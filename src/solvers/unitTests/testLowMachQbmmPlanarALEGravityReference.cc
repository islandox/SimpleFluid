/** Independently integrated one-dimensional forced two-family column on a prescribed moving domain. */
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <numbers>
#include <sstream>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace IO = Testing::QbmmRestart;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double gravity = -.5, lid_speed = 1. / 32., duration = 1. / 64.;
constexpr double liquid_density = 16., cv = 2., pressure_datum = 64., mass_scale = .125;
constexpr double length(double time)
{
    return 1. + lid_speed * time;
}

// Per unit *reference* length: liquid and two gas masses, total axial momentum, thermal internal energy.
// All three phases have unit molar mass, cp_liquid=cv_gas=2, R=1. The liquid is incompressible.
using Row = std::array<double, 5>;
using Column = std::vector<Row>;
double slip_speed(double x, bool supplied)
{
    if (!supplied || x <= .125 || x >= .875)
        return 0.;
    const double s = std::sin(std::numbers::pi * (x - .125) / .75);
    return .125 * s * s;
}

Row initial(double x)
{
    const double y = x - .5, gas_density = 8. + 2. * y;
    const double rho = .5 * (liquid_density + gas_density);
    // Hydrostatic background plus a resolved pressure disturbance whose derivative vanishes at both walls.
    const double p = pressure_datum + gravity * (12. * y + .5 * y * y) + .5 * std::cos(2. * std::numbers::pi * x);
    return {.5 * liquid_density, .25 * gas_density, .25 * gas_density, rho * lid_speed * x, cv * rho * p / gas_density};
}

struct Primitive
{
    double p, temperature, u, volume_velocity;
    std::array<double, 3> velocity;
};

Primitive recover(const Row& q, double x, double time, bool supplied)
{
    const double mass = q[0] + q[1] + q[2], gas = q[1] + q[2], slip = slip_speed(x, supplied);
    const double liquid_volume = q[0] / liquid_density, height = length(time);
    const double temperature = q[4] / (cv * mass);
    const double p = gas * temperature / (height - liquid_volume);
    const double u = (q[3] - slip * (q[1] - q[2])) / mass;
    const double gas_fraction = 1. - liquid_volume / height;
    return {p, temperature, u, u + gas_fraction * slip * (q[1] - q[2]) / gas, {u, u + slip, u - slip}};
}

// Eulerian balances pulled back by z=L(t)X. Face fluxes below use centered interpolation,
// independent of the production backward-Euler donor transport and pressure projection.
Column rhs(const Column& q, double time, bool supplied)
{
    const size_t n = q.size();
    Column flux(n + 1), result(n);
    std::vector<double> volume_velocity(n + 1);
    for (size_t j = 0; j < n; ++j)
    {
        const double x = (j + .5) / n;
        const auto a = recover(q[j], x, time, supplied);
        Row value{};
        value[3] = a.p;
        for (size_t p = 0; p < 3; ++p)
        {
            value[p] = q[j][p] / length(time) * (a.velocity[p] - lid_speed * x);
            value[3] += a.velocity[p] * value[p];
            value[4] += cv * a.temperature * value[p];
        }
        for (size_t face : {j, j + 1})
        {
            if (face == 0 || face == n)
            {
                // Impermeable end wall with prescribed constant speed: dp/dz = rho*g.
                flux[face][3] = a.p + gravity * (q[j][0] + q[j][1] + q[j][2]) * (face == 0 ? -.5 : .5) / n;
                volume_velocity[face] = face == 0 ? 0. : lid_speed;
                continue;
            }
            for (size_t k = 0; k < value.size(); ++k)
                flux[face][k] += .5 * value[k];
            volume_velocity[face] += .5 * a.volume_velocity;
        }
    }
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t k = 0; k < q[j].size(); ++k)
            result[j][k] = double(n) * (flux[j][k] - flux[j + 1][k]);
        result[j][3] += gravity * (q[j][0] + q[j][1] + q[j][2]);
        result[j][4] -=
            recover(q[j], (j + .5) / n, time, supplied).p * n * (volume_velocity[j + 1] - volume_velocity[j]);
    }
    return result;
}

Column increment(Column q, const Column& k, double h)
{
    for (size_t j = 0; j < q.size(); ++j)
        for (size_t i = 0; i < q[j].size(); ++i)
            q[j][i] += h * k[j][i];
    return q;
}

Column reference(size_t n, bool supplied)
{
    Column q(n);
    for (size_t j = 0; j < n; ++j)
        q[j] = initial((j + .5) / n);
    const int steps = int(n / 2);
    const double h = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const double t = step * h;
        const auto a = rhs(q, t, supplied), b = rhs(increment(q, a, h * .5), t + h * .5, supplied);
        const auto c = rhs(increment(q, b, h * .5), t + h * .5, supplied), d = rhs(increment(q, c, h), t + h, supplied);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < q[j].size(); ++i)
                q[j][i] += h / 6. * (a[j][i] + 2. * b[j][i] + 2. * c[j][i] + d[j][i]);
    }
    return q;
}

struct Vessel
{
    size_t n;
    bool supplied, reject_source = false;
    int targets = 0, sources = 0, rejected = 0;
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;

    Vessel(size_t layers, bool slip, bool batched = true, bool empty = false) : n(layers), supplied(slip)
    {
        const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
        const size_t across = empty ? std::min<size_t>(ranks, 2) : 1;
        ArrReal edges;
        for (size_t j = 0; j <= n; ++j)
            edges.push_back(double(j) / n);
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{ArrReal{0., 1.}, ArrReal{0., 1.}, edges}),
            Flow::Mesh::DistributionOptions{.allow_empty_partitions = true,
                .orthogonal_partitions = std::array<size_t, 3>{across, 1, ranks / across}});
        const auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "column material", 1., {{"M", 1}}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed}}});
        const R::ThermodynamicBounds bounds{2., 16., 16., 128.};
        const auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds,
            [](auto)
            {
                return R::LinearCompressibleMixtureLiquid{
                    8., pressure_datum, liquid_density, cv, 0., 0., cv * 8., 0., 0.};
            });
        const auto gas = R::make_ideal_gas_mixture(
            catalog, {{cv, 8., cv * 8.}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
        Flow::Options opts;
        opts.minimum_pressure = bounds.minimum_pressure;
        opts.maximum_pressure = bounds.maximum_pressure;
        opts.linear.preconditioner = LinearPreconditioner::None;
        opts.linear.tolerance = 1e-12;
        opts.use_multi_rhs_transport = batched;
        opts.qbmm_model.velocity =
            supplied ? R::QbmmModelOptions::Velocity::SuppliedPhaseSlip : R::QbmmModelOptions::Velocity::SharedCarrier;
        opts.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solvent"}, {Flow::PhaseRole::DispersedGas, "ascending"},
            {Flow::PhaseRole::DispersedGas, "descending"}};
        opts.qbmm_eos_law_identities = {
            R::ThermalLawIdentity{"constant column liquid", 1, catalog->fingerprint()}, std::nullopt, std::nullopt};
        opts.qbmm_families.resize(3);
        Flow::State state;
        state.qbmm_families.resize(3);
        state.thermodynamic_pressure = pressure_datum;
        for (size_t p = 1; p < 3; ++p)
        {
            opts.qbmm_families[p] = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(),
                opts.phase_roles[p].identity, opts.phase_roles[p], mass_scale,
                std::vector<R::QbmmMassFraction>{{47, 1.}});
            state.qbmm_families[p] = Flow::qbmm_family_stamp(*opts.qbmm_families[p]);
        }
        const std::vector eos{liquid, gas, gas};
        flow = std::make_shared<Flow>(mesh, eos, opts);
        for (size_t j = 0; j < mesh->num_owned_cells(); ++j)
        {
            const auto x = mesh->cell_centroid(static_cast<LO>(j)).z;
            const auto q = initial(x);
            const auto a = recover(q, x, 0., supplied);
            auto& cell = state.cells.emplace_back();
            cell.velocity.z = a.u;
            cell.mechanical_pressure = a.p - pressure_datum;
            cell.phase_numbers.resize(3);
            cell.qbmm_shape.resize(3);
            cell.drift_velocity.resize(3);
            for (size_t p = 0; p < 3; ++p)
            {
                const double m = q[p] / n;
                cell.phases.push_back({eos[p], {m}, a.p, m * cv * a.temperature, a.temperature});
                if (!p)
                    continue;
                const double w = m / (5. * mass_scale);
                cell.phase_numbers[p] = 3. * w;
                cell.qbmm_shape[p] = R::QbmmShapeState{11. * w, 29. * w};
                cell.drift_velocity[p].z = (p == 1 ? 1. : -1.) * slip_speed(x, supplied);
            }
        }
        flow->initialize(std::move(state));
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, gas->energy_convention());
        controls.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "ascending", {}},
            {R::SpeciesPhase::Dispersed, "descending", {}}};
        controls.source_identity = "inert gravity column";
        controls.source = [this](const R::CoupledReactionContext& ctx, double)
        {
            ++sources;
            if (reject_source && ctx.reaction.environment.location->geometryCellId == 0)
            {
                ++rejected;
                throw std::domain_error("injected moving-column source rejection");
            }
            return R::ReactionSourceInput{};
        };
        controls.body_acceleration = Flow::Vector{0., 0., gravity};
        controls.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
            R::MixturePlanarMotionProvider{mesh, {}, "constant column lid speed", 1, [this](double t, double h)
                {
                    ++targets;
                    return length(t + h);
                }});
        controller = std::make_unique<Coupler>(flow, controls);
        EXPECT_EQ(sum(mesh->num_owned_cells() == 0), ranks - ranks / across);
        for (const char* key : {"SIMPLEFLUID_Q8E_REFERENCE_RANKS", "SIMPLEFLUID_Q8E_RANKS"})
            if (const char* expected = std::getenv(key))
                EXPECT_EQ(ranks, std::stoul(expected));
    }
    double sum(double local) const
    {
        double total;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &total);
        return total;
    }

    void step(double h, bool require_crossing = false)
    {
        const auto trial = controller->preview(h);
        const auto& d = trial.diagnostics().flow;
        ASSERT_TRUE(d.impulse_parts);
        EXPECT_NEAR(d.impulse_parts->body.z, 12. * gravity * h, 4e-14);
        EXPECT_EQ(d.impulse_parts->other, Flow::Vector{});
        EXPECT_LE(d.maximum_continuity_residual, flow->options().continuity_tolerance);
        size_t crossing = 0;
        for (size_t i = 0; i < trial.owned_phase_transport().size(); ++i)
        {
            const auto face = static_cast<LO>(i);
            const auto& receipt = trial.owned_phase_transport()[i];
            if (mesh->is_geometry_exterior_face(face))
            {
                for (auto rate : receipt.relative_phase_volume_flux)
                    EXPECT_EQ(rate, 0.);
                continue;
            }
            const bool opposite =
                receipt.qbmm_flux[1]->derived_mass_flux * receipt.qbmm_flux[2]->derived_mass_flux < 0.;
            if (opposite &&
                (mesh->owned_cell_map()->getComm()->getSize() == 1 || !mesh->is_owned_cell(mesh->owner_cell(face)) ||
                    !mesh->is_owned_cell(mesh->neighbor_cell(face))))
                ++crossing;
        }
        if (require_crossing)
            EXPECT_GT(sum(crossing), 0.);
        controller->commit(trial);
    }
};

// Compare changes from the initial condition. Comparing raw pressure or stock would mask
// the pressure wave and gas redistribution beneath the datum and the stationary inventory.
using Observations = std::array<double, 4>;
Observations changes(const Row& q, double x, bool slip)
{
    const auto q0 = initial(x);
    const auto end = recover(q, x, duration, slip), start = recover(q0, x, 0., slip);
    return {end.p - start.p, end.u - start.u, q[1] - q0[1], q[2] - q0[2]};
}

Observations reference_error(const Vessel& f, const Column& oracle)
{
    Observations local{};
    const auto width = oracle.size() / f.n;
    for (size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
    {
        const auto id = static_cast<LO>(c);
        const size_t j = static_cast<size_t>(f.mesh->cell_centroid(id).z * f.n / length(duration));
        const double x = (j + .5) / f.n;
        const auto q0 = initial(x);
        const auto a0 = recover(q0, x, 0., f.supplied);
        const auto& end = f.flow->state().cells[c];
        const Observations measured{end.phases[1].absolute_pressure - a0.p, end.velocity.z - a0.u,
            f.n * end.phases[1].moles[0] - q0[1], f.n * end.phases[2].moles[0] - q0[2]};
        Observations expected{};
        for (size_t i = j * width; i < (j + 1) * width; ++i)
        {
            const auto fine = changes(oracle[i], (i + .5) / oracle.size(), f.supplied);
            for (size_t k = 0; k < fine.size(); ++k)
                expected[k] += fine[k] / width;
        }
        for (size_t k = 0; k < expected.size(); ++k)
            local[k] += std::abs(measured[k] - expected[k]) / f.n;
    }
    for (auto& v : local)
        v = f.sum(v);
    return local;
}

void metric(const std::string& name, double value)
{
    std::ostringstream message;
    message << std::setprecision(17) << value;
    ::testing::Test::RecordProperty(name, message.str());
}

void reference_case(bool slip)
{
    const auto fine = reference(256, slip), refined = reference(512, slip);
    const std::array<std::string, 4> names{"pressure_pa", "carrier_m_s", "rising_kg_m3", "falling_kg_m3"};
    Observations reference_change{}, reference_refinement{};
    for (size_t i = 0; i < fine.size(); ++i)
    {
        const auto a = changes(fine[i], (i + .5) / fine.size(), slip);
        const auto b = changes(refined[2 * i], (2 * i + .5) / refined.size(), slip);
        const auto c = changes(refined[2 * i + 1], (2 * i + 1.5) / refined.size(), slip);
        for (size_t k = 0; k < a.size(); ++k)
        {
            reference_change[k] += std::abs(.5 * (b[k] + c[k])) / fine.size();
            reference_refinement[k] += std::abs(a[k] - .5 * (b[k] + c[k])) / fine.size();
        }
    }
    for (size_t k = 0; k < names.size(); ++k)
    {
        metric("reference_change_" + names[k], reference_change[k]);
        metric("reference_refinement_" + names[k], reference_refinement[k]);
        EXPECT_LT(reference_refinement[k], .01 * reference_change[k]);
    }
    // At fixed 16 steps, spatial halving isolates the spatial error. Separately halve dt
    // at fixed 16 cells: no cancellation between changes of space and time in either check.
    const auto integrate = [&](int n, int steps)
    {
        Vessel f(n, slip);
        for (int i = 0; i < steps; ++i)
            f.step(duration / steps, slip);
        const auto err = reference_error(f, refined);
        const auto label = "n" + std::to_string(n) + "_steps" + std::to_string(steps);
        for (size_t k = 0; k < names.size(); ++k)
            metric(label + "_" + names[k], err[k]);
        if (n == 32)
        {
            for (size_t p : {1, 2})
            {
                // Each family starts with exactly 2 kg. Its centroid displacement
                // removes the stationary quadrature offset on each resolution.
                double displacement = 0., expected = 0., mass = 0.;
                for (size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
                {
                    const double z = f.mesh->cell_centroid(static_cast<LO>(c)).z;
                    const double x = z / length(duration);
                    const double m = f.flow->state().cells[c].phases[p].moles[0];
                    displacement += .5 * (z * m - x * initial(x)[p] / n);
                    mass += m;
                }
                for (size_t i = 0; i < refined.size(); ++i)
                {
                    const double x = (i + .5) / refined.size();
                    expected += .5 * x * (length(duration) * refined[i][p] - initial(x)[p]) / refined.size();
                }
                displacement = f.sum(displacement);
                const std::string family = p == 1 ? "rising" : "falling";
                metric(family + "_centroid_displacement_m", displacement);
                metric(family + "_reference_centroid_displacement_m", expected);
                EXPECT_NEAR(f.sum(mass), 2., 2e-11);
                EXPECT_NEAR(displacement, expected, .05 * std::abs(expected));
            }
        }
        return err;
    };
    const auto n4 = integrate(4, 16), n8 = integrate(8, 16), n16 = integrate(16, 16), n32 = integrate(32, 16);
    const auto two = integrate(16, 2), four = integrate(16, 4), eight = integrate(16, 8);
    for (size_t k = 0; k < names.size(); ++k)
    {
        SCOPED_TRACE(names[k]);
        EXPECT_LT(n8[k], n4[k]);
        EXPECT_LT(n16[k], n8[k]);
        EXPECT_LT(n32[k], n16[k]);
        EXPECT_LT(n32[k], .1 * reference_change[k]);
        EXPECT_LT(four[k], two[k]);
        EXPECT_LT(eight[k], four[k]);
        EXPECT_LT(n16[k], eight[k]);
    }
}

TEST(QbmmPlanarGravityReference, SharedColumnAgainstIndependentMovingCoordinateEvolution)
{
    reference_case(false);
}
TEST(QbmmPlanarGravityReference, CountercurrentColumnAgainstIndependentMovingCoordinateEvolution)
{
    reference_case(true);
}

void same_endpoint(const Vessel& a, const Vessel& b, double tolerance = 0.)
{
    const auto& left = a.flow->state();
    const auto& right = b.flow->state();
    EXPECT_EQ(left.time, right.time);
    EXPECT_EQ(left.revision, right.revision);
    EXPECT_NEAR(left.thermodynamic_pressure, right.thermodynamic_pressure, tolerance);
    ASSERT_EQ(left.cells.size(), right.cells.size());
    for (size_t c = 0; c < left.cells.size(); ++c)
    {
        const auto id = static_cast<LO>(c);
        EXPECT_EQ(a.mesh->cell_volume(id), b.mesh->cell_volume(id));
        EXPECT_EQ(a.mesh->cell_centroid(id), b.mesh->cell_centroid(id));
        const auto& x = left.cells[c];
        const auto& y = right.cells[c];
        EXPECT_NEAR(x.mechanical_pressure, y.mechanical_pressure, tolerance);
        EXPECT_EQ(x.drift_velocity, y.drift_velocity);
        for (size_t dim = 0; dim < 3; ++dim)
            EXPECT_NEAR(x.velocity.component(dim), y.velocity.component(dim), tolerance);
        for (size_t p = 0; p < 3; ++p)
        {
            EXPECT_NEAR(x.phases[p].moles[0], y.phases[p].moles[0], tolerance);
            EXPECT_NEAR(x.phases[p].internal_energy, y.phases[p].internal_energy, tolerance);
            EXPECT_NEAR(x.phases[p].temperature, y.phases[p].temperature, tolerance);
            EXPECT_NEAR(x.phases[p].absolute_pressure, y.phases[p].absolute_pressure, tolerance);
            if (p)
            {
                EXPECT_NEAR(*x.phase_numbers[p], *y.phase_numbers[p], tolerance);
                EXPECT_NEAR(x.qbmm_shape[p]->H2, y.qbmm_shape[p]->H2, tolerance);
                EXPECT_NEAR(x.qbmm_shape[p]->H3, y.qbmm_shape[p]->H3, tolerance);
            }
        }
    }
    for (size_t i = 0; i < a.mesh->num_faces(); ++i)
    {
        const auto face = static_cast<LO>(i);
        EXPECT_NEAR(a.flow->volume_flux().local_value(face), b.flow->volume_flux().local_value(face), tolerance);
    }
    const auto& x = a.controller->diagnostics();
    const auto& y = b.controller->diagnostics();
    EXPECT_NEAR(x.flow.pressure_work, y.flow.pressure_work, tolerance);
    EXPECT_EQ(x.flow.body_acceleration_m_per_s2, y.flow.body_acceleration_m_per_s2);
    EXPECT_NEAR(x.flow.external_impulse.z, y.flow.external_impulse.z, tolerance);
    EXPECT_EQ(x.external_mass, y.external_mass);
    EXPECT_EQ(x.external_heat, y.external_heat);
    if (tolerance == 0.)
    {
        EXPECT_EQ(x.correctors, y.correctors);
        EXPECT_EQ(x.flow.impulse_parts, y.flow.impulse_parts);
    }
}

TEST(QbmmPlanarGravityReference, GroupedAndScalarStocksAcrossPartitionsAndEmptyOwners)
{
    for (bool slip : {false, true})
        for (bool empty : {false, true})
        {
            Vessel grouped(8, slip, true, empty), scalar(8, slip, false, empty);
            const bool crossing = slip && (!empty || grouped.mesh->owned_cell_map()->getComm()->getSize() == 4);
            for (int i = 0; i < 2; ++i)
            {
                grouped.step(duration / 2, crossing);
                scalar.step(duration / 2, crossing);
                EXPECT_GT(grouped.controller->diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
                EXPECT_EQ(scalar.controller->diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
                same_endpoint(grouped, scalar, 3e-9);
            }
        }
}

void check_tag(std::span<const std::byte> bytes)
{
    std::vector<std::byte> expected;
    for (auto word : std::array<std::uint64_t, 3>{0x5453524d4d425153ULL, 9, 12})
        IO::append_restart_word(expected, word);
    ASSERT_GE(bytes.size(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), bytes.begin()));
}

void portable(bool supplied)
{
    const char* writer = std::getenv("SIMPLEFLUID_Q8E_REFERENCE_WRITE");
    const char* reader = std::getenv("SIMPLEFLUID_Q8E_REFERENCE_READ");
    ASSERT_FALSE(writer && reader);
    Vessel original(8, supplied, true, true);
    const auto comm = original.mesh->owned_cell_map()->getComm();
    const double h = duration / 2;
    std::vector<std::byte> first;
    if (reader)
    {
        first = IO::read_restart(*comm, reader, ".first");
        original.controller->restore_qbmm_restart(IO::read_restart(*comm, reader, ".second"));
        EXPECT_EQ(original.targets, 0);
        EXPECT_EQ(original.sources, 0);
    }
    else
    {
        original.step(h);
        first = original.controller->serialize_qbmm_restart();
        check_tag(first);
        original.step(h);
        if (writer)
        {
            IO::write_restart(*comm, writer, ".first", first);
            IO::write_restart(*comm, writer, ".second", original.controller->serialize_qbmm_restart());
            return;
        }
    }
    check_tag(first);
    Vessel resumed(8, supplied, true, true);
    resumed.controller->restore_qbmm_restart(first);
    EXPECT_EQ(resumed.sources, 0);
    EXPECT_EQ(resumed.targets, 0);
    const auto restored = resumed.controller->qbmm_report();
    EXPECT_EQ(restored.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_EQ(restored.body_acceleration_m_per_s2, (Flow::Vector{0., 0., gravity}));
    ASSERT_TRUE(restored.external_impulses_kg_m_per_s);
    EXPECT_NEAR(restored.external_impulses_kg_m_per_s->body.z, gravity * 12. * h, 4e-14);
    const auto accepted = resumed.controller->serialize_qbmm_restart();
    const auto epoch = resumed.mesh->geometry_epoch();
    const auto retained = resumed.controller->preview(h);
    resumed.reject_source = true;
    EXPECT_THROW((void) resumed.controller->preview(.5 * h), std::exception);
    EXPECT_EQ(resumed.sum(resumed.rejected), 1.);
    resumed.reject_source = false;
    const auto counts = std::pair{resumed.targets, resumed.sources};
    // Recompute shard and global checksums after independently changing the rank-local body impulse.
    constexpr size_t local_body_z = 19 * sizeof(std::uint64_t);
    auto corrupt = first;
    ASSERT_GT(corrupt.size(), local_body_z + 3 * sizeof(std::uint64_t));
    corrupt.resize(corrupt.size() - 2 * sizeof(std::uint64_t));
    if (comm->getRank() == 0)
        corrupt[local_body_z + 6] ^= std::byte{0x10};
    IO::append_restart_checksums(*comm, corrupt);
    EXPECT_THROW(resumed.controller->restore_qbmm_restart(corrupt), std::exception);
    EXPECT_EQ(resumed.controller->serialize_qbmm_restart(), accepted);
    EXPECT_EQ(resumed.mesh->geometry_epoch(), epoch);
    EXPECT_EQ(resumed.flow->state().time, h);
    resumed.controller->commit(retained);
    EXPECT_EQ((std::pair{resumed.targets, resumed.sources}), counts);
    same_endpoint(original, resumed);

    // Decode valid first-interval stocks, force, motion law and wall data on the second
    // accepted surface. The geometry digest is checked only after staging the earlier surface.
    auto wrong_geometry = first;
    constexpr size_t geometry_word = 26; // Follows the tag-12 interval's rank-local external impulse.
    wrong_geometry.resize(wrong_geometry.size() - 2 * sizeof(std::uint64_t));
    if (comm->getRank() == 0)
        wrong_geometry.at(geometry_word * sizeof(std::uint64_t)) ^= std::byte{1};
    IO::append_restart_checksums(*comm, wrong_geometry);
    const auto second = original.controller->serialize_qbmm_restart();
    const auto second_epoch = original.mesh->geometry_epoch();
    const auto third = original.controller->preview(h);
    const auto third_callbacks = std::pair{original.targets, original.sources};
    try
    {
        original.controller->restore_qbmm_restart(wrong_geometry);
        ADD_FAILURE() << "Restored an inconsistent accepted geometry fingerprint";
    }
    catch (const std::exception& e)
    {
        const std::string_view expected = comm->getRank() == 0 ? "accepted planar geometry" : "restart physical state";
        EXPECT_NE(std::string_view(e.what()).find(expected), std::string_view::npos) << e.what();
    }
    EXPECT_EQ(original.mesh->geometry_epoch(), second_epoch);
    EXPECT_EQ(original.controller->serialize_qbmm_restart(), second);
    same_endpoint(original, resumed);
    original.controller->commit(third);
    EXPECT_EQ((std::pair{original.targets, original.sources}), third_callbacks);
    resumed.step(h);
    same_endpoint(original, resumed);
}

TEST(QbmmPlanarGravityReference, SharedPortableContinuation)
{
    portable(false);
}
TEST(QbmmPlanarGravityReference, SuppliedSlipPortableContinuation)
{
    portable(true);
}
} // namespace
