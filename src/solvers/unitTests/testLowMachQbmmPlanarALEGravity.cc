/** Forced NativeImplicit planar ALE on closed hydrostatic columns and an analytic multiphase piston. */
#include "QbmmKineticTestSupport.hh"
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <tuple>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
using Coupler = R::ReactingMixtureCoupler;
using Velocity = Model::Velocity;
using Vec = Flow::Vector;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

enum class Geometry
{
    Cartesian,
    Sector,
    Extruded
};

constexpr double p0 = 64., t0 = 8., rho_liquid = 16., rho_gas = 8., cv = 2., dt = 1. / 64.;
// Molecular mass and gas constant are both one in SI units, so mass equals mole amount.
constexpr double scale = .125;

SP<Flow::Mesh> column_mesh(Geometry kind, size_t layers)
{
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    if (const char* expected = std::getenv("SIMPLEFLUID_Q8E_RANKS"))
        EXPECT_EQ(ranks, std::stoul(expected));
    testing::Test::RecordProperty("mpi_ranks", int(ranks));
    ArrReal z;
    for (size_t j = 0; j <= layers; ++j)
        z.push_back(double(j) / layers);
    Flow::Mesh::DistributionOptions partition{.allow_empty_partitions = true};
    if (kind == Geometry::Extruded)
    {
        using Semi = Meshes::SemiStructuredXY_Z;
        partition.xy_partitions = 1;
        partition.z_partitions = ranks;
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Semi>(
                Arr<Semi::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}, Arr<Arr<unsigned>>{{0, 1, 2, 3}}, z),
            partition);
    }
    partition.orthogonal_partitions = std::array<size_t, 3>{1, 1, ranks};
    if (kind == Geometry::Sector)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1, 2}, {0, 1}, z}), partition);
    return std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{ArrReal{0, 1}, {0, 1}, z}), partition);
}

double gas_fraction(size_t j)
{
    return (j + 1.) / 8.;
}
double layer_density(size_t j)
{
    return (1. - gas_fraction(j)) * rho_liquid + gas_fraction(j) * rho_gas;
}

struct Column
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;
    std::vector<double> pressure;
    double target = 1.;

    Column(Geometry kind, Velocity velocity, size_t layers, std::optional<double> gravity, bool moving,
        bool count_only = false, double absolute_linear_tolerance = 0.)
        : mesh(column_mesh(kind, layers)), pressure(layers, p0)
    {
        using Phase = R::SpeciesPhase;
        auto species = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "column material", 1., {{"G", 1}}, {Phase::Liquid, Phase::Dispersed}}});
        const R::ThermodynamicBounds bounds{2., 16., 16., 128.};
        auto liquid = R::make_linear_compressible_mixture_liquid(species, bounds,
            [](auto) { return R::LinearCompressibleMixtureLiquid{t0, p0, rho_liquid, cv, 0., 0., cv * t0, 0., 0.}; });
        auto gas = R::make_ideal_gas_mixture(
            species, {{cv, t0, cv * t0}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
        Flow::Options options;
        options.minimum_pressure = bounds.minimum_pressure;
        options.maximum_pressure = bounds.maximum_pressure;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.linear.absolute_tolerance = absolute_linear_tolerance;
        options.qbmm_model.velocity = velocity;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "bubbles"}};
        options.qbmm_eos_law_identities = {R::ThermalLawIdentity{"column liquid", 1, species->fingerprint()}, {}};
        Flow::State state;
        if (!count_only)
        {
            options.qbmm_families = {
                nullptr, std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "bubbles",
                             options.phase_roles[1], scale, std::vector<R::QbmmMassFraction>{{47, 1.}})};
            state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1])};
        }
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas}, options);
        for (size_t j = 1; j < layers; ++j)
            pressure[j] =
                pressure[j - 1] + .5 * (layer_density(j - 1) + layer_density(j)) * gravity.value_or(0.) / layers;
        state.thermodynamic_pressure = std::accumulate(pressure.begin(), pressure.end(), 0.) / layers;
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const size_t j = static_cast<size_t>(mesh->cell_centroid(static_cast<LO>(i)).z * layers);
            const double v = mesh->cell_volume(static_cast<LO>(i)), p = pressure[j], t = p / rho_gas;
            const double ng = gas_fraction(j) * v * rho_gas, nl = (1. - gas_fraction(j)) * v * rho_liquid;
            auto& cell = state.cells.emplace_back();
            cell.mechanical_pressure = p - state.thermodynamic_pressure;
            cell.phases = {{liquid, {nl}, p, nl * cv * t, t}, {gas, {ng}, p, ng * cv * t, t}};
            // Weights 2w and w at dimensionless masses 1 and 3.
            const double w = ng / (5. * scale);
            cell.phase_numbers = {std::nullopt, 3. * w};
            if (!count_only)
                cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{11. * w, 29. * w}};
        }
        flow->initialize(std::move(state));
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            species, std::vector<R::ReactionDefinition>{}, liquid->energy_convention());
        controls.phases = {{Phase::Liquid, {}, {}}, {Phase::Dispersed, "bubbles", {}}};
        controls.source_identity = "inert column";
        controls.source = [](const auto&, double) { return R::ReactionSourceInput{}; };
        if (gravity)
            controls.body_acceleration = Vec{0., 0., *gravity};
        if (moving)
            controls.planar_motion =
                std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
                    mesh, {}, "prescribed column end", 1, [this](double, double) { return target; }});
        controller = std::make_unique<Coupler>(flow, std::move(controls));
    }

    double sum(double local) const { return reduce(*mesh, Teuchos::REDUCE_SUM, local); }
};

double mass(const Flow::CellState& c)
{
    return c.phases[0].moles[0] + c.phases[1].moles[0];
}

void check_impulse(const Column& column, const Coupler::Trial& trial, double gravity, bool stationary)
{
    const auto& diagnostics = trial.diagnostics().flow;
    EXPECT_EQ(diagnostics.body_acceleration_m_per_s2, (Vec{0., 0., gravity}));
    ASSERT_TRUE(diagnostics.impulse_parts);
    double local_mass = 0.;
    Vec local_delta{};
    for (size_t i = 0; i < trial.state().cells.size(); ++i)
    {
        const auto& before = column.flow->state().cells[i];
        const auto& after = trial.state().cells[i];
        local_mass += mass(before);
        local_delta = local_delta + after.velocity * mass(after) - before.velocity * mass(before);
    }
    const Vec body{0., 0., column.sum(local_mass) * gravity * dt};
    for (size_t k = 0; k < 3; ++k)
    {
        EXPECT_NEAR(diagnostics.impulse_parts->body.component(k), body.component(k), 3e-13);
        EXPECT_EQ(diagnostics.impulse_parts->other.component(k), 0.);
        EXPECT_NEAR(diagnostics.external_impulse.component(k), body.component(k), 3e-13);
        EXPECT_NEAR(
            column.sum(local_delta.component(k)), body.component(k) + diagnostics.pressure_impulse.component(k), 1e-10);
        if (stationary)
            EXPECT_NEAR(diagnostics.pressure_impulse.component(k), -body.component(k), 1e-10);
    }
}

void check_stocks(const Flow::CellState& endpoint, const Flow::CellState& initial)
{
    for (size_t p = 0; p < initial.phases.size(); ++p)
        EXPECT_NEAR(endpoint.phases[p].moles[0], initial.phases[p].moles[0], 2e-10);
    EXPECT_NEAR(endpoint.phase_numbers[1].value(), initial.phase_numbers[1].value(), 2e-10);
    if (!initial.qbmm_shape.empty())
    {
        EXPECT_NEAR(endpoint.qbmm_shape[1]->H2, initial.qbmm_shape[1]->H2, 2e-9);
        EXPECT_NEAR(endpoint.qbmm_shape[1]->H3, initial.qbmm_shape[1]->H3, 2e-9);
    }
}

void metric(const std::string& name, double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    testing::Test::RecordProperty(name, stream.str());
}

void hydrostatic(Geometry kind, Velocity velocity, double gravity)
{
    SCOPED_TRACE(gravity);
    constexpr size_t layers = 4;
    Column fixed(kind, velocity, layers, gravity, false);
    Column moving(kind, velocity, layers, gravity, true);
    const auto reference = fixed.flow->state();
    EXPECT_EQ(moving.sum(moving.mesh->num_owned_cells()), layers);
    EXPECT_EQ(moving.sum(moving.mesh->num_owned_cells() == 0), 0.);
    double speed = 0., rate = 0., edge = 0.;
    for (int step = 0; step < 2; ++step)
    {
        auto a = fixed.controller->preview(dt);
        auto b = moving.controller->preview(dt);
        check_impulse(moving, b, gravity, true);
        check_impulse(fixed, a, gravity, true);
        fixed.controller->commit(a);
        moving.controller->commit(b);
        std::array<double, layers> pressure{}, density{};
        for (size_t i = 0; i < reference.cells.size(); ++i)
        {
            const auto id = static_cast<LO>(i);
            const auto j = static_cast<size_t>(moving.mesh->cell_centroid(id).z * layers);
            const auto& cell = moving.flow->state().cells[i];
            const auto& stationary = fixed.flow->state().cells[i];
            pressure[j] = moving.flow->state().thermodynamic_pressure + cell.mechanical_pressure;
            density[j] = mass(cell) / moving.mesh->cell_volume(id);
            EXPECT_NEAR(pressure[j], fixed.flow->state().thermodynamic_pressure + stationary.mechanical_pressure, 1e-6);
            EXPECT_NEAR(pressure[j], moving.pressure[j], 1e-6);
            EXPECT_NEAR(density[j], layer_density(j), 1e-9);
            speed = std::max(speed, cell.velocity.norm());
            EXPECT_LT(cell.velocity.norm(), 1e-9);
            EXPECT_LT(stationary.velocity.norm(), 1e-9);
            check_stocks(cell, reference.cells[i]);
            for (size_t p = 0; p < cell.phases.size(); ++p)
                EXPECT_NEAR(cell.phases[p].internal_energy, reference.cells[i].phases[p].internal_energy, 1e-8);
        }
        for (size_t j = 0; j < layers; ++j)
        {
            pressure[j] = moving.sum(pressure[j]);
            density[j] = moving.sum(density[j]);
            if (j)
            {
                const double head = gravity * .5 * (density[j - 1] + density[j]) / layers;
                const double defect = std::abs(pressure[j] - pressure[j - 1] - head);
                edge = std::max(edge, defect);
                EXPECT_LT(defect, 1e-6);
            }
        }
        for (const auto face : moving.flow->volume_flux().owned_face_ids())
        {
            const double absolute_rate = std::abs(moving.flow->volume_flux().owned_value(face));
            rate = std::max(rate, absolute_rate);
            EXPECT_LT(absolute_rate, 1e-10);
            EXPECT_LT(std::abs(fixed.flow->volume_flux().owned_value(face)), 1e-10);
            const auto& receipt = b.owned_phase_transport()[moving.flow->volume_flux().owned_row(face)];
            EXPECT_EQ(receipt.mesh_volume_flux, 0.);
            for (double relative : receipt.relative_phase_volume_flux)
                EXPECT_LT(std::abs(relative), 1e-10);
        }
        EXPECT_NEAR(b.diagnostics().flow.pressure_work, 0., 2e-9);
    }
    const std::string sign = gravity < 0. ? "down_" : gravity > 0. ? "up_" : "zero_";
    metric(sign + "max_speed_m_s", reduce(*moving.mesh, Teuchos::REDUCE_MAX, speed));
    metric(sign + "max_face_rate_m3_s", reduce(*moving.mesh, Teuchos::REDUCE_MAX, rate));
    metric(sign + "max_pressure_edge_defect_Pa", edge);
}

class QbmmPlanarGravityColumn : public testing::TestWithParam<std::tuple<Geometry, Velocity>>
{
};

TEST_P(QbmmPlanarGravityColumn, StationaryTargetMatchesFixedHydrostatics)
{
    const auto [kind, velocity] = GetParam();
    for (double g : {-1., 1., 0.})
        hydrostatic(kind, velocity, g);
}

TEST_P(QbmmPlanarGravityColumn, CompressionAndExpansionCloseGasWorkGclAndMomentum)
{
    const auto [kind, velocity] = GetParam();
    constexpr double gravity = -.75;
    Column column(kind, velocity, 1, gravity, true);
    const auto original = column.flow->state();
    const int ranks = column.mesh->owned_cell_map()->getComm()->getSize();
    EXPECT_EQ(column.sum(column.mesh->num_owned_cells() == 0), ranks - 1);
    // At fixed column base, V = base area * target. Each interval uses BE pressure work
    // C (T1-T0) = -p1 (V1-V0), ideal gas p1*(V1-Vliquid) = n_g*T1.
    const double area = kind == Geometry::Sector ? 1.5 : 1.;
    const double liquid_v = (1. - gas_fraction(0)) * area, gas_n = gas_fraction(0) * area * rho_gas;
    const double heat_capacity = cv * (gas_n + rho_liquid * liquid_v);
    double previous_volume = area, previous_temperature = t0;
    for (const double height : {1. - 1. / 128., 1. + 1. / 128.})
    {
        SCOPED_TRACE(height);
        column.target = height;
        const double v1 = area * height, dv = v1 - previous_volume;
        const double t1 = heat_capacity * previous_temperature / (heat_capacity + gas_n * dv / (v1 - liquid_v));
        const double p1 = gas_n * t1 / (v1 - liquid_v), work = -p1 * dv;
        const auto trial = column.controller->preview(dt);
        const auto& d = trial.diagnostics().flow;
        check_impulse(column, trial, gravity, false);
        double local_delta_u = 0., local_sweep = 0.;
        for (size_t i = 0; i < trial.state().cells.size(); ++i)
        {
            const auto& cell = trial.state().cells[i];
            check_stocks(cell, original.cells[i]);
            EXPECT_NEAR(trial.transport_snapshot().old_control_volumes[i], previous_volume, 1e-13);
            EXPECT_NEAR(trial.transport_snapshot().new_control_volumes[i], v1, 1e-13);
            EXPECT_NEAR(trial.state().thermodynamic_pressure + cell.mechanical_pressure, p1, 1e-6);
            for (size_t p = 0; p < cell.phases.size(); ++p)
            {
                EXPECT_NEAR(cell.phases[p].temperature, t1, 2e-7);
                local_delta_u +=
                    cell.phases[p].internal_energy - column.flow->state().cells[i].phases[p].internal_energy;
            }
        }
        for (const auto& face : trial.owned_phase_transport())
        {
            EXPECT_EQ(face.carrier_volume_flux, face.mesh_volume_flux);
            for (double q : face.relative_phase_volume_flux)
                EXPECT_EQ(q, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->derived_mass_flux, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->population_number_flux, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->shape_flux, R::QbmmShapeState{});
            EXPECT_EQ(face.qbmm_flux[1]->internal_energy_flux, 0.);
            local_sweep += face.mesh_volume_flux;
        }
        const double sweep = column.sum(local_sweep) * dt, delta_u = column.sum(local_delta_u);
        const std::string step = dv < 0. ? "compression_" : "expansion_";
        metric(step + "gcl_defect_m3", std::abs(sweep - dv));
        metric(step + "energy_defect_J", std::abs(delta_u - work));
        metric(step + "work_defect_J", std::abs(d.pressure_work - work));
        EXPECT_NEAR(sweep, dv, 1e-13);
        EXPECT_NEAR(delta_u, work, 2e-7);
        EXPECT_NEAR(d.pressure_work, work, 2e-7);
        EXPECT_LE(d.maximum_continuity_residual, column.flow->options().continuity_tolerance);
        EXPECT_LE(d.maximum_occupancy_error, column.flow->options().volume_relative_tolerance);
        column.controller->commit(trial);
        previous_volume = v1;
        previous_temperature = t1;
    }
}

std::string column_case_name(const testing::TestParamInfo<std::tuple<Geometry, Velocity>>& info)
{
    const auto [kind, velocity] = info.param;
    constexpr std::array names{"Cartesian", "Sector", "Extruded"};
    return std::string(names[static_cast<size_t>(kind)]) + (velocity == Velocity::SharedCarrier ? "Shared" : "Slip");
}

INSTANTIATE_TEST_SUITE_P(NativeImplicit, QbmmPlanarGravityColumn,
    testing::Combine(testing::Values(Geometry::Cartesian, Geometry::Sector, Geometry::Extruded),
        testing::Values(Velocity::SharedCarrier, Velocity::SuppliedPhaseSlip)),
    column_case_name);

void compare_count_only_endpoints(const Column& uninterrupted, const Column& resumed)
{
    const auto& expected = uninterrupted.flow->state();
    const auto& actual = resumed.flow->state();
    EXPECT_EQ(actual.time, expected.time);
    EXPECT_EQ(actual.revision, expected.revision);
    EXPECT_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    EXPECT_TRUE(actual.qbmm_families.empty());
    const auto& first = uninterrupted.controller->diagnostics().flow;
    const auto& second = resumed.controller->diagnostics().flow;
    EXPECT_EQ(second.body_acceleration_m_per_s2, first.body_acceleration_m_per_s2);
    EXPECT_EQ(second.impulse_parts, first.impulse_parts);
    EXPECT_EQ(second.pressure_impulse, first.pressure_impulse);
    EXPECT_EQ(second.pressure_work, first.pressure_work);
    for (size_t i = 0; i < actual.cells.size(); ++i)
    {
        const auto& a = expected.cells.at(i);
        const auto& b = actual.cells[i];
        EXPECT_EQ(b.velocity, a.velocity);
        EXPECT_EQ(b.drift_velocity, a.drift_velocity);
        EXPECT_EQ(b.mechanical_pressure, a.mechanical_pressure);
        EXPECT_EQ(b.phase_numbers, a.phase_numbers);
        EXPECT_TRUE(b.qbmm_shape.empty());
        for (size_t p = 0; p < b.phases.size(); ++p)
        {
            EXPECT_EQ(b.phases[p].moles, a.phases.at(p).moles);
            EXPECT_EQ(b.phases[p].temperature, a.phases[p].temperature);
            EXPECT_EQ(b.phases[p].absolute_pressure, a.phases[p].absolute_pressure);
            EXPECT_EQ(b.phases[p].internal_energy, a.phases[p].internal_energy);
        }
        const auto id = static_cast<LO>(i);
        EXPECT_EQ(resumed.mesh->cell_volume(id), uninterrupted.mesh->cell_volume(id));
        EXPECT_EQ(resumed.flow->density().owned_value(id), uninterrupted.flow->density().owned_value(id));
        EXPECT_EQ(resumed.flow->velocity().owned_value(id), uninterrupted.flow->velocity().owned_value(id));
    }
    for (const auto face : resumed.flow->volume_flux().owned_face_ids())
        EXPECT_EQ(resumed.flow->volume_flux().owned_value(face), uninterrupted.flow->volume_flux().owned_value(face));
}

TEST(QbmmPlanarGravity, CountOnlyMovingControllerRetainsDispersedNumber)
{
    for (const double gravity : {0., .5})
        for (const auto velocity : {Velocity::SharedCarrier, Velocity::SuppliedPhaseSlip})
        {
            SCOPED_TRACE(gravity);
            SCOPED_TRACE(static_cast<int>(velocity));
            Column column(Geometry::Cartesian, velocity, 1, gravity, true, true);
            const auto initial = column.flow->state();
            const auto advance = [&](Column& c, double target)
            {
                c.target = target;
                const auto candidate = c.controller->preview(dt);
                check_impulse(c, candidate, gravity, false);
                EXPECT_TRUE(candidate.state().qbmm_families.empty());
                for (size_t i = 0; i < initial.cells.size(); ++i)
                {
                    const auto& old = c.flow->state().cells[i];
                    const auto& cell = candidate.state().cells[i];
                    const double liquid_v = old.phases[0].moles[0] / rho_liquid;
                    const double gas_n = old.phases[1].moles[0], capacity = cv * mass(old);
                    const double dv = target - c.mesh->cell_volume(static_cast<LO>(i));
                    const double expected_t =
                        old.phases[1].temperature / (1. + gas_n * dv / (capacity * (target - liquid_v)));
                    const double expected_p = gas_n * expected_t / (target - liquid_v);
                    check_stocks(cell, initial.cells[i]);
                    for (const auto& phase : cell.phases)
                    {
                        EXPECT_NEAR(phase.absolute_pressure, expected_p, 1e-6);
                        EXPECT_NEAR(phase.temperature, expected_t, 2e-7);
                    }
                    EXPECT_NEAR(candidate.diagnostics().flow.pressure_work, -expected_p * dv, 2e-7);
                }
                c.controller->commit(candidate);
            };
            advance(column, 1. + 1. / 512.);
            const auto archive = column.controller->serialize_qbmm_restart();
            std::vector<std::byte> header;
            for (auto word : std::array<std::uint64_t, 3>{0x5453524d4d425153ULL, 9, 12})
                Testing::QbmmRestart::append_restart_word(header, word);
            EXPECT_TRUE(std::ranges::starts_with(archive, header));
            if (gravity != 0. && velocity == Velocity::SharedCarrier)
            {
                Column mismatch(Geometry::Cartesian, velocity, 1, gravity, true, true, 1e-14);
                const auto before = mismatch.controller->serialize_qbmm_restart();
                const auto epoch = mismatch.mesh->geometry_epoch();
                EXPECT_THROW(mismatch.controller->restore_qbmm_restart(archive), std::exception);
                EXPECT_EQ(mismatch.controller->serialize_qbmm_restart(), before);
                EXPECT_EQ(mismatch.mesh->geometry_epoch(), epoch);
            }
            Column resumed(Geometry::Cartesian, velocity, 1, gravity, true, true);
            resumed.controller->restore_qbmm_restart(archive);
            compare_count_only_endpoints(column, resumed);
            advance(column, 1. - 1. / 512.);
            advance(resumed, 1. - 1. / 512.);
            EXPECT_EQ(resumed.flow->state().time, 2. * dt);
            compare_count_only_endpoints(column, resumed);
        }
}
} // namespace
