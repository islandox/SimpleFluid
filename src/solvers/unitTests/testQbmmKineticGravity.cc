/** Closed Cartesian gravity and independently prescribed two-size kinetic transport. */
#include "QbmmKineticTestSupport.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
using Coupler = R::ReactingMixtureCoupler;
using Vector = Flow::Vector;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double p0 = 64., temperature0 = 8., liquid_density = 16., gas_density = 8., bubble_mass = 1. / 16.;
constexpr double dt = 1. / 64., rise_speed = .125;
enum class Profile
{
    Uniform,
    Stratified,
    Plume,
    UniformRoundedColumn,
    Opposing
};

struct Case
{
    Profile profile = Profile::Uniform;
    bool lateral = false, rise = false;
    std::optional<Vector> force;
    bool equilibrium_head = false;
    double top_pressure_offset = 0.;
    bool retain_finer_top_pressure = false;
    size_t column_cells = 4;
};

using Sizes = std::array<double, 2>;

Sizes size_counts(double count, double scaled_mass)
{
    // Independent inversion on the fixture's two fixed mass classes, xi=1 and xi=3.
    const double coarse = .5 * (scaled_mass - count);
    return {count - coarse, coarse};
}

Sizes counts(const Flow::CellState& c)
{
    return size_counts(*c.phase_numbers[1], c.phases[1].moles[0] / bubble_mass);
}

double mass(const Flow::CellState& c)
{
    return c.phases[0].moles[0] + c.phases[1].moles[0];
}

Vector slip(Case spec, size_t size, double z)
{
    if (spec.profile == Profile::Opposing)
        return {0., (z < .5 ? 1. : -1.) * (size == 0 ? .0625 : .25), size == 0 ? rise_speed : -2. * rise_speed};
    return {0., 0., spec.rise ? rise_speed * (size + 1.) : 0.};
}

struct Fixture
{
    Case spec;
    size_t nx, nz;
    SP<Flow::Mesh> grid;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;

    explicit Fixture(Case input)
        : spec(input), nx(input.lateral ? 2 : 1), nz(input.profile == Profile::Opposing ? 2 : input.column_cells)
    {
        const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
        const ArrReal x = nx == 2 ? ArrReal{0., .5, 1.} : ArrReal{0., 1.};
        const ArrReal z = nz == 2 ? ArrReal{0., .5, 1.} : ArrReal{0., .25, .5, .75, 1.};
        grid = std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{x, {0., 1.}, z}),
            Flow::Mesh::DistributionOptions{
                .allow_empty_partitions = true, .orthogonal_partitions = std::array<size_t, 3>{1, 1, ranks}});
        record_ownership(*grid, static_cast<int>(nx * nz));
        const auto species = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "column material", 1., {{"M", 1}}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed}}});
        const R::ThermodynamicBounds bounds{2., 14., 16., 128.};
        const auto liquid = R::make_linear_compressible_mixture_liquid(species, bounds, [](auto)
            { return R::LinearCompressibleMixtureLiquid{temperature0, p0, liquid_density, 2., 0., 0., 16., 0., 0.}; });
        const auto gas = R::make_ideal_gas_mixture(
            species, {{2., temperature0, 16.}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
        Flow::Options opts;
        opts.minimum_pressure = bounds.minimum_pressure;
        opts.maximum_pressure = bounds.maximum_pressure;
        opts.linear.preconditioner = LinearPreconditioner::None;
        opts.linear.tolerance = 1e-12;
        opts.phase_roles = {
            {Flow::PhaseRole::BulkLiquid, "solvent"}, {Flow::PhaseRole::DispersedGas, "column bubbles"}};
        opts.qbmm_eos_law_identities = {
            R::ThermalLawIdentity{"column constant liquid", 1, species->fingerprint()}, std::nullopt};
        opts.qbmm_model.family_transport = Model::FamilyTransport::KineticEuler;
        opts.qbmm_model.velocity = Model::Velocity::NodeDependentSlip;
        opts.qbmm_model.kinetic_cfl = spec.profile == Profile::Opposing ? 1. / 100. : 1.;
        const R::QbmmLawMetadata metadata{"two prescribed column speeds", 1, R::QbmmLawKind::NodeSlip,
            "physical node mass and cell coordinate", "m/s"};
        R::QbmmClosureVersions versions;
        versions.laws.push_back(metadata);
        opts.qbmm_families = {
            nullptr, std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "column bubbles",
                         opts.phase_roles[1], bubble_mass, std::vector<R::QbmmMassFraction>{{47, 1.}}, versions)};
        opts.qbmm_node_slip_laws = {nullptr, std::make_shared<const R::QbmmNodeSlipLaw>(
                                                 R::QbmmNodeSlipLaw{metadata, [input](const R::QbmmNodeSlipInput& node)
                                                     {
                                                         const auto v = slip(input,
                                                             node.mass_kg_per_bubble < 2. * bubble_mass ? 0 : 1,
                                                             node.location.value().centroid[2]);
                                                         return std::array{v.x, v.y, v.z};
                                                     }})};
        flow = std::make_shared<Flow>(grid, std::vector{liquid, gas}, opts);
        Flow::State initial;
        initial.thermodynamic_pressure = p0;
        initial.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*opts.qbmm_families[1])};
        initial.cells.resize(grid->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const auto pos = grid->cell_centroid(static_cast<LO>(c));
            const auto j = static_cast<size_t>(nz * pos.z);
            double alpha = gas_fraction(j);
            if (spec.profile == Profile::Plume)
                alpha = pos.x < .5 && (j == 1 || j == 2) ? .5 : 0.;
            const bool rounded_column = spec.profile == Profile::UniformRoundedColumn;
            const double p = rounded_column ? 80. + j : head(pos) + (j + 1 == nz ? spec.top_pressure_offset : 0.);
            const double t = p / (rounded_column ? liquid_density : gas_density);
            const double volume = grid->cell_volume(static_cast<LO>(c));
            const double gas_moles = alpha * volume * p / t, liquid_moles = (1. - alpha) * volume * liquid_density;
            const double ratio = spec.profile == Profile::Opposing ? 4. : 2.;
            const double coarse = gas_moles / ((ratio + 3.) * bubble_mass), fine = ratio * coarse;
            auto& cell = initial.cells[c];
            cell.mechanical_pressure = p - p0;
            if (spec.retain_finer_top_pressure && j + 1 == nz)
                cell.mechanical_pressure = std::nextafter(cell.mechanical_pressure, 0.);
            // Store the declared EOS caloric value at the manufactured T,p. For the liquid,
            // evaluating h−p*v can differ by one ULP from algebraically simplifying u to 2*T.
            const std::array composition{1.};
            const double liquid_u = liquid->evaluate(t, p, composition).internal_energy;
            const double gas_u = gas->evaluate(t, p, composition).internal_energy;
            cell.phases = {{liquid, {liquid_moles}, p, liquid_moles * liquid_u, t},
                {gas, {gas_moles}, p, gas_moles * gas_u, gas_moles > 0. ? t : 0.}};
            cell.phase_numbers = {std::nullopt, fine + coarse};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{fine + 9. * coarse, fine + 27. * coarse}};
            if (spec.profile == Profile::Opposing)
                cell.velocity.y = pos.z < .5 ? .125 : -.125;
        }
        flow->initialize(std::move(initial));
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            species, std::vector<R::ReactionDefinition>{}, liquid->energy_convention());
        controls.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "column bubbles", {}}};
        controls.source_identity = "inert column";
        controls.source = [](const auto&, double) { return R::ReactionSourceInput{}; };
        controls.body_acceleration = spec.force;
        controller = std::make_unique<Coupler>(flow, std::move(controls));
    }

    double gas_fraction(size_t j) const { return spec.profile == Profile::Stratified ? (j + 1.) / 8. : .25; }
    double row_density(size_t j) const { return liquid_density + (gas_density - liquid_density) * gas_fraction(j); }

    double head(Vector p) const
    {
        if (!spec.equilibrium_head)
            return p0;
        // Each phase has a prescribed physical occupancy. Set T_j = p_j / 8, so the synthetic
        // ideal gas has rho_g=8 exactly even in a nonuniform head. The liquid is incompressible.
        // Integrate the trapezoidal edge mass density, then subtract its volume mean (gauge).
        // The density, mesh widths and gauge mean are dyadic in this fixture. Apply the
        // non-dyadic acceleration only after those exact geometric coefficients are known.
        std::array<double, 4> head_per_acceleration{};
        for (size_t j = 1; j < nz; ++j)
            head_per_acceleration[j] = head_per_acceleration[j - 1] + .5 * (row_density(j - 1) + row_density(j)) / nz;
        const double mean = std::accumulate(head_per_acceleration.begin(), head_per_acceleration.begin() + nz, 0.) / nz;
        const double lateral_head = spec.force->x * row_density(0) * (p.x - .5);
        return std::fma(head_per_acceleration[static_cast<size_t>(p.z * nz)] - mean, spec.force->z, p0 + lateral_head);
    }

    size_t index(LO cell) const
    {
        const auto p = grid->cell_centroid(cell);
        return static_cast<size_t>(p.z * nz) * nx + static_cast<size_t>(p.x * nx);
    }

    template<class T> T sum(T value) const { return reduce(*grid, Teuchos::REDUCE_SUM, value); }
    double maximum(double value) const { return reduce(*grid, Teuchos::REDUCE_MAX, value); }

    void metric(const std::string& label, double value) const
    {
        if (grid->owned_cell_map()->getComm()->getRank() == 0)
            std::cout << "Q8b " << label << '=' << std::setprecision(15) << value << '\n';
        std::ostringstream formatted;
        formatted << std::setprecision(17) << value;
        ::testing::Test::RecordProperty(label, formatted.str());
    }
};

void check_impulse(const Fixture& f, const Coupler::Trial& trial, double h)
{
    const auto acc = f.spec.force.value_or(Vector{});
    double local_mass = 0., energy_change = 0.;
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        local_mass += mass(cell);
        for (size_t a = 0; a < 2; ++a)
            energy_change += cell.phases[a].internal_energy - f.flow->state().cells[c].phases[a].internal_energy;
    }
    const double total_mass = f.sum(local_mass);
    const auto& d = trial.diagnostics().flow;
    for (int k = 0; k < 3; ++k)
    {
        EXPECT_NEAR(d.external_impulse.component(k), total_mass * acc.component(k) * h, 2e-12);
        if (f.spec.force)
        {
            EXPECT_TRUE(d.impulse_parts);
            if (d.impulse_parts)
            {
                EXPECT_NEAR(d.impulse_parts->body.component(k), total_mass * acc.component(k) * h, 2e-12);
                EXPECT_EQ(d.impulse_parts->other.component(k), 0.);
            }
        }
    }
    EXPECT_NEAR(f.sum(energy_change), d.pressure_work, 2e-9 + d.internal_energy_roundoff_bound);
    EXPECT_LE(d.maximum_occupancy_error, Flow::Options{}.volume_relative_tolerance);
    EXPECT_LE(d.maximum_continuity_residual, Flow::Options{}.continuity_tolerance);
    EXPECT_LE(d.maximum_shape_transport_residual, 1.);
    EXPECT_LE(d.maximum_species_transport_residual, 1.);
}

void equilibrium(Case spec)
{
    Fixture f(spec);
    const auto initial = f.flow->state();
    double speed = 0., rate = 0., defect = 0.;
    for (int step = 0; step < 3; ++step)
    {
        const auto trial = f.controller->preview(dt);
        check_impulse(f, trial, dt);
        f.controller->commit(trial);
        Flow::Traits::scalar_cell_type pi(f.grid, 0., "observed_column_pressure");
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const auto id = static_cast<LO>(c);
            pi.set_owned_value(id, f.flow->state().cells[c].mechanical_pressure);
            const auto& cell = f.flow->state().cells[c];
            const double p = f.flow->state().thermodynamic_pressure + cell.mechanical_pressure;
            EXPECT_NEAR(p, f.head(f.grid->cell_centroid(id)), 1e-6);
            EXPECT_NEAR(mass(cell) / f.grid->cell_volume(id),
                f.row_density(static_cast<size_t>(f.grid->cell_centroid(id).z * f.nz)), 2e-11);
            for (size_t size = 0; size < 2; ++size)
            {
                EXPECT_GT(counts(cell)[size], 0.);
                EXPECT_NEAR(counts(cell)[size], counts(initial.cells[c])[size], 2e-11);
            }
            speed = std::max(speed, std::sqrt(cell.velocity.x * cell.velocity.x + cell.velocity.y * cell.velocity.y +
                                              cell.velocity.z * cell.velocity.z));
        }
        pi.sync_ghosts();
        for (const auto face : f.flow->volume_flux().owned_face_ids())
        {
            rate = std::max(rate, std::abs(f.flow->volume_flux().owned_value(face)));
            if (!f.grid->is_interior_face(face))
                continue;
            const auto l = f.grid->owner_cell(face), r = f.grid->neighbor_cell(face);
            const auto dx = f.grid->cell_centroid(r) - f.grid->cell_centroid(l);
            const auto g = spec.force.value_or(Vector{});
            const double head = .5 * (f.flow->density().local_value(l) + f.flow->density().local_value(r)) *
                                (g.x * dx.x + g.y * dx.y + g.z * dx.z);
            defect = std::max(defect, std::abs(pi.local_value(r) - pi.local_value(l) - head));
        }
    }
    speed = f.maximum(speed);
    rate = f.maximum(rate);
    defect = f.maximum(defect);
    f.metric("rest_maximum_speed_m_s", speed);
    f.metric("rest_maximum_face_rate_m3_s", rate);
    f.metric("rest_maximum_edge_defect_Pa", defect);
    EXPECT_LT(speed, 1e-9);
    EXPECT_LT(rate, 1e-10);
    EXPECT_LT(defect, 1e-6);
}

TEST(QbmmKineticGravity, StratifiedRestAndReversedHead)
{
    for (double g : {-1., 1.})
        equilibrium({.profile = Profile::Stratified, .force = Vector{0., 0., g}, .equilibrium_head = true});
}

TEST(QbmmKineticGravity, StratifiedRestWithNonDyadicAcceleration)
{
    for (size_t cells : {4u, 2u})
        equilibrium({.profile = Profile::Stratified,
            .force = Vector{0., 0., -.981},
            .equilibrium_head = true,
            .column_cells = cells});
}

TEST(QbmmKineticGravity, ResolvedPerturbationOfNonDyadicHeadAccelerates)
{
    for (const double sign : {-1., 1.})
    {
        Fixture f({.profile = Profile::Stratified,
            .force = Vector{0., 0., -.981},
            .equilibrium_head = true,
            .top_pressure_offset = sign * 1.e-3});
        const auto trial = f.controller->preview(dt);
        check_impulse(f, trial, dt);
        double top_velocity = 0.;
        for (size_t c = 0; c < f.grid->num_owned_cells(); ++c)
            if (f.index(static_cast<LO>(c)) + 1 == f.nz)
                top_velocity = trial.state().cells[c].velocity.z;
        const double response = f.sum(top_velocity);
        f.metric("non_dyadic_perturbed_top_velocity_m_s", response);
        EXPECT_LT(sign * response, -1.e-7);
    }
}

void expect_unrepresentable_motion(Fixture& f)
{
    const auto revision = f.flow->state().revision;
    std::string rejection;
    try
    {
        (void) f.controller->preview(dt);
    }
    catch (const std::exception& e)
    {
        rejection = e.what();
    }
    EXPECT_FALSE(rejection.empty());
    EXPECT_GT(f.sum(int(rejection.find("too small to update its authoritative stock") != std::string::npos)), 0);
    EXPECT_EQ(f.flow->state().revision, revision);
}

TEST(QbmmKineticGravity, FinerMechanicalPressureIsAuthoritativeAtRest)
{
    Fixture f({.profile = Profile::Stratified,
        .force = Vector{0., 0., -.981},
        .equilibrium_head = true,
        .retain_finer_top_pressure = true});
    // The pressure datum carries a residual below an absolute-pressure ULP. It must reach ordinary transport.
    expect_unrepresentable_motion(f);
}

TEST(QbmmKineticGravity, AccumulatedHeadDefectExceedsOnePressureDatumInterval)
{
    const double ulp = std::nextafter(80., 81.) - 80.;
    const double g = (1. - .75 * ulp) / 4.;
    Fixture f({.profile = Profile::UniformRoundedColumn, .force = Vector{0., 0., g}});
    // The EOS gives rho=16 kg/m³ at every cell. Each measured pressure jump is 1 Pa,
    // only 3/4 ULP above the true edge head. Three such jumps cannot fit one common datum.
    const double edge_defect = 1. - 16. * .25 * g;
    EXPECT_EQ(edge_defect, .75 * ulp);
    EXPECT_GT(3. * edge_defect, ulp);
    for (size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_EQ(f.flow->density().owned_value(static_cast<LO>(c)), 16.);
    expect_unrepresentable_motion(f);
}

TEST(QbmmKineticGravity, ConstantDensityObliqueHeadAndAbsentForceRest)
{
    equilibrium({.lateral = true, .force = Vector{.5, 0., -1.}, .equilibrium_head = true});
    equilibrium({.profile = Profile::Stratified});
}

TEST(QbmmKineticGravity, FlatPressureStartAcceleratesWithSignedForce)
{
    for (double g : {-1., 1.})
    {
        Fixture f({.profile = Profile::Stratified, .force = Vector{0., 0., g}});
        const auto trial = f.controller->preview(dt);
        check_impulse(f, trial, dt);
        f.controller->commit(trial);
        double pressure_difference = 0., mean_velocity = 0.;
        for (size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        {
            const auto& cell = f.flow->state().cells[c];
            const size_t j = f.index(static_cast<LO>(c));
            pressure_difference += ((j == f.nz - 1) - (j == 0)) * cell.mechanical_pressure;
            mean_velocity += cell.velocity.z * f.grid->cell_volume(static_cast<LO>(c));
        }
        const double head = f.sum(pressure_difference), u = f.sum(mean_velocity);
        EXPECT_GT(g * head, 1e-6);
        EXPECT_GT(g * u, 1e-8);
        f.metric("flat_start_top_minus_bottom_Pa", head);
        f.metric("flat_start_bulk_vertical_velocity_m_s", u);
    }
}

struct PlumeMeasures
{
    Sizes inventory{}, first_moment{}, top{}, through_top_plane{};
    double gas_centroid = 0., liquid_downflow = 0., circulation = 0., pressure_head = 0., top_mass = 0.;
};

PlumeMeasures plume_stocks(const Fixture& f, const Flow::State& state)
{
    PlumeMeasures out;
    double gas_mass = 0., gas_height = 0.;
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto pos = f.grid->cell_centroid(static_cast<LO>(c));
        const auto& cell = state.cells[c];
        // Trapezoidal circulation along a closed rectangle joining the eight cell centres.
        // The liquid velocity equals the carrier velocity everywhere in this fixture.
        const bool bottom = pos.z < .25, top = pos.z > .75;
        out.circulation += .25 * (bottom - top) * cell.velocity.x +
                           .25 * (pos.x > .5 ? 1. : -1.) * (bottom || top ? .5 : 1.) * cell.velocity.z;
        const auto n = counts(cell);
        for (size_t a = 0; a < n.size(); ++a)
        {
            out.inventory[a] += n[a];
            out.first_moment[a] += n[a] * pos.z;
            if (pos.z > .75)
                out.top[a] += n[a];
        }
        gas_mass += cell.phases[1].moles[0];
        gas_height += cell.phases[1].moles[0] * pos.z;
        if (pos.z > .75)
            out.top_mass += cell.phases[1].moles[0];
        out.pressure_head += ((pos.z < .25) - (pos.z > .75)) * cell.mechanical_pressure / f.nx;
    }
    for (size_t a = 0; a < 2; ++a)
    {
        out.inventory[a] = f.sum(out.inventory[a]);
        out.first_moment[a] = f.sum(out.first_moment[a]);
        out.top[a] = f.sum(out.top[a]);
    }
    out.top_mass = f.sum(out.top_mass);
    out.gas_centroid = f.sum(gas_height) / f.sum(gas_mass);
    out.pressure_head = f.sum(out.pressure_head);
    out.circulation = f.sum(out.circulation);
    return out;
}

PlumeMeasures plume(bool rise, bool force)
{
    Fixture f({.profile = Profile::Plume,
        .lateral = true,
        .rise = rise,
        .force = force ? std::optional<Vector>{Vector{0., 0., -1.}} : std::nullopt});
    const auto initial = plume_stocks(f, f.flow->state());
    auto before = initial;
    Sizes crossing{};
    double liquid_return = 0.;
    int local_shared_transport = 0;
    for (int step = 0; step < 4; ++step)
    {
        SCOPED_TRACE(step);
        const auto trial = f.controller->preview(dt);
        check_impulse(f, trial, dt);
        Sizes vertical{}, to_top{};
        for (const auto face : f.flow->volume_flux().owned_face_ids())
        {
            const auto& trace = trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)];
            const auto& gas = trace.qbmm_flux.at(1).value();
            EXPECT_EQ(trace.phase_volume_flux[0], trace.carrier_volume_flux);
            if (!f.grid->is_interior_face(face))
            {
                EXPECT_EQ(trace.material_volume_flux, 0.);
                EXPECT_EQ(gas.population_number_flux, 0.);
                EXPECT_EQ(gas.derived_mass_flux, 0.);
                EXPECT_EQ(gas.shape_flux.H2, 0.);
                EXPECT_EQ(gas.shape_flux.H3, 0.);
                continue;
            }
            const auto l = f.grid->owner_cell(face), r = f.grid->neighbor_cell(face);
            const double dz = f.grid->cell_centroid(r).z - f.grid->cell_centroid(l).z;
            const auto rates = size_counts(gas.population_number_flux, gas.derived_mass_flux / bubble_mass);
            EXPECT_NEAR(gas.shape_flux.H2, rates[0] + 9. * rates[1], 2e-11);
            EXPECT_NEAR(gas.shape_flux.H3, rates[0] + 27. * rates[1], 2e-11);
            if (dz == 0.)
                continue;
            if ((!f.grid->is_owned_cell(l) || !f.grid->is_owned_cell(r)) && std::abs(rates[1]) > 1e-8)
                ++local_shared_transport;
            const double z = .5 * (f.grid->cell_centroid(r).z + f.grid->cell_centroid(l).z);
            const double sign = std::copysign(1., dz);
            const double liquid_rate = sign * trace.carrier_volume_flux * trace.material_fraction[0];
            if (z == .5)
                liquid_return += dt * std::max(-liquid_rate, 0.);
            for (size_t a = 0; a < 2; ++a)
            {
                vertical[a] += dt * dz * rates[a];
                if (z == .75)
                    to_top[a] += dt * sign * rates[a];
            }
        }
        const auto after = plume_stocks(f, trial.state());
        for (size_t a = 0; a < 2; ++a)
        {
            EXPECT_NEAR(after.inventory[a], initial.inventory[a], 2e-12);
            EXPECT_NEAR(after.first_moment[a] - before.first_moment[a], f.sum(vertical[a]), 2e-12);
            crossing[a] += f.sum(to_top[a]);
            EXPECT_NEAR(after.top[a] - initial.top[a], crossing[a], 2e-12);
        }
        EXPECT_NEAR(after.top_mass, bubble_mass * (after.top[0] + 3. * after.top[1]), 2e-12);
        f.controller->commit(trial);
        before = after;
    }
    before.through_top_plane = crossing;
    before.liquid_downflow = f.sum(liquid_return);
    const auto shared = f.sum(local_shared_transport);
    if (rise && f.grid->owned_cell_map()->getComm()->getSize() > 1)
        EXPECT_GT(shared, 0);
    if (rise)
    {
        const double fine = before.first_moment[0] / before.inventory[0];
        const double coarse = before.first_moment[1] / before.inventory[1];
        EXPECT_GT(coarse, fine + 1e-4);
        EXPECT_GT(before.gas_centroid, initial.gas_centroid + 1e-4);
        EXPECT_GT(before.through_top_plane[0], 0.);
        EXPECT_GT(before.through_top_plane[1] / before.inventory[1],
            before.through_top_plane[0] / before.inventory[0] + 1e-4);
        EXPECT_GT(before.top_mass, 0.);
        const std::string prefix = force ? "combined_" : "slip_only_";
        f.metric(prefix + "fine_number_centroid_m", fine);
        f.metric(prefix + "coarse_number_centroid_m", coarse);
        f.metric(prefix + "gas_mass_centroid_m", before.gas_centroid);
        f.metric(prefix + "mean_fine_count_rate_into_top_layer_per_s", crossing[0] / (4. * dt));
        f.metric(prefix + "mean_coarse_count_rate_into_top_layer_per_s", crossing[1] / (4. * dt));
        f.metric(prefix + "gas_retained_in_top_layer_kg", before.top_mass);
        f.metric(prefix + "liquid_return_volume_m3", before.liquid_downflow);
        f.metric(prefix + "liquid_circulation_m2_s", before.circulation);
        f.metric(prefix + "bottom_minus_top_Pa", before.pressure_head);
    }
    return before;
}

TEST(QbmmKineticGravity, LocalizedPlumeSeparatesSizesAndRetainsBothAtClosedLid)
{
    const auto quiet = plume(false, false), gravity = plume(false, true);
    const auto slip_only = plume(true, false), combined = plume(true, true);
    EXPECT_NEAR(quiet.gas_centroid, .5, 1e-12);
    EXPECT_NEAR(quiet.top_mass, 0., 1e-12);
    EXPECT_NEAR(quiet.pressure_head, 0., 1e-10);
    EXPECT_GT(combined.gas_centroid, gravity.gas_centroid + 1e-4);
    EXPECT_GT(combined.top_mass, gravity.top_mass + 1e-5);
    EXPECT_GT(combined.liquid_downflow, 1e-7);
    EXPECT_GT(std::abs(combined.circulation), 1e-5);
    EXPECT_NEAR(quiet.circulation, 0., 1e-11);
    EXPECT_GT(combined.pressure_head, slip_only.pressure_head + 1e-3);
}

TEST(QbmmKineticGravity, ExplicitZeroUsesInteriorPressureGradientInDrivenColumn)
{
    constexpr double h = .0625, dz = .5;
    for (const bool supplied : {false, true})
    {
        Fixture f({.profile = Profile::Opposing, .force = supplied ? std::optional{Vector{}} : std::nullopt});
        const auto old = f.flow->state();
        const auto trial = f.controller->preview(h);
        double pressure_jump = 0., momentum_change = 0.;
        for (size_t c = 0; c < old.cells.size(); ++c)
        {
            const auto& before = old.cells[c];
            const auto& after = trial.state().cells[c];
            pressure_jump += (f.index(static_cast<LO>(c)) == 0 ? -1. : 1.) * after.mechanical_pressure;
            momentum_change += mass(after) * after.velocity.z - mass(before) * before.velocity.z +
                               after.qbmm_kinetic_closure[1]->drift_momentum.z -
                               before.qbmm_kinetic_closure[1]->drift_momentum.z;
        }
        pressure_jump = f.sum(pressure_jump);
        ASSERT_GT(std::abs(pressure_jump), 1e-6);
        // The two unit-area rows occupy 1 m³. An interior LS sample contributes dz² to the
        // normal equation. A homogeneous-Neumann wall adds (dz/2)² to its diagonal and zero to its RHS.
        const double fraction = supplied ? 1. : 4. / 5.;
        const double pressure_impulse = -h * fraction * pressure_jump / dz;
        EXPECT_NEAR(trial.diagnostics().flow.pressure_impulse.z, pressure_impulse, 2e-10);
        EXPECT_NEAR(f.sum(momentum_change), pressure_impulse, 2e-10);
    }
}

TEST(QbmmKineticGravity, OpposingSubcycledAtomsReplayDirectionalMomentumAndOneBodyImpulse)
{
    constexpr double h = .0625, dz = .5, area = 1., volume = dz * area;
    Fixture f({.profile = Profile::Opposing, .force = Vector{0., 0., -.5}});
    const auto old = f.flow->state();
    const auto trial = f.controller->preview(h);
    check_impulse(f, trial, h);
    const auto report = f.controller->qbmm_report(trial);
    ASSERT_EQ(f.sum(int(bool(report.interval))), f.grid->owned_cell_map()->getComm()->getSize());
    ASSERT_GT(report.kinetic_substeps, 1);
    std::vector<double> duration(report.kinetic_substeps, 0.);
    for (const auto& law : report.interval->local_kinetic_law_receipts)
        duration.at(law.substep) = law.duration;
    for (auto& d : duration)
        d = f.maximum(d);
    EXPECT_NEAR(std::accumulate(duration.begin(), duration.end(), 0.), h, 1e-16);

    // Reconstruct both donors at each actual substep on this two-cell mesh, from the
    // uniform initial EOS inventory and the frozen carrier face rate, without inversion.
    double phi = 0.;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
        if (f.grid->is_interior_face(face))
            phi += f.grid->face_area_vector(face).z *
                   trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)].carrier_volume_flux;
    phi = f.sum(phi);
    constexpr double fine = 4. * .25 * volume * gas_density / (7. * bubble_mass), coarse = .25 * fine;
    std::array<Sizes, 2> weights{{{fine, coarse}, {fine, coarse}}};
    std::array<double, 2> signed_mass{};
    std::array<Vector, 2> signed_drift{};
    for (double d : duration)
    {
        const auto predecessor = weights;
        for (size_t donor = 0; donor < 2; ++donor)
            for (size_t a = 0; a < 2; ++a)
            {
                const auto v = slip(f.spec, a, donor == 0 ? .25 : .75);
                const double oriented = phi + v.z * area, sign = donor == 0 ? 1. : -1.;
                const double n = d * std::max(sign * oriented, 0.) * predecessor[donor][a] / volume;
                const double kg = n * bubble_mass * (a == 0 ? 1. : 3.);
                weights[donor][a] -= n;
                weights[1 - donor][a] += n;
                signed_mass[donor] += sign * kg;
                signed_drift[donor].y += sign * kg * v.y;
                signed_drift[donor].z += sign * kg * v.z;
            }
    }
    EXPECT_GT(signed_mass[0], 0.);
    EXPECT_LT(signed_mass[1], 0.);
    double y_momentum_flux = 0., local_drift_change = 0., local_delta_physical_z = 0.;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
    {
        if (!f.grid->is_interior_face(face))
            continue;
        const auto sign = f.grid->face_area_vector(face).z;
        const auto& p = trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)];
        const auto& packets = p.qbmm_flux.at(1)->kinetic.value();
        for (size_t donor = 0; donor < 2; ++donor)
        {
            const auto& directional = ((donor == 0) == (sign > 0.)) ? packets.forward : packets.backward;
            EXPECT_NEAR(directional.mass_flux, sign * signed_mass[donor] / h, 2e-12);
            EXPECT_NEAR(directional.drift_momentum_flux.y, sign * signed_drift[donor].y / h, 2e-12);
            EXPECT_NEAR(directional.drift_momentum_flux.z, sign * signed_drift[donor].z / h, 2e-12);
        }
        const double liquid_rate = phi * p.material_fraction[0] * liquid_density;
        const double upward = signed_mass[0] / h + std::max(liquid_rate, 0.);
        const double downward = -signed_mass[1] / h + std::max(-liquid_rate, 0.);
        EXPECT_NEAR(p.forward_mass_flux, sign > 0. ? upward : downward, 2e-12);
        EXPECT_NEAR(p.backward_mass_flux, sign > 0. ? downward : upward, 2e-12);
        const auto& bottom = sign > 0. ? p.momentum_owner_velocity : p.momentum_neighbor_velocity;
        const auto& top = sign > 0. ? p.momentum_neighbor_velocity : p.momentum_owner_velocity;
        const double carrier = upward * bottom.y - downward * top.y;
        EXPECT_NEAR(sign * p.implicit_carrier_momentum_flux.y, carrier, 2e-12);
        y_momentum_flux = carrier + (signed_drift[0].y + signed_drift[1].y) / h;
    }
    y_momentum_flux = f.sum(y_momentum_flux);
    for (size_t c = 0; c < old.cells.size(); ++c)
    {
        const auto j = f.index(static_cast<LO>(c));
        const auto& after = trial.state().cells[c];
        Vector d_old{}, d_new{};
        for (size_t a = 0; a < 2; ++a)
        {
            EXPECT_NEAR(counts(after)[a], weights[j][a], 2e-12);
            const double kg = bubble_mass * (a == 0 ? 1. : 3.);
            const auto v = slip(f.spec, a, j == 0 ? .25 : .75);
            const double old_kg = (a == 0 ? fine : coarse) * kg, new_kg = weights[j][a] * kg;
            d_old.y += old_kg * v.y;
            d_old.z += old_kg * v.z;
            d_new.y += new_kg * v.y;
            d_new.z += new_kg * v.z;
        }
        const auto& stored = after.qbmm_kinetic_closure.at(1)->drift_momentum;
        EXPECT_NEAR(stored.y, d_new.y, 2e-12);
        EXPECT_NEAR(stored.z, d_new.z, 2e-12);
        local_drift_change += std::abs(d_new.y - d_old.y);
        // Tangential momentum has no pressure gradient, viscosity or body impulse.
        // This row detects both a missing D change and substitution of net-rate upwinding.
        const double change_y =
            mass(after) * after.velocity.y - mass(old.cells[c]) * old.cells[c].velocity.y + d_new.y - d_old.y;
        EXPECT_NEAR(change_y, (j == 0 ? -h : h) * y_momentum_flux, 2e-10);
        local_delta_physical_z +=
            mass(after) * after.velocity.z - mass(old.cells[c]) * old.cells[c].velocity.z + d_new.z - d_old.z;
        const auto& gross = report.interval->local_gross_events.at(c);
        // There is a single nonzero source channel in this inert, unheated problem.
        EXPECT_EQ(std::count_if(gross.begin(), gross.end(), [](double x) { return x != 0.; }), 1);
        EXPECT_NEAR(std::accumulate(gross.begin(), gross.end(), 0.), mass(after) * .5 * h, 2e-12);
    }
    const auto& d = trial.diagnostics().flow;
    EXPECT_GT(f.sum(local_drift_change), 1e-5);
    EXPECT_NEAR(f.sum(local_delta_physical_z), d.pressure_impulse.z + d.external_impulse.z, 2e-10);
    f.metric("opposing_substeps", report.kinetic_substeps);
    f.metric("upward_gas_mass_kg", signed_mass[0]);
    f.metric("downward_gas_mass_kg", -signed_mass[1]);
    f.metric("body_impulse_kg_m_per_s", d.external_impulse.z);
}
} // namespace
