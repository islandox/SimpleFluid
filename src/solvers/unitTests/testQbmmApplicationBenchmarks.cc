/** Q6 manufactured application benchmarks, not empirical bubble-model validation.
 *
 * Closed unit-depth x/z cavity, initially at rest, with a supplied common phase
 * slip, gravity, differential wall heat flux and interior Fourier contact.
 * Wall power enters the adjacent CV through the existing extensive heat source;
 * no open material boundary or prescribed-temperature reservoir is implied.
 * Spatial/time sensitivity and population-closure sensitivity are separate tests.
 */
#include "ConditionedThermalTestSupport.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace Thermal = Testing::ConditionedTransfer;
using Solver = R::LowMachMixtureSolver;
using Controller = R::ReactingMixtureCoupler;
using Phase = R::SpeciesPhase;
using LO = Solver::Pack::local_ordinal_type;
using Operations = std::vector<R::QbmmEventOperation>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double mass_scale = 1. / 1024, p0 = 100000, t0 = 300;
constexpr double liquid_u0 = 2 * Thermal::cv / Thermal::gas_constant * p0;
constexpr double step_size = 1. / 64, horizon = 1. / 8;
constexpr double mean_wall_power = 512, differential_wall_power = 8192, conductivity = 64;
constexpr double slip_speed = 1. / 64;
constexpr std::array<double, 2> molar_mass{1, .028};

enum class PopulationLaw
{
    Application,
    Nonlinear,
    None
};

// Independent moment summation, also used to initialize a broad three-atom
// measure in the closure study. No production moment summation/inversion oracle.
R::QbmmMomentBlock moments(std::span<const R::QbmmAtom> atoms)
{
    R::QbmmMomentBlock result{};
    for (const auto& atom : atoms)
        for (unsigned k = 0; k < 4; ++k)
            result[k] += atom.count * std::pow(atom.mass_over_scale, k);
    return result;
}

R::QbmmMaterialTransfer liquid_transfer(double scaled_mass)
{
    return {R::QbmmPhaseExchange{Phase::Liquid}, {mass_scale * scaled_mass, 0}};
}

// Synthetic nonlinear jump-growth, Brownian-like coalescence and binary breakup.
// Coordinate is x=m/m*, weights are counts per bulk m3. The fixed support and
// rate cutoffs are part of this manufactured model, identical for both closures.
// These coefficients are numerical fixtures, without an empirical validity claim.
constexpr double maximum_size = 16, minimum_breakup_size = .25, birth_rate = .125;
double growth_rate(double x)
{
    return x * 1.125 <= maximum_size ? .04 * std::pow(x, 2. / 3) : 0;
}
double breakup_rate(double x)
{
    return x >= minimum_breakup_size ? .02 * std::pow(x, .75) : 0;
}
double collision_kernel(double x, double y)
{
    return x + y <= maximum_size ? .01 * std::pow(std::cbrt(x) + std::cbrt(y), 2) / std::sqrt(1 + x + y) : 0;
}

Operations nonlinear_events(const R::MixtureQbmmContext& in, double volume)
{
    Operations result{
        R::QbmmBirth{{{{birth_rate * volume * in.time_step, 1}, liquid_transfer(birth_rate * volume * in.time_step)}}}};
    for (unsigned i = 0; i < in.quadrature.active_atoms; ++i)
    {
        const auto [weight, x] = in.quadrature.atoms[i];
        const double grow = in.time_step * weight * growth_rate(x);
        const double split = in.time_step * weight * breakup_rate(x);
        if (grow > 0)
            result.emplace_back(R::QbmmResize{i, grow, 1.125 * x, liquid_transfer(grow * .125 * x)});
        if (split > 0)
            result.emplace_back(R::QbmmBreakup{i, split, {{2, x / 2}}});
        for (unsigned j = i; j < in.quadrature.active_atoms; ++j)
        {
            const auto [other, y] = in.quadrature.atoms[j];
            const double pairs = in.time_step * collision_kernel(x, y) * weight * other / volume * (i == j ? .5 : 1);
            if (pairs > 0)
                result.emplace_back(R::QbmmCoalescence{i, j, pairs});
        }
    }
    return result;
}

struct CaseOptions
{
    int cells_per_side = 4;
    double gravity = -9.81, differential_heat = differential_wall_power, mean_heat = mean_wall_power;
    double slip = slip_speed;
    PopulationLaw law = PopulationLaw::Application;
    bool broad_initial_distribution = false;
};

struct Fixture
{
    CaseOptions config;
    std::shared_ptr<Solver::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<const R::InterphaseConfiguration> transfers;
    std::shared_ptr<Solver> solver;
    int population_calls = 0;

    explicit Fixture(CaseOptions options = {}, std::optional<std::array<size_t, 3>> partitions = {}) : config(options)
    {
        ArrReal cuts;
        for (int i = 0; i <= config.cells_per_side; ++i)
            cuts.push_back(double(i) / config.cells_per_side);
        Solver::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        distribution.orthogonal_partitions = partitions;
        mesh = std::make_shared<Solver::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{cuts, ArrReal{0, 1}, cuts}), distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{44, "solvent/vapor", 1, {{"S", 1}}, {Phase::Liquid, Phase::Dispersed}},
                {709, "background gas", .028, {{"G", 1}}, {Phase::Dissolved, Phase::Resolved}}});
        constexpr R::ThermodynamicBounds bounds{250, 1200, 50000, 1000000};
        constexpr auto convention = R::EnergyConvention::SensibleEnergy;
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{t0, p0, 1, 1000, 0, 0, liquid_u0, 0, 0}; }, convention);
        gas = R::make_ideal_gas_mixture(catalog,
            std::vector<R::IdealGasCaloricSpecies>(2, {Thermal::cv, t0, Thermal::cv * t0}), bounds, convention);
        mechanism =
            std::make_shared<const R::ReactionMechanism>(catalog, std::vector<R::ReactionDefinition>{}, convention);
        R::InterphaseConfiguration slots;
        slots.metadata = {"Q6 finite solvent to family", 1, catalog->fingerprint()};
        slots.stages = {{"paired growth and birth", R::InterphaseStageKind::Material, {{44, 0, 1}}, {}}};
        auto partials = std::make_shared<const R::PartialMolarEnthalpyLaw>(
            R::PartialMolarEnthalpyLaw{{"Q6 analytic liquid enthalpy", 1, catalog->fingerprint()}, bounds,
                [](double t, double p, std::span<const double>)
                {
                    const double h = liquid_u0 + 1000 * (t - t0) + p;
                    return R::PartialMolarEnthalpyPoint{{h, .028 * h}, {1000, 28}, {1, .028}};
                }});
        slots.phase_enthalpy = {partials, nullptr, nullptr};
        transfers = std::make_shared<const R::InterphaseConfiguration>(std::move(slots));
        Solver::Options controls;
        controls.minimum_pressure = bounds.minimum_pressure;
        controls.maximum_pressure = bounds.maximum_pressure;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.phase_roles = {{Solver::PhaseRole::BulkLiquid, "solution"}, {Solver::PhaseRole::DispersedGas, "vapor"},
            {Solver::PhaseRole::BulkGas, "background"}};
        controls.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
        R::QbmmClosureVersions policies;
        policies.source_identity = "Q6 conservative population law";
        auto family = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "vapor",
            controls.phase_roles[1], mass_scale, std::vector<R::QbmmMassFraction>{{44, 1}, {709, 0}}, policies);
        controls.qbmm_families = {nullptr, family, nullptr};
        solver = std::make_shared<Solver>(mesh, std::vector{liquid, gas, gas}, controls);
        Solver::State initial;
        initial.thermodynamic_pressure = p0;
        initial.qbmm_families = {std::nullopt, Solver::qbmm_family_stamp(*family), std::nullopt};
        initial.cells.resize(mesh->num_owned_cells());
        const std::vector<R::QbmmAtom> atoms = config.broad_initial_distribution
                                                   ? std::vector<R::QbmmAtom>{{2, 1}, {1, 2}, {1, 4}}
                                                   : std::vector<R::QbmmAtom>{{2, 1}, {3, 4}};
        const auto m = moments(atoms);
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            auto& cell = initial.cells[c];
            const double volume = mesh->cell_volume(static_cast<LO>(c));
            const double solvent = .375 * volume, vapor = m[1] * mass_scale * volume;
            const double background = .625 * volume * p0 / (Thermal::gas_constant * t0) - vapor;
            cell.phases = {{liquid, {solvent, 0}, p0, solvent * liquid_u0, t0},
                {gas, {vapor, 0}, p0, vapor * Thermal::cv * t0, t0},
                {gas, {0, background}, p0, background * Thermal::cv * t0, t0}};
            cell.phase_numbers = {std::nullopt, m[0] * volume, std::nullopt};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{m[2] * volume, m[3] * volume}, std::nullopt};
            cell.drift_velocity = {{}, {0, 0, config.slip}, {}};
            // No seeded vortex: heating and supplied forces must produce circulation.
            cell.velocity = {};
        }
        solver->initialize(std::move(initial));
    }

    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }

    std::vector<double> sum(const std::vector<double>& local) const
    {
        std::vector<double> global(local.size());
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
            local.data(), global.data());
        return global;
    }

    double heat_density(size_t c) const
    {
        const double x = mesh->cell_centroid(static_cast<LO>(c)).x;
        const double dx = 1. / config.cells_per_side;
        // Integrate the two wall Neumann powers [W/m2] over their actual unit-depth
        // face areas: V/dx=A. On the one-cell case both walls contribute.
        return (x < dx ? (config.mean_heat + config.differential_heat) / dx : 0) +
               (x > 1 - dx ? (config.mean_heat - config.differential_heat) / dx : 0);
    }

    R::ReactingMixtureOptions options()
    {
        R::ReactingMixtureOptions result;
        result.mechanism = mechanism;
        result.phases = {{Phase::Liquid, {}, {Phase::Liquid, Phase::Dissolved}}, {Phase::Dispersed, "vapor", {}},
            {Phase::Resolved, {}, {}}};
        result.source_identity = "Q6 opposing wall flux and independent net heat";
        // Keep the native owner's existing evaluation budget for every control.
        // All source/transport/thermal/conservation tolerances remain defaults.
        result.maximum_correctors = solver->options().maximum_correctors;
        if (config.gravity != 0)
            result.body_acceleration = Solver::Vector{0, 0, config.gravity};
        result.source = [this](const R::CoupledReactionContext& in, double)
        {
            R::ReactionSourceInput input;
            input.heat.rate = R::VolumetricHeatRate{heat_density(in.owned_cell), R::SourceVolumeBasis::BulkVolume};
            return input;
        };
        R::MixtureFaceTransportOptions contacts;
        contacts.identity = "Q6 liquid Fourier contacts";
        contacts.schema_fingerprint = catalog->fingerprint();
        contacts.physical_phase = 0;
        contacts.fourier = R::MixtureFourierLaw{"Q6 k*A/d", 1, catalog->fingerprint(),
            [](const R::MixtureFaceContext& face) { return conductivity * face.area / face.center_distance; }};
        result.face_transport = {std::move(contacts)};
        result.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{{"Q6 conservative population law", 1, catalog->fingerprint()}, transfers,
                [this](const R::MixtureQbmmContext& in) -> Operations
                {
                    ++population_calls;
                    const double volume = mesh->cell_volume(static_cast<LO>(in.owned_cell));
                    if (config.law == PopulationLaw::None)
                        return {};
                    if (config.law == PopulationLaw::Nonlinear)
                        return nonlinear_events(in, volume);
                    const double birth = volume * in.time_step * (.25 + .5 * in.accepted_time);
                    const auto [a, x] = in.quadrature.atoms[0];
                    const unsigned last = in.quadrature.active_atoms - 1;
                    const auto [b, y] = in.quadrature.atoms[last];
                    const double events = in.time_step * std::min(a, b) / 8;
                    return {R::QbmmBirth{{{{birth, 1}, liquid_transfer(birth)}}},
                        R::QbmmResize{0, events, 1.125 * x, liquid_transfer(events * .125 * x)},
                        R::QbmmCoalescence{0, last, events}, R::QbmmBreakup{last, events, {{2, y / 2}}}};
                }});
        return result;
    }
};

double phase_mass(const R::PhaseThermalState& phase)
{
    return std::inner_product(phase.moles.begin(), phase.moles.end(), molar_mass.begin(), 0.);
}

double phase_volume(const R::PhaseThermalState& phase, size_t p)
{
    return p == 0 ? phase_mass(phase)
                  : std::accumulate(phase.moles.begin(), phase.moles.end(), 0.) * Thermal::gas_constant *
                        phase.temperature / phase.absolute_pressure;
}

struct Inventory
{
    std::array<double, 2> species{};
    R::QbmmMomentBlock population{};
    std::array<double, 3> momentum{};
    double mass = 0, energy = 0;
};

Inventory inventory(const Fixture& f, const Solver::State& state)
{
    Inventory result;
    for (const auto& cell : state.cells)
    {
        result.population[0] += *cell.phase_numbers[1];
        result.population[1] += cell.phases[1].moles[0] / mass_scale;
        result.population[2] += cell.qbmm_shape[1]->H2;
        result.population[3] += cell.qbmm_shape[1]->H3;
        for (size_t p = 0; p < cell.phases.size(); ++p)
        {
            const auto& phase = cell.phases[p];
            result.energy += phase.internal_energy;
            for (size_t s = 0; s < 2; ++s)
                result.species[s] += phase.moles[s];
            // Physical momentum includes all prescribed phase-relative motion.
            for (size_t d = 0; d < 3; ++d)
                result.momentum[d] +=
                    phase_mass(phase) * (cell.velocity.component(d) + cell.drift_velocity[p].component(d));
        }
    }
    for (double& value : result.species)
        value = f.sum(value);
    for (double& value : result.population)
        value = f.sum(value);
    for (double& value : result.momentum)
        value = f.sum(value);
    result.energy = f.sum(result.energy);
    result.mass = std::inner_product(result.species.begin(), result.species.end(), molar_mass.begin(), 0.);
    return result;
}

void expect_relative(double actual, double expected, double scale, double tolerance = 2e-10)
{
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_NEAR(actual, expected, tolerance * std::max(scale, std::numeric_limits<double>::min()));
}

// Complete canonical endpoint and accepted flux; thermodynamic provider pointers
// intentionally differ after portable restart into a fresh owner.
struct Endpoint
{
    Solver::State state;
    std::vector<double> flux;
    explicit Endpoint(const Fixture& f) : state(f.solver->state())
    {
        for (size_t face = 0; face < f.mesh->num_owned_faces(); ++face)
            flux.push_back(f.solver->volume_flux().owned_value(static_cast<LO>(face)));
    }
    void expect_same(const Fixture& f) const
    {
        const auto& other = f.solver->state();
        EXPECT_EQ(other.time, state.time);
        EXPECT_EQ(other.revision, state.revision);
        EXPECT_EQ(other.thermodynamic_pressure, state.thermodynamic_pressure);
        EXPECT_EQ(other.qbmm_families, state.qbmm_families);
        EXPECT_EQ(other.cells.size(), state.cells.size());
        for (size_t c = 0; c < std::min(other.cells.size(), state.cells.size()); ++c)
        {
            const auto& a = state.cells[c];
            const auto& b = other.cells[c];
            EXPECT_EQ(a.phase_numbers, b.phase_numbers);
            EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
            EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
            for (size_t d = 0; d < 3; ++d)
                EXPECT_EQ(a.velocity.component(d), b.velocity.component(d));
            for (size_t p = 0; p < a.phases.size(); ++p)
            {
                EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
                EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
                EXPECT_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
                EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
                for (size_t d = 0; d < 3; ++d)
                    EXPECT_EQ(a.drift_velocity[p].component(d), b.drift_velocity[p].component(d));
            }
        }
        EXPECT_EQ(flux.size(), f.mesh->num_owned_faces());
        for (size_t face = 0; face < flux.size(); ++face)
            EXPECT_EQ(flux[face], f.solver->volume_flux().owned_value(static_cast<LO>(face)));
    }
};

double checked_step(Fixture& f, Controller& controller, double dt)
{
    const auto before = inventory(f, f.solver->state());
    const Endpoint accepted(f);
    const auto trial = controller.preview(dt);
    accepted.expect_same(f);
    const auto after = inventory(f, trial.state());
    const auto report = controller.qbmm_report(trial);
    const auto& diag = trial.diagnostics().flow;
    EXPECT_GE(diag.source_evaluations, 2);
    EXPECT_LE(diag.maximum_event_residual, 1);
    EXPECT_LE(diag.maximum_shape_source_residual, 1);
    EXPECT_LE(diag.maximum_shape_transport_residual, 1);
    EXPECT_LE(diag.maximum_species_transport_residual, 1);
    EXPECT_LE(diag.maximum_energy_transport_residual, 1);
    EXPECT_LE(diag.maximum_population_transport_residual, 1);
    EXPECT_LE(diag.maximum_occupancy_error, f.solver->options().volume_relative_tolerance);
    EXPECT_LE(diag.maximum_continuity_residual, f.solver->options().continuity_tolerance);
    for (size_t s = 0; s < 2; ++s)
        expect_relative(after.species[s], before.species[s], before.species[s]);
    expect_relative(after.mass, before.mass, before.mass);
    const double heat = 2 * f.config.mean_heat * dt;
    expect_relative(trial.diagnostics().external_heat, heat, std::max(std::abs(heat), 1.), 1e-12);
    expect_relative(trial.diagnostics().net_thermal_energy, heat, std::max(std::abs(heat), 1.), 2e-9);

    constexpr size_t phases = 3;
    const auto global_cells = f.mesh->owned_cell_map()->getGlobalNumElements();
    const auto global_faces = f.mesh->owned_face_map()->getGlobalNumElements();
    const std::array wall_heat{
        dt * (f.config.mean_heat + f.config.differential_heat), dt * (f.config.mean_heat - f.config.differential_heat)};
    std::vector<double> material_change(phases * global_cells), fourier_endpoints(2 * global_faces);
    const auto face_packets = trial.owned_phase_transport();
    const auto& thermal_plan = trial.interphase_receipt();
    double work = 0, heat_from_cells = 0;
    std::array<double, 2> wall_areas{}, heat_from_walls{};
    R::QbmmMomentBlock event_delta{};
    std::array<double, 4> activity{};
    collective_detail::collective_local_validation(*f.mesh, "Q6 application material face balances",
        [&]
        {
            if (face_packets.size() != f.mesh->num_owned_faces())
                throw std::runtime_error("Missing application face packets.");
            for (size_t face = 0; face < face_packets.size(); ++face)
            {
                const auto id = static_cast<LO>(face);
                const auto& packet = face_packets[face];
                EXPECT_EQ(packet.mesh_volume_flux, 0);
                const auto owner = f.mesh->cell_geometry_global_id(f.mesh->owner_cell(id));
                const bool exterior = f.mesh->is_geometry_exterior_face(id);
                const auto neighbor = exterior ? owner : f.mesh->cell_geometry_global_id(f.mesh->neighbor_cell(id));
                for (size_t p = 0; p < phases; ++p)
                {
                    const double traced_rate = packet.relative_phase_volume_flux.at(p) * packet.material_fraction.at(p);
                    double rate = traced_rate;
                    if (p == 1)
                    {
                        rate = packet.qbmm_flux.at(p).value().material_volume_flux;
                        EXPECT_DOUBLE_EQ(rate, traced_rate);
                    }
                    if (exterior)
                        EXPECT_EQ(rate, 0);
                    else
                    {
                        material_change.at(phases * owner + p) -= dt * rate;
                        material_change.at(phases * neighbor + p) += dt * rate;
                    }
                }
            }
        });
    // An owned face contributes to both geometric cells, including the remote half of a shared face.
    material_change = f.sum(material_change);
    // All rank-local layout/constitutive failures are fenced before reductions.
    collective_detail::collective_local_validation(*f.mesh, "Q6 application budgets",
        [&]
        {
            const auto nc = f.mesh->num_owned_cells();
            if (!thermal_plan || thermal_plan->cells.size() != nc || trial.local_sources().size() != nc ||
                trial.final_thermal_evaluations().size() != nc || trial.state().cells.size() != nc)
                throw std::runtime_error("Missing application cell source or thermal receipts.");
            std::vector<std::array<double, 2>> cell_wall_area(nc);
            for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
            {
                const auto face = static_cast<LO>(i);
                if (!f.mesh->is_geometry_exterior_face(face))
                    continue;
                const double nx = f.mesh->face_normal(face).x;
                if (nx == 0)
                    continue;
                const size_t wall = nx < 0 ? 0 : 1;
                const double area = f.mesh->face_area(face);
                wall_areas[wall] += area;
                cell_wall_area.at(f.mesh->owner_cell(face))[wall] += area;
            }
            for (size_t c = 0; c < trial.state().cells.size(); ++c)
            {
                const auto id = static_cast<LO>(c);
                const auto geometry_cell = f.mesh->cell_geometry_global_id(id);
                const auto& cell = trial.state().cells[c];
                const auto& transported = trial.transport_snapshot().cells.at(c);
                const double volume = f.mesh->cell_volume(id);
                const double external_heat = trial.local_sources().at(c).externalHeat;
                const auto& area = cell_wall_area[c];
                const double wall_packet = std::inner_product(area.begin(), area.end(), wall_heat.begin(), 0.);
                expect_relative(external_heat, wall_packet, std::max(std::abs(wall_packet), 1.), 1e-12);
                heat_from_cells += external_heat;
                if (global_cells > 1)
                    for (size_t wall = 0; wall < area.size(); ++wall)
                        if (area[wall] > 0)
                            heat_from_walls[wall] += external_heat;

                std::vector<LO> contacts;
                for (const auto face : f.mesh->faces(id))
                    if (!f.mesh->is_geometry_exterior_face(face))
                        contacts.push_back(face);
                // The sole face stage appends one raw entry per incident contact in geometry-ID order.
                const auto& raw = thermal_plan->cells.at(c).raw_static_energy.at(0);
                if (raw.size() < contacts.size())
                    throw std::runtime_error("Missing applied Fourier contact entries.");
                std::ranges::sort(contacts, {}, [&](LO face) { return f.mesh->face_geometry_global_id(face); });
                const auto first_contact = raw.end() - static_cast<std::ptrdiff_t>(contacts.size());
                expect_relative(static_cast<double>(std::accumulate(raw.begin(), first_contact, 0.L)), external_heat,
                    std::max(std::abs(external_heat), 1.), 1e-12);
                for (size_t i = 0; i < contacts.size(); ++i)
                {
                    const auto face = contacts[i];
                    const auto geometry_face = f.mesh->face_geometry_global_id(face);
                    const bool neighbor = f.mesh->owner_cell(face) != id;
                    fourier_endpoints.at(2 * geometry_face + neighbor) = static_cast<double>(first_contact[i]);
                }

                double occupied = 0;
                for (size_t p = 0; p < phases; ++p)
                {
                    const auto& phase = cell.phases.at(p);
                    const auto& advected = transported.phases.at(p);
                    const double v = phase_volume(phase, p);
                    occupied += v;
                    const double old_volume = phase_volume(accepted.state.cells.at(c).phases.at(p), p);
                    const double advected_volume = old_volume + material_change.at(phases * geometry_cell + p);
                    expect_relative(advected.advected_material_volume, advected_volume, volume, 1e-12);
                    const double pdv = -phase.absolute_pressure * (v - advected_volume);
                    work += pdv;
                    const auto& applied = trial.final_thermal_evaluations()[c].applied_phase_energy;
                    expect_relative(phase.internal_energy, advected.internal_energy + applied.at(p) + pdv,
                        std::abs(phase.internal_energy), 4e-9);
                }
                expect_relative(occupied, volume, volume, f.solver->options().volume_relative_tolerance);
                const auto& family = report.cells.at(c).families.at(1).value();
                EXPECT_EQ(family.quadrature.status, R::InversionStatus::TwoNodes);
                EXPECT_TRUE(family.closure_evaluated);
                EXPECT_FALSE(family.closure_validated);
                EXPECT_GT(family.geometry->area_density, 0);
                EXPECT_GT(family.geometry->d32.value(), 0);
                for (const auto residual : family.quadrature.reconstruction.relative)
                    EXPECT_LE(residual, R::qbmm_reconstruction_tolerance);

                // Compute every signed source moment from raw atom removals and
                // products, independently of receipt moment_change/gross arrays.
                for (const auto& event : trial.local_qbmm_sources().at(c).at(1)->events)
                {
                    for (const auto& donor : event.donors)
                        for (unsigned k = 0; k < 4; ++k)
                            event_delta[k] -= donor.atom.count * std::pow(donor.atom.mass_over_scale, k);
                    for (const auto& product : event.products)
                        for (unsigned k = 0; k < 4; ++k)
                            event_delta[k] += product.count * std::pow(product.mass_over_scale, k);
                    if (event.kind == R::QbmmEventKind::Birth)
                        activity[0] += 1;
                    if (event.kind == R::QbmmEventKind::Growth)
                        activity[1] += 1;
                    if (event.kind == R::QbmmEventKind::Coalescence)
                        activity[2] += 1;
                    if (event.kind == R::QbmmEventKind::Breakup)
                        activity[3] += 1;
                }
            }
            for (size_t face = 0; face < trial.owned_phase_transport().size(); ++face)
                if (f.mesh->is_geometry_exterior_face(static_cast<LO>(face)))
                {
                    const auto& flux = trial.owned_phase_transport()[face];
                    for (double q : flux.phase_volume_flux)
                        EXPECT_EQ(q, 0);
                    const auto& qbmm = flux.qbmm_flux.at(1).value();
                    EXPECT_EQ(qbmm.population_number_flux, 0);
                    EXPECT_EQ(qbmm.shape_flux.H2, 0);
                    EXPECT_EQ(qbmm.shape_flux.H3, 0);
                    EXPECT_EQ(qbmm.internal_energy_flux, 0);
                    for (double n : qbmm.species_molar_flux)
                        EXPECT_EQ(n, 0);
                }
        });
    const double global_work = f.sum(work);
    for (const double area : wall_areas)
        expect_relative(f.sum(area), 1, 1, 1e-12);
    expect_relative(f.sum(heat_from_cells), heat, std::max(std::abs(heat), 1.), 1e-12);
    if (global_cells > 1)
        for (size_t wall = 0; wall < wall_heat.size(); ++wall)
            expect_relative(
                f.sum(heat_from_walls[wall]), wall_heat[wall], std::max(std::abs(wall_heat[wall]), 1.), 1e-12);
    fourier_endpoints = f.sum(fourier_endpoints);
    double fourier_activity = 0;
    for (size_t face = 0; face < global_faces; ++face)
    {
        EXPECT_EQ(fourier_endpoints[2 * face], -fourier_endpoints[2 * face + 1]) << "geometry face " << face;
        fourier_activity += std::abs(fourier_endpoints[2 * face]);
    }
    expect_relative(diag.pressure_work, global_work, std::max(std::abs(global_work), 1.), 2e-8);
    // Sensible-energy tolerance is based on thermal storage, independent of heat
    // cancellation and independent of the Krylov stopping tolerance.
    expect_relative(after.energy - before.energy, heat + global_work, before.energy, 4e-10);
    for (unsigned k = 0; k < 4; ++k)
        expect_relative(after.population[k] - before.population[k], f.sum(event_delta[k]), before.population[k]);
    for (size_t d = 0; d < 3; ++d)
    {
        const double impulse = d == 2 ? dt * f.config.gravity * before.mass : 0;
        // Closed internal transfers conserve global mass, so accepted, transported
        // and candidate global masses give the same constant-gravity impulse.
        expect_relative(diag.external_impulse.component(d), impulse, std::max(std::abs(impulse), before.mass), 2e-10);
        const double change = after.momentum[d] - before.momentum[d];
        const double accounted = impulse + diag.pressure_impulse.component(d);
        const double scale = std::max({std::abs(before.momentum[d]), std::abs(after.momentum[d]), std::abs(impulse),
            std::abs(diag.pressure_impulse.component(d)), before.mass * slip_speed});
        expect_relative(change, accounted, scale, 4e-10);
    }
    if (f.config.law != PopulationLaw::None)
        for (double count : activity)
            EXPECT_GT(f.sum(count), 0);
    EXPECT_TRUE(report.balances.has_value());
    if (report.balances)
    {
        const auto& balances = *report.balances;
        EXPECT_EQ(balances.material_basis, Solver::QbmmMaterialBalanceBasis::DeclaredReservoirs);
        expect_relative(balances.mass_defect_kg, 0, before.mass);
        for (const auto& [element, defect] : balances.element_defects_mol)
            expect_relative(defect, 0, before.species[element == "S" ? 0 : 1]);
        expect_relative(balances.internal_energy_defect_j, 0, before.energy, 4e-10);
    }
    const int calls = f.population_calls;
    controller.commit(trial);
    EXPECT_EQ(f.population_calls, calls);
    EXPECT_EQ(f.solver->state().time, accepted.state.time + dt);
    return fourier_activity;
}

struct Observables
{
    double circulation = 0, thermal_dipole = 0, vertical_pressure_dipole = 0, speed_squared = 0, area = 0, d32 = 0;
};

Observables observe(const Fixture& f, const Controller& controller)
{
    const auto report = controller.qbmm_report();
    Observables result;
    double diameter3 = 0, diameter2 = 0;
    for (size_t c = 0; c < report.cells.size(); ++c)
    {
        const auto center = f.mesh->cell_centroid(static_cast<LO>(c));
        const double volume = f.mesh->cell_volume(static_cast<LO>(c));
        const auto& cell = f.solver->state().cells[c];
        result.circulation +=
            volume * ((center.z - .5) * cell.velocity.component(0) - (center.x - .5) * cell.velocity.component(2));
        result.thermal_dipole += volume * (.5 - center.x) * (cell.phases[0].temperature - t0);
        result.vertical_pressure_dipole += volume * (center.z - .5) * cell.mechanical_pressure;
        for (size_t d = 0; d < 3; ++d)
            result.speed_squared += volume * std::pow(cell.velocity.component(d), 2);
        const auto& geometry = *report.cells[c].families[1]->geometry;
        result.area += volume * geometry.area_density;
        diameter3 += volume * geometry.diameter_moment_density[3];
        diameter2 += volume * geometry.diameter_moment_density[2];
    }
    result.circulation = f.sum(result.circulation);
    result.thermal_dipole = f.sum(result.thermal_dipole);
    result.vertical_pressure_dipole = f.sum(result.vertical_pressure_dipole);
    result.speed_squared = f.sum(result.speed_squared);
    result.area = f.sum(result.area);
    result.d32 = f.sum(diameter3) / f.sum(diameter2);
    return result;
}

void record_observables(const char* prefix, const Observables& values)
{
    testing::Test::RecordProperty(std::string(prefix) + "_circulation_m2_per_s", values.circulation);
    testing::Test::RecordProperty(std::string(prefix) + "_thermal_dipole_K_m", values.thermal_dipole);
    testing::Test::RecordProperty(
        std::string(prefix) + "_vertical_pressure_dipole_Pa_m", values.vertical_pressure_dipole);
    testing::Test::RecordProperty(std::string(prefix) + "_rms_speed_m_per_s", std::sqrt(values.speed_squared));
    testing::Test::RecordProperty(std::string(prefix) + "_area_m2", values.area);
    testing::Test::RecordProperty(std::string(prefix) + "_d32_m", values.d32);
}

TEST(QbmmApplicationBenchmarks, HeatedClosedCirculationBudgetsCheckpointAndPortableRestartTrajectory)
{
    Fixture f;
    Controller controller(f.solver, f.options());
    EXPECT_EQ(observe(f, controller).speed_squared, 0);
    for (int step = 0; step < 4; ++step)
        checked_step(f, controller, step_size);
    const auto checkpoint = controller.checkpoint();
    const auto archive = f.solver->serialize_qbmm_restart();
    const Endpoint midpoint(f);
    const auto middle_report = controller.qbmm_report();
    std::vector<Endpoint> trajectory;
    for (int step = 0; step < 4; ++step)
    {
        checked_step(f, controller, step_size);
        trajectory.emplace_back(f);
    }
    const auto final = observe(f, controller);
    record_observables("coupled", final);
    EXPECT_EQ(f.solver->state().time, horizon);
    EXPECT_GT(final.thermal_dipole, 0);
    EXPECT_GT(std::abs(final.circulation), 1e-10);
    EXPECT_GT(final.speed_squared, 1e-16);
    // Births are integrated at the accepted left endpoint. Coalescence and
    // breakup have equal event counts; growth preserves H0.
    const double count = 5 + .25 * horizon + .25 * horizon * (horizon - step_size);
    expect_relative(inventory(f, f.solver->state()).population[0], count, count);

    const auto calls = f.population_calls;
    controller.restore(checkpoint);
    EXPECT_EQ(f.population_calls, calls);
    midpoint.expect_same(f);
    EXPECT_EQ(controller.qbmm_report().provenance.origin, Solver::QbmmStateOrigin::Checkpoint);
    for (const auto& endpoint : trajectory)
    {
        checked_step(f, controller, step_size);
        endpoint.expect_same(f);
    }
    Fixture fresh;
    fresh.solver->restore_qbmm_restart(archive);
    midpoint.expect_same(fresh);
    Controller resumed(fresh.solver, fresh.options());
    const auto restart = resumed.qbmm_report();
    EXPECT_EQ(restart.provenance.origin, Solver::QbmmStateOrigin::PortableRestart);
    EXPECT_EQ(restart.provenance.archived_generation, middle_report.provenance.generation);
    EXPECT_EQ(restart.receipt_history, Solver::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_FALSE(restart.interval);
    for (const auto& endpoint : trajectory)
    {
        checked_step(fresh, resumed, step_size);
        endpoint.expect_same(fresh);
    }
}

Observables run_case(CaseOptions options, double dt, const char* label = nullptr)
{
    Fixture f(options);
    Controller controller(f.solver, f.options());
    int maximum_evaluations = 0;
    double maximum_fourier_uncertainty = 0;
    for (int step = 0; step < std::lround(horizon / dt); ++step)
    {
        try
        {
            checked_step(f, controller, dt);
        }
        catch (const std::exception& error)
        {
            std::ostringstream message;
            message << "Q6 case " << (label ? label : "sensitivity") << ": N=" << options.cells_per_side
                    << ", dt=" << dt << ", accepted time=" << f.solver->state().time << ", gravity=" << options.gravity
                    << ", wall differential=" << options.differential_heat << ": " << error.what();
            throw std::runtime_error(message.str());
        }
        maximum_evaluations = std::max(maximum_evaluations, controller.diagnostics().flow.source_evaluations);
        const auto uncertainty = controller.diagnostics().flow.maximum_fourier_gross_comparison_uncertainty_j;
        EXPECT_TRUE(uncertainty.has_value());
        maximum_fourier_uncertainty = std::max(maximum_fourier_uncertainty, uncertainty.value_or(0.));
    }
    if (label)
    {
        testing::Test::RecordProperty(std::string(label) + "_maximum_source_evaluations", maximum_evaluations);
        testing::Test::RecordProperty(
            std::string(label) + "_maximum_fourier_operand_uncertainty_J", maximum_fourier_uncertainty);
    }
    EXPECT_EQ(f.solver->state().time, horizon);
    return observe(f, controller);
}

TEST(QbmmApplicationBenchmarks, FourierOperandResolutionRetainsResolvedHeatAndGrossTurnover)
{
    namespace Source = R::source_stage_detail;
    using Face = Solver::FourierFaceResolution;
    static_assert(!std::is_default_constructible_v<Solver::FourierGrossResolution>);
    static_assert(!std::is_constructible_v<Solver::FourierGrossResolution, size_t, std::vector<Face>>);
    constexpr double tolerance = 1e-8;
    // At 300 K, each rounding cell has a half-ULP radius. With dt*g=1 J/K
    // the two temperature radii sum to one ULP in the canonical heat packet.
    const double quantum = std::nextafter(300., 301.) - 300.;
    const auto comparison = [&](double now, double old, double uncertainty)
    {
        const std::array current{Face{1, now, uncertainty, uncertainty}};
        const std::array previous{Face{1, old, uncertainty, uncertainty}};
        return Source::fourier_face_error(current, previous, tolerance).residual;
    };
    const double heat = 6580000 * quantum;
    EXPECT_EQ(comparison(heat + 2 * quantum, heat, quantum), 0);
    EXPECT_GT(comparison(heat + 3 * quantum, heat, quantum), 1);
    EXPECT_EQ(comparison(0, 2 * quantum, quantum), 0);
    EXPECT_GT(comparison(0, 3 * quantum, quantum), 1);
    EXPECT_EQ(comparison(quantum, -quantum, quantum), 0);
    EXPECT_GT(comparison(2 * quantum, -2 * quantum, quantum), 1);
    EXPECT_GT(comparison(3 * quantum * 1e-200, 0, quantum * 1e-200), 1);
    EXPECT_GT(Source::event_error({{1e-250, 1e-250}}, {{2e-250, 2e-250}}, tolerance), 1);
}

TEST(QbmmApplicationBenchmarks, FourierResolutionMatchesContactsAndUsesAsymmetricMidpoints)
{
    namespace Source = R::source_stage_detail;
    using Face = Solver::FourierFaceResolution;
    constexpr double tolerance = 1e-8;
    const auto contact = [](int id, double owner, double neighbor, double integrated_conductance)
    {
        const auto [decrease, increase] = Source::fourier_temperature_bounds(owner, neighbor, integrated_conductance);
        return Face{id, integrated_conductance * (owner - neighbor), decrease, increase};
    };
    const double u = std::nextafter(300., 301.) - 300.;
    const auto isothermal = contact(3, 300., 300., 1024.);
    const std::array current{isothermal, contact(5, 300. + 100 * u, 300., 1.)};
    const std::array previously_zero{isothermal, contact(5, 300., 300., 1.)};
    const std::array previously_reversed{isothermal, contact(5, 300., 300. + 100 * u, 1.)};
    EXPECT_DOUBLE_EQ(isothermal.increase, 1024 * u);
    const auto on = Source::fourier_face_error(current, previously_zero, tolerance);
    const auto reversal = Source::fourier_face_error(current, previously_reversed, tolerance);
    EXPECT_DOUBLE_EQ(on.residual, .98 / tolerance);
    EXPECT_DOUBLE_EQ(reversal.residual, 1.98 / tolerance);
    EXPECT_DOUBLE_EQ(on.fourier_uncertainty, 2 * u);
    EXPECT_DOUBLE_EQ(reversal.fourier_uncertainty, 2 * u);
    const std::array other_face{isothermal, contact(6, 300., 300., 1.)};
    EXPECT_THROW(Source::fourier_face_error(current, other_face, tolerance), std::invalid_argument);

    // At 256 K the predecessor is s/2 away and the successor is s away.
    // The two packet centers are +s and -s/2. Each has only s/2 of
    // uncertainty toward the other, leaving a distinguishable gap of s/2.
    const double s = std::nextafter(256., 257.) - 256.;
    const std::array now{contact(1, 256., 256. - s, 1.)};
    const std::array old{contact(1, 256. - s / 2, 256., 1.)};
    EXPECT_DOUBLE_EQ(now[0].decrease, s / 2);
    EXPECT_DOUBLE_EQ(now[0].increase, 3 * s / 4);
    EXPECT_DOUBLE_EQ(old[0].decrease, 3 * s / 4);
    EXPECT_DOUBLE_EQ(old[0].increase, s / 2);
    const auto boundary = Source::fourier_face_error(now, old, tolerance);
    EXPECT_DOUBLE_EQ(boundary.fourier_uncertainty, s);
    EXPECT_DOUBLE_EQ(boundary.residual, .5 / tolerance);
}

TEST(QbmmApplicationBenchmarks, FourierSubnormalBoundsUseStrictComparison)
{
    namespace Source = R::source_stage_detail;
    using Face = Solver::FourierFaceResolution;
    constexpr double tolerance = 1e-8;
    constexpr double quantum = std::numeric_limits<double>::denorm_min();
    const double u = std::nextafter(300., 301.) - 300.;
    // One temperature ULP transfers 1.25 heat quanta. Rounding both bounds
    // outward to two quanta would hide the change from one to five quanta.
    constexpr double conductance = 0x1.4p-1030;
    const auto packet = [&](double owner)
    {
        const auto [decrease, increase] = Source::fourier_temperature_bounds(owner, 300., conductance);
        const double energy = static_cast<double>(static_cast<long double>(conductance) * (owner - 300.));
        EXPECT_EQ(decrease, 0.);
        EXPECT_EQ(increase, 0.);
        return std::array{Face{1, energy, decrease, increase}};
    };
    const auto zero = packet(300.);
    const auto small = packet(300. + u);
    const auto large = packet(300. + 4 * u);
    const auto negative = packet(300. - u);
    ASSERT_EQ(zero[0].energy, 0.);
    ASSERT_EQ(small[0].energy, quantum);
    ASSERT_EQ(large[0].energy, 5 * quantum);
    ASSERT_EQ(negative[0].energy, -quantum);
    EXPECT_DOUBLE_EQ(Source::fourier_face_error(small, zero, tolerance).residual, 1. / tolerance);
    const auto change = Source::fourier_face_error(large, small, tolerance);
    EXPECT_DOUBLE_EQ(change.residual, .8 / tolerance);
    EXPECT_EQ(change.fourier_uncertainty, 0.);
    EXPECT_DOUBLE_EQ(Source::fourier_face_error(small, negative, tolerance).residual, 2. / tolerance);

    // At the binade boundary the increase uses 3*u/4. This conductance
    // gives min-normal minus a quarter heat quantum, which rounds to
    // min-normal in binary64 before the cutoff can inspect it.
    const auto boundary = Source::fourier_temperature_bounds(256., std::nextafter(256., 0.), 0x1.5555555555555p-978);
    EXPECT_EQ(boundary[0], 0.);
    EXPECT_EQ(boundary[1], 0.);
    const auto normal = Source::fourier_temperature_bounds(300., 300., 0x1p-977);
    EXPECT_EQ(normal[0], 2 * std::numeric_limits<double>::min());
    EXPECT_EQ(normal[1], 2 * std::numeric_limits<double>::min());
}

TEST(QbmmApplicationBenchmarks, FourierResolutionRowsRequireReactingControllerAuthority)
{
    CaseOptions options;
    options.cells_per_side = 1;
    options.gravity = options.slip = 0;
    Fixture f(options);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const bool declares_rows = f.mesh->num_owned_cells() != 0;
    EXPECT_EQ(f.sum(declares_rows ? 1 : 0), 1);

    // Reuse the thermodynamic state with only the legacy population count bound.
    auto controls = f.solver->options();
    controls.qbmm_families.clear();
    controls.qbmm_model = {};
    auto initial = f.solver->state();
    initial.qbmm_families.clear();
    for (auto& cell : initial.cells)
        cell.qbmm_shape.clear();
    auto count_only = std::make_shared<Solver>(f.mesh, std::vector{f.liquid, f.gas, f.gas}, controls);
    count_only->initialize(std::move(initial));
    {
        const auto lease = count_only->acquire_controller();
        const auto retained = count_only->preview(step_size, {}, &lease);
        const auto revision = count_only->state().revision;
        const auto time = count_only->state().time;
        const auto provider = std::make_shared<const Solver::SourceStageProvider>(
            Solver::SourceStageProvider{"Q6 untrusted Fourier resolution rows", 1, f.catalog->fingerprint(),
                [declares_rows](const Solver::SourceStageContext& context)
                {
                    Solver::SourceStageResult result;
                    const auto owned = context.transport.cells.size();
                    result.gross_events.assign(owned, {1., 0.});
                    if (declares_rows)
                        result.fourier_gross_resolution.resize(owned); // Valid layout, with no constructed receipts.
                    return result;
                }});
        bool rejected = false;
        try
        {
            (void) count_only->preview_with_sources(step_size, provider, &lease);
        }
        catch (const std::exception& error)
        {
            rejected = true;
            const std::string_view message = declares_rows ? "private reacting-controller lease"
                                                           : "Source-stage returned packets failed on another rank";
            EXPECT_NE(std::string_view(error.what()).find(message), std::string_view::npos);
        }
        EXPECT_EQ(f.sum(rejected ? 1 : 0), comm->getSize());
        EXPECT_EQ(count_only->state().revision, revision);
        EXPECT_EQ(count_only->state().time, time);
        count_only->commit(retained, &lease);
        EXPECT_EQ(count_only->state().time, time + step_size);
    }
    auto reacting = f.options();
    reacting.qbmm_population.reset();
    Controller controller(count_only, std::move(reacting));
    const auto trusted = controller.preview(step_size);
    EXPECT_EQ(trusted.diagnostics().flow.maximum_fourier_gross_comparison_uncertainty_j, 0.);
    controller.commit(trusted);
    EXPECT_EQ(count_only->state().time, 2 * step_size);
}

TEST(QbmmApplicationBenchmarks, DifferentialHeatingAndGravityHaveIndependentCirculationControls)
{
    CaseOptions options;
    const auto driven = run_case(options, step_size, "driven");
    options.gravity = 0;
    const auto no_gravity = run_case(options, step_size, "no_gravity");
    options.gravity = -9.81;
    options.differential_heat = 0;
    const auto no_differential_heat = run_case(options, step_size, "uniform_heating");
    options.differential_heat = -differential_wall_power;
    const auto reversed = run_case(options, step_size, "reversed_heating");
    options.gravity = options.differential_heat = 0;
    const auto neither = run_case(options, step_size, "no_gravity_uniform_heating");
    record_observables("driven", driven);
    record_observables("no_gravity", no_gravity);
    record_observables("uniform_heating", no_differential_heat);
    record_observables("reversed_heating", reversed);
    record_observables("no_gravity_uniform_heating", neither);
    EXPECT_GT(driven.thermal_dipole, 0);
    EXPECT_LT(reversed.thermal_dipole, 0);
    EXPECT_LT(driven.circulation * reversed.circulation, 0);
    EXPECT_GT(std::abs(driven.circulation - no_gravity.circulation), 1e-10);
    EXPECT_GT(std::abs(driven.circulation), 10 * std::abs(no_differential_heat.circulation) + 1e-10);
    EXPECT_LT(driven.vertical_pressure_dipole, no_gravity.vertical_pressure_dipole);
    EXPECT_LT(no_differential_heat.vertical_pressure_dipole, neither.vertical_pressure_dipole);
    // Both equal-power controls retain mean heating and supplied vertical slip.
    // The extra circulation from differential heating is larger with gravity,
    // when the horizontal density gradient makes gravity drive a lateral roll.
    EXPECT_GT(driven.circulation - no_differential_heat.circulation,
        std::abs(no_gravity.circulation - neither.circulation) + 1e-10);
    EXPECT_GT(driven.thermal_dipole, 10 * std::abs(no_differential_heat.thermal_dipole) + 1e-10);
    EXPECT_GT(no_gravity.thermal_dipole, 10 * std::abs(neither.thermal_dipole) + 1e-10);
}

enum class Refinement
{
    Mesh,
    TimeStep
};

// Report the two successive differences in this short transient, including
// their signs and magnitudes relative to the finest observed value.
void record_sensitivity(
    Refinement refinement, const Observables& coarse, const Observables& medium, const Observables& fine)
{
    const std::string name = refinement == Refinement::Mesh ? "mesh" : "dt";
    record_observables((name + "_coarse").c_str(), coarse);
    record_observables((name + "_medium").c_str(), medium);
    record_observables((name + "_fine").c_str(), fine);
    const std::array a{coarse.circulation, coarse.thermal_dipole, coarse.area, coarse.d32};
    const std::array b{medium.circulation, medium.thermal_dipole, medium.area, medium.d32};
    const std::array c{fine.circulation, fine.thermal_dipole, fine.area, fine.d32};
    constexpr std::array labels{"circulation", "thermal_dipole", "area", "d32"};
    for (size_t i = 0; i < a.size(); ++i)
    {
        SCOPED_TRACE(labels[i]);
        const double coarse_change = b[i] - a[i], fine_change = c[i] - b[i];
        const double scale = std::max({std::abs(a[i]), std::abs(b[i]), std::abs(c[i])});
        ASSERT_TRUE(std::isfinite(coarse_change) && std::isfinite(fine_change) && std::isfinite(scale));
        ASSERT_GT(std::abs(c[i]), 0);
        const double roundoff = 1024 * std::numeric_limits<double>::epsilon() * scale;
        const std::string key = name + "_" + labels[i];
        testing::Test::RecordProperty(key + "_coarse_difference", std::abs(coarse_change));
        testing::Test::RecordProperty(key + "_fine_difference", std::abs(fine_change));
        testing::Test::RecordProperty(key + "_coarse_relative_change", coarse_change / std::abs(c[i]));
        testing::Test::RecordProperty(key + "_fine_relative_change", fine_change / std::abs(c[i]));
        if (i < 2)
        {
            EXPECT_GT(a[i], 0);
            EXPECT_GT(b[i], 0);
            EXPECT_GT(c[i], 0);
            EXPECT_GT(std::abs(coarse_change), roundoff);
            EXPECT_GT(std::abs(fine_change), roundoff);
        }
        // Grid refinement changes the wall-adjacent heating volumes and the
        // redistribution of the population. Its global area and d32 need not
        // move monotonically as the flow and the size distribution both change.
        if (refinement == Refinement::TimeStep || i < 2)
            EXPECT_LE(std::abs(fine_change), std::abs(coarse_change) + roundoff);
    }
}

TEST(QbmmApplicationBenchmarks, MeasuredMeshSensitivityAtMatchedTimestepAndHorizon)
{
    CaseOptions options;
    constexpr double dt = step_size / 2;
    options.cells_per_side = 2;
    const auto coarse = run_case(options, dt, "mesh_coarse");
    options.cells_per_side = 4;
    const auto medium = run_case(options, dt, "mesh_medium");
    options.cells_per_side = 8;
    const auto fine = run_case(options, dt, "mesh_fine");
    record_sensitivity(Refinement::Mesh, coarse, medium, fine);
    testing::Test::RecordProperty("mesh_coarse_cells_per_side", 2);
    testing::Test::RecordProperty("mesh_medium_cells_per_side", 4);
    testing::Test::RecordProperty("mesh_fine_cells_per_side", 8);
    testing::Test::RecordProperty("matched_dt_s", dt);
    testing::Test::RecordProperty("matched_horizon_s", horizon);
}

TEST(QbmmApplicationBenchmarks, MeasuredTimestepSensitivityAtMatchedMeshAndHorizon)
{
    CaseOptions options;
    const auto coarse = run_case(options, step_size);
    const auto medium = run_case(options, step_size / 2);
    const auto fine = run_case(options, step_size / 4);
    record_sensitivity(Refinement::TimeStep, coarse, medium, fine);
    testing::Test::RecordProperty("matched_cells_per_side", options.cells_per_side);
    testing::Test::RecordProperty("matched_horizon_s", horizon);
}

TEST(QbmmApplicationBenchmarks, SingleCellEmptyOwnersRetainImpulseHeatInventoriesAndRestart)
{
    CaseOptions options;
    options.cells_per_side = 1;
    Fixture f(options);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    EXPECT_EQ(f.sum(f.mesh->num_owned_cells() == 0 ? 1 : 0), comm->getSize() - 1);
    Controller controller(f.solver, f.options());
    checked_step(f, controller, step_size);
    const auto checkpoint = controller.checkpoint();
    const auto bytes = f.solver->serialize_qbmm_restart();
    const Endpoint middle(f);
    checked_step(f, controller, step_size);
    const Endpoint end(f);
    controller.restore(checkpoint);
    middle.expect_same(f);
    checked_step(f, controller, step_size);
    end.expect_same(f);
    Fixture fresh(options);
    fresh.solver->restore_qbmm_restart(bytes);
    middle.expect_same(fresh);
    Controller resumed(fresh.solver, fresh.options());
    checked_step(fresh, resumed, step_size);
    end.expect_same(fresh);
}

TEST(QbmmApplicationBenchmarks, SingleCellRetainsEachSignedWallHeatPacket)
{
    // The one-cell ledger combines both walls in one entry. Isolate each wall to observe its applied sign.
    constexpr std::array signed_heat{136., -120.};
    for (size_t wall = 0; wall < signed_heat.size(); ++wall)
    {
        SCOPED_TRACE(wall);
        CaseOptions options;
        options.cells_per_side = 1;
        const double wall_power =
            wall == 0 ? mean_wall_power + differential_wall_power : mean_wall_power - differential_wall_power;
        options.mean_heat = wall_power / 2;
        options.differential_heat = wall == 0 ? options.mean_heat : -options.mean_heat;
        Fixture f(options);
        Controller controller(f.solver, f.options());
        EXPECT_EQ(checked_step(f, controller, step_size), 0);
        EXPECT_NEAR(controller.diagnostics().external_heat, signed_heat[wall], 1e-10);
        EXPECT_NEAR(controller.diagnostics().net_thermal_energy, signed_heat[wall], 1e-8);
    }
}

TEST(QbmmApplicationBenchmarks, ResolvedCavityRetainsSharedFacesWithEmptyOwners)
{
    CaseOptions options;
    options.cells_per_side = 2;
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    Fixture f(options, std::array<size_t, 3>{ranks, 1, 1});
    const auto empty = f.sum(f.mesh->num_owned_cells() == 0 ? 1 : 0);
    EXPECT_EQ(empty, ranks - std::min(ranks, size_t{2}));
    EXPECT_EQ(f.sum(static_cast<double>(f.mesh->num_owned_cells())), 4);
    double interior = 0, shared = 0;
    for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
    {
        const auto face = static_cast<LO>(i);
        if (f.mesh->is_geometry_exterior_face(face))
            continue;
        ++interior;
        const auto owner = f.mesh->owner_cell(face), neighbor = f.mesh->neighbor_cell(face);
        EXPECT_NE(owner, Solver::Mesh::invalid_local_id());
        EXPECT_NE(neighbor, Solver::Mesh::invalid_local_id());
        if (f.mesh->is_owned_cell(owner) != f.mesh->is_owned_cell(neighbor))
            ++shared;
    }
    const double global_shared = f.sum(shared);
    EXPECT_EQ(f.sum(interior), 4);
    EXPECT_EQ(global_shared, ranks > 1 ? 2 : 0);
    testing::Test::RecordProperty("mpi_ranks", static_cast<int>(ranks));
    testing::Test::RecordProperty("empty_owners", empty);
    testing::Test::RecordProperty("shared_faces", global_shared);
    Controller controller(f.solver, f.options());
    checked_step(f, controller, step_size);
    EXPECT_GT(checked_step(f, controller, step_size), 0);
}

/** Independent mass-conserving fixed-pivot population reference.
 * Logarithmic pivots include 1,2,4 and exact half-mass daughters. Products are
 * deposited on adjacent pivots with weights preserving count AND first mass.
 * No production inversion, event allocator or native flow operation is used.
 * The source-only native comparison uses exactly these rates, initial H0..H3,
 * event duration and bulk volume; no circulation discretization enters here.
 */
struct FineBins
{
    std::vector<double> x, n;
    double largest_budget_defect = 0;
    explicit FineBins(int pivots_per_octave)
    {
        for (int i = 0; i <= 7 * pivots_per_octave; ++i)
            x.push_back(std::exp2(-3. + double(i) / pivots_per_octave));
        n.resize(x.size());
        n[3 * pivots_per_octave] = 2;
        n[4 * pivots_per_octave] = 1;
        n[5 * pivots_per_octave] = 1;
    }
    void deposit(std::vector<double>& delta, double count, double size) const
    {
        if (size < x.front() || size > x.back())
            throw std::runtime_error("Q6 fine-bin product outside the declared support.");
        const auto upper = std::lower_bound(x.begin(), x.end(), size);
        const size_t hi = static_cast<size_t>(upper - x.begin());
        if (*upper == size || hi == 0)
        {
            delta[hi] += count;
            return;
        }
        const double high_weight = (size - x[hi - 1]) / (x[hi] - x[hi - 1]);
        delta[hi - 1] += count * (1 - high_weight);
        delta[hi] += count * high_weight;
    }
    R::QbmmMomentBlock stocks() const
    {
        R::QbmmMomentBlock total{};
        for (size_t i = 0; i < x.size(); ++i)
            for (unsigned k = 0; k < 4; ++k)
                total[k] += n[i] * std::pow(x[i], k);
        return total;
    }
    void advance(double dt, bool only_topology = false, bool constant_kernel = false)
    {
        const auto before = stocks();
        std::vector<double> delta(n.size());
        double count_change = 0, mass_change = 0;
        if (!only_topology)
        {
            deposit(delta, birth_rate * dt, 1);
            count_change += birth_rate * dt;
            mass_change += birth_rate * dt;
        }
        for (size_t i = 0; i < n.size(); ++i)
        {
            if (n[i] == 0)
                continue;
            // Intentionally repeat formulas instead of calling native-law
            // helpers, so wrong pair multiplicity/powers do not share an oracle.
            if (!only_topology && 1.125 * x[i] <= 16)
            {
                const double count = dt * .04 * std::pow(x[i], 2. / 3) * n[i];
                delta[i] -= count;
                deposit(delta, count, 1.125 * x[i]);
                mass_change += count * .125 * x[i];
            }
            if (!constant_kernel && x[i] >= .25)
            {
                const double count = dt * .02 * std::pow(x[i], .75) * n[i];
                delta[i] -= count;
                deposit(delta, 2 * count, x[i] / 2);
                count_change += count;
            }
            for (size_t j = i; j < n.size(); ++j)
                if (n[j] > 0 && x[i] + x[j] <= x.back())
                {
                    const double kernel = constant_kernel ? 1
                                                          : .01 * std::pow(std::cbrt(x[i]) + std::cbrt(x[j]), 2) /
                                                                std::sqrt(1 + x[i] + x[j]);
                    const double count = dt * kernel * n[i] * n[j] / (i == j ? 2 : 1);
                    delta[i] -= count;
                    delta[j] -= count;
                    deposit(delta, count, x[i] + x[j]);
                    count_change -= count;
                }
        }
        for (size_t i = 0; i < n.size(); ++i)
        {
            // Positivity must hold before publication; no clipping/renormalizing.
            if (!(n[i] + delta[i] >= 0))
                throw std::runtime_error("Q6 fine-bin explicit donor overdraw.");
            n[i] += delta[i];
        }
        const auto after = stocks();
        largest_budget_defect =
            std::max({largest_budget_defect, std::abs(after[0] - before[0] - count_change) / before[0],
                std::abs(after[1] - before[1] - mass_change) / before[1]});
    }
    double area_mass_integral() const
    {
        double area = 0;
        for (size_t i = 0; i < x.size(); ++i)
            area += n[i] * std::pow(x[i], 2. / 3);
        return area;
    }
};

struct ClosureResult
{
    R::QbmmMomentBlock m;
    double area_integral;
};

struct ClosureTrajectory
{
    ClosureResult initial, final;
};

ClosureResult run_bins(int resolution, double dt, double end_time = .5)
{
    FineBins reference(resolution);
    const int steps = std::lround(end_time / dt);
    for (int step = 0; step < steps; ++step)
        reference.advance(dt);
    EXPECT_LT(reference.largest_budget_defect, 2e-14);
    testing::Test::RecordProperty("fine_bin_maximum_relative_inventory_defect", reference.largest_budget_defect);
    return {reference.stocks(), reference.area_mass_integral()};
}

ClosureResult observe_closure(const Fixture& f, const Controller& controller)
{
    const auto stocks = inventory(f, f.solver->state());
    const auto report = controller.qbmm_report();
    double area = 0;
    for (const auto& cell : report.cells)
    {
        const auto& q = cell.families[1]->quadrature;
        for (unsigned i = 0; i < q.active_atoms; ++i)
            area += q.atoms[i].count * std::pow(q.atoms[i].mass_over_scale, 2. / 3);
    }
    return {stocks.population, f.sum(area)};
}

ClosureTrajectory run_two_node(double dt)
{
    CaseOptions options;
    options.cells_per_side = 1;
    options.gravity = options.slip = options.mean_heat = options.differential_heat = 0;
    options.law = PopulationLaw::Nonlinear;
    options.broad_initial_distribution = true;
    Fixture f(options);
    Controller controller(f.solver, f.options());
    const auto initial = observe_closure(f, controller);
    const std::array<double, 4> initial_moments{4, 8, 22, 74};
    EXPECT_EQ(initial.m, initial_moments);
    for (int step = 0; step < std::lround(.5 / dt); ++step)
        checked_step(f, controller, dt);
    return {initial, observe_closure(f, controller)};
}

TEST(QbmmApplicationBenchmarks, FineBinReferencePreservesMassAndCountBeforeClosureComparison)
{
    FineBins bins(16);
    EXPECT_EQ(bins.stocks(), (R::QbmmMomentBlock{4, 8, 22, 74}));
    const double mass = bins.stocks()[1];
    for (int step = 0; step < 32; ++step)
        bins.advance(1. / 128, true);
    expect_relative(bins.stocks()[1], mass, mass, 2e-14);
    EXPECT_LT(bins.largest_budget_defect, 2e-14);
    EXPECT_GT(bins.stocks()[0], 0);
    EXPECT_GT(bins.area_mass_integral(), 0);
}

TEST(QbmmApplicationBenchmarks, ConstantKernelFineBinsConvergeToIndependentDiscreteSolution)
{
    // K=1, unit monomer stock, t=1: n_j=(1/2)^(j-1)/(3/2)^(j+1).
    // Integer pivots give exact deposition for every represented collision.
    // The discarded tail beyond j=64 is bounded by this geometric solution.
    const R::QbmmMomentBlock exact{2. / 3, 1, 2, 5.5};
    FineBins geometric(16);
    geometric.x.resize(64);
    geometric.n.resize(64);
    double exact_fractional = 0;
    for (size_t i = 0; i < 64; ++i)
    {
        const double j = i + 1;
        geometric.x[i] = j;
        geometric.n[i] = std::pow(.5, j - 1) / std::pow(1.5, j + 1);
        exact_fractional += geometric.n[i] * std::pow(j, 2. / 3);
    }
    const auto represented = geometric.stocks();
    for (unsigned k = 0; k < 4; ++k)
        expect_relative(represented[k], exact[k], exact[k], 1e-14);
    EXPECT_NEAR(exact_fractional, .849396443577582, 2e-15);
    // Independent Gaussian roots of x^2-b*x+d, orthogonal to 1 and x.
    const auto [m0, m1, m2, m3] = exact;
    const double determinant = m0 * m2 - m1 * m1;
    const double b = (m0 * m3 - m1 * m2) / determinant;
    const double d = (m1 * m3 - m2 * m2) / determinant;
    const double hi = .5 * (b + std::sqrt(b * b - 4 * d)), lo = d / hi;
    const double high_count = (m1 - m0 * lo) / (hi - lo);
    const double projected = (m0 - high_count) * std::pow(lo, 2. / 3) + high_count * std::pow(hi, 2. / 3);
    EXPECT_NEAR(projected, .851439464059812, 2e-15);
    testing::Test::RecordProperty("constant_kernel_exact_fractional_moment", exact_fractional);
    testing::Test::RecordProperty("constant_kernel_two_node_fractional_moment", projected);
    testing::Test::RecordProperty("constant_kernel_relative_closure_gap", projected / exact_fractional - 1);

    const auto integrate = [&](int steps)
    {
        auto bins = geometric;
        std::fill(bins.n.begin(), bins.n.end(), 0);
        bins.n[0] = 1;
        for (int step = 0; step < steps; ++step)
            bins.advance(1. / steps, true, true);
        EXPECT_LT(bins.largest_budget_defect, 2e-14);
        expect_relative(bins.stocks()[1], 1, 1, 2e-14);
        return bins;
    };
    const auto coarse = integrate(256), fine = integrate(512);
    for (unsigned k : {0u, 2u, 3u})
    {
        const double a = std::abs(coarse.stocks()[k] - exact[k]);
        const double c = std::abs(fine.stocks()[k] - exact[k]);
        // H2 has exactly constant derivative for K=1 and may integrate exactly.
        EXPECT_LE(c, .55 * a + 128 * std::numeric_limits<double>::epsilon() * exact[k]);
        EXPECT_LT(c / exact[k], .005);
    }
    EXPECT_LT(std::abs(fine.area_mass_integral() - exact_fractional),
        .55 * std::abs(coarse.area_mass_integral() - exact_fractional));
}

TEST(QbmmApplicationBenchmarks, MatchedNonlinearPopulationClosureSensitivityVersusRefinedFineBins)
{
    // Homogeneous, source-only problem. Initial H0..H3, source law, donor stocks,
    // bulk volume, interval and dt match. The fine-bin full initial measure has
    // three atoms; the two-node projection preserves its first four moments.
    const auto bins_coarse = run_bins(32, 1. / 256);
    const auto bins_medium = run_bins(64, 1. / 256);
    const auto bins_fine = run_bins(128, 1. / 256);
    const auto bins_half_dt = run_bins(128, 1. / 512);
    const FineBins initial_bins(128);
    const ClosureResult bin_initial{initial_bins.stocks(), initial_bins.area_mass_integral()};
    const auto q_coarse = run_two_node(1. / 128).final;
    const auto q_fine = run_two_node(1. / 256).final;
    const auto q_trajectory = run_two_node(1. / 512);
    const auto& q_half_dt = q_trajectory.final;
    EXPECT_EQ(q_trajectory.initial.m, bin_initial.m);
    // The monic quadratic orthogonal to 1 and x under (4, 8, 22, 74)
    // is x^2 - 5*x + 4.5. This independently fixes the initial area gap.
    const double lo = (5 - std::sqrt(7.)) / 2, hi = (5 + std::sqrt(7.)) / 2;
    const double high_count = (8 - 4 * lo) / (hi - lo);
    const double projected_area = (4 - high_count) * std::pow(lo, 2. / 3) + high_count * std::pow(hi, 2. / 3);
    expect_relative(q_trajectory.initial.area_integral, projected_area, projected_area, 1e-13);
    EXPECT_GT(projected_area, bin_initial.area_integral);
    constexpr std::array labels{"H0", "H1", "H2", "H3", "area_mass_integral"};
    double coarse_reference_norm = 0, fine_reference_norm = 0;
    for (unsigned k = 0; k < 5; ++k)
    {
        const auto value = [k](const ClosureResult& result) { return k == 4 ? result.area_integral : result.m[k]; };
        const double initial_reference = value(bin_initial), initial_q = value(q_trajectory.initial);
        const double initial_gap = initial_q - initial_reference;
        const double reference = value(bins_half_dt);
        const double space_coarse = std::abs(value(bins_coarse) - value(bins_medium));
        const double space_fine = std::abs(value(bins_medium) - value(bins_fine));
        const double time_bins = std::abs(value(bins_fine) - reference);
        const double time_q_coarse = std::abs(value(q_coarse) - value(q_fine));
        const double time_q_fine = std::abs(value(q_fine) - value(q_half_dt));
        const double evolved_gap = value(q_half_dt) - reference;
        const double closure = std::abs(evolved_gap);
        const double arithmetic = 1e-12 * std::abs(reference);
        SCOPED_TRACE(labels[k]);
        const std::string key = std::string("nonlinear_") + labels[k];
        testing::Test::RecordProperty(key + "_initial_fine_bin", initial_reference);
        testing::Test::RecordProperty(key + "_initial_two_node", initial_q);
        testing::Test::RecordProperty(key + "_initial_signed_relative_closure_gap", initial_gap / initial_reference);
        testing::Test::RecordProperty(key + "_fine_bin_reference", reference);
        testing::Test::RecordProperty(key + "_two_node", value(q_half_dt));
        testing::Test::RecordProperty(key + "_bin_space_difference", space_fine);
        testing::Test::RecordProperty(key + "_bin_time_difference", time_bins);
        testing::Test::RecordProperty(key + "_two_node_time_difference", time_q_fine);
        testing::Test::RecordProperty(key + "_relative_closure_gap", closure / std::abs(reference));
        testing::Test::RecordProperty(key + "_evolved_signed_relative_closure_gap", evolved_gap / reference);
        testing::Test::RecordProperty(key + "_signed_relative_gap_change", (evolved_gap - initial_gap) / reference);
        coarse_reference_norm += std::pow(space_coarse / reference, 2);
        fine_reference_norm += std::pow(space_fine / reference, 2);
        EXPECT_LE(time_q_fine, .65 * time_q_coarse + arithmetic);
        EXPECT_LT(space_fine / std::abs(reference), 2e-3);
        EXPECT_LT(time_bins / std::abs(reference), 2e-4);
        EXPECT_LT(closure / std::abs(reference), .15);
        // Area and H2 must retain a distinguishable closure gap after bounding
        // reference/time uncertainty. H3 can have cancelling closure errors;
        // its measured gap is reported without claiming it is resolved here.
        if (k == 2 || k == 4)
            EXPECT_GT(closure, 2 * (space_fine + time_bins + time_q_fine) + arithmetic);
    }
    // The discontinuous support cutoffs need not produce componentwise monotone
    // errors (notably the small difference of breakup and coalescence in H0).
    // Require reference contraction in the dimensionless vector norm, retain
    // individual error caps above, and report each component's measured error.
    testing::Test::RecordProperty("bin_coarse_relative_difference_norm", std::sqrt(coarse_reference_norm));
    testing::Test::RecordProperty("bin_fine_relative_difference_norm", std::sqrt(fine_reference_norm));
    EXPECT_LT(fine_reference_norm, coarse_reference_norm);
}
} // namespace
