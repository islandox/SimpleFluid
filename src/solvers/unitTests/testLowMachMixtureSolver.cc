/** @file testLowMachMixtureSolver.cc
 * @brief Native low-Mach physical budgets, pressure projection and immutable transaction regressions.
 */
#include "FVM/CellOperators.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>

namespace
{
namespace R = SimpleFluid::Reacting;
using namespace SimpleFluid;
using Solver = R::LowMachMixtureSolver;
using LO = Solver::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double gas_constant = 8.31446261815324;

SP<Solver::Mesh> mesh()
{
    Solver::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Solver::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                              Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, 0.5, 1}}),
        distribution);
}
std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    const std::vector phases{
        R::SpeciesPhase::Liquid, R::SpeciesPhase::Dissolved, R::SpeciesPhase::Resolved, R::SpeciesPhase::Dispersed};
    return std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {10, "vapor", 0.018, {{"A", 1}}, phases}, {29, "noncondensable", 0.032, {{"B", 1}}, phases}});
}
std::shared_ptr<const R::MixtureThermodynamics> ideal(std::shared_ptr<const R::SpeciesCatalog> species)
{
    return R::make_ideal_gas_mixture(species, {{20, 300, 6000}, {20, 300, 6000}}, {250, 600, 10000, 2000000});
}
Solver::Options options()
{
    Solver::Options result;
    result.minimum_pressure = 20000;
    result.maximum_pressure = 1000000;
    result.linear.tolerance = 1e-12;
    result.linear.preconditioner = LinearPreconditioner::None;
    return result;
}
Solver::State state(
    const Solver::Mesh& grid, std::shared_ptr<const R::MixtureThermodynamics> gas, bool varying_composition = false)
{
    Solver::State result;
    result.thermodynamic_pressure = 100000;
    result.cells.resize(grid.num_owned_cells());
    for (size_t c = 0; c < result.cells.size(); ++c)
    {
        const double volume = grid.cell_volume(static_cast<LO>(c));
        const double n = 100000 * volume / (gas_constant * 300);
        const double fraction = varying_composition && grid.cell_global_id(static_cast<LO>(c)) % 2 ? 0.8 : 0.2;
        result.cells[c].phases = {{gas, {fraction * n, (1 - fraction) * n}, 100000, 20 * n * 300, 300}};
    }
    return result;
}

TEST(LowMachMixtureSolverTest, UniformClosedHeatingRecoversPressureTemperatureAndActualMixtureDensity)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas, true));
    const auto initial = solver.state();
    std::vector<Solver::CellSources> sources(initial.cells.size());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {0.1 * initial.cells[c].phases[0].internal_energy};
    const auto diagnostics = solver.advance(0.01, sources);
    EXPECT_NEAR(solver.state().thermodynamic_pressure, 110000, 1e-3);
    EXPECT_LE(diagnostics.maximum_occupancy_error, options().volume_relative_tolerance);
    EXPECT_LE(diagnostics.maximum_continuity_residual, options().continuity_tolerance);
    EXPECT_NEAR(diagnostics.mass_error, 0, 1e-12);
    EXPECT_NEAR(diagnostics.internal_energy_error, 0, 1e-6);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        EXPECT_NEAR(solver.state().cells[c].phases[0].temperature, 330, 1e-6);
        EXPECT_EQ(solver.state().cells[c].phases[0].moles, initial.cells[c].phases[0].moles);
        const double mass = initial.cells[c].phases[0].moles[0] * 0.018 + initial.cells[c].phases[0].moles[1] * 0.032;
        EXPECT_NEAR(solver.density().value(static_cast<LO>(c)), mass / grid->cell_volume(static_cast<LO>(c)), 1e-12);
    }
}

TEST(LowMachMixtureSolverTest, ImmutablePreviewRetainsAcceptedStateAndStaleTrialsReject)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    const auto before = solver.state();
    const auto first = solver.preview(0.01);
    EXPECT_EQ(solver.state().revision, before.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, before.time);
    for (size_t c = 0; c < before.cells.size(); ++c)
        EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].internal_energy, before.cells[c].phases[0].internal_energy);
    const auto second = solver.preview(0.01);
    EXPECT_NO_THROW(solver.commit(first));
    EXPECT_EQ(solver.state().revision, before.revision + 1);
    EXPECT_THROW(solver.commit(second), std::exception);
}

TEST(LowMachMixtureSolverTest, ControllerLeaseExcludesStandaloneAndForeignAdvancement)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    {
        auto lease = solver.acquire_controller();
        EXPECT_THROW((void) solver.advance(0.01), std::exception);
        EXPECT_THROW((void) solver.acquire_controller(), std::exception);
        const auto trial = solver.preview(0.01, {}, &lease);
        EXPECT_THROW(solver.commit(trial), std::exception);
        solver.commit(trial, &lease);
    }
    EXPECT_NO_THROW((void) solver.advance(0.01));
}

TEST(LowMachMixtureSolverTest, CheckpointReplayAndRejectedSourceLeaveAllAcceptedInventoriesUnchanged)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    const auto saved = solver.checkpoint();
    const auto initial = solver.state();
    std::vector<Solver::CellSources> sources(initial.cells.size());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {100};
    solver.advance(0.01, sources);
    const auto expected = solver.state();
    solver.restore(saved);
    solver.advance(0.01, sources);
    EXPECT_NEAR(solver.state().thermodynamic_pressure, expected.thermodynamic_pressure, 1e-8);
    EXPECT_EQ(solver.state().revision, expected.revision);
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].species_moles = {{-1e100, 0}};
    EXPECT_THROW((void) solver.advance(0.01, sources), std::exception);
    EXPECT_EQ(solver.state().revision, expected.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, expected.time);
    for (size_t c = 0; c < initial.cells.size(); ++c)
        EXPECT_EQ(solver.state().cells[c].phases[0].moles, expected.cells[c].phases[0].moles);
}

TEST(LowMachMixtureSolverTest, SeparatePhaseTemperaturesRetainIndependentConservativeBinEnergies)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    auto controls = options();
    controls.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
    Solver solver(grid, {gas, gas}, controls);
    auto initial = state(*grid, gas);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        const double volume = grid->cell_volume(static_cast<LO>(c));
        const double first = 100000 * 0.4 * volume / (gas_constant * 350);
        const double second = 100000 * 0.6 * volume / (gas_constant * 300);
        initial.cells[c].phases = {
            {gas, {first, 0}, 100000, first * 20 * 350, 350}, {gas, {0, second}, 100000, second * 20 * 300, 300}};
    }
    solver.initialize(initial);
    solver.advance(0.01);
    for (const auto& cell : solver.state().cells)
    {
        EXPECT_NEAR(cell.phases[0].temperature, 350, 1e-6);
        EXPECT_NEAR(cell.phases[1].temperature, 300, 1e-6);
    }
}

TEST(LowMachMixtureSolverTest, NonuniformHeatingDrivesNativeConservativeFlowWithVariableComposition)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas, true));
    std::vector<Solver::CellSources> sources(grid->num_owned_cells());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {grid->cell_global_id(static_cast<LO>(c)) == 0 ? 0.01 : 0};
    const auto result = solver.advance(0.01, sources);
    EXPECT_LE(result.maximum_occupancy_error, options().volume_relative_tolerance);
    EXPECT_LE(result.maximum_continuity_residual, options().continuity_tolerance);
    for (double error : result.species_errors)
        EXPECT_NEAR(error, 0, 1e-10);
    EXPECT_NEAR(result.mass_error, 0, 1e-12);
    EXPECT_GT(result.correctors, 1);
}
TEST(LowMachMixtureSolverTest, OrdinaryThermodynamicStateRetainsPhysicalGatesAtSmallTimeSteps)
{
    // The heat rate and ordinary 300 K/1 bar state stay fixed as dt decreases.
    // Requiring a fixed RHS-relative Krylov tolerance, or sub-ulp global volume
    // cancellation, must not prevent a physically resolved step.
    for (const double dt : {1e-4, 1e-6, 1e-8})
    {
        SCOPED_TRACE(dt);
        Solver::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        auto grid = std::make_shared<Solver::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{ArrReal{0., .01}, ArrReal{0., .01}, ArrReal{0., .005, .01}}), distribution);
        const auto gas = ideal(catalog());
        const auto controls = options();
        Solver solver(grid, {gas}, controls);
        solver.initialize(state(*grid, gas, true));
        const auto initial = solver.state();
        std::vector<Solver::CellSources> sources(grid->num_owned_cells());
        for (size_t c = 0; c < sources.size(); ++c)
            sources[c].internal_energy = {grid->cell_centroid(static_cast<LO>(c)).z < .005 ? dt : 0.};
        Solver::Diagnostics result;
        ASSERT_NO_THROW(result = solver.advance(dt, sources));
        EXPECT_EQ(solver.state().revision, initial.revision + 1);
        EXPECT_DOUBLE_EQ(solver.state().time, dt);
        EXPECT_LE(result.maximum_occupancy_error, controls.volume_relative_tolerance);
        EXPECT_LE(result.maximum_continuity_residual, controls.continuity_tolerance);
        EXPECT_LE(std::abs(result.internal_energy_error),
            controls.conservation_relative_tolerance * result.caloric_energy_scale +
                result.internal_energy_roundoff_bound);
        EXPECT_NEAR(result.mass_error, 0., 1e-12);
        for (const double error : result.species_errors)
            EXPECT_NEAR(error, 0., 1e-12);
    }
}

TEST(LowMachMixtureSolverTest, UnresolvedLargeVolumeContinuityTargetRejectsWithoutPublishing)
{
    // At O(1 m3), binary64 EOS-minus-transported volume differences divided
    // by a tiny dt can exceed the original 1e-10 m3/s physical gate. The inner
    // pressure normalization must never turn that unresolved target into a pass.
    int rejections = 0;
    for (const double dt : {1e-4, 1e-6, 1e-8})
    {
        SCOPED_TRACE(dt);
        auto grid = mesh();
        const auto gas = ideal(catalog());
        Solver solver(grid, {gas}, options());
        solver.initialize(state(*grid, gas, true));
        const auto initial = solver.state();
        std::vector<Solver::CellSources> sources(initial.cells.size());
        for (size_t c = 0; c < sources.size(); ++c)
            sources[c].internal_energy = {grid->cell_centroid(static_cast<LO>(c)).z < .5 ? dt : 0.};
        try
        {
            const auto trial = solver.preview(dt, sources);
            EXPECT_LE(trial.diagnostics().maximum_occupancy_error, options().volume_relative_tolerance);
            EXPECT_LE(trial.diagnostics().maximum_continuity_residual, options().continuity_tolerance);
        }
        catch (const std::exception& error)
        {
            ++rejections;
            const std::string message = error.what();
            EXPECT_NE(message.find("global pressure compatibility: residual="), std::string::npos) << message;
            EXPECT_NE(message.find("original limit="), std::string::npos) << message;
            EXPECT_NE(message.find("dt="), std::string::npos) << message;
        }
        EXPECT_EQ(solver.state().revision, initial.revision);
        EXPECT_DOUBLE_EQ(solver.state().time, initial.time);
        EXPECT_DOUBLE_EQ(solver.state().thermodynamic_pressure, initial.thermodynamic_pressure);
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            EXPECT_EQ(solver.state().cells[c].phases[0].moles, initial.cells[c].phases[0].moles);
            EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].internal_energy,
                initial.cells[c].phases[0].internal_energy);
            EXPECT_DOUBLE_EQ(solver.velocity().value(static_cast<LO>(c)).z, 0.);
        }
        for (const auto face : solver.volume_flux().owned_face_ids())
            EXPECT_DOUBLE_EQ(solver.volume_flux().value(face), 0.);
    }
    EXPECT_GT(rejections, 0);
}

void ordinary_liquid_gas_contrast(int iteration_budget)
{
    const auto species = catalog();
    const auto gas = ideal(species);
    double maximum_rhs_relative_pressure_residual = 0, maximum_absolute_pressure_residual = 0;
    int maximum_pressure_iterations = 0;
    const auto run_case = [&](double compressibility, double dt)
    {
        SCOPED_TRACE(compressibility);
        SCOPED_TRACE(dt);
        Solver::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        auto grid = std::make_shared<Solver::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{ArrReal{0., .01}, ArrReal{0., .01},
                ArrReal{0., .00125, .0025, .00375, .005, .00625, .0075, .00875, .01}}), distribution);
        const auto liquid = R::make_linear_compressible_mixture_liquid(species, {250, 600, 10000, 2000000},
            [compressibility](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, compressibility, 0, 0, 0}; });
        auto controls = options();
        if (iteration_budget > 0)
            controls.linear.max_iterations = iteration_budget;
        Solver solver(grid, {liquid, gas}, controls);
        Solver::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(grid->num_owned_cells());
        std::vector<Solver::CellSources> sources(initial.cells.size());
        constexpr double gas_fractions[] = {1e-8, 1e-10, 1e-6, .01, .2, .8, .2, .01};
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const size_t index = std::min(size_t(grid->cell_centroid(cell).z / .00125), size_t(7));
            const double volume = grid->cell_volume(cell), alpha = gas_fractions[index];
            const double liquid_mass = 1000 * (1 - alpha) * volume;
            const double gas_moles = 100000 * alpha * volume / (gas_constant * 300);
            initial.cells[c].phases = {{liquid, {liquid_mass / .018, 0.}, 100000, 0., 300},
                {gas, {0., gas_moles}, 100000, 6000 * gas_moles, gas_moles > 0 ? 300. : 0.}};
            sources[c].internal_energy = {(1 + double(index)) * dt, 0.};
        }
        solver.initialize(initial);
        std::optional<Solver::Trial> staged;
        ASSERT_NO_THROW(staged.emplace(solver.preview(dt, sources)));
        const auto& trial = *staged;
        const auto& result = trial.diagnostics();
        EXPECT_TRUE(result.pressure_linear.converged);
        EXPECT_LE(result.pressure_linear.iterations, controls.linear.max_iterations);
        EXPECT_LE(result.pressure_linear.achieved_tolerance, controls.linear.tolerance);
        const double absolute_residual = result.pressure_linear.achieved_tolerance *
                                         std::max(result.pressure_linear.rhs_norm,
                                             result.pressure_linear_rhs_norm_floor);
        maximum_absolute_pressure_residual = std::max(maximum_absolute_pressure_residual, absolute_residual);
        if (result.pressure_linear.rhs_norm > 0)
            maximum_rhs_relative_pressure_residual =
                std::max(maximum_rhs_relative_pressure_residual, absolute_residual / result.pressure_linear.rhs_norm);
        maximum_pressure_iterations = std::max(maximum_pressure_iterations, result.pressure_linear.iterations);
        EXPECT_LE(controls.linear.tolerance * result.pressure_linear_rhs_norm_floor,
            controls.continuity_tolerance / 8 * (1 + 8 * std::numeric_limits<double>::epsilon()));
        // Rebuild final physical material continuity from public EOS/source
        // receipts and final carrier flux, independently of the Krylov metric.
        Solver::Traits::scalar_face_type material_flux(grid, 0., "independent_material_continuity");
        for (size_t f = 0; f < grid->num_owned_faces(); ++f)
        {
            const auto& receipt = trial.owned_phase_transport()[f];
            const double theta =
                std::accumulate(receipt.material_fraction.begin(), receipt.material_fraction.end(), 0.);
            material_flux.set_owned_value(static_cast<LO>(f), theta * trial.owned_volume_flux()[f]);
        }
        material_flux.sync_ghosts();
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            double eos_volume = 0, advected_volume = 0;
            for (size_t a = 0; a < 2; ++a)
            {
                const auto& phase = trial.state().cells[c].phases[a];
                const double amount = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
                auto fractions = phase.moles;
                double mass = 0;
                for (size_t species_index = 0; species_index < fractions.size(); ++species_index)
                {
                    mass += phase.moles[species_index] * species->species()[species_index].molarMass;
                    fractions[species_index] /= amount;
                }
                eos_volume += mass / phase.thermodynamics
                                         ->evaluate(phase.temperature, phase.absolute_pressure, fractions)
                                         .density;
                advected_volume += trial.transport_snapshot().cells[c].phases[a].advected_material_volume;
            }
            const auto cell = static_cast<LO>(c);
            EXPECT_LE(std::abs(eos_volume - grid->cell_volume(cell)) / grid->cell_volume(cell),
                controls.volume_relative_tolerance);
            EXPECT_LE(std::abs(FVM::cell_flux_balance(*grid, material_flux, cell) -
                               (eos_volume - advected_volume) / dt), controls.continuity_tolerance);
        }
        EXPECT_EQ(solver.state().revision, 0U);
        EXPECT_NO_THROW(solver.commit(trial));
        EXPECT_LE(result.maximum_occupancy_error, controls.volume_relative_tolerance);
        EXPECT_LE(result.maximum_continuity_residual, controls.continuity_tolerance);
        EXPECT_LE(std::abs(result.internal_energy_error),
            controls.conservation_relative_tolerance * result.caloric_energy_scale +
                result.internal_energy_roundoff_bound);
        for (const double error : result.species_errors)
            EXPECT_NEAR(error, 0., 1e-12);
    };
    for (const double compressibility : {4.5e-10, 0.})
        for (const double dt : {1e-2, 1e-4, 1e-6, 1e-8})
            run_case(compressibility, dt);
    const auto format = [](double value)
    {
        std::ostringstream text;
        text.precision(17);
        text << value;
        return text.str();
    };
    testing::Test::RecordProperty(
        "maximum_pressure_rhs_relative_residual", format(maximum_rhs_relative_pressure_residual));
    testing::Test::RecordProperty("maximum_pressure_absolute_residual", format(maximum_absolute_pressure_residual));
    testing::Test::RecordProperty("maximum_pressure_iterations", maximum_pressure_iterations);
}

struct EmptyGasTransportFixture
{
    SP<Solver::Mesh> grid;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    Solver::Options controls = options();
    Solver::State initial;
    std::vector<Solver::CellSources> sources;
    double dt = 1e-8, gas_reference = 6000.;

    explicit EmptyGasTransportFixture(bool captured_case, bool entirely_empty = false, double reference = 6000.)
        : gas_reference(reference)
    {
        Solver::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        ArrReal z = captured_case ? ArrReal{0., .00125, .0025, .00375, .005, .00625, .0075, .00875, .01}
                                  : ArrReal{0., .01 / 3, .02 / 3, .01};
        grid = std::make_shared<Solver::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                  Vec3D<ArrReal>{ArrReal{0., .01}, ArrReal{0., .01}, std::move(z)}),
            distribution);
        const auto species = catalog();
        liquid =
            R::make_linear_compressible_mixture_liquid(species, {250, 600, 10000, 2000000}, [](std::span<const double>)
                { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 4.5e-10, 0, 0, 0}; });
        gas = R::make_ideal_gas_mixture(
            species, {{20, 300, reference}, {20, 300, reference}}, {250, 600, 10000, 2000000});
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(grid->num_owned_cells());
        sources.resize(initial.cells.size());
        constexpr double contrast[] = {0., 1e-10, 1e-6, .01, .2, .8, .2, .01};
        for (std::size_t c = 0; c < initial.cells.size(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            const std::size_t index =
                std::min(std::size_t(grid->cell_centroid(cell).z / (captured_case ? .00125 : .01 / 3)),
                    captured_case ? std::size_t(7) : std::size_t(2));
            const double alpha = entirely_empty ? 0. : captured_case ? contrast[index] : index == 0 ? 0. : 1e-10;
            const double volume = grid->cell_volume(cell), mass = 1000 * (1 - alpha) * volume;
            const double n = 100000 * alpha * volume / (gas_constant * 300);
            initial.cells[c].phases = {{liquid, {mass / .018, 0.}, 100000, 0., 300},
                {gas, {captured_case ? 0. : .25 * n, captured_case ? n : .75 * n}, 100000, reference * n,
                    n > 0 ? 300. : 0.}};
            if (!captured_case)
                // Closed ALE walls require zero normal slip in their adjacent cells.
                // Interior drift still sends a tiny physical gas packet into the empty bottom cell.
                initial.cells[c].drift_velocity = {Solver::Vector{}, Solver::Vector{0., 0., index == 1 ? -1e-7 : 0.}};
            sources[c].internal_energy = {(1 + double(index)) * dt, 0.};
        }
    }
};

double independent_phase_volume(const R::PhaseThermalState& phase)
{
    const double n = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
    if (n == 0.)
        return 0.;
    double mass = 0.;
    auto x = phase.moles;
    for (std::size_t s = 0; s < x.size(); ++s)
    {
        mass += x[s] * phase.thermodynamics->catalog().species()[s].molarMass;
        x[s] /= n;
    }
    return mass / phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, x).density;
}

/** Rebuild stage-used implicit upwind fluxes from public snapshots. No final
 * pressure-corrected flux, EOS donor-volume denominator, or absolute stock floor enters this oracle. */
void expect_transport_face_balances(const EmptyGasTransportFixture& f, const Solver::Trial& trial)
{
    using CellField = Solver::Traits::scalar_cell_type;
    using FaceField = Solver::Traits::scalar_face_type;
    const auto& packet = trial.transport_snapshot();
    std::array<double, 3> local_maximum{}, maximum{};
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t channel = 0; channel < 4; ++channel)
        {
            CellField concentration(f.grid, 0., "independent_transport_concentration");
            FaceField face_flux(f.grid, 0., "independent_transport_face_flux");
            const auto amount = [channel](const Solver::TransportPhaseInventory& phase)
            {
                return channel < 2    ? phase.moles[channel]
                       : channel == 2 ? phase.internal_energy
                                      : phase.advected_material_volume;
            };
            for (std::size_t c = 0; c < packet.cells.size(); ++c)
                concentration.set_owned_value(
                    static_cast<LO>(c), amount(packet.cells[c].phases[a]) / packet.new_control_volumes[c]);
            concentration.sync_ghosts();
            for (std::size_t i = 0; i < f.grid->num_owned_faces(); ++i)
            {
                const LO face = static_cast<LO>(i);
                const auto& receipt = trial.owned_phase_transport()[i];
                const double phi = receipt.relative_phase_volume_flux[a];
                if (f.grid->is_geometry_exterior_face(face))
                {
                    EXPECT_DOUBLE_EQ(phi, 0.);
                    continue;
                }
                const LO owner = f.grid->owner_cell(face);
                const LO donor = phi >= 0 ? owner : f.grid->opposite_or_periodic_neighbor_cell(face, owner);
                face_flux.set_owned_value(
                    face, phi * (channel == 3 ? receipt.material_fraction[a] : concentration.local_value(donor)));
            }
            face_flux.sync_ghosts();
            for (std::size_t c = 0; c < packet.cells.size(); ++c)
            {
                const LO cell = static_cast<LO>(c);
                const auto& old = f.initial.cells[c].phases[a];
                const double before = channel < 2    ? old.moles[channel]
                                      : channel == 2 ? old.internal_energy
                                                     : independent_phase_volume(old);
                const double after = amount(packet.cells[c].phases[a]);
                long double integrated = 0., magnitude = 0.;
                for (std::size_t i = 0; i < f.grid->num_faces(); ++i)
                {
                    const LO face = static_cast<LO>(i);
                    const int sign = f.grid->owner_cell(face) == cell ? 1
                                     : !f.grid->is_geometry_exterior_face(face) && f.grid->neighbor_cell(face) == cell
                                         ? -1
                                         : 0;
                    if (sign)
                    {
                        const long double contribution =
                            static_cast<long double>(packet.time_step) * face_flux.local_value(face);
                        integrated += sign * contribution;
                        magnitude += std::abs(contribution);
                    }
                }
                const long double residual = static_cast<long double>(after) - before + integrated;
                const long double scale = std::abs(static_cast<long double>(before)) + std::abs(after) + magnitude;
                if (scale == 0.)
                    EXPECT_EQ(residual, 0.);
                else
                {
                    const double relative = static_cast<double>(std::abs(residual) / scale);
                    EXPECT_LE(relative, f.controls.conservation_relative_tolerance);
                    auto& channel_maximum = local_maximum[channel < 2 ? 0 : channel - 1];
                    channel_maximum = std::max(channel_maximum, relative);
                }
            }
        }
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, static_cast<int>(maximum.size()),
        local_maximum.data(), maximum.data());
    constexpr const char* names[] = {
        "maximum_species_face_balance", "maximum_energy_face_balance", "maximum_material_volume_face_balance"};
    for (std::size_t channel = 0; channel < maximum.size(); ++channel)
    {
        std::ostringstream value;
        value.precision(17);
        value << maximum[channel];
        testing::Test::RecordProperty(names[channel], value.str());
    }
}

TEST(LowMachMixtureSolverTest, EmptyGasFrontRetainsTinyPhysicalFaceArrivalAtOrdinaryWaterConditions)
{
    EmptyGasTransportFixture f(true);
    Solver solver(f.grid, {f.liquid, f.gas}, f.controls);
    solver.initialize(f.initial);
    std::optional<Solver::Trial> staged;
    ASSERT_NO_THROW(staged.emplace(solver.preview(f.dt, f.sources)));
    const auto& trial = *staged;
    expect_transport_face_balances(f, trial);
    EXPECT_GT(trial.diagnostics().maximum_initial_transport_residual, 1.);
    EXPECT_GT(trial.diagnostics().transport_consistency_refinements, 0);
    EXPECT_LE(trial.diagnostics().maximum_species_transport_residual, 1.);
    EXPECT_LE(trial.diagnostics().maximum_energy_transport_residual, 1.);
    testing::Test::RecordProperty(
        "transport_consistency_refinements", std::to_string(trial.diagnostics().transport_consistency_refinements));
    testing::Test::RecordProperty(
        "maximum_initial_transport_residual", std::to_string(trial.diagnostics().maximum_initial_transport_residual));
    std::array<double, 3> arrival{}, global_arrival{};
    for (std::size_t c = 0; c < f.initial.cells.size(); ++c)
        if (f.grid->cell_centroid(static_cast<LO>(c)).z < .00125)
        {
            const auto& gas = trial.transport_snapshot().cells[c].phases[1];
            arrival = {gas.moles[0] + gas.moles[1], gas.internal_energy, gas.advected_material_volume};
            EXPECT_GT(arrival[0], 0.);
            EXPECT_GT(gas.internal_energy, 0.);
            EXPECT_GT(gas.advected_material_volume, 0.);
            EXPECT_EQ(trial.state().cells[c].phases[1].moles, gas.moles);
            EXPECT_GT(trial.state().cells[c].phases[1].temperature, 0.);
        }
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(arrival.size()),
        arrival.data(), global_arrival.data());
    constexpr const char* names[] = {
        "empty_front_arriving_moles", "empty_front_arriving_internal_energy", "empty_front_advected_material_volume"};
    for (std::size_t channel = 0; channel < global_arrival.size(); ++channel)
    {
        EXPECT_GT(global_arrival[channel], 0.);
        std::ostringstream value;
        value.precision(17);
        value << global_arrival[channel];
        testing::Test::RecordProperty(names[channel], value.str());
    }
    EXPECT_NO_THROW(solver.commit(trial));
}

TEST(LowMachMixtureSolverTest, EmptyGasFrontAdvectsSignedCaloricReferencesInFixedAndALESteps)
{
    for (const bool moving : {false, true})
        for (const double reference : {-6000., 0., 6000.})
        {
            SCOPED_TRACE(moving);
            SCOPED_TRACE(reference);
            EmptyGasTransportFixture f(false, false, reference);
            Solver solver(f.grid, {f.liquid, f.gas}, f.controls);
            solver.initialize(f.initial);
            Solver::PlanarMotion motion(f.grid);
            std::optional<Solver::Trial> staged;
            ASSERT_NO_THROW(staged.emplace(moving ? solver.preview_ale(f.dt, motion, .01 * (1 + 1e-12), f.sources)
                                                  : solver.preview(f.dt, f.sources)));
            const auto& trial = *staged;
            EXPECT_FALSE(motion.has_active_trial());
            expect_transport_face_balances(f, trial);
            for (std::size_t c = 0; c < f.initial.cells.size(); ++c)
            {
                const auto& gas = trial.transport_snapshot().cells[c].phases[1];
                const double n = gas.moles[0] + gas.moles[1];
                EXPECT_EQ(trial.state().cells[c].phases[1].moles, gas.moles);
                if (n > 0.)
                    EXPECT_GT(trial.state().cells[c].phases[1].temperature, 0.);
                const double expected = reference * n;
                const double scale = std::abs(expected) + std::abs(gas.internal_energy);
                if (reference == 0.)
                    EXPECT_DOUBLE_EQ(gas.internal_energy, 0.);
                else
                    EXPECT_LE(
                        std::abs(gas.internal_energy - expected), f.controls.conservation_relative_tolerance * scale);
                if (f.grid->cell_centroid(static_cast<LO>(c)).z < .01 / 3)
                {
                    EXPECT_GT(n, 0.);
                    EXPECT_GT(gas.advected_material_volume, 0.);
                }
                // Phase energy after caloric closure includes pressure work and equilibrium heat sharing.
                // Proportionality above intentionally checks the pre-source transported U, not that final U.
            }
            if (moving)
                EXPECT_NO_THROW(solver.commit_ale(trial, motion));
            else
                EXPECT_NO_THROW(solver.commit(trial));
            EXPECT_EQ(solver.state().revision, 1U);
            int local_empty = f.grid->num_owned_cells() == 0, empty = 0;
            Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
            EXPECT_EQ(empty, std::max(0, f.grid->owned_cell_map()->getComm()->getSize() - 3));
        }
}

TEST(LowMachMixtureSolverTest, AbsentGasRemainsExactlyAbsentWithNoFictitiousEnergyOrVolume)
{
    for (const bool moving : {false, true})
    {
        EmptyGasTransportFixture f(false, true);
        Solver solver(f.grid, {f.liquid, f.gas}, f.controls);
        solver.initialize(f.initial);
        Solver::PlanarMotion motion(f.grid);
        const auto trial =
            moving ? solver.preview_ale(f.dt, motion, .01 * (1 + 1e-12), f.sources) : solver.preview(f.dt, f.sources);
        expect_transport_face_balances(f, trial);
        for (std::size_t c = 0; c < f.initial.cells.size(); ++c)
        {
            const auto& transported = trial.transport_snapshot().cells[c].phases[1];
            EXPECT_EQ(transported.moles, std::vector<double>({0., 0.}));
            EXPECT_DOUBLE_EQ(transported.internal_energy, 0.);
            EXPECT_DOUBLE_EQ(transported.advected_material_volume, 0.);
            const auto& accepted = trial.state().cells[c].phases[1];
            EXPECT_EQ(accepted.moles, transported.moles);
            EXPECT_DOUBLE_EQ(accepted.internal_energy, 0.);
            EXPECT_DOUBLE_EQ(accepted.temperature, 0.);
        }
        if (moving)
            solver.commit_ale(trial, motion);
        else
            solver.commit(trial);
    }
}

TEST(LowMachMixtureSolverTest, InsufficientTransportAccuracyBudgetRejectsWithoutPublishing)
{
    for (const bool loose_krylov : {false, true})
    {
        SCOPED_TRACE(loose_krylov);
        EmptyGasTransportFixture f(true);
        f.controls.linear.max_iterations = 1;
        if (loose_krylov)
            f.controls.linear.tolerance = 2.;
        for (auto& cell : f.initial.cells)
            cell.drift_velocity = {Solver::Vector{}, Solver::Vector{0., 0., -1.}};
        Solver solver(f.grid, {f.liquid, f.gas}, f.controls);
        solver.initialize(f.initial);
        const auto initial = solver.state();
        const auto initial_flux = solver.volume_flux().owned_data().getLocalViewHost(Tpetra::Access::ReadOnly);
        std::vector<double> flux;
        for (std::size_t i = 0; i < f.grid->num_owned_faces(); ++i)
            flux.push_back(initial_flux(i, 0));
        std::string failure;
        try
        {
            // At CFL=0.8, a deliberately loose Krylov stop passes its own metric.
            // One remaining balance sweep cannot satisfy the original local physical gate.
            (void) solver.preview(.001);
        }
        catch (const std::exception& error)
        {
            failure = error.what();
        }
        EXPECT_FALSE(failure.empty());
        if (loose_krylov)
            EXPECT_NE(failure.find("Low-Mach local transport exhausted the original scalar iteration budget:"),
                std::string::npos)
                << failure;
        else
            EXPECT_TRUE(
                failure.find("Low-Mach phase transport failed true linear residual gate.") != std::string::npos ||
                failure.find("Low-Mach local transport exhausted the original scalar iteration budget:") !=
                    std::string::npos)
                << failure;
        EXPECT_EQ(solver.state().revision, initial.revision);
        EXPECT_DOUBLE_EQ(solver.state().time, initial.time);
        EXPECT_DOUBLE_EQ(solver.state().thermodynamic_pressure, initial.thermodynamic_pressure);
        for (std::size_t c = 0; c < initial.cells.size(); ++c)
            for (std::size_t a = 0; a < 2; ++a)
            {
                EXPECT_EQ(solver.state().cells[c].phases[a].moles, initial.cells[c].phases[a].moles);
                EXPECT_DOUBLE_EQ(
                    solver.state().cells[c].phases[a].internal_energy, initial.cells[c].phases[a].internal_energy);
            }
        for (std::size_t i = 0; i < f.grid->num_owned_faces(); ++i)
            EXPECT_DOUBLE_EQ(solver.volume_flux().local_value(static_cast<LO>(i)), flux[i]);
    }
}

TEST(LowMachMixtureSolverTest, OrdinaryLiquidGasSusceptibilityContrastRetainsPhysicalGatesAtSmallTimeSteps)
{
    ordinary_liquid_gas_contrast(0);
}

TEST(LowMachMixtureSolverTest, PhysicalPressureAccuracyRespectsAnEightIterationBudgetAtSmallTimeSteps)
{
    // Eight unknowns permit a full Krylov space; the unchanged total iteration
    // cap must not be spent chasing a tiny RHS-relative roundoff remainder
    // after the original physical continuity/occupancy gates are resolved.
    ordinary_liquid_gas_contrast(8);
}

TEST(LowMachMixtureSolverTest, IndependentNonidealMixtureEOSClosesUniformHeatingWithoutIdealVolumeSubstitution)
{
    auto grid = mesh();
    const auto species = catalog();
    constexpr double b = 0.001;
    const auto gas = std::make_shared<const R::MixtureThermodynamics>(species,
        R::ThermodynamicBounds{250, 600, 10000, 2000000}, R::EOSKind::Gas,
        [species](double t, double p, std::span<const double> x)
        {
            const double mw = x[0] * species->species()[0].molarMass + x[1] * species->species()[1].molarMass;
            const double v = (gas_constant * t / p + b) / mw;
            R::MixtureThermodynamicPoint point{1 / v, 20 * t / mw, 20 * t / mw + p * v, 20 / mw, 0,
                -gas_constant / (mw * p * v * v), gas_constant * t / (mw * p * p * v * v), {}, {}};
            for (double fraction : x)
                point.fugacities.push_back(fraction * p * std::exp(b * p / (gas_constant * t)));
            return point;
        });
    auto initial = state(*grid, gas, true);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        const double n = 100000 * grid->cell_volume(static_cast<LO>(c)) / (gas_constant * 300 + 100000 * b);
        initial.cells[c].phases[0].moles = {0.2 * n, 0.8 * n};
        initial.cells[c].phases[0].internal_energy = 20 * n * 300;
    }
    Solver solver(grid, {gas}, options());
    solver.initialize(initial);
    std::vector<Solver::CellSources> sources(initial.cells.size());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {0.1 * initial.cells[c].phases[0].internal_energy};
    const auto result = solver.advance(0.01, sources);
    EXPECT_NEAR(solver.state().thermodynamic_pressure, 110000, 1e-3);
    EXPECT_NEAR(result.internal_energy_error, 0, 1e-6);
    for (const auto& cell : solver.state().cells)
        EXPECT_NEAR(cell.phases[0].temperature, 330, 1e-6);
}

TEST(LowMachMixtureSolverTest, FullGasBinConversionOwnsPressureWorkAndCreatesNoEmptyPhaseEnergy)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    auto initial = state(*grid, gas);
    for (auto& cell : initial.cells)
        cell.phases.push_back({gas, {0, 0}, 100000, 0, 0});
    Solver solver(grid, {gas, gas}, options());
    solver.initialize(initial);
    std::vector<Solver::CellSources> sources(initial.cells.size());
    for (size_t c = 0; c < sources.size(); ++c)
    {
        const auto& donor = solver.state().cells[c].phases[0];
        const auto moles = donor.moles;
        const double amount = moles[0] + moles[1];
        const double mass = moles[0] * 0.018 + moles[1] * 0.032;
        const auto properties = gas->evaluate(
            donor.temperature, donor.absolute_pressure, std::vector<double>{moles[0] / amount, moles[1] / amount});
        const double carried_enthalpy = mass * properties.enthalpy;
        sources[c].species_moles = {{-moles[0], -moles[1]}, moles};
        sources[c].internal_energy = {-carried_enthalpy, carried_enthalpy};
    }
    const auto result = solver.advance(0.01, sources);
    EXPECT_NEAR(result.internal_energy_error, 0, 1e-6);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].internal_energy, 0);
        EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].temperature, 0);
        EXPECT_EQ(solver.state().cells[c].phases[1].moles, initial.cells[c].phases[0].moles);
        EXPECT_NEAR(solver.state().cells[c].phases[1].temperature, 300, 1e-6);
    }
}

TEST(LowMachMixtureSolverTest, RankFixtureVerifiesPartitionFacesAndActualEmptyOwners)
{
    auto grid = mesh();
    const auto comm = grid->owned_cell_map()->getComm();
    const int local_occupied = grid->num_owned_cells() > 0 ? 1 : 0;
    int occupied = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_occupied, &occupied);
    EXPECT_EQ(occupied, std::min(comm->getSize(), 2));
    if (comm->getSize() >= 4)
        EXPECT_EQ(comm->getSize() - occupied, comm->getSize() - 2);
    int local_shared = 0;
    for (size_t f = 0; f < grid->num_owned_faces(); ++f)
    {
        const LO face = static_cast<LO>(f);
        if (grid->is_boundary_face(face))
            continue;
        const LO owner = grid->owner_cell(face), neighbor = grid->opposite_or_periodic_neighbor_cell(face, owner);
        if (static_cast<size_t>(owner) >= grid->num_owned_cells() ||
            static_cast<size_t>(neighbor) >= grid->num_owned_cells())
            ++local_shared;
    }
    int shared = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_shared, &shared);
    if (comm->getSize() > 1)
        EXPECT_GT(shared, 0);
}
TEST(LowMachMixtureSolverTest, FailedPreviewRetainsThePreviouslyValidImmutableTrial)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    const auto valid = solver.preview(0.01);
    std::vector<Solver::CellSources> invalid(grid->num_owned_cells());
    for (auto& source : invalid)
        source.species_moles = {{-1e100, 0}};
    EXPECT_THROW((void) solver.preview(0.01, invalid), std::exception);
    EXPECT_NO_THROW(solver.commit(valid));
    EXPECT_EQ(solver.state().revision, 1);
}
std::shared_ptr<const R::MixtureThermodynamics> reference_gas(
    std::shared_ptr<const R::SpeciesCatalog> species, double common_mass_reference)
{
    return R::make_ideal_gas_mixture(species,
        {{20, 300, .018 * common_mass_reference}, {20, 300, .032 * common_mass_reference}}, {250, 600, 10000, 2000000});
}
Solver::State reference_state(
    const Solver::Mesh& grid, std::shared_ptr<const R::MixtureThermodynamics> gas, double common_mass_reference)
{
    auto initial = state(grid, gas, true);
    for (auto& cell : initial.cells)
    {
        const double mass = .018 * cell.phases[0].moles[0] + .032 * cell.phases[0].moles[1];
        cell.phases[0].internal_energy = mass * common_mass_reference;
    }
    return initial;
}

TEST(LowMachMixtureSolverTest, ZeroInternalEnergyReferenceSupportsStationaryAndHeatOnlyStates)
{
    auto grid = mesh();
    const auto gas = reference_gas(catalog(), 0);
    Solver solver(grid, {gas}, options());
    solver.initialize(reference_state(*grid, gas, 0));
    const auto stationary = solver.advance(.01);
    EXPECT_GT(stationary.caloric_energy_scale, 0);
    EXPECT_NEAR(stationary.internal_energy_error, 0, 1e-6);
    for (const auto& cell : solver.state().cells)
        EXPECT_NEAR(cell.phases[0].temperature, 300, 1e-7);
    const auto before_heat = solver.state();
    std::vector<Solver::CellSources> sources(grid->num_owned_cells());
    for (auto& source : sources)
        source.internal_energy = {1000};
    const auto heated = solver.advance(.01, sources);
    EXPECT_NEAR(heated.internal_energy_error, 0, 1e-6);
    for (size_t c = 0; c < before_heat.cells.size(); ++c)
    {
        const double n = std::accumulate(
            before_heat.cells[c].phases[0].moles.begin(), before_heat.cells[c].phases[0].moles.end(), 0.0);
        EXPECT_NEAR(solver.state().cells[c].phases[0].temperature, 300 + 1000 / (20 * n), 1e-6);
    }
}

TEST(LowMachMixtureSolverTest, EquivalentCommonMassEnergyReferenceShiftPreservesTemperaturePressureAndFlow)
{
    auto grid = mesh();
    const auto species = catalog();
    const auto zero = reference_gas(species, 0), shifted = reference_gas(species, 1e8);
    Solver baseline(grid, {zero}, options()), alternative(grid, {shifted}, options());
    baseline.initialize(reference_state(*grid, zero, 0));
    alternative.initialize(reference_state(*grid, shifted, 1e8));
    std::vector<Solver::CellSources> sources(grid->num_owned_cells());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {grid->cell_global_id(static_cast<LO>(c)) == 0 ? 100.0 : 0.0};
    Solver::Diagnostics a, b;
    ASSERT_NO_THROW(a = baseline.advance(.01, sources));
    ASSERT_NO_THROW(b = alternative.advance(.01, sources));
    EXPECT_NEAR(baseline.state().thermodynamic_pressure, alternative.state().thermodynamic_pressure, .001);
    for (size_t c = 0; c < sources.size(); ++c)
    {
        EXPECT_NEAR(
            baseline.state().cells[c].phases[0].temperature, alternative.state().cells[c].phases[0].temperature, 1e-6);
        for (size_t component = 0; component < 3; ++component)
            EXPECT_NEAR(baseline.state().cells[c].velocity.component(component),
                alternative.state().cells[c].velocity.component(component), 1e-8);
    }
    for (const auto face : baseline.volume_flux().owned_face_ids())
        EXPECT_NEAR(baseline.volume_flux().value(face), alternative.volume_flux().value(face), 1e-9);
    EXPECT_NEAR(a.caloric_energy_scale, b.caloric_energy_scale, 1e-3);
    EXPECT_GT(b.internal_energy_roundoff_bound, a.internal_energy_roundoff_bound);
}

TEST(LowMachMixtureSolverTest, IllConditionedFormationReferenceCannotSilentlyLoosenCaloricAccuracy)
{
    auto grid = mesh();
    const auto gas = reference_gas(catalog(), 1e10);
    Solver solver(grid, {gas}, options());
    EXPECT_THROW(solver.initialize(reference_state(*grid, gas, 1e10)), std::exception);
    EXPECT_FALSE(solver.initialized());
}

SP<Solver::Mesh> extended_mesh(bool semi_structured = false)
{
    Solver::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    if (!semi_structured)
        return std::make_shared<Solver::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(
                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .125, .25, .375, .5, .625, .75, .875, 1}}),
            distribution);
    using Semi = Meshes::SemiStructuredXY_Z;
    Arr<Semi::Vec3> nodes;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 9; ++x)
            nodes.push_back({double(x), double(y), 0});
    Arr<Arr<unsigned>> cells;
    for (unsigned parity = 0; parity < 2; ++parity)
        for (unsigned x = parity; x < 8; x += 2)
            cells.push_back({x, x + 1, x + 10, x + 9});
    distribution.xy_partitions = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    distribution.z_partitions = 1;
    return std::make_shared<Solver::Mesh>(std::make_shared<Semi>(nodes, cells, ArrReal{0, .2, .6, 1}), distribution);
}

TEST(LowMachMixtureSolverTest, ExtendedPartitionHaloSupportsTheOriginalStrongHeatingGate)
{
    auto grid = extended_mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas, true));
    std::vector<Solver::CellSources> sources(grid->num_owned_cells());
    for (size_t c = 0; c < sources.size(); ++c)
        sources[c].internal_energy = {grid->cell_centroid(static_cast<LO>(c)).z < .125 ? 100.0 : 0.0};
    const auto trial = solver.preview(.01, sources);
    EXPECT_LE(trial.diagnostics().maximum_occupancy_error, options().volume_relative_tolerance);
    EXPECT_LE(trial.diagnostics().maximum_continuity_residual, options().continuity_tolerance);
    EXPECT_EQ(solver.state().revision, 0U);
    EXPECT_NO_THROW(solver.commit(trial));
    EXPECT_EQ(solver.state().revision, 1U);
}

TEST(LowMachPressureEquationTest, ManufacturedPotentialProjectsAcrossFringesAndContiguousSemiAlgebraicMaps)
{
    using Pressure = R::LowMachPressureEquation;
    for (const bool semi : {false, true})
    {
        SCOPED_TRACE(semi ? "decomposed XY/Z geometry with separate algebraic IDs" : "extended Cartesian halo");
        auto grid = extended_mesh(semi);
        Pressure equation(grid);
        Pressure::ScalarField density(grid, 2.0, "projection_density"), pressure(grid, 0.0, "projection_pressure");
        Pressure::VectorField velocity(grid, "projection_velocity");
        Pressure::FaceField predictor(grid, 0.0, "projection_predictor"), corrected(grid, 7.0, "projection_output");
        velocity.owned_data().putScalar(0);
        density.sync_ghosts();
        pressure.sync_ghosts();
        velocity.sync_ghosts();
        const auto potential = [&](LO cell)
        {
            const auto center = grid->cell_centroid(cell);
            return center.x * center.x + .3 * center.z * center.z;
        };
        const auto gauge = grid->owned_cell_map()->getMinAllGlobalIndex();
        double local_gauge = 0, gauge_value = 0;
        int local_different_ids = 0, different_ids = 0, local_fringe = 0, fringe = 0;
        for (size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            if (grid->cell_global_id(cell) == gauge)
                local_gauge = potential(cell);
            local_different_ids += grid->cell_global_id(cell) != grid->cell_geometry_global_id(cell);
        }
        for (size_t f = 0; f < grid->num_faces(); ++f)
        {
            const LO face = static_cast<LO>(f);
            if (!grid->is_geometry_exterior_face(face) && (grid->owner_cell(face) < 0 || grid->neighbor_cell(face) < 0))
                ++local_fringe;
        }
        for (const auto face : predictor.owned_face_ids())
        {
            if (grid->is_geometry_exterior_face(face))
                continue;
            const LO owner = grid->owner_cell(face), neighbor = grid->neighbor_cell(face);
            const double mobility = .01 / 2 * grid->face_area(face) / grid->face_cell_center_distance(face);
            predictor.set_owned_value(face, mobility * (potential(owner) - potential(neighbor)));
            local_different_ids += grid->face_global_id(face) != grid->face_geometry_global_id(face);
        }
        predictor.sync_ghosts();
        const auto comm = grid->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_gauge, &gauge_value);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_different_ids, &different_ids);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_fringe, &fringe);
        if (comm->getSize() > 1)
            EXPECT_GT(fringe, 0);
        if (semi && comm->getSize() > 1)
        {
            EXPECT_TRUE(grid->owned_cell_map()->isContiguous());
            EXPECT_TRUE(grid->owned_face_map()->isContiguous());
            EXPECT_GT(different_ids, 0);
        }
        std::vector<double> target(grid->num_owned_cells(), 0);
        const auto result =
            equation.project(density, .01, target, predictor, corrected, pressure, velocity, options().linear);
        EXPECT_LE(result.maximum_continuity_residual, options().continuity_tolerance);
        for (size_t c = 0; c < target.size(); ++c)
            EXPECT_NEAR(pressure.value(static_cast<LO>(c)), gauge_value - potential(static_cast<LO>(c)), 1e-8);
        for (const auto face : corrected.owned_face_ids())
            EXPECT_NEAR(corrected.value(face), 0, 1e-10);
        std::vector<double> pressure_before(target.size());
        for (size_t c = 0; c < target.size(); ++c)
            pressure_before[c] = pressure.value(static_cast<LO>(c));
        std::vector<double> flux_before;
        for (const auto face : corrected.owned_face_ids())
            flux_before.push_back(corrected.value(face));
        if (!target.empty())
            target[0] = 1;
        EXPECT_THROW(
            (void) equation.project(density, .01, target, predictor, corrected, pressure, velocity, options().linear),
            std::exception);
        for (size_t c = 0; c < target.size(); ++c)
            EXPECT_DOUBLE_EQ(pressure.value(static_cast<LO>(c)), pressure_before[c]);
        size_t index = 0;
        for (const auto face : corrected.owned_face_ids())
            EXPECT_DOUBLE_EQ(corrected.value(face), flux_before[index++]);
    }
}

TEST(LowMachMixtureSolverTest, RestoreUsesOnlyTheAcceptedCheckpointAndDoesNotInvokeConstitutiveProviders)
{
    auto grid = mesh();
    const auto species = catalog();
    const auto base = ideal(species);
    const auto unavailable = std::make_shared<bool>(false);
    const auto gas = std::make_shared<const R::MixtureThermodynamics>(
        species, base->bounds(), base->kind(),
        [base, unavailable](double t, double p, std::span<const double> x)
        {
            if (*unavailable)
                throw std::runtime_error("Constitutive evaluation must not run during checkpoint restore.");
            return base->evaluate(t, p, x);
        },
        base->energy_convention());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    const auto initial = solver.state();
    const auto saved = solver.checkpoint();
    std::vector<Solver::CellSources> sources(grid->num_owned_cells());
    for (auto& source : sources)
        source.internal_energy = {100};
    solver.advance(.01, sources);
    const auto expected = solver.state();
    const auto superseded = solver.preview(.01);
    *unavailable = true;
    EXPECT_NO_THROW(solver.restore(saved));
    EXPECT_EQ(solver.state().revision, initial.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, initial.time);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        EXPECT_EQ(solver.state().cells[c].phases[0].moles, initial.cells[c].phases[0].moles);
        EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].internal_energy, initial.cells[c].phases[0].internal_energy);
        EXPECT_DOUBLE_EQ(solver.state().cells[c].phases[0].temperature, initial.cells[c].phases[0].temperature);
    }
    EXPECT_THROW(solver.commit(superseded), std::exception);
    *unavailable = false;
    solver.advance(.01, sources);
    EXPECT_DOUBLE_EQ(solver.state().time, expected.time);
    EXPECT_EQ(solver.state().revision, expected.revision);
    EXPECT_NEAR(solver.state().thermodynamic_pressure, expected.thermodynamic_pressure, 1e-8);
}

TEST(LowMachMixtureSolverTest, UnrepresentableClockAndRevisionRejectWithoutPublishing)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    for (const bool revision_overflow : {false, true})
    {
        auto initial = state(*grid, gas);
        initial.time = revision_overflow ? 0 : 1e300;
        initial.revision = revision_overflow ? std::numeric_limits<std::uint64_t>::max() : 0;
        solver.initialize(initial);
        EXPECT_THROW((void) solver.preview(.01), std::exception);
        EXPECT_DOUBLE_EQ(solver.state().time, initial.time);
        EXPECT_EQ(solver.state().revision, initial.revision);
    }
    solver.initialize(state(*grid, gas));
    const auto valid = solver.preview(.01);
    auto invalid = solver.state();
    if (grid->owned_cell_map()->getComm()->getRank() == 0)
        invalid.time = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(solver.initialize(invalid), std::exception);
    EXPECT_NO_THROW(solver.commit(valid));
}

TEST(LowMachMixtureSolverTest, RankDivergentSchemaEOSBoundsAndClosurePresenceRejectCollectively)
{
    auto grid = mesh();
    const auto comm = grid->owned_cell_map()->getComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Rank-asymmetric metadata requires multiple ranks.";
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto definitions = catalog()->species();
        if (scenario == 0 && comm->getRank() + 1 == comm->getSize())
            definitions[0].molarMass *= 2;
        const auto schema = std::make_shared<const R::SpeciesCatalog>(std::move(definitions));
        auto bounds = R::ThermodynamicBounds{250, 600, 10000, 2000000};
        if (scenario == 1 && comm->getRank() + 1 == comm->getSize())
            bounds.maximum_temperature = 650;
        const auto gas = R::make_ideal_gas_mixture(schema, {{20, 300, 6000}, {20, 300, 6000}}, bounds);
        auto controls = options();
        if (scenario == 2 && comm->getRank() + 1 == comm->getSize())
            controls.mixture_dynamic_viscosity = [](std::span<const R::PhaseThermalState>, std::span<const double>)
            { return 0.; };
        EXPECT_THROW((void) Solver(grid, {gas}, controls), std::exception);
    }
}

TEST(LowMachMixtureSolverTest, StrongCoolingSafeguardsPressureAndCaloricRangesWithoutWeakeningTheEnergyGate)
{
    auto grid = mesh();
    const auto species = catalog();
    const auto gas = R::make_ideal_gas_mixture(
        species, {{20, 300, 6000}, {20, 300, 6000}}, R::ThermodynamicBounds{119, 600, 10000, 2000000});
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas, true));
    const auto initial = solver.state();
    const auto saved = solver.checkpoint();
    std::vector<Solver::CellSources> source(initial.cells.size());
    for (size_t c = 0; c < source.size(); ++c)
        source[c].internal_energy = {-.6 * initial.cells[c].phases[0].internal_energy};
    const auto staged = solver.preview(.01, source);
    EXPECT_DOUBLE_EQ(solver.state().thermodynamic_pressure, initial.thermodynamic_pressure);
    EXPECT_EQ(solver.state().revision, initial.revision);
    EXPECT_NEAR(staged.state().thermodynamic_pressure, 40000, 1e-5);
    EXPECT_LE(staged.diagnostics().maximum_occupancy_error, options().volume_relative_tolerance);
    EXPECT_LE(staged.diagnostics().maximum_continuity_residual, options().continuity_tolerance);
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        EXPECT_NEAR(staged.state().cells[c].phases[0].temperature, 120, 1e-7);
        EXPECT_NEAR(
            staged.state().cells[c].phases[0].internal_energy, .4 * initial.cells[c].phases[0].internal_energy, 1e-6);
        EXPECT_EQ(staged.state().cells[c].phases[0].moles, initial.cells[c].phases[0].moles);
    }
    auto invalid = source;
    for (size_t c = 0; c < invalid.size(); ++c)
        invalid[c].internal_energy = {-2 * initial.cells[c].phases[0].internal_energy};
    EXPECT_THROW((void) solver.preview(.01, invalid), std::exception);
    EXPECT_NO_THROW(solver.commit(staged));
    const auto expected = solver.state();
    solver.restore(saved);
    solver.advance(.01, source);
    EXPECT_EQ(solver.state().revision, expected.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, expected.time);
    EXPECT_NEAR(solver.state().thermodynamic_pressure, expected.thermodynamic_pressure, 1e-8);
}

TEST(LowMachMixtureSolverTest, ConstitutiveProviderFailuresRemainHardErrorsDuringPressureSearch)
{
    auto grid = mesh();
    const auto species = catalog();
    const auto base = R::make_ideal_gas_mixture(
        species, {{20, 300, 6000}, {20, 300, 6000}}, R::ThermodynamicBounds{119, 600, 10000, 2000000});
    const auto failures = std::make_shared<int>(0);
    const auto gas = std::make_shared<const R::MixtureThermodynamics>(
        species, base->bounds(), base->kind(),
        [base, failures](double t, double p, std::span<const double> x)
        {
            if (p < 80000)
            {
                ++*failures;
                throw std::runtime_error("Actual constitutive failure; not a caloric search condition.");
            }
            return base->evaluate(t, p, x);
        },
        base->energy_convention());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    const auto initial = solver.state();
    const auto retained = solver.preview(.01);
    std::vector<Solver::CellSources> source(initial.cells.size());
    for (size_t c = 0; c < source.size(); ++c)
        source[c].internal_energy = {-.6 * initial.cells[c].phases[0].internal_energy};
    EXPECT_THROW((void) solver.preview(.01, source), std::exception);
    EXPECT_EQ(*failures, grid->num_owned_cells() ? 1 : 0);
    EXPECT_EQ(solver.state().revision, initial.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, initial.time);
    EXPECT_NO_THROW(solver.commit(retained));
}

TEST(LowMachMixtureSolverTest, RankAsymmetricLeaseReleaseRejectsBeforeAnyTrialPublication)
{
    auto grid = mesh();
    const auto comm = grid->owned_cell_map()->getComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Rank-asymmetric ownership requires multiple ranks.";
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    solver.initialize(state(*grid, gas));
    std::optional<Solver::ControllerLease> lease;
    lease.emplace(solver.acquire_controller());
    if (comm->getRank() == 0)
        lease.reset();
    EXPECT_THROW((void) solver.preview(.01, {}, lease ? &*lease : nullptr), std::exception);
    EXPECT_EQ(solver.state().revision, 0U);
    EXPECT_DOUBLE_EQ(solver.state().time, 0);
    lease.reset();
    EXPECT_NO_THROW((void) solver.advance(.01));
}

TEST(LowMachMixtureSolverTest, FixedGeometryRejectsChangedEpochBeforeInitializationOrProjection)
{
    auto grid = mesh();
    const auto gas = ideal(catalog());
    Solver solver(grid, {gas}, options());
    R::LowMachPressureEquation pressure_equation(grid);
    PlanarALEMeshMotion<Solver::Pack> motion(grid);
    motion.begin_trial(1.1, .01);
    EXPECT_THROW(solver.initialize(state(*grid, gas)), std::exception);
    EXPECT_FALSE(solver.initialized());
    using Pressure = R::LowMachPressureEquation;
    Pressure::ScalarField density(grid, 1.0, "epoch_density"), pressure(grid, 0.0, "epoch_pressure");
    Pressure::VectorField velocity(grid, "epoch_velocity");
    velocity.owned_data().putScalar(0);
    Pressure::FaceField predictor(grid, 0.0, "epoch_predictor"), corrected(grid, 9.0, "epoch_output");
    density.sync_ghosts();
    pressure.sync_ghosts();
    velocity.sync_ghosts();
    predictor.sync_ghosts();
    const std::vector<double> target(grid->num_owned_cells(), 0);
    EXPECT_THROW((void) pressure_equation.project(
                     density, .01, target, predictor, corrected, pressure, velocity, options().linear),
        std::exception);
    for (const auto face : corrected.owned_face_ids())
        EXPECT_DOUBLE_EQ(corrected.value(face), 9);
    motion.rollback_trial();
    Solver fresh(grid, {gas}, options());
    EXPECT_NO_THROW(fresh.initialize(state(*grid, gas)));
}
} // namespace
