/** @file testBoussinesqSpeciesFreeSurface.cc
 * @brief fixed-grid typed species/headspace/follower and rollback regressions.
 * Registered for serial and two-/four-rank execution, including empty owners.
 */
#include "ReactingMixtureTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ClosedMixtureHeadspaceModel.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>

namespace
{
using namespace SimpleFluid;
namespace M = Testing::ReactingMixture;
namespace R = Reacting;
using Pack = DefaultTpetraTypes;
using Grid = MeshHandle<Pack>;
using Solver = BoussinesqSolver<Pack>;
using Coupler = MultiphysicsSolverCoupler<Pack>;
using Model = Coupler::reacting_gas_model_type;
using LO = Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

BoundaryConditionSet closed(const Grid& grid)
{
    BoundaryConditionSet result;
    for (const auto& [id, batch] : grid.boundary_batches())
    {
        const auto name = grid.boundary_batch_name(id);
        result.velocity[name] = {BoundaryConditionType::NoSlip, {0, 0, 0}};
        result.pressure[name] = {BoundaryConditionType::Neumann, 0};
        result.temperature[name] = {BoundaryConditionType::Neumann, 0};
    }
    return result;
}
double integral(const Model::field_type& field)
{
    double local = 0, global = 0;
    for (std::size_t i = 0; i < field.mesh().num_owned_cells(); ++i)
        local += field.value(static_cast<LO>(i)) * field.mesh().cell_volume(static_cast<LO>(i));
    Teuchos::reduceAll(*field.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
struct Fixture
{
    M::Mixture chemistry{false, true, R::EnergyConvention::SensibleEnergy, 0};
    SP<Grid> grid;
    std::shared_ptr<Solver> solver;
    std::unique_ptr<Coupler> coupling;
    Model* gas = nullptr;
    double pressure;
    explicit Fixture(bool bubbles = true, double reference_pressure = 100000) : pressure(reference_pressure)
    {
        Grid::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        grid = std::make_shared<Grid>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                          Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        TimeStepperOptions time;
        time.time_step = .001;
        time.kinematic_viscosity = 0;
        time.thermal_diffusivity = 0;
        time.thermal_expansion = 0;
        time.gravity_x = 0;
        time.gravity_y = 0;
        time.gravity_z = 0;
        BoussinesqModelOptions material;
        material.reference_density = 1000;
        material.density = 1000;
        material.specific_heat_capacity = 4000;
        material.dynamic_viscosity = .001;
        material.thermal_conductivity = 0;
        LinearSolverOptions linear;
        linear.preconditioner = LinearPreconditioner::None;
        linear.tolerance = 1e-12;
        solver = std::make_shared<Solver>(grid, closed(*grid), time, linear, material);
        coupling = std::make_unique<Coupler>(solver);
        M::InitialMixture seed;
        seed.product_concentration = .2;
        if (bubbles)
            seed.populations = {{"micro", 1e4, 1e-9, .25}, {"large", 2e3, 2e-9, .6}};
        auto options = M::field_options(chemistry, seed);
        options.reference_pressure = pressure;
        options.species_properties = {
            chemistry.transport(M::gas_one, 0, 1e-5), chemistry.transport(M::gas_two, 0, 2e-5)};
        options.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
        options.constant_slip_velocity = .02;
        options.free_surface_patches = {"zmax"};
        // A finite reverse reaction consumes both dissolved gas components.
        options.species_source = [](const R::ReactionContext& context, double dt)
        {
            R::ReactionSourceInput input;
            input.integratedExtents = {{M::conversion, -1e-4 * dt * context.environment.bulkVolume}};
            return input;
        };
        gas = &coupling->configure_reacting_gas(options);
        coupling->initialize_heated_box(300, 300);
    }
    FreeSurfaceOptions surface_options(bool closed_headspace = false) const
    {
        FreeSurfaceOptions options;
        options.enabled = true;
        options.mode = FreeSurfaceMode::PlanarVolumeBudget;
        options.vessel.bottom_elevation = 0;
        options.vessel.top_elevation = 2;
        options.vessel.cross_section_area = 1;
        options.vessel.total_internal_volume = 2;
        options.liquid_mass.mode = LiquidVolumeMode::CellMassInventory;
        options.liquid_mass.initial_liquid_mass = integral(gas->liquid_mass_density());
        options.headspace.mode = closed_headspace ? HeadspaceMode::Closed : HeadspaceMode::Vented;
        options.headspace.total_internal_volume = 2;
        options.headspace.ambient_pressure = pressure;
        options.headspace.initial_pressure = pressure;
        options.headspace.initial_temperature = 300;
        options.coupling.minimum_absolute_pressure = M::bounds.minimum_pressure;
        options.coupling.maximum_absolute_pressure = M::bounds.maximum_pressure;
        if (closed_headspace)
        {
            options.headspace.infer_initial_moles = false;
            // Independent excluded-volume law for the explicitly 50/50 headspace.
            const double pool = gas->evaluate_liquid_volume(pressure) + gas->evaluate_submerged_bubble_volume(pressure);
            const double n = pressure * (2 - pool) / (M::gas_constant * 300 + .002 * pressure);
            options.headspace.initial_moles = {{"gas_one", .5 * n}, {"gas_two", .5 * n}};
        }
        return options;
    }
    void configure_surface(FreeSurfaceOptions options)
    {
        if (options.headspace.mode == HeadspaceMode::Closed)
            coupling->configure_free_surface(options, std::make_unique<ClosedMixtureHeadspaceModel>(
                                                          options.headspace, chemistry.gas, "manufactured-mixture", 1));
        else
            coupling->configure_free_surface(options);
    }
};
TEST(BoussinesqSpeciesFreeSurfaceTest, ConsumesBothActualGasSpeciesAndFollowsLiquidWithoutSecondTransport)
{
    Fixture f(false);
    f.configure_surface(f.surface_options());
    const auto before = f.coupling->free_surface_diagnostics();
    const auto mass = f.coupling->liquid_mass_inventory().totalMass();
    EXPECT_TRUE(f.coupling->liquid_mass_inventory().speciesBound());
    EXPECT_ANY_THROW((void) f.coupling->liquid_mass_inventory().previewCellwiseAdvance(
        .001, f.solver->pressure_corrected_face_fluxes(), nullptr, nullptr, LinearSolverOptions{}));
    f.coupling->step();
    const auto after = f.coupling->free_surface_diagnostics();
    for (const auto& name : {"gas_one", "gas_two"})
    {
        EXPECT_NEAR(after.consumed_gas_moles.at(name), 1e-7, 1e-18);
        EXPECT_NEAR(after.net_source_gas_moles->at(name), -1e-7, 1e-18);
        EXPECT_NEAR(before.dissolved_gas_moles.at(name) - after.dissolved_gas_moles.at(name), 1e-7, 1e-14);
        EXPECT_NEAR(after.gas_closure_by_species.at(name), 0., 1e-13);
    }
    EXPECT_NEAR(f.coupling->liquid_mass_inventory().totalMass(), mass, 1e-10);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_DOUBLE_EQ(f.coupling->liquid_mass_inventory().cellMassInventory().value(static_cast<LO>(c)),
            f.gas->liquid_mass_density().value(static_cast<LO>(c)));
    EXPECT_EQ(f.gas->species_event_receipt().revision, 1u);
    int local_empty = f.grid->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (f.grid->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(BoussinesqSpeciesFreeSurfaceTest, TransfersOnlyExactAcceptedGasEscapePacketsToTheVent)
{
    Fixture f;
    f.configure_surface(f.surface_options());
    f.coupling->step();
    GasMolesBySpecies expected;
    for (const auto& event : f.gas->physical_escape_events())
        if (event.phase == R::SpeciesPhase::Dispersed || event.phase == R::SpeciesPhase::Resolved)
            expected[f.chemistry.catalog->species()[f.chemistry.catalog->index(event.species)].name] += event.moles;
    const auto after = f.coupling->free_surface_diagnostics();
    for (const auto& name : {"gas_one", "gas_two"})
    {
        EXPECT_GT(expected.at(name), 0);
        EXPECT_DOUBLE_EQ(after.escaped_gas_moles_this_step.at(name), expected.at(name));
        EXPECT_DOUBLE_EQ(after.vented_gas_moles.at(name), expected.at(name));
    }
    EXPECT_TRUE(after.submerged_population_gas_moles.contains("micro"));
    EXPECT_TRUE(after.submerged_population_gas_moles.contains("large"));
}
TEST(BoussinesqSpeciesFreeSurfaceTest, RejectsManuallyAdvancedModelWithoutLosingItsEvent)
{
    Fixture f(false);
    f.configure_surface(f.surface_options());
    const auto surface = f.coupling->free_surface_diagnostics();
    f.gas->advance(.001, .001, f.solver->temperature(), f.solver->pressure(), f.solver->velocity(),
        f.solver->pressure_corrected_face_fluxes(), f.solver->material_properties(), nullptr);
    const auto receipt = f.gas->species_event_receipt();
    const M::SavedReactingState<Model> saved(*f.gas);
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_EQ(f.gas->species_event_receipt(), receipt);
    saved.expect_restored(*f.gas);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    EXPECT_EQ(f.coupling->free_surface_diagnostics().dissolved_gas_moles, surface.dissolved_gas_moles);
}
TEST(BoussinesqSpeciesFreeSurfaceTest, ClosedEOSFailureRollsBackFieldsReceiptHeadspaceAndCanRetry)
{
    Fixture f(true, 1500000);
    auto options = f.surface_options(true);
    options.headspace.temperature_mode = HeadspaceTemperatureMode::Prescribed;
    options.headspace.prescribed_temperature_times = {0, .001};
    options.headspace.prescribed_temperature_values = {300, 500};
    f.configure_surface(options);
    const auto receipt = f.gas->species_event_receipt();
    const auto surface = f.coupling->free_surface_diagnostics();
    const auto mass = f.coupling->liquid_mass_inventory().totalMass();
    const auto history = f.coupling->free_surface_history().size();
    const M::SavedReactingState<Model> saved(*f.gas);
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_EQ(f.gas->species_event_receipt(), receipt);
    saved.expect_restored(*f.gas);
    EXPECT_DOUBLE_EQ(f.coupling->free_surface_diagnostics().headspace.pressure, surface.headspace.pressure);
    EXPECT_DOUBLE_EQ(f.coupling->liquid_mass_inventory().totalMass(), mass);
    EXPECT_EQ(f.coupling->free_surface_history().size(), history);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    f.solver->set_time_step(.001 / 16);
    EXPECT_NO_THROW(f.coupling->step());
    EXPECT_EQ(f.gas->species_event_receipt().revision, receipt.revision + 1);
}
TEST(BoussinesqSpeciesFreeSurfaceTest, ClosedHeadspaceRejectsMissingForeignOrInferredSchema)
{
    Fixture f(false);
    auto options = f.surface_options(true);
    EXPECT_ANY_THROW(f.coupling->configure_free_surface(options));
    M::Mixture foreign(true, true, R::EnergyConvention::SensibleEnergy, 0);
    EXPECT_ANY_THROW(f.coupling->configure_free_surface(
        options, std::make_unique<ClosedMixtureHeadspaceModel>(options.headspace, foreign.gas, "foreign", 1)));
    options.headspace.infer_initial_moles = true;
    options.headspace.initial_moles.clear();
    EXPECT_ANY_THROW(f.coupling->configure_free_surface(
        options, std::make_unique<ClosedMixtureHeadspaceModel>(options.headspace, f.chemistry.gas, "inferred", 1)));
}

TEST(BoussinesqSpeciesFreeSurfaceTest, ModelOnlyRestoreCannotReuseAConsumedLogicalRevision)
{
    Fixture f(false);
    f.configure_surface(f.surface_options());
    const auto before = f.gas->snapshot();
    f.coupling->step();
    const auto consumed = f.gas->species_event_receipt();
    const auto surface = f.coupling->free_surface_diagnostics();
    f.gas->restore(before);
    f.gas->advance(.001, .001, f.solver->temperature(), f.solver->pressure(), f.solver->velocity(),
        f.solver->pressure_corrected_face_fluxes(), f.solver->material_properties(), nullptr);
    const auto replacement = f.gas->species_event_receipt();
    EXPECT_EQ(replacement.revision, consumed.revision);
    EXPECT_DOUBLE_EQ(replacement.interval_end, consumed.interval_end);
    EXPECT_GT(replacement.publication_nonce, consumed.publication_nonce);
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_EQ(f.gas->species_event_receipt(), replacement);
    EXPECT_EQ(f.coupling->free_surface_diagnostics().dissolved_gas_moles, surface.dissolved_gas_moles);
    EXPECT_DOUBLE_EQ(f.solver->time(), .001);
}

struct HeadspaceProbe
{
    int rank = 0, trial_calls = 0, commit_calls = 0;
};
class FaultHeadspace final : public HeadspaceModel
{
    std::unique_ptr<HeadspaceModel> core;
    std::shared_ptr<HeadspaceProbe> probe;
    bool finite_divergence;
    static HeadspaceMetadata binding(const std::shared_ptr<const R::MixtureThermodynamics>& eos)
    {
        return {eos->catalog().fingerprint(), "manufactured-fault-headspace", 1,
            {eos->bounds().minimum_temperature, eos->bounds().maximum_temperature, eos->bounds().minimum_pressure,
                eos->bounds().maximum_pressure},
            true};
    }
    FaultHeadspace(const FaultHeadspace& other)
        : HeadspaceModel(other.options(), other.metadata()), core(other.core->clone()), probe(other.probe),
          finite_divergence(other.finite_divergence)
    {
        setState(other.state());
    }

public:
    FaultHeadspace(HeadspaceOptions options, std::shared_ptr<const R::MixtureThermodynamics> eos,
        std::shared_ptr<HeadspaceProbe> trace, bool divergent)
        : HeadspaceModel(options, binding(eos)), core(std::make_unique<ClosedMixtureHeadspaceModel>(std::move(options),
                                                     std::move(eos), "manufactured-fault-headspace", 1)),
          probe(std::move(trace)), finite_divergence(divergent)
    {
    }
    std::unique_ptr<HeadspaceModel> clone() const override
    {
        return std::unique_ptr<HeadspaceModel>(new FaultHeadspace(*this));
    }
    HeadspaceMode mode() const noexcept override { return HeadspaceMode::Closed; }
    void initialize(real_t pool, real_t temperature) override
    {
        core->initialize(pool, temperature);
        setState(core->state());
    }
    HeadspaceState trialState(real_t pool, real_t temperature, const GasMolesBySpecies& escaped) const override
    {
        ++probe->trial_calls; // Diagnostic instrumentation, not a physical source/history.
        auto trial = core->trialState(pool, temperature, escaped);
        double amount = 0;
        for (const auto& [name, value] : escaped)
            amount += value;
        if (amount > 0 && probe->rank == 0)
        {
            if (!finite_divergence)
                throw std::runtime_error("manufactured rank-local headspace EOS failure");
            trial.pressure *= 1.0001; // Finite; must reject before the next branch/callback.
        }
        return trial;
    }
    void commit(const HeadspaceState& state, const GasMolesBySpecies& escaped) override
    {
        ++probe->commit_calls;
        core->commit(state, escaped);
        setState(core->state());
    }
    const GasMolesBySpecies& headspaceMoles() const noexcept override { return core->headspaceMoles(); }
    const GasMolesBySpecies& ventedMoles() const noexcept override { return core->ventedMoles(); }
};
void expect_fault_headspace_rolls_back(bool finite_divergence)
{
    Fixture f;
    const auto comm = f.grid->owned_cell_map()->getComm();
    auto trace = std::make_shared<HeadspaceProbe>();
    trace->rank = comm->getRank();
    auto options = f.surface_options(true);
    auto closure = std::make_unique<FaultHeadspace>(options.headspace, f.chemistry.gas, trace, finite_divergence);
    const auto* identity = closure.get();
    f.coupling->configure_free_surface(options, std::move(closure));
    EXPECT_EQ(&f.coupling->find_free_surface_model()->headspace(), identity);
    const auto receipt = f.gas->species_event_receipt();
    const auto headspace = f.coupling->free_surface_diagnostics().headspace;
    const auto history = f.coupling->free_surface_history().size();
    const M::SavedReactingState<Model> saved(*f.gas);
    const int commits = trace->commit_calls, calls = trace->trial_calls;
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_EQ(trace->commit_calls, commits);
    EXPECT_GT(trace->trial_calls, calls);
    EXPECT_EQ(f.gas->species_event_receipt(), receipt);
    saved.expect_restored(*f.gas);
    EXPECT_DOUBLE_EQ(f.coupling->free_surface_diagnostics().headspace.pressure, headspace.pressure);
    EXPECT_EQ(f.coupling->free_surface_history().size(), history);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &trace->trial_calls, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &trace->trial_calls, &maximum);
    EXPECT_EQ(minimum, maximum);
}
TEST(BoussinesqSpeciesFreeSurfaceTest, ThrowingHeadspaceRejectsCollectivelyBeforeAnyCommit)
{
    expect_fault_headspace_rolls_back(false);
}
TEST(BoussinesqSpeciesFreeSurfaceTest, FiniteDivergentHeadspaceRejectsCollectivelyBeforeAnyCommit)
{
    if (Tpetra::getDefaultComm()->getSize() == 1)
        GTEST_SKIP() << "Finite rank divergence requires the proposed registered 2/4-rank cases; serial is not a pass.";
    expect_fault_headspace_rolls_back(true);
}

TEST(BoussinesqSpeciesFreeSurfaceTest, HistoryCSVUsesCatalogAndPopulationColumnsWithoutReservedGasSlots)
{
    Fixture f(false);
    f.configure_surface(f.surface_options());
    f.coupling->step();
    const auto filename = std::filesystem::temp_directory_path() /
                          ("simplefluid-stage5-species-history-" +
                              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    f.coupling->write_free_surface_history_csv(filename.string());
    if (f.grid->owned_cell_map()->getComm()->getRank() == 0)
    {
        std::ifstream input(filename);
        const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        EXPECT_NE(contents.find("schema_fingerprint"), std::string::npos);
        EXPECT_NE(contents.find("species_name"), std::string::npos);
        EXPECT_NE(contents.find("gas_subset_consumed_mol"), std::string::npos);
        EXPECT_NE(contents.find("population_moles:micro"), std::string::npos);
        EXPECT_NE(contents.find("population_moles:large"), std::string::npos);
        EXPECT_NE(contents.find("gas_one"), std::string::npos);
        EXPECT_NE(contents.find("gas_two"), std::string::npos);
        EXPECT_EQ(contents.find("h2_"), std::string::npos);
        EXPECT_EQ(contents.find("hydrogen"), std::string::npos);
        EXPECT_EQ(std::count(contents.begin(), contents.end(), '\n'),
            1 + f.chemistry.catalog->species().size() * f.coupling->free_surface_history().size());
        std::filesystem::remove(filename);
    }
}

} // namespace
