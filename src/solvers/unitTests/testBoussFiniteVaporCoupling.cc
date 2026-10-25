/** @file testBoussFiniteVaporCoupling.cc
 * @brief Positive fixed-grid finite-solvent/cp*T Bouss adapter coverage.
 * Designed for serial and 2/4 ranks; two cells give actual empty owners on four ranks.
 * Source-only draft: no qualification is claimed until registered runs complete.
 */
#include "ReactingMixtureTestSupport.hh"
#include "equations/BoussFiniteVapor.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace B = BubbleGasPhysics;
using P = R::SpeciesPhase;
using Pack = DefaultTpetraTypes;
using Grid = MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
class Solver : public BoussinesqSolver<Pack>
{
public:
    using BoussinesqSolver<Pack>::BoussinesqSolver;
    auto& temperature_owner()
    {
        return this->FluidSolver<Pack>::d_problem.template object<TemperatureDiffusionEquation<Pack, Grid>>(
            "temperature_equation");
    }
};
using Coupler = MultiphysicsSolverCoupler<Pack>;
using Model = Coupler::reacting_gas_model_type;
constexpr R::SpeciesId solvent = 29, donor = 73, gasOne = 5, gasTwo = 101;
constexpr R::ReactionId gasReaction = 47;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

BoundaryConditionSet closed(const Grid& mesh)
{
    BoundaryConditionSet result;
    for (const auto& [id, batch] : mesh.boundary_batches())
    {
        const auto name = mesh.boundary_batch_name(id);
        result.velocity[name] = {BoundaryConditionType::NoSlip, {0, 0, 0}};
        result.pressure[name] = {BoundaryConditionType::Neumann, 0};
        result.temperature[name] = {BoundaryConditionType::Neumann, 0};
    }
    return result;
}
double integral(const Model::field_type& field)
{
    double local = 0, global = 0;
    for (std::size_t c = 0; c < field.mesh().num_owned_cells(); ++c)
        local += field.value(static_cast<LO>(c)) * field.mesh().cell_volume(static_cast<LO>(c));
    Teuchos::reduceAll(*field.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
struct Fixture
{
    std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{gasTwo, "ordinary_second_gas", 1, {{"X", 1}}, {P::Resolved, P::Dispersed}},
            {solvent, "finite_solvent", 1, {{"S", 1}}, {P::Liquid, P::Resolved, P::Dispersed}},
            {donor, "finite_gas_donor", 2, {{"X", 2}}, {P::Liquid}},
            {gasOne, "ordinary_first_gas", 1, {{"X", 1}}, {P::Resolved, P::Dispersed}}},
        19);
    std::shared_ptr<const R::ReactionMechanism> mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{{gasReaction, "two_gases",
                     {{donor, P::Liquid, -1}, {gasOne, P::Resolved, 1}, {gasTwo, P::Resolved, 1}}, 0}});
    SP<Grid> grid;
    std::shared_ptr<Solver> solver;
    std::unique_ptr<Coupler> coupling;
    Model* model = nullptr;
    ScalarVoidFractionModel<Pack, Grid>* scalar = nullptr;
    std::shared_ptr<const R::BoussReferenceVolume> reference = std::make_shared<const R::BoussReferenceVolume>(
        R::BoussReferenceVolume{"manufactured.fixed_rho_cp", 1, catalog->fingerprint(), 2, 2, 5, {.2, 1, 0, .1}});
    Fixture(double temperature = 383, double alpha = .19, double maximum = .2, double dt = 1, double solventDensity = 2,
        double initialVapor = 0, bool mixedMicro = false, double wallFlux = 0, double conductivity = 0,
        BoundaryConditionType wallType = BoundaryConditionType::Neumann, double sourceHeating = 0)
    {
        Grid::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        grid = std::make_shared<Grid>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                          Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        TimeStepperOptions time;
        time.time_step = dt;
        time.kinematic_viscosity = 0;
        time.thermal_diffusivity = 0;
        time.thermal_expansion = 0;
        time.gravity_x = time.gravity_y = time.gravity_z = 0;
        BoussinesqModelOptions material;
        material.reference_density = material.density = 2;
        material.specific_heat_capacity = 5;
        material.dynamic_viscosity = 0;
        material.thermal_conductivity = conductivity;
        LinearSolverOptions linear;
        linear.preconditioner = LinearPreconditioner::None;
        linear.tolerance = 1e-12;
        auto boundaries = closed(*grid);
        if (wallFlux != 0)
        {
            if (conductivity <= 0)
                throw std::invalid_argument("Manufactured nonzero wall heat needs k>0.");
            boundaries.temperature["zmax"] = {
                wallType, wallType == BoundaryConditionType::Neumann ? wallFlux / conductivity : wallFlux};
        }
        solver = std::make_shared<Solver>(grid, boundaries, time, linear, material);
        coupling = std::make_unique<Coupler>(solver);
        ScalarVoidFractionOptions voidOptions;
        voidOptions.initial_alpha = alpha;
        voidOptions.alpha_max = maximum;
        scalar = &coupling->configure_scalar_void_fraction(voidOptions);
        auto initialSources = std::make_shared<R::BoussCoordinatedSources>();
        initialSources->identity = "manufactured.coordinated_sources";
        initialSources->revision = 1;
        initialSources->schemaFingerprint = catalog->fingerprint();
        ReactingGasOptions options;
        options.mode = ReactingGasMode::Species;
        options.alpha_max = maximum;
        options.reaction_mechanism = mechanism;
        options.bouss_reference = reference;
        options.bouss_sources = initialSources;
        options.reference_pressure = 100000;
        if (sourceHeating != 0)
            options.species_source = [sourceHeating](const R::ReactionContext& context, double dt)
            {
                R::ReactionSourceInput input;
                input.heat.integratedEnergy = sourceHeating * context.environment.bulkVolume * dt;
                return input;
            };
        options.population_ids = {"micro", "large", "vapor_bin"};
        options.min_radius = 1e-10;
        options.max_radius = 10;
        options.initial_species_state = [schema = catalog, alpha, solventDensity, initialVapor, mixedMicro](
                                            const R::ReactionEnvironment& env, std::int64_t)
        {
            R::CellSpeciesState state(*schema);
            for (const auto* population : {"micro", "large", "vapor_bin"})
                state.dispersedPopulations.push_back({population, std::vector<double>(schema->species().size())});
            state.set_amount(*schema, solvent, P::Liquid, solventDensity * env.bulkVolume);
            state.set_amount(*schema, donor, P::Liquid, env.bulkVolume);
            state.set_amount(*schema, gasOne, P::Resolved, alpha * env.bulkVolume / .1);
            state.set_amount(*schema, solvent, P::Resolved, initialVapor * env.bulkVolume);
            if (mixedMicro)
            {
                state.set_amount(*schema, gasOne, P::Dispersed, .01 * env.bulkVolume, "micro");
                state.set_amount(*schema, gasTwo, P::Dispersed, .02 * env.bulkVolume, "micro");
            }
            return state;
        };
        options.initial_population_numbers = [mixedMicro](const R::ReactionContext& context)
        { return std::vector<double>{mixedMicro ? 10 * context.environment.bulkVolume : 0, 0, 0}; };
        model = &coupling->configure_reacting_gas(options);
        coupling->initialize_heated_box(temperature, temperature);
    }
    std::shared_ptr<R::BoussCoordinatedSources> laws() const
    {
        auto result = std::make_shared<R::BoussCoordinatedSources>(*model->options().bouss_sources);
        ++result->revision;
        return result;
    }
    void vapor(double latent = 10, bool population = false)
    {
        auto sources = laws();
        sources->vapor = R::BoussVaporBinding{
            solvent, population ? P::Dispersed : P::Resolved, population ? "vapor_bin" : "", 373, latent, 1, 0};
        if (population)
            sources->populationEvents = [schema = catalog](const R::BoussAcceptedMaterialContext& context)
            {
                const double stock = context.acceptedMaterial.amount(*schema, solvent, P::Dispersed, "vapor_bin");
                return std::vector<B::PopulationNumberEvent>{{"vapor_bin", context.vapor.evaporation * 1000,
                    stock == 0 ? context.transportedNumbers[2].number : 0}};
            };
        coupling->configure_bouss_sources(sources);
    }
    void gas(double rate = 1)
    {
        auto sources = laws();
        sources->gas = R::BoussGasAdmissionLaw{"manufactured.molar_gas_request", 1, catalog->fingerprint(),
            [rate](const R::ReactionContext& context, double dt)
            {
                return std::vector<R::IntegratedReactionExtent>{
                    {gasReaction, rate * dt * context.environment.bulkVolume}};
            }};
        coupling->configure_bouss_sources(sources);
    }
    void wall(double flux, double fraction, bool bulk = true)
    {
        auto sources = laws();
        if (!sources->vapor)
            sources->vapor = R::BoussVaporBinding{solvent, P::Resolved, {}, 373, 10, 1, 0};
        sources->vapor->bulkEnabled = bulk;
        sources->vapor->walls = {{"zmax", flux, fraction}};
        coupling->configure_bouss_sources(sources);
    }
    void heat(double value)
    {
        PrescribedHeatSourceOptions heat;
        heat.profile = PrescribedHeatProfile::Constant;
        heat.power_density = value;
        coupling->configure_prescribed_heat_source(heat);
    }
};

TEST(BoussFiniteVaporCouplingTest, AdmittedLatentPreservesOriginalFixedCapacityScalarLimitOracle)
{
    Fixture f;
    f.vapor();
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Resolved)), .01, 1e-13);
    EXPECT_NEAR(integral(f.model->thermal_source()), -.1, 1e-13);
    EXPECT_NEAR(integral(f.scalar->alpha_g()), .2, 1e-13);
    EXPECT_NEAR(integral(f.scalar->source_alpha_total()), .01, 1e-13);
    EXPECT_NEAR(integral(*f.model->output_fields().at("bouss_evaporation_moles_rate")), .01, 1e-13);
    EXPECT_NEAR(integral(*f.model->output_fields().at("bouss_latent_power")), -.1, 1e-13);
    EXPECT_NEAR(integral(*f.model->output_fields().at("bouss_vapor_volume_source")), .01, 1e-13);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        EXPECT_NEAR(f.solver->temperature().value(cell), 382.99, 1e-10);
        const auto& receipt = f.model->bouss_receipts()[c];
        EXPECT_NEAR(-receipt.vapor.latentEnergy, receipt.vapor.evaporation * 10, 1e-13);
        EXPECT_DOUBLE_EQ(receipt.vapor.sensibleEnergy, 0);
    }
}
TEST(BoussFiniteVaporCouplingTest, BulkVaporCannotSpendMoreThanStepSensibleSuperheat)
{
    Fixture f(383, 0, .95, 2);
    f.vapor(1000);
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Resolved)), .1, 1e-13);
    EXPECT_NEAR(integral(f.model->thermal_source()), -50, 1e-10);
    EXPECT_NEAR(integral(f.scalar->source_alpha_total()), .05, 1e-13);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_NEAR(f.solver->temperature().value(static_cast<LO>(c)), 373, 1e-10);
}
TEST(BoussFiniteVaporCouplingTest, LateGasLawConfigurationPreservesEvolvedVaporVoidAndDefaultCapacity)
{
    Fixture f(383, 0, .95, .1);
    f.vapor(1000);
    f.coupling->step();
    const auto vapor = integral(f.model->species_field(solvent, P::Resolved));
    const auto voidFraction = integral(f.scalar->alpha_g());
    const auto receipt = f.model->species_event_receipt();
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    f.gas();
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.scalar->options().alpha_max, .95);
    EXPECT_DOUBLE_EQ(integral(f.scalar->alpha_g()), voidFraction);
    EXPECT_DOUBLE_EQ(integral(f.model->species_field(solvent, P::Resolved)), vapor);
    EXPECT_EQ(f.model->species_event_receipt(), receipt);
    f.coupling->step();
    EXPECT_GT(integral(f.model->species_field(gasTwo, P::Resolved)), 0);
    EXPECT_GT(integral(f.model->species_field(solvent, P::Resolved)), vapor);
}
TEST(BoussFiniteVaporCouplingTest, OrdinaryMolarGasRequestUsesAuthoritativeScalarLimitWithoutHeatYield)
{
    Fixture f(300, .19, .2, .1);
    f.gas(10);
    f.heat(100); // Independent heating does not determine the actual molar request.
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->source_alpha_reaction()), .1, 1e-13);
    EXPECT_NEAR(integral(f.scalar->source_alpha_total()), .1, 1e-13);
    EXPECT_NEAR(integral(f.scalar->alpha_g()), .2, 1e-13);
    EXPECT_NEAR(integral(f.scalar->alpha_l()), .8, 1e-13);
    const auto secondGas = integral(f.model->species_field(gasTwo, P::Resolved));
    EXPECT_NEAR(secondGas, .01 / .3, 1e-13);
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->source_alpha_reaction()), 0, 1e-13);
    EXPECT_NEAR(integral(f.scalar->source_alpha_total()), 0, 1e-13);
    EXPECT_NEAR(integral(f.model->species_field(gasTwo, P::Resolved)), secondGas, 1e-13);
}
TEST(BoussFiniteVaporCouplingTest, GasAndVaporSetupOrderIsEquivalentAndSharesCapacity)
{
    Fixture gasFirst(383, .19, .2, .1), vaporFirst(383, .19, .2, .1);
    gasFirst.gas(.1);
    gasFirst.vapor(10);
    vaporFirst.vapor(10);
    vaporFirst.gas(.1);
    gasFirst.coupling->step();
    vaporFirst.coupling->step();
    EXPECT_NEAR(integral(gasFirst.scalar->alpha_g()), .2, 1e-13);
    EXPECT_NEAR(integral(gasFirst.model->species_field(gasTwo, P::Resolved)), .01, 1e-13);
    EXPECT_NEAR(integral(gasFirst.model->species_field(solvent, P::Resolved)), .007, 1e-13);
    for (const auto& species : gasFirst.catalog->species())
        for (const auto phase : {P::Liquid, P::Resolved})
            EXPECT_DOUBLE_EQ(integral(gasFirst.model->species_field(species.id, phase)),
                integral(vaporFirst.model->species_field(species.id, phase)));
    for (std::size_t c = 0; c < gasFirst.grid->num_owned_cells(); ++c)
        EXPECT_DOUBLE_EQ(gasFirst.solver->temperature().value(static_cast<LO>(c)),
            vaporFirst.solver->temperature().value(static_cast<LO>(c)));
}
TEST(BoussFiniteVaporCouplingTest, NativeHeatGasVaporAndMechanicalFeedbackRemainPositiveWithOneThermalOwner)
{
    Fixture f(383, .1, .9, .1);
    f.gas(.001);
    f.vapor(1000);
    f.heat(100);
    MaterialFeedbackOptions feedback;
    feedback.density_mode = DensityFeedbackMode::Mixture;
    feedback.reference_density = feedback.liquid_density = 2;
    feedback.gas_density = 1;
    feedback.reference_dynamic_viscosity = feedback.min_viscosity = 0;
    auto& material = f.coupling->configure_material_feedback(feedback);
    f.coupling->step();
    EXPECT_FALSE(f.grid->legacy_mesh());
    EXPECT_EQ(f.solver->temperature().mesh_ptr(), f.grid);
    EXPECT_EQ(f.solver->material_properties().density.mesh_ptr(), f.grid);
    EXPECT_EQ(f.scalar->alpha_g().mesh_ptr(), f.grid);
    EXPECT_EQ(f.model->thermal_source().mesh_ptr(), f.grid);
    EXPECT_EQ(material.density_feedback().mesh_ptr(), f.grid);
    EXPECT_GT(integral(f.model->species_field(gasTwo, P::Resolved)), 0);
    EXPECT_GT(integral(f.model->species_field(solvent, P::Resolved)), 0);
    EXPECT_NEAR(integral(f.model->thermal_source()), -100, 1e-10);
    // 100 W/m3 independent heat and admitted 100 W/m3 latent sink cancel at C=2*5.
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_NEAR(f.solver->temperature().value(static_cast<LO>(c)), 383, 1e-10);
}
TEST(BoussFiniteVaporCouplingTest, NamedVaporHasExplicitCountsAndFiniteSolventExhaustion)
{
    Fixture f(383, 0, .9, 1, .003);
    f.vapor(10, true);
    f.coupling->step();
    EXPECT_DOUBLE_EQ(integral(f.model->species_field(solvent, P::Liquid)), 0);
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Dispersed, "vapor_bin")), .003, 1e-15);
    EXPECT_NEAR(integral(f.model->population_number_field("vapor_bin")), 3, 1e-13);
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->population_number_field("vapor_bin")), 3, 1e-13);
    EXPECT_DOUBLE_EQ(integral(f.model->thermal_source()), 0);
}
TEST(BoussFiniteVaporCouplingTest, FailedAdmittedCountProviderRestoresEveryPublishedFieldAndCanReplay)
{
    Fixture trial(383, .1, .9, .1), control(383, .1, .9, .1);
    trial.vapor(1000, true);
    control.vapor(1000, true);
    trial.coupling->step();
    control.coupling->step();
    auto fail = std::make_shared<bool>(true);
    auto sources = trial.laws();
    const auto valid = sources->populationEvents;
    sources->populationEvents = [fail, valid, rank = trial.grid->owned_cell_map()->getComm()->getRank()](
                                    const R::BoussAcceptedMaterialContext& context)
    {
        if (*fail && rank == 0)
            throw std::runtime_error("injected physical number-law failure");
        return valid(context);
    };
    trial.coupling->configure_bouss_sources(sources);
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*trial.model);
    const auto receipt = trial.model->species_event_receipt();
    const auto oldVoid = integral(trial.scalar->alpha_g());
    const auto oldHeat = integral(trial.model->thermal_source());
    const auto oldReceipts = trial.model->bouss_receipts();
    std::vector<std::array<double, 8>> oldFlow;
    for (std::size_t c = 0; c < trial.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto velocity = trial.solver->velocity().owned_read_view();
        oldFlow.push_back({trial.solver->temperature().value(cell), trial.solver->pressure().value(cell),
            trial.solver->material_properties().density.value(cell),
            trial.solver->material_properties().specific_heat_capacity.value(cell), velocity(c, 0), velocity(c, 1),
            velocity(c, 2), trial.scalar->source_alpha_total().value(cell)});
    }
    std::vector<double> oldFlux;
    for (const auto face : trial.solver->pressure_corrected_face_fluxes().owned_face_ids())
        oldFlux.push_back(trial.solver->pressure_corrected_face_fluxes().value(face));
    const auto oldTime = trial.solver->time();
    EXPECT_ANY_THROW(trial.coupling->step());
    saved.expect_restored(*trial.model);
    EXPECT_EQ(trial.model->species_event_receipt(), receipt);
    EXPECT_DOUBLE_EQ(integral(trial.scalar->alpha_g()), oldVoid);
    EXPECT_DOUBLE_EQ(integral(trial.model->thermal_source()), oldHeat);
    EXPECT_DOUBLE_EQ(trial.solver->time(), oldTime);
    EXPECT_EQ(trial.model->bouss_receipts().size(), oldReceipts.size());
    for (std::size_t c = 0; c < trial.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto velocity = trial.solver->velocity().owned_read_view();
        EXPECT_DOUBLE_EQ(trial.solver->temperature().value(cell), oldFlow[c][0]);
        EXPECT_DOUBLE_EQ(trial.solver->pressure().value(cell), oldFlow[c][1]);
        EXPECT_DOUBLE_EQ(trial.solver->material_properties().density.value(cell), oldFlow[c][2]);
        EXPECT_DOUBLE_EQ(trial.solver->material_properties().specific_heat_capacity.value(cell), oldFlow[c][3]);
        for (std::size_t axis = 0; axis < 3; ++axis)
            EXPECT_DOUBLE_EQ(velocity(c, axis), oldFlow[c][4 + axis]);
        EXPECT_DOUBLE_EQ(trial.scalar->source_alpha_total().value(cell), oldFlow[c][7]);
        const auto& restored = trial.model->bouss_receipts()[c];
        EXPECT_EQ(restored.configurationSignature, oldReceipts[c].configurationSignature);
        EXPECT_EQ(restored.vapor.latentEnergy, oldReceipts[c].vapor.latentEnergy);
        EXPECT_EQ(restored.vapor.sensibleEnergy, oldReceipts[c].vapor.sensibleEnergy);
        EXPECT_DOUBLE_EQ(restored.vapor.evaporation, oldReceipts[c].vapor.evaporation);
        EXPECT_DOUBLE_EQ(restored.vapor.condensation, oldReceipts[c].vapor.condensation);
    }
    std::size_t faceIndex = 0;
    for (const auto face : trial.solver->pressure_corrected_face_fluxes().owned_face_ids())
        EXPECT_DOUBLE_EQ(trial.solver->pressure_corrected_face_fluxes().value(face), oldFlux[faceIndex++]);
    *fail = false;
    trial.coupling->step();
    control.coupling->step();
    for (const auto& species : trial.catalog->species())
        for (const auto phase : {P::Liquid, P::Resolved})
            EXPECT_DOUBLE_EQ(integral(trial.model->species_field(species.id, phase)),
                integral(control.model->species_field(species.id, phase)));
    EXPECT_DOUBLE_EQ(integral(trial.model->species_field(solvent, P::Dispersed, "vapor_bin")),
        integral(control.model->species_field(solvent, P::Dispersed, "vapor_bin")));
    EXPECT_DOUBLE_EQ(integral(trial.model->population_number_field("vapor_bin")),
        integral(control.model->population_number_field("vapor_bin")));
    EXPECT_EQ(trial.model->species_event_receipt().revision, 2u);
    int localEmpty = trial.grid->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*trial.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &localEmpty, &empty);
    if (trial.grid->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(BoussFiniteVaporCouplingTest, FiniteCondensationReturnsActualSolventAndOnlyAdmittedLatent)
{
    Fixture f(363, .19, .2, 1, 2, .005);
    f.vapor();
    f.coupling->step();
    EXPECT_DOUBLE_EQ(integral(f.model->species_field(solvent, P::Resolved)), 0);
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Liquid)), 2.005, 1e-13);
    EXPECT_NEAR(integral(f.model->thermal_source()), .05, 1e-13);
    EXPECT_NEAR(integral(f.scalar->alpha_g()), .19, 1e-13);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_NEAR(f.solver->temperature().value(static_cast<LO>(c)), 363.005, 1e-10);
}
TEST(BoussFiniteVaporCouplingTest, AcceptedSourceHeatingBoundsCondensationWithoutLosingItsRequest)
{
    for (const double heating : {80.0, 100.0, 120.0})
    {
        SCOPED_TRACE(heating);
        Fixture f(363, 0, .9, 1, 2, .5, false, 0, 0, BoundaryConditionType::Neumann, heating);
        f.vapor(100);
        f.coupling->step();
        const double condensed = heating < 100 ? .2 : 0;
        EXPECT_NEAR(integral(f.model->species_field(solvent, P::Resolved)), .5 - condensed, 1e-13);
        EXPECT_NEAR(integral(f.model->species_field(solvent, P::Liquid)), 2 + condensed, 1e-13);
        EXPECT_NEAR(integral(f.model->thermal_source()), heating + 100 * condensed, 1e-11);
        EXPECT_NEAR(integral(f.scalar->alpha_g()), .5 - condensed, 1e-13);
        for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const auto volume = f.grid->cell_volume(cell);
            const auto& receipt = f.model->bouss_receipts()[c];
            EXPECT_NEAR(receipt.vapor.requestedCondensation, volume, 1e-13);
            EXPECT_NEAR(receipt.vapor.condensation, condensed * volume, 1e-13);
            EXPECT_NEAR(receipt.vapor.latentEnergy, 100 * condensed * volume, 1e-11);
            EXPECT_NEAR(f.solver->temperature().value(cell), heating <= 100 ? 373 : 375, 1e-10);
        }
    }
}
TEST(BoussFiniteVaporCouplingTest, RejectsVariableCpAndIndependentLegacyBoilingBeforeAdvancement)
{
    Fixture f;
    f.vapor();
    BoilingSourceOptions legacy;
    legacy.enable_bulk_boiling = true;
    legacy.saturation_temperature = 373;
    legacy.latent_heat = 10;
    legacy.gas_density = 1;
    EXPECT_ANY_THROW(f.coupling->configure_boiling_source(legacy));
    auto called = std::make_shared<bool>(false);
    f.coupling->set_material_updater(
        [called](const auto&, auto& properties)
        {
            *called = true;
            properties.specific_heat_capacity.put_scalar(6);
        });
    const auto revision = f.model->species_event_receipt();
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_FALSE(*called);
    EXPECT_EQ(f.model->species_event_receipt(), revision);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
}
TEST(BoussFiniteVaporCouplingTest, ExplicitMixedMicroToLargePacketPreservesBothOrdinaryGasesAndCounts)
{
    Fixture f(300, .1, .9, .1, 2, 0, true);
    auto sources = f.laws();
    sources->populationProcess = [](const R::BoussPopulationContext& context, double)
    {
        B::MultispeciesPopulationPacket packet;
        for (const auto species : {gasOne, gasTwo})
            packet.transfers.push_back({species, P::Dispersed, P::Dispersed, "micro", "large",
                context.chemistry.state.amount(context.chemistry.catalog, species, P::Dispersed, "micro")});
        packet.numberEvents = {
            {"micro", 0, context.populations[0].number}, {"large", context.populations[0].number, 0}};
        return packet;
    };
    f.coupling->configure_bouss_sources(sources);
    f.coupling->step();
    EXPECT_DOUBLE_EQ(integral(f.model->population_number_field("micro")), 0);
    EXPECT_NEAR(integral(f.model->population_number_field("large")), 10, 1e-13);
    EXPECT_NEAR(integral(f.model->species_field(gasOne, P::Dispersed, "large")), .01, 1e-15);
    EXPECT_NEAR(integral(f.model->species_field(gasTwo, P::Dispersed, "large")), .02, 1e-15);
    EXPECT_DOUBLE_EQ(integral(f.model->species_field(gasOne, P::Dispersed, "micro")), 0);
    EXPECT_DOUBLE_EQ(integral(f.model->species_field(gasTwo, P::Dispersed, "micro")), 0);
    EXPECT_DOUBLE_EQ(integral(f.model->thermal_source()), 0);
}
TEST(BoussFiniteVaporCouplingTest, ActualWallBoundaryHeatEntersOnceAndOnlyAdmittedLatentLeavesScalarStorage)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 40, 1);
    f.wall(40, .25, false);
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Resolved)), .1, 1e-13);
    EXPECT_NEAR(integral(f.model->thermal_source()), -10, 1e-12);
    EXPECT_NEAR(10 * (integral(f.solver->temperature()) - 373), 3, 1e-10);
    EXPECT_NEAR(f.solver->temperature_owner().last_prescribed_boundary_power(), -40, 1e-13);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
    for (const auto& cell : f.model->bouss_receipts())
        for (const auto& origin : cell.vapor.evaporationAdmissions)
            if (origin.wall)
                EXPECT_EQ(origin.patch, "zmax");
}
TEST(BoussFiniteVaporCouplingTest, WallAndBulkCompeteAtTheSameFiniteVoidCapacity)
{
    Fixture f(383, .19, .2, 1, 2, 0, false, 100, 1);
    f.wall(100, 1);
    f.coupling->step();
    EXPECT_NEAR(integral(f.model->species_field(solvent, P::Resolved)), .01, 1e-13);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
    {
        if (f.grid->cell_geometry_global_id(static_cast<LO>(c)) != 1)
            continue;
        const auto& origins = f.model->bouss_receipts()[c].vapor.evaporationAdmissions;
        ASSERT_EQ(origins.size(), 2);
        // Top bulk sensible budget is 50J ->5mol; wall100J ->10mol.
        EXPECT_NEAR(origins[0].admittedMoles, .005 / 3, 1e-13);
        EXPECT_NEAR(origins[1].admittedMoles, .010 / 3, 1e-13);
    }
}
TEST(BoussFiniteVaporCouplingTest, MismatchedWallHeatFailsWithoutChangingStocksFluxOwnerOrClock)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 40, 1);
    f.wall(41, .25, false);
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    EXPECT_ANY_THROW(f.coupling->step());
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
}
TEST(BoussFiniteVaporCouplingTest, ChangedConductivityAfterAdmissionRejectsAndRestoresExactWallOwner)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 40, 1);
    f.wall(40, .25, false);
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    auto fail = std::make_shared<bool>(true);
    auto sources = f.laws();
    sources->populationEvents = [solver = f.solver, grid = f.grid, fail](const R::BoussAcceptedMaterialContext&)
    {
        // Intentional MPI-free invalid provider mutation; immutable-owner check must reject.
        if (*fail)
            for (std::size_t c = 0; c < grid->num_owned_cells(); ++c)
                if (grid->cell_geometry_global_id(static_cast<LO>(c)) == 1)
                    solver->material_properties().thermal_conductivity.set_owned_value(static_cast<LO>(c), 2);
        return std::vector<B::PopulationNumberEvent>{};
    };
    f.coupling->configure_bouss_sources(sources);
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    EXPECT_ANY_THROW(f.coupling->step());
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_DOUBLE_EQ(f.solver->material_properties().thermal_conductivity.value(static_cast<LO>(c)), 1);
    *fail = false;
    f.coupling->step();
    EXPECT_NEAR(10 * (integral(f.solver->temperature()) - 373), 3, 1e-10);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
    int localEmpty = f.grid->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &localEmpty, &empty);
    if (f.grid->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}

TEST(BoussFiniteVaporCouplingTest, BoundaryConductivityCacheCannotMaskChangedOwnedConductivity)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 40, 1);
    f.wall(40, .25, false);
    // The resolved wall owns a nonnull, molecular-only boundary cache while the
    // interior effective conductivity is a separate field with its own lifetime.
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        f.solver->material_properties().dynamic_viscosity.set_owned_value(static_cast<LO>(c), .001);
    f.solver->material_properties().dynamic_viscosity.sync_ghosts();
    TurbulenceModelOptions options;
    options.model = TurbulenceModelType::StandardKEpsilon;
    options.wall_treatment = TurbulenceWallTreatmentType::ResolvedLowReKEpsilon;
    options.wall_options.boundary_names = {"zmax"};
    options.initial_turbulent_kinetic_energy = 1e-8;
    options.initial_dissipation_rate = 1e-6;
    auto& turbulence = f.solver->configure_turbulence(options);
    EXPECT_NE(turbulence.effective_thermal_conductivity_boundary_cache(), nullptr);
    std::vector<double> acceptedConductivity;
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        acceptedConductivity.push_back(turbulence.effective_thermal_conductivity().value(static_cast<LO>(c)));
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    auto fail = std::make_shared<bool>(true);
    auto reached = std::make_shared<bool>(false);
    auto sources = f.laws();
    sources->populationEvents = [solver = f.solver, grid = f.grid, fail, reached](
                                   const R::BoussAcceptedMaterialContext&)
    {
        auto* model = solver->find_turbulence_model();
        if (!model || !model->effective_thermal_conductivity_boundary_cache())
            throw std::logic_error("This regression requires an actual effective wall cache.");
        for (std::size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            if (grid->cell_geometry_global_id(cell) != 1)
                continue;
            *reached = true;
            if (*fail)
            {
                // Intentional invalid provider mutation. Keep the boundary cache
                // untouched so checking only its selected coefficient would miss this.
                auto& conductivity = const_cast<Model::field_type&>(model->effective_thermal_conductivity());
                conductivity.set_owned_value(cell, conductivity.value(cell) + 1);
            }
        }
        return std::vector<B::PopulationNumberEvent>{};
    };
    f.coupling->configure_bouss_sources(sources);
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    const auto acceptedTime = f.solver->time();
    EXPECT_ANY_THROW(f.coupling->step());
    int localReached = *reached, globalReached = 0;
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &localReached, &globalReached);
    EXPECT_EQ(globalReached, 1); // A setup/turbulence failure cannot substitute for the owner gate.
    for (std::size_t c = 0; c < acceptedConductivity.size(); ++c)
        EXPECT_DOUBLE_EQ(turbulence.effective_thermal_conductivity().value(static_cast<LO>(c)), acceptedConductivity[c]);
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), acceptedTime);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
    *fail = false;
    f.coupling->step();
    EXPECT_NEAR(10 * (integral(f.solver->temperature()) - 373), 3, 1e-10);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
}
TEST(BoussFiniteVaporCouplingTest, ExplicitHeatFluxSnapshotOwnsWallHeatAndIsRestoredByIdentity)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 0, 1);
    using Cache = TemperatureDiffusionEquation<Pack, Grid>::boundary_cache_type;
    auto supplied = std::make_shared<Cache>();
    supplied->mesh = f.grid;
    for (const auto& [id, batch] : f.grid->boundary_batches())
        if (f.grid->boundary_batch_name(id) == "zmax")
            supplied->value[id] = Arr<double>(batch.face_lids.size(), -40);
    f.solver->temperature_owner().set_boundary_heat_flux_snapshot(supplied);
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    f.wall(40, .25, false);
    f.coupling->step();
    EXPECT_NEAR(10 * (integral(f.solver->temperature()) - 373), 3, 1e-10);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
    double localWallEnergy = 0, wallEnergy = 0;
    for (const auto& receipt : f.model->bouss_receipts())
        for (const auto& origin : receipt.vapor.evaporationAdmissions)
            if (origin.wall)
                localWallEnergy += static_cast<double>(origin.suppliedWallEnergy);
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &localWallEnergy, &wallEnergy);
    EXPECT_NEAR(wallEnergy, 4, 1e-13);
}
TEST(BoussFiniteVaporCouplingTest, BoundSolventCannotBypassLatentOwnerThroughPopulationOnlyPacket)
{
    Fixture f(300, 0, .9, .1);
    f.vapor(1000);
    auto sources = f.laws();
    sources->populationProcess = [](const R::BoussPopulationContext& context, double)
    {
        B::MultispeciesPopulationPacket packet;
        packet.transfers = {{solvent, P::Liquid, P::Resolved, {}, {}, .001 * context.chemistry.environment.bulkVolume}};
        return packet;
    };
    f.coupling->configure_bouss_sources(sources);
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    EXPECT_ANY_THROW(f.coupling->step());
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
}
TEST(BoussFiniteVaporCouplingTest, UnknownDirichletWallHeatRejectsWithoutAnExplicitFluxOwner)
{
    Fixture f(373, 0, .9, .1, 2, 0, false, 383, 1, BoundaryConditionType::Dirichlet);
    f.wall(40, .25, false);
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    const auto original = f.solver->temperature_owner().boundary_heat_flux();
    EXPECT_ANY_THROW(f.coupling->step());
    saved.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    EXPECT_EQ(f.solver->temperature_owner().boundary_heat_flux().get(), original.get());
}
TEST(BoussFiniteVaporCouplingTest, FirstStepFailurePreservesInitializationAuthorityAndRetryDoesNotResetStocks)
{
    Fixture f;
    auto calls = std::make_shared<std::size_t>(0);
    auto options = f.model->options();
    const auto initial = options.initial_species_state;
    options.initial_species_state = [initial, calls](const R::ReactionEnvironment& e, std::int64_t cell)
    {
        ++*calls;
        return initial(e, cell);
    };
    f.model = &f.coupling->configure_reacting_gas(options);
    f.vapor();
    auto fail = std::make_shared<bool>(true);
    auto sources = f.laws();
    sources->populationEvents = [fail](const R::BoussAcceptedMaterialContext&)
    {
        if (*fail)
            throw std::runtime_error("Deliberate first-step source failure.");
        return std::vector<B::PopulationNumberEvent>{};
    };
    f.coupling->configure_bouss_sources(sources);
    const auto initializedCalls = *calls;
    EXPECT_EQ(initializedCalls, f.grid->num_owned_cells());
    const auto receipt = f.model->species_event_receipt();
    const Testing::ReactingMixture::SavedReactingState<Model> saved(*f.model);
    EXPECT_ANY_THROW(f.coupling->step());
    EXPECT_EQ(*calls, initializedCalls);
    saved.expect_restored(*f.model);
    EXPECT_EQ(f.model->species_event_receipt(), receipt);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    *fail = false;
    EXPECT_NO_THROW(f.coupling->step());
    EXPECT_EQ(*calls, initializedCalls);
    EXPECT_GT(integral(f.model->species_field(solvent, P::Resolved)), 0);
    EXPECT_EQ(f.model->species_event_receipt().initialization_generation, receipt.initialization_generation);
}

TEST(BoussFiniteVaporCouplingTest, LateLawReplacementPreservesAcceptedProvenanceButInvalidatesSnapshots)
{
    Fixture f(383, 0, .9, .1);
    f.vapor(1000);
    f.coupling->step();
    const auto snapshot = f.model->snapshot();
    const auto event = f.model->species_event_receipt();
    const auto history = f.model->species_histories();
    const Testing::ReactingMixture::SavedReactingState<Model> accepted(*f.model);
    std::vector<std::uint64_t> configurations;
    for (const auto& receipt : f.model->bouss_receipts())
        configurations.push_back(receipt.configurationSignature);
    f.gas(.01);
    EXPECT_EQ(f.model->species_event_receipt(), event);
    EXPECT_EQ(f.model->species_histories(), history);
    accepted.expect_restored(*f.model);
    for (std::size_t c = 0; c < configurations.size(); ++c)
        EXPECT_EQ(f.model->bouss_receipts()[c].configurationSignature, configurations[c]);
    EXPECT_ANY_THROW(f.model->restore(snapshot));
    accepted.expect_restored(*f.model);
    EXPECT_NO_THROW(f.coupling->step());
    EXPECT_EQ(f.model->species_event_receipt().initialization_generation, event.initialization_generation);
    EXPECT_EQ(f.model->species_event_receipt().revision, event.revision + 1);
    EXPECT_NEAR(f.solver->time(), .2, 1e-15);
    EXPECT_GT(integral(f.model->species_field(gasTwo, P::Resolved)), 0);
}
TEST(BoussFiniteVaporCouplingTest, RankAsymmetricValidBoundsRejectBeforeMutatingAnyAcceptedOwner)
{
    Fixture f(383, .19, .2, .1);
    const auto comm = f.grid->owned_cell_map()->getComm();
    const auto snapshot = f.model->snapshot();
    const auto* owner = f.model;
    const auto sources = f.model->options().bouss_sources;
    const Testing::ReactingMixture::SavedReactingState<Model> accepted(*f.model);
    auto options = f.model->options();
    if (comm->getRank() == 0)
        options.alpha_max = .3;
    EXPECT_NO_THROW(validate_reacting_gas_options(options));
    EXPECT_ANY_THROW((void) f.coupling->configure_reacting_gas(options));
    EXPECT_EQ(f.coupling->find_reacting_gas_model(), owner);
    EXPECT_EQ(f.model->options().bouss_sources.get(), sources.get());
    EXPECT_DOUBLE_EQ(f.model->options().alpha_max, .2);
    EXPECT_DOUBLE_EQ(f.scalar->options().alpha_max, .2);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        EXPECT_DOUBLE_EQ(f.solver->temperature().value(static_cast<LO>(c)), 383);
    accepted.expect_restored(*f.model);
    EXPECT_NO_THROW(f.model->restore(snapshot));
}
TEST(BoussFiniteVaporCouplingTest, RankAsymmetricValidSurfacePressureRejectsBeforeConstructionOrMutation)
{
    namespace M = Testing::ReactingMixture;
    const M::Mixture mixture;
    Grid::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    auto grid = std::make_shared<Grid>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                           Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
        distribution);
    TimeStepperOptions time;
    time.time_step = .01;
    time.kinematic_viscosity = time.thermal_diffusivity = time.thermal_expansion = 0;
    time.gravity_x = time.gravity_y = time.gravity_z = 0;
    BoussinesqModelOptions material;
    material.reference_density = material.density = 1000;
    material.specific_heat_capacity = 4000;
    material.dynamic_viscosity = .001;
    material.thermal_conductivity = 0;
    auto solver = std::make_shared<Solver>(grid, closed(*grid), time, LinearSolverOptions{}, material);
    Coupler coupling(solver);
    auto options = M::field_options(mixture);
    auto& model = coupling.configure_reacting_gas(options);
    // Actual model inventory is accepted, while flow primary initialization
    // remains incomplete so the installed surface is deliberately uninitialized.
    solver->temperature().put_scalar(300);
    model.initialize_state(
        0, solver->temperature(), solver->pressure(), solver->velocity(), solver->material_properties());
    FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = FreeSurfaceMode::PlanarVolumeBudget;
    surface.vessel.bottom_elevation = 0;
    surface.vessel.top_elevation = 2;
    surface.vessel.cross_section_area = 1;
    surface.vessel.total_internal_volume = 2;
    surface.liquid_mass.mode = LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = FreeSurfaceRangePolicy::Error;
    surface.liquid_mass.initial_liquid_mass = 1000;
    surface.headspace.mode = HeadspaceMode::Vented;
    surface.headspace.total_internal_volume = 2;
    surface.headspace.ambient_pressure = surface.headspace.initial_pressure = 100000;
    surface.headspace.initial_temperature = 300;
    surface.coupling.minimum_absolute_pressure = M::bounds.minimum_pressure;
    surface.coupling.maximum_absolute_pressure = M::bounds.maximum_pressure;
    auto* surfaceOwner = coupling.configure_free_surface(surface);
    EXPECT_FALSE(surfaceOwner->initialized());
    const auto snapshot = model.snapshot();
    const M::SavedReactingState<Model> accepted(model);
    const auto comm = grid->owned_cell_map()->getComm();
    if (comm->getRank() == 0)
    {
        options.pressure_mode = ReactingPressureMode::PrescribedHistory;
        options.pressure_history_times = {0, 1};
        options.pressure_history_values = {100000, 120000};
    }
    EXPECT_NO_THROW(validate_reacting_gas_options(options));
    EXPECT_ANY_THROW((void) coupling.configure_reacting_gas(options));
    EXPECT_EQ(coupling.find_reacting_gas_model(), &model);
    EXPECT_EQ(coupling.find_free_surface_model(), surfaceOwner);
    EXPECT_FALSE(surfaceOwner->initialized());
    EXPECT_EQ(model.options().pressure_mode, ReactingPressureMode::Constant);
    EXPECT_DOUBLE_EQ(solver->time(), 0);
    accepted.expect_restored(model);
    EXPECT_NO_THROW(model.restore(snapshot));
}
TEST(BoussFiniteVaporCouplingTest, RawBoussCountDeathCannotBorrowTheFollowingPacketBirth)
{
    Fixture f(300, 0, .9, .1, 2, 0, true);
    auto options = f.model->options();
    options.initial_population_numbers = [](const R::ReactionContext& context)
    { return std::vector<double>{1e20 * context.environment.bulkVolume, 0, 0}; };
    options.reaction_advance.maximumRetries = 2;
    auto sources = f.laws();
    sources->populationEvents = [](const R::BoussAcceptedMaterialContext&)
    { return std::vector<B::PopulationNumberEvent>{{"micro", 0, 1}}; };
    sources->populationProcess = [](const R::BoussPopulationContext& context, double)
    {
        const auto found =
            std::ranges::find(context.populations, std::string{"micro"}, &R::BoussPopulationCondition::population);
        B::MultispeciesPopulationPacket packet;
        packet.numberEvents = {{"micro", found->number, found->number}};
        return packet;
    };
    options.bouss_sources = sources;
    f.model = &f.coupling->configure_reacting_gas(options);
    const Testing::ReactingMixture::SavedReactingState<Model> accepted(*f.model);
    EXPECT_ANY_THROW(f.coupling->step());
    accepted.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
}
TEST(BoussFiniteVaporCouplingTest, RawCondensationDebitCannotBorrowFollowingGasTransferCredit)
{
    Fixture f(363, 0, .9, 1);
    auto options = f.model->options();
    auto reference = std::make_shared<R::BoussReferenceVolume>(*f.reference);
    ++reference->revision;
    reference->gasMolarVolumes[f.catalog->index(solvent)] = 1e-24; // Manufactured small occupancy of large stock.
    options.bouss_reference = reference;
    options.initial_species_state = [schema = f.catalog](const R::ReactionEnvironment& env, std::int64_t)
    {
        R::CellSpeciesState state(*schema);
        for (const auto* name : {"micro", "large", "vapor_bin"})
            state.dispersedPopulations.push_back({name, std::vector<double>(schema->species().size())});
        state.set_amount(*schema, solvent, P::Liquid, 2 * env.bulkVolume);
        state.set_amount(*schema, donor, P::Liquid, env.bulkVolume);
        state.set_amount(*schema, solvent, P::Resolved, 1e20 * env.bulkVolume);
        state.set_amount(*schema, solvent, P::Dispersed, 1e20 * env.bulkVolume, "micro");
        return state;
    };
    options.initial_population_numbers = [](const R::ReactionContext& context)
    { return std::vector<double>{1e10 * context.environment.bulkVolume, 0, 0}; };
    options.reaction_advance.maximumRetries = 2;
    auto sources = f.laws();
    sources->vapor = R::BoussVaporBinding{solvent, P::Resolved, {}, 373, 1, 50, 0};
    sources->populationProcess = [](const R::BoussPopulationContext& context, double)
    {
        const auto& chemistry = context.chemistry;
        const double stock = chemistry.state.amount(chemistry.catalog, solvent, P::Resolved);
        const double incoming =
            std::min(stock, chemistry.state.amount(chemistry.catalog, solvent, P::Dispersed, "micro"));
        B::MultispeciesPopulationPacket packet;
        packet.transfers = {{solvent, P::Resolved, P::Dispersed, {}, "micro", stock},
            {solvent, P::Dispersed, P::Resolved, "micro", {}, incoming}};
        return packet;
    };
    options.bouss_sources = sources;
    f.model = &f.coupling->configure_reacting_gas(options);
    const Testing::ReactingMixture::SavedReactingState<Model> accepted(*f.model);
    // Each .5m3 cell condenses exactly1mol: C=5, deltaT10, dt1, tau50,L1.
    // Rounded gas stock is unchanged, but outgoing old stock is short by1mol;
    // the simultaneous incoming bin transfer cannot finance it.
    EXPECT_ANY_THROW(f.coupling->step());
    accepted.expect_restored(*f.model);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
}
} // namespace
