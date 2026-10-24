/** @file
 * @brief Explicit schema configuration and finite population ownership regressions.
 */
#include "ReactingMixtureTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Grid = MeshHandle<Pack>;
namespace M = Testing::ReactingMixture;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Fixture
{
    SP<Grid> grid;
    std::shared_ptr<BoussinesqSolver<Pack>> solver;
    std::unique_ptr<MultiphysicsSolverCoupler<Pack>> coupling;
    Fixture()
    {
        Grid::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        grid = std::make_shared<Grid>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                          Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        TimeStepperOptions time;
        time.time_step = .01;
        time.kinematic_viscosity = time.thermal_diffusivity = time.thermal_expansion = 0;
        time.gravity_x = time.gravity_y = time.gravity_z = 0;
        BoussinesqModelOptions material;
        material.reference_density = material.density = 1000;
        material.specific_heat_capacity = 4000;
        material.dynamic_viscosity = material.thermal_conductivity = 0;
        solver = std::make_shared<BoussinesqSolver<Pack>>(
            grid, BoundaryConditionSet{}, time, LinearSolverOptions{}, material);
        coupling = std::make_unique<MultiphysicsSolverCoupler<Pack>>(solver);
    }
};
TEST(ReactingLegacyCleanupTest, RankLocalRetiredKeyFailsBeforeConstructionAndExplicitSchemaRetryWorks)
{
    Fixture f;
    const auto comm = f.grid->owned_cell_map()->getComm();
    Database database;
    if (comm->getRank() == 0)
        database.set("hydrogen_yield_mol_per_j", real_t{0});
    EXPECT_ANY_THROW((void) f.coupling->configure_reacting_gas(database));
    EXPECT_EQ(f.coupling->find_reacting_gas_model(), nullptr);
    EXPECT_DOUBLE_EQ(f.solver->time(), 0);
    const M::Mixture mixture;
    auto options = M::field_options(mixture);
    EXPECT_NO_THROW((void) f.coupling->configure_reacting_gas(options));
    EXPECT_NE(f.coupling->find_reacting_gas_model(), nullptr);
    int empty = f.grid->num_owned_cells() == 0, globalEmpty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &globalEmpty);
    if (comm->getSize() == 4)
        EXPECT_EQ(globalEmpty, 2);
}
TEST(ReactingLegacyCleanupTest, ReconfigurationRemovesOldSchemaOutputPointersAndRejectsStaleSnapshots)
{
    Fixture f;
    const M::Mixture mixture;
    auto& model = f.coupling->configure_reacting_gas(M::field_options(mixture));
    f.coupling->initialize_heated_box(300, 300);
    EXPECT_NO_THROW(f.coupling->step());
    const auto accepted = model.species_event_receipt();
    ASSERT_TRUE(accepted.accepted);
    EXPECT_TRUE(model.liquid_transport_volume_receipt().event.accepted);
    const auto saved = model.snapshot();
    ASSERT_GT(model.output_fields().count("species_101_phase_1"), 0u);
    model.configure({});
    EXPECT_FALSE(model.initial_state_initialized());
    EXPECT_FALSE(model.species_event_receipt().accepted);
    EXPECT_GT(model.species_event_receipt().initialization_generation, accepted.initialization_generation);
    EXPECT_EQ(model.species_event_receipt().schema_fingerprint, 0u);
    EXPECT_EQ(model.population_escape_receipt().event, model.species_event_receipt());
    EXPECT_TRUE(model.population_escape_receipt().events.empty());
    EXPECT_EQ(model.liquid_transport_volume_receipt().event, model.species_event_receipt());
    EXPECT_FALSE(model.liquid_transport_volume_receipt().complete_carrier);
    EXPECT_TRUE(model.source_ledgers().empty());
    EXPECT_TRUE(model.population_ledgers().empty());
    EXPECT_TRUE(model.escape_ledgers().empty());
    EXPECT_TRUE(model.species_histories().empty());
    EXPECT_TRUE(model.species_step_events().empty());
    EXPECT_EQ(model.output_fields().count("species_101_phase_1"), 0u);
    EXPECT_EQ(model.output_fields().count("C_H2"), 0u);
    EXPECT_THROW(model.restore(saved), std::invalid_argument);

    // Full reconfiguration starts new physical initialization authority; it
    // never rewinds the allocator used to distinguish accepted publications.
    f.coupling->configure_reacting_gas(M::field_options(mixture));
    EXPECT_FALSE(model.species_event_receipt().accepted);
    EXPECT_FALSE(model.liquid_transport_volume_receipt().event.accepted);
    EXPECT_EQ(model.liquid_transport_volume_receipt().event, model.species_event_receipt());
    EXPECT_NO_THROW(f.coupling->step());
    EXPECT_TRUE(model.species_event_receipt().accepted);
    EXPECT_GT(model.species_event_receipt().publication_nonce, accepted.publication_nonce);
    EXPECT_GT(model.species_event_receipt().initialization_generation, accepted.initialization_generation);
    EXPECT_EQ(model.liquid_transport_volume_receipt().event, model.species_event_receipt());
    EXPECT_THROW(model.restore(saved), std::invalid_argument);
}
TEST(ReactingLegacyCleanupTest, DeclaredBuiltInCompleteDissolutionPublishesAnExplicitDeath)
{
    Fixture f;
    const M::Mixture mixture;
    M::InitialMixture seed;
    seed.populations = {{"micro", 100, 1e-5, .25}, {"large"}};
    auto options = M::field_options(mixture, seed);
    options.micro_dissolution_frequency = 1e6; // exp(-frequency*.01) underflows to zero.
    options.species_properties.clear();        // This oracle isolates the declared first-order operator, not film.
    auto& model = f.coupling->configure_reacting_gas(options);
    f.coupling->initialize_heated_box(300, 300);
    const auto before = model.global_species_moles(M::gas_one);
    f.coupling->step();
    EXPECT_NEAR(model.global_species_moles(M::gas_one), before, 64 * std::numeric_limits<double>::epsilon() * before);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(c);
        EXPECT_DOUBLE_EQ(model.population_number_field("micro").value(cell), 0);
        EXPECT_DOUBLE_EQ(model.species_field(M::gas_one, Reacting::SpeciesPhase::Dispersed, "micro").value(cell), 0);
        EXPECT_DOUBLE_EQ(model.species_field(M::gas_two, Reacting::SpeciesPhase::Dispersed, "micro").value(cell), 0);
    }
}
TEST(ReactingLegacyCleanupTest, UpstreamCompleteChemistryCannotBorrowDefaultCountDeathAndRollbackIsAtomic)
{
    using P = Reacting::SpeciesPhase;
    for (const bool conversion : {false, true})
    {
        SCOPED_TRACE(conversion);
        Fixture f;
        const M::Mixture mixture(false, false, Reacting::EnergyConvention::SensibleEnergy, 0);
        M::InitialMixture seed;
        seed.populations = {{"micro", 100, 1e-5, .25}, {"large", 200, 1e-5, .5}};
        auto options = M::field_options(mixture, seed);
        options.species_properties.clear(); // Isolate declared first-order dissolution/conversion, not film.
        options.micro_dissolution_frequency = conversion ? 0 : 1e6;
        options.micro_conversion_frequency = conversion ? 1e6 : 0;
        options.reaction_mechanism = std::make_shared<const Reacting::ReactionMechanism>(mixture.catalog,
            std::vector<Reacting::ReactionDefinition>{
                {711, "chemical relocation of first gas",
                    {{M::gas_one, P::Dispersed, -1, "micro"}, {M::gas_one, P::Dissolved, 1}}, 0},
                {713, "chemical relocation of second gas",
                    {{M::gas_two, P::Dispersed, -1, "micro"}, {M::gas_two, P::Dissolved, 1}}, 0}},
            Reacting::EnergyConvention::SensibleEnergy);
        options.species_source = [](const Reacting::ReactionContext& context, double duration)
        {
            Reacting::ReactionSourceInput input;
            if (duration > .005)
                input.integratedExtents = {
                    {711, context.state.amount(context.catalog, M::gas_one, P::Dispersed, "micro")},
                    {713, context.state.amount(context.catalog, M::gas_two, P::Dispersed, "micro")}};
            return input;
        };
        auto& model = f.coupling->configure_reacting_gas(options);
        f.coupling->initialize_heated_box(300, 300);
        const M::SavedReactingState saved(model);
        const auto receipt = model.species_event_receipt();
        const auto count_escape = model.population_escape_receipt();
        const auto histories = model.species_histories();
        const std::array<double, 2> initial_moles{
            model.global_species_moles(M::gas_one), model.global_species_moles(M::gas_two)};
        const auto values = [&](const auto& field)
        {
            using Value = std::remove_cvref_t<decltype(field.value(Pack::local_ordinal_type{}))>;
            std::vector<Value> result;
            for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
                result.push_back(field.value(static_cast<Pack::local_ordinal_type>(c)));
            return result;
        };
        const auto temperature = values(f.solver->temperature()), pressure = values(f.solver->pressure()),
                   density = values(f.solver->material_properties().density);
        const auto velocity = values(f.solver->velocity());
        std::vector<double> flux;
        for (std::size_t face = 0; face < f.grid->num_owned_faces(); ++face)
            flux.push_back(
                f.solver->pressure_corrected_face_fluxes().value(static_cast<Pack::local_ordinal_type>(face)));
        EXPECT_ANY_THROW(f.coupling->step()); // .01s: chemistry removes material before the default number law.
        EXPECT_DOUBLE_EQ(f.solver->time(), 0);
        EXPECT_EQ(values(f.solver->temperature()), temperature);
        EXPECT_EQ(values(f.solver->pressure()), pressure);
        EXPECT_EQ(values(f.solver->material_properties().density), density);
        const auto restored_velocity = values(f.solver->velocity());
        for (std::size_t c = 0; c < velocity.size(); ++c)
            for (std::size_t axis = 0; axis < 3; ++axis)
                EXPECT_DOUBLE_EQ(restored_velocity[c].component(axis), velocity[c].component(axis));
        for (std::size_t face = 0; face < flux.size(); ++face)
            EXPECT_DOUBLE_EQ(
                f.solver->pressure_corrected_face_fluxes().value(static_cast<Pack::local_ordinal_type>(face)),
                flux[face]);
        saved.expect_restored(model);
        EXPECT_EQ(model.species_event_receipt(), receipt);
        EXPECT_EQ(model.population_escape_receipt(), count_escape);
        EXPECT_EQ(model.species_histories(), histories);
        // Retry with no upstream chemistry: this operator genuinely owns complete material/count removal.
        f.solver->set_time_step(.001);
        EXPECT_NO_THROW(f.coupling->step());
        EXPECT_DOUBLE_EQ(f.solver->time(), .001);
        EXPECT_NEAR(model.global_species_moles(M::gas_one), initial_moles[0],
            64 * std::numeric_limits<double>::epsilon() * initial_moles[0]);
        EXPECT_NEAR(model.global_species_moles(M::gas_two), initial_moles[1],
            64 * std::numeric_limits<double>::epsilon() * initial_moles[1]);
        for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(c);
            EXPECT_DOUBLE_EQ(model.population_number_field("micro").value(cell), 0);
            EXPECT_DOUBLE_EQ(model.species_field(M::gas_one, P::Dispersed, "micro").value(cell), 0);
            EXPECT_DOUBLE_EQ(model.species_field(M::gas_two, P::Dispersed, "micro").value(cell), 0);
        }
        const auto comm = f.grid->owned_cell_map()->getComm();
        int local_empty = f.grid->num_owned_cells() == 0, actual_empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &actual_empty);
        EXPECT_EQ(actual_empty, std::max(0, comm->getSize() - 2));
    }
}
TEST(ReactingLegacyCleanupTest, ExplicitNumberOwnerMayCloseUpstreamChemicalDisappearance)
{
    using P = Reacting::SpeciesPhase;
    Fixture f;
    const M::Mixture mixture(false, false, Reacting::EnergyConvention::SensibleEnergy, 0);
    M::InitialMixture seed;
    seed.populations = {{"micro", 100, 1e-5, .25}, {"large"}};
    auto options = M::field_options(mixture, seed);
    options.micro_dissolution_frequency = 1e6; // The explicit packet law, not this default, owns the count.
    options.species_properties.clear();
    options.reaction_mechanism = std::make_shared<const Reacting::ReactionMechanism>(mixture.catalog,
        std::vector<Reacting::ReactionDefinition>{
            {711, "first explicit chemical relocation",
                {{M::gas_one, P::Dispersed, -1, "micro"}, {M::gas_one, P::Dissolved, 1}}, 0},
            {713, "second explicit chemical relocation",
                {{M::gas_two, P::Dispersed, -1, "micro"}, {M::gas_two, P::Dissolved, 1}}, 0}},
        Reacting::EnergyConvention::SensibleEnergy);
    options.species_source = [](const Reacting::ReactionContext& context, double)
    {
        Reacting::ReactionSourceInput input;
        input.integratedExtents = {{711, context.state.amount(context.catalog, M::gas_one, P::Dispersed, "micro")},
            {713, context.state.amount(context.catalog, M::gas_two, P::Dispersed, "micro")}};
        return input;
    };
    options.population_process = [](const BubbleGasPhysics::MultispeciesPopulationContext& context, double)
    {
        BubbleGasPhysics::MultispeciesPopulationPacket packet;
        for (const auto& population : context.populations)
            if (population.id == "micro")
                packet.numberEvents.push_back({"micro", 0, population.number});
        return packet;
    };
    auto& model = f.coupling->configure_reacting_gas(options);
    f.coupling->initialize_heated_box(300, 300);
    const double initial = model.global_species_moles(M::gas_one);
    EXPECT_NO_THROW(f.coupling->step());
    EXPECT_DOUBLE_EQ(f.solver->time(), .01);
    EXPECT_NEAR(model.global_species_moles(M::gas_one), initial, 64 * std::numeric_limits<double>::epsilon() * initial);
    EXPECT_TRUE(model.species_event_receipt().accepted);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(c);
        EXPECT_DOUBLE_EQ(model.population_number_field("micro").value(cell), 0);
        EXPECT_DOUBLE_EQ(model.species_field(M::gas_one, P::Dispersed, "micro").value(cell), 0);
        EXPECT_DOUBLE_EQ(model.species_field(M::gas_two, P::Dispersed, "micro").value(cell), 0);
    }
}
TEST(ReactingLegacyCleanupTest, FailedTypedReinitializationPreservesAcceptedStateAndNeverReusesTentativeAuthority)
{
    namespace Mixture = SimpleFluid::Testing::ReactingMixture;
    using Coupler = SimpleFluid::MultiphysicsSolverCoupler<Pack>;
    using Model = Coupler::reacting_gas_model_type;
    Fixture f;
    f.solver->temperature().put_scalar(300.);
    const auto comm = f.grid->owned_cell_map()->getComm();
    int local_empty = f.grid->num_owned_cells() == 0, actual_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &actual_empty);
    EXPECT_EQ(actual_empty, std::max(0, comm->getSize() - 2));
    const Mixture::Mixture chemistry;
    auto options = Mixture::field_options(chemistry, {}, .01, .25);
    for (auto& species : options.species_properties)
        species.diffusivity = 0;
    Model* model = nullptr;
    std::optional<Model::StateSnapshot> tentative;
    bool reject_derived_eos = false;
    const auto base = options.liquid_thermodynamics;
    options.liquid_thermodynamics = std::make_shared<const Reacting::MixtureThermodynamics>(
        chemistry.catalog, base->bounds(), Reacting::EOSKind::Liquid,
        [base, &model, &tentative, &reject_derived_eos](double t, double p, std::span<const double> x)
        {
            if (reject_derived_eos)
            {
                // This callback runs after initialize_species staged its new authority.
                tentative = model->snapshot();
                throw std::runtime_error("Deliberate post-publication derived EOS rejection.");
            }
            return base->evaluate(t, p, x);
        },
        base->energy_convention());
    auto& gas = f.coupling->configure_reacting_gas(options);
    model = &gas;
    gas.initialize_state(0., f.solver->temperature(), f.solver->pressure(), f.solver->velocity(),
        f.solver->material_properties());
    Model::face_flux_field_type flux(f.grid, 0.0, "initialization-regression-zero-flux");
    gas.advance(.001, .001, f.solver->temperature(), f.solver->pressure(), f.solver->velocity(), flux,
        f.solver->material_properties());
    const Mixture::SavedReactingState accepted(gas);
    const auto accepted_snapshot = gas.snapshot();
    const auto accepted_number_escape = gas.population_escape_receipt();
    tentative = accepted_snapshot; // Empty owners have no EOS callback; their local snapshot stays valid.
    reject_derived_eos = true;
    EXPECT_THROW(gas.initialize_state(.001, f.solver->temperature(), f.solver->pressure(), f.solver->velocity(),
                     f.solver->material_properties(), true),
        std::runtime_error);
    accepted.expect_restored(gas);
    EXPECT_TRUE(gas.initial_state_initialized());
    EXPECT_EQ(gas.population_escape_receipt(), accepted_number_escape);
    EXPECT_THROW(gas.restore(*tentative), std::invalid_argument); // Actual owners expose the rejected generation.
    accepted.expect_restored(gas);
    reject_derived_eos = false;
    ASSERT_NO_THROW(gas.initialize_state(.001, f.solver->temperature(), f.solver->pressure(), f.solver->velocity(),
        f.solver->material_properties(), true));
    // A monotonic allocator prevents the rejected transient from becoming valid on retry.
    EXPECT_THROW(gas.restore(*tentative), std::invalid_argument);
    EXPECT_THROW(gas.restore(accepted_snapshot), std::invalid_argument); // Explicit reinitialization remains a boundary.
    const auto before_configure = gas.snapshot();
    gas.configure(options);
    EXPECT_THROW(gas.restore(before_configure), std::invalid_argument); // Configuration authority remains strict too.
}
} // namespace
