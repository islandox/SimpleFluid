/** @file
 * @brief Independent finite-donor and heat-budget checks for Boussinesq vapor sources.
 */
#include "equations/BoussFiniteVapor.hh"
#include "utils/CompensatedSum.hh"
#include <gtest/gtest.h>

namespace SimpleFluid
{
namespace
{
namespace R = Reacting;
namespace B = BubbleGasPhysics;
using P = R::SpeciesPhase;
constexpr R::SpeciesId solvent = 29, donor = 73, gasOne = 5, gasTwo = 101;
constexpr R::ReactionId gasReaction = 47;
struct Fixture
{
    std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{gasTwo, "ordinary_second_gas", 1, {{"X", 1}}, {P::Resolved, P::Dispersed}},
            {solvent, "finite_solvent", 1, {{"S", 1}}, {P::Liquid, P::Resolved, P::Dispersed}},
            {donor, "finite_gas_donor", 2, {{"X", 2}}, {P::Liquid}},
            {gasOne, "ordinary_first_gas", 1, {{"X", 1}}, {P::Resolved, P::Dispersed}}},
        19);
    R::ReactionMechanism mechanism{catalog,
        {{gasReaction, "two_gases", {{donor, P::Liquid, -1}, {gasOne, P::Resolved, 1}, {gasTwo, P::Resolved, 1}}, 0}}};
    R::BoussReferenceVolume volumes{
        "manufactured.reference_volume", 1, catalog->fingerprint(), 2, 2, 5, {.2, 1, 0, .1}};
    R::BoussCoordinatedSources sources{"manufactured.coordinator", 1, catalog->fingerprint()};
    R::ReactionEnvironment environment{0, 383, 100000, 1, 1, R::ReactionVolumeClosure::ReferenceControlVolume};
    R::CellSpeciesState initial{*catalog};
    Fixture()
    {
        initial.set_amount(*catalog, solvent, P::Liquid, 2);
        initial.set_amount(*catalog, donor, P::Liquid, 1);
        initial.set_amount(*catalog, gasOne, P::Resolved, 1.9); // .19 m3, explicitly supplied physical stock.
        sources.vapor = R::BoussVaporBinding{solvent, P::Resolved, {}, 373, 10, 1, 0};
    }
    R::BoussSourceAdvance step(double dt = 1, double maximum = .2, R::ReactionSourceInput actual = {}) const
    {
        auto currentEnvironment = environment;
        currentEnvironment.liquidVolume = R::bouss_liquid_volume(*catalog, volumes, initial);
        return R::advance_bouss_sources(
            mechanism, volumes, sources, initial, {}, currentEnvironment, dt, maximum, actual);
    }
};
TEST(BoussFiniteVaporTest, AdmittedLatentAndReferenceThermalStorageMatchIndependentScalarLimit)
{
    Fixture f;
    const auto accepted = f.step();
    const double mass = accepted.vapor.evaporation;
    EXPECT_NEAR(mass, .01, 1e-15);
    EXPECT_NEAR(R::bouss_gas_volume(*f.catalog, f.volumes, accepted.state), .2, 1e-15);
    EXPECT_NEAR(accepted.vapor.latentEnergy, -.1, 1e-14);
    EXPECT_NEAR(383 + static_cast<double>(accepted.thermalEnergy) / (2 * 5), 382.99, 1e-13);
    EXPECT_NEAR(accepted.vapor.evaporation * 10, -accepted.vapor.latentEnergy, 1e-15);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, gasOne, P::Resolved), 1.9);
}
TEST(BoussFiniteVaporTest, LongStepCannotSpendMoreThanSensibleSuperheat)
{
    Fixture f;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.vapor->latentHeat = 1000;
    const auto accepted = f.step(2, .9);
    EXPECT_NEAR(accepted.vapor.evaporation, .1, 1e-15);
    EXPECT_NEAR(accepted.thermalEnergy, -100, 1e-12);
    EXPECT_NEAR(383 + static_cast<double>(accepted.thermalEnergy) / 10, 373, 1e-13);
    EXPECT_NEAR(-static_cast<double>(accepted.vapor.latentEnergy) / 2, 50, 1e-13);
}
TEST(BoussFiniteVaporTest, SharedGasRequestAdmissionPrecedesVaporWithoutScalingActualSources)
{
    Fixture f;
    unsigned requests = 0;
    f.sources.gas = R::BoussGasAdmissionLaw{"manufactured.gas_request", 1, f.catalog->fingerprint(),
        [&requests](const R::ReactionContext&, double)
        {
            ++requests;
            return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1}};
        }};
    const auto accepted = f.step();
    EXPECT_EQ(requests, 1U); // Representability previews never re-evaluate the supplied request law.
    ASSERT_EQ(accepted.admissions.size(), 1);
    EXPECT_DOUBLE_EQ(accepted.admissions[0].requestedMoles, 1);
    EXPECT_NEAR(accepted.admissions[0].admittedMoles, .01 / .3, 1e-15);
    EXPECT_NEAR(R::bouss_gas_volume(*f.catalog, f.volumes, accepted.state), .2, 1e-15);
    EXPECT_LE(accepted.vapor.evaporation, 1e-15);
    // Independently sum the actual stored products, without a capacity tolerance
    // or the production helper's rounded double volume report.
    detail::CompensatedSum<long double> remaining;
    remaining += static_cast<long double>(f.environment.bulkVolume) * .2;
    for (std::size_t i = 0; i < f.catalog->species().size(); ++i)
    {
        const auto id = f.catalog->species()[i].id;
        remaining += -static_cast<long double>(accepted.state.amount(*f.catalog, id, P::Resolved)) *
                     f.volumes.gasMolarVolumes[i];
    }
    EXPECT_GE(remaining.value(), 0);
    EXPECT_GT(accepted.admissions[0].admittedMoles, 0);
    EXPECT_DOUBLE_EQ(accepted.chemistry.ledger.reactionExtents[0], accepted.admissions[0].admittedMoles);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, donor, P::Liquid), 1 - accepted.admissions[0].admittedMoles);
    R::ReactionSourceInput actual;
    actual.integratedExtents = {{gasReaction, 1}};
    EXPECT_THROW(f.step(1, .2, actual), std::invalid_argument);
}
TEST(BoussFiniteVaporTest, GasAndVaporShareFiniteSolventAcrossExplicitSequentialStages)
{
    Fixture f;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, .003);
    const auto accepted = f.step(1, .9);
    EXPECT_DOUBLE_EQ(accepted.vapor.evaporation, .003);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Liquid), 0);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Resolved), .003);
    const auto second =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, accepted.state, {}, f.environment, 1, .9, {});
    EXPECT_DOUBLE_EQ(second.vapor.evaporation, 0);
}
TEST(BoussFiniteVaporTest, CondensationConsumesActualVaporAndReleasesItsAdmittedEnergyOnce)
{
    Fixture f;
    f.environment.temperature = 363;
    f.initial.set_amount(*f.catalog, solvent, P::Resolved, .005);
    const auto accepted = f.step();
    EXPECT_DOUBLE_EQ(accepted.vapor.condensation, .005);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Resolved), 0);
    EXPECT_NEAR(accepted.state.amount(*f.catalog, solvent, P::Liquid), 2.005, 1e-15);
    EXPECT_NEAR(accepted.thermalEnergy, .05, 1e-15);
    ASSERT_EQ(accepted.transfers.size(), 1);
    EXPECT_EQ(accepted.transfers[0].donor, P::Resolved);
    EXPECT_EQ(accepted.transfers[0].receiver, P::Liquid);
}
TEST(BoussFiniteVaporTest, CondensationUsesSubcoolingLeftByActualAndAdmittedSourceHeating)
{
    // C=100 J/K and initial subcooling=1000 J. Test independent heat,
    // actual reaction heat, and capacity-admitted reaction heat separately.
    for (const double heating : {980.0, 1000.0, 1020.0})
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            SCOPED_TRACE(::testing::Message() << "heating=" << heating << " channel=" << channel);
            Fixture f;
            f.environment.temperature = 363;
            f.environment.bulkVolume = 10;
            f.initial.set_amount(*f.catalog, solvent, P::Resolved, 5);
            auto definitions = f.mechanism.reactions();
            definitions[0].heatPerExtent = heating;
            const R::ReactionMechanism mechanism(f.catalog, definitions, R::EnergyConvention::SensibleEnergy);
            R::ReactionSourceInput actual;
            if (channel == 0)
                actual.heat.integratedEnergy = heating;
            else if (channel == 1)
                actual.integratedExtents = {{gasReaction, 1}};
            else
                f.sources.gas = R::BoussGasAdmissionLaw{"heated_admitted_request", 1, f.catalog->fingerprint(),
                    [](const R::ReactionContext&, double)
                    { return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1}}; }};
            const auto accepted = R::advance_bouss_sources(
                mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, actual);
            const double condensed = heating < 1000 ? 2 : 0;
            EXPECT_DOUBLE_EQ(accepted.vapor.requestedCondensation, 100);
            EXPECT_DOUBLE_EQ(accepted.vapor.condensation, condensed);
            EXPECT_DOUBLE_EQ(accepted.vapor.evaporation, 0);
            EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Resolved), 5 - condensed);
            EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Liquid), 2 + condensed);
            EXPECT_EQ(accepted.vapor.latentEnergy, 10 * condensed);
            EXPECT_EQ(accepted.thermalEnergy, heating + 10 * condensed);
            const double endpoint = 363 + static_cast<double>(accepted.thermalEnergy) / 100;
            EXPECT_DOUBLE_EQ(endpoint, heating <= 1000 ? 373 : 373.2);
            if (channel == 2)
            {
                ASSERT_EQ(accepted.admissions.size(), 1);
                EXPECT_DOUBLE_EQ(accepted.admissions[0].admittedMoles, 1);
            }
        }
}
TEST(BoussFiniteVaporTest, CondensationSubcoolingRetainsSmallRemainderAfterCancelingSourceHeats)
{
    Fixture f;
    f.environment.temperature = 363;
    f.environment.bulkVolume = 10;
    f.initial.set_amount(*f.catalog, solvent, P::Resolved, 5);
    auto definitions = f.mechanism.reactions();
    definitions[0].heatPerExtent = -1e20;
    const R::ReactionMechanism mechanism(f.catalog, definitions, R::EnergyConvention::SensibleEnergy);
    R::ReactionSourceInput actual;
    actual.heat.integratedEnergy = 1e20;
    actual.integratedExtents = {{gasReaction, 1}};
    // Transfer energy is an independent accepted source channel, not inferred
    // from carried enthalpy; the net heating is exactly 990 J.
    actual.externalTransfers = {{"declared_energy_feed", donor, P::Liquid, 0, 0, 990}};
    const auto accepted = R::advance_bouss_sources(
        mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, actual);
    EXPECT_DOUBLE_EQ(accepted.vapor.requestedCondensation, 100);
    EXPECT_DOUBLE_EQ(accepted.vapor.condensation, 1);
    EXPECT_EQ(accepted.vapor.latentEnergy, 10);
    EXPECT_EQ(accepted.thermalEnergy, 1000);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Resolved), 4);
}
TEST(BoussFiniteVaporTest, IndependentHeatIsNotConvertedToChemistryOrDoubleDebited)
{
    Fixture f;
    R::ReactionSourceInput input;
    input.heat.integratedEnergy = 100;
    const auto accepted = f.step(1, .2, input);
    EXPECT_NEAR(accepted.thermalEnergy, 99.9, 1e-13);
    EXPECT_NEAR(accepted.vapor.latentEnergy, -.1, 1e-14);
    EXPECT_DOUBLE_EQ(accepted.chemistry.ledger.reactionExtents[0], 0);
}
TEST(BoussFiniteVaporTest, ExplicitSensibleCorrectionChangesOnlyDeclaredScalarHeat)
{
    Fixture f;
    f.sources.vapor->sensibleCorrection = 2;
    const auto accepted = f.step();
    EXPECT_NEAR(accepted.vapor.evaporation, .01, 1e-15);
    EXPECT_NEAR(accepted.vapor.latentEnergy, -.1, 1e-14);
    EXPECT_NEAR(accepted.vapor.sensibleEnergy, .02, 1e-14);
    EXPECT_NEAR(accepted.thermalEnergy, -.08, 1e-14);
}
TEST(BoussFiniteVaporTest, NamedVaporRequiresActualExplicitBirthAndFullDisappearanceDeath)
{
    Fixture f;
    f.sources.vapor->vaporPhase = P::Dispersed;
    f.sources.vapor->population = "vapor_bin";
    f.initial.dispersedPopulations = {{"vapor_bin", std::vector<double>(f.catalog->species().size())}};
    const std::vector<B::PopulationNumberState> zero{{"vapor_bin", 0}};
    EXPECT_THROW(R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, zero, f.environment, 1, .2, {}),
        std::invalid_argument);
    f.sources.populationEvents = [](const R::BoussAcceptedMaterialContext& context)
    {
        return std::vector<B::PopulationNumberEvent>{{"vapor_bin", context.vapor.evaporation * 1000,
            context.vapor.condensation > 0 ? context.transportedNumbers[0].number : 0}};
    };
    const auto born =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, zero, f.environment, 1, .2, {});
    EXPECT_NEAR(born.numbers[0].number, 10, 1e-12);
    f.environment.temperature = 363;
    const auto dissolved =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, born.state, born.numbers, f.environment, 1, .2, {});
    EXPECT_DOUBLE_EQ(dissolved.numbers[0].number, 0);
    EXPECT_DOUBLE_EQ(dissolved.state.amount(*f.catalog, solvent, P::Dispersed, "vapor_bin"), 0);
}
TEST(BoussFiniteVaporTest, GrossCountDeathsCannotBorrowBirthsOrHideOneBeyondHugeStock)
{
    Fixture f;
    f.initial.dispersedPopulations = {{"micro", std::vector<double>(f.catalog->species().size())}};
    f.initial.set_amount(*f.catalog, gasOne, P::Dispersed, .01, "micro");
    f.sources.vapor.reset();
    f.sources.populationEvents = [](const R::BoussAcceptedMaterialContext&)
    { return std::vector<B::PopulationNumberEvent>{{"micro", 1e20, 1e20}, {"micro", 0, 1}}; };
    const std::vector<B::PopulationNumberState> numbers{{"micro", 1e20}};
    EXPECT_THROW(
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, numbers, f.environment, 1, .3, {}),
        std::invalid_argument);
}
TEST(BoussFiniteVaporTest, FailedProviderAndInvalidBindingLeaveInputsAndReplayUnchanged)
{
    Fixture f;
    const auto first = f.step();
    const auto saved = f.initial.amounts;
    f.sources.gas = R::BoussGasAdmissionLaw{"throwing.request", 1, f.catalog->fingerprint(),
        [](const R::ReactionContext&, double) -> std::vector<R::IntegratedReactionExtent>
        { throw std::runtime_error("manufactured source failure"); }};
    EXPECT_THROW(f.step(), std::runtime_error);
    EXPECT_EQ(f.initial.amounts, saved);
    f.sources.gas.reset();
    const auto replay = f.step();
    EXPECT_EQ(first.state.amounts, replay.state.amounts);
    EXPECT_EQ(first.thermalEnergy, replay.thermalEnergy);
    f.sources.schemaFingerprint ^= 1;
    EXPECT_THROW(f.step(), std::invalid_argument);
}
TEST(BoussFiniteVaporTest, RawGasCapacityDoesNotLoseOneMoleBesideHugeExistingStock)
{
    Fixture f;
    f.sources.vapor.reset();
    f.volumes.gasMolarVolumes[f.catalog->index(gasOne)] = 1;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 1e20);
    f.environment.bulkVolume = 2e20;
    R::ReactionSourceInput actual;
    actual.externalTransfers.push_back({"explicit.one_mole_reservoir", gasOne, P::Resolved, 1, 1});
    EXPECT_THROW(f.step(1, .5, actual), std::invalid_argument);
    EXPECT_DOUBLE_EQ(f.initial.amount(*f.catalog, gasOne, P::Resolved), 1e20);
}
TEST(BoussFiniteVaporTest, MicroLargeConversionPreservesEverySpeciesAndExplicitCounts)
{
    Fixture f;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.initial.dispersedPopulations = {{"micro", std::vector<double>(f.catalog->species().size())},
        {"large", std::vector<double>(f.catalog->species().size())}};
    f.initial.set_amount(*f.catalog, gasOne, P::Dispersed, .01, "micro");
    f.initial.set_amount(*f.catalog, gasTwo, P::Dispersed, .02, "micro");
    std::vector<B::PopulationNumberState> numbers{{"micro", 10}, {"large", 0}};
    B::MultispeciesPopulationPacket packet;
    packet.transfers = {{gasOne, P::Dispersed, P::Dispersed, "micro", "large", .01},
        {gasTwo, P::Dispersed, P::Dispersed, "micro", "large", .02}};
    packet.numberEvents = {{"micro", 0, 10}, {"large", 10, 0}};
    const auto accepted =
        R::advance_bouss_population_exchange(*f.catalog, f.volumes, f.initial, numbers, f.environment, .2, packet);
    EXPECT_DOUBLE_EQ(accepted.numbers[0].number, 0);
    EXPECT_DOUBLE_EQ(accepted.numbers[1].number, 10);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, gasOne, P::Dispersed, "micro"), 0);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, gasTwo, P::Dispersed, "micro"), 0);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, gasOne, P::Dispersed, "large"), .01);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, gasTwo, P::Dispersed, "large"), .02);
    ASSERT_EQ(accepted.ledger.reactionExtents.size(), packet.transfers.size());
    for (std::size_t i = 0; i < packet.transfers.size(); ++i)
        EXPECT_DOUBLE_EQ(accepted.ledger.reactionExtents[i], packet.transfers[i].moles);
    packet.numberEvents.clear();
    EXPECT_THROW(
        R::advance_bouss_population_exchange(*f.catalog, f.volumes, f.initial, numbers, f.environment, .2, packet),
        std::invalid_argument);
}
TEST(BoussFiniteVaporTest, AdmittedRequestsReserveRawDonorsJointlyWithoutLosingSmallPriorDebit)
{
    Fixture f;
    f.sources.vapor.reset();
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.initial.set_amount(*f.catalog, donor, P::Liquid, 1e20);
    f.environment.bulkVolume = 1e21;
    f.sources.gas = R::BoussGasAdmissionLaw{"two_competing_requests", 1, f.catalog->fingerprint(),
        [](const R::ReactionContext&, double)
        { return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1}, {gasReaction, 1e20}}; }};
    // The maximal mathematical second extent is not representable as a double.
    // Reject that proposed extent rather than borrowing its already-spent one mole.
    EXPECT_THROW(f.step(1, .9), std::invalid_argument);
    EXPECT_DOUBLE_EQ(f.initial.amount(*f.catalog, donor, P::Liquid), 1e20);
}
TEST(BoussFiniteVaporTest, FormationEnergyConventionCannotOwnReferenceCpTPlusLatentHeat)
{
    Fixture f;
    const R::ReactionMechanism formation(
        f.catalog, f.mechanism.reactions(), R::EnergyConvention::IncludesFormationEnergy);
    EXPECT_THROW(R::validate_bouss_reference(formation, f.volumes), std::invalid_argument);
    EXPECT_NO_THROW(R::validate_bouss_reference(f.mechanism, f.volumes));
}
R::BoussWallHeatBudget wall_budget(const Fixture& f, std::vector<R::BoussWallHeatPacket> packets)
{
    return {
        &f, 1, R::bouss_configuration_signature(f.volumes, f.sources), f.environment.time + 1, 1, std::move(packets)};
}
TEST(BoussFiniteVaporTest, WallAndBulkRequestsShareOneProportionalMaterialAndCapacityAdmission)
{
    Fixture f;
    f.sources.vapor->walls = {{"top", 100, 1}};
    auto wall = wall_budget(f, {{0, 91, 100}});
    const auto accepted =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .2, {}, {}, &wall);
    ASSERT_EQ(accepted.vapor.evaporationAdmissions.size(), 2);
    EXPECT_NEAR(accepted.vapor.evaporationAdmissions[0].admittedMoles, .005, 1e-15);
    EXPECT_NEAR(accepted.vapor.evaporationAdmissions[1].admittedMoles, .005, 1e-15);
    EXPECT_NEAR(accepted.vapor.evaporation, .01, 1e-15);
    EXPECT_NEAR(accepted.thermalEnergy, -.1, 1e-14);
    EXPECT_EQ(accepted.vapor.evaporationAdmissions[1].patch, "top");
    EXPECT_DOUBLE_EQ(accepted.vapor.evaporationAdmissions[1].suppliedWallEnergy, 100);
    EXPECT_DOUBLE_EQ(accepted.vapor.evaporationAdmissions[1].allocatedWallEnergy, 100);
}
TEST(BoussFiniteVaporTest, WallHeatFundsOnlyAcceptedLatentAndIsNotAddedAgainAsAHeatSource)
{
    Fixture f;
    f.environment.temperature = 373;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.vapor->bulkEnabled = false;
    f.sources.vapor->walls = {{"top", 40, .25}};
    auto wall = wall_budget(f, {{0, 91, 40}});
    const auto accepted =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, {}, {}, &wall);
    EXPECT_NEAR(accepted.vapor.evaporation, .9, 1e-15); // Capacity, not the requested 1kg, owns admission.
    EXPECT_NEAR(accepted.thermalEnergy, -9, 1e-13);
    EXPECT_NEAR(373 + (40 + static_cast<double>(accepted.thermalEnergy)) / 10, 376.1, 1e-13);
    EXPECT_DOUBLE_EQ(accepted.chemistry.ledger.externalHeat, 0);
}
TEST(BoussFiniteVaporTest, WallBulkCannotSpendMoreThanSharedFiniteSolvent)
{
    Fixture f;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, .003);
    f.sources.vapor->walls = {{"top", 100, 1}};
    auto wall = wall_budget(f, {{0, 91, 100}});
    const auto accepted =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, {}, {}, &wall);
    long double remaining = f.initial.amount(*f.catalog, solvent, P::Liquid);
    for (const auto& origin : accepted.vapor.evaporationAdmissions)
        remaining -= origin.admittedMoles;
    EXPECT_GE(remaining, 0);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Liquid), static_cast<double>(remaining));
    EXPECT_NEAR(accepted.vapor.evaporationAdmissions[0].admittedMoles, .0015, 1e-17);
    EXPECT_NEAR(accepted.vapor.evaporationAdmissions[1].admittedMoles, .0015, 1e-17);
    EXPECT_NEAR(accepted.vapor.evaporation, .003, 1e-17);
}
TEST(BoussFiniteVaporTest, CondensationUsesOldVaporAndDoesNotCreditNewLiquidToWallDonors)
{
    Fixture f;
    f.environment.temperature = 363;
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, 0);
    f.initial.set_amount(*f.catalog, solvent, P::Resolved, .005);
    f.sources.vapor->walls = {{"top", .1, 1}};
    auto wall = wall_budget(f, {{0, 91, .1}});
    const auto accepted =
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .2, {}, {}, &wall);
    EXPECT_DOUBLE_EQ(accepted.vapor.condensation, .005);
    EXPECT_DOUBLE_EQ(accepted.vapor.evaporation, 0); // No borrowing the same stage's incoming solvent.
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, solvent, P::Liquid), .005);
    EXPECT_NEAR(accepted.thermalEnergy, .05, 1e-15);
}
TEST(BoussFiniteVaporTest, CertifiedWallBudgetRejectsMissingStaleOrForeignInputs)
{
    Fixture f;
    f.sources.vapor->walls = {{"top", 100, 1}};
    EXPECT_THROW(f.step(), std::invalid_argument);
    auto wall = wall_budget(f, {{0, 91, 100}});
    wall.configurationSignature ^= 1;
    EXPECT_THROW(
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .2, {}, {}, &wall),
        std::invalid_argument);
    wall = wall_budget(f, {{0, 91, 100}});
    wall.duration = .5;
    EXPECT_THROW(
        R::advance_bouss_sources(f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .2, {}, {}, &wall),
        std::invalid_argument);
}
TEST(BoussFiniteVaporTest, RawWallOriginEnergyRetainsOneJouleBesideHugeCancelingChemicalHeat)
{
    Fixture f;
    f.environment.temperature = 373;
    f.environment.bulkVolume = 1e22;
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, 1e21);
    f.sources.vapor->bulkEnabled = false;
    f.sources.vapor->latentHeat = 1;
    f.sources.vapor->walls = {{"top", 1e20, 1}, {"side", 1, 1}};
    auto wall = wall_budget(f, {{0, 91, 1e20}, {1, 93, 1}});
    R::ReactionSourceInput actual;
    actual.heat.integratedEnergy = 1e20;
    const auto accepted = R::advance_bouss_sources(
        f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, actual, {}, &wall);
    ASSERT_EQ(accepted.vapor.evaporationAdmissions.size(), 2);
    EXPECT_DOUBLE_EQ(accepted.vapor.evaporationAdmissions[1].admittedMoles, 1);
    EXPECT_EQ(accepted.thermalEnergy, -1.0L);
}
TEST(BoussFiniteVaporTest, RawSameReactionStageHeatSurvivesLaterWallLatentCancellation)
{
    Fixture f;
    auto definitions = f.mechanism.reactions();
    definitions[0].heatPerExtent = 1;
    const R::ReactionMechanism mechanism(f.catalog, definitions, R::EnergyConvention::SensibleEnergy);
    f.environment.temperature = 373;
    f.environment.bulkVolume = 1e23;
    f.initial.set_amount(*f.catalog, donor, P::Liquid, 1e21);
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, 1e21);
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.gas = R::BoussGasAdmissionLaw{"one_extra_chemical_joule", 1, f.catalog->fingerprint(),
        [](const R::ReactionContext&, double) { return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1}}; }};
    f.sources.vapor->bulkEnabled = false;
    f.sources.vapor->latentHeat = 1;
    f.sources.vapor->walls = {{"top", 1e20, 1}};
    auto wall = wall_budget(f, {{0, 91, 1e20}});
    R::ReactionSourceInput actual;
    actual.integratedExtents = {{gasReaction, 1e20}};
    const auto accepted = R::advance_bouss_sources(
        mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, actual, {}, &wall);
    EXPECT_EQ(accepted.thermalEnergy, 1.0L);
    EXPECT_GT(accepted.thermalContributions.size(), accepted.chemistry.localReceipt.reactionHeatContributions.size());
}
TEST(BoussFiniteVaporTest, RawNumberReceiptRetainsSmallBirthAfterCompleteHugeCountDeath)
{
    Fixture f;
    f.initial.dispersedPopulations = {{"micro", std::vector<double>(f.catalog->species().size())}};
    f.initial.set_amount(*f.catalog, gasOne, P::Dispersed, .01, "micro");
    const std::vector<B::PopulationNumberState> numbers{{"micro", 1e20}};
    B::MultispeciesPopulationPacket packet;
    packet.numberEvents = {{"micro", 1, 1e20}};
    const auto accepted =
        R::advance_bouss_population_exchange(*f.catalog, f.volumes, f.initial, numbers, f.environment, .3, packet);
    EXPECT_DOUBLE_EQ(accepted.numbers[0].number, 1);
    ASSERT_EQ(accepted.localNumberChanges.size(), 1);
    detail::CompensatedSum<long double> endpoint;
    endpoint += numbers[0].number;
    endpoint += accepted.localNumberChanges[0].high;
    endpoint += accepted.localNumberChanges[0].correction;
    EXPECT_EQ(endpoint.value(), 1.0L);
}
TEST(BoussFiniteVaporTest, RawActualDonorDebitCannotFundAMaximalAdmittedRequest)
{
    Fixture f;
    f.environment.bulkVolume = 1e23;
    f.initial.set_amount(*f.catalog, donor, P::Liquid, 1e20);
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.vapor.reset();
    f.sources.gas = R::BoussGasAdmissionLaw{"finite_after_actual", 1, f.catalog->fingerprint(),
        [](const R::ReactionContext&, double)
        { return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1e20}}; }};
    R::ReactionSourceInput actual;
    actual.integratedExtents = {{gasReaction, 1}};
    // Actual outgoing1 plus admitted outgoing1e20 exceeds actual initial1e20,
    // even though the intermediate public stock rounds unchanged.
    EXPECT_THROW(f.step(1, .9, actual), std::invalid_argument);
    EXPECT_DOUBLE_EQ(f.initial.amount(*f.catalog, donor, P::Liquid), 1e20);
}
TEST(BoussFiniteVaporTest, CompositeRawEndpointRetainsActualCreditAndAdmittedReactionBeforeReporting)
{
    Fixture f;
    f.environment.bulkVolume = 1e23;
    f.initial.set_amount(*f.catalog, donor, P::Liquid, 1e20);
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.vapor.reset();
    f.sources.gas = R::BoussGasAdmissionLaw{"one_funded_request", 1, f.catalog->fingerprint(),
        [](const R::ReactionContext&, double) { return std::vector<R::IntegratedReactionExtent>{{gasReaction, 1}}; }};
    R::ReactionSourceInput actual;
    actual.integratedExtents = {{gasReaction, 1e20}};
    actual.externalTransfers = {{"finite_feed", donor, P::Liquid, 1, 2, 0}};
    const auto accepted = f.step(1, .9, actual);
    EXPECT_DOUBLE_EQ(accepted.state.amount(*f.catalog, donor, P::Liquid), 0);
    EXPECT_DOUBLE_EQ(accepted.admissions.front().admittedMoles, 1);
    const auto slot = static_cast<std::size_t>(P::Resolved) * f.catalog->species().size() + f.catalog->index(gasOne);
    ASSERT_EQ(accepted.chemistry.localReceipt.postSourceAmounts.size(), accepted.chemistry.state.amounts.size());
    detail::CompensatedSum<long double> residual;
    residual += accepted.chemistry.localReceipt.postSourceAmounts[slot][0];
    residual += accepted.chemistry.localReceipt.postSourceAmounts[slot][1];
    residual += -1e20L;
    EXPECT_EQ(residual.value(), 1.0L); // Exact actual chemistry1e20 plus admitted1, before coarse reporting.
}
TEST(BoussFiniteVaporTest, RawActualSolventWithdrawalCannotFundFullWallEvaporation)
{
    Fixture f;
    f.environment.temperature = 373;
    f.environment.bulkVolume = 1e23;
    f.initial.set_amount(*f.catalog, solvent, P::Liquid, 1e20);
    f.initial.set_amount(*f.catalog, gasOne, P::Resolved, 0);
    f.sources.vapor->bulkEnabled = false;
    f.sources.vapor->latentHeat = 1;
    f.sources.vapor->walls = {{"top", 1e20, 1}};
    auto wall = wall_budget(f, {{0, 91, 1e20}});
    R::ReactionSourceInput actual;
    actual.externalTransfers = {{"finite_liquid_exit", solvent, P::Liquid, -1, -1, 0}};
    EXPECT_THROW(R::advance_bouss_sources(
                     f.mechanism, f.volumes, f.sources, f.initial, {}, f.environment, 1, .9, actual, {}, &wall),
        std::invalid_argument);
    EXPECT_DOUBLE_EQ(f.initial.amount(*f.catalog, solvent, P::Liquid), 1e20);
}
} // namespace
} // namespace SimpleFluid
