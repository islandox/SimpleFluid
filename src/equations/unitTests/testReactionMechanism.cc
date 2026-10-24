/** @file testReactionMechanism.cc
 * @brief Independent analytic and transactional checks for supplied multispecies chemistry.
 */
#include "equations/ReactionMechanism.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
namespace R = SimpleFluid::Reacting;
constexpr R::SpeciesId A = 11, B = 22, G1 = 51, G2 = 73;
constexpr R::ReactionId split = 8;

std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    using P = R::SpeciesPhase;
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{A, "donor", 0.010, {{"X", 5}}, {P::Liquid}},
            {B, "liquid product", 0.004, {{"X", 2}}, {P::Liquid}},
            {G1, "gas one", 0.002, {{"X", 1}}, {P::Dissolved, P::Dispersed, P::Resolved}},
            {G2, "gas two", 0.004, {{"X", 2}}, {P::Dissolved, P::Dispersed, P::Resolved}}},
        19);
}

R::ReactionDefinition split_reaction(double heat = 0.0)
{
    using P = R::SpeciesPhase;
    return {split, "A to B plus two different gases",
        {{A, P::Liquid, -1}, {B, P::Liquid, 1}, {G1, P::Dissolved, 1}, {G2, P::Dissolved, 1}}, heat};
}

R::ReactionEnvironment environment()
{
    return {0.0, 300.0, 100000.0, 0.2, 0.5};
}

R::CellSpeciesState state(const R::SpeciesCatalog& schema, double donor = 1.0)
{
    R::CellSpeciesState result(schema);
    result.set_amount(schema, A, R::SpeciesPhase::Liquid, donor);
    return result;
}

R::ReactionRateProvider constant_rate(
    const R::SpeciesCatalog& schema, double value, R::SourceVolumeBasis basis = R::SourceVolumeBasis::LiquidVolume)
{
    R::ReactionRateProvider result;
    result.identity = "constant verification source";
    result.schemaRevision = schema.revision();
    result.schemaFingerprint = schema.fingerprint();
    result.validityDescription = "Positive absolute pressure and temperature; prescribed manufactured rates.";
    result.parameterUnits = "mol/(m3 s)";
    result.validate = [](const R::ReactionContext&) {};
    result.evaluate = [value, basis](const R::ReactionContext&)
    { return std::vector<R::ReactionRate>{{split, value, basis}}; };
    return result;
}

R::ReactionSourceInput decay(const R::SpeciesCatalog& schema, double k)
{
    R::ReactionSourceInput result;
    auto provider = constant_rate(schema, 0.0);
    provider.identity = "first-order analytical batch fixture";
    provider.parameterUnits = "k in 1/s";
    provider.evaluate = [k](const R::ReactionContext& context)
    {
        const auto volume = context.environment.liquidVolume;
        const auto concentration =
            volume == 0.0 ? 0.0 : context.state.amount(context.catalog, A, R::SpeciesPhase::Liquid) / volume;
        return std::vector<R::ReactionRate>{{split, k * concentration, R::SourceVolumeBasis::LiquidVolume}};
    };
    result.rates = std::move(provider);
    return result;
}

long double mass(const R::SpeciesCatalog& schema, const R::CellSpeciesState& values)
{
    long double result = 0.0;
    for (const auto& species : schema.species())
        for (std::size_t phase = 0; phase < 4; ++phase)
            result += static_cast<long double>(species.molarMass) *
                      values.phase_amount(schema, species.id, static_cast<R::SpeciesPhase>(phase));
    return result;
}

TEST(ReactionMechanismTest, IntegratedMultigasStoichiometryAndIndependentHeat)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction(20.0)});
    R::ReactionSourceInput input;
    input.integratedExtents = {{split, 0.25}};
    input.heat.integratedEnergy = -3.0;
    const auto initial = state(*schema);
    const auto result = R::advance_reactions(mechanism, initial, environment(), 7.0, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 0.75);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, B, R::SpeciesPhase::Liquid), 0.25);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, R::SpeciesPhase::Dissolved), 0.25);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G2, R::SpeciesPhase::Dissolved), 0.25);
    EXPECT_DOUBLE_EQ(result.ledger.reactionHeat, 5.0);
    EXPECT_DOUBLE_EQ(result.ledger.externalHeat, -3.0);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 2.0);
    EXPECT_NEAR(static_cast<double>(mass(*schema, result.state) - mass(*schema, initial)), 0.0, 1e-17);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, A, R::SpeciesPhase::Liquid), 1.0);
}

TEST(ReactionMechanismTest, AnalyticalBatchAndSecondOrderTimeConvergence)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    const auto input = decay(*schema, 1.0);
    const auto coarse = R::advance_reactions(mechanism, state(*schema), environment(), 1.0, input, {0.1});
    const auto fine = R::advance_reactions(mechanism, state(*schema), environment(), 1.0, input, {0.05});
    const auto exact = std::exp(-1.0);
    const auto coarseError = std::abs(coarse.state.amount(*schema, A, R::SpeciesPhase::Liquid) - exact);
    const auto fineError = std::abs(fine.state.amount(*schema, A, R::SpeciesPhase::Liquid) - exact);
    EXPECT_GT(coarseError / fineError, 3.8);
    EXPECT_LT(coarseError / fineError, 4.3);
    EXPECT_LT(fineError, 2e-4);
    EXPECT_NEAR(fine.state.amount(*schema, G1, R::SpeciesPhase::Dissolved), 1.0 - exact, 2e-4);
    EXPECT_DOUBLE_EQ(fine.state.amount(*schema, G1, R::SpeciesPhase::Dissolved),
        fine.state.amount(*schema, G2, R::SpeciesPhase::Dissolved));
    EXPECT_NEAR(static_cast<double>(mass(*schema, fine.state)), 0.01, 2e-17);
}

TEST(ReactionMechanismTest, LiquidBulkAndIntegratedInputsScaleExactlyOnce)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    const auto initial = state(*schema, 100.0);
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 2.0, R::SourceVolumeBasis::LiquidVolume);
    auto result = R::advance_reactions(mechanism, initial, environment(), 0.25, input);
    EXPECT_NEAR(result.ledger.reactionExtents[0], 0.1, 1e-16);
    input.rates = constant_rate(*schema, 2.0, R::SourceVolumeBasis::BulkVolume);
    result = R::advance_reactions(mechanism, initial, environment(), 0.25, input);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], 0.25);
    input.rates.reset();
    input.integratedExtents = {{split, 0.4}};
    result = R::advance_reactions(mechanism, initial, environment(), 20.0, input);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], 0.4);
}

TEST(ReactionMechanismTest, CompetingReactionsReserveSharedDonorAtomically)
{
    const auto schema = catalog();
    auto alternate = split_reaction();
    alternate.id = 99;
    alternate.name = "competing channel";
    const R::ReactionMechanism mechanism(schema, {split_reaction(), alternate});
    const auto initial = state(*schema);
    R::ReactionSourceInput input;
    input.integratedExtents = {{split, 0.6}, {99, 0.6}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, A, R::SpeciesPhase::Liquid), 1.0);
    input.integratedExtents = {{split, 0.5}, {99, 0.5}};
    const auto result = R::advance_reactions(mechanism, initial, environment(), 1.0, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 0.0);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G2, R::SpeciesPhase::Dissolved), 1.0);
}

TEST(ReactionMechanismTest, ReverseReactionConsumesEveryProductReservoir)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction(12.0)});
    auto initial = state(*schema, 0.0);
    initial.set_amount(*schema, B, R::SpeciesPhase::Liquid, 0.5);
    initial.set_amount(*schema, G1, R::SpeciesPhase::Dissolved, 0.5);
    initial.set_amount(*schema, G2, R::SpeciesPhase::Dissolved, 0.4);
    R::ReactionSourceInput input;
    input.integratedExtents = {{split, -0.5}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::runtime_error);
    input.integratedExtents = {{split, -0.4}};
    const auto result = R::advance_reactions(mechanism, initial, environment(), 1.0, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 0.4);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G2, R::SpeciesPhase::Dissolved), 0.0);
    EXPECT_NEAR(result.ledger.reactionHeat, -4.8, 1e-15);
}

TEST(ReactionMechanismTest, SimultaneousProductionCannotMaskMissingGrossDonors)
{
    const auto schema = catalog();
    auto duplicate = split_reaction();
    duplicate.id = 99;
    duplicate.name = "opposite channel";
    const R::ReactionMechanism mechanism(schema, {split_reaction(), duplicate});
    R::ReactionSourceInput input;
    input.integratedExtents = {{split, 1.0}, {99, -1.0}};
    EXPECT_THROW(
        (void) R::advance_reactions(mechanism, state(*schema, 0.0), environment(), 1.0, input), std::runtime_error);
}

TEST(ReactionMechanismTest, BoundedSubcyclingRetriesJointDonorsWithoutClipping)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    const auto result = R::advance_reactions(mechanism, state(*schema), environment(), 1.0, decay(*schema, 10.0));
    EXPECT_GT(result.ledger.retries, 0u);
    EXPECT_GT(result.ledger.substeps, 1u);
    EXPECT_GE(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 0.0);
    EXPECT_NEAR(static_cast<double>(mass(*schema, result.state)), 0.01, 2e-17);
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 1.0);
    EXPECT_THROW((void) R::advance_reactions(mechanism, state(*schema, 0.0), environment(), 1.0, input, {1.0, 10, 3}),
        std::runtime_error);
}

TEST(ReactionMechanismTest, SeparateDispersedPopulationsCarryTwoGasDonors)
{
    const auto schema = catalog();
    using P = R::SpeciesPhase;
    const R::ReactionMechanism mechanism(
        schema, {{1, "micro gas one dissolution", {{G1, P::Dispersed, -1, "micro"}, {G1, P::Dissolved, 1}}},
                    {2, "large gas two dissolution", {{G2, P::Dispersed, -1, "large:7"}, {G2, P::Dissolved, 1}}}});
    auto initial = state(*schema, 0.0);
    initial.dispersedPopulations = {{"micro", {0.0, 0.0, 0.3, 0.1}}, {"large:7", {0.0, 0.0, 0.2, 0.5}}};
    R::ReactionSourceInput input;
    input.integratedExtents = {{1, 0.2}, {2, 0.4}};
    const auto result = R::advance_reactions(mechanism, initial, environment(), 1.0, input);
    const auto& raw = result.localReceipt.chemicalChanges;
    ASSERT_EQ(raw.size(), 4U);
    EXPECT_EQ(raw[0].species, G1);
    EXPECT_EQ(raw[0].phase, P::Dissolved);
    EXPECT_TRUE(raw[0].population.empty());
    EXPECT_EQ(raw[0].moles, static_cast<long double>(.2));
    EXPECT_EQ(raw[1].species, G2);
    EXPECT_EQ(raw[1].phase, P::Dissolved);
    EXPECT_EQ(raw[1].moles, static_cast<long double>(.4));
    EXPECT_EQ(raw[2].species, G1);
    EXPECT_EQ(raw[2].phase, P::Dispersed);
    EXPECT_EQ(raw[2].population, "micro");
    EXPECT_EQ(raw[2].moles, -static_cast<long double>(.2));
    EXPECT_EQ(raw[3].species, G2);
    EXPECT_EQ(raw[3].phase, P::Dispersed);
    EXPECT_EQ(raw[3].population, "large:7");
    EXPECT_EQ(raw[3].moles, -static_cast<long double>(.4));
    EXPECT_NEAR(result.state.amount(*schema, G1, P::Dispersed, "micro"), 0.1, 1e-16);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, P::Dispersed, "large:7"), 0.2);
    EXPECT_NEAR(result.state.phase_amount(*schema, G2, P::Dispersed), 0.2, 1e-16);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G2, P::Dissolved), 0.4);
    EXPECT_NEAR(static_cast<double>(mass(*schema, result.state) - mass(*schema, initial)), 0.0, 1e-17);
    input.integratedExtents = {{1, 0.4}}; // Other populations cannot lend their gas to micro.
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::runtime_error);
}

TEST(ReactionMechanismTest, ZeroLiquidVolumeHasZeroLiquidBasedProduction)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    auto context = environment();
    context.liquidVolume = 0.0;
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 17.0);
    const auto result = R::advance_reactions(mechanism, state(*schema, 0.0), context, 2.0, input);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], 0.0);
    EXPECT_DOUBLE_EQ(result.state.phase_amount(*schema, G1, R::SpeciesPhase::Dissolved), 0.0);
    context.bulkVolume = 0.0;
    input.rates = constant_rate(*schema, 17.0, R::SourceVolumeBasis::BulkVolume);
    EXPECT_NO_THROW((void) R::advance_reactions(mechanism, state(*schema, 0.0), context, 2.0, input));
}

TEST(ReactionMechanismTest, HeatOnlyAndReactionOnlyAreIndependent)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    R::ReactionSourceInput input;
    input.heat.rate = R::VolumetricHeatRate{10.0, R::SourceVolumeBasis::LiquidVolume};
    auto result = R::advance_reactions(mechanism, state(*schema), environment(), 0.5, input);
    EXPECT_DOUBLE_EQ(result.ledger.externalHeat, 1.0);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], 0.0);
    input.heat.rate->basis = R::SourceVolumeBasis::BulkVolume;
    result = R::advance_reactions(mechanism, state(*schema), environment(), 0.5, input);
    EXPECT_DOUBLE_EQ(result.ledger.externalHeat, 2.5);
    input = decay(*schema, 1.0);
    result = R::advance_reactions(mechanism, state(*schema), environment(), 0.5, input, {0.01});
    EXPECT_GT(result.ledger.reactionExtents[0], 0.0);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 0.0);
}

TEST(ReactionMechanismTest, FormationEnergyRejectsDuplicateReactionHeat)
{
    const auto schema = catalog();
    EXPECT_THROW(R::ReactionMechanism(schema, {split_reaction(1.0)}, R::EnergyConvention::IncludesFormationEnergy),
        std::invalid_argument);
    EXPECT_NO_THROW(R::ReactionMechanism(schema, {split_reaction()}, R::EnergyConvention::IncludesFormationEnergy));
}

TEST(ReactionMechanismTest, ExternalReservoirRequiresMassAndDebitsWithdrawals)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {});
    R::ReactionSourceInput input;
    input.externalTransfers = {{"supply", G1, R::SpeciesPhase::Resolved, 2.0, 0.004, -5.0, {1.0, 2.0, 3.0}}};
    const auto initial = state(*schema, 0.0);
    auto result = R::advance_reactions(mechanism, initial, environment(), 0.1, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, R::SpeciesPhase::Resolved), 2.0);
    EXPECT_DOUBLE_EQ(result.ledger.externalMass, 0.004);
    EXPECT_DOUBLE_EQ(result.ledger.transferEnergy, -5.0);
    EXPECT_DOUBLE_EQ(result.ledger.externalMomentum[1], 2.0);
    input.externalTransfers[0].mass = 0.0;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 0.1, input), std::invalid_argument);
    input.externalTransfers[0].moles = -2.0;
    input.externalTransfers[0].mass = -0.004;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 0.1, input), std::runtime_error);
    input.externalTransfers[0].reservoir.clear();
    EXPECT_THROW(
        (void) R::advance_reactions(mechanism, result.state, environment(), 0.1, input), std::invalid_argument);
}

TEST(ReactionMechanismTest, SchemaRejectsMassElementAndArbitrarySpeciesCreation)
{
    const auto original = catalog();
    auto species = original->species();
    species[1].molarMass = 0.005;
    auto schema = std::make_shared<const R::SpeciesCatalog>(species);
    EXPECT_THROW(R::ReactionMechanism(schema, {split_reaction()}), std::invalid_argument);
    species = original->species();
    species[1].elements["X"] = 3.0;
    schema = std::make_shared<const R::SpeciesCatalog>(species);
    EXPECT_THROW(R::ReactionMechanism(schema, {split_reaction()}), std::invalid_argument);
    EXPECT_THROW(R::ReactionMechanism(original, {{1, "creation", {{G1, R::SpeciesPhase::Dissolved, 1}}}}),
        std::invalid_argument);
    species = original->species();
    species[1].id = A;
    EXPECT_THROW((void) R::SpeciesCatalog(species), std::invalid_argument);
    species = original->species();
    species[0].molarMass = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) R::SpeciesCatalog(species), std::invalid_argument);
}

TEST(ReactionMechanismTest, NonfiniteSourceSchemaAndMixedFormsRejectAtomically)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    auto initial = state(*schema);
    auto input = decay(*schema, 1.0);
    input.rates->schemaRevision = 18;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    input.rates = constant_rate(*schema, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    input.rates = constant_rate(*schema, 0.0);
    input.integratedExtents = {{split, 0.1}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    input.rates.reset();
    input.integratedExtents[0].moles = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    input = {};
    input.heat.rate = R::VolumetricHeatRate{1.0};
    input.heat.integratedEnergy = 1.0;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    initial.schemaRevision = 18;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0), std::invalid_argument);
}

TEST(ReactionMechanismTest, CompensatedEventLedgersKeepTransfersSmallerThanInventoryUlp)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    R::ReactionSourceInput input;
    input.integratedExtents = {{split, 1.0}};
    input.externalTransfers = {{"energy 1", G1, R::SpeciesPhase::Resolved, 0.0, 0.0, 1e18},
        {"small transfer", G1, R::SpeciesPhase::Resolved, 0.0, 0.0, 1.0},
        {"energy 2", G1, R::SpeciesPhase::Resolved, 0.0, 0.0, -1e18}};
    const auto result = R::advance_reactions(mechanism, state(*schema, 1e20), environment(), 1.0, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 1e20);
    EXPECT_DOUBLE_EQ(result.ledger.chemicalChange.amount(*schema, A, R::SpeciesPhase::Liquid), -1.0);
    EXPECT_DOUBLE_EQ(result.ledger.transferEnergy, 1.0);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, R::SpeciesPhase::Dissolved), 1.0);
}

TEST(ReactionMechanismTest, DeterministicReplayPreservesStateAndEventHistory)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction(13.0)});
    const auto initial = state(*schema);
    const auto input = decay(*schema, 0.7);
    const auto first = R::advance_reactions(mechanism, initial, environment(), 1.0, input, {0.03});
    const auto second = R::advance_reactions(mechanism, initial, environment(), 1.0, input, {0.03});
    EXPECT_EQ(first.state.amounts, second.state.amounts);
    EXPECT_EQ(first.ledger.chemicalChange.amounts, second.ledger.chemicalChange.amounts);
    EXPECT_EQ(first.ledger.reactionExtents, second.ledger.reactionExtents);
    EXPECT_DOUBLE_EQ(first.ledger.reactionHeat, second.ledger.reactionHeat);
    EXPECT_EQ(first.ledger.substeps, second.ledger.substeps);
}

TEST(ReactionMechanismTest, SameRevisionForeignOrReorderedSchemasRejectState)
{
    const auto schema = catalog();
    const auto initial = state(*schema);
    auto species = schema->species();
    std::swap(species[0], species[1]);
    const auto reordered = std::make_shared<const R::SpeciesCatalog>(species, schema->revision());
    EXPECT_NE(reordered->fingerprint(), schema->fingerprint());
    EXPECT_THROW(initial.validate(*reordered), std::invalid_argument);
    const R::ReactionMechanism mechanism(reordered, {split_reaction()});
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0), std::invalid_argument);
    species = schema->species();
    species[0].name = "foreign donor";
    const R::SpeciesCatalog foreign(species, schema->revision());
    EXPECT_THROW(initial.validate(foreign), std::invalid_argument);
    const R::SpeciesCatalog equivalent(schema->species(), schema->revision());
    EXPECT_EQ(equivalent.fingerprint(), schema->fingerprint());
    EXPECT_NO_THROW(initial.validate(equivalent));
}

TEST(ReactionMechanismTest, InvalidEnumInputsRejectBeforeStorageAccess)
{
    const auto schema = catalog();
    auto initial = state(*schema);
    EXPECT_THROW((void) initial.amount(*schema, A, static_cast<R::SpeciesPhase>(999)), std::invalid_argument);
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 1.0, static_cast<R::SourceVolumeBasis>(999));
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
}
TEST(ReactionMechanismTest, ProviderRejectsSameRevisionForeignOrderedSchema)
{
    const auto original = catalog();
    auto species = original->species();
    std::swap(species[0], species[1]);
    const auto reordered = std::make_shared<const R::SpeciesCatalog>(species, original->revision());
    const R::ReactionMechanism mechanism(reordered, {split_reaction()});
    const auto input = decay(*original, 1.0);
    EXPECT_THROW(
        (void) R::advance_reactions(mechanism, state(*reordered), environment(), 1.0, input), std::invalid_argument);
}

TEST(ReactionMechanismTest, AbsoluteIntervalAndEveryCallbackStageMustAdvanceRepresentableTime)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    auto context = environment();
    context.time = 1e20;
    EXPECT_THROW((void) R::advance_reactions(mechanism, state(*schema), context, 1.0), std::invalid_argument);
    EXPECT_THROW((void) R::advance_reactions(mechanism, state(*schema), context, 32768.0, decay(*schema, 1.0), {1.0}),
        std::invalid_argument);
    EXPECT_NO_THROW((void) R::advance_reactions(mechanism, state(*schema), context, 0.0));
}

TEST(ReactionMechanismTest, InitialOccupiedSpeciesNeedTheirPhysicalCarrierVolume)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    auto context = environment();
    context.liquidVolume = 0.0;
    EXPECT_THROW((void) R::advance_reactions(mechanism, state(*schema), context, 1.0), std::invalid_argument);
    auto initial = state(*schema, 0.0);
    initial.set_amount(*schema, G1, R::SpeciesPhase::Dissolved, 1.0);
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, context, 1.0), std::invalid_argument);
    initial.set_amount(*schema, G1, R::SpeciesPhase::Dissolved, 0.0);
    initial.set_amount(*schema, G1, R::SpeciesPhase::Resolved, 1.0);
    context.bulkVolume = 0.0;
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, context, 1.0), std::invalid_argument);
}

TEST(ReactionMechanismTest, PhaseTransferCanCreatePreviouslyAbsentLiquidForOuterEOSRecovery)
{
    const auto schema = catalog();
    using P = R::SpeciesPhase;
    const R::ReactionMechanism mechanism(schema, {{1, "condensation", {{G1, P::Resolved, -1}, {G1, P::Dissolved, 1}}}});
    auto context = environment();
    context.liquidVolume = 0.0;
    auto initial = state(*schema, 0.0);
    initial.set_amount(*schema, G1, P::Resolved, 1.0);
    R::ReactionSourceInput input;
    input.integratedExtents = {{1, 0.5}};
    const auto result = R::advance_reactions(mechanism, initial, context, 1.0, input);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, P::Dissolved), 0.5);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, P::Resolved), 0.5);
}

TEST(ReactionMechanismTest, NonrepresentableProductsRejectInsteadOfDiscardingMassByUnderflow)
{
    const auto schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {A, "A", 1.0, {{"X", 1}}, {R::SpeciesPhase::Liquid}}, {B, "B", 1.0, {{"X", 1}}, {R::SpeciesPhase::Liquid}},
        {G1, "G", 1.0, {{"X", 1}}, {R::SpeciesPhase::Dissolved}}});
    const R::ReactionMechanism mechanism(
        schema, {{1, "half-size products",
                    {{A, R::SpeciesPhase::Liquid, -1}, {B, R::SpeciesPhase::Liquid, 0.5},
                        {G1, R::SpeciesPhase::Dissolved, 0.5}}}});
    const auto initial = state(*schema, std::numeric_limits<double>::denorm_min());
    R::ReactionSourceInput input;
    input.integratedExtents = {{1, std::numeric_limits<double>::denorm_min()}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, environment(), 1.0, input), std::invalid_argument);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, A, R::SpeciesPhase::Liquid), std::numeric_limits<double>::denorm_min());
}
TEST(ReactionMechanismTest, RepeatedSubUlpConsumptionAccumulatesIntoRepresentableStoredChange)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 5.0); // 5 mol/(m3 s) * .2 m3 = 1 mol/s.
    const auto initial = state(*schema, 1e20);
    const auto result = R::advance_reactions(mechanism, initial, environment(), 10000.0, input, {1.0, 10000});
    EXPECT_DOUBLE_EQ(result.ledger.chemicalChange.amount(*schema, A, R::SpeciesPhase::Liquid), -10000.0);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, A, R::SpeciesPhase::Liquid),
        static_cast<double>(static_cast<long double>(1e20) - 10000.0L));
    EXPECT_LT(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), 1e20);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G1, R::SpeciesPhase::Dissolved), 10000.0);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, G2, R::SpeciesPhase::Dissolved), 10000.0);
}
TEST(ReactionMechanismTest, PredictorEOSVolumeDefersOnlyOccupancyToTheFlowOwner)
{
    const auto schema = catalog();
    const R::ReactionMechanism mechanism(schema, {split_reaction()});
    const auto initial = state(*schema, .2);
    auto context = environment();
    context.liquidVolume = .6;
    R::ReactionSourceInput input;
    input.rates = constant_rate(*schema, 2.);
    input.heat.rate = R::VolumetricHeatRate{2., R::SourceVolumeBasis::BulkVolume};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, context, .1, input), std::invalid_argument);
    context.volumeClosure = R::ReactionVolumeClosure::ThermodynamicPredictor;
    input.rates->validate = [](const R::ReactionContext& value)
    {
        EXPECT_DOUBLE_EQ(value.environment.liquidVolume, .6);
        EXPECT_DOUBLE_EQ(value.environment.bulkVolume, .5);
        EXPECT_EQ(value.environment.volumeClosure, R::ReactionVolumeClosure::ThermodynamicPredictor);
    };
    const auto result = R::advance_reactions(mechanism, initial, context, .1, input);
    EXPECT_NEAR(result.ledger.reactionExtents[0], .12, 1e-16);
    EXPECT_NEAR(result.state.amount(*schema, A, R::SpeciesPhase::Liquid), .08, 1e-16);
    EXPECT_DOUBLE_EQ(result.ledger.externalHeat, .1);
    for (int invalid = 0; invalid < 4; ++invalid)
    {
        auto bad = context;
        if (invalid == 0)
            bad.volumeClosure = static_cast<R::ReactionVolumeClosure>(71);
        if (invalid == 1)
            bad.bulkVolume = 0;
        if (invalid == 2)
            bad.liquidVolume = -1;
        if (invalid == 3)
            bad.liquidVolume = std::numeric_limits<double>::infinity();
        EXPECT_THROW((void) R::advance_reactions(mechanism, initial, bad, .1, input), std::invalid_argument);
    }
    R::ReactionSourceInput overdraw;
    overdraw.integratedExtents = {{split, .3}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, context, .1, overdraw), std::exception);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, A, R::SpeciesPhase::Liquid), .2);
}
} // namespace
TEST(ReactionDirectionalHistoryTest, AcceptedStagesRetainGrossReversalWithoutChangingNetInventoryOrHeat)
{
    namespace R = SimpleFluid::Reacting;
    using P = R::SpeciesPhase;
    const auto schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {5, "left", 0.01, {{"X", 1}}, {P::Liquid}}, {17, "right", 0.01, {{"X", 1}}, {P::Liquid}}});
    const R::ReactionMechanism mechanism(schema, {{61, "reversible", {{5, P::Liquid, -1}, {17, P::Liquid, 1}}, 10.0}});
    R::CellSpeciesState initial(*schema);
    initial.set_amount(*schema, 5, P::Liquid, 2.0);
    initial.set_amount(*schema, 17, P::Liquid, 2.0);
    R::ReactionSourceInput input;
    R::ReactionRateProvider provider;
    provider.identity = "linear sign change";
    provider.schemaRevision = schema->revision();
    provider.schemaFingerprint = schema->fingerprint();
    provider.validityDescription = "manufactured finite time";
    provider.parameterUnits = "mol/(m3 s)";
    provider.validate = [](const R::ReactionContext&) {};
    provider.evaluate = [](const R::ReactionContext& context)
    {
        return std::vector<R::ReactionRate>{
            {61, 2.0 * context.environment.time - 1.0, R::SourceVolumeBasis::BulkVolume}};
    };
    input.rates = provider;
    const R::ReactionEnvironment environment{0.0, 300.0, 100000.0, 1.0, 1.0};
    for (const double maximumStep : {1.0, 0.25})
    {
        R::ReactionAdvanceOptions options;
        options.maximumSubstep = maximumStep;
        const auto result = R::advance_reactions(mechanism, initial, environment, 1.0, input, options);
        EXPECT_DOUBLE_EQ(result.state.amount(*schema, 5, P::Liquid), 2.0);
        EXPECT_DOUBLE_EQ(result.state.amount(*schema, 17, P::Liquid), 2.0);
        ASSERT_EQ(result.ledger.forwardReactionExtents.size(), 1);
        ASSERT_EQ(result.ledger.reverseReactionExtents.size(), 1);
        EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], 0.0);
        const double expectedDirection = maximumStep == 1.0 ? 0.5 : 0.25;
        EXPECT_DOUBLE_EQ(result.ledger.forwardReactionExtents[0], expectedDirection);
        EXPECT_DOUBLE_EQ(result.ledger.reverseReactionExtents[0], expectedDirection);
        EXPECT_DOUBLE_EQ(result.ledger.reactionHeat, 0.0);
    }
    input.rates.reset();
    input.integratedExtents = {{61, -0.125}};
    const auto reverse = R::advance_reactions(mechanism, initial, environment, 1.0, input);
    EXPECT_DOUBLE_EQ(reverse.ledger.forwardReactionExtents[0], 0.0);
    EXPECT_DOUBLE_EQ(reverse.ledger.reverseReactionExtents[0], 0.125);
    EXPECT_DOUBLE_EQ(reverse.ledger.reactionExtents[0], -0.125);
    EXPECT_DOUBLE_EQ(reverse.ledger.reactionHeat, -1.25);
}

TEST(ReactionMechanismTest, LocalAcceptedReceiptPreservesHeatAndMaterialLostByRoundedExtentReports)
{
    namespace R = SimpleFluid::Reacting;
    using P = R::SpeciesPhase;
    const auto schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {11, "left", .1, {{"X", 1}}, {P::Liquid}},
        {29, "right", .1, {{"X", 1}}, {P::Liquid}}});
    const double large = std::ldexp(1., 60), quantum = std::ldexp(1., -60);
    const R::ReactionMechanism mechanism(schema,
        {{41, "forward", {{11, P::Liquid, -quantum}, {29, P::Liquid, quantum}}, 1.},
            {83, "opposing", {{29, P::Liquid, -quantum}, {11, P::Liquid, quantum}}, -1.}});
    R::CellSpeciesState initial(*schema);
    initial.set_amount(*schema, 11, P::Liquid, 2.);
    initial.set_amount(*schema, 29, P::Liquid, 2.);
    R::ReactionRateProvider provider;
    provider.identity = "two accepted intervals with one small surviving event";
    provider.schemaRevision = schema->revision();
    provider.schemaFingerprint = schema->fingerprint();
    provider.validityDescription = "manufactured t in [0,1], finite donors";
    provider.parameterUnits = "mol/(m3 s)";
    provider.validate = [](const R::ReactionContext&) {};
    provider.evaluate = [large](const R::ReactionContext& context)
    {
        const double time = context.environment.time;
        // The first .5-s SSPRK2 interval accepts 2^60 for each reaction.
        // The second accepts exactly one additional forward extent, without a large donor stock.
        return std::vector<R::ReactionRate>{
            {41, time == 0 ? 4 * large : (time == 1 ? 4. : 0.), R::SourceVolumeBasis::BulkVolume},
            {83, time == 0 ? 4 * large : 0., R::SourceVolumeBasis::BulkVolume}};
    };
    R::ReactionSourceInput input;
    input.rates = provider;
    const R::ReactionEnvironment environment{0., 300., 100000., 1., 1.};
    const auto result = R::advance_reactions(mechanism, initial, environment, 1., input, {.5, 4});
    ASSERT_EQ(result.ledger.substeps, 2U);
    ASSERT_EQ(result.ledger.reactionExtents.size(), 2U);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0], large);
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[1], large);
    // Reconstructing from those reporting views invents zero heat and zero chemical change.
    EXPECT_DOUBLE_EQ(result.ledger.reactionExtents[0] - result.ledger.reactionExtents[1], 0.);
    EXPECT_DOUBLE_EQ(quantum * (result.ledger.reactionExtents[0] - result.ledger.reactionExtents[1]), 0.);
    EXPECT_DOUBLE_EQ(result.ledger.reactionHeat, 1.);
    const auto& receipt = result.localReceipt;
    EXPECT_EQ(receipt.schemaFingerprint, schema->fingerprint());
    ASSERT_EQ(receipt.reactionHeatContributions.size(), 2U);
    const long double actual_heat = receipt.reactionHeatContributions[0] + receipt.reactionHeatContributions[1];
    EXPECT_EQ(actual_heat, 1.L);
    EXPECT_EQ(receipt.externalHeat, 0.L);
    ASSERT_EQ(receipt.chemicalChanges.size(), 2U);
    EXPECT_EQ(receipt.chemicalChanges[0].species, 11U);
    EXPECT_EQ(receipt.chemicalChanges[0].phase, P::Liquid);
    EXPECT_TRUE(receipt.chemicalChanges[0].population.empty());
    EXPECT_EQ(receipt.chemicalChanges[0].moles, -static_cast<long double>(quantum));
    EXPECT_EQ(receipt.chemicalChanges[1].species, 29U);
    EXPECT_EQ(receipt.chemicalChanges[1].moles, static_cast<long double>(quantum));
    EXPECT_DOUBLE_EQ(result.ledger.chemicalChange.amount(*schema, 11, P::Liquid), -quantum);
    EXPECT_DOUBLE_EQ(result.ledger.chemicalChange.amount(*schema, 29, P::Liquid), quantum);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, 11, P::Liquid), 2.);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, 29, P::Liquid), 2.);
}

TEST(ReactionMechanismTest, LocalReceiptRetainsVolumetricHeatBeforeItsDoubleReportingView)
{
    namespace R = SimpleFluid::Reacting;
    const auto schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {7, "ordinary carrier", .1, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
    const R::ReactionMechanism mechanism(schema, {}, R::EnergyConvention::IncludesFormationEnergy);
    R::CellSpeciesState initial(*schema);
    const double factor = std::ldexp(1., 30);
    R::ReactionSourceInput input;
    input.heat.rate = R::VolumetricHeatRate{factor + 1., R::SourceVolumeBasis::BulkVolume};
    const R::ReactionEnvironment environment{0., 300., 100000., 0., factor - 1.};
    const auto result = R::advance_reactions(mechanism, initial, environment, 1., input);
    EXPECT_DOUBLE_EQ(result.ledger.externalHeat, std::ldexp(1., 60));
    EXPECT_EQ(result.localReceipt.externalHeat, std::ldexp(1.L, 60) - 1.L);
    EXPECT_TRUE(result.localReceipt.chemicalChanges.empty());
    EXPECT_TRUE(result.localReceipt.reactionHeatContributions.empty());
    EXPECT_DOUBLE_EQ(result.ledger.reactionHeat, 0.);
}

namespace
{
constexpr R::SpeciesId margin_donor = 101, margin_sink = 103, margin_feed = 107;
std::shared_ptr<const R::SpeciesCatalog> margin_schema()
{
    using P = R::SpeciesPhase;
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{margin_donor, "finite donor", 1., {{"X", 1}}, {P::Liquid}},
            {margin_sink, "finite product", 1., {{"X", 1}}, {P::Liquid}},
            {margin_feed, "explicit incoming feed", 1., {{"X", 1}}, {P::Liquid}}});
}
std::vector<R::ReactionDefinition> margin_reactions()
{
    using P = R::SpeciesPhase;
    return {{211, "large donor demand", {{margin_donor, P::Liquid, -1}, {margin_sink, P::Liquid, 1}}, 0},
        {223, "independent one-mole donor demand", {{margin_donor, P::Liquid, -1}, {margin_sink, P::Liquid, 1}}, 0},
        {227, "simultaneous donor credit", {{margin_feed, P::Liquid, -1}, {margin_donor, P::Liquid, 1}}, 0}};
}
R::CellSpeciesState margin_state(const R::SpeciesCatalog& schema, double donor = 1e20, double feed = 1e20)
{
    R::CellSpeciesState result(schema);
    result.set_amount(schema, margin_donor, R::SpeciesPhase::Liquid, donor);
    result.set_amount(schema, margin_feed, R::SpeciesPhase::Liquid, feed);
    return result;
}
R::ReactionRateProvider margin_provider(const R::SpeciesCatalog& schema)
{
    auto provider = constant_rate(schema, 0, R::SourceVolumeBasis::BulkVolume);
    provider.identity = "raw stage donor margin";
    return provider;
}
R::ExternalSpeciesTransfer margin_transfer(const char* reservoir, double moles)
{
    return {reservoir, margin_donor, R::SpeciesPhase::Liquid, moles, moles, 0., {}, {}};
}
const R::ReactionEnvironment margin_environment{0., 300., 100000., 1., 1.};
} // namespace

TEST(ReactionMechanismTest, GrossChemicalDemandCannotBorrowHugeSimultaneousCreditOrLoseOneMole)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, margin_reactions());
    const auto initial = margin_state(*schema);
    R::ReactionSourceInput input;
    input.integratedExtents = {{211, 1e20}, {223, 1}, {227, 1e20}};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_feed, R::SpeciesPhase::Liquid), 1e20);
}

TEST(ReactionMechanismTest, GrossExternalDemandCannotBorrowHugeSimultaneousCreditOrLoseOneMole)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, {});
    const auto initial = margin_state(*schema);
    R::ReactionSourceInput input;
    input.externalTransfers = {margin_transfer("large withdrawal", -1e20),
        margin_transfer("independent withdrawal", -1), margin_transfer("explicit supply", 1e20)};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
    std::reverse(input.externalTransfers.begin(), input.externalTransfers.end());
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input), std::runtime_error);
}

TEST(ReactionMechanismTest, ChemicalAndExternalRawDemandsShareOneFiniteMargin)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, margin_reactions());
    const auto initial = margin_state(*schema);
    R::ReactionSourceInput input;
    input.integratedExtents = {{211, 1e20}};
    input.externalTransfers = {margin_transfer("one extra withdrawal", -1), margin_transfer("incoming credit", 1e20)};
    EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
}

TEST(ReactionMechanismTest, EachSSPRK2EulerStageChecksRawGrossDemandAgainstItsOwnStock)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, margin_reactions());
    const auto initial = margin_state(*schema);
    for (bool second_only : {false, true})
    {
        auto provider = margin_provider(*schema);
        provider.evaluate = [second_only](const R::ReactionContext& context)
        {
            const bool active = !second_only || context.environment.time > 0.;
            return std::vector<R::ReactionRate>{{211, active ? 1e20 : 0., R::SourceVolumeBasis::BulkVolume},
                {223, active ? 1. : 0., R::SourceVolumeBasis::BulkVolume},
                {227, active ? 1e20 : 0., R::SourceVolumeBasis::BulkVolume}};
        };
        R::ReactionSourceInput input;
        input.rates = std::move(provider);
        EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input, {1., 4, 0}),
            std::runtime_error); // No retry may replace the deliberately overdrawn Euler stage.
        EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
    }
}

TEST(ReactionMechanismTest, FullDepletionRetainsSeparateFiniteExternalAndChemicalCredits)
{
    const auto schema = margin_schema();
    const auto initial = margin_state(*schema, 1e20, 1);
    const R::ReactionMechanism external_only(schema, {});
    R::ReactionSourceInput external;
    external.externalTransfers = {margin_transfer("complete withdrawal", -1e20), margin_transfer("finite credit", 1)};
    const auto supplied = R::advance_reactions(external_only, initial, margin_environment, 1., external);
    EXPECT_DOUBLE_EQ(supplied.state.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1.);
    ASSERT_EQ(supplied.ledger.externalTransfers.size(), 2U);
    EXPECT_DOUBLE_EQ(supplied.ledger.externalTransfers[1].moles, 1.);
    const R::ReactionMechanism chemical(schema, margin_reactions());
    R::ReactionSourceInput reactions;
    reactions.integratedExtents = {{211, 1e20}, {227, 1}};
    const auto created = R::advance_reactions(chemical, initial, margin_environment, 1., reactions);
    EXPECT_DOUBLE_EQ(created.state.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1.);
    EXPECT_DOUBLE_EQ(created.state.amount(*schema, margin_feed, R::SpeciesPhase::Liquid), 0.);
    EXPECT_DOUBLE_EQ(created.ledger.reactionExtents[2], 1.);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
}

TEST(ReactionMechanismTest, SSPRK2SecondStageMayUseItsActualFirstStageProduct)
{
    using P = R::SpeciesPhase;
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(
        schema, {{241, "first Euler stage", {{margin_donor, P::Liquid, -1}, {margin_sink, P::Liquid, 1}}, 0},
                    {251, "second Euler stage", {{margin_sink, P::Liquid, -1}, {margin_feed, P::Liquid, 1}}, 0}});
    auto provider = margin_provider(*schema);
    provider.evaluate = [](const R::ReactionContext& context)
    {
        const bool first = context.environment.time == 0.;
        return std::vector<R::ReactionRate>{{241, first ? 1. : 0., R::SourceVolumeBasis::BulkVolume},
            {251, first ? 0. : 1., R::SourceVolumeBasis::BulkVolume}};
    };
    R::ReactionSourceInput input;
    input.rates = std::move(provider);
    const auto result =
        R::advance_reactions(mechanism, margin_state(*schema, 1., 0.), margin_environment, 1., input, {1., 4, 0});
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, margin_donor, P::Liquid), .5);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, margin_sink, P::Liquid), 0.);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, margin_feed, P::Liquid), .5);
}

TEST(ReactionMechanismTest, AcceptedSmallDepletionRemainsInTheNextEulerDonorMargin)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, margin_reactions());
    const auto initial = margin_state(*schema, 1e20, 0.);
    auto provider = margin_provider(*schema);
    provider.evaluate = [](const R::ReactionContext& context)
    {
        const double t = context.environment.time;
        return std::vector<R::ReactionRate>{
            {211, t == 0. ? 4. : (t == 1. ? 2e20 : 0.), R::SourceVolumeBasis::BulkVolume}};
    };
    R::ReactionSourceInput input;
    input.rates = std::move(provider);
    // First accepted half-step consumes one mole. The next Euler stage may not spend the rounded 1e20 view.
    EXPECT_THROW(
        (void) R::advance_reactions(mechanism, initial, margin_environment, 1., input, {.5, 4, 0}), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), 1e20);
}

TEST(ReactionMechanismTest, SSPRK2PrivateHalfEventsDoNotPrematurelyNarrowARepresentableFinalTransfer)
{
    const auto schema = margin_schema();
    const R::ReactionMechanism mechanism(schema, margin_reactions());
    const double quantum = std::numeric_limits<double>::denorm_min();
    auto provider = margin_provider(*schema);
    provider.evaluate = [quantum](const R::ReactionContext&)
    { return std::vector<R::ReactionRate>{{211, quantum, R::SourceVolumeBasis::BulkVolume}}; };
    R::ReactionSourceInput input;
    input.rates = std::move(provider);
    const auto initial = margin_state(*schema, 2 * quantum, 0.);
    if constexpr (std::numeric_limits<long double>::min_exponent < std::numeric_limits<double>::min_exponent)
    {
        const auto result = R::advance_reactions(mechanism, initial, margin_environment, 1., input, {1., 4, 0});
        EXPECT_EQ(result.state.amount(*schema, margin_donor, R::SpeciesPhase::Liquid), quantum);
        EXPECT_EQ(result.state.amount(*schema, margin_sink, R::SpeciesPhase::Liquid), quantum);
        EXPECT_EQ(result.ledger.reactionExtents[0], quantum);
    }
    else
    {
        // Without exponent headroom the half-event cannot be retained: reject rather than silently erase it.
        EXPECT_THROW((void) R::advance_reactions(mechanism, initial, margin_environment, 1., input, {1., 4, 0}),
            std::invalid_argument);
    }
}
