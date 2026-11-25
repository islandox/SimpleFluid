/** @file testQbmmEvents.cc
 * @brief Constructive mass-coordinate events, gross receipts and finite pre-substep donor reservations.
 */
#include "equations/QbmmLawInputs.hh"
#include "equations/QbmmPopulationEvents.hh"
#include "equations/QbmmPopulationLawAdapters.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace
{
namespace R = SimpleFluid::Reacting;
using P = R::SpeciesPhase;
using Status = R::QbmmEventStatus;
using Kind = R::QbmmEventKind;

struct Binding
{
    enum class Role
    {
        DispersedGas
    } role;
    std::string identity;
};

class QbmmEventsTest : public ::testing::Test
{
protected:
    const std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{6, "A", 1, {{"A", 1}}, {P::Dispersed, P::Dissolved}},
            {41, "B", 2, {{"B", 1}}, {P::Dispersed, P::Dissolved}}});
    const R::QbmmPopulationSchema schema{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, 0.25}, {41, 0.75}}};

    // These fixtures use binary-exact material arithmetic in the A,B catalog order.
    static std::vector<double> moles(double mass) { return {mass / 4, 3 * mass / 8}; }

    R::CellSpeciesState stock(double mass, double dissolved = 0) const
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"unrelated", {1, 2}}, {"small", moles(mass)}};
        const auto donor = moles(dissolved);
        state.set_amount(*catalog, 6, P::Dissolved, donor[0]);
        state.set_amount(*catalog, 41, P::Dissolved, donor[1]);
        return state;
    }

    static R::QbmmMaterialTransfer internal(double mass) { return {R::QbmmPhaseExchange{P::Dissolved}, moles(mass)}; }

    static R::QbmmMaterialTransfer external(double mass)
    {
        return {R::QbmmReservoirExchange{"feed", 2.5, {-1., 0., 3.}}, moles(mass)};
    }

    R::QbmmEventBatch batch(std::initializer_list<R::QbmmEventOperation> operations) const
    {
        R::QbmmEventBatch result{schema.fingerprint(), 19, 0.125, {}};
        for (const auto& operation : operations)
            result.events.push_back({"test-law", 3, operation});
        return result;
    }

    static R::QbmmCellView cell_view(const R::CellSpeciesState& state, const double& count,
        const R::QbmmShapeState& shape, std::uint64_t family_stamp)
    {
        const auto& row = state.dispersedPopulations.at(1);
        return {row.id, state.schemaFingerprint, row.amounts, count, shape, family_stamp};
    }

    R::QbmmPopulationAdvance advance(const R::CellSpeciesState& state, const double& count,
        const R::QbmmShapeState& shape, const R::QbmmEventBatch& changes) const
    {
        return R::advance_qbmm_population(schema, state, cell_view(state, count, shape, schema.fingerprint()), changes);
    }

    R::QbmmPopulationAdvance advance(
        const R::QbmmPopulationEndpoint& predecessor, const R::QbmmEventBatch& changes) const
    {
        const auto view =
            cell_view(predecessor.species, predecessor.count, predecessor.shape, predecessor.family_fingerprint);
        return R::advance_qbmm_population(schema, predecessor.species, view, changes);
    }

    void expect_block(const R::QbmmPopulationEndpoint& endpoint, R::QbmmMomentBlock expected) const
    {
        ASSERT_EQ(endpoint.species.dispersedPopulations.size(), 2u);
        EXPECT_EQ(endpoint.species.dispersedPopulations[0].amounts, (std::vector<double>{1, 2}));
        const auto& material = endpoint.species.dispersedPopulations[1].amounts;
        const auto amounts = moles(expected[1]);
        ASSERT_EQ(material.size(), amounts.size());
        for (unsigned i = 0; i < amounts.size(); ++i)
            EXPECT_NEAR(material[i], amounts[i], amounts[i] * 1e-13);
        EXPECT_EQ(endpoint.family_fingerprint, schema.fingerprint());
        const auto block = R::qbmm_cell_moments(
            schema, cell_view(endpoint.species, endpoint.count, endpoint.shape, endpoint.family_fingerprint));
        ASSERT_TRUE(endpoint.quadrature.valid());
        for (unsigned k = 0; k < block.size(); ++k)
        {
            EXPECT_NEAR(block[k], expected[k], expected[k] * 1e-13);
            long double reconstructed = 0;
            for (unsigned i = 0; i < endpoint.quadrature.active_atoms; ++i)
                reconstructed += endpoint.quadrature.atoms[i].count *
                                 std::pow(static_cast<long double>(endpoint.quadrature.atoms[i].mass_over_scale), k);
            EXPECT_NEAR(reconstructed, expected[k], expected[k] * 1e-13);
        }
    }

    static void rejected(const R::QbmmPopulationAdvance& result, Status status)
    {
        EXPECT_EQ(result.status, status) << result.diagnostic;
        EXPECT_FALSE(result.endpoint);
        EXPECT_FALSE(result.diagnostic.empty());
    }

    static R::QbmmQuadrature rate_quadrature()
    {
        // Independently specified positive measure: four mass-two and two mass-eight bubbles.
        R::QbmmQuadrature result;
        result.status = R::InversionStatus::TwoNodes;
        result.active_atoms = 2;
        result.atoms = {R::QbmmAtom{4, 2}, R::QbmmAtom{2, 8}};
        return result;
    }

    R::QbmmLawInputs rate_inputs(std::optional<double> dt = .25, double volume = 2, double mass_scale = 1,
        R::QbmmQuadrature quadrature = rate_quadrature(), const R::QbmmPopulationSchema* bound_family = nullptr) const
    {
        const auto& family = bound_family ? *bound_family : schema;
        const bool absent = quadrature.status == R::InversionStatus::Absent;
        return {quadrature, std::nullopt, mass_scale, volume, 100000,
            absent ? std::nullopt : std::optional<double>(300), dt, absent ? 0. : .25, .5,
            absent ? std::nullopt : std::optional<std::vector<double>>(std::vector<double>{7, 11}),
            std::vector<double>{13, 17}, 310, 999, family.fingerprint(), family.catalog().fingerprint(),
            family.population()};
    }

    R::QbmmEventBatch rate_batch(std::vector<R::QbmmEventOperation> operations) const
    {
        auto result = batch({});
        result.duration = .25;
        for (auto& operation : operations)
            result.events.push_back({"supplied-SI-rate", 1, std::move(operation)});
        return result;
    }

    static R::QbmmLawMetadata slip_metadata(R::QbmmApplicabilityGate gate = R::QbmmApplicabilityGate::RejectOutside)
    {
        return {"supplied-slip", 2, R::QbmmLawKind::PhaseSlip, "common family drift relative to carrier, m/s", "m/s",
            gate, true, {{R::QbmmRangeVariable::SlipSpeedMPerS, 5, 5, std::nullopt}}};
    }
};

TEST_F(QbmmEventsTest, BirthRateUsesBulkVolumeActualDurationAndExplicitMaterialSpectrum)
{
    R::QbmmQuadrature absent;
    absent.status = R::InversionStatus::Absent;
    const auto inputs = rate_inputs(.25, 2, 1, absent);
    const std::array births{R::QbmmBirthRate{3, 2, R::QbmmPhaseExchange{P::Dissolved}},
        R::QbmmBirthRate{0, 19, R::QbmmPhaseExchange{P::Dissolved}},
        R::QbmmBirthRate{2, 8, R::QbmmPhaseExchange{P::Dissolved}}};
    const auto operations = R::qbmm_birth_from_rate(schema, inputs, births);
    ASSERT_EQ(operations.size(), 1u);
    const auto& birth = std::get<R::QbmmBirth>(operations.front());
    ASSERT_EQ(birth.atoms.size(), 2u);
    EXPECT_EQ(birth.atoms[0].atom.count, 1.5);
    EXPECT_EQ(birth.atoms[0].atom.mass_over_scale, 2);
    EXPECT_EQ(birth.atoms[0].transfer.moles, (std::vector<double>{.75, 1.125}));
    EXPECT_EQ(birth.atoms[1].atom.count, 1);
    EXPECT_EQ(birth.atoms[1].atom.mass_over_scale, 8);
    EXPECT_EQ(birth.atoms[1].transfer.moles, (std::vector<double>{2, 3}));
    const auto result = advance(stock(0, 11), 0, {}, rate_batch(operations));
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    expect_block(*result.endpoint, {2.5, 11, 70, 524});
    EXPECT_EQ(result.endpoint->species.amount(*catalog, 6, P::Dissolved), 0);
    EXPECT_EQ(result.endpoint->species.amount(*catalog, 41, P::Dissolved), 0);
    EXPECT_TRUE(R::qbmm_birth_from_rate(schema, rate_inputs(), {}).empty());
}

TEST_F(QbmmEventsTest, BirthRateRejectsInvalidMassRateScaleOverflowAndUnderflow)
{
    for (const double bad : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        const std::array births{R::QbmmBirthRate{bad, 2, R::QbmmPhaseExchange{P::Dissolved}}};
        EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, rate_inputs(), births), std::invalid_argument);
    }
    for (const double bad : {0., -1., std::numeric_limits<double>::infinity()})
    {
        const std::array births{R::QbmmBirthRate{1, bad, R::QbmmPhaseExchange{P::Dissolved}}};
        EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, rate_inputs(), births), std::invalid_argument);
    }
    const std::array valid{R::QbmmBirthRate{1, 2, R::QbmmPhaseExchange{P::Dissolved}}};
    EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, rate_inputs(.25, 2, 2), valid), std::invalid_argument);
    const std::array huge{R::QbmmBirthRate{std::numeric_limits<double>::max(), 2, R::QbmmPhaseExchange{P::Dissolved}}};
    EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, rate_inputs(1, 4), huge), std::invalid_argument);
    const std::array tiny{
        R::QbmmBirthRate{std::numeric_limits<double>::denorm_min(), 2, R::QbmmPhaseExchange{P::Dissolved}}};
    EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, rate_inputs(.25, .5), tiny), std::invalid_argument);
}

TEST_F(QbmmEventsTest, CoalescenceKernelUsesUnorderedExpectedPairsAndConservesMaterial)
{
    const std::array kernel{std::array{.5, .25}, std::array{.25, 1.}};
    const auto operations = R::qbmm_coalescence_from_kernel(rate_inputs(), kernel);
    ASSERT_EQ(operations.size(), 3u);
    const auto& aa = std::get<R::QbmmCoalescence>(operations[0]);
    const auto& ab = std::get<R::QbmmCoalescence>(operations[1]);
    const auto& bb = std::get<R::QbmmCoalescence>(operations[2]);
    EXPECT_EQ(aa.first, 0u);
    EXPECT_EQ(aa.second, 0u);
    EXPECT_EQ(aa.count, .5); // Expected N^2 pairs; an integer N*(N-1) law would give .375.
    EXPECT_EQ(ab.first, 0u);
    EXPECT_EQ(ab.second, 1u);
    EXPECT_EQ(ab.count, .25);
    EXPECT_EQ(bb.first, 1u);
    EXPECT_EQ(bb.second, 1u);
    EXPECT_EQ(bb.count, .25);
    const auto result = advance(stock(24), 6, {144, 1056}, rate_batch(operations));
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    // Remaining atoms: 2.75 at 2 and 1.25 at 8. Products: .5 at 4, .25 at 10, .25 at 16.
    expect_block(*result.endpoint, {5, 24, 188, 1968});
    EXPECT_EQ(result.endpoint->species.dispersedPopulations[1].amounts, (std::vector<double>{6, 9}));
    EXPECT_TRUE(R::qbmm_coalescence_from_kernel(rate_inputs(), {}).empty());
}

TEST_F(QbmmEventsTest, CoalescenceRejectsInvalidKernelsAndLeavesDonorOverdrawToTheOwner)
{
    for (const auto& kernel :
        {std::array{std::array{1., .25}, std::array{.5, 1.}}, std::array{std::array{-1., 0.}, std::array{0., 1.}},
            std::array{std::array{std::numeric_limits<double>::infinity(), 0.}, std::array{0., 1.}}})
        EXPECT_THROW((void) R::qbmm_coalescence_from_kernel(rate_inputs(), kernel), std::invalid_argument);
    const std::array excessive{std::array{100., 0.}, std::array{0., 0.}};
    const auto operations = R::qbmm_coalescence_from_kernel(rate_inputs(), excessive);
    ASSERT_EQ(operations.size(), 1u);
    EXPECT_EQ(std::get<R::QbmmCoalescence>(operations[0]).count, 100);
    rejected(advance(stock(24), 6, {144, 1056}, rate_batch(operations)), Status::CountDonorOverdraw);
}

TEST_F(QbmmEventsTest, BreakupFrequencyUsesPerParentDaughterFractionsAndConservesMass)
{
    const std::array frequencies{.5, 1.};
    const std::array daughters{
        std::vector<R::QbmmDaughterFraction>{{2, .5}}, std::vector<R::QbmmDaughterFraction>{{1, .25}, {1, .75}}};
    const auto operations = R::qbmm_breakup_from_frequency(rate_inputs(), frequencies, daughters);
    ASSERT_EQ(operations.size(), 2u);
    const auto& first = std::get<R::QbmmBreakup>(operations[0]);
    const auto& second = std::get<R::QbmmBreakup>(operations[1]);
    EXPECT_EQ(first.donor, 0u);
    EXPECT_EQ(first.count, .5);
    ASSERT_EQ(first.daughters.size(), 1u);
    EXPECT_EQ(first.daughters[0].count, 2);
    EXPECT_EQ(first.daughters[0].mass_over_scale, 1);
    EXPECT_EQ(second.donor, 1u);
    EXPECT_EQ(second.count, .5);
    ASSERT_EQ(second.daughters.size(), 2u);
    EXPECT_EQ(second.daughters[0].mass_over_scale, 2);
    EXPECT_EQ(second.daughters[1].mass_over_scale, 6);
    const auto result = advance(stock(24), 6, {144, 1056}, rate_batch(operations));
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    expect_block(*result.endpoint, {7, 24, 131, 909});
    EXPECT_TRUE(R::qbmm_breakup_from_frequency(rate_inputs(), std::array{0., 0.}, daughters).empty());
}

TEST_F(QbmmEventsTest, BreakupRejectsInvalidFrequencyDaughtersAndNodeLayout)
{
    const std::array daughters{
        std::vector<R::QbmmDaughterFraction>{{2, .5}}, std::vector<R::QbmmDaughterFraction>{{2, .5}}};
    EXPECT_THROW(
        (void) R::qbmm_breakup_from_frequency(rate_inputs(), std::array{1.}, daughters), std::invalid_argument);
    for (const double bad : {-1., std::numeric_limits<double>::infinity()})
        EXPECT_THROW((void) R::qbmm_breakup_from_frequency(rate_inputs(), std::array{bad, 0.}, daughters),
            std::invalid_argument);
    for (const auto& bad : {std::vector<R::QbmmDaughterFraction>{}, std::vector<R::QbmmDaughterFraction>{{2, .25}},
             std::vector<R::QbmmDaughterFraction>{{0, .5}}, std::vector<R::QbmmDaughterFraction>{{2, -1}},
             std::vector<R::QbmmDaughterFraction>{{2, std::numeric_limits<double>::quiet_NaN()}}})
    {
        const std::array invalid{bad, daughters[1]};
        EXPECT_THROW(
            (void) R::qbmm_breakup_from_frequency(rate_inputs(), std::array{1., 0.}, invalid), std::invalid_argument);
    }
}

TEST_F(QbmmEventsTest, SignedMassRatePairsGrowthAndShrinkageWithTheSameRoundedDestinations)
{
    const std::array rates{2., -4.};
    const std::array<R::QbmmTransferCounterpart, 2> counterparts{
        R::QbmmPhaseExchange{P::Dissolved}, R::QbmmPhaseExchange{P::Dissolved}};
    const auto operations = R::qbmm_resize_from_mass_rate(schema, rate_inputs(), rates, counterparts);
    ASSERT_EQ(operations.size(), 2u);
    const auto& growth = std::get<R::QbmmResize>(operations[0]);
    const auto& shrinkage = std::get<R::QbmmResize>(operations[1]);
    EXPECT_EQ(growth.count, 4);
    EXPECT_EQ(growth.mass_over_scale, 2.5);
    EXPECT_EQ(shrinkage.count, 2);
    EXPECT_EQ(shrinkage.mass_over_scale, 7);
    EXPECT_EQ(growth.transfer.moles, (std::vector<double>{.5, .75}));
    EXPECT_EQ(shrinkage.transfer.moles, (std::vector<double>{.5, .75}));
    const auto result = advance(stock(24, 2), 6, {144, 1056}, rate_batch(operations));
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    expect_block(*result.endpoint, {6, 24, 123, 748.5});
    EXPECT_EQ(result.endpoint->species.amount(*catalog, 6, P::Dissolved), .5);
    EXPECT_EQ(result.endpoint->species.amount(*catalog, 41, P::Dissolved), .75);
    EXPECT_TRUE(R::qbmm_resize_from_mass_rate(schema, rate_inputs(), std::array{0., 0.}, counterparts).empty());
}

TEST_F(QbmmEventsTest, MassRateRejectsZeroCrossingsUnrepresentableChangeAndIncompatibleLayout)
{
    const std::array<R::QbmmTransferCounterpart, 2> counterparts{
        R::QbmmPhaseExchange{P::Dissolved}, R::QbmmPhaseExchange{P::Dissolved}};
    for (const double bad :
        {-8., -16., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::denorm_min()})
        EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(schema, rate_inputs(), std::array{bad, 0.}, counterparts),
            std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(schema, rate_inputs(), std::array{1.}, counterparts),
        std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(schema, rate_inputs(.25, 2, 2), std::array{1., 0.}, counterparts),
        std::invalid_argument);
}

TEST_F(QbmmEventsTest, SignedMassRateUsesTransferScaleWhenRoundingTheDestination)
{
    constexpr double count = 0x1p52;
    R::QbmmQuadrature quadrature;
    quadrature.status = R::InversionStatus::RankOne;
    quadrature.active_atoms = 1;
    quadrature.atoms[0] = {count, 1};
    const auto inputs = rate_inputs(1, 2, 1, quadrature);
    const std::array<R::QbmmTransferCounterpart, 1> counterparts{R::QbmmPhaseExchange{P::Dissolved}};

    // The ±.75 kg requests cannot be represented by the adjacent destinations.
    // At unit scaled mass the neighboring destinations move +1 kg or -.5 kg.
    for (const double rate : {3 * 0x1p-54, -3 * 0x1p-54})
        EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(schema, inputs, std::array{rate}, counterparts),
            std::invalid_argument);

    for (const double rate : {std::nextafter(0x1p-52, 0.), std::nextafter(-0x1p-53, 0.)})
    {
        SCOPED_TRACE(rate);
        const auto events = R::qbmm_resize_from_mass_rate(schema, inputs, std::array{rate}, counterparts);
        ASSERT_EQ(events.size(), 1u);
        const auto& resize = std::get<R::QbmmResize>(events[0]);
        EXPECT_EQ(resize.count, count);
        EXPECT_EQ(resize.mass_over_scale, std::nextafter(1., rate > 0 ? 2. : 0.));
        EXPECT_EQ(resize.transfer.moles, moles(rate > 0 ? 1. : .5));
    }
}

TEST_F(QbmmEventsTest, MaterialRateAdaptersRequireFrozenFamilyAndCatalogBinding)
{
    const auto inputs = rate_inputs();
    const std::array births{R::QbmmBirthRate{3, 2, R::QbmmPhaseExchange{P::Dissolved}}};
    const std::array<R::QbmmTransferCounterpart, 2> counterparts{
        R::QbmmPhaseExchange{P::Dissolved}, R::QbmmPhaseExchange{P::Dissolved}};
    const std::array rates{2., 0.};
    const auto reject = [&](const char* label, const R::QbmmPopulationSchema& family, const R::QbmmLawInputs& frozen)
    {
        SCOPED_TRACE(label);
        EXPECT_THROW((void) R::qbmm_birth_from_rate(family, frozen, births), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(family, frozen, rates, counterparts), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_birth_from_rate(family, frozen, {}), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(family, frozen, std::array{0., 0.}, counterparts),
            std::invalid_argument);
    };
    const R::QbmmPopulationSchema wrong_composition{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, .75}, {41, .25}}};
    const R::QbmmPopulationSchema wrong_population{catalog, catalog->fingerprint(), "large",
        Binding{Binding::Role::DispersedGas, "large"}, 1, {{6, .25}, {41, .75}}};
    reject("equal-scale family with different composition", wrong_composition, inputs);
    reject("equal-scale family with different population", wrong_population, inputs);

    const R::SpeciesCatalog other_catalog(catalog->species(), catalog->revision() + 1);
    const auto restamp = [&](std::uint64_t family, std::uint64_t catalog_stamp, const std::string& population)
    {
        return R::QbmmLawInputs{inputs.quadrature, inputs.geometry, inputs.mass_scale_kg, inputs.bulk_volume_m3,
            inputs.absolute_pressure_pa, inputs.temperature_K, inputs.time_step_s, inputs.phase_volume_fraction,
            inputs.gas_volume_fraction, inputs.phase_concentration_mol_per_m3,
            inputs.carrier_liquid_concentration_mol_per_m3, inputs.carrier_liquid_temperature_K,
            inputs.slip_speed_m_per_s, family, catalog_stamp, population};
    };
    reject("wrong catalog stamp", schema, restamp(schema.fingerprint(), other_catalog.fingerprint(), "small"));
    reject("wrong population stamp", schema, restamp(schema.fingerprint(), catalog->fingerprint(), "large"));
    reject("missing binding", schema, restamp(0, 0, {}));

    const auto birth = R::qbmm_birth_from_rate(schema, inputs, births);
    const auto resize = R::qbmm_resize_from_mass_rate(schema, inputs, rates, counterparts);
    ASSERT_EQ(birth.size(), 1u);
    ASSERT_EQ(resize.size(), 1u);
    EXPECT_EQ(std::get<R::QbmmBirth>(birth[0]).atoms.at(0).transfer.moles, moles(3));
    EXPECT_EQ(std::get<R::QbmmResize>(resize[0]).transfer.moles, moles(2));
}

TEST_F(QbmmEventsTest, MaterialRateAdaptersRespectNonunitMassScaleAndExplicitReservoirHandoffs)
{
    const R::QbmmPopulationSchema scaled{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 2, {{6, .25}, {41, .75}}};
    const auto inputs = rate_inputs(.25, 2, 2, rate_quadrature(), &scaled);
    const R::QbmmReservoirExchange reservoir{"supplied-feed", 7.25, {1, -2, 4}};
    const std::array births{R::QbmmBirthRate{3, 4, reservoir}};
    const auto birth_events = R::qbmm_birth_from_rate(scaled, inputs, births);
    ASSERT_EQ(birth_events.size(), 1u);
    const auto& birth = std::get<R::QbmmBirth>(birth_events[0]).atoms.at(0);
    EXPECT_EQ(birth.atom.count, 1.5);
    EXPECT_EQ(birth.atom.mass_over_scale, 2);
    EXPECT_EQ(birth.transfer.moles, (std::vector<double>{1.5, 2.25})); // Six physical kilograms.
    const auto& handoff = std::get<R::QbmmReservoirExchange>(birth.transfer.counterpart);
    EXPECT_EQ(handoff.reservoir, "supplied-feed");
    EXPECT_EQ(handoff.energy_j, 7.25);
    EXPECT_EQ(handoff.momentum, (std::array<double, 3>{1, -2, 4}));

    const std::array<R::QbmmTransferCounterpart, 2> counterparts{reservoir, reservoir};
    const auto resize_events = R::qbmm_resize_from_mass_rate(scaled, inputs, std::array{4., -8.}, counterparts);
    ASSERT_EQ(resize_events.size(), 2u);
    EXPECT_EQ(std::get<R::QbmmResize>(resize_events[0]).mass_over_scale, 2.5);
    EXPECT_EQ(std::get<R::QbmmResize>(resize_events[1]).mass_over_scale, 7);
    for (const auto& event : resize_events)
    {
        const auto& transfer = std::get<R::QbmmResize>(event).transfer;
        EXPECT_EQ(transfer.moles, (std::vector<double>{1, 1.5})); // Four physical kilograms each way.
        EXPECT_EQ(std::get<R::QbmmReservoirExchange>(transfer.counterpart).energy_j, 7.25);
        EXPECT_EQ(std::get<R::QbmmReservoirExchange>(transfer.counterpart).momentum, (std::array<double, 3>{1, -2, 4}));
    }
}

TEST_F(QbmmEventsTest, RateAdaptersRejectMissingDurationAndInvalidFrozenPhysicalInputs)
{
    const auto check = [&](const R::QbmmLawInputs& input)
    {
        EXPECT_THROW((void) R::qbmm_birth_from_rate(schema, input, {}), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_coalescence_from_kernel(input, {}), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_breakup_from_frequency(input, {}, {}), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_resize_from_mass_rate(schema, input, {}, {}), std::invalid_argument);
    };
    check(rate_inputs(std::nullopt));
    check(rate_inputs(0));
    check(rate_inputs(-1));
    check(rate_inputs(std::numeric_limits<double>::quiet_NaN()));
    check(rate_inputs(.25, 0));
    check(rate_inputs(.25, 2, 0));
    R::QbmmQuadrature invalid;
    check(rate_inputs(.25, 2, 1, invalid));
}

TEST_F(QbmmEventsTest, SuppliedSlipUsesFieldMagnitudeInclusiveBoundsAndPreservesFrozenInputs)
{
    const auto inputs = rate_inputs(std::nullopt);
    const auto result = R::evaluate_qbmm_supplied_slip(slip_metadata(), inputs, {3, -4, 0});
    EXPECT_EQ(result.velocity_m_per_s, (std::array<double, 3>{3, -4, 0}));
    EXPECT_TRUE(result.applicability.applicable);
    EXPECT_TRUE(result.applicability.gate_passed);
    ASSERT_EQ(result.applicability.ranges.size(), 1u);
    EXPECT_EQ(result.applicability.ranges[0].status, R::QbmmRangeStatus::WithinRange);
    EXPECT_EQ(result.applicability.ranges[0].observed_minimum, 5);
    EXPECT_EQ(result.applicability.ranges[0].observed_maximum, 5);
    EXPECT_EQ(inputs.slip_speed_m_per_s, 999);
    EXPECT_FALSE(inputs.time_step_s); // Initialization needs no invented unit interval.
}

TEST_F(QbmmEventsTest, SuppliedSlipChecksTheEntireFamilyEvenWithANodeSelector)
{
    auto law = slip_metadata();
    law.ranges = {{R::QbmmRangeVariable::DiameterM, 1., 1.5, std::nullopt}};
    const R::QbmmLawInputs inputs{.quadrature = rate_quadrature(),
        .geometry = R::QbmmGeometryView{.active_atoms = 2, .diameter_m = {1., 2.},
            .min_diameter = 1., .max_diameter = 2.},
        .mass_scale_kg = 1., .bulk_volume_m3 = 2., .absolute_pressure_pa = 100000.,
        .phase_volume_fraction = .25, .node_diameter_m = 1.};
    EXPECT_TRUE(R::evaluate_qbmm_applicability(law, inputs).gate_passed);
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(law, inputs, {3, 4, 0}), std::invalid_argument);
}

TEST_F(QbmmEventsTest, ApplicabilityKeepsPhysicalLiquidConcentrationsSeparateAndExactAbsenceUndefined)
{
    auto law = slip_metadata();
    law.ranges = {{R::QbmmRangeVariable::PhaseConcentrationMolPerM3, 7, 7, 0},
        {R::QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3, 13, 13, 0},
        {R::QbmmRangeVariable::CarrierLiquidTemperatureK, 310, 310, std::nullopt}};
    const auto result = R::evaluate_qbmm_supplied_slip(law, rate_inputs(), {3, 4, 0});
    ASSERT_EQ(result.applicability.ranges.size(), 3u);
    EXPECT_TRUE(result.applicability.applicable);
    EXPECT_EQ(result.applicability.ranges[0].observed_minimum, 7);
    EXPECT_EQ(result.applicability.ranges[1].observed_minimum, 13);
    const auto concentrations = R::qbmm_phase_concentrations(std::array{2., 0.}, .5);
    ASSERT_TRUE(concentrations);
    EXPECT_EQ(*concentrations, (std::vector<double>{4, 0}));
    EXPECT_FALSE(R::qbmm_phase_concentrations(std::array{0., 0.}, 0));
    EXPECT_THROW((void) R::qbmm_phase_concentrations(std::array{2., 0.}, 0), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_phase_concentrations(std::array{0., 0.}, .5), std::invalid_argument);
}

TEST_F(QbmmEventsTest, SuppliedSlipRejectsNonfiniteFieldsAndHonorsReportOnlyAndMissingInputGates)
{
    const auto report_only = slip_metadata(R::QbmmApplicabilityGate::ReportOnly);
    const auto result = R::evaluate_qbmm_supplied_slip(report_only, rate_inputs(), {0, 0, 6});
    EXPECT_FALSE(result.applicability.applicable);
    EXPECT_TRUE(result.applicability.gate_passed);
    EXPECT_EQ(result.applicability.ranges[0].status, R::QbmmRangeStatus::OutsideRange);
    EXPECT_THROW(
        (void) R::evaluate_qbmm_supplied_slip(slip_metadata(), rate_inputs(), {0, 0, 6}), std::invalid_argument);
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        EXPECT_THROW(
            (void) R::evaluate_qbmm_supplied_slip(report_only, rate_inputs(), {bad, 0, 0}), std::invalid_argument);
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(report_only, rate_inputs(),
                     {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0}),
        std::invalid_argument);
    auto duration_gate = slip_metadata();
    duration_gate.ranges.push_back({R::QbmmRangeVariable::TimeStepS, .1, .5, std::nullopt});
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(duration_gate, rate_inputs(std::nullopt), {3, 4, 0}),
        std::invalid_argument);
    duration_gate.gate = R::QbmmApplicabilityGate::ReportOnly;
    const auto missing = R::evaluate_qbmm_supplied_slip(duration_gate, rate_inputs(std::nullopt), {3, 4, 0});
    EXPECT_EQ(missing.applicability.ranges[1].status, R::QbmmRangeStatus::MissingInput);
    EXPECT_FALSE(missing.applicability.applicable);
    EXPECT_TRUE(missing.applicability.gate_passed);
    duration_gate.kind = R::QbmmLawKind::Coalescence;
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(duration_gate, rate_inputs(), {3, 4, 0}), std::invalid_argument);
}

TEST_F(QbmmEventsTest, SuppliedSlipAtExactAbsenceSkipsPopulationRangesButAlwaysChecksFiniteField)
{
    R::QbmmQuadrature absent;
    absent.status = R::InversionStatus::Absent;
    const auto inputs = rate_inputs(std::nullopt, 2, 1, absent);
    const auto skipped = R::evaluate_qbmm_supplied_slip(slip_metadata(), inputs, {0, 0, 6});
    EXPECT_FALSE(skipped.applicability.applicable);
    EXPECT_TRUE(skipped.applicability.gate_passed);
    EXPECT_EQ(skipped.applicability.ranges[0].status, R::QbmmRangeStatus::NotApplicable);
    auto always = slip_metadata();
    always.requires_population = false;
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(always, inputs, {0, 0, 6}), std::invalid_argument);
    EXPECT_THROW((void) R::evaluate_qbmm_supplied_slip(
                     slip_metadata(), inputs, {0, std::numeric_limits<double>::quiet_NaN(), 0}),
        std::invalid_argument);
}

TEST_F(QbmmEventsTest, FrozenLawInputBuilderUsesPhysicalVolumesAndDoesNotInventDuration)
{
    const auto material = moles(24);
    const auto inputs =
        R::make_qbmm_law_inputs(schema, {rate_quadrature(), std::nullopt, material, 4, .5, 100000, 300, std::nullopt,
                                            .25, std::vector<double>{13, 17}, 310, {3, -4, 0}});
    EXPECT_EQ(inputs.phase_volume_fraction, .125);
    EXPECT_EQ(inputs.phase_concentration_mol_per_m3, (std::optional<std::vector<double>>{{12, 18}}));
    EXPECT_EQ(inputs.carrier_liquid_concentration_mol_per_m3, (std::optional<std::vector<double>>{{13, 17}}));
    EXPECT_EQ(inputs.slip_speed_m_per_s, 5);
    EXPECT_FALSE(inputs.time_step_s);
    EXPECT_EQ(inputs.family_fingerprint, schema.fingerprint());
    auto law = slip_metadata();
    law.ranges.push_back({R::QbmmRangeVariable::TimeStepS, .1, .5, std::nullopt});
    const auto evaluation = R::evaluate_qbmm_applicability(law, inputs);
    EXPECT_EQ(evaluation.ranges[1].status, R::QbmmRangeStatus::MissingInput);
    EXPECT_FALSE(evaluation.gate_passed);
}

TEST_F(QbmmEventsTest, FrozenLawInputBuilderPreservesExactPopulationAndLiquidAbsence)
{
    R::QbmmQuadrature absent;
    absent.status = R::InversionStatus::Absent;
    const auto material = moles(0);
    R::QbmmFrozenLawState frozen{
        absent, std::nullopt, material, 4, 0, 100000, 300, .025, 0, std::nullopt, std::nullopt, {0, 0, 0}};
    const auto inputs = R::make_qbmm_law_inputs(schema, frozen);
    EXPECT_EQ(inputs.phase_volume_fraction, 0);
    EXPECT_FALSE(inputs.temperature_K);
    EXPECT_FALSE(inputs.phase_concentration_mol_per_m3);
    EXPECT_FALSE(inputs.carrier_liquid_concentration_mol_per_m3);
    EXPECT_FALSE(inputs.carrier_liquid_temperature_K);
    frozen.phase_volume_m3 = 1;
    EXPECT_THROW((void) R::make_qbmm_law_inputs(schema, frozen), std::invalid_argument);
}

TEST_F(QbmmEventsTest, FamilyLawBindingsIncludeRangesGatesAndExactFamilyDeclarations)
{
    auto a = slip_metadata(R::QbmmApplicabilityGate::ReportOnly);
    a.kind = R::QbmmLawKind::Coalescence;
    a.input_basis = "bulk number density, count/m3";
    a.output_units = "m3/s";
    auto b = a;
    b.ranges[0].maximum = 6; // Same identity/revision, distinct immutable family contract.
    R::QbmmClosureVersions va, vb;
    va.laws = {a};
    vb.laws = {b};
    const R::QbmmPopulationSchema fa{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, .25}, {41, .75}}, va};
    const R::QbmmPopulationSchema fb{catalog, catalog->fingerprint(), "large",
        Binding{Binding::Role::DispersedGas, "large"}, 1, {{6, .25}, {41, .75}}, vb};
    EXPECT_TRUE(fa.declares_law(a));
    EXPECT_FALSE(fa.declares_law(b));
    EXPECT_TRUE(fb.declares_law(b));
    EXPECT_FALSE(fb.declares_law(a));
    EXPECT_FALSE(schema.declares_law(a));
    const R::QbmmPopulationSchema changed_ranges{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, .25}, {41, .75}}, vb};
    EXPECT_NE(fa.fingerprint(), changed_ranges.fingerprint());
    EXPECT_NE(R::qbmm_law_signature(a), R::qbmm_law_signature(b));
    b.gate = R::QbmmApplicabilityGate::RejectOutside;
    EXPECT_FALSE(fb.declares_law(b));
    EXPECT_NE(R::qbmm_law_signature(a), R::qbmm_law_signature(b));
    vb.laws = {b};
    const R::QbmmPopulationSchema changed_gate{catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, .25}, {41, .75}}, vb};
    EXPECT_NE(changed_ranges.fingerprint(), changed_gate.fingerprint());
}

TEST_F(QbmmEventsTest, FiniteBirthCarriesIntegratedSpeciesAndProviderProvenanceExactlyOnce)
{
    const auto before = stock(0, 3);
    const auto source = batch({R::QbmmBirth{{{{1, 3}, internal(3)}}}});
    const auto result = advance(before, 0, {}, source);
    ASSERT_EQ(result.status, Status::Accepted) << result.diagnostic;
    ASSERT_TRUE(result.endpoint);
    const auto& after = *result.endpoint;
    expect_block(after, {1, 3, 9, 27});
    EXPECT_EQ(after.species.amount(*catalog, 6, P::Dissolved), 0);
    EXPECT_EQ(after.species.amount(*catalog, 41, P::Dissolved), 0);
    EXPECT_EQ(before.dispersedPopulations[1].amounts, moles(0));
    EXPECT_EQ(before.amount(*catalog, 6, P::Dissolved), 0.75);
    ASSERT_EQ(after.receipts.size(), 1u);
    const auto& receipt = after.receipts.front();
    EXPECT_EQ(receipt.kind, Kind::Birth);
    EXPECT_EQ(receipt.provider, "test-law");
    EXPECT_EQ(receipt.provider_revision, 3u);
    EXPECT_EQ(receipt.substep, 19u);
    EXPECT_EQ(receipt.duration, 0.125);
    EXPECT_EQ(receipt.family_fingerprint, schema.fingerprint());
    EXPECT_EQ(receipt.catalog_fingerprint, catalog->fingerprint());
    EXPECT_EQ(receipt.population, "small");
    EXPECT_TRUE(receipt.donors.empty());
    ASSERT_EQ(receipt.products.size(), 1u);
    EXPECT_EQ(receipt.products[0].count, 1);
    EXPECT_EQ(receipt.products[0].mass_over_scale, 3);
    ASSERT_EQ(receipt.transfers.size(), 1u);
    EXPECT_TRUE(receipt.transfers[0].into_family);
    EXPECT_EQ(receipt.transfers[0].transfer.moles, moles(3));
    EXPECT_EQ(receipt.material_mass_change_over_scale, 3);
    EXPECT_EQ(after.moment_change, (R::QbmmEventMoments{1, 3, 9, 27}));
    EXPECT_EQ(after.gross_created, after.moment_change);
    EXPECT_EQ(after.gross_consumed, (R::QbmmEventMoments{}));
}

TEST_F(QbmmEventsTest, ExhaustingTwoRoundedWeightsRemovesTheEntireCanonicalCount)
{
    R::QbmmPopulationSchema pure{
        catalog, catalog->fingerprint(), "small", Binding{Binding::Role::DispersedGas, "small"}, 1, {{6, 1}, {41, 0}}};
    const auto q = R::invert_qbmm_moments({1, 3, 11, 43});
    ASSERT_EQ(q.active_atoms, 2u);
    auto before = stock(0);
    before.dispersedPopulations[1].amounts = {3, 0};
    // Split three moles into two binary64 packets whose sum is exactly the original stock.
    const double heavy = 8. / 3.;
    R::QbmmEventBatch release{pure.fingerprint(), 0, 1, {}};
    for (size_t i = 0; i < 2; ++i)
        release.events.push_back({"release", 1,
            R::QbmmDisappearance{
                i, q.atoms[i].count, {R::QbmmPhaseExchange{P::Dissolved}, {i == 0 ? 3 - heavy : heavy, 0}}}});
    const auto result =
        R::advance_qbmm_population(pure, before, cell_view(before, 1, {11, 43}, pure.fingerprint()), release);
    ASSERT_EQ(result.status, Status::Accepted) << result.diagnostic;
    EXPECT_EQ(result.endpoint->count, 0);
    EXPECT_EQ(result.endpoint->shape, R::QbmmShapeState{});
    EXPECT_EQ(result.endpoint->quadrature.status, R::InversionStatus::Absent);
    EXPECT_EQ(result.endpoint->moment_change[0], -1);
    EXPECT_EQ(result.endpoint->species.amount(*catalog, 6, P::Dissolved), 3);
    EXPECT_EQ(result.endpoint->receipts[0].donors[0].atom.count, q.atoms[0].count);
    constexpr double newborn = 0x1p-12;
    release.events.push_back(
        {"birth", 1, R::QbmmBirth{{{{newborn, 1}, {R::QbmmReservoirExchange{"feed", 0, {}}, {newborn, 0}}}}}});
    const auto replacement =
        R::advance_qbmm_population(pure, before, cell_view(before, 1, {11, 43}, pure.fingerprint()), release);
    ASSERT_EQ(replacement.status, Status::Accepted) << replacement.diagnostic;
    EXPECT_EQ(replacement.endpoint->count, newborn);
    EXPECT_EQ(replacement.endpoint->shape, (R::QbmmShapeState{newborn, newborn}));
    EXPECT_EQ(replacement.endpoint->moment_change[0], newborn - 1);
    EXPECT_EQ(replacement.endpoint->quadrature.status, R::InversionStatus::RankOne);
    release.events.pop_back();
    auto& light = std::get<R::QbmmDisappearance>(release.events[0].operation);
    light.count -= newborn;
    light.transfer.moles.front() = 3 - newborn - heavy;
    const auto survivor =
        R::advance_qbmm_population(pure, before, cell_view(before, 1, {11, 43}, pure.fingerprint()), release);
    ASSERT_EQ(survivor.status, Status::Accepted) << survivor.diagnostic;
    EXPECT_EQ(survivor.endpoint->count, newborn);
    EXPECT_EQ(survivor.endpoint->shape, (R::QbmmShapeState{newborn, newborn}));
}

TEST_F(QbmmEventsTest, UniformGrowthFromBothOriginalNodesAndPositiveBirthWithThreeSizes)
{
    const auto before = stock(14);
    const auto growth = batch({R::QbmmResize{0, 2, 1.5, external(1)}, R::QbmmResize{1, 3, 4.5, external(1.5)}});
    const auto result = advance(before, 5, {50, 194}, growth);
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    expect_block(*result.endpoint, {5, 33. / 2, 261. / 4, 2241. / 8});
    ASSERT_EQ(result.endpoint->receipts.size(), 2u);
    for (const auto& receipt : result.endpoint->receipts)
    {
        EXPECT_EQ(receipt.kind, Kind::Growth);
        ASSERT_EQ(receipt.transfers.size(), 1u);
        const auto& raw = receipt.transfers.front();
        EXPECT_TRUE(raw.into_family);
        const auto* reservoir = std::get_if<R::QbmmReservoirExchange>(&raw.transfer.counterpart);
        ASSERT_NE(reservoir, nullptr);
        EXPECT_EQ(reservoir->reservoir, "feed");
        EXPECT_EQ(reservoir->energy_j, 2.5);
        EXPECT_EQ(reservoir->momentum, (std::array<double, 3>{-1, 0, 3}));
    }
    EXPECT_EQ(result.endpoint->moment_change, (R::QbmmEventMoments{0, 2.5, 61. / 4, 689. / 8}));

    // Independent positive measure has three distinct sizes before two-node compression.
    const auto born = advance(before, 5, {50, 194}, batch({R::QbmmBirth{{{{1, 3}, external(3)}}}}));
    ASSERT_TRUE(born.endpoint) << born.diagnostic;
    expect_block(*born.endpoint, {6, 17, 59, 221});
    EXPECT_EQ(born.endpoint->quadrature.active_atoms, 2u);
}

TEST_F(QbmmEventsTest, CoalescenceAndReverseBreakupHaveOppositeSignedShapeAndPreserveMaterial)
{
    const auto paired = stock(5);
    const auto merged = advance(paired, 2, {17, 65}, batch({R::QbmmCoalescence{0, 1, 1}}));
    ASSERT_TRUE(merged.endpoint) << merged.diagnostic;
    expect_block(*merged.endpoint, {1, 5, 25, 125});
    EXPECT_EQ(merged.endpoint->moment_change, (R::QbmmEventMoments{-1, 0, 8, 60}));
    EXPECT_EQ(merged.endpoint->gross_consumed, (R::QbmmEventMoments{2, 5, 17, 65}));
    EXPECT_EQ(merged.endpoint->gross_created, (R::QbmmEventMoments{1, 5, 25, 125}));
    ASSERT_EQ(merged.endpoint->receipts.size(), 1u);
    EXPECT_EQ(merged.endpoint->receipts[0].kind, Kind::Coalescence);
    EXPECT_EQ(merged.endpoint->receipts[0].material_mass_change_over_scale, 0);
    EXPECT_TRUE(merged.endpoint->receipts[0].transfers.empty());
    const auto split = advance(*merged.endpoint, batch({R::QbmmBreakup{0, 1, {{1, 1}, {1, 4}}}}));
    ASSERT_TRUE(split.endpoint) << split.diagnostic;
    expect_block(*split.endpoint, {2, 5, 17, 65});
    EXPECT_EQ(split.endpoint->moment_change, (R::QbmmEventMoments{1, 0, -8, -60}));
    ASSERT_EQ(split.endpoint->receipts.size(), 1u);
    EXPECT_EQ(split.endpoint->receipts[0].kind, Kind::Breakup);
    EXPECT_EQ(split.endpoint->gross_consumed, merged.endpoint->gross_created);
    EXPECT_EQ(split.endpoint->gross_created, merged.endpoint->gross_consumed);
}

TEST_F(QbmmEventsTest, ShrinkageKeepsCountAndCompleteDissolutionRequiresExplicitDeath)
{
    const auto before = stock(10);
    const auto shrunk = advance(before, 5, {20, 40}, batch({R::QbmmResize{0, 5, 1, internal(5)}}));
    ASSERT_TRUE(shrunk.endpoint) << shrunk.diagnostic;
    expect_block(*shrunk.endpoint, {5, 5, 5, 5});
    ASSERT_EQ(shrunk.endpoint->receipts.size(), 1u);
    EXPECT_EQ(shrunk.endpoint->receipts[0].kind, Kind::Shrinkage);
    EXPECT_FALSE(shrunk.endpoint->receipts[0].transfers.at(0).into_family);
    const auto dissolved = advance(*shrunk.endpoint, batch({R::QbmmDisappearance{0, 5, internal(5)}}));
    ASSERT_TRUE(dissolved.endpoint) << dissolved.diagnostic;
    expect_block(*dissolved.endpoint, {});
    EXPECT_EQ(dissolved.endpoint->quadrature.status, R::InversionStatus::Absent);
    EXPECT_EQ(dissolved.endpoint->species.amount(*catalog, 6, P::Dissolved), 2.5);
    EXPECT_EQ(dissolved.endpoint->species.amount(*catalog, 41, P::Dissolved), 3.75);
    EXPECT_EQ(dissolved.endpoint->moment_change, (R::QbmmEventMoments{-5, -5, -5, -5}));
    EXPECT_EQ(dissolved.endpoint->receipts.at(0).kind, Kind::Disappearance);
    rejected(advance(before, 5, {20, 40}, batch({R::QbmmResize{0, 5, 0, internal(10)}})), Status::InvalidEvent);
}

TEST_F(QbmmEventsTest, EqualBirthAndDeathRetainBothGrossChannelsAndCannotFundEachOther)
{
    const auto exchange = batch({R::QbmmDisappearance{0, 1, internal(2)}, R::QbmmBirth{{{{1, 2}, internal(2)}}}});
    const auto turnover = advance(stock(2, 2), 1, {4, 8}, exchange);
    ASSERT_TRUE(turnover.endpoint) << turnover.diagnostic;
    expect_block(*turnover.endpoint, {1, 2, 4, 8});
    EXPECT_EQ(turnover.endpoint->moment_change, (R::QbmmEventMoments{}));
    EXPECT_EQ(turnover.endpoint->gross_consumed, (R::QbmmEventMoments{1, 2, 4, 8}));
    EXPECT_EQ(turnover.endpoint->gross_created, turnover.endpoint->gross_consumed);
    EXPECT_EQ(turnover.endpoint->receipts.size(), 2u);
    const auto before = stock(2);
    rejected(advance(before, 1, {4, 8}, exchange), Status::SpeciesDonorOverdraw);
    EXPECT_EQ(before.amount(*catalog, 6, P::Dissolved), 0);
    EXPECT_EQ(before.dispersedPopulations[1].amounts, moles(2));

    const auto competing = batch({R::QbmmBirth{{{{1, 1}, internal(1)}}}, R::QbmmBirth{{{{1, 1}, internal(1)}}}});
    const auto overdraw = advance(stock(0, 1.5), 0, {}, competing);
    rejected(overdraw, Status::SpeciesDonorOverdraw);
    ASSERT_EQ(overdraw.failed_event, 1u);
    ASSERT_TRUE(overdraw.rejected_event);
    const auto& failure = *overdraw.rejected_event;
    EXPECT_EQ(failure.reason, Status::SpeciesDonorOverdraw);
    EXPECT_EQ(failure.population, "small");
    EXPECT_EQ(failure.family_fingerprint, schema.fingerprint());
    EXPECT_EQ(failure.catalog_fingerprint, catalog->fingerprint());
    EXPECT_EQ(failure.attempt.provider, "test-law");
    EXPECT_EQ(failure.attempt.provider_revision, 3u);
    EXPECT_EQ(failure.substep, 19u);
    EXPECT_EQ(failure.duration, 0.125);
    ASSERT_NE(std::get_if<R::QbmmBirth>(&failure.attempt.operation), nullptr);
    ASSERT_TRUE(failure.donor_shortfall);
    const auto& shortage = *failure.donor_shortfall;
    const auto* donor = std::get_if<R::QbmmSpeciesDonor>(&shortage.stock);
    ASSERT_NE(donor, nullptr);
    EXPECT_EQ(donor->species, 6u);
    EXPECT_EQ(donor->phase, P::Dissolved);
    EXPECT_TRUE(donor->population.empty());
    EXPECT_EQ(shortage.available, 0.125);
    EXPECT_EQ(shortage.requested, 0.25);
}

TEST_F(QbmmEventsTest, CompleteReplacementRetainsTinyNewMaterialCountAndHigherMoments)
{
    const double original = 0x1p40, born = 0x1p-28;
    const auto before = stock(original);
    // Count and material are debited from the original size-one family. The new size-two
    // measure must survive even though adding either stock to its predecessor rounds away.
    const auto birth = R::QbmmBirth{{{{born, 2}, external(2 * born)}}};
    const auto death = R::QbmmDisappearance{0, original, internal(original)};
    for (const auto& changes : {batch({birth, death}), batch({death, birth})})
    {
        const auto result = advance(before, original, {original, original}, changes);
        ASSERT_TRUE(result.endpoint) << result.diagnostic;
        const auto& after = *result.endpoint;
        expect_block(after, {born, 2 * born, 4 * born, 8 * born});
        EXPECT_EQ(after.quadrature.status, R::InversionStatus::RankOne);
        EXPECT_EQ(after.gross_consumed, (R::QbmmEventMoments{original, original, original, original}));
        EXPECT_EQ(after.gross_created, (R::QbmmEventMoments{born, 2 * born, 4 * born, 8 * born}));
        ASSERT_EQ(after.receipts.size(), 2u);
        EXPECT_EQ(after.species.amount(*catalog, 6, P::Dissolved), original / 4);
        EXPECT_EQ(after.species.amount(*catalog, 41, P::Dissolved), 3 * original / 8);
    }
}

TEST_F(QbmmEventsTest, CompetingDebitsRetainSubUlpShortfallAndPositiveRemainder)
{
    constexpr double original = 0x1p40, small = 0x1p-28;
    // Leave a one-count gap so both packets' 1/4 and 3/8 molar components are exactly representable.
    constexpr double bulk_debit = original - 1;
    for (const bool use_atom_donor : {true, false})
    {
        const auto before = use_atom_donor ? stock(original) : stock(0, original);
        const double count = use_atom_donor ? original : 0.;
        const auto request = [&](double first, double second)
        {
            auto events = batch({});
            for (const double debit : {first, second})
                if (use_atom_donor)
                    events.events.push_back({"dissolve", 1, R::QbmmDisappearance{0, debit, internal(debit)}});
                else
                    events.events.push_back({"birth", 1, R::QbmmBirth{{{{debit, 1}, internal(debit)}}}});
            return advance(before, count, {count, count}, events);
        };

        // The small debit rounds away in a binary64 running stock. Reserving the full
        // original stock afterward must still reject and identify the actual shortage.
        ASSERT_EQ(original - small, original);
        const auto excessive = request(small, original);
        rejected(excessive, use_atom_donor ? Status::CountDonorOverdraw : Status::SpeciesDonorOverdraw);
        ASSERT_TRUE(excessive.rejected_event && excessive.rejected_event->donor_shortfall);
        EXPECT_EQ(excessive.rejected_event->donor_shortfall->shortfall, use_atom_donor ? small : small / 4);

        // The two dyadic requests add to original-small. The positive donor stock
        // survives cancellation, including both shape moments for a size-one atom.
        const auto accepted = request(1 - small, bulk_debit);
        ASSERT_TRUE(accepted.endpoint) << accepted.diagnostic;
        const auto& after = *accepted.endpoint;
        if (use_atom_donor)
            expect_block(after, {small, small, small, small});
        else
        {
            EXPECT_EQ(after.species.amount(*catalog, 6, P::Dissolved), small / 4);
            EXPECT_EQ(after.species.amount(*catalog, 41, P::Dissolved), 3 * small / 8);
        }
    }
}

TEST_F(QbmmEventsTest, IdenticalDonorPairsAndCompetingProcessesReserveOriginalCount)
{
    const auto before = stock(3); // Three size-one donors support 1.5 identical pairs.
    const auto all = advance(before, 3, {3, 3}, batch({R::QbmmCoalescence{0, 0, 1.5}}));
    ASSERT_TRUE(all.endpoint) << all.diagnostic;
    expect_block(*all.endpoint, {1.5, 3, 6, 12});
    const auto too_many = batch({R::QbmmCoalescence{0, 0, std::nextafter(1.5, 2.)}});
    rejected(advance(before, 3, {3, 3}, too_many), Status::CountDonorOverdraw);
    const auto competing = batch({R::QbmmCoalescence{0, 0, 1}, R::QbmmResize{0, 1.5, 2, external(1.5)}});
    const auto overdraw = advance(before, 3, {3, 3}, competing);
    rejected(overdraw, Status::CountDonorOverdraw);
    ASSERT_EQ(overdraw.failed_event, 1u);
    ASSERT_TRUE(overdraw.rejected_event);
    ASSERT_TRUE(overdraw.rejected_event->donor_shortfall);
    const auto& shortage = *overdraw.rejected_event->donor_shortfall;
    const auto* index = std::get_if<std::size_t>(&shortage.stock);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(*index, 0u);
    EXPECT_EQ(shortage.available, 1);
    EXPECT_EQ(shortage.requested, 1.5);
    const auto borrow_new = batch({R::QbmmBirth{{{{2, 1}, external(2)}}}, R::QbmmCoalescence{0, 0, 2}});
    rejected(advance(before, 3, {3, 3}, borrow_new), Status::CountDonorOverdraw);
}

TEST_F(QbmmEventsTest, RejectsForeignShapeStampAndDetachedOrWrongNamedSpeciesRowBeforeEvaluatingEvents)
{
    auto initial = stock(1);
    const double count = 1;
    const R::QbmmShapeState shape{1, 1};
    const auto changes = batch({R::QbmmBirth{{{{1, 1}, external(1)}}}});
    const R::QbmmPopulationSchema different_scale(catalog, catalog->fingerprint(), "small",
        Binding{Binding::Role::DispersedGas, "small"}, 2, {{6, 0.25}, {41, 0.75}});
    const auto foreign = cell_view(initial, count, shape, different_scale.fingerprint());
    const auto stale = R::advance_qbmm_population(schema, initial, foreign, changes);
    rejected(stale, Status::InvalidState);
    EXPECT_FALSE(stale.failed_event);

    // Both invalid views hold precisely the same values and schema as the correct row.
    // The source must debit the named owner, so the borrowed storage identity matters.
    initial.dispersedPopulations[0].amounts = moles(1);
    const auto copy = moles(1);
    auto material = cell_view(initial, count, shape, schema.fingerprint());
    const std::array<std::span<const double>, 2> invalid_rows{copy, initial.dispersedPopulations[0].amounts};
    for (auto wrong : invalid_rows)
    {
        material.species_moles = wrong;
        const auto result = R::advance_qbmm_population(schema, initial, material, changes);
        rejected(result, Status::InvalidState);
        EXPECT_FALSE(result.failed_event);
        EXPECT_EQ(initial.dispersedPopulations[1].amounts, moles(1));
    }
}

TEST_F(QbmmEventsTest, RejectsSelectiveOrUnpairedMaterialEvenWhenNetBatchLooksCompatible)
{
    const auto before = stock(1);
    auto left = external(1), right = external(1);
    // Same total mass and correct net composition, with opposing per-atom fraction errors.
    left.moles = {0.5, 0.25};
    right.moles = {0, 0.5};
    const auto masked = batch({R::QbmmBirth{{{{1, 1}, left}, {{1, 1}, right}}}});
    rejected(advance(before, 1, {1, 1}, masked), Status::InvalidTransfer);
    const auto wrong_mass = batch({R::QbmmResize{0, 1, 2, external(0.5)}});
    rejected(advance(before, 1, {1, 1}, wrong_mass), Status::InvalidTransfer);
    rejected(advance(before, 1, {1, 1}, batch({R::QbmmBirth{}})), Status::InvalidEvent);
    rejected(advance(before, 1, {1, 1}, batch({R::QbmmBreakup{0, 1, {{2, 1}}}})), Status::InvalidEvent);
    // Daughter expected counts conserve mass, but one daughter exceeds an entire parent's mass.
    const auto oversized = batch({R::QbmmBreakup{0, 1, {{0.25, 2}, {2, 0.25}}}});
    rejected(advance(before, 1, {1, 1}, oversized), Status::InvalidEvent);
    rejected(advance(before, 1, {1, 1}, batch({R::QbmmResize{1, 1, 2, external(1)}})), Status::InvalidEvent);
    auto missing = stock(1);
    missing.dispersedPopulations.erase(missing.dispersedPopulations.begin() + 1);
    const double count = 1;
    const R::QbmmShapeState shape{1, 1};
    const auto original = cell_view(before, count, shape, schema.fingerprint());
    rejected(R::advance_qbmm_population(schema, missing, original, batch({})), Status::InvalidState);
    auto foreign = before;
    ++foreign.schemaFingerprint;
    rejected(advance(foreign, 1, {1, 1}, batch({})), Status::InvalidState);
    auto incompatible = batch({R::QbmmBirth{{{{1, 1}, external(1)}}}});
    ++incompatible.family_fingerprint;
    rejected(advance(before, 1, {1, 1}, incompatible), Status::InvalidProvenance);
    EXPECT_EQ(before.dispersedPopulations[1].amounts, moles(1));
}
} // namespace
