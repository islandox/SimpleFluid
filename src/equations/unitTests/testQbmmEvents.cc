/** @file testQbmmEvents.cc
 * @brief Constructive mass-coordinate events, gross receipts and finite pre-substep donor reservations.
 */
#include "equations/QbmmPopulationEvents.hh"

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
};

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
