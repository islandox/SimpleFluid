/** @file testQbmmBulkGasConversion.cc
 * @brief Local resolved-gas pool conversion, shared donor reservations and pressure-conditioned energy handoff.
 */
#include "equations/PressureConditionedThermal.hh"
#include "equations/QbmmBulkGasPool.hh"
#include "equations/QbmmLawInputs.hh"
#include "equations/QbmmPopulationLawAdapters.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace
{
namespace R = SimpleFluid::Reacting;
using P = R::SpeciesPhase;
using Direction = R::QbmmPoolConversionDirection;
using Status = R::QbmmEventStatus;
using Operations = std::vector<R::QbmmEventOperation>;
using Moles = std::vector<double>;
constexpr R::SpeciesId A = 7, B = 42;

enum class Role
{
    DispersedGas,
    BulkGas,
    BulkLiquid
};
struct Phase
{
    Role role;
    std::string identity;
};

class QbmmBulkGasConversionTest : public ::testing::Test
{
protected:
    const std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{A, "A", 1, {{"A", 1}}, {P::Liquid, P::Dissolved, P::Resolved, P::Dispersed}},
            {B, "B", 2, {{"B", 1}}, {P::Dissolved, P::Resolved, P::Dispersed}}});
    const std::array<Phase, 4> roles{Phase{Role::DispersedGas, "small"}, Phase{Role::BulkGas, "pool"},
        Phase{Role::DispersedGas, "large"}, Phase{Role::BulkLiquid, "liquid"}};
    const std::array<R::SpeciesId, 2> ids{A, B};
    const std::array<bool, 4> bound{true, false, true, false};
    const R::QbmmPopulationSchema small{catalog, catalog->fingerprint(), "small", roles[0], 1, {{A, .5}, {B, .5}}};
    const R::QbmmPopulationSchema large{catalog, catalog->fingerprint(), "large", roles[2], 1, {{A, .5}, {B, .5}}};
    const R::QbmmBulkGasPoolBinding pool{
        catalog, catalog->fingerprint(), std::span(roles), 1, ids, Direction::Bidirectional, bound};

    static Moles moles(double mass) { return {.5 * mass, .25 * mass}; }

    R::CellSpeciesState stock(double small_mass, double pool_mass = 0) const
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"small", moles(small_mass)}, {"large", moles(0)}};
        state.set_amount(*catalog, A, P::Liquid, 1);
        for (unsigned i = 0; i < ids.size(); ++i)
            state.set_amount(*catalog, ids[i], P::Resolved, moles(pool_mass)[i]);
        return state;
    }

    R::QbmmLawInputs inputs(const R::QbmmPopulationSchema& family, R::QbmmMomentBlock moments = {3, 5, 11, 29},
        std::optional<double> dt = .25, double volume = 2, std::optional<Moles> pool_stock = Moles{0., 0.}) const
    {
        const bool absent = moments[0] == 0;
        const bool occupied = pool_stock && std::ranges::any_of(*pool_stock, [](double n) { return n > 0; });
        const auto family_stock = moles(moments[1] * family.mass_scale_kg());
        R::QbmmFrozenLawState state{R::invert_qbmm_moments(moments), {}, family_stock, volume,
            absent ? 0. : .5 * volume, 1024., absent ? std::nullopt : std::optional{320.}, dt, .75, {}, {}, {}};
        state.bulk_gas_pool_moles = pool_stock;
        if (pool_stock)
            state.bulk_gas_pool_fingerprint = pool.fingerprint();
        if (occupied)
        {
            state.bulk_gas_pool_volume_m3 = 1.;
            state.bulk_gas_pool_temperature_K = 320.;
            state.bulk_gas_pool_concentration_mol_per_m3 = pool_stock;
        }
        return R::make_qbmm_law_inputs(family, state);
    }

    static R::QbmmEventBatch batch(const R::QbmmPopulationSchema& family, const Operations& operations)
    {
        R::QbmmEventBatch result{family.fingerprint(), 17, .25, {}};
        for (const auto& operation : operations)
            result.events.push_back({"supplied-law", 11, operation});
        return result;
    }

    static R::QbmmCellView view(const R::QbmmPopulationSchema& family, const R::CellSpeciesState& state,
        const double& count, const R::QbmmShapeState& shape)
    {
        const auto& rows = state.dispersedPopulations;
        const auto row = std::ranges::find(rows, family.population(), &R::SpeciesPopulationState::id);
        return {row->id, state.schemaFingerprint, row->amounts, count, shape, family.fingerprint()};
    }

    auto advance(const R::CellSpeciesState& state, double count, const R::QbmmShapeState& shape,
        const Operations& operations) const
    {
        const auto changes = batch(small, operations);
        const std::array requests{R::QbmmFamilyEventInput{small, view(small, state, count, shape), changes}};
        return R::advance_qbmm_populations(state, requests, &pool);
    }

    Operations births(const R::QbmmPopulationSchema& family, R::QbmmMomentBlock moments, double count, double mass,
        double pool_mass) const
    {
        const std::array spectrum{R::QbmmPoolBirthAtom{count, mass}};
        return R::qbmm_pool_birth_from_spectrum(
            family, inputs(family, moments, .25, 2, moles(pool_mass)), pool, spectrum);
    }

    static void reject(const R::QbmmJointPopulationAdvance& result, Status expected)
    {
        EXPECT_EQ(result.status, expected) << result.diagnostic;
        EXPECT_FALSE(result.endpoint);
        EXPECT_FALSE(result.diagnostic.empty());
    }

    void check_endpoint(const R::QbmmJointPopulationAdvance& result, R::QbmmMomentBlock moments, double pool_mass) const
    {
        ASSERT_EQ(result.status, Status::Accepted) << result.diagnostic;
        ASSERT_TRUE(result.endpoint);
        const auto& [state, families] = *result.endpoint;
        ASSERT_EQ(families.size(), 1u);
        const auto& endpoint = families[0];
        EXPECT_EQ(R::qbmm_cell_moments(small, view(small, state, endpoint.count, endpoint.shape)), moments);
        EXPECT_EQ(state.dispersedPopulations[0].amounts, moles(moments[1]));
        EXPECT_EQ(state.dispersedPopulations[1].amounts, moles(0));
        EXPECT_EQ(state.amount(*catalog, A, P::Liquid), 1.);
        for (unsigned i = 0; i < ids.size(); ++i)
            EXPECT_EQ(state.amount(*catalog, ids[i], P::Resolved), moles(pool_mass)[i]);
    }

    void check_transfer(const R::QbmmEventReceipt& receipt, Direction direction, double mass) const
    {
        EXPECT_EQ(receipt.population, "small");
        EXPECT_EQ(receipt.provider, "supplied-law");
        EXPECT_EQ(receipt.provider_revision, 11u);
        EXPECT_EQ(receipt.substep, 17u);
        EXPECT_EQ(receipt.duration, .25);
        EXPECT_EQ(receipt.family_fingerprint, small.fingerprint());
        EXPECT_EQ(receipt.catalog_fingerprint, catalog->fingerprint());
        ASSERT_EQ(receipt.transfers.size(), 1u);
        const auto& transfer = receipt.transfers[0];
        EXPECT_EQ(transfer.into_family, direction == Direction::PoolToDispersed);
        EXPECT_EQ(transfer.transfer.moles, moles(mass));
        const auto& counterpart = std::get<R::QbmmPhaseExchange>(transfer.transfer.counterpart);
        EXPECT_EQ(counterpart.phase, P::Resolved);
        ASSERT_TRUE(counterpart.resolved_pool);
        EXPECT_EQ(counterpart.resolved_pool->direction, direction);
        EXPECT_EQ(counterpart.resolved_pool->pool_fingerprint, pool.fingerprint());
    }
};

TEST_F(QbmmBulkGasConversionTest, SelectedDonorAndSuppliedBirthSpectrumConserveSpeciesAndLoseOriginalShape)
{
    const auto drain = R::qbmm_pool_drain_from_fraction(small, inputs(small), pool, std::array{0., 1.});
    const auto deposited = advance(stock(5), 3, {11, 29}, drain);
    ASSERT_NO_FATAL_FAILURE(check_endpoint(deposited, {2, 2, 2, 2}, 3));
    const auto& first = deposited.endpoint->families[0];
    ASSERT_EQ(first.receipts.size(), 1u);
    const auto& death = first.receipts.front();
    check_transfer(death, Direction::DispersedToPool, 3);
    ASSERT_EQ(death.donors.size(), 1u);
    EXPECT_EQ(death.donors[0].index, 1u);
    EXPECT_EQ(death.donors[0].atom.count, 1.);
    EXPECT_EQ(death.donors[0].atom.mass_over_scale, 3.);
    EXPECT_EQ(death.gross_consumed, (R::QbmmEventMoments{1, 3, 9, 27}));
    EXPECT_EQ(death.gross_created, R::QbmmEventMoments{});
    const auto returned =
        advance(deposited.endpoint->species, first.count, first.shape, births(small, {2, 2, 2, 2}, 2, 1.5, 3));
    ASSERT_NO_FATAL_FAILURE(check_endpoint(returned, {4, 5, 6.5, 8.75}, 0));
    const auto& birth = returned.endpoint->families[0].receipts.at(0);
    check_transfer(birth, Direction::PoolToDispersed, 3);
    EXPECT_EQ(birth.gross_created, (R::QbmmEventMoments{2, 3, 4.5, 6.75}));
    EXPECT_EQ(birth.gross_consumed, R::QbmmEventMoments{});
}

TEST_F(QbmmBulkGasConversionTest, CompleteDrainGivesExactAbsenceAndExplicitRegeneration)
{
    const auto drain = R::qbmm_pool_drain_from_fraction(small, inputs(small), pool, std::array{1., 1.});
    const auto deposited = advance(stock(5), 3, {11, 29}, drain);
    ASSERT_NO_FATAL_FAILURE(check_endpoint(deposited, {}, 5));
    const auto& empty = deposited.endpoint->families[0];
    EXPECT_EQ(empty.quadrature.status, R::InversionStatus::Absent);
    EXPECT_EQ(empty.quadrature.active_atoms, 0u);
    const auto returned = advance(deposited.endpoint->species, empty.count, empty.shape, births(small, {}, 10, .5, 5));
    ASSERT_NO_FATAL_FAILURE(check_endpoint(returned, {10, 5, 2.5, 1.25}, 0));
    EXPECT_EQ(returned.endpoint->families[0].quadrature.status, R::InversionStatus::RankOne);
    const auto absent = R::make_qbmm_bulk_gas_pool_state(pool, returned.endpoint->species, 0.);
    EXPECT_EQ(absent.volume_m3, 0);
    EXPECT_EQ(absent.moles, moles(0));
    EXPECT_FALSE(absent.temperature_K);
    EXPECT_FALSE(absent.concentration_mol_per_m3);
}

TEST_F(QbmmBulkGasConversionTest, FullTwoNodeDrainPartitionsActualNonbinarySpeciesStockExactly)
{
    // Nominal per-node packets round 2/3 and 1 separately, leaving a positive residual in 5/3.
    const auto ternary = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{A, "A", 3., {{"A", 1}}, {P::Resolved, P::Dispersed}}});
    const R::QbmmPopulationSchema family{ternary, ternary->fingerprint(), "small", roles[0], 1., {{A, 1.}}};
    const R::QbmmBulkGasPoolBinding binding{
        ternary, ternary->fingerprint(), std::span(roles), 1, std::array{A}, Direction::Bidirectional, bound};
    R::CellSpeciesState original(*ternary);
    const double original_moles = 5. / 3.;
    original.dispersedPopulations = {{"small", {original_moles}}};
    R::QbmmFrozenLawState frozen{R::invert_qbmm_moments({3, 5, 11, 29}), {}, original.dispersedPopulations[0].amounts,
        2., 1., 1024., 320., {}, .5, {}, {}, {}};
    frozen.bulk_gas_pool_moles = Moles{0.};
    frozen.bulk_gas_pool_fingerprint = binding.fingerprint();
    const auto law = R::make_qbmm_law_inputs(family, frozen);
    const auto drain = R::qbmm_pool_drain_from_fraction(family, law, binding, std::array{1., 1.});
    const auto changes = batch(family, drain);
    const double count = 3;
    const R::QbmmShapeState shape{11, 29};
    const std::array request{R::QbmmFamilyEventInput{family, view(family, original, count, shape), changes}};
    const auto result = R::advance_qbmm_populations(original, request, &binding);
    ASSERT_TRUE(result.endpoint) << result.diagnostic;
    EXPECT_EQ(result.endpoint->species.amount(*ternary, A, P::Resolved), original_moles);
    EXPECT_EQ(result.endpoint->species.dispersedPopulations[0].amounts, Moles{0.});
    const auto& post = result.endpoint->families.at(0);
    EXPECT_EQ(post.count, 0.);
    EXPECT_EQ(post.shape, R::QbmmShapeState{});
    EXPECT_EQ(post.quadrature.status, R::InversionStatus::Absent);
    ASSERT_EQ(post.receipts.size(), 2u);
    const long double packet_sum = static_cast<long double>(post.receipts[0].transfers.at(0).transfer.moles.at(0)) +
                                   post.receipts[1].transfers.at(0).transfer.moles.at(0);
    EXPECT_EQ(packet_sum, static_cast<long double>(original_moles));
}

TEST_F(QbmmBulkGasConversionTest, IntegratedSpectraNeedNoRateDuration)
{
    const auto frozen = inputs(small, {}, std::nullopt, 2, moles(3));
    const std::array spectrum{R::QbmmPoolBirthAtom{2., 1.5}};
    const auto ops = R::qbmm_pool_birth_from_spectrum(small, frozen, pool, spectrum);
    ASSERT_EQ(ops.size(), 1u);
    const auto result = advance(stock(0, 3), 0, {}, ops);
    ASSERT_NO_FATAL_FAILURE(check_endpoint(result, {2., 3., 4.5, 6.75}, 0));
    EXPECT_THROW((void) R::qbmm_pool_birth_from_rate(small, frozen, pool, std::array{R::QbmmPoolBirthRate{2., 1.5}}),
        std::invalid_argument);
}

TEST_F(QbmmBulkGasConversionTest, OriginalNodeCountIsSharedByCoalescenceDissolutionAndPoolDrain)
{
    for (const double fraction : {.25, std::nextafter(.25, 1.)})
    {
        // Four unit parents. One equal-node coalescence uses two and dissolution uses one.
        auto changes = R::qbmm_pool_drain_from_fraction(small, inputs(small, {4, 4, 4, 4}), pool, std::array{fraction});
        changes.insert(changes.begin(), R::QbmmDisappearance{0, 1, {R::QbmmPhaseExchange{P::Dissolved}, moles(1)}});
        changes.insert(changes.begin(), R::QbmmCoalescence{0, 0, 1});
        const auto result = advance(stock(4), 4, {4, 4}, changes);
        if (fraction == .25)
        {
            ASSERT_NO_FATAL_FAILURE(check_endpoint(result, {1, 2, 4, 8}, 1));
            EXPECT_EQ(result.endpoint->species.amount(*catalog, B, P::Dissolved), .25);
        }
        else
        {
            reject(result, Status::CountDonorOverdraw);
            ASSERT_TRUE(result.rejected_event);
            EXPECT_EQ(result.rejected_event->population, "small");
            ASSERT_TRUE(result.rejected_event->donor_shortfall);
            EXPECT_EQ(result.rejected_event->donor_shortfall->shortfall, std::nextafter(1., 2.) - 1);
        }
    }
}

TEST_F(QbmmBulkGasConversionTest, FamiliesJointlyReserveOriginalPoolAndCannotBorrowSameBatchDeposits)
{
    const auto state = stock(5, 1);
    const double count = 3, zero = 0;
    const R::QbmmShapeState shape{11, 29}, empty{};
    const auto small_view = view(small, state, count, shape);
    const auto large_view = view(large, state, zero, empty);
    for (bool deposit : {false, true})
        for (bool excess : {false, true})
        {
            auto small_ops = births(small, {3, 5, 11, 29}, .5, 1, 1);
            if (deposit)
                small_ops = R::qbmm_pool_drain_from_fraction(small, inputs(small), pool, std::array{0., 1.});
            const double available = deposit ? 1 : .5;
            const auto small_batch = batch(small, small_ops);
            const auto large_batch =
                batch(large, births(large, {}, excess ? std::nextafter(available, 2.) : available, 1, 1));
            for (bool reverse : {false, true})
            {
                const R::QbmmFamilyEventInput first{small, small_view, small_batch},
                    second{large, large_view, large_batch};
                const std::array requests = reverse ? std::array{second, first} : std::array{first, second};
                const auto result = R::advance_qbmm_populations(state, requests, &pool);
                if (excess)
                {
                    reject(result, Status::SpeciesDonorOverdraw);
                    ASSERT_TRUE(result.rejected_event);
                    ASSERT_TRUE(result.rejected_event->donor_shortfall);
                    const std::string expected_family = deposit ? "large" : "small";
                    EXPECT_EQ(result.rejected_event->population, expected_family);
                    ASSERT_TRUE(result.failed_family);
                    EXPECT_EQ(requests[*result.failed_family].schema.population(), expected_family);
                    const auto& shortage = *result.rejected_event->donor_shortfall;
                    EXPECT_EQ(std::get<R::QbmmSpeciesDonor>(shortage.stock).phase, P::Resolved);
                    EXPECT_GT(shortage.shortfall, 0.);
                }
                else
                {
                    ASSERT_TRUE(result.endpoint) << result.diagnostic;
                    const auto& families = result.endpoint->families;
                    ASSERT_EQ(families.size(), 2u);
                    for (std::size_t i = 0; i < requests.size(); ++i)
                    {
                        EXPECT_EQ(families[i].family_fingerprint, requests[i].schema.fingerprint());
                        EXPECT_EQ(families[i].count,
                            requests[i].schema.population() == "small" ? (deposit ? 2. : 3.5) : available);
                    }
                    const auto& post = result.endpoint->species;
                    EXPECT_EQ(post.amount(*catalog, A, P::Resolved), deposit ? 1.5 : 0.);
                    EXPECT_EQ(post.amount(*catalog, B, P::Resolved), deposit ? .75 : 0.);
                    EXPECT_EQ(post.dispersedPopulations[0].amounts, moles(deposit ? 2. : 5.5));
                    EXPECT_EQ(post.dispersedPopulations[1].amounts, moles(available));
                }
            }
        }
}

TEST_F(QbmmBulkGasConversionTest, JointOwnerCannotFundLaterBirthWithEarlierDepositInOneBatch)
{
    auto changes = R::qbmm_pool_drain_from_fraction(small, inputs(small), pool, std::array{0., 1.});
    const R::QbmmMaterialTransfer withdrawal{
        R::QbmmPhaseExchange{
            P::Resolved, R::QbmmResolvedPoolConversion{pool.fingerprint(), Direction::PoolToDispersed}},
        moles(3)};
    changes.emplace_back(R::QbmmBirth{{{{2, 1.5}, withdrawal}}});
    const auto result = advance(stock(5), 3, {11, 29}, changes);
    reject(result, Status::SpeciesDonorOverdraw);
    ASSERT_EQ(result.failed_event, 1u);
    ASSERT_TRUE(result.rejected_event);
    ASSERT_TRUE(result.rejected_event->donor_shortfall);
    EXPECT_EQ(result.rejected_event->donor_shortfall->available, 0.);
    EXPECT_EQ(result.rejected_event->donor_shortfall->requested, 1.5);
}

TEST_F(QbmmBulkGasConversionTest, GrossReservationChecksSmallDebitsAndPositiveRemainders)
{
    // The first, third and fourth requests sum to exactly one in real arithmetic.
    // Put the extra debit below the stock ULP and within the native compensated subtotal.
    const std::array quantities{0x1p-60, 0x1p-100, 1. - 0x1p-53, 0x1p-53 - 0x1p-60};
    for (bool extra : {false, true})
    {
        Operations births_from_pool, consume_parents;
        for (std::size_t i = 0; i < quantities.size(); ++i)
        {
            if (i == 1 && !extra)
                continue;
            const double amount = quantities[i];
            auto birth = births(small, {}, amount, 1., 1.);
            births_from_pool.push_back(std::move(birth.at(0)));
            if (i == 2)
                consume_parents.emplace_back(R::QbmmCoalescence{0, 0, amount / 2});
            else
                consume_parents.emplace_back(
                    R::QbmmDisappearance{0, amount, {R::QbmmPhaseExchange{P::Dissolved}, moles(amount)}});
        }
        const auto species = advance(stock(0, 1), 0, {}, births_from_pool);
        const auto count = advance(stock(1), 1, {1, 1}, consume_parents);
        if (extra)
        {
            reject(species, Status::SpeciesDonorOverdraw);
            ASSERT_EQ(species.failed_event, 3u);
            ASSERT_TRUE(species.rejected_event);
            ASSERT_TRUE(species.rejected_event->donor_shortfall);
            const auto& debit = *species.rejected_event->donor_shortfall;
            EXPECT_EQ(std::get<R::QbmmSpeciesDonor>(debit.stock).species, A);
            EXPECT_EQ(debit.shortfall, 0x1p-101); // The A stock holds 0.5 mol per scaled mass unit.
            reject(count, Status::CountDonorOverdraw);
            ASSERT_EQ(count.failed_event, 3u);
            ASSERT_TRUE(count.rejected_event);
            ASSERT_TRUE(count.rejected_event->donor_shortfall);
            EXPECT_EQ(count.rejected_event->donor_shortfall->shortfall, 0x1p-100);
        }
        else
        {
            ASSERT_NO_FATAL_FAILURE(check_endpoint(species, {1, 1, 1, 1}, 0));
            ASSERT_TRUE(count.endpoint) << count.diagnostic;
            EXPECT_EQ(count.endpoint->families[0].quadrature.status, R::InversionStatus::RankOne);
        }
    }
    // Six partial withdrawals leave exactly 2^-100 unit atoms, with a positive rank-one endpoint.
    const auto one_atom = inputs(small, {1., 1., 1., 1.});
    Operations partial_drains;
    for (const double count :
        {0x1p-60, 0x1p-100, 1. - 0x1p-53, 0x1p-53 - 0x1p-59, 0x1p-60 - 0x1p-79, 0x1p-79 - 0x1p-99})
    {
        auto drain = R::qbmm_pool_drain_from_fraction(small, one_atom, pool, std::array{count});
        partial_drains.push_back(std::move(drain.at(0)));
    }
    constexpr double residual = 0x1p-100;
    const auto residual_state = advance(stock(1), 1., {1., 1.}, partial_drains);
    ASSERT_NO_FATAL_FAILURE(check_endpoint(residual_state, {residual, residual, residual, residual}, 1.));
    EXPECT_EQ(residual_state.endpoint->families[0].quadrature.status, R::InversionStatus::RankOne);
}

TEST_F(QbmmBulkGasConversionTest, BindingAndPhysicalPoolRejectMalformedRolesInventoriesAndOccupancy)
{
    const auto bind = [&](std::size_t phase, std::span<const R::SpeciesId> species, std::span<const bool> mask)
    {
        return R::QbmmBulkGasPoolBinding(
            catalog, catalog->fingerprint(), std::span(roles), phase, species, Direction::Bidirectional, mask);
    };
    for (auto index : {0u, 2u, 3u, 4u})
        EXPECT_THROW((void) bind(index, ids, bound), std::invalid_argument);
    EXPECT_THROW((void) bind(1, {}, bound), std::invalid_argument);
    EXPECT_THROW((void) bind(1, std::array{A, A}, bound), std::invalid_argument);
    EXPECT_THROW((void) bind(1, ids, {}), std::invalid_argument);
    EXPECT_THROW((void) bind(1, ids, std::array{true, true, true, false}), std::invalid_argument);
    const auto only_a = bind(1, std::array{A}, bound);
    const auto state = stock(0, 2);
    EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(only_a, state, 2, 300), std::invalid_argument);
    for (const auto bad : {-1., 0., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, state, bad, 300), std::invalid_argument);
        EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, state, 2, bad), std::invalid_argument);
    }
    EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, state, 2), std::invalid_argument);
    EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, stock(0), 0, 300), std::invalid_argument);
    EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, stock(0), 2, 300), std::invalid_argument);
    EXPECT_THROW((void) R::make_qbmm_bulk_gas_pool_state(pool, stock(0), 0, {}, 0), std::invalid_argument);
    EXPECT_THROW(
        (void) R::make_qbmm_bulk_gas_pool_state(pool, stock(0), 0, {}, {}, R::QbmmShapeState{}), std::invalid_argument);
    const auto physical = R::make_qbmm_bulk_gas_pool_state(pool, state, 2, 300);
    EXPECT_EQ(physical.moles, moles(2));
    ASSERT_TRUE(physical.concentration_mol_per_m3);
    EXPECT_EQ(*physical.concentration_mol_per_m3, (Moles{.5, .25}));
    EXPECT_EQ(physical.temperature_K, 300.);
}

TEST_F(QbmmBulkGasConversionTest, IntegratedDrainAndBirthRatesUseFrozenDurationVolumeAndCoordinateScale)
{
    const R::QbmmPopulationSchema scaled{catalog, catalog->fingerprint(), "small", roles[0], 4, {{A, .5}, {B, .5}}};
    const auto frozen = inputs(scaled, {3, 5, 11, 29}, .25, 2, moles(30));
    const auto drain = R::qbmm_pool_drain_from_frequency(scaled, frozen, pool, std::array{4 * std::log(2.), 0.});
    ASSERT_EQ(drain.size(), 1u);
    const auto& death = std::get<R::QbmmDisappearance>(drain.front());
    EXPECT_DOUBLE_EQ(death.count, 1.);
    EXPECT_EQ(death.donor, 0u);
    EXPECT_EQ(death.transfer.moles, moles(4));
    const auto weak = R::qbmm_pool_drain_from_frequency(scaled, frozen, pool, std::array{0x1p-60, 0.});
    ASSERT_EQ(weak.size(), 1u);
    EXPECT_EQ(std::get<R::QbmmDisappearance>(weak[0]).count, 0x1p-61);
    EXPECT_TRUE(R::qbmm_pool_drain_from_fraction(scaled, frozen, pool, std::array{0., 0.}).empty());
    const std::array rates{R::QbmmPoolBirthRate{2., 4.}, R::QbmmPoolBirthRate{4., 6.}};
    const auto birth = R::qbmm_pool_birth_from_rate(scaled, frozen, pool, rates);
    ASSERT_EQ(birth.size(), 1u);
    const auto& atoms = std::get<R::QbmmBirth>(birth[0]).atoms;
    ASSERT_EQ(atoms.size(), 2u);
    EXPECT_EQ(atoms[0].atom.count, 1.);
    EXPECT_EQ(atoms[0].atom.mass_over_scale, 1.);
    EXPECT_EQ(atoms[0].transfer.moles, moles(4));
    EXPECT_EQ(atoms[1].atom.count, 2.);
    EXPECT_EQ(atoms[1].atom.mass_over_scale, 1.5);
    EXPECT_EQ(atoms[1].transfer.moles, moles(12));
    EXPECT_TRUE(R::qbmm_pool_birth_from_rate(scaled, frozen, pool, {}).empty());
}

TEST_F(QbmmBulkGasConversionTest, AdaptersRejectInvalidNodeLayoutFractionsRatesPacketsCompositionAndStamps)
{
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto tiny = std::numeric_limits<double>::denorm_min();
    const auto frozen = inputs(small, {3, 5, 11, 29}, .25, 2, moles(4));
    const std::array valid{R::QbmmPoolBirthAtom{1, 2}};
    const std::array rate{R::QbmmPoolBirthRate{1, 2}};
    for (const auto values : {std::vector<double>{}, std::vector{0.}, std::vector{0., 0., 0.}, std::vector{0., -1.},
             std::vector{nan, 0.}, std::vector{0., inf}})
    {
        EXPECT_THROW((void) R::qbmm_pool_drain_from_fraction(small, frozen, pool, values), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_pool_drain_from_frequency(small, frozen, pool, values), std::invalid_argument);
    }
    EXPECT_THROW((void) R::qbmm_pool_drain_from_fraction(small, frozen, pool, std::array{1., std::nextafter(1., 2.)}),
        std::invalid_argument);
    EXPECT_THROW(
        (void) R::qbmm_pool_drain_from_fraction(small, inputs(small, {.25, .25, .25, .25}), pool, std::array{tiny}),
        std::invalid_argument);
    // exp(-50) is positive, but the rounded disappearance count equals the entire original atom.
    EXPECT_THROW(
        (void) R::qbmm_pool_drain_from_frequency(small, frozen, pool, std::array{200., 0.}), std::invalid_argument);
    auto malformed = frozen.quadrature;
    malformed.status = R::InversionStatus::RankOne; // Two active nodes cannot represent a rank-one input.
    const auto family_stock = moles(5.);
    R::QbmmFrozenLawState bad_layout{malformed, {}, family_stock, 2., 1., 1024., 320., .25, .5, {}, {}, {}};
    bad_layout.bulk_gas_pool_moles = moles(0);
    bad_layout.bulk_gas_pool_fingerprint = pool.fingerprint();
    EXPECT_THROW((void) R::qbmm_pool_drain_from_fraction(
                     small, R::make_qbmm_law_inputs(small, bad_layout), pool, std::array{0., 0.}),
        std::invalid_argument);
    for (const auto& atom : {R::QbmmPoolBirthAtom{0., 2.}, R::QbmmPoolBirthAtom{-1., 2.}, R::QbmmPoolBirthAtom{nan, 2.},
             R::QbmmPoolBirthAtom{1., 0.}, R::QbmmPoolBirthAtom{1., inf}, R::QbmmPoolBirthAtom{tiny, tiny}})
        EXPECT_THROW(
            (void) R::qbmm_pool_birth_from_spectrum(small, frozen, pool, std::array{atom}), std::invalid_argument);
    for (const auto& bad : {R::QbmmPoolBirthRate{0., 2.}, R::QbmmPoolBirthRate{-1., 2.}, R::QbmmPoolBirthRate{inf, 2.},
             R::QbmmPoolBirthRate{1., 0.}, R::QbmmPoolBirthRate{0., nan}, R::QbmmPoolBirthRate{tiny, 2.}})
        EXPECT_THROW((void) R::qbmm_pool_birth_from_rate(small, frozen, pool, std::array{bad}), std::invalid_argument);
    for (const auto stock_moles : {std::optional<Moles>{}, std::optional{moles(0)}, std::optional{Moles{2., 0.}}})
    {
        const auto wrong = inputs(small, {}, .25, 2, stock_moles);
        EXPECT_THROW((void) R::qbmm_pool_birth_from_spectrum(small, wrong, pool, valid), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_pool_birth_from_rate(small, wrong, pool, rate), std::invalid_argument);
    }
    EXPECT_THROW(
        (void) R::qbmm_pool_drain_from_fraction(large, frozen, pool, std::array{0., 0.}), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_pool_birth_from_rate(large, frozen, pool, {}), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_pool_birth_from_spectrum(large, frozen, pool, {}), std::invalid_argument);
    auto other_roles = roles;
    other_roles[1].identity = "other-pool";
    const R::QbmmBulkGasPoolBinding other_pool{
        catalog, catalog->fingerprint(), std::span(other_roles), 1, ids, Direction::Bidirectional, bound};
    ASSERT_NE(other_pool.fingerprint(), pool.fingerprint());
    EXPECT_THROW((void) R::qbmm_pool_birth_from_spectrum(small, frozen, other_pool, valid), std::invalid_argument);
    EXPECT_THROW(
        (void) R::qbmm_pool_drain_from_fraction(small, frozen, other_pool, std::array{0., 1.}), std::invalid_argument);
    const R::QbmmBulkGasPoolBinding drain_only{
        catalog, catalog->fingerprint(), std::span(roles), 1, ids, Direction::DispersedToPool, bound};
    const R::QbmmBulkGasPoolBinding birth_only{
        catalog, catalog->fingerprint(), std::span(roles), 1, ids, Direction::PoolToDispersed, bound};
    EXPECT_THROW(
        (void) R::qbmm_pool_drain_from_fraction(small, frozen, birth_only, std::array{0., 0.}), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_pool_birth_from_rate(small, frozen, drain_only, rate), std::invalid_argument);
}

TEST_F(QbmmBulkGasConversionTest, JointOwnerRejectsSpoofedConversionsAndPreservesAttemptedReceipt)
{
    const auto state = stock(1, 1);
    const double count = 1;
    const R::QbmmShapeState shape{1, 1};
    auto attempt = [&](R::QbmmEventOperation operation, const R::QbmmBulkGasPoolBinding* binding)
    {
        const auto changes = batch(small, {std::move(operation)});
        const std::array requests{R::QbmmFamilyEventInput{small, view(small, state, count, shape), changes}};
        return R::advance_qbmm_populations(state, requests, binding);
    };
    auto corrupt = state;
    corrupt.amounts[0] = std::numeric_limits<double>::quiet_NaN();
    const auto no_events = batch(small, {});
    const std::array requests{R::QbmmFamilyEventInput{small, view(small, corrupt, count, shape), no_events}};
    reject(R::advance_qbmm_populations(corrupt, requests), Status::InvalidState);
    const R::QbmmResolvedPoolConversion good{pool.fingerprint(), Direction::DispersedToPool};
    for (const auto counterpart : {R::QbmmPhaseExchange{P::Resolved}, R::QbmmPhaseExchange{P::Liquid, good},
             R::QbmmPhaseExchange{P::Resolved, R::QbmmResolvedPoolConversion{pool.fingerprint() + 1, good.direction}},
             R::QbmmPhaseExchange{
                 P::Resolved, R::QbmmResolvedPoolConversion{pool.fingerprint(), Direction::PoolToDispersed}}})
    {
        const auto result = attempt(R::QbmmDisappearance{0, 1., {counterpart, moles(1)}}, &pool);
        reject(result, Status::InvalidTransfer);
        ASSERT_TRUE(result.rejected_event);
        EXPECT_EQ(result.failed_family, 0u);
        EXPECT_EQ(result.failed_event, 0u);
        EXPECT_EQ(result.rejected_event->attempt.provider_revision, 11u);
        const auto& death = std::get<R::QbmmDisappearance>(result.rejected_event->attempt.operation);
        EXPECT_EQ(death.transfer.moles, moles(1));
        const auto& rejected = std::get<R::QbmmPhaseExchange>(death.transfer.counterpart);
        EXPECT_EQ(rejected.phase, counterpart.phase);
        ASSERT_EQ(rejected.resolved_pool.has_value(), counterpart.resolved_pool.has_value());
        if (rejected.resolved_pool)
        {
            EXPECT_EQ(rejected.resolved_pool->direction, counterpart.resolved_pool->direction);
            EXPECT_EQ(rejected.resolved_pool->pool_fingerprint, counterpart.resolved_pool->pool_fingerprint);
        }
    }
    const R::QbmmMaterialTransfer material{R::QbmmPhaseExchange{P::Resolved, good}, moles(.5)};
    reject(attempt(R::QbmmResize{0, 1., .5, material}, &pool), Status::InvalidTransfer);
    reject(attempt(R::QbmmDisappearance{0, .5, material}, nullptr), Status::InvalidTransfer);
}

TEST_F(QbmmBulkGasConversionTest, PoolLawObservesPhysicalVolumeAndSpeciesTraceAndDistinguishesAbsenceFromMissingInput)
{
    using V = R::QbmmRangeVariable;
    using S = R::QbmmRangeStatus;
    const R::QbmmLawMetadata law{"pool-window", 2, R::QbmmLawKind::BulkGasConversion, "physical donor trace", "1",
        R::QbmmApplicabilityGate::RejectOutside, false,
        {{V::BulkGasPoolVolumeFraction, 0., .75, {}}, {V::BulkGasPoolVolumeM3, 0., 2., {}},
            {V::BulkGasPoolConcentrationMolPerM3, 1., 1., 1}, {V::BulkGasPoolTemperatureK, 320., 320., {}}}};
    const auto occupied = R::evaluate_qbmm_applicability(law, inputs(small, {}, .25, 2., moles(4)));
    ASSERT_TRUE(occupied.applicable);
    EXPECT_TRUE(occupied.gate_passed);
    EXPECT_EQ(occupied.ranges[0].observed_minimum, .5);
    EXPECT_EQ(occupied.ranges[1].observed_minimum, 1.);
    EXPECT_EQ(occupied.ranges[2].observed_minimum, 1.);
    EXPECT_EQ(occupied.ranges[3].observed_minimum, 320.);
    const auto absent = R::evaluate_qbmm_applicability(law, inputs(small, {}, .25, 2., moles(0)));
    EXPECT_FALSE(absent.applicable);
    EXPECT_TRUE(absent.gate_passed);
    for (const auto i : {0, 1})
    {
        EXPECT_EQ(absent.ranges[i].status, S::WithinRange);
        EXPECT_EQ(absent.ranges[i].observed_minimum, 0.);
    }
    for (const auto i : {2, 3})
    {
        EXPECT_EQ(absent.ranges[i].status, S::NotApplicable);
        EXPECT_FALSE(absent.ranges[i].observed_minimum);
    }
    const auto missing = R::evaluate_qbmm_applicability(law, inputs(small, {}, .25, 2, std::nullopt));
    EXPECT_FALSE(missing.applicable);
    EXPECT_FALSE(missing.gate_passed);
    for (const auto& range : missing.ranges)
        EXPECT_EQ(range.status, S::MissingInput);
}

TEST_F(QbmmBulkGasConversionTest, LiquidTraceLawIsInactiveAtExactDryoutAndFailsMissingOccupiedInput)
{
    using V = R::QbmmRangeVariable;
    using S = R::QbmmRangeStatus;
    const R::QbmmLawMetadata film{"liquid-dependent", 3, R::QbmmLawKind::MassTransfer, "liquid EOS trace", "kg/s",
        R::QbmmApplicabilityGate::RejectOutside, true,
        {{V::CarrierLiquidConcentrationMolPerM3, 0, 1, 0}, {V::CarrierLiquidTemperatureK, 300, 325, {}}}};
    const auto material = moles(5.);
    R::QbmmFrozenLawState frozen{
        R::invert_qbmm_moments({3, 5, 11, 29}), {}, material, 2., 1., 1024., 320., .25, .5, {}, {}, {}};
    const auto missing = R::evaluate_qbmm_applicability(film, R::make_qbmm_law_inputs(small, frozen));
    EXPECT_FALSE(missing.applicable);
    EXPECT_FALSE(missing.gate_passed);
    for (const auto& range : missing.ranges)
        EXPECT_EQ(range.status, S::MissingInput);
    frozen.carrier_liquid_volume_m3 = 0.;
    const auto dry = R::evaluate_qbmm_applicability(film, R::make_qbmm_law_inputs(small, frozen));
    EXPECT_FALSE(dry.applicable);
    EXPECT_TRUE(dry.gate_passed);
    for (const auto& range : dry.ranges)
    {
        EXPECT_EQ(range.status, S::NotApplicable);
        EXPECT_FALSE(range.observed_minimum);
        EXPECT_FALSE(range.observed_maximum);
    }
    frozen.carrier_liquid_temperature_K = 320.;
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_volume_m3 = 1.;
    frozen.carrier_liquid_concentration_mol_per_m3 = Moles{.25, 0.};
    const auto wet = R::evaluate_qbmm_applicability(film, R::make_qbmm_law_inputs(small, frozen));
    EXPECT_TRUE(wet.applicable);
    EXPECT_TRUE(wet.gate_passed);
    EXPECT_EQ(wet.ranges[0].observed_minimum, .25);
    EXPECT_EQ(wet.ranges[1].observed_minimum, 320.);
}

TEST_F(QbmmBulkGasConversionTest, PressureConditionedTransferClosesSpeciesPartialEnthalpyRedistributionAndVolumeWork)
{
    // Initial family has (2.5, 1.25) mol, empty pool. Drain one size-three bubble.
    // A passive, constant-volume liquid makes the aggregate gas fraction nontrivial.
    const auto deposited = advance(
        stock(5), 3, {11, 29}, R::qbmm_pool_drain_from_fraction(small, inputs(small), pool, std::array{0., 1.}));
    ASSERT_NO_FATAL_FAILURE(check_endpoint(deposited, {2, 2, 2, 2}, 3));
    const auto& transfer = deposited.endpoint->families[0].receipts.at(0).transfers.at(0);
    const auto& phase = std::get<R::QbmmPhaseExchange>(transfer.transfer.counterpart);
    ASSERT_EQ(phase.phase, P::Resolved);
    ASSERT_FALSE(transfer.into_family);

    constexpr double p = 1024., t0 = 320., gas_constant = 8.;
    const R::ThermodynamicBounds bounds{100, 1000, 512, 2048};
    const auto gas = [&](double cv_a, double cv_b)
    {
        return R::make_ideal_gas_mixture(catalog, {{cv_a, t0, cv_a * t0}, {cv_b, t0, cv_b * t0}}, bounds,
            R::EnergyConvention::IncludesFormationEnergy, gas_constant);
    };
    const auto donor = gas(16., 32.);
    const auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{320., 1024., .25, 16., 0., 0., 5120., 0., 0.}; });
    const auto partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(R::PartialMolarEnthalpyLaw{
        {"dispersed-ideal-partials", 1, catalog->fingerprint()}, bounds, [](double t, double, std::span<const double>)
        { return R::PartialMolarEnthalpyPoint{{24. * t, 40. * t}, {24., 40.}, {0., 0.}}; }});
    auto config = std::make_shared<R::InterphaseConfiguration>();
    config->metadata = {"single-pool-conversion", 1, catalog->fingerprint()};
    config->phase_enthalpy = {partial, nullptr, nullptr, nullptr};
    config->stages = {{"drain", R::InterphaseStageKind::Material, {{A, 0, 1}, {B, 0, 1}}, {}}};
    const std::array<double, 4> energy_before{80 * t0, 0., 0., 16 * t0};
    const std::array<double, 4> volume_before{9.375, 0., 0., 4.};
    for (bool matched : {true, false})
    {
        SCOPED_TRACE(matched);
        const auto receiver = matched ? donor : gas(24., 48.);
        const std::array before{R::PhaseThermalState{donor, moles(5), p, energy_before[0], t0},
            R::PhaseThermalState{receiver, moles(0), p, 0., 0.}, R::PhaseThermalState{donor, moles(0), p, 0., 0.},
            R::PhaseThermalState{liquid, {1., 0.}, p, energy_before[3], t0}};
        const auto plan = R::compile_interphase_plan(config, before, {transfer.transfer.moles}, {{}, {}, {}, {}}, {});
        auto after = before;
        after[0].moles = deposited.endpoint->species.dispersedPopulations[0].amounts;
        for (unsigned i = 0; i < ids.size(); ++i)
            after[1].moles[i] = deposited.endpoint->species.amount(*catalog, ids[i], P::Resolved);
        const auto result = R::evaluate_pressure_conditioned_thermal(
            plan, after, energy_before, volume_before, p, R::PhaseThermalMode::Equilibrium);
        // At prescribed p the total enthalpy is fixed. The constant liquid pV cancels.
        // dH/dT (J/K) for the old gas is 110. After transfer the donor retains 44,
        // the pool has 66 or 90, and the passive liquid contributes 16 on both sides.
        const double pool_cp = matched ? 66. : 90.;
        const double t = 126. * t0 / (44. + pool_cp + 16.);
        const std::array<double, 4> volume{1.5 * gas_constant * t / p, 2.25 * gas_constant * t / p, 0., 4.};
        const std::array<double, 4> redistribution{44. * (t - t0), pool_cp * t - 66. * t0, 0., 16. * (t - t0)};
        const std::array<double, 4> carried{-66. * t0, 66. * t0, 0., 0.};
        ASSERT_EQ(result.material_events.size(), 2u);
        EXPECT_DOUBLE_EQ(result.material_events[0].molar_enthalpy, 24 * t0);
        EXPECT_DOUBLE_EQ(result.material_events[1].molar_enthalpy, 40 * t0);
        EXPECT_DOUBLE_EQ(result.material_events[0].carried_energy, 1.5 * 24 * t0);
        EXPECT_DOUBLE_EQ(result.material_events[1].carried_energy, .75 * 40 * t0);
        double work = 0;
        for (std::size_t a = 0; a < volume.size(); ++a)
        {
            EXPECT_EQ(result.phases[a].absolute_pressure, p);
            EXPECT_NEAR(result.phases[a].temperature, a == 2 ? 0. : t, 5e-12);
            EXPECT_NEAR(result.volumes[a], volume[a], 1e-13);
            EXPECT_NEAR(result.raw_material_enthalpy[a], carried[a], 1e-9);
            EXPECT_NEAR(result.raw_equilibrium_redistribution[a], redistribution[a], 1e-9);
            EXPECT_EQ(result.raw_conductive_heat[a], 0.);
            const double phase_work = -p * (volume[a] - volume_before[a]);
            EXPECT_NEAR(
                result.phases[a].internal_energy - energy_before[a], carried[a] + redistribution[a] + phase_work, 1e-9);
            work += phase_work;
        }
        EXPECT_NEAR(result.pressure_work, work, 1e-9);
        if (matched)
        {
            const double work_roundoff = std::numeric_limits<double>::epsilon() * p * result.total_volume;
            EXPECT_EQ(result.phases[0].temperature, t0);
            EXPECT_EQ(result.phases[1].temperature, t0);
            EXPECT_EQ(result.total_volume, 13.375);
            EXPECT_EQ((result.volumes[0] + result.volumes[1]) / result.total_volume, 9.375 / 13.375);
            EXPECT_NEAR(result.pressure_work, 0., work_roundoff);

            const auto& transferred_state = deposited.endpoint->species;
            const auto frozen_pool =
                R::make_qbmm_bulk_gas_pool_state(pool, transferred_state, result.volumes[1], t0);
            const R::QbmmFrozenLawState frozen{deposited.endpoint->families[0].quadrature, {},
                transferred_state.dispersedPopulations[0].amounts, result.total_volume, result.volumes[0], p, t0,
                .25, 9.375 / 13.375, {}, {}, {}};
            const auto returned = advance(transferred_state, 2., {2., 2.},
                R::qbmm_pool_birth_from_spectrum(small, R::make_qbmm_law_inputs(small, frozen, frozen_pool),
                    pool, std::array{R::QbmmPoolBirthAtom{2., 1.5}}));
            ASSERT_NO_FATAL_FAILURE(check_endpoint(returned, {4, 5, 6.5, 8.75}, 0));
            const auto& return_packet =
                returned.endpoint->families[0].receipts.at(0).transfers.at(0).transfer.moles;
            auto reversed_config = std::make_shared<R::InterphaseConfiguration>(*config);
            reversed_config->phase_enthalpy = {nullptr, partial, nullptr, nullptr};
            reversed_config->stages = {{"return", R::InterphaseStageKind::Material, {{A, 1, 0}, {B, 1, 0}}, {}}};
            const auto reversed_plan = R::compile_interphase_plan(
                reversed_config, result.phases, {return_packet}, {{}, {}, {}, {}}, {});
            auto returned_templates = result.phases;
            returned_templates[0].moles = returned.endpoint->species.dispersedPopulations[0].amounts;
            returned_templates[1].moles = moles(0);
            const std::array returned_energy{result.phases[0].internal_energy, result.phases[1].internal_energy,
                result.phases[2].internal_energy, result.phases[3].internal_energy};
            const auto restored = R::evaluate_pressure_conditioned_thermal(reversed_plan, returned_templates,
                returned_energy, result.volumes, p, R::PhaseThermalMode::Equilibrium);
            EXPECT_EQ(restored.phases[0].absolute_pressure, p);
            EXPECT_EQ(restored.phases[0].temperature, t0);
            EXPECT_EQ(restored.phases[1].temperature, 0.);
            EXPECT_EQ(restored.volumes[1], 0.);
            EXPECT_EQ(restored.total_volume, result.total_volume);
            EXPECT_EQ(restored.volumes[0] / restored.total_volume, 9.375 / 13.375);
            EXPECT_NEAR(restored.pressure_work, 0., work_roundoff);
        }
        else
        {
            EXPECT_NEAR(result.phases[1].temperature, 268.8, 5e-12);
            EXPECT_NEAR(result.pressure_work, 1536., 1e-9);
        }
    }
}
} // namespace
