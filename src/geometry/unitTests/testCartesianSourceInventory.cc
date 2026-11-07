#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/CartesianSourceInventory.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <gtest/gtest.h>
#include <span>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
CartesianSourceInventory::CapturedSource capture(std::shared_ptr<const CartesianIncidenceSource> source)
{
    CartesianSourceInventory::CapturedSource result;
    result.source = std::move(source);
    if (std::fegetenv(&result.environment))
        throw std::runtime_error("Cannot capture source test environment.");
    return result;
}
TEST(CartesianSourceInventory, ActiveBatchesRejectCrossLevelAncestorsAndAdmitTouchingMixedLevelNeighbors)
{
    const auto value = grid();
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto arena = CartesianGroupAuthority::make_arena(*budget, budget);
    const ContactHaloContext context{1, 2, 3, 4, 5};
    const auto parent = cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {0, 0, 0}}}, {'d'});
    const auto neighbor = cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {1, 0, 0}}}, {'d'});
    const auto child = cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{1, {0, 0, 0}}}, {'d'});
    auto p = CartesianIncidenceSource::make(value, std::span(&parent, 1), {'d'}, 0, 1, context, *budget, budget);
    auto n = CartesianIncidenceSource::make(value, std::span(&neighbor, 1), {'d'}, 0, 1, context, *budget, budget);
    auto c = CartesianIncidenceSource::make(value, std::span(&child, 1), {'d'}, 0, 1, context, *budget, budget);
    const std::array invalid{capture(p), capture(c)};
    const auto retained = budget->used;
    EXPECT_THROW((void) CartesianSourceInventory::make(context, 0, 1, invalid, *budget, budget, arena), std::exception);
    EXPECT_EQ(budget->used, retained);
    const std::array touching{capture(n), capture(c)};
    auto allowed = CartesianSourceInventory::make(context, 0, 1, touching, *budget, budget, arena);
    EXPECT_EQ(allowed->cell_count(), 2);
}
TEST(CartesianSourceInventory, RefineCoarsenAndFailedTransitionsKeepExactImmutableCapturedBatches)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const auto value = grid();
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto arena = CartesianGroupAuthority::make_arena(*budget, budget);
    const ContactHaloContext initial{1, 2, 3, 4, 5};
    auto next = initial;
    ++next.round;
    const std::array coarse{
        cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {0, 0, 0}}}, {'d'}),
        cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {1, 0, 0}}}, {'d'})};
    auto source = CartesianIncidenceSource::make(value, coarse, {'d'}, 0, 1, initial, *budget, budget);
    const std::array captured{capture(source)};
    auto accepted = CartesianSourceInventory::make(initial, 0, 1, captured, *budget, budget, arena);
    const auto old_identity = accepted->encode_identity(*budget).bytes();
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    std::vector<AdaptiveLeaf> fine;
    for (std::uint64_t z = 0; z != 2; ++z)
        for (std::uint64_t y = 0; y != 2; ++y)
            for (std::uint64_t x = 0; x != 2; ++x)
                fine.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{1, {x, y, z}}}, {'d'}));
    auto refined_source = CartesianIncidenceSource::make(value, fine, {'d'}, 0, 2, next, *budget, budget);
    const std::array added{capture(refined_source)};
    EXPECT_THROW((void) accepted->derive(next, 2, {}, added, *budget, budget, arena), std::exception);
    const std::array retired{CartesianSourceInventory::Retirement{coarse.front().key, coarse.front().revision}};
    auto refined = accepted->derive(next, 2, retired, added, *budget, budget, arena);
    ASSERT_EQ(refined->batches().size(), 2);
    EXPECT_EQ(refined->cell_count(), 9);
    EXPECT_EQ(refined->batches().front().source(), source);
    EXPECT_EQ(refined->batches().back().source(), refined_source);
    EXPECT_NE(source->evaluation_environment(), refined_source->evaluation_environment());
    EXPECT_EQ(accepted->encode_identity(*budget).bytes(), old_identity);
    EXPECT_TRUE(accepted->contains(coarse.front().key));
    EXPECT_FALSE(refined->contains(coarse.front().key));
    auto stale = retired;
    ++stale.front().revision;
    EXPECT_THROW((void) accepted->derive(next, 2, stale, added, *budget, budget, arena), std::exception);
    const std::array duplicate{retired.front(), retired.front()};
    EXPECT_THROW((void) accepted->derive(next, 2, duplicate, added, *budget, budget, arena), std::exception);
    EXPECT_THROW((void) accepted->derive(next, 1, retired, added, *budget, budget, arena), std::exception);
    auto back = next;
    ++back.round;
    std::vector<CartesianSourceInventory::Retirement> fine_retired;
    for (const auto& leaf : fine)
        fine_retired.push_back({leaf.key, leaf.revision});
    auto parent_source =
        CartesianIncidenceSource::make(value, std::span(coarse).first(1), {'d'}, 0, 3, back, *budget, budget);
    const std::array parent{capture(parent_source)};
    auto coarsened = refined->derive(back, 3, fine_retired, parent, *budget, budget, arena);
    EXPECT_EQ(coarsened->cell_count(), 2);
    EXPECT_TRUE(coarsened->contains(coarse.front().key));
    EXPECT_TRUE(coarsened->contains(coarse.back().key));
    auto rejected = std::make_shared<Budget>(Budget{0});
    EXPECT_THROW(CartesianGroupAuthority::make_arena(*rejected, rejected), std::exception);
    EXPECT_EQ(rejected->used, 0);
    EXPECT_EQ(accepted->encode_identity(*budget).bytes(), old_identity);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
}
} // namespace
