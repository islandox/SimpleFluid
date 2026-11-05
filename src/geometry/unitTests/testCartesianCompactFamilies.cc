/** Focused draft regression; register with the existing Cartesian source test
 * closure after integration. Run again with the producer compiled with
 * SIMPLEFLUID_CARTESIAN_LINE_FORCE_WIDE to exercise the actual wide store. */
#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/CartesianSourceLineRanges.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <limits>

#ifndef SIMPLEFLUID_TEST_EXPECT_LINE_INDEX_WIDTH
#error "Supply the independently expected retained index width (4 or 8) for this producer run."
#endif
#include <set>

namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(CartesianCompactFamilies, WidthBoundaryKeepsZeroAndFullUint32DistinctFromAbsence)
{
    using cartesian_line_detail::compact_indices_fit;
    constexpr auto largest = std::uint64_t(std::numeric_limits<std::uint32_t>::max());
    EXPECT_TRUE(compact_indices_fit({}));
    EXPECT_TRUE(compact_indices_fit({0, 1, largest - 1, largest}));
    EXPECT_FALSE(compact_indices_fit({largest + 1}));
    EXPECT_FALSE(compact_indices_fit({std::numeric_limits<std::uint64_t>::max()}));
    EXPECT_FALSE(compact_indices_fit({0, largest}, true));
}

TEST(CartesianCompactFamilies, BorrowedWordsFollowLiteralSignedByteBoundaryOrder)
{
    // This literal expectation is the existing full byte comparator's order,
    // not a numeric sort or a replay-derived permutation. 32767 = 0x7fff
    // starts with byte 0xff in the unchanged little-endian canonical word;
    // it follows 255 for signed char and is last for unsigned char.
    const std::array<std::uint64_t, 8> signed_order{128, 255, 32767, 32768, 0, 256, 1, 127};
    const std::array<std::uint64_t, 8> unsigned_order{0, 256, 32768, 1, 127, 128, 255, 32767};
    const auto& ordered = std::numeric_limits<char>::is_signed ? signed_order : unsigned_order;
    AdaptiveBytes prototype(24, char(0));
    const auto before = [&](std::uint64_t a, std::uint64_t b)
    {
        CartesianSupportRangeKeyView av{prototype, 8, a}, bv{prototype, 8, b};
        for (std::size_t i = 0; i != av.size(); ++i)
            if (av[i] != bv[i])
                return av[i] < bv[i];
        return false;
    };
    for (std::size_t i = 0; i != ordered.size(); ++i)
        for (std::size_t j = 0; j != ordered.size(); ++j)
            EXPECT_EQ(before(ordered[i], ordered[j]), i < j);
}

TEST(CartesianCompactFamilies, UniformActualPassRetainsExactFamilyCapacityAndLiteralPointValues)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    IdentityBudget identities;
    auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    DistributedAdaptiveIncidenceSession session(*kernel, comm, {}, {}, *budget, identities, budget);
    const auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
    { return *std::find_if(native.begin(), native.end(), [&](const auto& leaf) { return leaf.key == key; }); };
    const auto rows = session.bootstrap(native, lookup);
    ASSERT_EQ(rows.size(), 8);
    const auto snapshot = session.point_pool().source();
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->size(), 27);
    std::set<std::array<long double, 3>> actual;
    for (std::size_t i = 0; i != snapshot->size(); ++i)
        actual.insert(PLIC::details::GeometryIdentityAccess::point(snapshot->value({i})));
    std::set<std::array<long double, 3>> expected;
    for (auto x : {0.0L, 0.5L, 1.0L})
        for (auto y : {0.0L, 0.5L, 1.0L})
            for (auto z : {0.0L, 0.5L, 1.0L})
                expected.insert({x, y, z});
    EXPECT_EQ(actual, expected);

    std::size_t families = 0, general_membership_buffers = 0, prototype_tables = 0, observed_recipes = 0;
    std::set<std::size_t> observed_descriptor_bytes;
    session.for_each_borrowed_payload(
        [&](const ContactPayloadAllocation& allocation)
        {
            if (allocation.identity.role == ContactPayloadRole::CartesianLineRecipeObject)
            {
                ++observed_recipes;
                const auto* recipe = static_cast<const CartesianSourceLineRangeRecipe*>(allocation.identity.address);
                EXPECT_EQ(recipe->index_width_bytes(), std::size_t(SIMPLEFLUID_TEST_EXPECT_LINE_INDEX_WIDTH));
                RecordProperty("actual_retained_index_width_bytes", recipe->index_width_bytes());
            }
            if (allocation.identity.role == ContactPayloadRole::CartesianLineFamilies)
            {
                ++families;
                observed_descriptor_bytes.insert(allocation.element_size);
                EXPECT_GT(allocation.live_count, 0);
                EXPECT_EQ(allocation.live_count, allocation.capacity);
            }
            if (allocation.identity.role == ContactPayloadRole::CartesianLineMask ||
                allocation.identity.role == ContactPayloadRole::CartesianLineRanges)
                ++general_membership_buffers;
            if (allocation.identity.role == ContactPayloadRole::CartesianLinePrototypeEntries)
            {
                ++prototype_tables;
                EXPECT_EQ(allocation.live_count, allocation.capacity);
            }
        });
    EXPECT_GT(observed_recipes, 0);
    EXPECT_EQ(observed_descriptor_bytes.size(), 1);
    if (!observed_descriptor_bytes.empty())
        RecordProperty("actual_retained_family_element_bytes", *observed_descriptor_bytes.begin());
    EXPECT_GT(families, 0);
    EXPECT_GT(prototype_tables, 0);
    EXPECT_EQ(general_membership_buffers, 0);
}
} // namespace
