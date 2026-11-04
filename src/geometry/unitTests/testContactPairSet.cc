#include "geometry/mesh/AMR/ContactPairSet.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <vector>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Pair = D::ContactPairSet::Pair;
constexpr std::size_t ample = 16 * 1024 * 1024;
std::vector<Pair> values(const D::ContactPairSet::SortedPairs& sorted)
{
    return {sorted.begin(), sorted.end()};
}
TEST(ContactPairSet, CompleteValuesDeduplicateAndReturnOriginalAscendingOrder)
{
    D::Budget budget{ample};
    {
        D::ContactPairSet pairs(budget);
        EXPECT_EQ(pairs.storage_bytes(), 0u);
        EXPECT_EQ(budget.used, 0u);
        std::set<Pair> expected;
        std::uint64_t random = 1234567;
        for (unsigned i = 0; i < 10000; ++i)
        {
            random = random * 6364136223846793005ULL + 1;
            const auto a = std::size_t((random >> 16) % 1000);
            random = random * 6364136223846793005ULL + 1;
            const auto b = std::size_t((random >> 16) % 1000);
            if (a == b)
                continue;
            const Pair pair{std::min(a, b), std::max(a, b)};
            EXPECT_EQ(pairs.insert(pair), expected.insert(pair).second);
        }
        EXPECT_THROW((void) pairs.insert({4, 4}), std::invalid_argument);
        EXPECT_THROW((void) pairs.insert({5, 4}), std::invalid_argument);
        const auto before = pairs.size();
        auto sorted = pairs.take_sorted();
        EXPECT_EQ(sorted.size(), before);
        EXPECT_EQ(values(sorted), (std::vector<Pair>(expected.begin(), expected.end())));
        EXPECT_EQ(pairs.size(), 0u);
        EXPECT_EQ(pairs.storage_bytes(), 0u);
        const auto cumulative = budget.cumulative;
        sorted.clear();
        EXPECT_EQ(budget.used, 0u);
        EXPECT_EQ(budget.cumulative, cumulative);
    }
    EXPECT_EQ(budget.used, 0u);
}
TEST(ContactPairSet, FullWidthFallbackDoesNotAliasLowWords)
{
    if (std::numeric_limits<std::size_t>::digits <= 32)
        GTEST_SKIP() << "Wide local IDs require size_t wider than32 bits.";
    D::Budget budget{ample};
    D::ContactPairSet pairs(budget);
    const auto high = std::size_t{1} << 32;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const std::vector<Pair> expected{{0, 1}, {0, high + 1}, {1, high + 1}, {high, high + 1}, {maximum - 1, maximum}};
    for (auto it = expected.rbegin(); it != expected.rend(); ++it)
        EXPECT_TRUE(pairs.insert(*it));
    for (const auto& pair : expected)
        EXPECT_FALSE(pairs.insert(pair));
    auto sorted = pairs.take_sorted();
    EXPECT_EQ(values(sorted), expected);
}
TEST(ContactPairSet, GrowthAndSortingRejectBeforeReplacingLiveStorage)
{
    D::Budget budget{ample};
    D::ContactPairSet pairs(budget);
    for (std::size_t i = 1; i <= 7; ++i)
        EXPECT_TRUE(pairs.insert({0, i}));
    const auto before = budget.used;
    budget.limit = before;
    EXPECT_THROW((void) pairs.insert({0, 8}), std::invalid_argument);
    EXPECT_EQ(pairs.size(), 7u);
    EXPECT_EQ(budget.used, before);
    const auto sorted_bytes = pairs.size() * sizeof(std::uint64_t);
    budget.limit = before + sorted_bytes - 1;
    EXPECT_THROW((void) pairs.take_sorted(), std::invalid_argument);
    EXPECT_EQ(pairs.size(), 7u);
    EXPECT_EQ(budget.used, before);
    ++budget.limit;
    auto sorted = pairs.take_sorted();
    EXPECT_EQ(sorted.size(), 7u);
    EXPECT_EQ(budget.peak, budget.limit);
    EXPECT_EQ(budget.used, sorted_bytes);
    for (std::size_t i = 0; const auto& pair : sorted)
        EXPECT_EQ(pair, (Pair{0, ++i}));
}
TEST(ContactPairSet, SortedOwnerMovesAndCanOutliveTheEmptiedSet)
{
    D::Budget budget{ample};
    D::ContactPairSet::SortedPairs retained;
    {
        D::ContactPairSet pairs(budget);
        EXPECT_TRUE(pairs.insert({1, 9}));
        EXPECT_TRUE(pairs.insert({0, 4}));
        retained = pairs.take_sorted();
    }
    EXPECT_EQ(values(retained), (std::vector<Pair>{{0, 4}, {1, 9}}));
    D::ContactPairSet::SortedPairs moved(std::move(retained));
    EXPECT_EQ(retained.size(), 0u);
    EXPECT_EQ(values(moved), (std::vector<Pair>{{0, 4}, {1, 9}}));
    moved.clear();
    EXPECT_EQ(budget.used, 0u);
}
TEST(ContactPairSet, N32InteriorSlabHasExactPairsAndLowerPeakThanTreeNodes)
{
    constexpr int n = 32, low = 8, high = 16;
    const auto id = [](int x, int y, int z) { return std::size_t(x + n * (y + n * z)); };
    D::Budget budget{ample};
    D::ContactPairSet pairs(budget);
    std::set<Pair> oracle;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = low; x < high; ++x)
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (!(dx || dy || dz) || x + dx < 0 || x + dx >= n || y + dy < 0 || y + dy >= n ||
                                z + dz < 0 || z + dz >= n)
                                continue;
                            const auto a = id(x, y, z), b = id(x + dx, y + dy, z + dz);
                            const Pair pair{std::min(a, b), std::max(a, b)};
                            EXPECT_EQ(pairs.insert(pair), oracle.insert(pair).second);
                        }
    // Independent Cartesian neighbor count: local/local edges plus two
    // interfaces, including edge/corner contacts across each interface.
    constexpr std::size_t transverse = (3 * n - 2) * (3 * n - 2);
    constexpr std::size_t expected = (transverse * (3 * (high - low) - 2) - n * n * (high - low)) / 2 + 2 * transverse;
    ASSERT_EQ(expected, 110772u);
    ASSERT_EQ(pairs.size(), expected);
    const auto table = pairs.storage_bytes();
    auto sorted = pairs.take_sorted();
    EXPECT_EQ(values(sorted), (std::vector<Pair>(oracle.begin(), oracle.end())));
    const auto tree_bytes = expected * (sizeof(Pair) + 4 * sizeof(void*));
    EXPECT_LT(budget.peak, tree_bytes);
    EXPECT_EQ(sorted.storage_bytes(), expected * sizeof(std::uint64_t));
    RecordProperty("pair_count", std::to_string(expected));
    RecordProperty("table_bytes", std::to_string(table));
    RecordProperty("sorted_bytes", std::to_string(sorted.storage_bytes()));
    RecordProperty("peak_live_bytes", std::to_string(budget.peak));
    RecordProperty("tree_node_bytes", std::to_string(tree_bytes));
}
} // namespace
