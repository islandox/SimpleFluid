#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/CartesianSourceLineRanges.hh"
#include "geometry/mesh/AMR/PooledLineRegistry.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
namespace
{
using namespace cartesian_incidence_test;
static_assert(!std::is_default_constructible_v<CartesianLineRangeAdmission>);
template<class F> bool invalid(F&& f)
{
    try
    {
        f();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}
struct ConstantBytesHash
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const AdaptiveBytes&) const noexcept { return 7; }
};
TEST(CartesianSourceRanges, BorrowedWordSelectorsKeepOriginalSignedByteOrderingAndFullCollisionEquality)
{
    const std::array<std::uint64_t, 17> values{0, 1, 126, 127, 128, 129, 254, 255, 256, 257, 32767, 32768, 65535, 65536,
        0xffffffffULL, 0x100000000ULL, std::numeric_limits<std::uint64_t>::max()};
    Budget budget{ample};
    IdentityPool<AdaptiveBytes> normal(budget);
    IdentityPool<AdaptiveBytes, ConstantBytesHash> collisions(budget);
    Writer prototype;
    prototype.word(17);
    prototype.word(0);
    prototype.word(23);
    std::vector<AdaptiveBytes> original;
    for (const auto value : values)
    {
        Writer writer;
        writer.word(17);
        writer.word(value);
        writer.word(23);
        CartesianSupportRangeKeyView view{prototype.data, 8, value};
        AdaptiveBytes reconstructed;
        for (std::size_t i = 0; i != view.size(); ++i)
            reconstructed.push_back(view[i]);
        EXPECT_EQ(reconstructed, writer.data);
        EXPECT_EQ(IdentityPoolHasher<AdaptiveBytes>{}(view), adaptive_route_hash(writer.data));
        const auto slot = normal.intern(writer.data);
        EXPECT_EQ(collisions.intern(writer.data), slot);
        EXPECT_EQ(normal.find_borrowed(view), slot);
        EXPECT_EQ(collisions.find_borrowed(view), slot);
        original.push_back(std::move(writer.data));
    }
    for (std::size_t i = 0; i != values.size(); ++i)
        for (std::size_t j = 0; j != values.size(); ++j)
        {
            CartesianSupportRangeKeyView a{prototype.data, 8, values[i]}, b{prototype.data, 8, values[j]};
            const auto compare = [&]
            {
                for (std::size_t byte = 0; byte != a.size(); ++byte)
                    if (a[byte] != b[byte])
                        return a[byte] < b[byte];
                return false;
            };
            EXPECT_EQ(compare(), original[i] < original[j]);
        }
    auto absent = prototype.data;
    absent.front() ^= 1;
    EXPECT_FALSE(normal.find_borrowed(CartesianSupportRangeKeyView{absent, 8, 0}));
    EXPECT_FALSE(collisions.find_borrowed(CartesianSupportRangeKeyView{absent, 8, 0}));
}
TEST(CartesianSourceRanges, RealCompleteLineRunPreservesOrderedResultsForkAndThrowingPublisher)
{
    auto arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    IdentityBudget identities;
    IncidencePools pools(*arena, identities, arena);
    PooledLineRegistry registry(*arena, pools);
    Writer key;
    key.bytes(std::array<char, 1>{'d'});
    key.bytes(PLIC::canonical_key(PLIC::details::GeometryIdentityAccess::line({0, {1, 0, 0}, {0, 0, 0}})));
    const auto line = pools.lines.intern(std::move(key.data));
    std::vector<PooledLineEdit> edits;
    for (unsigned i = 0; i < 16; ++i)
    {
        PooledLineSubscriber sub;
        sub.support = pools.supports.intern(AdaptiveBytes{'s', char(i)});
        sub.resident = sub.support_authority = 0;
        sub.first = pools.points.intern(PLIC::details::GeometryIdentityAccess::point({(long double) i, 0, 0}));
        sub.last = pools.points.intern(PLIC::details::GeometryIdentityAccess::point({(long double) (i + 1), 0, 0}));
        sub.source_generation = 1;
        sub.native_source = true;
        edits.push_back({line, std::move(sub), false});
    }
    registry.begin_replay(true);
    registry.apply(edits);
    const auto before = registry.active_storage();
    std::size_t replies = 0;
    registry.recompute(
        [&](auto actual_line, auto query)
        {
            EXPECT_TRUE(actual_line == line && query < edits.size());
            const auto result = registry.query_result(line, query);
            EXPECT_TRUE(result.result_size() == 2 && result.subscriber.source_generation == 1 &&
                        result.subscriber.native_source);
            EXPECT_TRUE(
                PLIC::details::GeometryIdentityAccess::point(pools.points.get(result.result_at(0)))[0] == query);
            EXPECT_TRUE(
                PLIC::details::GeometryIdentityAccess::point(pools.points.get(result.result_at(1)))[0] == query + 1);
            ++replies;
        });
    registry.finish_replay(0);
    const auto after = registry.active_storage();
    EXPECT_TRUE(replies == 16 && after.run_records == 16 && after.runs == 1 && after.explicit_records == 0);
    EXPECT_TRUE(after.run_bytes < before.explicit_bytes);
    auto candidate_arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    IdentityBudget candidate_identities;
    IncidencePools candidate_pools(pools, *candidate_arena, candidate_identities, candidate_arena);
    PooledLineRegistry candidate(registry, *candidate_arena, candidate_pools);
    candidate.begin_replay(false);
    auto stale = edits.front();
    stale.remove = true;
    ++stale.subscriber.source_generation;
    EXPECT_TRUE(invalid([&] { candidate.apply(std::span(&stale, 1)); }));
    auto remove = edits.front();
    remove.remove = true;
    candidate.apply(std::span(&remove, 1));
    candidate.recompute([](auto, auto) {});
    candidate.finish_replay(0);
    EXPECT_TRUE(candidate.active_storage().run_records == 15 && registry.active_storage().run_records == 16);
    // A throwing publisher cannot retain complete-line construction scratch.
    PooledLineRegistry throwing(*arena, pools);
    throwing.begin_replay(true);
    throwing.apply(edits);
    const auto charged = arena->used;
    const auto old_body = throwing.active_storage().explicit_bytes;
    EXPECT_TRUE(invalid([&] { throwing.recompute([](auto, auto) { throw std::runtime_error("line publication"); }); }));
    const auto new_body = throwing.active_storage().run_bytes;
    EXPECT_TRUE(new_body && arena->used == charged - old_body + new_body);
}
} // namespace
