#include "cartesian_incidence_test_helpers.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
static_assert(!std::is_default_constructible_v<CartesianCompletionReceipt>);
static_assert(!std::is_default_constructible_v<CartesianCompleteSupport>);
static_assert(!std::is_default_constructible_v<CartesianCompleteLine>);
static_assert(!std::is_default_constructible_v<CartesianSealedTile>);
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
std::vector<AdaptiveLeaf> cells(const AMRGrid& value)
{
    return leaves(value);
}
TEST(CartesianIncidenceAuthority, ContextForkAndExplicitContributorGenerationsAreAuthenticated)
{
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.max_level = 0;
    auto native = cells(grid);
    auto arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    const ContactHaloContext old{1, 2, 3, 4, 5}, fresh{9, 10, 11, 12, 13};
    auto domain = CartesianGroupAuthority::make_arena(*arena, arena);
    auto stale = CartesianIncidenceSource::make(grid, native, {'d'}, 0, 1, old, *arena, arena);
    auto current = CartesianIncidenceSource::make(grid, native, {'d'}, 0, 1, fresh, *arena, arena);
    auto authority = CartesianGroupAuthority::make(fresh, *arena, arena, AdaptiveClosureOptions{}.tile_budget, domain);
    std::array stale_sources{stale};
    EXPECT_TRUE(invalid([&] { authority->replace_support_contributors(stale_sources, {}); }));
    auto leaf = native.front();
    leaf.surfaces = cartesian_adaptive_kernel(grid)->surfaces(leaf);
    auto stale_explicit = CartesianExplicitSupportSource::make(leaf, 0, 1, 1, *arena, arena, old);
    std::array old_support{stale_explicit};
    EXPECT_TRUE(invalid([&] { authority->replace_support_contributors({}, old_support); }));
    auto current_explicit = CartesianExplicitSupportSource::make(leaf, 0, 1, 1, *arena, arena, fresh);
    std::array explicit_support{current_explicit};
    authority->replace_support_contributors({}, explicit_support);
    auto group = leaf.surfaces.front().key;
    auto receipt = authority->support_completions(group, *arena, arena);
    auto output = authority->complete_support(group, receipt->receipts(), *arena, arena);
    EXPECT_TRUE(output->tile_count() == 1);
    auto point = PLIC::details::GeometryIdentityAccess::point({0, 0, 0});
    const std::array subscriptions{LineSubscription{point, PLIC::details::GeometryIdentityAccess::point({1, 0, 0})}};
    auto old_line = CartesianExplicitLineSource::make(
        std::array<char, 1>{'i'}, std::array<char, 1>{'l'}, subscriptions, 0, 1, 1, *arena, arena, UINT64_MAX, old);
    std::array old_lines{old_line};
    EXPECT_TRUE(invalid([&] { authority->replace_line_contributors({}, old_lines); }));
    auto current_line = CartesianExplicitLineSource::make(
        std::array<char, 1>{'i'}, std::array<char, 1>{'l'}, subscriptions, 0, 1, 1, *arena, arena, UINT64_MAX, fresh);
    std::array new_lines{current_line};
    authority->replace_line_contributors({}, new_lines);
    auto complete = authority->line_completions(std::array<char, 1>{'l'}, 0, *arena, arena);
    auto line = authority->complete_line(std::array<char, 1>{'l'}, 0, complete->receipts(), *arena, arena);
    EXPECT_TRUE(line->subscription_count() == 1);
    CartesianReplayRecipe recipe;
    auto certificate = CartesianReplayCertificate::from_line(*line, 0, recipe, *arena, arena);
    auto child_budget = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    auto child_domain = CartesianGroupAuthority::make_arena(*child_budget, child_budget);
    auto advanced = fresh;
    ++advanced.round;
    auto child = authority->fork(advanced, *child_budget, child_budget, child_domain);
    std::size_t parent_values = 0, child_values = 0;
    certificate->replay(*authority, *arena, [&](auto) { ++parent_values; });
    certificate->replay(*child, *child_budget, [&](auto) { ++child_values; });
    EXPECT_TRUE(parent_values == 2 && child_values == 2);
    auto rebound_empty = child->support_completions(std::array<char, 1>{'z'}, *child_budget, child_budget);
    EXPECT_TRUE(rebound_empty->receipts().size() == 1 && rebound_empty->receipts()[0]->member_count() == 0);
    auto revised_stale = CartesianExplicitLineSource::make(std::array<char, 1>{'i'}, std::array<char, 1>{'l'},
        subscriptions, 0, 2, 2, *child_budget, child_budget, UINT64_MAX, fresh);
    std::array stale_revision{revised_stale};
    EXPECT_TRUE(invalid([&] { child->replace_line_contributors({}, stale_revision); }));
    certificate->replay(*authority, *arena, [](auto) {});
    auto empty_source = CartesianExplicitLineSource::make(std::array<char, 1>{'e'}, std::array<char, 1>{'e'}, {}, 0, 2,
        2, *child_budget, child_budget, UINT64_MAX, advanced);
    std::array empty_lines{empty_source};
    child->replace_line_contributors({}, empty_lines);
    auto empty_receipts = child->line_completions(std::array<char, 1>{'e'}, 0, *child_budget, child_budget);
    auto empty_result =
        child->complete_line(std::array<char, 1>{'e'}, 0, empty_receipts->receipts(), *child_budget, child_budget);
    EXPECT_TRUE(empty_result->subscription_count() == 0);
    EXPECT_TRUE(invalid([&] { certificate->replay(*child, *child_budget, [](auto) {}); }));
    auto bad_context = advanced;
    ++bad_context.topology;
    EXPECT_TRUE(invalid([&] { auto bad = authority->fork(bad_context, *child_budget, child_budget, child_domain); }));
    certificate->replay(*authority, *arena, [](auto) {});
}
void literal_lines(bool tied)
{
    using Access = PLIC::details::GeometryIdentityAccess;
    auto point = [](long double x) { return Access::point(std::array<long double, 3>{x, 0, 0}); };
    const auto end = tied ? std::ldexp(1.L, -8000) : 2.L;
    const auto first = tied ? std::ldexp(1.L, -9000) : .5L;
    const auto second = tied ? std::ldexp(1.L, -9001) : 1.5L;
    std::vector<LineSubscription> input{{point(0), point(end)}, {point(end), point(0)}, {point(0), point(end)},
        {point(first), point(tied ? end : second)}};
    if (tied)
        input.push_back({point(second), point(end)});
    Writer key;
    key.bytes(std::array<char, 1>{'l'});
    key.bytes(PLIC::canonical_key(Access::line({0, {1, 0, 0}, {0, 0, 0}})));
    auto arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    auto explicit_source =
        CartesianExplicitLineSource::make(std::array<char, 1>{'q'}, key.data, input, 0, 7, 11, *arena, arena);
    auto empty = CartesianExplicitLineSource::make(
        std::array<char, 1>{'e'}, std::array<char, 1>{'z'}, {}, 0, 7, 11, *arena, arena);
    auto authority_domain = CartesianGroupAuthority::make_arena(*arena, arena);
    auto authority =
        CartesianGroupAuthority::make({}, *arena, arena, AdaptiveClosureOptions{}.tile_budget, authority_domain);
    authority->replace_support_contributors({}, {});
    std::array sources{explicit_source, empty};
    authority->replace_line_contributors({}, sources);
    auto receipts = authority->line_completions(key.data, 0, *arena, arena);
    EXPECT_TRUE(receipts->receipts().size() == 2 && receipts->receipts()[1]->member_count() == 0);
    EXPECT_TRUE(invalid([&] { authority->complete_line(key.data, 0, receipts->receipts().first(1), *arena, arena); }));
    std::array duplicate{receipts->receipts()[0], receipts->receipts()[0]};
    EXPECT_TRUE(invalid([&] { authority->complete_line(key.data, 0, duplicate, *arena, arena); }));
    auto complete = authority->complete_line(key.data, 0, receipts->receipts(), *arena, arena);
    EXPECT_TRUE(complete->subscription_count() == input.size());
    complete->visit_queries(*arena,
        [&](std::size_t q, const auto& output)
        {
            std::vector<long double> expected;
            if (tied)
                expected = q == 1 ? std::vector<long double>{end, 0} : std::vector<long double>{0, end};
            else if (q == 1)
                expected = {end, second, first, 0};
            else if (q == 3)
                expected = {first, second};
            else
                expected = {0, first, second, end};
            EXPECT_TRUE(output.size() == expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_TRUE(output.point(i) == point(expected[i]));
        });
    const auto before = arena->used;
    EXPECT_TRUE(invalid([&]
        { complete->visit_queries(*arena, [](auto, const auto&) { throw std::runtime_error("query callback"); }); }));
    EXPECT_TRUE(arena->used == before);
    CartesianReplayRecipe recipe;
    auto certificate = CartesianReplayCertificate::from_line(*complete, 0, recipe, *arena, arena);
    EXPECT_TRUE(!certificate->implicit());
    auto child_arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
    auto child_domain = CartesianGroupAuthority::make_arena(*child_arena, child_arena);
    auto child = authority->fork(*child_arena, child_arena, child_domain);
    std::size_t n = 0;
    certificate->replay(*child, *child_arena, [&](auto) { ++n; });
    EXPECT_TRUE(n == (tied ? 2 : 4));
    authority->replace_line_contributors({}, sources);
    EXPECT_TRUE(invalid([&] { certificate->replay(*authority, *arena, [](auto) {}); }));
    n = 0;
    certificate->replay(*child, *child_arena, [&](auto) { ++n; });
    EXPECT_TRUE(n == (tied ? 2 : 4));
}

TEST(CartesianIncidenceAuthority, CompleteLineRetainsDuplicateReversalAndProjectionTieOrder)
{
    literal_lines(false);
    literal_lines(true);
}
TEST(CartesianIncidenceAuthority, InvalidLayerAndMissingRegistryRejectBeforeIndexOrEnvironmentRefresh)
{
    RestoreEnvironment restore;
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto arena = CartesianGroupAuthority::make_arena(*budget, budget);
    auto authority = CartesianGroupAuthority::make({}, *budget, budget, 262144, arena);
    const AdaptiveBytes group{'g'};
    EXPECT_THROW(authority->support_completions(group, *budget, budget), std::exception);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    EXPECT_THROW(authority->support_completions(group, *budget, budget), std::exception);
    EXPECT_THROW(authority->complete_line(group, 64, {}, *budget, budget), std::exception);
}
TEST(CartesianIncidenceAuthority, FailedCompleteSupportUnwindsPartialGeometryAndLedgerAdmission)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto selected = native.front();
    selected.surfaces = kernel->surfaces(selected);
    AdaptiveBytes group, expected_face;
    for (const auto& surface : selected.surfaces)
        if (!surface.exterior)
        {
            group = surface.key;
            expected_face = PLIC::canonical_face_key(*surface.geometry);
            break;
        }
    ASSERT_FALSE(group.empty());
    struct Result
    {
        std::size_t floor = 0, peak = 0, cumulative = 0;
        bool completed = false;
        std::string failure;
    };
    // Every attempt uses a fresh equivalent authority/ledger. Rejected
    // construction is never resumed through a poisoned geometry ledger.
    const auto attempt = [&](std::optional<std::size_t> cap, bool full_oracle)
    {
        Result result;
        auto budget = std::make_shared<Budget>(Budget{ample});
        {
            auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
            const auto immutable_identity = source->identity();
            auto arena = CartesianGroupAuthority::make_arena(*budget, budget);
            auto authority = CartesianGroupAuthority::make({}, *budget, budget, 262144, arena);
            std::array sources{source};
            authority->replace_support_contributors(sources, {});
            auto receipts = authority->support_completions(group, *budget, budget);
            result.floor = budget->used;
            if (cap)
                budget->limit = *cap;
            const auto spent = budget->cumulative;
            try
            {
                auto complete = authority->complete_support(group, receipts->receipts(), *budget, budget);
                result.peak = budget->peak; // Complete-operation bound, before optional replay observation.
                EXPECT_EQ(complete->tile_count(), 1);
                if (full_oracle)
                {
                    std::size_t tiles = 0;
                    complete->visit_tiles(*budget,
                        [&](const auto& handle, const auto& tile)
                        {
                            ++tiles;
                            EXPECT_EQ(handle.layer(), 0);
                            EXPECT_EQ(PLIC::canonical_face_key(*tile.face), expected_face);
                            EXPECT_TRUE(tile.second_member.has_value());
                            EXPECT_TRUE(tile.second_image.is_zero());
                        });
                    EXPECT_EQ(tiles, 1);
                }
                result.completed = true;
            }
            catch (const std::exception& failure)
            {
                result.failure = failure.what();
            }
            EXPECT_EQ(budget->used, result.floor) << result.failure;
            EXPECT_GE(budget->cumulative, spent);
            EXPECT_EQ(source->identity(), immutable_identity);
            result.cumulative = budget->cumulative;
            budget->limit = ample;
        }
        EXPECT_EQ(budget->used, 0);
        return result;
    };
    const auto reference = attempt({}, true);
    ASSERT_TRUE(reference.completed) << reference.failure;
    ASSERT_GT(reference.peak, reference.floor);
    RecordProperty("complete_support_reference_peak", reference.peak);
    RecordProperty("complete_support_retained_floor", reference.floor);
    RecordProperty("actual_group_occurrences", 2);
    std::size_t failures = 0;
    for (std::size_t cap = reference.floor; cap < reference.peak; cap += 256)
    {
        const auto result = attempt(cap, false);
        EXPECT_EQ(result.floor, reference.floor);
        EXPECT_FALSE(result.completed) << cap;
        EXPECT_FALSE(result.failure.empty());
        ++failures;
    }
    EXPECT_GT(failures, 0);
    const auto exact = attempt(reference.peak, false);
    EXPECT_TRUE(exact.completed) << exact.failure;
    const auto short_one = attempt(reference.peak - 1, false);
    EXPECT_FALSE(short_one.completed);
    EXPECT_NE(short_one.failure.find("budget exhausted"), std::string::npos) << short_one.failure;
}

} // namespace
