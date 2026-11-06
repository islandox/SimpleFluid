/** Persistent physical-owner counts, admission overlap and affected-path work. */
#include "geometry/mesh/AMR/ContactOwnerUsage.hh"
#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Usage = D::ContactOwnerUsage;
constexpr std::size_t ample = 64 * 1024 * 1024;
auto arena(std::size_t limit = ample)
{
    return std::make_shared<D::Budget>(D::Budget{limit});
}

std::size_t census(const Usage& usage)
{
    std::size_t bytes = 0, count = 0;
    std::set<const void*> identities;
    usage.visit_pages(
        [&](Usage::Page page)
        {
            EXPECT_TRUE(identities.insert(page.identity).second);
            EXPECT_NE(page.origin, nullptr);
            EXPECT_GT(page.object_bytes, 0u);
            EXPECT_EQ(page.bytes, page.object_bytes + 128);
            bytes += page.bytes;
            ++count;
        });
    EXPECT_EQ(bytes, usage.storage_bytes());
    EXPECT_EQ(count, usage.distinct_owners());
    return bytes;
}

TEST(ContactOwnerUsage, PhysicalRolesCountsAndConflictingDescriptionsRemainDistinct)
{
    const auto memory = arena(), other_origin = arena();
    Usage usage(memory);
    std::array<int, 2> owners{};
    const Usage::Key first{&owners[0], 1}, second{&owners[1], 1}, other_role{&owners[0], 2};
    EXPECT_TRUE(usage.retain(first, 101, memory.get()).inserted());
    const auto duplicate = usage.retain(first, 101, memory.get());
    EXPECT_EQ(duplicate.before, 1u);
    EXPECT_EQ(duplicate.after, 2u);
    EXPECT_FALSE(duplicate.inserted());
    EXPECT_TRUE(usage.retain(second, 101, memory.get()).inserted());
    EXPECT_TRUE(usage.retain(other_role, 7, memory.get()).inserted());
    EXPECT_EQ(usage.distinct_owners(), 3u);
    EXPECT_EQ(usage.payload_bytes(), 209u);
    const auto before = memory->used;
    EXPECT_THROW((void) usage.retain(first, 102, memory.get()), std::invalid_argument);
    EXPECT_THROW((void) usage.retain(first, 101, other_origin.get()), std::invalid_argument);
    EXPECT_THROW((void) usage.retain({}, 1, memory.get()), std::invalid_argument);
    EXPECT_THROW((void) usage.release({&owners[1], 2}), std::invalid_argument);
    EXPECT_EQ(memory->used, before);
    ASSERT_TRUE(usage.find(first));
    EXPECT_EQ(usage.find(first)->references, 2u);
    EXPECT_FALSE(usage.release(first).retired());
    EXPECT_TRUE(usage.release(first).retired());
    EXPECT_EQ(usage.payload_bytes(), 108u);
    EXPECT_FALSE(usage.find(first));
    EXPECT_EQ(memory->used, census(usage));
}

TEST(ContactOwnerUsage, MixedUpdatesMatchIndependentReferenceCountsAndAllocationCensus)
{
    const auto memory = arena();
    Usage usage(memory);
    std::array<int, 127> owners{};
    std::array<std::size_t, owners.size()> counts{};
    std::uint32_t state = 31;
    for (std::size_t step = 0; step < 1500; ++step)
    {
        state = state * 1664525u + 1013904223u;
        const auto index = state % owners.size();
        const Usage::Key key{&owners[index], 1};
        const auto bytes = index % 17 + 1;
        if (!counts[index] || ((state >> 16) & 1))
        {
            const auto changed = usage.retain(key, bytes, memory.get());
            EXPECT_EQ(changed.before, counts[index]);
            EXPECT_EQ(changed.after, ++counts[index]);
        }
        else
        {
            const auto changed = usage.release(key);
            EXPECT_EQ(changed.before, counts[index]);
            EXPECT_EQ(changed.after, --counts[index]);
        }
        if (step % 25)
            continue;
        std::size_t expected_bytes = 0, expected_count = 0;
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            const auto found = usage.find({&owners[i], 1});
            EXPECT_EQ(bool(found), counts[i] != 0);
            if (counts[i])
            {
                expected_bytes += i % 17 + 1;
                ++expected_count;
                ASSERT_TRUE(found);
                EXPECT_EQ(found->references, counts[i]);
            }
        }
        EXPECT_EQ(usage.payload_bytes(), expected_bytes);
        EXPECT_EQ(usage.distinct_owners(), expected_count);
        EXPECT_EQ(memory->used, census(usage));
    }
}

TEST(ContactOwnerUsage, ForkImportsCachedTotalsAndSparseEditsShareBothTestSizes)
{
    for (const std::size_t count : {4096u, 8192u})
    {
        SCOPED_TRACE(count);
        std::vector<int> owners(count + 1);
        const auto source_memory = arena(), candidate_memory = arena();
        Usage source(source_memory);
        for (std::size_t i = 0; i < count; ++i)
            (void) source.retain({&owners[i], 1}, 17, source_memory.get());
        std::set<const void*> old_pages;
        source.visit_pages([&](Usage::Page page) { old_pages.insert(page.identity); });
        auto candidate = source.fork(candidate_memory);
        EXPECT_EQ(candidate_memory->used, source.storage_bytes());
        EXPECT_EQ(candidate.last_edit_work().visited_nodes, 0u);
        EXPECT_EQ(candidate.last_edit_work().created_nodes, 0u);
        (void) candidate.retain({&owners[count / 2], 1}, 17, source_memory.get());
        const auto first_work = candidate.last_edit_work();
        (void) candidate.retain({&owners.back(), 1}, 23, candidate_memory.get());
        const auto second_work = candidate.last_edit_work();
        EXPECT_LE(first_work.visited_nodes, 4 * std::bit_width(count));
        EXPECT_LE(second_work.visited_nodes, 4 * std::bit_width(count));
        EXPECT_LE(first_work.created_nodes, 2 * std::bit_width(count));
        EXPECT_LE(second_work.created_nodes, 2 * std::bit_width(count));
        EXPECT_GT(candidate.overlap_bytes(), 0u);
        candidate.finish_overlap();
        EXPECT_EQ(candidate.overlap_bytes(), 0u);
        EXPECT_EQ(candidate_memory->used, census(candidate));
        EXPECT_EQ(source_memory->used, census(source));
        EXPECT_EQ(candidate.payload_bytes(), 17 * count + 23);
        std::size_t shared = 0;
        candidate.visit_pages([&](Usage::Page page) { shared += old_pages.contains(page.identity); });
        EXPECT_GE(shared, count - 4 * std::bit_width(count));
        EXPECT_EQ(source.find({&owners[count / 2], 1})->references, 1u);
        EXPECT_EQ(candidate.find({&owners[count / 2], 1})->references, 2u);
    }
}

TEST(ContactOwnerUsage, ExactPeakRejectsOneByteShortWithoutChangingEitherRoot)
{
    std::array<int, 258> owners{};
    const auto source_memory = arena();
    Usage source(source_memory);
    for (std::size_t i = 0; i < owners.size() - 1; ++i)
        (void) source.retain({&owners[i], 1}, 31, source_memory.get());
    const auto old_bytes = source_memory->used;
    const auto measured_memory = arena();
    {
        auto measured = source.fork(measured_memory);
        (void) measured.retain({&owners.back(), 1}, 19, measured_memory.get());
        measured.finish_overlap();
    }
    EXPECT_EQ(measured_memory->used, 0u);
    const auto required = measured_memory->peak;
    ASSERT_GT(required, source.storage_bytes());
    const auto short_memory = arena(required - 1);
    {
        auto rejected = source.fork(short_memory);
        const auto before = short_memory->used;
        EXPECT_THROW((void) rejected.retain({&owners.back(), 1}, 19, short_memory.get()), std::invalid_argument);
        EXPECT_EQ(short_memory->used, before);
        EXPECT_EQ(rejected.storage_bytes(), source.storage_bytes());
        EXPECT_EQ(rejected.payload_bytes(), source.payload_bytes());
        EXPECT_FALSE(rejected.find({&owners.back(), 1}));
    }
    EXPECT_EQ(short_memory->used, 0u);
    const auto exact_memory = arena(required);
    {
        auto retry = source.fork(exact_memory);
        EXPECT_TRUE(retry.retain({&owners.back(), 1}, 19, exact_memory.get()).inserted());
        retry.finish_overlap();
        EXPECT_EQ(exact_memory->used, census(retry));
    }
    EXPECT_EQ(exact_memory->used, 0u);
    EXPECT_EQ(source_memory->used, old_bytes);
    EXPECT_FALSE(source.find({&owners.back(), 1}));
}

TEST(ContactOwnerUsage, ThirdGenerationReleasesUnneededAncestorMetadataExactly)
{
    std::array<int, 65> owners{};
    auto first_memory = arena(), second_memory = arena();
    const auto third_memory = arena();
    std::weak_ptr first_weak(first_memory), second_weak(second_memory);
    auto first = std::make_unique<Usage>(first_memory);
    for (std::size_t i = 0; i < 64; ++i)
        (void) first->retain({&owners[i], 1}, 3, nullptr);
    auto second = std::make_unique<Usage>(*first, second_memory);
    (void) second->retain({&owners.back(), 1}, 5, nullptr);
    second->finish_overlap();
    auto third = second->fork(third_memory);
    first.reset();
    second.reset();
    first_memory.reset();
    second_memory.reset();
    EXPECT_FALSE(first_weak.expired());
    EXPECT_FALSE(second_weak.expired());
    for (auto& owner : owners)
        EXPECT_TRUE(third.release({&owner, 1}).retired());
    third.finish_overlap();
    EXPECT_EQ(third.distinct_owners(), 0u);
    EXPECT_EQ(third.payload_bytes(), 0u);
    EXPECT_EQ(third.storage_bytes(), 0u);
    EXPECT_EQ(third_memory->used, 0u);
    EXPECT_TRUE(first_weak.expired());
    EXPECT_TRUE(second_weak.expired());
}

TEST(ContactOwnerUsage, ActiveOwnerAuthenticationRejectsForeignControlWithoutAdmission)
{
    const auto memory = arena();
    Usage usage(memory);
    const auto actual = std::make_shared<int>(19);
    const std::shared_ptr<const void> owner = actual;
    const std::shared_ptr<const void> alias(actual, actual.get());
    const std::shared_ptr<const void> foreign(actual.get(), [](const void*) {});
    const std::shared_ptr<const void> unowned(std::shared_ptr<const void>{}, actual.get());
    const Usage::Key key{actual.get(), 7};
    EXPECT_TRUE(usage.retain(key, sizeof(int), nullptr, owner).inserted());
    EXPECT_TRUE(usage.authentic(key, alias));
    EXPECT_FALSE(usage.authentic(key, foreign));
    EXPECT_FALSE(usage.authentic(key, unowned));
    const auto used = memory->used, cumulative = memory->cumulative;
    EXPECT_THROW((void) usage.retain(key, sizeof(int), nullptr, foreign), std::invalid_argument);
    EXPECT_THROW((void) usage.retain(key, sizeof(int), nullptr, unowned), std::invalid_argument);
    EXPECT_THROW((void) usage.release(key, foreign), std::invalid_argument);
    EXPECT_THROW((void) usage.retain(key, sizeof(int), nullptr), std::invalid_argument);
    EXPECT_THROW((void) usage.release(key), std::invalid_argument);
    EXPECT_EQ(memory->used, used);
    EXPECT_EQ(memory->cumulative, cumulative);
    ASSERT_TRUE(usage.find(key));
    EXPECT_EQ(usage.find(key)->references, 1u);
    EXPECT_EQ(usage.retain(key, sizeof(int), nullptr, alias).after, 2u);
    EXPECT_EQ(usage.release(key, alias).after, 1u);
    EXPECT_TRUE(usage.release(key, owner).retired());
    EXPECT_EQ(memory->used, 0u);
}

TEST(ContactOwnerUsage, ControlledForkRetainsExactAuthenticationAcrossBudgetRejection)
{
    std::vector<std::shared_ptr<const int>> owners;
    const auto source_memory = arena();
    Usage source(source_memory);
    for (std::size_t i = 0; i < 65; ++i)
    {
        owners.push_back(std::make_shared<const int>(int(i)));
        (void) source.retain({owners.back().get(), 1}, sizeof(int), nullptr, owners.back());
    }
    const Usage::Key key{owners[32].get(), 1};
    const auto measured_memory = arena();
    {
        auto measured = source.fork(measured_memory);
        EXPECT_TRUE(measured.authentic(key, owners[32]));
        (void) measured.retain(key, sizeof(int), nullptr, owners[32]);
        measured.finish_overlap();
    }
    EXPECT_EQ(measured_memory->used, 0u);
    const auto short_memory = arena(measured_memory->peak - 1);
    {
        auto rejected = source.fork(short_memory);
        const auto before = short_memory->used;
        EXPECT_THROW((void) rejected.retain(key, sizeof(int), nullptr, owners[32]), std::invalid_argument);
        EXPECT_EQ(short_memory->used, before);
        EXPECT_EQ(rejected.find(key)->references, 1u);
        EXPECT_TRUE(rejected.authentic(key, owners[32]));
    }
    const auto exact_memory = arena(measured_memory->peak);
    {
        auto exact = source.fork(exact_memory);
        EXPECT_EQ(exact.retain(key, sizeof(int), nullptr, owners[32]).after, 2u);
        exact.finish_overlap();
        EXPECT_EQ(exact_memory->used, census(exact));
    }
    EXPECT_EQ(short_memory->used, 0u);
    EXPECT_EQ(exact_memory->used, 0u);
    EXPECT_EQ(source.find(key)->references, 1u);
}
} // namespace
