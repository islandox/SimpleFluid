#include "geometry/mesh/AMR/CompactTileIndex.hh"
#include "geometry/mesh/AMR/TileBodyPages.hh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace
{
struct AllocationRecord
{
    void* address = nullptr;
    std::size_t bytes = 0, allocated_at = 0, released_at = 0;
    bool failed = false;
};
struct ActualHeap
{
    std::array<AllocationRecord, 512> records{};
    SimpleFluid::Meshes::AMR::adaptive_detail::Budget* budget = nullptr;
    std::size_t count = 0, attempts = 0, clock = 0, live = 0, peak = 0, unadmitted = 0;
    std::size_t fail_on_attempt = 0;
    bool overflow = false;
};
thread_local ActualHeap* active_heap = nullptr;
class ScopedActualHeap
{
public:
    explicit ScopedActualHeap(ActualHeap& heap) : previous_(std::exchange(active_heap, &heap)) {}
    ~ScopedActualHeap() { active_heap = previous_; }

private:
    ActualHeap* previous_;
};
void observe_delete(void* address) noexcept
{
    if (!active_heap || !address)
        return;
    auto& heap = *active_heap;
    for (std::size_t i = heap.count; i != 0;)
    {
        auto& record = heap.records[--i];
        if (record.address == address && !record.released_at)
        {
            record.released_at = ++heap.clock;
            heap.live -= record.bytes;
            return;
        }
    }
}
} // namespace
void* operator new(std::size_t bytes)
{
    AllocationRecord* record = nullptr;
    if (active_heap)
    {
        auto& heap = *active_heap;
        ++heap.attempts;
        if (heap.count == heap.records.size())
            heap.overflow = true;
        else
        {
            record = &heap.records[heap.count++];
            record->bytes = bytes;
            record->allocated_at = ++heap.clock;
        }
        if (heap.fail_on_attempt == heap.attempts)
        {
            if (record)
                record->failed = true;
            throw std::bad_alloc();
        }
    }
    if (auto* address = std::malloc(bytes ? bytes : 1))
    {
        if (record)
        {
            record->address = address;
            auto& heap = *active_heap;
            heap.live += bytes;
            heap.peak = std::max(heap.peak, heap.live);
            if (heap.budget && heap.live > heap.budget->used)
                heap.unadmitted = std::max(heap.unadmitted, heap.live - heap.budget->used);
        }
        return address;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes)
{
    return ::operator new(bytes);
}
void operator delete(void* address) noexcept
{
    observe_delete(address);
    std::free(address);
}
void operator delete(void* address, std::size_t) noexcept
{
    ::operator delete(address);
}
void operator delete[](void* address) noexcept
{
    ::operator delete(address);
}
void operator delete[](void* address, std::size_t) noexcept
{
    ::operator delete(address);
}

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
constexpr std::size_t ample = 64 * 1024 * 1024;
constexpr auto u32max = std::numeric_limits<std::uint32_t>::max();
constexpr auto u64max = std::numeric_limits<std::uint64_t>::max();
struct Value
{
    int literal = 0;
    std::shared_ptr<int> owner;
};
using Index = D::CompactTileIndex<Value>;
using Key = Index::Key;

TEST(CompactTileIndex, LiteralUnsortedKeysRetainFullEmissionAndSupportBoundary)
{
    D::Budget budget{ample};
    {
        Index index(budget);
        {
            Index::Batch batch(index);
            batch.insert({u32max, u64max}, {71, {}});
            batch.insert({3, 7}, {19, {}});
            batch.insert({3, 0}, {23, {}});
            batch.insert({0, u64max}, {11, {}});
            batch.publish();
        }
        const std::array<Key, 4> expected{{{0, u64max}, {3, 0}, {3, 7}, {u32max, u64max}}};
        const std::array<int, 4> values{11, 23, 19, 71};
        ASSERT_EQ(index.size(), 4);
        EXPECT_FALSE(index.wide());
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_EQ(index.key_at(i), expected[i]);
            EXPECT_EQ(index.value_at(i).literal, values[i]);
            EXPECT_EQ(index.find(expected[i]), &index.value_at(i));
        }
        EXPECT_EQ(index.lower_bound({3, 1}), 2);
        EXPECT_EQ(index.lower_bound({u32max, u64max}), 3);
        EXPECT_EQ(index.lower_bound({std::size_t(u32max) + 1, 0}), 4);
        EXPECT_EQ(index.find({3, u64max}), nullptr);
        {
            Index::Batch batch(index);
            batch.insert({std::size_t(u32max) + 1, 0}, {89, {}});
            batch.publish();
        }
        EXPECT_TRUE(index.wide());
        EXPECT_EQ(index.at({std::size_t(u32max) + 1, 0}).literal, 89);
        EXPECT_EQ(index.at({u32max, u64max}).literal, 71);
        EXPECT_EQ(budget.used, index.metadata_bytes());
    }
    EXPECT_EQ(budget.used, 0);
}

TEST(CompactTileIndex, OrderedDuplicateRejectionPreservesPublishedValuesAndOwners)
{
    D::Budget budget{ample};
    Index index(budget);
    auto owner = std::make_shared<int>(51);
    {
        Index::Batch batch(index);
        batch.insert({8, 2}, {51, owner});
        batch.publish();
    }
    const auto* address = index.find({8, 2});
    {
        Index::Batch batch(index);
        batch.replace_support(8);
        batch.insert({8, u64max}, {3, {}});
        batch.insert({1, 9}, {4, {}});
        batch.insert({8, u64max}, {5, {}});
        EXPECT_THROW(batch.publish(), std::invalid_argument);
        EXPECT_EQ(index.find({8, 2}), address);
        EXPECT_EQ(index.at({8, 2}).owner, owner);
    }
    {
        Index::Batch batch(index);
        batch.insert({8, 2}, {99, {}});
        EXPECT_THROW(batch.publish(), std::invalid_argument);
    }
    EXPECT_EQ(index.size(), 1);
    EXPECT_EQ(index.at({8, 2}).literal, 51);
    EXPECT_EQ(owner.use_count(), 2);
    {
        Index::Batch batch(index);
        batch.replace_support(8);
        batch.replace_support(8);
        EXPECT_THROW(batch.publish(), std::invalid_argument);
    }
    EXPECT_EQ(index.find({8, 2}), address);
}

TEST(CompactTileIndex, MultiGroupDeleteReinsertPreservesUnchangedAddressesAndOldReaders)
{
    D::Budget budget{ample};
    Index index(budget);
    {
        Index::Batch batch(index);
        batch.insert({1, 0}, {10, {}});
        batch.insert({1, 5}, {15, {}});
        batch.insert({2, 0}, {20, {}});
        batch.insert({3, 7}, {37, {}});
        batch.publish();
    }
    const auto* untouched = index.find({2, 0});
    const auto* removed = index.find({1, 0});
    {
        Index::Batch batch(index);
        batch.replace_support(3);
        batch.replace_support(1);
        batch.insert({1, 0}, {110, {}});
        batch.insert({4, 6}, {46, {}});
        EXPECT_EQ(removed->literal, 10);
        EXPECT_EQ(index.find({1, 0}), removed);
        EXPECT_EQ(index.size(), 4);
        batch.publish();
        EXPECT_EQ(index.find({2, 0}), untouched);
        EXPECT_NE(index.find({1, 0}), removed);
    }
    ASSERT_EQ(index.size(), 3);
    EXPECT_EQ(index.key_at(0), (Key{1, 0}));
    EXPECT_EQ(index.key_at(1), (Key{2, 0}));
    EXPECT_EQ(index.key_at(2), (Key{4, 6}));
    EXPECT_EQ(index.at({1, 0}).literal, 110);
    EXPECT_EQ(index.find({1, 5}), nullptr);
    EXPECT_EQ(index.find({3, 7}), nullptr);
    for (int repeat = 0; repeat < 40; ++repeat)
    {
        Index::Batch batch(index);
        batch.replace_support(1);
        batch.insert({1, 0}, {repeat, {}});
        batch.publish();
    }
    EXPECT_EQ(index.find({2, 0}), untouched);
    EXPECT_EQ(index.at({1, 0}).literal, 39);
    EXPECT_EQ(budget.used, index.metadata_bytes());
}

TEST(CompactTileIndex, ConservativeSyntheticSlotCeilingForcesWideWithoutTruncation)
{
    using Small = D::CompactTileIndex<Value, 1>;
    D::Budget budget{ample};
    Small index(budget);
    {
        Small::Batch batch(index);
        batch.insert({0, u64max}, {9, {}});
        batch.insert({0, 0}, {8, {}});
        batch.publish();
    }
    EXPECT_FALSE(index.wide());
    const auto* old = index.find({0, u64max});
    {
        Small::Batch batch(index);
        batch.insert({0, 7}, {7, {}});
        batch.publish();
    }
    EXPECT_TRUE(index.wide());
    EXPECT_EQ(index.find({0, u64max}), old);
    EXPECT_EQ(index.at({0, u64max}).literal, 9);
    EXPECT_EQ(index.at({0, 7}).literal, 7);
}

TEST(CompactTileIndex, EmptyPageReclamationPacksDirectoryWithoutMovingTheLastSurvivor)
{
    D::Budget budget{ample};
    Index index(budget);
    {
        Index::Batch batch(index);
        for (std::size_t i = 0; i < 96; ++i)
            batch.insert({i, 0}, {int(i), {}});
        batch.publish();
    }
    const auto old_used = budget.used;
    const auto* survivor = index.find({95, 0});
    {
        Index::Batch batch(index);
        for (std::size_t i = 0; i < 95; ++i)
            batch.replace_support(i);
        batch.publish();
    }
    EXPECT_LT(budget.used, old_used);
    ASSERT_EQ(index.size(), 1);
    EXPECT_EQ(index.find({95, 0}), survivor);
    EXPECT_EQ(survivor->literal, 95);
    {
        Index::Batch batch(index);
        batch.insert({2, u64max}, {29, {}});
        batch.publish();
    }
    EXPECT_EQ(index.find({95, 0}), survivor);
    EXPECT_EQ(index.at({2, u64max}).literal, 29);
    EXPECT_EQ(budget.used, index.metadata_bytes());
}

TEST(CompactTileIndex, NoOpBatchesAllocateNothingAndResidentLimitChecksTheFinalPublication)
{
    D::Budget budget{ample};
    Index index(budget);
    {
        Index::Batch batch(index);
        batch.insert({7, 0}, {70, {}});
        batch.publish(1);
    }
    const auto admitted = budget.cumulative;
    {
        Index::Batch batch(index);
        batch.publish(1);
    }
    EXPECT_EQ(budget.cumulative, admitted);
    {
        Index::Batch batch(index);
        batch.replace_support(7);
        batch.insert({7, 4}, {74, {}});
        batch.publish(1);
    }
    const auto* address = index.find({7, 4});
    {
        Index::Batch batch(index);
        batch.insert({8, 0}, {80, {}});
        EXPECT_THROW(batch.publish(1), std::invalid_argument);
    }
    EXPECT_EQ(index.find({7, 4}), address);
    EXPECT_EQ(index.at({7, 4}).literal, 74);
    EXPECT_EQ(index.find({8, 0}), nullptr);
}

TEST(CompactTileIndex, SourceDestructionAfterClonePreservesAdoptedBodyAndHistory)
{
    struct BodyValue
    {
        D::TileBodyHandle body;
        std::shared_ptr<int> history;
        bool initialized = false;
    };
    using Bodies = D::CompactTileIndex<BodyValue>;
    D::Budget source_budget{ample}, candidate_budget{ample};
    auto source_pages = std::make_unique<D::TileBodyPages>(source_budget);
    auto source = std::make_unique<Bodies>(source_budget);
    D::PooledTile tile;
    tile.support = 31;
    tile.emission = u64max;
    tile.intrinsic_loop = {19, 2, 31, 7, 11};
    std::weak_ptr<int> history;
    {
        Bodies::Batch batch(*source);
        auto owner = std::make_shared<int>(63);
        history = owner;
        batch.insert({31, u64max}, {source_pages->append(tile), std::move(owner), true});
        batch.publish();
    }
    source_pages->freeze();
    {
        D::TileBodyPages candidate_pages(*source_pages, candidate_budget);
        Bodies candidate(candidate_budget);
        std::size_t calls = 0;
        candidate.clone_from(*source,
            [&](const BodyValue& value)
            {
                ++calls;
                return BodyValue{candidate_pages.adopt(value.body), value.history, value.initialized};
            });
        ASSERT_EQ(calls, 1);
        EXPECT_NE(candidate.find({31, u64max}), source->find({31, u64max}));
        source.reset();
        source_pages.reset();
        EXPECT_EQ(source_budget.used, 0);
        const auto& retained = candidate.at({31, u64max});
        EXPECT_EQ(*retained.history, 63);
        EXPECT_TRUE(retained.initialized);
        EXPECT_EQ(retained.body.scalars().emission, u64max);
        const auto ring = retained.body.intrinsic_loop();
        ASSERT_EQ(ring.size(), 5);
        EXPECT_EQ(ring[0], 19);
        EXPECT_EQ(ring[1], 2);
        EXPECT_EQ(ring[2], 31);
        EXPECT_EQ(ring[3], 7);
        EXPECT_EQ(ring[4], 11);
        EXPECT_FALSE(history.expired());
    }
    EXPECT_TRUE(history.expired());
    EXPECT_EQ(candidate_budget.used, 0);
}

TEST(CompactTileIndex, ExactPeakBudgetAndOneByteShortHaveLiteralAcceptanceBoundary)
{
    const auto populate = [](Index& index)
    {
        Index::Batch batch(index);
        batch.insert({3, u64max}, {31, {}});
        batch.insert({1, 4}, {14, {}});
        batch.publish();
    };
    std::size_t required = 0;
    {
        D::Budget measured{ample};
        {
            Index index(measured);
            populate(index);
            required = measured.peak;
        }
        EXPECT_EQ(measured.used, 0);
    }
    ASSERT_GT(required, 0);
    D::Budget exact{required}, short_budget{required - 1};
    {
        Index index(exact);
        EXPECT_NO_THROW(populate(index));
        EXPECT_EQ(index.at({3, u64max}).literal, 31);
    }
    {
        Index index(short_budget);
        EXPECT_THROW(populate(index), std::invalid_argument);
        EXPECT_EQ(index.size(), 0);
        EXPECT_GT(short_budget.cumulative, 0);
    }
    EXPECT_EQ(exact.used, 0);
    EXPECT_EQ(short_budget.used, 0);
}

TEST(CompactTileIndex, FailedMergeUnwindsTemporaryChargesAndPreservesPriorPublication)
{
    D::Budget budget{ample};
    Index index(budget);
    auto owner = std::make_shared<int>(27);
    {
        Index::Batch batch(index);
        batch.insert({3, 9}, {27, owner});
        batch.publish();
    }
    const auto initial_used = budget.used;
    const auto* address = index.find({3, 9});
    std::size_t spent_before_failure = 0;
    {
        Index::Batch batch(index);
        batch.replace_support(3);
        batch.insert({3, 11}, {39, {}});
        spent_before_failure = budget.cumulative;
        budget.limit = budget.used + sizeof(Index::CompactDescriptor) - 1;
        EXPECT_THROW(batch.publish(), std::invalid_argument);
        EXPECT_EQ(index.find({3, 9}), address);
        EXPECT_EQ(index.find({3, 11}), nullptr);
        EXPECT_EQ(index.at({3, 9}).owner, owner);
    }
    EXPECT_EQ(budget.used, initial_used);
    EXPECT_EQ(budget.cumulative, spent_before_failure);
    budget.limit = ample;
    {
        Index::Batch batch(index);
        batch.replace_support(3);
        batch.insert({3, 11}, {39, {}});
        batch.publish();
    }
    EXPECT_EQ(index.find({3, 9}), nullptr);
    EXPECT_EQ(index.at({3, 11}).literal, 39);
    EXPECT_EQ(owner.use_count(), 1);
}

TEST(CompactTileIndex, ManySupportReplacementBatchesHaveBoundedComparisonWork)
{
    constexpr std::size_t count = 12000;
    D::Budget budget{ample};
    Index index(budget);
    {
        Index::Batch batch(index);
        for (std::size_t i = count; i != 0; --i)
            batch.insert({i - 1, u64max}, {int(i), {}});
        batch.publish();
    }
    const auto* survivor = index.find({1, u64max});
    {
        Index::Batch batch(index);
        for (std::size_t i = count; i != 0; --i)
            if ((i - 1) % 2 == 0)
            {
                batch.replace_support(i - 1);
                batch.insert({i - 1, 17}, {-int(i), {}});
            }
        batch.publish();
    }
    ASSERT_EQ(index.size(), count);
    RecordProperty("published_tiles", int(count));
    RecordProperty("replacement_tiles", int(count / 2));
    RecordProperty("merge_comparisons", int(index.last_merge_comparisons()));
    EXPECT_LT(index.last_merge_comparisons(), 64 * count);
    EXPECT_EQ(index.find({1, u64max}), survivor);
    for (std::size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(index.key_at(i), (Key{i, i % 2 == 0 ? 17 : u64max}));
        EXPECT_EQ(index.value_at(i).literal, i % 2 == 0 ? -int(i + 1) : int(i + 1));
    }
    EXPECT_EQ(budget.used, index.metadata_bytes());
}

TEST(CompactTileIndex, ActualAllocationRequestsAdmitCookieFreeDirectoryGrowthAndOverlap)
{
    D::Budget budget{ample};
    ActualHeap heap;
    heap.budget = &budget;
    std::array<std::size_t, 3> directory_records{};
    std::array<std::size_t, 3> steady_actual{}, steady_admitted{};
    {
        ScopedActualHeap observe(heap);
        Index index(budget);
        for (std::size_t round = 0; round < 3; ++round)
        {
            Index::Batch batch(index);
            const auto first = round * 16;
            // Each round starts with a full set of sixteen-slot pages and an
            // empty staging array: staging allocation, then directory growth.
            directory_records[round] = heap.count + 1;
            batch.insert({first, 0}, {int(first), {}});
            for (std::size_t i = first + 1; i < first + 16; ++i)
                batch.insert({i, 0}, {int(i), {}});
            batch.publish();
            steady_actual[round] = heap.live;
            steady_admitted[round] = budget.used;
        }
    }
    EXPECT_FALSE(heap.overflow);
    EXPECT_EQ(heap.live, 0);
    EXPECT_EQ(heap.unadmitted, 0);
    EXPECT_EQ(budget.used, 0);
    for (std::size_t round = 0; round < 3; ++round)
    {
        const auto& directory = heap.records[directory_records[round]];
        EXPECT_FALSE(directory.failed);
        // Independent byte oracle: one native owning pointer per directory
        // slot, not metadata_bytes() and not an assumed ABI cookie allowance.
        EXPECT_EQ(directory.bytes, (std::size_t(1) << round) * sizeof(std::unique_ptr<int>));
        EXPECT_EQ(steady_actual[round], steady_admitted[round]);
        if (round)
        {
            const auto& previous = heap.records[directory_records[round - 1]];
            EXPECT_LT(previous.allocated_at, directory.allocated_at);
            EXPECT_GT(previous.released_at, directory.allocated_at);
        }
    }
    RecordProperty("actual_peak_requested_bytes", int(heap.peak));
    RecordProperty("maximum_unadmitted_requested_bytes", int(heap.unadmitted));
}

TEST(CompactTileIndex, ActualDirectoryAllocationFailureUnwindsWithoutPublishingOrLosingSource)
{
    D::Budget budget{ample};
    Index index(budget);
    {
        Index::Batch batch(index);
        for (std::size_t i = 0; i < 16; ++i)
            batch.insert({i, 0}, {int(10 + i), {}});
        batch.publish();
    }
    const auto* original = index.find({9, 0});
    const auto initial_used = budget.used, initial_cumulative = budget.cumulative;
    ActualHeap heap;
    heap.fail_on_attempt = 2; // stage descriptor succeeds; directory growth fails.
    bool failed = false;
    {
        ScopedActualHeap observe(heap);
        Index::Batch batch(index);
        try
        {
            batch.insert({17, u64max}, {71, {}});
        }
        catch (const std::bad_alloc&)
        {
            failed = true;
        }
    }
    EXPECT_TRUE(failed);
    EXPECT_FALSE(heap.overflow);
    ASSERT_EQ(heap.count, 2);
    EXPECT_EQ(heap.records[0].bytes, sizeof(Index::WideDescriptor));
    EXPECT_EQ(heap.records[1].bytes, 2 * sizeof(std::unique_ptr<int>));
    EXPECT_TRUE(heap.records[1].failed);
    EXPECT_NE(heap.records[0].released_at, 0);
    EXPECT_EQ(heap.live, 0);
    EXPECT_EQ(budget.used, initial_used);
    EXPECT_EQ(budget.cumulative - initial_cumulative, sizeof(Index::WideDescriptor) + 2 * sizeof(std::unique_ptr<int>));
    EXPECT_EQ(index.find({9, 0}), original);
    EXPECT_EQ(index.at({9, 0}).literal, 19);
    EXPECT_EQ(index.find({17, u64max}), nullptr);
    EXPECT_EQ(index.size(), 16);
}
} // namespace
