/** Full-value identity, shared-page lifetime and measured admission oracles. */
#include "geometry/mesh/AMR/IdentityPool.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
struct HeapActivity
{
    std::size_t calls = 0, requested_bytes = 0;
};
thread_local HeapActivity* active_heap_probe = nullptr;
class HeapProbe
{
    HeapActivity* previous_;

public:
    explicit HeapProbe(HeapActivity& activity) : previous_(std::exchange(active_heap_probe, &activity)) {}
    ~HeapProbe() { active_heap_probe = previous_; }
};
} // namespace

// As in testFrozenTopology, count actual operator-new requests only in the
// explicitly scoped operation. Assertions and input construction stay outside.
void* operator new(std::size_t size)
{
    if (void* value = std::malloc(size ? size : 1))
    {
        if (active_heap_probe)
        {
            ++active_heap_probe->calls;
            active_heap_probe->requested_bytes += size;
        }
        return value;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* value) noexcept
{
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    std::free(value);
}

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
constexpr std::size_t ample = 1024 * 1024;
using Words = std::array<std::uint64_t, 3>;
struct Identity
{
    inline static std::size_t live = 0, copies = 0, moves = 0, destroyed = 0;
    Words words;
    explicit Identity(Words value) : words(value) { ++live; }
    Identity(const Identity& value) : words(value.words)
    {
        ++live;
        ++copies;
    }
    Identity(Identity&& value) noexcept : words(value.words)
    {
        ++live;
        ++moves;
    }
    ~Identity()
    {
        --live;
        ++destroyed;
    }
    bool operator<(const Identity& other) const { return words < other.words; }
    bool operator==(const Identity& other) const { return words == other.words; }
};
using Pool = D::IdentityPool<Identity>;
std::vector<Identity> inputs(std::size_t count)
{
    std::vector<Identity> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        result.emplace_back(Words{1000 - i, i % 3, 7 * i});
    return result;
}
template<class Hash> void fill(D::IdentityPool<Identity, Hash>& pool, const std::vector<Identity>& values)
{
    for (const auto& value : values)
        (void) pool.intern(value);
}
template<class Hash>
void expect_prefix(const D::IdentityPool<Identity, Hash>& pool, const std::vector<Identity>& values,
    const std::vector<const Identity*>& addresses)
{
    EXPECT_GE(pool.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(pool.find(values[i]), std::optional<std::size_t>(i));
        EXPECT_EQ(pool.get(i).words, values[i].words);
        EXPECT_EQ(&pool.get(i), addresses[i]);
    }
}
template<class Hash> std::vector<const Identity*> addresses(const D::IdentityPool<Identity, Hash>& pool)
{
    std::vector<const Identity*> result;
    for (std::size_t i = 0; i < pool.size(); ++i)
        result.push_back(&pool.get(i));
    return result;
}
class AdaptiveIdentityPool : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(Identity::live, 0u);
        Identity::copies = Identity::moves = Identity::destroyed = 0;
    }
    void TearDown() override
    {
        EXPECT_EQ(Identity::live, 0u);
        EXPECT_EQ(active_heap_probe, nullptr);
    }
};

TEST_F(AdaptiveIdentityPool, FullValuesKeepInsertionSlotsAndStableAddressesAcrossPages)
{
    D::Budget budget{ample};
    constexpr std::size_t caller_live = 113;
    budget.add(caller_live);
    {
        Pool pool(budget);
        const std::array<Words, 5> literal{
            {{9, 4, 7}, {2, 4, 7}, {9, 4, 8}, {9, 3, 7}, {0, std::numeric_limits<std::uint64_t>::max(), 0}}};
        std::array<const Identity*, 5> original{};
        for (std::size_t i = 0; i < literal.size(); ++i)
        {
            Identity value(literal[i]);
            EXPECT_EQ(pool.intern(value), i);
            original[i] = &pool.get(i);
            value.words = {99, 99, 99};
            EXPECT_EQ(pool.get(i).words, literal[i]);
        }
        const auto extra = inputs(32);
        for (std::size_t i = 0; i < extra.size(); ++i)
            EXPECT_EQ(pool.intern(extra[i]), literal.size() + i);
        EXPECT_EQ(pool.size(), 37u);
        EXPECT_EQ(pool.owner_pages(), 3u);
        EXPECT_EQ(pool.record_capacity(), 48u);
        for (std::size_t i = 0; i < literal.size(); ++i)
        {
            EXPECT_EQ(&pool.get(i), original[i]);
            EXPECT_EQ(pool.find(Identity(literal[i])), std::optional<std::size_t>(i));
            EXPECT_EQ(pool.get(i).words, literal[i]);
        }
        EXPECT_FALSE(pool.find(Identity({9, 4, 9})));
        EXPECT_THROW((void) pool.get(pool.size()), std::invalid_argument);
        EXPECT_THROW((void) pool.get(std::numeric_limits<std::size_t>::max()), std::invalid_argument);
        EXPECT_EQ(pool.retained_bytes(), budget.used - caller_live);
        EXPECT_GT(pool.retained_bytes(), pool.metadata_bytes());
    }
    EXPECT_EQ(budget.used, caller_live);
}

TEST_F(AdaptiveIdentityPool, DuplicateInternAndQueriesMakeNoCopiesOrAllocations)
{
    const auto values = inputs(33);
    D::Budget budget{ample};
    Pool pool(budget);
    fill(pool, values);
    const auto before_used = budget.used, before_cumulative = budget.cumulative;
    const auto before_copies = Identity::copies, before_moves = Identity::moves;
    const auto before_pages = pool.owner_pages(), before_capacity = pool.record_capacity();
    HeapActivity activity;
    bool correct = true;
    {
        HeapProbe probe(activity);
        for (unsigned repetition = 0; repetition < 10; ++repetition)
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                correct &= pool.intern(values[i]) == i;
                correct &= pool.find(values[i]) == i;
                correct &= pool.get(i) == values[i];
            }
    }
    EXPECT_TRUE(correct);
    EXPECT_EQ(activity.calls, 0u);
    EXPECT_EQ(activity.requested_bytes, 0u);
    EXPECT_EQ(Identity::copies, before_copies);
    EXPECT_EQ(Identity::moves, before_moves);
    EXPECT_EQ(budget.used, before_used);
    EXPECT_EQ(budget.cumulative, before_cumulative);
    EXPECT_EQ(pool.owner_pages(), before_pages);
    EXPECT_EQ(pool.record_capacity(), before_capacity);
}

TEST_F(AdaptiveIdentityPool, ForksAppendPrivatelyAndSurviveSourceAndSourceBudgetDestruction)
{
    const auto values = inputs(17); // Fork with an incompletely occupied final page.
    const Identity source_only({4001, 7, 1}), candidate_only({4002, 7, 1});
    auto source_budget = std::make_unique<D::Budget>(D::Budget{ample});
    source_budget->add(103);
    D::Budget candidate_budget{ample}, sibling_budget{ample};
    candidate_budget.add(211);
    sibling_budget.add(307);
    auto source = std::make_unique<Pool>(*source_budget);
    fill(*source, values);
    const auto prefix = addresses(*source);
    const auto copies_before_fork = Identity::copies, moves_before_fork = Identity::moves;
    auto candidate = std::make_unique<Pool>(*source, candidate_budget);
    auto sibling = std::make_unique<Pool>(*source, sibling_budget);
    EXPECT_EQ(Identity::copies, copies_before_fork);
    EXPECT_EQ(Identity::moves, moves_before_fork);
    EXPECT_EQ(candidate->owner_pages(), 2u);
    EXPECT_EQ(sibling->owner_pages(), 2u);
    EXPECT_EQ(source->intern(source_only), 17u);
    EXPECT_EQ(candidate->intern(candidate_only), 17u);
    EXPECT_EQ(source->owner_pages(), 3u);
    EXPECT_EQ(candidate->owner_pages(), 3u);
    EXPECT_EQ(sibling->owner_pages(), 2u);
    EXPECT_EQ(sibling->size(), 17u);
    EXPECT_FALSE(source->find(candidate_only));
    EXPECT_FALSE(candidate->find(source_only));
    EXPECT_FALSE(sibling->find(source_only));
    EXPECT_FALSE(sibling->find(candidate_only));
    EXPECT_NE(&source->get(17), &candidate->get(17));
    expect_prefix(*source, values, prefix);
    expect_prefix(*candidate, values, prefix);
    expect_prefix(*sibling, values, prefix);
    sibling.reset();
    EXPECT_EQ(sibling_budget.used, 307u);
    const auto destroyed_before_source = Identity::destroyed;
    source.reset();
    EXPECT_EQ(source_budget->used, 103u);
    EXPECT_EQ(Identity::destroyed, destroyed_before_source + 1); // Only the source's private appended value dies.
    source_budget.reset();
    expect_prefix(*candidate, values, prefix);
    EXPECT_EQ(candidate->get(17), candidate_only);
    EXPECT_EQ(candidate->retained_bytes(), candidate_budget.used - 211);
    candidate.reset();
    EXPECT_EQ(candidate_budget.used, 211u);
    EXPECT_EQ(Identity::live, values.size() + 2); // Only test inputs remain.
}

TEST_F(AdaptiveIdentityPool, ExactObservedPeakAdmitsAndOneByteLessRejects)
{
    const auto values = inputs(33);
    constexpr std::size_t caller_live = 157;
    D::Budget probe{ample};
    probe.add(caller_live);
    std::size_t retained = 0;
    {
        Pool pool(probe);
        fill(pool, values);
        retained = pool.retained_bytes();
        EXPECT_EQ(pool.size(), values.size());
    }
    const auto required_peak = probe.peak;
    EXPECT_EQ(probe.used, caller_live);
    D::Budget exact{required_peak};
    exact.add(caller_live);
    {
        Pool pool(exact);
        EXPECT_NO_THROW(fill(pool, values));
        EXPECT_EQ(pool.size(), values.size());
        EXPECT_EQ(pool.retained_bytes(), retained);
        EXPECT_EQ(exact.peak, required_peak);
    }
    EXPECT_EQ(exact.used, caller_live);
    D::Budget short_budget{required_peak - 1};
    short_budget.add(caller_live);
    std::size_t spent_after_rejection = 0;
    {
        Pool pool(short_budget);
        EXPECT_THROW(fill(pool, values), std::invalid_argument);
        EXPECT_LT(pool.size(), values.size());
        EXPECT_GT(short_budget.cumulative, caller_live);
        spent_after_rejection = short_budget.cumulative;
    }
    EXPECT_EQ(short_budget.used, caller_live);
    EXPECT_EQ(short_budget.cumulative, spent_after_rejection);
    RecordProperty("required_peak_bytes", std::to_string(required_peak));
    RecordProperty("retained_pool_bytes", std::to_string(retained));
}

TEST_F(AdaptiveIdentityPool, FailedForkAndPartialAppendReturnOnlyTheirOwnAdmission)
{
    const auto values = inputs(17);
    const Identity inserted({4003, 7, 1});
    D::Budget source_budget{ample};
    source_budget.add(79);
    Pool source(source_budget);
    fill(source, values);
    const auto prefix = addresses(source);
    const auto source_used = source_budget.used, source_spent = source_budget.cumulative;
    constexpr std::size_t caller_live = 313;
    std::size_t fork_peak = 0, append_peak = 0;
    {
        D::Budget probe{ample};
        probe.add(caller_live);
        {
            Pool candidate(source, probe);
            fork_peak = probe.peak;
            (void) candidate.intern(inserted);
            append_peak = probe.peak;
        }
        EXPECT_EQ(probe.used, caller_live);
    }
    ASSERT_GT(append_peak, fork_peak);
    D::Budget failed_fork{fork_peak - 1};
    failed_fork.add(caller_live);
    EXPECT_THROW((void) Pool(source, failed_fork), std::invalid_argument);
    EXPECT_EQ(failed_fork.used, caller_live);
    EXPECT_GT(failed_fork.cumulative, caller_live);
    D::Budget failed_append{append_peak - 1};
    failed_append.add(caller_live);
    std::size_t spent_after_failure = 0;
    {
        Pool candidate(source, failed_append);
        const auto before_append = failed_append.cumulative;
        EXPECT_THROW((void) candidate.intern(inserted), std::invalid_argument);
        EXPECT_GT(failed_append.cumulative, before_append); // Failure follows partial successful admission.
        spent_after_failure = failed_append.cumulative;
        expect_prefix(source, values, prefix);
        EXPECT_FALSE(source.find(inserted));
    }
    EXPECT_EQ(failed_append.used, caller_live);
    EXPECT_EQ(failed_append.cumulative, spent_after_failure);
    EXPECT_EQ(source_budget.used, source_used);
    EXPECT_EQ(source_budget.cumulative, source_spent);
    expect_prefix(source, values, prefix);
    RecordProperty("fork_peak_bytes", std::to_string(fork_peak));
    RecordProperty("append_peak_bytes", std::to_string(append_peak));
}

// Explicit comparison model for the replaced ownership pattern: one shared
// object per identity plus a full-value map and an insertion-order slot vector.
// Its owner count is separate from the independently intercepted heap requests.
class PerRecordOwnerReference
{
    using Owner = std::shared_ptr<const Identity>;
    struct Less
    {
        using is_transparent = void;
        bool operator()(const Owner& a, const Owner& b) const { return *a < *b; }
        bool operator()(const Owner& a, const Identity& b) const { return *a < b; }
        bool operator()(const Identity& a, const Owner& b) const { return a < *b; }
    };
    std::map<Owner, std::size_t, Less> values_;
    std::vector<Owner> slots_;

public:
    std::size_t intern(const Identity& value)
    {
        const auto found = values_.find(value);
        if (found != values_.end())
            return found->second;
        const auto slot = slots_.size();
        auto owner = std::make_shared<const Identity>(value);
        values_.emplace(owner, slot);
        slots_.push_back(std::move(owner));
        return slot;
    }
    std::size_t owner_count() const { return slots_.size(); }
};

TEST_F(AdaptiveIdentityPool, MeasuredPageAllocationsReplacePerRecordOwnersWithoutPayloadMoves)
{
    constexpr std::size_t count = 96;
    const auto values = inputs(count);
    D::Budget budget{ample};
    Pool pool(budget);
    PerRecordOwnerReference reference;
    HeapActivity pooled_heap, reference_heap;
    const auto copies_before_pool = Identity::copies, moves_before_pool = Identity::moves;
    bool pooled_slots = true, reference_slots = true;
    {
        HeapProbe probe(pooled_heap);
        for (std::size_t i = 0; i < count; ++i)
            pooled_slots &= pool.intern(values[i]) == i;
    }
    EXPECT_EQ(Identity::copies - copies_before_pool, count);
    EXPECT_EQ(Identity::moves, moves_before_pool);
    const auto copies_before_reference = Identity::copies;
    {
        HeapProbe probe(reference_heap);
        for (std::size_t i = 0; i < count; ++i)
            reference_slots &= reference.intern(values[i]) == i;
    }
    EXPECT_TRUE(pooled_slots);
    EXPECT_TRUE(reference_slots);
    EXPECT_EQ(Identity::copies - copies_before_reference, count);
    EXPECT_EQ(pool.owner_pages(), 6u);
    EXPECT_EQ(pool.record_capacity(), count);
    EXPECT_EQ(reference.owner_count(), count);
    EXPECT_LT(pool.owner_pages(), reference.owner_count());
    EXPECT_LT(pooled_heap.calls, reference_heap.calls);
    EXPECT_LT(pooled_heap.requested_bytes, reference_heap.requested_bytes);
    RecordProperty("pooled_heap_calls", std::to_string(pooled_heap.calls));
    RecordProperty("reference_heap_calls", std::to_string(reference_heap.calls));
    RecordProperty("pooled_heap_requested_bytes", std::to_string(pooled_heap.requested_bytes));
    RecordProperty("reference_heap_requested_bytes", std::to_string(reference_heap.requested_bytes));
    RecordProperty("pooled_owner_pages", std::to_string(pool.owner_pages()));
    RecordProperty("reference_identity_owners", std::to_string(reference.owner_count()));
    RecordProperty("pooled_retained_bytes", std::to_string(pool.retained_bytes()));
    RecordProperty("pooled_metadata_bytes", std::to_string(pool.metadata_bytes()));
    RecordProperty("pooled_record_capacity", std::to_string(pool.record_capacity()));
    RecordProperty("pooled_unused_capacity", std::to_string(pool.record_capacity() - pool.size()));
}

TEST_F(AdaptiveIdentityPool, SmallPoolsReportTheirUnusedPageCapacity)
{
    for (const std::size_t count : {1u, 16u, 17u, 33u})
    {
        SCOPED_TRACE(count);
        const auto values = inputs(count);
        D::Budget budget{ample};
        Pool pool(budget);
        fill(pool, values);
        // Sixteen records per immutable owner page is the documented storage
        // contract; unused records are admitted capacity, not live payloads.
        const auto expected_pages = (count + 15) / 16;
        EXPECT_EQ(pool.owner_pages(), expected_pages);
        EXPECT_EQ(pool.record_capacity(), 16 * expected_pages);
        EXPECT_EQ(pool.retained_bytes(), budget.used);
        EXPECT_EQ(Identity::live, 2 * count);
        if (count > 1)
            EXPECT_LT(pool.owner_pages(), count);
        const auto prefix = "records_" + std::to_string(count);
        RecordProperty(prefix + "_owner_pages", std::to_string(pool.owner_pages()));
        RecordProperty(prefix + "_record_capacity", std::to_string(pool.record_capacity()));
        RecordProperty(prefix + "_unused_capacity", std::to_string(pool.record_capacity() - pool.size()));
        RecordProperty(prefix + "_retained_bytes", std::to_string(pool.retained_bytes()));
        RecordProperty(prefix + "_metadata_bytes", std::to_string(pool.metadata_bytes()));
    }
}

struct DynamicIdentity
{
    std::uint64_t key;
    std::vector<std::uint64_t> words;
    bool operator<(const DynamicIdentity& other) const { return key < other.key; }
};
TEST_F(AdaptiveIdentityPool, StatefulPayloadPolicyForkAdmitsDynamicCapacityAndCopiesCapturedState)
{
    std::size_t observed_bytes = 0;
    const auto payload = [&observed_bytes](const DynamicIdentity& value)
    {
        observed_bytes = value.words.capacity() * sizeof(std::uint64_t);
        return observed_bytes;
    };
    using DynamicPool = D::IdentityPool<DynamicIdentity, D::NoIdentityPoolHash, decltype(payload)>;
    static_assert(!std::is_pointer_v<DynamicPool::PayloadPolicy>);
    D::Budget baseline_budget{ample}, source_budget{ample}, candidate_budget{ample};
    std::size_t baseline_retained = 0;
    {
        DynamicPool baseline(baseline_budget, {}, {}, payload);
        EXPECT_EQ(baseline.intern(DynamicIdentity{1, {}}), 0u);
        baseline_retained = baseline.retained_bytes();
    }
    EXPECT_EQ(baseline_budget.used, 0u);
    DynamicIdentity first{1, {4, 5, 6}};
    first.words.reserve(23);
    const auto first_bytes = first.words.capacity() * sizeof(std::uint64_t);
    DynamicPool source(source_budget, {}, {}, payload);
    EXPECT_EQ(source.intern(std::move(first)), 0u);
    EXPECT_EQ(observed_bytes, first_bytes);
    EXPECT_EQ(source.retained_bytes(), baseline_retained + first_bytes);
    EXPECT_EQ(source_budget.used, source.retained_bytes());
    const auto* retained = &source.get(0);
    std::size_t append_peak = 0;
    {
        DynamicPool candidate(source, candidate_budget);
        EXPECT_EQ(&candidate.get(0), retained);
        EXPECT_EQ(candidate.retained_bytes(), source.retained_bytes());
        DynamicIdentity tail{2, {7, 8}};
        tail.words.reserve(43);
        const auto tail_bytes = tail.words.capacity() * sizeof(std::uint64_t);
        observed_bytes = 0;
        EXPECT_EQ(candidate.intern(std::move(tail)), 1u);
        EXPECT_EQ(observed_bytes, tail_bytes); // The fork retained the capturing policy.
        EXPECT_EQ(candidate.get(1).words, (std::vector<std::uint64_t>{7, 8}));
        EXPECT_EQ(source.size(), 1u);
        append_peak = candidate_budget.peak;
    }
    EXPECT_EQ(candidate_budget.used, 0u);
    for (const bool admit : {true, false})
    {
        D::Budget replay_budget{append_peak - (admit ? 0 : 1)};
        {
            DynamicPool candidate(source, replay_budget);
            DynamicIdentity tail{2, {7, 8}};
            tail.words.reserve(43);
            if (admit)
            {
                EXPECT_EQ(candidate.intern(std::move(tail)), 1u);
                EXPECT_EQ(replay_budget.peak, append_peak);
            }
            else
                EXPECT_THROW((void) candidate.intern(std::move(tail)), std::invalid_argument);
        }
        EXPECT_EQ(replay_budget.used, 0u);
        EXPECT_EQ(&source.get(0), retained);
        EXPECT_EQ(source.get(0).words, (std::vector<std::uint64_t>{4, 5, 6}));
    }
}

struct ConstantHash
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const Identity&) const noexcept { return 0; }
};
struct WordHash
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const Identity& value) const noexcept
    {
        std::uint64_t hash = 14695981039346656037ULL;
        for (const auto word : value.words)
        {
            hash ^= word;
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};
using FlatPool = D::IdentityPool<Identity, WordHash>;
static_assert(!Pool::uses_hashing);
static_assert(FlatPool::uses_hashing);
static_assert(std::is_empty_v<FlatPool::HashPolicy>);
static_assert(!std::is_pointer_v<FlatPool::HashPolicy>);
struct SeededWordHash
{
    static constexpr bool enabled = true;
    std::uint64_t seed = 0;
    std::uint64_t operator()(const Identity& value) const noexcept { return WordHash{}(value) ^ seed; }
};
TEST_F(AdaptiveIdentityPool, StatefulHashPolicyForkPreservesExistingLookupConfiguration)
{
    using StatefulPool = D::IdentityPool<Identity, SeededWordHash>;
    const auto values = inputs(17);
    D::Budget source_budget{ample}, candidate_budget{ample};
    StatefulPool source(source_budget, {}, SeededWordHash{137});
    fill(source, values);
    const auto retained = addresses(source);
    StatefulPool candidate(source, candidate_budget);
    expect_prefix(candidate, values, retained);
    const Identity appended(Words{9000, 3, 4});
    EXPECT_EQ(candidate.intern(appended), 17u);
    EXPECT_EQ(candidate.find(appended), 17u);
    EXPECT_EQ(source.find(appended), std::nullopt);
}
TEST_F(AdaptiveIdentityPool, FlatLookupChecksEveryFullKeyEvenWhenAllHashesCollide)
{
    const auto values = inputs(257);
    D::Budget budget{ample};
    D::IdentityPool<Identity, ConstantHash> pool(budget);
    fill(pool, values);
    const auto retained = addresses(pool);
    expect_prefix(pool, values, retained);
    EXPECT_EQ(pool.find(Identity(Words{9999, 0, 0})), std::nullopt);
    HeapActivity heap;
    {
        HeapProbe probe(heap);
        for (std::size_t i = 0; i < values.size(); ++i)
            if (pool.intern(values[i]) != i)
                std::abort();
    }
    EXPECT_EQ(heap.calls, 0u);
    EXPECT_EQ(pool.size(), values.size());
}
TEST_F(AdaptiveIdentityPool, FlatLookupForksKeepSlotsAndPagesAfterEitherBranchGrows)
{
    const auto values = inputs(17);
    D::Budget source_budget{ample}, candidate_budget{ample};
    auto source = std::make_unique<FlatPool>(source_budget);
    fill(*source, values);
    const auto retained = addresses(*source);
    FlatPool candidate(*source, candidate_budget);
    const Identity source_only(Words{5000, 1, 2}), candidate_only(Words{6000, 2, 3});
    EXPECT_EQ(source->intern(source_only), 17u);
    EXPECT_EQ(candidate.intern(candidate_only), 17u);
    EXPECT_EQ(source->find(candidate_only), std::nullopt);
    EXPECT_EQ(candidate.find(source_only), std::nullopt);
    source.reset();
    EXPECT_EQ(source_budget.used, 0u);
    expect_prefix(candidate, values, retained);
    EXPECT_EQ(candidate.get(17).words, candidate_only.words);
}
TEST_F(AdaptiveIdentityPool, FlatLookupGrowthAdmitsOldAndNewTablesBeforeReplacingEither)
{
    const auto values = inputs(8);
    D::Budget budget{ample};
    FlatPool pool(budget);
    for (std::size_t i = 0; i < 7; ++i)
        EXPECT_EQ(pool.intern(values[i]), i);
    ASSERT_EQ(pool.lookup_bucket_capacity(), 8u);
    const auto retained = addresses(pool);
    const auto before = budget.used, cumulative = budget.cumulative;
    const auto new_table_bytes = 2 * pool.lookup_bucket_capacity() * sizeof(std::size_t);
    budget.limit = before + new_table_bytes - 1;
    EXPECT_THROW((void) pool.intern(values[7]), std::invalid_argument);
    EXPECT_EQ(budget.used, before);
    EXPECT_EQ(budget.cumulative, cumulative);
    EXPECT_EQ(pool.size(), 7u);
    for (std::size_t i = 0; i < 7; ++i)
    {
        EXPECT_EQ(pool.find(values[i]), std::optional<std::size_t>(i));
        EXPECT_EQ(&pool.get(i), retained[i]);
    }
    ++budget.limit;
    EXPECT_EQ(pool.intern(values[7]), 7u);
    EXPECT_EQ(budget.peak, budget.limit);
    EXPECT_EQ(pool.lookup_bucket_capacity(), 16u);
}
TEST_F(AdaptiveIdentityPool, FlatLookupMeasuresFewerAllocationsThanTreeMetadata)
{
    const auto values = inputs(96);
    D::Budget tree_budget{ample}, flat_budget{ample};
    Pool tree(tree_budget);
    FlatPool flat(flat_budget);
    HeapActivity tree_heap, flat_heap;
    {
        HeapProbe probe(tree_heap);
        fill(tree, values);
    }
    {
        HeapProbe probe(flat_heap);
        fill(flat, values);
    }
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(tree.find(values[i]), flat.find(values[i]));
        EXPECT_EQ(tree.get(i).words, flat.get(i).words);
    }
    EXPECT_LT(flat_heap.calls, tree_heap.calls);
    EXPECT_LT(flat_heap.requested_bytes, tree_heap.requested_bytes);
    EXPECT_LT(flat.metadata_bytes(), tree.metadata_bytes());
    RecordProperty("flat_lookup_heap_calls", std::to_string(flat_heap.calls));
    RecordProperty("tree_lookup_heap_calls", std::to_string(tree_heap.calls));
    RecordProperty("flat_lookup_heap_requested_bytes", std::to_string(flat_heap.requested_bytes));
    RecordProperty("tree_lookup_heap_requested_bytes", std::to_string(tree_heap.requested_bytes));
    RecordProperty("flat_lookup_metadata_bytes", std::to_string(flat.metadata_bytes()));
    RecordProperty("tree_lookup_metadata_bytes", std::to_string(tree.metadata_bytes()));
}
} // namespace
