/** Borrowed-owner credit, alias identity and physical retirement order oracles. */
#include "geometry/mesh/AMR/DeferredOwnerCharges.hh"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
struct NodeTrace
{
    struct Allocation
    {
        void* pointer = nullptr;
        bool freed = false;
        std::size_t used_when_freed = 0;
    };
    D::Budget* budget = nullptr;
    std::array<Allocation, 4> allocations{};
    std::size_t count = 0;
    bool capture = false;
};
thread_local NodeTrace* active_node_trace = nullptr;
thread_local bool fail_next_allocation = false;
void observe_delete(void* pointer) noexcept
{
    if (active_node_trace)
        for (auto& allocation : active_node_trace->allocations)
            if (pointer && allocation.pointer == pointer && !allocation.freed)
            {
                allocation.freed = true;
                allocation.used_when_freed = active_node_trace->budget->used;
            }
}
struct TraceScope
{
    NodeTrace* previous;
    explicit TraceScope(NodeTrace& trace) : previous(std::exchange(active_node_trace, &trace)) {}
    ~TraceScope() { active_node_trace = previous; }
};
} // namespace

// Restrict observations to helper-node construction and destruction, following
// the existing geometry allocation tests. Payload controls use their allocator.
void* operator new(std::size_t size)
{
    if (std::exchange(fail_next_allocation, false))
        throw std::bad_alloc();
    if (void* value = std::malloc(size ? size : 1))
    {
        if (active_node_trace && active_node_trace->capture)
        {
            if (active_node_trace->count < active_node_trace->allocations.size())
                active_node_trace->allocations[active_node_trace->count].pointer = value;
            ++active_node_trace->count;
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
    observe_delete(value);
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    observe_delete(value);
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    observe_delete(value);
    std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    observe_delete(value);
    std::free(value);
}

namespace
{
constexpr std::size_t ample = 1024 * 1024;
struct Events
{
    D::Budget* budget;
    std::size_t body_destructions = 0, allocations = 0, deallocations = 0, allocated_bytes = 0;
    std::size_t used_at_body_destruction = 0, used_at_control_deallocation = 0;
};
struct Body
{
    Events* events;
    int first = 17, second = 29;
    explicit Body(Events& observer) : events(&observer) {}
    ~Body()
    {
        ++events->body_destructions;
        events->used_at_body_destruction = events->budget->used;
    }
};
template<class T> struct ObservedAllocator
{
    using value_type = T;
    Events* events;
    explicit ObservedAllocator(Events& observer) : events(&observer) {}
    template<class U> ObservedAllocator(const ObservedAllocator<U>& other) : events(other.events) {}
    T* allocate(std::size_t count)
    {
        auto* result = std::allocator<T>{}.allocate(count);
        ++events->allocations;
        events->allocated_bytes += count * sizeof(T);
        return result;
    }
    void deallocate(T* value, std::size_t count) noexcept
    {
        ++events->deallocations;
        events->used_at_control_deallocation = events->budget->used;
        std::allocator<T>{}.deallocate(value, count);
    }
    template<class U> bool operator==(const ObservedAllocator<U>& other) const { return events == other.events; }
};
std::shared_ptr<Body> body(Events& events)
{
    // This deleter never returns budget credit. The body is manually charged
    // in each admission domain; the helper owns only that transferred credit.
    return std::allocate_shared<Body>(ObservedAllocator<Body>(events), events);
}
std::size_t node_credit()
{
    D::Budget budget{ample};
    auto value = std::make_shared<int>(7);
    budget.add(257);
    D::DeferredOwnerCharges retired(budget);
    if (!retired.adopt(value, 257))
        throw std::logic_error("Fresh owner was unexpectedly already deferred.");
    return retired.metadata_bytes();
}
class DeferredOwnerCharges : public ::testing::Test
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(active_node_trace, nullptr);
        EXPECT_FALSE(fail_next_allocation);
    }
};

TEST_F(DeferredOwnerCharges, PinnedSourcePreventsSpendingBorrowedBodyCreditUntilReap)
{
    const auto node = node_credit();
    constexpr std::size_t baseline = 53, body_bytes = 4096, scratch = 23;
    D::Budget budget{baseline + body_bytes + node + scratch};
    budget.add(baseline);
    budget.add(body_bytes);
    std::size_t component_credit = body_bytes;
    Events events{&budget};
    auto source = body(events), candidate = source;
    {
        D::DeferredOwnerCharges retired(budget);
        const auto strong = source.use_count();
        const bool transferred = retired.adopt(candidate, component_credit);
        ASSERT_TRUE(transferred);
        if (transferred)
            component_credit -= body_bytes;
        EXPECT_EQ(source.use_count(), strong);
        EXPECT_EQ(component_credit, 0u);
        EXPECT_EQ(retired.body_bytes(), body_bytes);
        EXPECT_EQ(retired.metadata_bytes(), node);
        candidate.reset();
        retired.reap();
        EXPECT_EQ(events.body_destructions, 0u);
        EXPECT_EQ(source->first, 17);
        EXPECT_EQ(budget.used, baseline + body_bytes + node);
        budget.add(scratch);
        EXPECT_THROW(budget.add(1), std::invalid_argument);
        const auto spent = budget.cumulative;
        source.reset();
        EXPECT_EQ(events.body_destructions, 1u);
        EXPECT_EQ(events.deallocations, 0u); // The weak entry still retains the inline control allocation.
        EXPECT_EQ(budget.used, budget.limit);
        retired.reap();
        EXPECT_EQ(events.deallocations, 1u);
        EXPECT_EQ(retired.owners(), 0u);
        EXPECT_EQ(retired.retained_bytes(), 0u);
        EXPECT_EQ(budget.used, baseline + scratch);
        EXPECT_EQ(budget.cumulative, spent);
        EXPECT_NO_THROW(budget.add(body_bytes));
        budget.used -= body_bytes + scratch;
    }
    EXPECT_EQ(budget.used, baseline);
}

TEST_F(DeferredOwnerCharges, AliasesWithDifferentPointersTransferOneOwnerCredit)
{
    constexpr std::size_t baseline = 37, body_bytes = 512;
    D::Budget budget{ample};
    budget.add(baseline + body_bytes);
    Events events{&budget};
    auto source = body(events);
    std::shared_ptr<const int> first(source, &source->first), second(source, &source->second);
    ASSERT_NE(first.get(), second.get());
    {
        D::DeferredOwnerCharges retired(budget);
        EXPECT_TRUE(retired.adopt(first, body_bytes));
        EXPECT_TRUE(retired.contains(source));
        EXPECT_TRUE(retired.contains(second));
        const auto used = budget.used, spent = budget.cumulative, metadata = retired.metadata_bytes();
        EXPECT_FALSE(retired.adopt(second, body_bytes));
        EXPECT_EQ(retired.owners(), 1u);
        EXPECT_EQ(retired.body_bytes(), body_bytes);
        EXPECT_EQ(retired.metadata_bytes(), metadata);
        EXPECT_EQ(budget.used, used);
        EXPECT_EQ(budget.cumulative, spent);
        source.reset();
        first.reset();
        retired.reap();
        EXPECT_EQ(events.body_destructions, 0u);
        EXPECT_TRUE(retired.contains(second));
        EXPECT_FALSE(retired.adopt(second, body_bytes)); // The last local alias also observes the prior transfer.
        EXPECT_EQ(budget.used, used);
        second.reset();
        retired.reap();
        EXPECT_EQ(events.body_destructions, 1u);
        EXPECT_EQ(events.deallocations, 1u);
        EXPECT_EQ(retired.owners(), 0u);
        EXPECT_EQ(budget.used, baseline);
        retired.reap();
        EXPECT_EQ(budget.used, baseline);
        EXPECT_EQ(budget.cumulative, spent);
    }
    EXPECT_EQ(budget.used, baseline);
}

TEST_F(DeferredOwnerCharges, EqualValuesWithDistinctOwnersKeepIndependentCredits)
{
    constexpr std::size_t baseline = 19, first_bytes = 127, second_bytes = 251;
    const auto node = node_credit();
    D::Budget budget{ample};
    budget.add(baseline + first_bytes + second_bytes);
    Events first_events{&budget}, second_events{&budget};
    auto first = body(first_events), second = body(second_events);
    EXPECT_EQ(first->first, second->first);
    EXPECT_EQ(first->second, second->second);
    D::DeferredOwnerCharges retired(budget);
    EXPECT_TRUE(retired.adopt(first, first_bytes));
    EXPECT_FALSE(retired.contains(second));
    EXPECT_TRUE(retired.adopt(second, second_bytes));
    EXPECT_EQ(retired.owners(), 2u);
    EXPECT_EQ(retired.body_bytes(), first_bytes + second_bytes);
    EXPECT_EQ(retired.metadata_bytes(), 2 * node);
    const auto spent = budget.cumulative;
    first.reset();
    retired.reap();
    EXPECT_EQ(retired.owners(), 1u);
    EXPECT_EQ(retired.body_bytes(), second_bytes);
    EXPECT_EQ(budget.used, baseline + second_bytes + node);
    EXPECT_EQ(first_events.deallocations, 1u);
    EXPECT_EQ(second_events.body_destructions, 0u);
    EXPECT_TRUE(retired.contains(second));
    second.reset();
    retired.reap();
    EXPECT_EQ(second_events.deallocations, 1u);
    EXPECT_EQ(budget.used, baseline);
    EXPECT_EQ(budget.cumulative, spent);
}

TEST_F(DeferredOwnerCharges, ExactWeakNodeCreditAcceptsAndOneByteShortPreservesCallerState)
{
    const auto node = node_credit();
    ASSERT_GT(node, 0u);
    constexpr std::size_t baseline = 61, body_bytes = 1024;
    auto source = std::make_shared<int>(41);
    const auto* address = source.get();
    D::Budget exact{baseline + body_bytes + node};
    exact.add(baseline + body_bytes);
    std::size_t component_credit = body_bytes;
    {
        D::DeferredOwnerCharges retired(exact);
        const bool transferred = retired.adopt(source, component_credit);
        EXPECT_TRUE(transferred);
        if (transferred)
            component_credit -= body_bytes;
        EXPECT_EQ(exact.used, exact.limit);
        EXPECT_EQ(retired.metadata_bytes(), node);
    }
    EXPECT_EQ(exact.used, baseline);
    EXPECT_EQ(component_credit, 0u);
    EXPECT_EQ(source.get(), address);
    EXPECT_EQ(*source, 41);
    EXPECT_EQ(source.use_count(), 1);

    D::Budget short_budget{baseline + body_bytes + node - 1};
    short_budget.add(baseline + body_bytes);
    component_credit = body_bytes;
    const auto before_used = short_budget.used, before_spent = short_budget.cumulative;
    {
        D::DeferredOwnerCharges retired(short_budget);
        EXPECT_THROW(
            {
                if (retired.adopt(source, component_credit))
                    component_credit -= body_bytes;
            },
            std::invalid_argument);
        EXPECT_EQ(component_credit, body_bytes);
        EXPECT_FALSE(retired.contains(source));
        EXPECT_EQ(retired.owners(), 0u);
        EXPECT_EQ(retired.retained_bytes(), 0u);
        EXPECT_EQ(short_budget.used, before_used);
        EXPECT_EQ(short_budget.cumulative, before_spent);
        EXPECT_EQ(source.get(), address);
        EXPECT_EQ(*source, 41);
        EXPECT_EQ(source.use_count(), 1);
    }
    EXPECT_EQ(short_budget.used, before_used);
    short_budget.used -= component_credit;
    EXPECT_EQ(short_budget.used, baseline);
    RecordProperty("weak_node_credit_bytes", std::to_string(node));
}

TEST_F(DeferredOwnerCharges, InvalidAndConflictingDuplicateCreditsLeaveOneAdmission)
{
    constexpr std::size_t baseline = 43, body_bytes = 613;
    D::Budget budget{ample};
    budget.add(baseline + body_bytes);
    auto source = std::make_shared<int>(67);
    {
        D::DeferredOwnerCharges retired(budget);
        EXPECT_TRUE(retired.adopt(source, body_bytes));
        const auto used = budget.used, spent = budget.cumulative, metadata = retired.metadata_bytes();
        EXPECT_THROW((void) retired.adopt(source, body_bytes - 1), std::invalid_argument);
        EXPECT_THROW((void) retired.adopt(source, body_bytes + 1), std::invalid_argument);
        EXPECT_THROW((void) retired.adopt(source, 0), std::invalid_argument);
        EXPECT_THROW((void) retired.adopt(source, used + 1), std::invalid_argument);
        EXPECT_THROW((void) retired.adopt(std::shared_ptr<int>{}, body_bytes), std::invalid_argument);
        EXPECT_TRUE(retired.contains(source));
        EXPECT_EQ(retired.owners(), 1u);
        EXPECT_EQ(retired.body_bytes(), body_bytes);
        EXPECT_EQ(retired.metadata_bytes(), metadata);
        EXPECT_EQ(budget.used, used);
        EXPECT_EQ(budget.cumulative, spent);
        EXPECT_EQ(*source, 67);
    }
    EXPECT_EQ(budget.used, baseline);
    EXPECT_EQ(source.use_count(), 1);
}

TEST_F(DeferredOwnerCharges, FailedNodeAllocationRefundsLiveMetadataButKeepsCumulativeAdmission)
{
    constexpr std::size_t baseline = 31, body_bytes = 337;
    const auto node = node_credit();
    D::Budget budget{ample};
    budget.add(baseline + body_bytes);
    auto source = std::make_shared<int>(71);
    std::size_t component_credit = body_bytes;
    {
        D::DeferredOwnerCharges retired(budget);
        const auto used = budget.used, spent = budget.cumulative;
        bool failed = false;
        fail_next_allocation = true;
        try
        {
            if (retired.adopt(source, component_credit))
                component_credit -= body_bytes;
        }
        catch (const std::bad_alloc&)
        {
            failed = true;
        }
        fail_next_allocation = false;
        EXPECT_TRUE(failed);
        EXPECT_EQ(component_credit, body_bytes);
        EXPECT_EQ(budget.used, used);
        EXPECT_EQ(budget.cumulative, spent + node);
        EXPECT_EQ(retired.owners(), 0u);
        EXPECT_EQ(retired.retained_bytes(), 0u);
        EXPECT_FALSE(retired.contains(source));
        EXPECT_EQ(source.use_count(), 1);
        EXPECT_TRUE(retired.adopt(source, component_credit));
        component_credit = 0;
        source.reset();
        retired.reap();
        EXPECT_EQ(budget.used, baseline);
        EXPECT_EQ(budget.cumulative, spent + 2 * node);
    }
    EXPECT_EQ(budget.used, baseline);
}

TEST_F(DeferredOwnerCharges, ReapDestroysControlAndBookkeepingBeforeReturningTheirCredit)
{
    constexpr std::size_t baseline = 97;
    D::Budget budget{ample};
    budget.add(baseline);
    Events events{&budget};
    auto source = body(events);
    const auto body_bytes = events.allocated_bytes;
    ASSERT_EQ(events.allocations, 1u);
    ASSERT_GT(body_bytes, 0u);
    budget.add(body_bytes);
    D::DeferredOwnerCharges retired(budget);
    NodeTrace trace;
    trace.budget = &budget;
    {
        TraceScope scope(trace);
        trace.capture = true;
        const bool transferred = retired.adopt(source, body_bytes);
        trace.capture = false;
        EXPECT_TRUE(transferred);
        ASSERT_GT(trace.count, 0u);
        ASSERT_LE(trace.count, trace.allocations.size());
        const auto admitted = budget.used, spent = budget.cumulative;
        source.reset();
        EXPECT_EQ(events.body_destructions, 1u);
        EXPECT_EQ(events.used_at_body_destruction, admitted);
        EXPECT_EQ(events.deallocations, 0u);
        retired.reap();
        EXPECT_EQ(events.deallocations, 1u);
        EXPECT_EQ(events.used_at_control_deallocation, admitted);
        for (std::size_t i = 0; i < trace.count; ++i)
        {
            EXPECT_TRUE(trace.allocations[i].freed);
            EXPECT_EQ(trace.allocations[i].used_when_freed, admitted);
        }
        EXPECT_EQ(budget.used, baseline);
        EXPECT_EQ(budget.cumulative, spent);
        EXPECT_EQ(retired.retained_bytes(), 0u);
        RecordProperty("observed_control_allocation_bytes", std::to_string(body_bytes));
        RecordProperty("observed_helper_node_allocations", std::to_string(trace.count));
    }
}

TEST_F(DeferredOwnerCharges, AdmissionDomainExitReleasesCreditWithoutChangingSurvivingOwner)
{
    D::Budget budget{ample};
    Events events{&budget};
    auto source = body(events);
    const auto* address = source.get();
    budget.add(events.allocated_bytes);
    std::size_t component_credit = events.allocated_bytes, spent = 0;
    {
        D::DeferredOwnerCharges retired(budget);
        const bool transferred = retired.adopt(source, component_credit);
        EXPECT_TRUE(transferred);
        if (transferred)
            component_credit = 0;
        spent = budget.cumulative;
        EXPECT_EQ(retired.owners(), 1u);
        EXPECT_EQ(source.use_count(), 1);
    }
    EXPECT_EQ(component_credit, 0u);
    EXPECT_EQ(budget.used, 0u);
    EXPECT_EQ(budget.cumulative, spent);
    EXPECT_EQ(source.get(), address);
    EXPECT_EQ(source->first, 17);
    EXPECT_EQ(source->second, 29);
    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(events.body_destructions, 0u);
    EXPECT_EQ(events.deallocations, 0u);
    source.reset();
    EXPECT_EQ(events.body_destructions, 1u);
    EXPECT_EQ(events.deallocations, 1u);
}
} // namespace
