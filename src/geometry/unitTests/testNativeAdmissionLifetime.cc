/** Header-only independent requested-C++ native admission qualification. */
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/details/NativeConstructorStorage.hh"
#include "pipeline_allocation_operators.inc"
#include "pipeline_allocation_trace.hh"
#include <cstdint>
#include <gtest/gtest.h>
namespace
{
namespace N = SimpleFluid::geometry_detail;
constexpr std::size_t cap = 4 * 1024 * 1024;
struct Counter
{
    std::size_t limit, used = 0, peak = 0, cumulative = 0;
    N::NativeCounterAllocationEnvelope native_allocation;
    void add(std::size_t bytes)
    {
        if (used > limit || bytes > limit - used || bytes > std::numeric_limits<std::size_t>::max() - cumulative)
            throw std::invalid_argument("Test counter admission rejected.");
        used += bytes;
        cumulative += bytes;
        peak = std::max(peak, used);
    }
};
struct Trace
{
    Trace() { pipeline_trace::begin(); }
    ~Trace() { pipeline_trace::recording.store(false); }
};
struct alignas(128) Aligned
{
    std::array<std::byte, 128> bytes{};
};
bool raw_extent_live(const void* address, std::size_t bytes)
{
    for (const auto& entry : pipeline_trace::table)
        if (entry.live && entry.address == address && entry.bytes == bytes)
            return true;
    return false;
}
void raw_live_matches(const N::NativeConstructionAdmission& admission)
{
    const auto raw = pipeline_trace::summary();
    const auto measured = admission.statistics();
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(measured.actual_live, raw.all_live);
}
struct ConstructionFailure
{
};
struct DestroyedChild
{
    unsigned* destroyed;
    explicit DestroyedChild(unsigned& count) : destroyed(&count) {}
    ~DestroyedChild() { ++*destroyed; }
};
struct CompositePayload
{
    // Concrete children must die before their storage receipts retire.
    N::NativeConstructorStorage storage;
    std::vector<Aligned> cells;
    std::string label;
    std::unique_ptr<DestroyedChild> child;
    unsigned* destroyed;
    CompositePayload(
        const N::NativeConstructionAdmission& admission, unsigned& children, unsigned& objects, bool fail = false)
        : storage(admission), destroyed(&objects)
    {
        storage.reserve(cells, 7);
        cells.resize(7);
        storage.string(label, "a detached payload label with a concrete heap allocation");
        child = storage.object<DestroyedChild>(children);
        if (fail)
            throw ConstructionFailure{};
    }
    ~CompositePayload() { ++*destroyed; }
};
struct GraphCensus
{
    std::array<std::weak_ptr<const void>, 8> owners;
    std::array<std::size_t, 8> sizes{};
    std::size_t count = 0, bytes = 0, edges = 0;
    bool expected_prepaid = false, valid = true;
    static bool edge(void* context, const std::shared_ptr<const void>& owner, std::size_t bytes, bool prepaid)
    {
        auto& self = *static_cast<GraphCensus*>(context);
        ++self.edges;
        self.valid = self.valid && prepaid == self.expected_prepaid;
        for (std::size_t i = 0; i != self.count; ++i)
            if (!self.owners[i].owner_before(owner) && !owner.owner_before(self.owners[i]))
            {
                self.valid = self.valid && self.sizes[i] == bytes;
                return false;
            }
        if (self.count == self.owners.size())
        {
            self.valid = false;
            return false;
        }
        self.owners[self.count] = owner;
        self.sizes[self.count++] = bytes;
        self.bytes += bytes;
        return true;
    }
};
struct MirrorCounter
{
    std::size_t limit = cap, used = 4096, peak = used;
    static bool acquire(void* context, std::size_t bytes) noexcept
    {
        auto& self = *static_cast<MirrorCounter*>(context);
        if (self.used > self.limit || bytes > self.limit - self.used)
            return false;
        self.used += bytes;
        self.peak = std::max(self.peak, self.used);
        return true;
    }
    static void retire(void* context, std::size_t bytes) noexcept
    {
        auto& self = *static_cast<MirrorCounter*>(context);
        if (bytes > self.used)
            std::terminate();
        self.used -= bytes;
    }
};
void raw_matches(const N::NativeConstructionAdmission& admission)
{
    const auto raw = pipeline_trace::summary();
    const auto measured = admission.statistics();
    const auto counters = admission.counters();
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(measured.actual_live, raw.all_live);
    EXPECT_EQ(measured.actual_cumulative, raw.phase[0].bytes);
    EXPECT_EQ(
        counters.used, measured.actual_live + measured.imported_live + measured.profiled_live + measured.reserved_live);
}
TEST(NativeAdmissionLifetime, CounterBirthPreAdmitsPrefixAndKeepsItSeparateFromRawAllocations)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096);
    const N::NativeConstructionAdmission admission(counter);
    raw_matches(admission);
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(admission.statistics().imported_live, 4096u);
        EXPECT_EQ(admission.counters().used, pipeline_trace::summary().all_live + 4096);
    }
    admission.release_prefix(4096);
    raw_matches(admission);
}

TEST(NativeAdmissionLifetime, ActualAlignedNodesBucketsControlsAndWeakTailMatchRawCppRequests)
{
    const Trace trace;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        raw_matches(admission);
        const auto baseline = admission.counters().used;
        {
            auto aligned = admission.unique<Aligned>();
            {
                const pipeline_trace::Pause pause;
                EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned.get()) % 128, 0u);
            }
            raw_matches(admission);
            N::NativeConstructorMap<unsigned, std::uint64_t> ordered(
                std::less<unsigned>{}, admission.allocator<std::pair<const unsigned, std::uint64_t>>());
            N::NativeConstructorHash<unsigned, std::uint64_t> hashed(0, std::hash<unsigned>{},
                std::equal_to<unsigned>{}, admission.allocator<std::pair<const unsigned, std::uint64_t>>());
            for (unsigned key = 0; key != 130; ++key)
            {
                ordered.emplace(key, 7 * key);
                hashed.emplace(key, 11 * key);
            }
            raw_matches(admission);
        }
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(admission.counters().used, baseline);
        }
        auto strong = admission.shared<Aligned>();
        std::weak_ptr<Aligned> weak = strong;
        const auto alive = admission.counters().used;
        strong.reset();
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(weak.expired());
            EXPECT_EQ(admission.counters().used, alive);
        }
        raw_matches(admission);
        weak.reset();
        raw_matches(admission);
    }
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}

TEST(NativeAdmissionLifetime, ProfileTransfersRetireByScopeWithoutFictitiousAllocatorWork)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    const auto actual = admission.statistics();
    const auto baseline = admission.counters().used;
    {
        auto phase = admission.reserve_profile({2048});
        auto retained = phase.retain_profile(512);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(admission.statistics().reserved_live, 1536u);
            EXPECT_EQ(admission.statistics().profiled_live, 512u);
            EXPECT_EQ(admission.statistics().actual_cumulative, actual.actual_cumulative);
            EXPECT_EQ(admission.counters().used, baseline + 2048);
        }
        phase.finish_scratch();
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(admission.counters().used, baseline + 512);
        }
        retained.reset();
    }
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(admission.counters().used, baseline);
        EXPECT_EQ(admission.statistics().profiled_cumulative_upper, 2048u);
        EXPECT_EQ(admission.statistics().admission_peak, baseline + 2048);
    }
    raw_matches(admission);
}

TEST(NativeAdmissionLifetime, ConcreteVectorAndStringEarlyReleaseDestroyStorageBeforeRetirement)
{
    const pipeline_trace::Pause setup;
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    pipeline_trace::recording.store(true);
    pipeline_trace::begin();
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        N::NativeConstructorStorage receipts(admission);
        std::vector<Aligned> vector;
        std::string short_string, long_string;
        const auto baseline = admission.counters().used;
        receipts.reserve(vector, 9);
        vector.resize(9);
        receipts.string(short_string, "tiny");
        receipts.string(long_string, "a long immutable boundary label which exceeds the local string capacity");
        raw_matches(admission);
        receipts.reset_buffer(vector);
        raw_matches(admission);
        receipts.reset_buffer(long_string);
        raw_matches(admission);
        const pipeline_trace::Pause pause;
        EXPECT_EQ(admission.counters().used, baseline);
        EXPECT_EQ(receipts.exclusive_bytes(), 0u);
        EXPECT_EQ(short_string, "tiny");
    }
    pipeline_trace::recording.store(false);
}

TEST(NativeAdmissionLifetime, OneByteShortRejectsBeforeTheRequestedAllocation)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    constexpr std::size_t request = 17 * sizeof(Aligned);
    const auto before = admission.counters();
    counter.owner()->limit = before.used + request - 1;
    bool denied = false;
    try
    {
        (void) admission.allocator<Aligned>().allocate(17);
    }
    catch (const std::invalid_argument&)
    {
        denied = true;
    }
    const auto cumulative = admission.statistics().actual_cumulative;
    bool raw_request = false;
    for (const auto& entry : pipeline_trace::table)
        raw_request = raw_request || (entry.used && entry.bytes == request);
    const pipeline_trace::Pause pause;
    EXPECT_TRUE(denied);
    EXPECT_FALSE(raw_request);
    EXPECT_EQ(admission.counters().used, before.used);
    EXPECT_EQ(cumulative, before.cumulative);
}

TEST(NativeAdmissionLifetime, ReusingStateAuthenticatesItsActualSharedControl)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission original(counter);
    const auto reused = N::NativeConstructionAdmission::from_state(original.state());
    EXPECT_EQ(reused.state(), original.state());
    const std::shared_ptr<const N::NativeAllocationState> forged(original.state().get(), [](const auto*) {});
    EXPECT_THROW((void) N::NativeConstructionAdmission::from_state(forged), std::invalid_argument);
}

TEST(NativeAdmissionLifetime, EmptyUntrackedHeapStringRejectsBeforeShortAssignment)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    // This allocation predates the domain: it cannot be silently reused as
    // admitted storage merely because clear() removed its characters.
    std::string destination(173, 'x');
    destination.clear();
    const auto* buffer = destination.data();
    const auto capacity = destination.capacity();
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    N::NativeConstructorStorage receipts(admission);
    const auto before = admission.counters();
    bool denied = false;
    try
    {
        receipts.string(destination, "tiny");
    }
    catch (const std::invalid_argument&)
    {
        denied = true;
    }
    raw_live_matches(admission); // Exception text is a separate, already-freed raw request.
    const pipeline_trace::Pause pause;
    EXPECT_TRUE(denied);
    EXPECT_TRUE(destination.empty());
    EXPECT_EQ(destination.data(), buffer);
    EXPECT_EQ(destination.capacity(), capacity);
    EXPECT_EQ(receipts.exclusive_bytes(), 0u);
    EXPECT_EQ(admission.counters().used, before.used);
    EXPECT_EQ(admission.counters().cumulative, before.cumulative);
}

TEST(NativeAdmissionLifetime, ClearedTrackedHeapStringKeepsItsReceiptUntilExplicitReset)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    N::NativeConstructorStorage receipts(admission);
    std::string destination;
    const auto baseline = admission.counters().used;
    receipts.string(destination, "the first tracked heap label remains live after clear");
    const auto* first = destination.data();
    const auto bytes = destination.capacity() + 1;
    const auto before = admission.counters();
    const auto revision = receipts.revision();
    destination.clear();
    bool denied = false;
    try
    {
        receipts.string(destination, "a replacement long label must not retire the previous buffer implicitly");
    }
    catch (const std::invalid_argument&)
    {
        denied = true;
    }
    raw_live_matches(admission);
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(denied);
        EXPECT_TRUE(raw_extent_live(first, bytes));
        EXPECT_EQ(destination.data(), first);
        EXPECT_EQ(receipts.revision(), revision);
        EXPECT_EQ(admission.counters().used, before.used);
        EXPECT_EQ(admission.counters().cumulative, before.cumulative);
    }
    receipts.reset_buffer(destination);
    raw_live_matches(admission);
    {
        const pipeline_trace::Pause pause;
        EXPECT_FALSE(raw_extent_live(first, bytes));
        EXPECT_EQ(receipts.exclusive_bytes(), 0u);
        EXPECT_EQ(admission.counters().used, baseline);
    }
    receipts.string(destination, "the explicit reset permits a new tracked heap label");
    raw_live_matches(admission);
    receipts.reset_buffer(destination);
    raw_live_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(admission.counters().used, baseline);
    EXPECT_EQ(receipts.exclusive_bytes(), 0u);
}

TEST(NativeAdmissionLifetime, ExactAllocationLimitSucceedsAndOneByteShortNeverRequestsItsBuffer)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    constexpr std::size_t count = 43, bytes = count * sizeof(Aligned);
    const auto baseline = admission.counters();
    counter.owner()->limit = baseline.used + bytes - 1;
    bool denied = false;
    try
    {
        (void) admission.allocator<Aligned>().allocate(count);
    }
    catch (const std::invalid_argument&)
    {
        denied = true;
    }
    bool requested = false;
    for (const auto& entry : pipeline_trace::table)
        requested = requested || (entry.used && entry.bytes == bytes);
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(denied);
        EXPECT_FALSE(requested);
        EXPECT_EQ(admission.counters().used, baseline.used);
        EXPECT_EQ(admission.counters().cumulative, baseline.cumulative);
    }
    counter.owner()->limit = baseline.used + bytes;
    auto* allocation = admission.allocator<Aligned>().allocate(count);
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(raw_extent_live(allocation, bytes));
        EXPECT_EQ(admission.counters().used, baseline.used + bytes);
        EXPECT_EQ(admission.counters().peak, baseline.used + bytes);
    }
    admission.allocator<Aligned>().deallocate(allocation, count);
    raw_live_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(raw_extent_live(allocation, bytes));
    EXPECT_EQ(admission.counters().used, baseline.used);
}

TEST(NativeAdmissionLifetime, PartialTargetConstructionReleasesEveryConcreteChildAndReceipt)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    const auto baseline = admission.counters().used;
    unsigned children = 0, objects = 0;
    bool failed = false;
    try
    {
        (void) N::NativeAllocationEnvelopeAccess::make<CompositePayload>(admission, admission, children, objects, true);
    }
    catch (const ConstructionFailure&)
    {
        failed = true;
    }
    raw_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_TRUE(failed);
    EXPECT_EQ(children, 1u);
    EXPECT_EQ(objects, 0u);
    EXPECT_EQ(admission.counters().used, baseline);
}

TEST(NativeAdmissionLifetime, EveryEnvelopeAndConcreteChildPrefixUnwindsAtItsOneByteShortBoundary)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    struct Request
    {
        std::uint64_t serial;
        std::size_t bytes;
    };
    std::array<Request, 16> requests{};
    std::size_t count = 0;
    {
        const Trace trace;
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        unsigned children = 0, objects = 0;
        const auto before = pipeline_trace::serial;
        auto payload =
            N::NativeAllocationEnvelopeAccess::make<CompositePayload>(admission, admission, children, objects);
        // Capture independently observed requests, including STL buffers and
        // receipt objects, instead of duplicating the implementation's sizes.
        for (const auto& entry : pipeline_trace::table)
            if (entry.live && entry.serial > before)
            {
                const pipeline_trace::Pause pause;
                ASSERT_LT(count, requests.size());
                requests[count++] = {entry.serial, entry.bytes};
            }
        std::sort(requests.begin(), requests.begin() + count,
            [](const Request& a, const Request& b) { return a.serial < b.serial; });
        raw_matches(admission);
        const pipeline_trace::Pause pause;
        ASSERT_GT(count, 4u);
    }
    for (std::size_t failure = 0; failure != count; ++failure)
    {
        const Trace trace;
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        const auto baseline = admission.counters();
        std::size_t prefix = 0;
        for (std::size_t i = 0; i != failure; ++i)
            prefix += requests[i].bytes;
        counter.owner()->limit = baseline.used + prefix + requests[failure].bytes - 1;
        unsigned children = 0, objects = 0;
        bool denied = false;
        try
        {
            (void) N::NativeAllocationEnvelopeAccess::make<CompositePayload>(admission, admission, children, objects);
        }
        catch (const std::invalid_argument&)
        {
            denied = true;
        }
        raw_live_matches(admission);
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(denied) << failure;
        EXPECT_EQ(admission.counters().used, baseline.used) << failure;
        EXPECT_EQ(admission.counters().cumulative, baseline.cumulative + prefix) << failure;
        EXPECT_EQ(objects, failure + 1 == count ? 1u : 0u) << failure;
        EXPECT_EQ(children, objects) << failure;
    }
}

TEST(NativeAdmissionLifetime, FixedGraphAndDynamicChildrenMatchRawLifetimesAcrossWeakTails)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const Trace trace;
    std::weak_ptr<CompositePayload> target_tail;
    std::weak_ptr<const N::NativeAllocationEnvelope> envelope_tail;
    std::weak_ptr<const N::NativeAllocationState> state_tail;
    unsigned children = 0, objects = 0;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        state_tail = admission.state();
        const auto baseline = admission.counters().used;
        auto payload =
            N::NativeAllocationEnvelopeAccess::make<CompositePayload>(admission, admission, children, objects);
        target_tail = payload.owner;
        envelope_tail = payload.envelope;
        const auto* buffer = payload.owner->cells.data();
        const auto buffer_bytes = payload.owner->cells.capacity() * sizeof(Aligned);
        const auto* object = payload.owner->child.get();
        const auto fixed = payload.envelope->fixed_allocation_bytes();
        {
            GraphCensus graph;
            N::NativeAllocationGraphAccess::fixed(payload.envelope, {&graph, GraphCensus::edge}, false);
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(graph.valid);
            EXPECT_EQ(graph.count, 3u); // Envelope's four blocks, State, and counter's two blocks.
            EXPECT_EQ(graph.bytes, fixed + baseline);
            EXPECT_GT(graph.edges, graph.count);
            EXPECT_EQ(graph.bytes + payload.owner->storage.exclusive_bytes(), pipeline_trace::summary().all_live);
        }
        pipeline_trace::prepare_census();
        payload.envelope->visit_allocations(
            [](N::NativeAllocationExtent extent) { pipeline_trace::mark(extent.address, 1); });
        payload.owner->storage.visit_allocations(
            [](N::NativeAllocationExtent extent) { pipeline_trace::mark(extent.address, 1); });
        pipeline_trace::mark(admission.state()->allocation.address, 1);
        pipeline_trace::mark(counter.allocations().object.address, 1);
        pipeline_trace::mark(counter.allocations().control.address, 1);
        raw_matches(admission);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(pipeline_trace::unmarked_graph_extents().second, 0u);
            EXPECT_TRUE(raw_extent_live(buffer, buffer_bytes));
            EXPECT_TRUE(raw_extent_live(object, sizeof(DestroyedChild)));
        }
        payload.owner.reset();
        raw_matches(admission);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(children, 1u);
            EXPECT_EQ(objects, 1u);
            EXPECT_TRUE(target_tail.expired());
            EXPECT_FALSE(payload.envelope->object_alive());
            EXPECT_FALSE(raw_extent_live(buffer, buffer_bytes));
            EXPECT_FALSE(raw_extent_live(object, sizeof(DestroyedChild)));
            EXPECT_EQ(admission.counters().used, baseline + fixed);
        }
        payload.envelope.reset();
        raw_matches(admission);
    }
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(envelope_tail.expired());
        EXPECT_FALSE(state_tail.expired()); // Allocators retained by weak controls keep State alive.
        EXPECT_GT(pipeline_trace::summary().all_live, 0u);
    }
    target_tail.reset();
    envelope_tail.reset();
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(state_tail.expired());
        EXPECT_GT(pipeline_trace::summary().all_live, 0u); // State's own weak-control tail.
    }
    state_tail.reset();
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}

TEST(NativeAdmissionLifetime, ForeignStateAndEnvelopeControlsRejectGraphAdmission)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    auto payload = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, 23);
    const std::shared_ptr<const N::NativeAllocationState> foreign_state(admission.state().get(), [](const auto*) {});
    const std::shared_ptr<const N::NativeAllocationEnvelope> foreign_envelope(
        payload.envelope.get(), [](const auto*) {});
    GraphCensus graph;
    EXPECT_THROW(N::NativeAllocationGraphAccess::state(foreign_state, {&graph, GraphCensus::edge}, false),
        std::invalid_argument);
    EXPECT_THROW(N::NativeAllocationGraphAccess::fixed(foreign_envelope, {&graph, GraphCensus::edge}, false),
        std::invalid_argument);
    EXPECT_EQ(graph.edges, 0u);
}

TEST(NativeAdmissionLifetime, MirroredSourceAdmissionRetiresActualProfileAndWeakControlExtents)
{
    auto source = std::make_shared<MirrorCounter>(); // Detached source arena exists before this operation.
    const Trace trace;
    std::weak_ptr<Aligned> weak;
    std::weak_ptr<Counter> counter_tail;
    {
        N::NativeAllocationMirror mirror{source, MirrorCounter::acquire, MirrorCounter::retire, {}};
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096, mirror);
        counter_tail = counter.owner();
        const N::NativeConstructionAdmission admission(counter);
        auto check_mirror = [&]
        {
            raw_matches(admission);
            const auto statistics = admission.statistics();
            const pipeline_trace::Pause pause;
            EXPECT_EQ(
                source->used, 4096 + statistics.actual_live + statistics.profiled_live + statistics.reserved_live);
        };
        check_mirror();
        {
            auto phase = admission.reserve_profile({1024});
            auto retained = phase.retain_profile(257);
            check_mirror();
            phase.finish_scratch();
            check_mirror();
            retained.reset();
            check_mirror();
        }
        auto strong = admission.shared<Aligned>();
        weak = strong;
        check_mirror();
        strong.reset();
        check_mirror();
        admission.release_prefix(4096); // Imported prefix was pre-existing, so it is not mirrored twice.
        check_mirror();
    }
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(source->used, 4096 + pipeline_trace::summary().all_live);
        EXPECT_FALSE(counter_tail.expired());
    }
    weak.reset();
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(counter_tail.expired());
        EXPECT_EQ(source->used, 4096 + pipeline_trace::summary().all_live);
        EXPECT_GT(pipeline_trace::summary().all_live, 0u);
    }
    counter_tail.reset();
    const pipeline_trace::Pause pause;
    EXPECT_EQ(source->used, 4096u);
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}

TEST(NativeAdmissionLifetime, SourceMirrorOneByteShortRejectsBeforeRawAllocationAndExactHeadroomSucceeds)
{
    auto source = std::make_shared<MirrorCounter>();
    const Trace trace;
    N::NativeAllocationMirror mirror{source, MirrorCounter::acquire, MirrorCounter::retire, {}};
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096, mirror);
    const N::NativeConstructionAdmission admission(counter);
    constexpr std::size_t count = 47, bytes = count * sizeof(Aligned);
    const auto baseline = admission.counters();
    const auto source_baseline = source->used;
    source->limit = source_baseline + bytes - 1;
    bool denied = false;
    try
    {
        (void) admission.allocator<Aligned>().allocate(count);
    }
    catch (const std::invalid_argument&)
    {
        denied = true;
    }
    bool requested = false;
    for (const auto& entry : pipeline_trace::table)
        requested = requested || (entry.used && entry.bytes == bytes);
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(denied);
        EXPECT_FALSE(requested);
        EXPECT_EQ(source->used, source_baseline);
        EXPECT_EQ(admission.counters().used, baseline.used);
        EXPECT_EQ(admission.counters().cumulative, baseline.cumulative);
    }
    source->limit = source_baseline + bytes;
    auto* allocation = admission.allocator<Aligned>().allocate(count);
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(raw_extent_live(allocation, bytes));
        EXPECT_EQ(source->used, source_baseline + bytes);
    }
    admission.allocator<Aligned>().deallocate(allocation, count);
    raw_live_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(source->used, source_baseline);
    EXPECT_FALSE(raw_extent_live(allocation, bytes));
}

TEST(NativeAdmissionLifetime, MirroredCounterAndStateBirthUnwindEachPartialAllocationPrefix)
{
    struct Request
    {
        std::uint64_t serial;
        std::size_t bytes;
    };
    std::array<Request, 3> requests{};
    {
        auto source = std::make_shared<MirrorCounter>();
        const Trace trace;
        N::NativeAllocationMirror mirror{source, MirrorCounter::acquire, MirrorCounter::retire, {}};
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096, mirror);
        const N::NativeConstructionAdmission admission(counter);
        std::size_t count = 0;
        for (const auto& entry : pipeline_trace::table)
            if (entry.live)
            {
                const pipeline_trace::Pause pause;
                ASSERT_LT(count, requests.size());
                requests[count++] = {entry.serial, entry.bytes};
            }
        std::sort(
            requests.begin(), requests.end(), [](const Request& a, const Request& b) { return a.serial < b.serial; });
        raw_matches(admission);
        const pipeline_trace::Pause pause;
        ASSERT_EQ(count, requests.size());
    }
    std::size_t prefix = 0;
    for (std::size_t failure = 0; failure != requests.size(); ++failure)
    {
        auto source = std::make_shared<MirrorCounter>();
        source->limit = 4096 + prefix + requests[failure].bytes - 1;
        const Trace trace;
        bool denied = false;
        try
        {
            N::NativeAllocationMirror mirror{source, MirrorCounter::acquire, MirrorCounter::retire, {}};
            const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096, mirror);
            const N::NativeConstructionAdmission admission(counter);
        }
        catch (const std::invalid_argument&)
        {
            denied = true;
        }
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(denied) << failure;
        EXPECT_EQ(source->used, 4096u) << failure;
        EXPECT_EQ(pipeline_trace::summary().all_live, 0u) << failure;
        EXPECT_FALSE(pipeline_trace::overflow.load());
        prefix += requests[failure].bytes;
    }
    auto source = std::make_shared<MirrorCounter>();
    source->limit = 4096 + prefix;
    const Trace trace;
    {
        N::NativeAllocationMirror mirror{source, MirrorCounter::acquire, MirrorCounter::retire, {}};
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 4096, mirror);
        const N::NativeConstructionAdmission admission(counter);
        raw_matches(admission);
        const pipeline_trace::Pause pause;
        EXPECT_EQ(source->used, source->limit);
        EXPECT_EQ(pipeline_trace::summary().all_live, prefix);
    }
    const pipeline_trace::Pause pause;
    EXPECT_EQ(source->used, 4096u);
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
} // namespace
