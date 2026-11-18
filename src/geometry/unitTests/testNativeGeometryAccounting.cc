/** Independent requested-C++ allocation and closed native ledger regressions. */
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/ContactPhysicalResidency.hh"
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "pipeline_allocation_operators.inc"
#include "pipeline_allocation_trace.hh"
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <unordered_map>
namespace
{
namespace N = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::PLIC::details;
namespace G = D::geometry_demand;
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
TEST(NativeGeometryAccounting, CounterBirthPreAdmitsPrefixAndKeepsItSeparateFromRawAllocations)
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
TEST(NativeGeometryAccounting, ExpiredOperationAuthorityRejectsMovedOutMutableLedger)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    SimpleFluid::PLIC::GeometryAllocationLedger escaped;
    {
        D::NativeGeometryAccounting accounting(admission);
        escaped = std::move(accounting.ordinary());
    }
    std::size_t foreign_cursor = 0;
    auto owner = admission.shared<unsigned>(23);
    EXPECT_THROW(
        escaped.retain_external_owner(owner, sizeof(unsigned), foreign_cursor, cap), SimpleFluid::PLIC::GeometryError);
}
TEST(NativeGeometryAccounting, ActualAlignedNodesBucketsControlsAndWeakTailMatchRawCppRequests)
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
TEST(NativeGeometryAccounting, ProfileTransfersRetireByScopeWithoutFictitiousAllocatorWork)
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
TEST(NativeGeometryAccounting, ConcreteVectorAndStringEarlyReleaseDestroyStorageBeforeRetirement)
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
TEST(NativeGeometryAccounting, OneByteShortRejectsBeforeTheRequestedAllocation)
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
TEST(NativeGeometryAccounting, NativeAllocationRetirementBetweenCertifiedImportsNeverDecreasesLedgerCursor)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    D::NativeGeometryAccounting accounting(admission);
    auto payload = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, 17);
    accounting.authorize_semantic(0, D::GeometryOwnerGraphEditor::external_kind(), payload.owner, payload.envelope);
    D::GeometryOwnerGraphEditor ordinary({}, 0, accounting.cursor(), cap, &accounting.ordinary());
    D::GeometryOwnerGraphEditor bounded({}, 1, accounting.cursor(), cap, &accounting.bounded());
    std::size_t overlap = 0;
    ordinary.pair_with(bounded, overlap);
    ordinary.external(payload.owner, 0, true);
    ordinary.native_fixed(payload.envelope, true);
    const auto roots = accounting.finish_pair(ordinary, bounded, overlap);
    D::GeometryOwnerImportAccess::import_pair(
        accounting.ordinary(), accounting.bounded(), roots, accounting.cursor(), cap);
    const auto cursor = accounting.cursor();
    const auto native_before = admission.counters().used;
    auto* scratch = admission.allocator<Aligned>().allocate(19);
    admission.allocator<Aligned>().deallocate(scratch, 19);
    const auto native_after = admission.counters().used;
    D::GeometryOwnerImportAccess::import_pair(
        accounting.ordinary(), accounting.bounded(), roots, accounting.cursor(), cap);
    raw_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(native_after, native_before);
    EXPECT_GE(accounting.cursor(), cursor);
    EXPECT_NE(&accounting.cursor(), admission.state()->used_counter);
    EXPECT_GT(admission.statistics().ledger_work_bytes, 0u);
}
TEST(NativeGeometryAccounting, PairedNativePayloadOverlapsByActualPhysicalReceipts)
{
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    D::NativeGeometryAccounting accounting(admission);
    auto payload = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, 17);
    D::GeometryOwnerGraphEditor ordinary({}, 0, accounting.cursor(), cap, &accounting.ordinary());
    D::GeometryOwnerGraphEditor bounded({}, 1, accounting.cursor(), cap, &accounting.bounded());
    std::size_t overlap = 0;
    ordinary.pair_with(bounded, overlap);
    ordinary.native_fixed(payload.envelope, true);
    bounded.native_fixed(payload.envelope, true);
    const auto state = admission.state();
    const auto expected = payload.envelope->fixed_allocation_bytes() + state->allocation_admission_bytes +
                          counter.allocations().object.bytes + counter.allocations().control.bytes;
    const auto roots = accounting.finish_pair(ordinary, bounded, overlap);
    raw_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(overlap, expected);
    EXPECT_EQ(roots.overlap_bytes(), expected);
    EXPECT_EQ(roots.payload_bytes(), expected);
}
TEST(NativeGeometryAccounting, ForeignPrepaidClaimsPoisonBothChannelsAndRejectAliasedControls)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    D::NativeGeometryAccounting accounting(admission);
    auto native = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, 17);
    const std::shared_ptr<const void> forged_control(native.owner.get(), [](const void*) {});
    EXPECT_THROW(
        accounting.authorize_semantic(0, D::GeometryOwnerGraphEditor::external_kind(), forged_control, native.envelope),
        SimpleFluid::PLIC::GeometryError);
    const auto foreign = std::make_shared<unsigned>(19);
    EXPECT_THROW(
        D::GeometryAllocationOwnerAccess::admit_graph_owner(accounting.ordinary(), 0,
            D::GeometryOwnerGraphEditor::external_kind(), foreign, sizeof(unsigned), accounting.cursor(), cap, true),
        SimpleFluid::PLIC::GeometryError);
    EXPECT_THROW(accounting.bounded().retain_external_owner(native.owner, 0, accounting.cursor(), cap),
        SimpleFluid::PLIC::GeometryError);
}
TEST(NativeGeometryAccounting, EarlyConcreteRetirementInvalidatesAnIssuedPaidStorageReceipt)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    D::NativeGeometryAccounting accounting(admission);
    const auto storage = N::native_constructor_storage(admission);
    std::vector<std::uint64_t> values;
    storage.owner->reserve(values, 23);
    accounting.authorize_storage(storage.owner, storage.envelope);
    const auto paid_bytes = storage.owner->exclusive_bytes();
    storage.owner->reset_buffer(values);
    EXPECT_THROW(
        D::GeometryAllocationOwnerAccess::admit_graph_owner(accounting.ordinary(), 0,
            D::GeometryOwnerGraphEditor::external_kind(), storage.owner, paid_bytes, accounting.cursor(), cap, true),
        SimpleFluid::PLIC::GeometryError);
}
TEST(NativeGeometryAccounting, OwnNativeIndexDomainsMatchIndependentRawCppLifetimeAcrossLongHistoryAndThreeGenerations)
{
    // All source fixture storage predates capture. Only actual native Counter,
    // State, index, control and construction scratch allocations enter the trace.
    constexpr std::size_t rows_count = 3072, generations = 80;
    std::vector<std::shared_ptr<const void>> inputs;
    inputs.reserve(rows_count);
    for (std::size_t index = 0; index != rows_count; ++index)
        inputs.push_back(std::make_shared<unsigned>(index));
    std::sort(inputs.begin(), inputs.end(),
        [](const auto& a, const auto& b) { return std::less<const void*>{}(a.get(), b.get()); });
    std::vector<G::OwnerRecord> rows;
    rows.reserve(rows_count);
    for (const auto& owner : inputs)
        rows.push_back({{0, D::GeometryOwnerGraphEditor::external_kind(), owner.get()}, owner, 1, 1});
    std::vector<std::weak_ptr<const N::NativeAllocationState>> probes;
    probes.reserve(generations + 4);
    struct Frame
    {
        D::OwnerManifestPair body;
        D::GeometryNativeAdministration administration;
    };
    std::array<Frame, 3> retained;
    const Trace trace;
    const auto make = [&](const Frame& previous, std::size_t generation)
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        probes.push_back(admission.state()); // pre-reserved test-only weak observation slots
        std::size_t cursor = 0;
        G::Admission demand{&cursor, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; },
            nullptr, admission.state()};
        Frame next;
        if (!generation)
            next.body.owners[0] = G::GeometryOwnerManifest::built(rows, demand);
        else
            next.body.owners[0] = previous.body.owners[0].retained(0, D::GeometryOwnerGraphEditor::external_kind(),
                inputs[((generation - 1) % 48) * G::leaf_capacity], 1, demand);
        D::GeometryOwnerUnionWork work;
        next.administration =
            previous.administration.updated(previous.body, next.body, {}, {}, cursor, cap, admission.state(), work);
        {
            const pipeline_trace::Pause pause;
            EXPECT_GT(work.native_domain_edges, 0u);
            EXPECT_GT(work.native_domain_batches, 0u);
        }
        return next;
    };
    retained[0] = make({}, 0);
    for (std::size_t generation = 1; generation != generations; ++generation)
    {
        const auto previous = (generation - 1) % 3;
        const auto slot = generation % 3;
        auto next = make(retained[previous], generation);
        retained[slot] = std::move(next);
        if (generation % 8 != 0)
            continue;
        // Drop every other generation before measuring the remaining actual
        // C++ allocations. No production owner visitor derives this oracle.
        Frame only = retained[slot];
        retained = {};
        for (auto& probe : probes)
            if (probe.expired())
                probe.reset();
        const auto raw = pipeline_trace::summary();
        const auto counted = only.body.index_admission_bytes() + only.administration.resident_bytes();
        {
            const pipeline_trace::Pause pause;
            EXPECT_FALSE(pipeline_trace::overflow.load());
            EXPECT_EQ(counted, raw.all_live) << "generation=" << generation;
        }
        retained[slot] = std::move(only);
    }
    const auto slot = (generations - 1) % 3;
    Frame current = retained[slot];
    retained = {};
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        std::size_t cursor = 0;
        D::GeometryOwnerUnionWork work;
        const auto empty =
            current.administration.updated(current.body, {}, {}, {}, cursor, cap, admission.state(), work);
        const auto repeated = empty.updated({}, {}, {}, {}, cursor, cap, admission.state(), work);
        const pipeline_trace::Pause pause;
        EXPECT_EQ(empty.resident_bytes(), 0u);
        EXPECT_EQ(repeated.resident_bytes(), 0u);
    }
    current = {};
    for (auto& probe : probes)
    {
        const auto expired = probe.expired();
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(expired);
        }
        probe.reset();
    }
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, ReusingStateAuthenticatesItsActualSharedControl)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission original(counter);
    const auto reused = N::NativeConstructionAdmission::from_state(original.state());
    EXPECT_EQ(reused.state(), original.state());
    const std::shared_ptr<const N::NativeAllocationState> forged(original.state().get(), [](const auto*) {});
    EXPECT_THROW((void) N::NativeConstructionAdmission::from_state(forged), std::invalid_argument);
}
TEST(NativeGeometryAccounting, NativeBookkeepingRetainsLogicalPayloadGatesAndRetiresActualAdminStorage)
{
    const auto input = std::make_shared<unsigned>(19);
    const Trace trace;
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    const auto baseline = admission.counters().used;
    {
        SimpleFluid::PLIC::GeometryAllocationLedger ledger;
        std::size_t logical = 0;
        D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ledger, admission, logical, cap);
        ledger.retain_external_owner(input, 37, logical, cap);
        raw_matches(admission);
        const pipeline_trace::Pause pause;
        EXPECT_EQ(ledger.payload_bytes(), 37u);
        EXPECT_GT(logical, 37u);
        EXPECT_GT(admission.counters().used, baseline);
    }
    const auto after = admission.counters().used;
    raw_matches(admission);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(after, baseline);
}
TEST(NativeGeometryAccounting, NativeBookkeepingCannotMintPrepaidPayloadAuthority)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    SimpleFluid::PLIC::GeometryAllocationLedger ledger;
    std::size_t logical = 0;
    D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ledger, admission, logical, cap);
    const auto input = std::make_shared<unsigned>(19);
    EXPECT_THROW(D::GeometryAllocationOwnerAccess::admit_prepaid(ledger, input, 37, logical, cap),
        SimpleFluid::PLIC::GeometryError);
    EXPECT_THROW(ledger.retain_external_owner(input, 37, logical, cap), SimpleFluid::PLIC::GeometryError);
}
TEST(NativeGeometryAccounting, ExpiredNativeObservationAccountsOnlyActualFixedAllocations)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    std::vector<std::byte> probe_storage(4 * 1024 * 1024);
    const Trace trace;
    D::NativeWeakOwnerObservation observation;
    std::weak_ptr<const N::NativeConstructorStorage> target;
    std::size_t live_body_bytes = 0;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
        const N::NativeConstructionAdmission admission(counter);
        auto storage = N::native_constructor_storage(admission);
        target = storage.owner;
        {
            std::vector<std::byte> buffer;
            storage.owner->reserve(buffer, 181);
            const auto charge = D::NativePaidExtentCharge::storage(storage);
            observation = charge.physical_observation();
            live_body_bytes = storage.owner->exclusive_bytes();
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(observation.matches_live(storage.owner));
            EXPECT_GT(live_body_bytes, buffer.capacity());
            const std::shared_ptr<const void> foreign(storage.owner.get(), [](const void*) {});
            EXPECT_FALSE(observation.matches_live(foreign));
        }
        const auto before = pipeline_trace::summary().all_live;
        storage.owner.reset();
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(target.expired());
            EXPECT_FALSE(observation.matches_live(storage.owner));
            EXPECT_EQ(before - pipeline_trace::summary().all_live, live_body_bytes - 181);
        }
    }
    target.reset();
    const auto raw = pipeline_trace::summary().all_live;
    {
        const pipeline_trace::Pause pause;
        D::retention::ProbeScratch scratch(probe_storage);
        const std::array roots{D::retention::root(observation)};
        const auto report = D::retention::inspect_retention(roots, {}, scratch);
        ASSERT_TRUE(report.complete()) << report.detail;
        EXPECT_EQ(report.owner_union.bytes, raw);

        std::size_t cursor = 0;
        SimpleFluid::PLIC::GeometryAllocationLedger ledger;
        D::GeometryAllocationOwnerAccess::retain_weak_observation(ledger, 0, observation, cursor, cap);
        D::GeometryAllocationOwnerAccess::retain_weak_observation(ledger, 0, observation, cursor, cap);
        EXPECT_EQ(ledger.payload_bytes(), raw);

        D::GeometryOwnerGraphEditor retain({}, 0, cursor, cap);
        EXPECT_TRUE(retain.weak_observation(observation, true));
        EXPECT_FALSE(retain.weak_observation(observation, true));
        auto graph = retain.finish();
        EXPECT_EQ(graph.payload_bytes(), raw);
        D::GeometryOwnerGraphEditor retire(std::move(graph), 0, cursor, cap);
        EXPECT_FALSE(retire.weak_observation(observation, false));
        EXPECT_TRUE(retire.weak_observation(observation, false));
        EXPECT_EQ(retire.finish().owner_count(), 0u);
    }
    observation = {};
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, NativeZeroAllowanceRetainsItsAdministrationRole)
{
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    std::size_t cursor = 0;
    G::Admission demand{&cursor, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; },
        nullptr, admission.state()};
    const auto owner = std::make_shared<unsigned>(19);
    const D::IndexControlRecord live{owner, sizeof(unsigned), 0, true};
    const auto controls = D::GeometryIndexControls{}.edited(std::span(&live, 1), demand);
    EXPECT_TRUE(controls.holds(owner, sizeof(unsigned)));
    EXPECT_TRUE(controls.contains(owner, sizeof(unsigned), 0));
    EXPECT_EQ(controls.control_bytes(), 0u);
    const D::IndexControlRecord retired{owner, sizeof(unsigned), 0, false};
    const auto empty = controls.edited(std::span(&retired, 1), demand);
    EXPECT_FALSE(empty.holds(owner, sizeof(unsigned)));
}
TEST(NativeGeometryAccounting, CustomOddAdmissionRowsKeepTheirUnpackedNumericTotals)
{
    struct Row { unsigned key; std::size_t exclusive_admission_bytes() const noexcept { return 1; } };
    struct Less { bool operator()(const Row& a, const Row& b) const noexcept { return a.key < b.key; } };
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
    const N::NativeConstructionAdmission admission(counter);
    std::size_t logical = 0;
    const G::Admission demand{&logical, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; },
        nullptr, admission.state()};
    const std::array<Row, 3> rows{{{1}, {2}, {3}}};
    const auto root = G::OrderedPages<Row, Less>(demand).build(rows);
    EXPECT_EQ(root->subtree_owned_admission_bytes, 3u);
    EXPECT_EQ(root->owned_admission_bytes(), 3u);
}
TEST(NativeGeometryAccounting, PopulatedBorrowedLedgerKeeperSurvivesWrapperExpiryAndBothMovePaths)
{
    // Payloads predate the trace and carry no admission State. Only the actual
    // ledger holder and its deleter can keep either native domain alive.
    std::array<std::shared_ptr<const unsigned>, 96> payloads;
    for (unsigned index = 0; index != payloads.size(); ++index)
        payloads[index] = std::make_shared<const unsigned>(index);
    std::weak_ptr<const N::NativeAllocationState> first_state, second_state;
    std::weak_ptr<const void> first_counter, second_counter;
    const Trace trace;
    {
        SimpleFluid::PLIC::GeometryAllocationLedger first, second;
        {
            const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
            const N::NativeConstructionAdmission admission(counter);
            first_state = admission.state();
            first_counter = admission.state()->counter_owner;
            std::size_t logical = 0;
            D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(first, admission, logical, cap);
            for (const auto& payload : payloads)
                first.retain_external_owner(payload, sizeof(unsigned), logical, cap);
            raw_matches(admission);
        }
        {
            const pipeline_trace::Pause pause;
            EXPECT_FALSE(first_state.expired());
            EXPECT_FALSE(first_counter.expired());
        }
        SimpleFluid::PLIC::GeometryAllocationLedger moved(std::move(first));
        {
            const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
            const N::NativeConstructionAdmission admission(counter);
            second_state = admission.state();
            second_counter = admission.state()->counter_owner;
            std::size_t logical = 0;
            D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(second, admission, logical, cap);
            for (const auto& payload : payloads)
                second.retain_external_owner(payload, sizeof(unsigned), logical, cap);
            const auto old_domain = first_state.lock();
            const auto raw = pipeline_trace::summary();
            const pipeline_trace::Pause pause;
            ASSERT_TRUE(old_domain);
            EXPECT_EQ(raw.all_live, old_domain->statistics.actual_live + admission.statistics().actual_live);
        }
        // Move assignment destroys the populated destination's containers while
        // its own keeper still exists, then installs the incoming keeper.
        second = std::move(moved);
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(second_state.expired());
            EXPECT_FALSE(first_state.expired());
            EXPECT_FALSE(first_counter.expired());
            EXPECT_EQ(second.owner_count(), payloads.size());
            EXPECT_EQ(second.payload_bytes(), payloads.size() * sizeof(unsigned));
        }
        second_state.reset();
        second_counter.reset();
        {
            const auto live = first_state.lock();
            const auto raw = pipeline_trace::summary();
            const pipeline_trace::Pause pause;
            ASSERT_TRUE(live);
            EXPECT_EQ(raw.all_live, live->statistics.actual_live);
        }
    }
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(first_state.expired());
        // The expired State's still-live allocation control owns its bootstrap
        // allocator, which intentionally keeps the scalar Counter alive.
        EXPECT_FALSE(first_counter.expired());
    }
    first_state.reset();
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(first_counter.expired());
    }
    // A weak probe retains a real control allocation. Release every probe
    // before asking the independent requested-C++ trace for full retirement.
    first_state.reset();
    second_state.reset();
    first_counter.reset();
    second_counter.reset();
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, RealHolderCensusMatchesIndependentRawRequestsAndFailedNodeAllocation)
{
    std::array<std::shared_ptr<const unsigned>, 97> payloads;
    for (unsigned index = 0; index != payloads.size(); ++index)
        payloads[index] = std::make_shared<const unsigned>(index);
    std::vector<std::byte> probe_storage(2 * 1024 * 1024);
    std::weak_ptr<const N::NativeAllocationState> weak;
    const Trace trace;
    {
        SimpleFluid::PLIC::GeometryAllocationLedger ledger;
        std::size_t logical = 0;
        {
            const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap);
            const N::NativeConstructionAdmission admission(counter);
            weak = admission.state();
            D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ledger, admission, logical, cap);
            for (std::size_t index = 0; index != 96; ++index)
                ledger.retain_external_owner(payloads[index], sizeof(unsigned), logical, cap);
        }
        const auto inspect = [&]
        {
            D::retention::ProbeScratch scratch(probe_storage);
            const std::array<const SimpleFluid::PLIC::GeometryAllocationLedger*, 1> ledgers{&ledger};
            const auto report = D::retention::inspect_retention({}, ledgers, scratch);
            const auto raw = pipeline_trace::summary();
            const pipeline_trace::Pause pause;
            ASSERT_TRUE(report.complete()) << report.detail;
            EXPECT_EQ(report.ledger_bytes + report.ledger_tracking_bytes - 96 * sizeof(unsigned), raw.all_live);
        };
        inspect();
        const auto before = pipeline_trace::summary();
        {
            const auto state = weak.lock();
            ASSERT_TRUE(state);
            // No Counter/Admission wrapper remains. The authentic actual holder
            // supplies the domain; the next node must reject before allocation.
            auto* counter = static_cast<Counter*>(state->counter);
            counter->limit = counter->used;
        }
        {
            const pipeline_trace::Pause pause;
            EXPECT_THROW(ledger.retain_external_owner(payloads.back(), sizeof(unsigned), logical, cap), std::exception);
        }
        const auto after = pipeline_trace::summary();
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(after.all_live, before.all_live);
            EXPECT_EQ(after.phase[0].allocations, before.phase[0].allocations);
        }
        inspect();
    }
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(weak.expired());
    }
    weak.reset();
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
namespace A = SimpleFluid::Meshes::AMR::adaptive_detail;
struct PhysicalSourceFixture
{
    std::shared_ptr<A::Budget> body_arena = std::make_shared<A::Budget>(A::Budget{cap});
    std::shared_ptr<SimpleFluid::Meshes::NativeFragmentDescriptor> descriptor =
        std::make_shared<SimpleFluid::Meshes::NativeFragmentDescriptor>();
    SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf;
    A::AdaptiveLeafBodyUsage body;
    std::size_t physical_bytes = 0;
    PhysicalSourceFixture()
    {
        descriptor->view.assign(71, 'g');
        descriptor->parts.push_back(std::make_shared<const SimpleFluid::Meshes::AMR::AdaptiveBytes>(127, 's'));
        physical_bytes = sizeof(*descriptor) + descriptor->view.capacity() +
            descriptor->parts.capacity() * sizeof(descriptor->parts[0]) +
            sizeof(*descriptor->parts.front()) + descriptor->parts.front()->capacity();
        leaf.key = {0, SimpleFluid::Meshes::AMR::ConvexCellAncestry{1, {}}};
        leaf.native_recipe = descriptor;
        A::AdaptiveLeafBodyDelta delta(body_arena);
        delta.record(std::span(&leaf, 1), true);
        body = delta.apply({}).usage;
    }
};
struct RealFloor
{
    std::shared_ptr<A::Budget> arena = std::make_shared<A::Budget>(A::Budget{cap});
    std::shared_ptr<std::size_t> total = std::make_shared<std::size_t>(0);
    std::shared_ptr<A::ContactExternalResidencyLease> floor =
        std::make_shared<A::ContactExternalResidencyLease>(arena, total);
    explicit RealFloor(std::size_t bytes) { floor->resize(bytes); }
};
struct SourceUnionFrame
{
    A::NativeSourceOwnerStore store;
    D::GeometryOwnerUnion owners;
};
TEST(NativeGeometryAccounting, TypedEnvelopeFirstAndLastAreIndependentOfFlatScalarPins)
{
    auto arena = std::make_shared<A::Budget>(A::Budget{cap});
    const N::NativeConstructionAdmission admission(
        N::NativeConstructionCounterAccess::create<Counter>(cap, 0, A::NativeSourceOwnerStore::mirror(arena)));
    const auto target = N::NativeAllocationEnvelopeAccess::make<unsigned char>(admission);
    const auto sibling = N::NativeAllocationEnvelopeAccess::make<unsigned char>(admission);
    const auto& state = admission.state();
    const auto& counter = state->counter_owner;
    const auto counter_bytes = state->counter_allocations.object.bytes + state->counter_allocations.control.bytes;
    const auto kind = D::GeometryOwnerGraphEditor::external_kind();
    std::size_t work = 0;
    const auto edit = [&](const G::GeometryOwnerManifest& accepted, auto apply)
    {
        D::GeometryOwnerGraphEditor next(accepted, 0, work, cap, nullptr, nullptr, admission.state());
        apply(next);
        return next.finish();
    };
    const auto verify = [&](const G::GeometryOwnerManifest& manifest, std::array<std::size_t, 4> expected)
    {
        const std::array observed{manifest.references(0, kind, target.envelope), manifest.references(0, kind, state),
            manifest.references(0, kind, counter), manifest.references(0, kind, arena)};
        EXPECT_EQ(observed, expected);
        EXPECT_EQ(manifest.physical_edge_count(), std::accumulate(expected.begin(), expected.end(), std::size_t{0}));
    };
    const auto retain = [&](auto& graph) { graph.native_fixed(target.envelope, true); };
    const auto retire = [&](auto& graph) { graph.native_fixed(target.envelope, false); };
    const auto flat = [&](auto& graph, bool retain_flat)
    {
        graph.external(state, state->allocation_admission_bytes, retain_flat);
        graph.external(counter, counter_bytes, retain_flat);
    };
    // The envelope owns five State fields. State owns two Counter and two
    // keeper fields, and the Counter owns two additional keeper fields.
    const auto first = edit({}, retain);
    verify(first, {1, 5, 2, 4});
    const auto second = edit(first, retain);
    verify(second, {2, 5, 2, 4});
    const auto one = edit(second, retire);
    verify(one, {1, 5, 2, 4});
    const auto two_envelopes = edit(one, [&](auto& graph) { graph.native_fixed(sibling.envelope, true); });
    EXPECT_EQ(two_envelopes.references(0, kind, sibling.envelope), 1u);
    EXPECT_EQ(two_envelopes.references(0, kind, state), 10u);
    EXPECT_EQ(two_envelopes.references(0, kind, counter), 2u);
    EXPECT_EQ(two_envelopes.references(0, kind, arena), 4u);
    verify(edit(two_envelopes, [&](auto& graph) { graph.native_fixed(sibling.envelope, false); }), {1, 5, 2, 4});

    const auto flat_only = edit({}, [&](auto& graph) { flat(graph, true); });
    const auto mixed = edit(flat_only, retain);
    verify(mixed, {1, 6, 3, 4});
    const auto after_typed = edit(mixed, retire);
    verify(after_typed, {0, 1, 1, 0});
    verify(edit(after_typed, [&](auto& graph) { flat(graph, false); }), {0, 0, 0, 0});
    const auto after_flat = edit(mixed, [&](auto& graph) { flat(graph, false); });
    verify(after_flat, {1, 5, 2, 4});
    verify(edit(after_flat, retire), {0, 0, 0, 0});
    const auto opposite_admission = edit(first, [&](auto& graph) { flat(graph, true); });
    verify(opposite_admission, {1, 6, 3, 4});
    verify(edit(opposite_admission, retire), {0, 1, 1, 0});

    const auto witness = D::NativePaidExtentCharge::counter(state).detached_source_observation();
    ASSERT_TRUE(witness);
    const auto observed = edit(first, [&](auto& graph) { graph.weak_observation(witness, true); });
    verify(observed, {1, 5, 3, 5});
    const auto witness_only = edit(observed, retire);
    // Only the witness's keeper and the Counter's two keeper fields remain.
    verify(witness_only, {0, 0, 1, 3});
    verify(edit(witness_only, [&](auto& graph) { graph.weak_observation(witness, false); }), {0, 0, 0, 0});
}
TEST(NativeGeometryAccounting, TypedManifestReservationIncludesBorrowedMetadataAndRoleOnlyEdit)
{
    using Manifest = G::GeometryOwnerManifest;
    using Editor = D::GeometryOwnerGraphEditor;
    auto arena = std::make_shared<A::Budget>(A::Budget{cap});
    const N::NativeConstructionAdmission admission(
        N::NativeConstructionCounterAccess::create<Counter>(cap, 0, A::NativeSourceOwnerStore::mirror(arena)));
    const auto target = N::NativeAllocationEnvelopeAccess::make<unsigned char>(admission);
    const auto& state = admission.state();
    std::size_t used = 0;
    const auto edit = [&](const Manifest& accepted, auto mutate)
    {
        Editor graph(accepted, 0, used, cap, nullptr, nullptr, admission.state());
        mutate(graph);
        return graph.finish();
    };
    struct Borrowed
    {
        std::size_t nodes = 0, direct_children = 0, physical_rows = 0;
        const void* root = nullptr;
        const void* last_physical_leaf = nullptr;
    };
    const auto count_borrows = [](const Manifest& manifest)
    {
        Borrowed count;
        manifest.visit_borrowed_dependencies(
            {&count, [](void* raw, const void* owner, const void* parent, std::size_t)
                {
                    auto& c = *static_cast<Borrowed*>(raw);
                    if (!c.root) c.root = owner;
                    if (parent == c.root) ++c.direct_children;
                    ++c.nodes;
                    return true;
                }},
            {&count, [](void* raw, const void* parent, const G::OwnerRecord&)
                {
                    auto& c = *static_cast<Borrowed*>(raw);
                    ++c.physical_rows;
                    c.last_physical_leaf = parent;
                }});
        return count;
    };
    const auto typed = edit({}, [&](auto& graph) { graph.native_fixed(target.envelope, true); });
    const auto actual = count_borrows(typed);
    EXPECT_EQ(actual.physical_rows, 4u);
    EXPECT_EQ(actual.direct_children, 2u);
    EXPECT_EQ(actual.nodes, 3u);
    EXPECT_EQ(typed.metadata_owner_count_bound(), actual.nodes);

    const auto flat_edges = [&](auto& graph, bool retain)
    {
        const auto slot = [&](const auto& owner, std::size_t bytes, std::size_t count)
        {
            for (std::size_t i = 0; i != count; ++i) graph.external(owner, bytes, retain);
        };
        slot(target.envelope, target.envelope->fixed_allocation_bytes(), 1);
        slot(state, state->allocation_admission_bytes, 5);
        slot(state->counter_owner, state->counter_allocations.object.bytes + state->counter_allocations.control.bytes, 2);
        slot(arena, state->mirror.owner_proof.bytes(), 4);
    };
    const auto flat = edit(typed, [&](auto& graph)
        {
            flat_edges(graph, true);
            graph.native_fixed(target.envelope, false);
        });
    const auto flat_borrows = count_borrows(flat);
    EXPECT_EQ(flat_borrows.nodes, 2u);
    EXPECT_EQ(flat_borrows.last_physical_leaf, actual.last_physical_leaf);
    EXPECT_EQ(flat.metadata_owner_count_bound(), Manifest::metadata_owner_count_bound(4));

    const auto before = used;
    const auto restored = edit(flat, [&](auto& graph)
        {
            graph.native_fixed(target.envelope, true);
            flat_edges(graph, false);
        });
    const auto restored_borrows = count_borrows(restored);
    EXPECT_EQ(restored_borrows.last_physical_leaf, actual.last_physical_leaf);
    EXPECT_EQ(restored_borrows.nodes, actual.nodes);
    EXPECT_EQ(restored.physical_edge_count(), flat.physical_edge_count());
    EXPECT_LE(used - before, Editor::edit_byte_bound(flat, 24, 0, 3));

    std::array<N::NativeTrackedOwner<unsigned char>, 62> siblings;
    for (auto& one : siblings)
        one = N::NativeAllocationEnvelopeAccess::make<unsigned char>(admission);
    const auto full = edit(restored, [&](auto& graph)
        {
            for (const auto& one : std::span(siblings).first(61))
                graph.native_fixed(one.envelope, true);
        });
    ASSERT_EQ(full.typed_owner_count(), 64u);
    const auto split_before = used;
    const auto split = edit(full, [&](auto& graph) { graph.native_fixed(siblings.back().envelope, true); });
    EXPECT_EQ(split.typed_owner_count(), 65u);
    const auto split_borrows = count_borrows(split);
    EXPECT_EQ(split_borrows.nodes, count_borrows(full).nodes + 2);
    EXPECT_LE(split_borrows.nodes, split.metadata_owner_count_bound());
    EXPECT_LE(used - split_before, Editor::edit_byte_bound(full, 6, 2, 2));
}
TEST(NativeGeometryAccounting, PaidReceiptBatchChecksReferenceOrderAndReducesPageCopies)
{
    auto arena = std::make_shared<A::Budget>(A::Budget{cap});
    const N::NativeConstructionAdmission admission(
        N::NativeConstructionCounterAccess::create<Counter>(cap, 0, A::NativeSourceOwnerStore::mirror(arena)));
    using Receipts = D::NativeOwnerReceipts;
    using Action = Receipts::Action;
    constexpr auto count = G::leaf_capacity + 1;
    std::array<N::NativeTrackedOwner<unsigned>, count> actual;
    std::array<D::NativePaidExtentCharge, count> charges;
    std::array<Receipts::Edit, count> edits;
    std::size_t work = 0;
    const G::Admission demand{&work, [](void* raw, std::size_t bytes)
        { *static_cast<std::size_t*>(raw) += bytes; }, nullptr, admission.state()};
    Receipts sequential;
    for (std::size_t i = 0; i != actual.size(); ++i)
    {
        actual[i] = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, static_cast<unsigned>(i));
        charges[i] = D::NativePaidExtentCharge::fixed(actual[i].envelope);
        edits[i] = {&charges[i], Action::Retain};
        sequential = sequential.retained(charges[i], demand);
    }
    const auto sequential_work = work;
    work = 0;
    const auto batch = Receipts{}.edited(edits, demand);
    EXPECT_LT(work, sequential_work);
    ASSERT_GT(work, 0u);
    const auto live = admission.statistics().actual_live;
    struct Limit
    {
        std::size_t used = 0, cap;
    } short_batch{0, work - 1};
    const G::Admission short_demand{&short_batch, [](void* raw, std::size_t bytes)
        {
            auto& budget = *static_cast<Limit*>(raw);
            if (bytes > budget.cap - budget.used) throw std::bad_alloc();
            budget.used += bytes;
        }, nullptr, admission.state()};
    EXPECT_THROW((void) Receipts{}.edited(edits, short_demand), std::bad_alloc);
    EXPECT_EQ(admission.statistics().actual_live, live);
    struct Compare
    {
        const Receipts& sequential;
        std::size_t rows = 0;
    } expected{sequential};
    batch.visit({&expected, [](void* raw, const D::NativePaidExtentCharge& charge, std::size_t references)
        {
            auto& expected = *static_cast<Compare*>(raw);
            EXPECT_TRUE(expected.sequential.contains(charge));
            EXPECT_EQ(references, 1u);
            ++expected.rows;
        }});
    EXPECT_EQ(expected.rows, actual.size());
    const auto charge = D::NativePaidExtentCharge::fixed(actual.front().envelope);
    std::array balanced{Receipts::Edit{&charge, Action::Retire}, Receipts::Edit{&charge, Action::Retain}};
    EXPECT_TRUE(batch.edited(balanced, demand).same_root(batch));
    std::array invalid{Receipts::Edit{&charge, Action::Retire}, Receipts::Edit{&charge, Action::Retire},
        Receipts::Edit{&charge, Action::Retain}};
    EXPECT_THROW((void) batch.edited(invalid, demand), SimpleFluid::PLIC::GeometryError);
    EXPECT_TRUE(batch.contains(charge));

    const auto stage = [&](G::ReceiptEdits& scratch)
    {
        for (const auto& item : charges)
        {
            scratch.append_borrowed(item, Action::Retire);
            scratch.append(item, Action::Retain);
        }
    };
    work = 0;
    std::size_t staged_work = 0;
    {
        G::ReceiptEdits scratch(demand);
        stage(scratch);
        staged_work = work;
        EXPECT_TRUE(batch.edited(scratch.rows(), demand).same_root(batch));
    }
    ASSERT_GT(staged_work, 0u);
    const auto attempt = [&]
    {
        G::ReceiptEdits scratch(short_demand);
        stage(scratch);
    };
    short_batch = {0, staged_work - 1};
    EXPECT_THROW(attempt(), std::bad_alloc);
    EXPECT_GT(short_batch.used, 0u);
    EXPECT_EQ(admission.statistics().actual_live, live);
    EXPECT_TRUE(batch.contains(charge));

    const auto incoming = N::NativeAllocationEnvelopeAccess::make<unsigned>(admission, count);
    const auto changed = [&]
    {
        const auto donor = Receipts{}.edited(edits, demand);
        G::ReceiptEdits scratch(demand);
        scratch.append_borrowed(charges.front(), Action::Retire);
        scratch.append(D::NativePaidExtentCharge::fixed(incoming.envelope), Action::Retain);
        return donor.edited(scratch.rows(), demand);
    }();
    EXPECT_FALSE(changed.contains(charges.front()));
    EXPECT_TRUE(changed.contains(D::NativePaidExtentCharge::fixed(incoming.envelope)));
    for (const auto& unchanged : std::span(charges).subspan(1))
        EXPECT_TRUE(changed.contains(unchanged));
}
SourceUnionFrame source_union_frame(const SourceUnionFrame& previous, const PhysicalSourceFixture& source,
    RealFloor& floor, pipeline_trace::Phase phase, std::size_t revision)
{
    const auto previous_phase = pipeline_trace::phase;
    pipeline_trace::phase = phase;
    struct Restore { pipeline_trace::Phase phase; ~Restore() { pipeline_trace::phase = phase; } } restore{previous_phase};
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
        A::NativeSourceOwnerStore::mirror(floor.arena));
    const N::NativeConstructionAdmission admission(counter);
    const auto coverage = A::NativeSourceOwnerStore::body_coverage(source.body, A::AdaptiveLeafBodyRole::Leaf,
        floor.floor, admission, revision);
    std::array<D::OwnerManifestPair, 4> components;
    components[0] = source.body.owners();
    SourceUnionFrame result;
    result.store = A::NativeSourceOwnerStore::assemble(previous.store, std::move(components), admission, nullptr,
        {}, {source.body.administration(), {}}, coverage);
    SimpleFluid::PLIC::GeometryAllocationLedger ordinary, bounded;
    std::size_t cursor = 0;
    D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ordinary, admission, cursor, cap);
    D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(bounded, admission, cursor, cap);
    result.owners = result.store.replace_in(previous.store, previous.owners, ordinary, bounded, cursor, cap);
    return result;
}
void expect_raw_origin(const SourceUnionFrame& frame, const RealFloor& floor, pipeline_trace::Phase phase,
    std::size_t group_bytes)
{
    // Every allocation during a phase uses that real arena's admitted allocator.
    // Original physical source/floor fixtures predate capture. The only
    // explicit nonallocator credit is the actual group and documented scalar
    // arena upper extent; no production owner visitor computes this oracle.
    const auto raw = pipeline_trace::summary();
    const auto native = raw.phase[static_cast<unsigned>(phase)].live;
    const auto scalar = native ? sizeof(A::Budget) + 4 * sizeof(void*) : 0;
    const auto paid = frame.owners.native_paid_bytes(floor.arena);
    const auto direct = frame.owners.direct_h_paid_bytes(floor.arena);
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(paid, native + scalar + group_bytes);
    EXPECT_EQ(direct, native + scalar);
}
TEST(NativeGeometryAccounting, OriginEditsKeepNonzeroTotalsAcrossByteBoundaries)
{
    using Assignment = D::NativePaidOriginTotals::Assignment;
    std::array<std::shared_ptr<const void>, 3> arenas{
        std::make_shared<A::Budget>(), std::make_shared<A::Budget>(), std::make_shared<A::Budget>()};
    std::sort(arenas.begin(), arenas.end(),
        [](const auto& a, const auto& b) { return std::less<const void*>{}(a.get(), b.get()); });
    Counter counter{cap};
    const G::Admission admission{&counter, [](void* raw, std::size_t bytes) { static_cast<Counter*>(raw)->add(bytes); }};
    const std::array initial{Assignment{arenas[0], 3016}, Assignment{arenas[1], 120}};
    const auto old = D::NativePaidOriginTotals{}.edited(initial, admission);
    const std::array changes{Assignment{arenas[0], 8704}, Assignment{arenas[1], 0}, Assignment{arenas[2], 256}};
    const auto next = old.edited(changes, admission);
    EXPECT_EQ(next.bytes(arenas[0]), 8704u);
    EXPECT_EQ(next.bytes(arenas[1]), 0u);
    EXPECT_EQ(next.bytes(arenas[2]), 256u);
    EXPECT_EQ(old.bytes(arenas[0]), 3016u);
    EXPECT_EQ(old.bytes(arenas[1]), 120u);
    EXPECT_EQ(old.bytes(arenas[2]), 0u);
}
TEST(NativeGeometryAccounting, MirroredUnionAndUnchangedMembershipArenaSwitchMatchIndependentRawOrigins)
{
    PhysicalSourceFixture source;
    RealFloor first(source.physical_bytes), second(source.physical_bytes), third(source.physical_bytes);
    ASSERT_EQ(source.body.payload_bytes(), source.physical_bytes);
    SourceUnionFrame current;
    const Trace trace;
    current = source_union_frame({}, source, first, pipeline_trace::Phase::Candidate, 1);
    expect_raw_origin(current, first, pipeline_trace::Phase::Candidate, source.physical_bytes);
    const auto old_membership = current.store.binding().source_group();
    auto old_group = *old_membership;
    auto next = source_union_frame(current, source, second, pipeline_trace::Phase::Directory, 2);
    {
        std::size_t visits = 0;
        const auto& next_group = *next.store.binding().source_group();
        next_group.visit_delta(old_group, {&visits, [](void* raw, auto, const auto&, auto, bool)
            { ++*static_cast<std::size_t*>(raw); }});
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(next_group.same_membership(old_group));
        EXPECT_EQ(visits, 0u);
    }
    old_group = {};
    // Preserve three real generations first. Their original arena categories
    // remain independently valid; equal membership/bytes cannot change origin.
    auto newest = source_union_frame(next, source, third, pipeline_trace::Phase::Transport, 3);
    {
        const pipeline_trace::Pause pause;
        EXPECT_NO_THROW(current.store.require_current());
        EXPECT_NO_THROW(next.store.require_current());
        EXPECT_NO_THROW(newest.store.require_current());
    }
    current = {};
    next = {};
    newest.store = {};
    first.floor.reset();
    second.floor.reset();
    third.floor.reset();
    {
        const auto raw = pipeline_trace::summary();
        const pipeline_trace::Pause pause;
        EXPECT_EQ(raw.phase[static_cast<unsigned>(pipeline_trace::Phase::Candidate)].live, 0u);
        EXPECT_EQ(raw.phase[static_cast<unsigned>(pipeline_trace::Phase::Directory)].live, 0u);
    }
    expect_raw_origin(newest, first, pipeline_trace::Phase::Candidate, 0);
    expect_raw_origin(newest, second, pipeline_trace::Phase::Directory, 0);
    expect_raw_origin(newest, third, pipeline_trace::Phase::Transport, source.physical_bytes);
    {
        const pipeline_trace::Pause pause;
        const auto foreign = std::shared_ptr<const void>(third.arena.get(), [](const void*) {});
        EXPECT_THROW((void) newest.owners.native_paid_bytes(foreign), std::exception);
        EXPECT_THROW((void) newest.owners.direct_h_paid_bytes(foreign), std::exception);
    }
    newest = {};
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, SharedAndCopiedReceiptLeavesKeepExactDetachedPinEdgesAfterStructuralRelease)
{
    PhysicalSourceFixture source;
    RealFloor floor(source.physical_bytes);
    std::vector<std::byte> probe_storage(4 * 1024 * 1024);
    const auto selected = [&]
    {
        G::OwnerRecord result;
        source.body.owners().owners[0].visit([&](const auto& row) { if (!result.owner) result = row; });
        return result;
    }();
    ASSERT_TRUE(selected.owner);
    D::NativeOwnerReceipts base, alias, copied;
    std::weak_ptr<const A::NativeSourceOwnerCoverage> weak_finance;
    std::weak_ptr<const A::NativeSourceCoverage> weak_source;
    const Trace trace;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
            A::NativeSourceOwnerStore::mirror(floor.arena));
        const N::NativeConstructionAdmission admission(counter);
        auto group = A::NativeSourceOwnerStore::body_coverage(source.body, A::AdaptiveLeafBodyRole::Leaf,
            floor.floor, admission, 1);
        weak_source = group;
        A::NativeSourceCoverageBroker broker;
        std::shared_ptr<const A::NativeSourceSubsetCoverage> subset;
        {
            A::NativeSourceCoverageTransaction transaction(broker, admission);
            subset = transaction.collect(group, std::span(&selected, 1));
            transaction.prepare();
            transaction.commit();
        }
        weak_finance = subset->owners().front();
        {
            std::size_t cursor = 0;
            D::GeometryOwnerGraphEditor capture({}, 0, cursor, cap, nullptr, nullptr, admission.state());
            A::NativeSourceOwnerStore::record_subset(subset, capture);
            // A receipt can be independently retained after its structural
            // subset wrapper disappears. It pins only the selected token.
            (void) capture.finish();
            base = capture.paid_receipts();
        }
        alias = base;
        std::size_t work = 0;
        const G::Admission demand{&work, [](void* raw, std::size_t bytes)
            { *static_cast<std::size_t*>(raw) += bytes; }, nullptr, admission.state()};
        copied = base.recorded(D::NativePaidExtentCharge::state(admission.state()), demand);
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(base.same_root(alias));
            EXPECT_FALSE(base.same_root(copied));
        }
        G::GeometryOwnerManifest physical;
        const auto edit = [&](const D::NativeOwnerReceipts& input, bool retain)
        {
            std::size_t cursor = 0;
            D::GeometryOwnerGraphEditor graph(physical, 0, cursor, cap, nullptr, nullptr, admission.state());
            struct Context { D::GeometryOwnerGraphEditor& graph; bool retain; } context{graph, retain};
            const auto sink = G::MetadataSink{&context,
                [](void* raw, const auto& owner, std::size_t payload, std::size_t)
                { auto& context = *static_cast<Context*>(raw); return context.graph.external(owner, payload, context.retain); },
                [](void* raw, const auto& owner, std::size_t bytes)
                { auto& context = *static_cast<Context*>(raw); return context.graph.external(owner, bytes, context.retain); }}
                .with_financial(graph, retain);
            input.visit_administration(sink);
            physical = graph.finish();
        };
        edit(base, true);
        edit(alias, true);
        edit(copied, true);
        {
            const auto finance = weak_finance.lock();
            const pipeline_trace::Pause pause;
            ASSERT_TRUE(finance);
            EXPECT_EQ(physical.references(0, D::GeometryOwnerGraphEditor::external_kind(), finance), 2u);
        }
        edit(base, false); // shared receipt header still has another root hold
        {
            const auto finance = weak_finance.lock();
            const pipeline_trace::Pause pause;
            EXPECT_EQ(physical.references(0, D::GeometryOwnerGraphEditor::external_kind(), finance), 2u);
        }
        edit(copied, false);
        {
            const auto finance = weak_finance.lock();
            const pipeline_trace::Pause pause;
            EXPECT_EQ(physical.references(0, D::GeometryOwnerGraphEditor::external_kind(), finance), 1u);
        }
        edit(alias, false);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(physical.owner_count(), 0u);
        }
        // Reverse removal order exercises the other shared frontier.
        edit(base, true); edit(alias, true); edit(copied, true);
        edit(alias, false); edit(base, false); edit(copied, false);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(physical.owner_count(), 0u);
        }
        subset.reset();
        group.reset();
        broker = {};
        floor.floor.reset();
        {
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(weak_source.expired());
            EXPECT_FALSE(weak_finance.expired());
            EXPECT_EQ(*floor.total, selected.bytes);
        }
        weak_source.reset(); // release its actual shared control tail before raw comparison
        D::retention::ProbeScratch scratch(probe_storage);
        const std::array roots{D::retention::root(base), D::retention::root(alias), D::retention::root(copied)};
        const auto report = D::retention::inspect_retention(roots, {}, scratch);
        const auto raw = pipeline_trace::summary();
        const auto scalar_profiles = sizeof(A::Budget) + sizeof(std::size_t) + 8 * sizeof(void*);
        const pipeline_trace::Pause pause;
        ASSERT_TRUE(report.complete()) << report.detail;
        EXPECT_EQ(report.owner_union.bytes, raw.all_live + scalar_profiles);
    }
    base = {}; alias = {}; copied = {};
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(weak_finance.expired());
        EXPECT_EQ(*floor.total, 0u);
    }
    weak_finance.reset();
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, EmptyRemovalDoesNotCreditLiveOutsideOutputAndCanReAddTheSamePhysicalSource)
{
    PhysicalSourceFixture source;
    RealFloor floor(source.physical_bytes);
    SourceUnionFrame current;
    N::NativeTrackedOwner<std::array<std::byte, 20>> outside;
    const Trace trace;
    current = source_union_frame({}, source, floor, pipeline_trace::Phase::Candidate, 1);
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
            A::NativeSourceOwnerStore::mirror(floor.arena));
        const N::NativeConstructionAdmission admission(counter);
        outside = N::NativeAllocationEnvelopeAccess::make<std::array<std::byte, 20>>(admission);
    }
    const auto remove_all = [&](SourceUnionFrame& frame)
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
            A::NativeSourceOwnerStore::mirror(floor.arena));
        const N::NativeConstructionAdmission admission(counter);
        SimpleFluid::PLIC::GeometryAllocationLedger ordinary, bounded;
        std::size_t cursor = 0;
        D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ordinary, admission, cursor, cap);
        D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(bounded, admission, cursor, cap);
        A::NativeSourceOwnerStore empty;
        auto owners = empty.replace_in(frame.store, frame.owners, ordinary, bounded, cursor, cap);
        frame = {std::move(empty), std::move(owners)};
    };
    remove_all(current);
    remove_all(current);
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(current.owners.resident_bytes(), 0u);
        EXPECT_EQ(current.owners.native_paid_bytes(floor.arena), 0u);
        EXPECT_EQ(current.owners.direct_h_paid_bytes(floor.arena), 0u);
        EXPECT_GT(floor.arena->used, source.physical_bytes);
        EXPECT_TRUE(outside.owner);
    }
    current = source_union_frame(current, source, floor, pipeline_trace::Phase::Directory, 2);
    {
        const auto direct = current.owners.direct_h_paid_bytes(floor.arena);
        const auto total = current.owners.native_paid_bytes(floor.arena);
        const pipeline_trace::Pause pause;
        EXPECT_EQ(total - direct, source.physical_bytes);
        EXPECT_NO_THROW(current.store.require_current());
    }
    remove_all(current);
    outside = {};
    floor.floor.reset();
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, ReceiptIndexExactPeakAndOneByteShortPreserveAcceptedRootAndFunding)
{
    PhysicalSourceFixture source;
    RealFloor floor(source.physical_bytes);
    G::OwnerRecord selected;
    source.body.owners().owners[0].visit([&](const auto& row) { if (!selected.owner) selected = row; });
    const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
        A::NativeSourceOwnerStore::mirror(floor.arena));
    const N::NativeConstructionAdmission admission(counter);
    auto group = A::NativeSourceOwnerStore::body_coverage(source.body, A::AdaptiveLeafBodyRole::Leaf,
        floor.floor, admission, 1);
    A::NativeSourceCoverageBroker broker;
    std::shared_ptr<const A::NativeSourceSubsetCoverage> subset;
    {
        A::NativeSourceCoverageTransaction transaction(broker, admission);
        subset = transaction.collect(group, std::span(&selected, 1));
        transaction.prepare(); transaction.commit();
    }
    D::NativeOwnerReceipts accepted;
    {
        std::size_t cursor = 0;
        D::GeometryOwnerGraphEditor graph({}, 0, cursor, cap, nullptr, nullptr, admission.state());
        A::NativeSourceOwnerStore::record_subset(subset, graph);
        (void) graph.finish(); accepted = graph.paid_receipts();
    }
    const auto original = accepted;
    const auto token = subset->owners().front();
    const auto floor_bytes = floor.floor->bytes(), arena_used = floor.arena->used, leased = *floor.total;
    const auto baseline = admission.counters().used;
    const auto charge = D::NativePaidExtentCharge::state(admission.state());
    std::size_t work = 0;
    const G::Admission demand{&work, [](void* raw, std::size_t bytes)
        { *static_cast<std::size_t*>(raw) += bytes; }, nullptr, admission.state()};
    counter.owner()->peak = baseline;
    {
        const auto pilot = accepted.recorded(charge, demand);
        ASSERT_FALSE(pilot.same_root(accepted));
    }
    const auto required = counter.owner()->peak - baseline;
    ASSERT_GT(required, 0u);
    ASSERT_EQ(admission.counters().used, baseline);
    counter.owner()->limit = baseline + required - 1;
    EXPECT_THROW((void) accepted.recorded(charge, demand), std::exception);
    EXPECT_TRUE(accepted.same_root(original));
    EXPECT_EQ(admission.counters().used, baseline);
    EXPECT_EQ(floor.floor->bytes(), floor_bytes);
    EXPECT_EQ(floor.arena->used, arena_used);
    EXPECT_EQ(*floor.total, leased);
    EXPECT_TRUE(token->matches(selected.key.kind, selected.owner, selected.bytes));
    counter.owner()->limit = baseline + required;
    const auto exact = accepted.recorded(charge, demand);
    EXPECT_FALSE(exact.same_root(accepted));
    EXPECT_LE(admission.counters().used, counter.owner()->limit);
    EXPECT_TRUE(accepted.same_root(original));
}
TEST(NativeGeometryAccounting, RealIntrinsicAndExternalSubsetCategoriesDeduplicateTheSamePhysicalOwner)
{
    auto arena = std::make_shared<A::Budget>(A::Budget{cap});
    std::vector<std::size_t> slots{0, 2, 4};
    std::size_t transferred = slots.capacity() * sizeof(slots[0]);
    const auto ring_body_bytes = sizeof(A::SharedSubfaceRing) + transferred;
    arena->add(transferred);
    auto ring = A::SharedSubfaceRing::transfer(slots, *arena, arena, transferred);
    ASSERT_EQ(transferred, 0u);
    const auto ring_bytes = ring->charged_bytes();
    auto total = std::make_shared<std::size_t>(0);
    auto floor = std::make_shared<A::ContactExternalResidencyLease>(arena, total);
    floor->resize(ring_body_bytes);
    SourceUnionFrame frame;
    const Trace trace;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(cap, 0,
            A::NativeSourceOwnerStore::mirror(arena));
        const N::NativeConstructionAdmission admission(counter);
        const G::OwnerRecord selected{
            {0, D::GeometryOwnerGraphEditor::external_kind(), ring.get()}, ring, ring_body_bytes, 1};
        // First detach a genuinely pre-admitted legacy body floor. The
        // selected manifest authenticates the original ring before a native
        // contact graph has registered its independent intrinsic receipt.
        std::size_t membership_cursor = 0;
        D::GeometryOwnerGraphEditor membership({}, 0, membership_cursor, cap, nullptr, nullptr, admission.state());
        membership.owner(selected, true);
        auto coverage = A::NativeSourceOwnerStore::manifest_coverage(
            A::NativeSourceOwnerStore::single(membership.finish()), floor, admission, 1);
        A::NativeSourceCoverageBroker broker;
        std::shared_ptr<const A::NativeSourceSubsetCoverage> subset;
        {
            A::NativeSourceCoverageTransaction transaction(broker, admission);
            subset = transaction.collect(coverage, std::span(&selected, 1));
            transaction.prepare(); transaction.commit();
        }
        auto contact = std::make_unique<A::ContactPhysicalResidency>(arena);
        contact->retain(A::ContactPhysicalResidency::Role::SubfaceRing, ring, ring_bytes, ring->origin());
        auto contact_owners = A::NativeSourceOwnerStore::single(contact->source_owners());
        {
            const pipeline_trace::Pause pause;
            ASSERT_TRUE(contact->source_owners().admits(0, selected.key.kind, selected.owner, ring_body_bytes));
            EXPECT_LT(ring_body_bytes, ring_bytes);
            EXPECT_GT(contact_owners.payload_bytes(), ring_bytes);
            ASSERT_EQ(subset->bytes(), ring_body_bytes);
            EXPECT_EQ(floor->bytes(), 0u);
        }
        {
            // An actual same-arena intrinsic receipt funds a new selection.
            // Use a fresh broker to exercise current collection independently
            // of the outstanding legacy lease and test both categories below.
            A::NativeSourceCoverageBroker current;
            A::NativeSourceCoverageTransaction transaction(current, admission);
            const auto redundant =
                transaction.collect(coverage, std::span(&selected, 1), &contact->source_paid_receipts());
            transaction.prepare(); transaction.commit();
            const pipeline_trace::Pause pause;
            EXPECT_TRUE(redundant->owners().empty());
            EXPECT_EQ(floor->bytes(), 0u);
            EXPECT_EQ(*total, ring_body_bytes);
        }
        std::array<D::NativeOwnerReceipts, 4> receipts;
        // Contact administration has this exact receipt root as a live
        // input. The detached floor has its own receipt root and its own
        // lifetime. Preserve both actual inputs when composing the source.
        receipts[0] = contact->source_paid_receipts();
        {
            std::size_t cursor = 0;
            D::GeometryOwnerGraphEditor graph({}, 0, cursor, cap, nullptr, nullptr, admission.state());
            A::NativeSourceOwnerStore::record_subset(subset, graph);
            (void) graph.finish();
            receipts[1] = graph.paid_receipts();
        }
        std::size_t direct = 0, external = 0;
        struct Categories { std::size_t& direct; std::size_t& external; const void* ring; }
            categories{direct, external, ring.get()};
        for (const auto& component : receipts)
            component.visit({&categories, [](void* raw, const D::NativePaidExtentCharge& charge, std::size_t)
                {
                    auto& context = *static_cast<Categories*>(raw);
                    if (charge.owner_address() == context.ring)
                        ++(charge.direct_h() ? context.direct : context.external);
                }});
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(direct, 1u);
            EXPECT_EQ(external, 1u);
        }
        {
            using Receipts = D::NativeOwnerReceipts;
            Receipts both = receipts[0];
            std::size_t memory = 0;
            const G::Admission demand{&memory, [](void* raw, std::size_t bytes)
                { *static_cast<std::size_t*>(raw) += bytes; }, nullptr, admission.state()};
            struct Project { Receipts& both; G::Admission demand; } projection{both, demand};
            receipts[1].visit({&projection, [](void* raw, const D::NativePaidExtentCharge& charge, std::size_t)
                {
                    auto& p = *static_cast<Project*>(raw);
                    p.both = p.both.retained(charge, p.demand);
                }});
            auto removed = selected;
            removed.references = 0;
            auto pruned = both.edited({}, demand, std::span(&removed, 1));
            const pipeline_trace::Pause pause;
            for (const auto& input : {receipts[0], receipts[1]})
                input.visit({&pruned, [](void* raw, const auto& receipt, std::size_t)
                    {
                        if (receipt.financial_address())
                            EXPECT_FALSE(static_cast<const Receipts*>(raw)->contains(receipt));
                    }});
        }
        std::array<D::OwnerManifestPair, 4> components;
        components[0] = std::move(contact_owners);
        frame.store = A::NativeSourceOwnerStore::assemble({}, std::move(components), admission, nullptr, {},
            {contact->source_administration(), {}}, coverage, {}, std::move(receipts));
        {
            SimpleFluid::PLIC::GeometryAllocationLedger ordinary, bounded;
            std::size_t cursor = 0;
            D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(ordinary, admission, cursor, cap);
            D::GeometryAllocationOwnerAccess::begin_native_bookkeeping(bounded, admission, cursor, cap);
            frame.owners = frame.store.replace_in({}, {}, ordinary, bounded, cursor, cap);
        }
        contact.reset(); subset.reset(); broker = {}; coverage.reset();
        // The SourceRoot now owns the real component sidecar and its exact
        // financial token; no controller or structural subset wrapper remains.
    }
    frame.store = {};
    floor.reset();
    const auto raw = pipeline_trace::summary();
    const auto direct_paid = frame.owners.direct_h_paid_bytes(arena);
    const auto all_paid = frame.owners.native_paid_bytes(arena);
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(frame.owners.native_group_covered_bytes(arena), 0u);
        EXPECT_EQ(all_paid, direct_paid);
        EXPECT_EQ(direct_paid, raw.all_live + ring_bytes + sizeof(A::Budget) + 4 * sizeof(void*));
    }
    frame = {};
    {
        const pipeline_trace::Pause pause;
        EXPECT_EQ(*total, 0u);
    }
    const pipeline_trace::Pause pause;
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeGeometryAccounting, ActualAdmittedCellAndFaceMatchFullGraphIndependentCensusRawAndWeakTails)
{
    using P = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>;
    // Certificate creation is physical input preparation, not a native-library
    // warm-up: capture starts before the real admitted decoder/factory calls.
    const auto packet = P::box({-.75, .125, -2}, {1.5, 2.75, 4})->encode_certificate();
    std::vector<std::byte> probe_storage(8 * 1024 * 1024);
    std::weak_ptr<const P> weak_cell;
    std::weak_ptr<const SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Ordinary>> weak_face;
    std::weak_ptr<const N::NativeAllocationState> weak_state;
    const Trace trace;
    {
        const auto counter = N::NativeConstructionCounterAccess::create<Counter>(16 * 1024 * 1024);
        const N::NativeConstructionAdmission admission(counter);
        weak_state = admission.state();
        auto cell = D::NativeCellCodecAccess::decode(packet, 16 * 1024 * 1024, admission);
        auto face = D::NativePlanarOwnerAccess::cell_face(*cell, 0, admission);
        auto alias = cell;
        const auto support = D::GeometryIdentityAccess::support_axis_admitted(0, .5L, admission);
        const auto translated_support = D::GeometryIdentityAccess::translated_support_admitted(support,
            {.25L, .125L, 0}, admission);
        weak_cell = cell; weak_face = face;
        const auto raw = pipeline_trace::summary();
        // Freeze the actual native request measurement. The following Graph,
        // full Ledger and PMR Census are independent diagnostic artifacts and
        // are not charged to the native factory's physical output.
        pipeline_trace::recording.store(false);
        {
            const pipeline_trace::Pause pause;
            EXPECT_EQ(admission.statistics().actual_live, raw.all_live);
            EXPECT_EQ(admission.statistics().reserved_live, 0u);
            EXPECT_EQ(admission.statistics().profiled_live, 0u);
        }
        std::size_t used = 0;
        G::GeometryOwnerManifest physical;
        {
            D::GeometryOwnerGraphEditor graph({}, 0, used, 16 * 1024 * 1024);
            graph.geometry(cell, true); graph.geometry(alias, true); graph.planar(face, true);
            graph.identity(translated_support, true);
            physical = graph.finish();
        }
        SimpleFluid::PLIC::GeometryAllocationLedger full;
        used = 0;
        full.retain(cell, used, 16 * 1024 * 1024);
        full.retain(alias, used, 16 * 1024 * 1024);
        full.retain_planar_face(face, used, 16 * 1024 * 1024);
        full.retain(translated_support, used, 16 * 1024 * 1024);
        D::retention::ProbeScratch scratch(probe_storage);
        const std::array roots{D::retention::root(cell), D::retention::root(face), D::retention::root(translated_support)};
        const std::array<const SimpleFluid::PLIC::GeometryAllocationLedger*, 1> ledgers{&full};
        const auto census = D::retention::inspect_retention(roots, ledgers, scratch, {10000, 10000, true});
        ASSERT_TRUE(census.complete()) << census.detail;
        EXPECT_EQ(physical.payload_bytes(), raw.all_live);
        EXPECT_EQ(full.payload_bytes(), raw.all_live);
        EXPECT_EQ(census.owner_union.bytes, raw.all_live);
        EXPECT_EQ(census.application_only.owners, 0u);
        EXPECT_EQ(census.ledger_only.owners, 0u);
        EXPECT_GT(census.maximum_identity_dependency_depth, 0u);
    }
    // The real object destructors have run. Weak observations deliberately keep
    // only their exact object/control envelopes alive, never semantic children.
    EXPECT_TRUE(weak_cell.expired());
    EXPECT_TRUE(weak_face.expired());
    if (const auto state = weak_state.lock())
        EXPECT_EQ(state->statistics.actual_live, pipeline_trace::summary().all_live);
    weak_cell.reset(); weak_face.reset();
    EXPECT_TRUE(weak_state.expired());
    weak_state.reset();
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
} // namespace
