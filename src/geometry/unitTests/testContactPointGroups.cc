#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/ContactPointGroups.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace
{
thread_local std::size_t* allocation_count = nullptr;
struct HeapAllocation
{
    void* address = nullptr;
    std::size_t bytes = 0;
    bool live = false;
};
struct HeapAudit
{
    std::array<HeapAllocation, 2048> allocations{};
    std::size_t count = 0;
    bool overflow = false;
    void allocated(void* address, std::size_t bytes) noexcept
    {
        if (count == allocations.size())
        {
            overflow = true;
            return;
        }
        allocations[count++] = {address, bytes, true};
    }
    void freed(void* address) noexcept
    {
        for (std::size_t i = count; i-- > 0;)
            if (allocations[i].address == address && allocations[i].live)
            {
                allocations[i].live = false;
                return;
            }
    }
};
thread_local HeapAudit* heap_audit = nullptr;
void release_heap(void* address) noexcept
{
    if (heap_audit)
        heap_audit->freed(address);
    std::free(address);
}
} // namespace
void* operator new(std::size_t size)
{
    if (auto* result = std::malloc(size ? size : 1))
    {
        if (allocation_count)
            ++*allocation_count;
        if (heap_audit)
            heap_audit->allocated(result, size);
        return result;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* value) noexcept
{
    release_heap(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    release_heap(value);
}
void operator delete[](void* value) noexcept
{
    release_heap(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    release_heap(value);
}

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Bytes = SimpleFluid::Meshes::AMR::AdaptiveBytes;
constexpr auto ample = std::size_t(64 * 1024 * 1024);
D::ContactPointIdentity point(long double x, long double y = 0, long double z = 0)
{
    return P::details::GeometryIdentityAccess::point({x, y, z});
}
Bytes old_key(const Bytes& domain, const D::ContactPointIdentity& value)
{
    D::Writer writer;
    writer.bytes(domain);
    writer.bytes(P::canonical_key(value));
    return std::move(writer.data);
}
std::optional<std::pair<const void*, D::CapturedPointReference>> physical_location(
    const D::PointFactDirectory& directory, D::ContactPointRef reference)
{
    std::optional<std::pair<const void*, D::CapturedPointReference>> location;
    directory.visit_borrowed_facts(
        [&](const D::PointFactDirectory::BorrowedFact& fact)
        {
            if (fact.reference == reference && fact.physical_page)
                location.emplace(fact.physical_page, fact.captured);
        });
    return location;
}
std::shared_ptr<D::ContactIncidenceRow> row(long double physical, std::uint64_t generation, long double normalized = 0)
{
    auto result = std::make_shared<D::ContactIncidenceRow>();
    result->incidence_generation = generation;
    result->points.push_back({point(normalized), point(physical)});
    return result;
}
struct ConstantHash
{
    std::uint64_t operator()(D::ContactPointKeyView) const noexcept { return 7; }
    std::uint64_t operator()(std::span<const char>) const noexcept { return 7; }
};
TEST(ContactPointGroups, EmptyConstructionPreflightIncludesEveryInitialDynamicOwner)
{
    constexpr auto caller_live = std::size_t{73};
    constexpr auto required = D::ContactPointGroups<>::empty_dynamic_admission_bytes();
    static_assert(required > 0);
    for (const bool admit : {true, false})
    {
        auto arena =
            std::make_shared<D::Budget>(D::Budget{caller_live + required - (admit ? std::size_t{0} : std::size_t{1})});
        arena->add(caller_live);
        const auto construct = [&]
        {
            D::ContactPointGroups groups(arena);
            EXPECT_EQ(groups.size(), 0u);
            EXPECT_EQ(arena->used, caller_live + required);
        };
        if (admit)
            EXPECT_NO_THROW(construct());
        else
            EXPECT_THROW(construct(), std::invalid_argument);
        EXPECT_EQ(arena->used, caller_live);
        EXPECT_GT(arena->cumulative, caller_live);
    }
}
TEST(ContactPointGroups, WireOrderingMatchesIndependentWriterOracle)
{
    std::vector<Bytes> domains{{}, {char(0x7f)}, {char(0x80)}, {char(0xff)}};
    for (const auto size : {127u, 128u, 255u, 256u})
        domains.emplace_back(size, char(0x80));
    const std::array points{point(0), point(-1), point(1), point(std::nextafter(1.L, 2.L)), point(1.1L, 2.3L, 4.7L)};
    D::ContactPointWireLess less;
    D::ContactPointWireHasher hash;
    for (const auto& first_domain : domains)
        for (const auto& first : points)
        {
            const auto a = old_key(first_domain, first);
            const D::ContactPointKeyView av{first_domain, &first};
            EXPECT_EQ(hash(av), hash(std::span<const char>(a)));
            for (const auto& second_domain : domains)
                for (const auto& second : points)
                {
                    const auto b = old_key(second_domain, second);
                    const D::ContactPointKeyView bv{second_domain, &second};
                    EXPECT_EQ(less(av, bv), a < b);
                    EXPECT_EQ(less(av, std::span<const char>(b)), a < b);
                    EXPECT_EQ(less(std::span<const char>(a), bv), a < b);
                }
        }
}
TEST(ContactPointGroups, WireQueriesAllocateNothingAndRestoreFloatingEnvironment)
{
    struct Restore
    {
        std::fenv_t value;
        Restore() { std::fegetenv(&value); }
        ~Restore() { std::fesetenv(&value); }
    } restore;
    const Bytes domain{char(0x80), char(0xff)};
    const auto first = point(1.1L, 2.3L, 4.7L), second = point(std::nextafter(1.1L, 2.L), 2.3L, 4.7L);
    const auto encoded = old_key(domain, first);
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> groups(arena);
    const auto expected = groups.intern(domain, first);
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
#if defined(__GLIBC__)
    const auto old_traps = fegetexcept();
    feenableexcept(FE_INVALID);
#endif
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    std::size_t allocations = 0;
    allocation_count = &allocations;
    const bool comparison = D::ContactPointWireLess{}({domain, &first}, {domain, &second});
    const auto hash = D::ContactPointWireHasher{}({domain, &first});
    const bool encoded_comparison = D::ContactPointWireLess{}({domain, &first}, std::span<const char>(encoded));
    const auto typed_found = groups.find({domain, &first});
    const auto encoded_found = groups.find_encoded(encoded);
    const auto flags_after = std::fetestexcept(FE_ALL_EXCEPT);
    const auto rounding_after = std::fegetround();
#if defined(__GLIBC__)
    const auto traps_after = fegetexcept();
#endif
    allocation_count = nullptr;
    EXPECT_FALSE(encoded_comparison);
    EXPECT_EQ(typed_found, expected);
    EXPECT_EQ(encoded_found, expected);
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(hash, D::ContactPointWireHasher{}(std::span<const char>(encoded)));
    EXPECT_EQ(comparison, old_key(domain, first) < old_key(domain, second));
    EXPECT_EQ(rounding_after, FE_UPWARD);
    EXPECT_EQ(flags_after, flags);
#if defined(__GLIBC__)
    EXPECT_EQ(traps_after, old_traps | FE_INVALID);
#endif
}
TEST(ContactPointGroups, FullValueCollisionEqualityAndArbitraryEncodedProbes)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::ContactPointGroups<ConstantHash> groups(arena);
        const Bytes domain{char(0x80)};
        std::vector<D::ContactGroupId> ids;
        for (unsigned i = 0; i < 40; ++i)
            ids.push_back(groups.intern(domain, point(i)));
        for (unsigned i = 0; i < 40; ++i)
        {
            const auto value = point(i);
            EXPECT_EQ(groups.find({domain, &value}), ids[i]);
            auto encoded = old_key(domain, value);
            EXPECT_EQ(groups.find_encoded(encoded), ids[i]);
            encoded.pop_back();
            EXPECT_FALSE(groups.find_encoded(encoded));
        }
        for (const Bytes encoded : {Bytes{}, Bytes{char(0xff)}, Bytes(29, char(0x80))})
            EXPECT_NO_THROW(EXPECT_FALSE(groups.find_encoded(encoded)));
        const auto ordered = groups.ordered_ids();
        std::vector<Bytes> actual;
        for (const auto id : ordered.ids())
        {
            const auto key = groups.key(id);
            actual.push_back(old_key(Bytes(key.domain.begin(), key.domain.end()), key.point));
        }
        EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));
        const std::array selected{ids.back(), ids.front(), ids.back()};
        const auto ordered_selected = groups.ordered_ids(std::span<const D::ContactGroupId>(selected));
        EXPECT_EQ(ordered_selected.ids().size(), 2u);
        EXPECT_EQ(groups.normalized(ids.front()), groups.key(ids.front()).point);
        EXPECT_EQ(
            groups.point_directory().get(groups.normalized_reference(ids.front())), groups.key(ids.front()).point);
        EXPECT_EQ(groups.point_directory().live_points(), ids.size());
        EXPECT_EQ(groups.domain(ids.front()), domain);
    }
    EXPECT_EQ(arena->used, 0u);
}
TEST(ContactPointGroups, SelectivePageLocatorDropsUnreferencedBatchPages)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::PointFactDirectory directory(arena);
        D::ContactPointRef retained{};
        std::weak_ptr<const D::ContactPointPageView> old_view;
        std::size_t before = 0;
        {
            D::PointFactDirectory::PointPool pool(*arena, arena);
            for (unsigned i = 0; i < 48; ++i)
                (void) pool.intern(point(i));
            D::ContactPointBatch batch(pool);
            auto array = D::ContactPointArray::pooled(batch, 3, *arena);
            for (const auto slot : {0u, 16u, 32u})
            {
                const auto reference = batch.capture(slot);
                array.push_reference(reference, reference, *arena);
            }
            batch.freeze();
            EXPECT_EQ(array.borrowed_page_view(), array.normalized_locator(1)->view);
            ASSERT_EQ(batch.page_count(), 3u);
            old_view = array.page_view();
            retained = directory.retain(*array.normalized_locator(1));
            EXPECT_EQ(directory.get(retained), point(16));
            before = arena->used;
            // A row owner normally releases these already admitted buffers.
            arena->used -= array.storage_bytes();
        }
        EXPECT_TRUE(old_view.expired());
        EXPECT_LT(arena->used, before);
        EXPECT_EQ(directory.live_pages(), 1u);
        EXPECT_EQ(directory.get(retained), point(16));
        directory.release(retained);
        directory.reclaim();
        EXPECT_EQ(directory.live_pages(), 0u);
        EXPECT_THROW((void) directory.get(retained), std::out_of_range);
        const auto replacement = directory.retain(point(99));
        EXPECT_EQ(replacement.slot, retained.slot);
        EXPECT_NE(replacement.generation, retained.generation);
    }
    EXPECT_EQ(arena->used, 0u);
}
TEST(ContactPointGroups, TargetPoolCreditReuseAndAncestorArenaIndependence)
{
    auto first = std::make_shared<D::Budget>(D::Budget{ample});
    auto second = std::make_shared<D::Budget>(D::Budget{ample});
    auto third = std::make_shared<D::Budget>(D::Budget{ample});
    std::unique_ptr<D::PointFactDirectory> descendant;
    D::ContactPointRef ref{};
    {
        D::PointFactDirectory::PointPool pool(*first, first);
        const auto slot = pool.intern(point(8));
        auto source = pool.page_view();
        const auto captured = source->capture(pool, slot);
        source->freeze();
        D::PointFactDirectory initial(first);
        ref = initial.retain({source.get(), captured});
        {
            D::PointFactDirectory::PointPool target(pool, *second, second);
            auto target_view = target.page_view();
            const auto target_captured = target_view->capture(target, slot);
            target_view->freeze();
            D::PointFactDirectory middle(initial, second, &target);
            EXPECT_EQ(middle.credit_identity(ref), target_view->page_credit_identity(target_captured));
            EXPECT_NE(middle.credit_identity(ref), source->page_credit_identity(captured));
            EXPECT_EQ(middle.inherited_credit_overhead_bytes(), 0u);
            descendant = std::make_unique<D::PointFactDirectory>(middle, third);
        }
        EXPECT_EQ(second->used, 0u);
    }
    EXPECT_EQ(first->used, 0u);
    ASSERT_TRUE(descendant);
    EXPECT_EQ(descendant->get(ref), point(8));
    descendant.reset();
    EXPECT_EQ(third->used, 0u);
}
TEST(ContactPointGroups, TouchedFanEditsPreserveSurvivorsAppendAndResponseOrder)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::ContactPointGroups<> groups(arena);
        const auto group = groups.intern({}, point(0));
        auto catalog = groups.catalog();
        const auto a = catalog->bind(row(10, 101), 10), b = catalog->bind(row(20, 102), 20);
        const auto c = catalog->bind(row(30, 103), 30), d = catalog->bind(row(40, 104), 40);
        const auto physical = catalog->points.retain(point(99));
        const std::array locals{
            D::ContactLocalOccurrence{a, 0}, D::ContactLocalOccurrence{b, 0}, D::ContactLocalOccurrence{c, 0}};
        const std::array remote{D::ContactRemoteOccurrence{7, physical}, D::ContactRemoteOccurrence{8, physical}};
        const std::array edit{D::ContactPointGroups<>::Edit{group, {}, locals, remote}};
        groups.replace_fans(edit);
        const auto old = groups.fan(group);
        const std::array removed{b};
        const std::array appended{D::ContactLocalOccurrence{d, 0}};
        const std::array next_remote{D::ContactRemoteOccurrence{9, physical}, D::ContactRemoteOccurrence{7, physical}};
        const std::array next_edit{D::ContactPointGroups<>::Edit{group, removed, appended, next_remote}};
        groups.replace_fans(next_edit);
        const auto next = groups.fan(group);
        ASSERT_EQ(next.local_count(), 3u);
        EXPECT_EQ(next.local(0).binding.record, 10u);
        EXPECT_EQ(next.local(1).binding.record, 30u);
        EXPECT_EQ(next.local(2).binding.record, 40u);
        EXPECT_EQ(next.remote(0).record, 9u);
        EXPECT_EQ(next.remote(1).record, 7u);
        EXPECT_EQ(old.local(1).binding.record, 20u);
        EXPECT_EQ(old.remote(0).record, 7u);
        EXPECT_FALSE(next.owner()->block().wide());
        for (const auto binding : {a, b, c, d})
            catalog->release(binding);
        catalog->points.release(physical);
    }
    EXPECT_EQ(arena->used, 0u);
}
TEST(ContactPointGroups, SnapshotPinsBindingAndGenerationUntilItRetires)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::ContactPointGroups<> groups(arena);
        const auto group = groups.intern({}, point(0));
        auto catalog = groups.catalog();
        const auto old_binding = catalog->bind(row(10, std::numeric_limits<std::uint64_t>::max()), 10);
        const std::array local{D::ContactLocalOccurrence{old_binding, 0}};
        const std::array edit{D::ContactPointGroups<>::Edit{group, {}, local, {}}};
        groups.replace_fans(edit);
        auto snapshot = groups.fan(group);
        const std::array removed{old_binding};
        const std::array empty_edit{D::ContactPointGroups<>::Edit{group, removed, {}, {}}};
        groups.replace_fans(empty_edit);
        catalog->release(old_binding);
        groups.reclaim();
        const auto other = catalog->bind(row(20, 12), 20);
        EXPECT_NE(other.slot, old_binding.slot);
        EXPECT_EQ(snapshot.local(0).binding.incidence_generation, std::numeric_limits<std::uint64_t>::max());
        snapshot = {};
        groups.reclaim();
        EXPECT_THROW((void) catalog->binding(old_binding), std::out_of_range);
        const auto replacement = catalog->bind(row(30, 13), 30);
        EXPECT_EQ(replacement.slot, old_binding.slot);
        EXPECT_NE(replacement.generation, old_binding.generation);
        catalog->release(other);
        catalog->release(replacement);
    }
    EXPECT_EQ(arena->used, 0u);
}
TEST(ContactPointGroups, OrderedIdsDelayGroupReuseAndRejectStaleIds)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> groups(arena);
    const auto old = groups.intern({}, point(1));
    {
        auto ordered = groups.ordered_ids();
        groups.retire(old);
        groups.reclaim();
        EXPECT_EQ(groups.key(old).point, point(1));
        const auto other = groups.intern({}, point(2));
        EXPECT_NE(other.slot, old.slot);
    }
    groups.reclaim();
    EXPECT_THROW((void) groups.key(old), std::out_of_range);
    const auto replacement = groups.intern({}, point(3));
    EXPECT_EQ(replacement.slot, old.slot);
    EXPECT_NE(replacement.generation, old.generation);
}
TEST(ContactPointGroups, SharedMultiGroupBlocksReleaseDeadSlicesAndNeverReviveRecycledSlots)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto fork_arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::ContactPointGroups<> groups(arena);
        auto catalog = groups.catalog();
        const auto first = groups.intern({}, point(0)), second = groups.intern({}, point(1));
        const auto a = catalog->bind(row(10, 101, 0), 10), b = catalog->bind(row(20, 102, 1), 20);
        const auto c = catalog->bind(row(30, 103, 0), 30);
        const std::array first_local{D::ContactLocalOccurrence{a, 0}};
        const std::array second_local{D::ContactLocalOccurrence{b, 0}};
        const std::array edits{D::ContactPointGroups<>::Edit{first, {}, first_local, {}},
            D::ContactPointGroups<>::Edit{second, {}, second_local, {}}};
        groups.replace_fans(edits);
        auto old_owner = groups.fan(second).owner();
        EXPECT_EQ(groups.fan(first).owner(), old_owner);
        EXPECT_EQ(groups.fan(second).group(), 1u);
        catalog->release(a);
        const std::array removed{a};
        const std::array appended{D::ContactLocalOccurrence{c, 0}};
        const std::array replacement{D::ContactPointGroups<>::Edit{first, removed, appended, {}}};
        groups.replace_fans(replacement);
        groups.reclaim();
        EXPECT_THROW((void) catalog->binding(a), std::out_of_range);
        const auto recycled = catalog->bind(row(40, 104), 40);
        EXPECT_EQ(recycled.slot, a.slot);
        EXPECT_NE(recycled.generation, a.generation);
        EXPECT_THROW((void) D::ContactFanView(old_owner, 0), std::out_of_range);
        EXPECT_EQ(groups.fan(second).local(0).binding.record, 20u);
        {
            D::ContactPointGroups<> fork(groups, fork_arena);
            EXPECT_EQ(fork.fan(first).local(0).binding.record, 30u);
            EXPECT_EQ(fork.fan(second).local(0).binding.record, 20u);
            EXPECT_EQ(&fork.fan(second).owner()->block(), &old_owner->block());
        }
        for (const auto binding : {b, c, recycled})
            catalog->release(binding);
    }
    EXPECT_EQ(arena->used, 0u);
    EXPECT_EQ(fork_arena->used, 0u);
}
TEST(ContactPointGroups, MismatchedTypedOccurrenceRejectsBeforeFanPublication)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> groups(arena);
    const auto group = groups.intern({}, point(0));
    const auto wrong = groups.catalog()->bind(row(7, 8, 1), 7);
    const std::array local{D::ContactLocalOccurrence{wrong, 0}};
    const std::array edit{D::ContactPointGroups<>::Edit{group, {}, local, {}}};
    EXPECT_THROW(groups.replace_fans(edit), std::invalid_argument);
    EXPECT_EQ(groups.fan(group).local_count(), 0u);
    groups.catalog()->release(wrong);
}
TEST(ContactPointGroups, ExactRowGenerationMismatchRejectsReaderAndStillReleases)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactFanCatalog catalog(arena);
    auto mutable_owner = row(1, 100);
    const auto id = catalog.bind(mutable_owner, 1);
    const auto same = catalog.bind(mutable_owner, 1);
    const auto other_record = catalog.bind(mutable_owner, 2);
    EXPECT_EQ(same, id);
    EXPECT_NE(other_record, id);
    catalog.release(same);
    catalog.release(other_record);
    mutable_owner->incidence_generation = 101;
    EXPECT_THROW((void) catalog.binding(id), std::invalid_argument);
    EXPECT_NO_THROW(catalog.release(id));
    EXPECT_NO_THROW(catalog.reclaim());
    EXPECT_THROW((void) catalog.binding(id), std::out_of_range);
}
TEST(ContactPointGroups, ForkSharesImmutableBlockWithIndependentCatalogAndArena)
{
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto candidate_arena = std::make_shared<D::Budget>(D::Budget{ample});
    std::unique_ptr<D::ContactPointGroups<>> candidate;
    D::ContactGroupId id{};
    D::ContactFanView escaped;
    {
        D::ContactPointGroups<> source(source_arena);
        id = source.intern({}, point(0));
        const auto binding = source.catalog()->bind(row(5, 9), 5);
        const std::array local{D::ContactLocalOccurrence{binding, 0}};
        const std::array edit{D::ContactPointGroups<>::Edit{id, {}, local, {}}};
        source.replace_fans(edit);
        source.catalog()->release(binding);
        escaped = source.fan(id);
        candidate = std::make_unique<D::ContactPointGroups<>>(source, candidate_arena);
        const auto forked = candidate->fan(id);
        EXPECT_EQ(&forked.owner()->block(), &escaped.owner()->block());
        EXPECT_NE(forked.owner()->catalog_identity(), escaped.owner()->catalog_identity());
        candidate->retire(id);
        candidate->reclaim();
        EXPECT_EQ(source.fan(id).local(0).binding.record, 5u);
        const auto probe = point(0);
        EXPECT_FALSE(candidate->find({{}, &probe}));
    }
    EXPECT_EQ(escaped.local(0).binding.incidence_generation, 9u);
    escaped = {};
    EXPECT_EQ(source_arena->used, 0u);
    candidate.reset();
    EXPECT_EQ(candidate_arena->used, 0u);
}
TEST(ContactPointGroups, WideFallbackIsCheckedAndLossless)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactGroupAllocation allocation(arena);
    const auto wide = std::size_t(std::numeric_limits<std::uint32_t>::max()) + 1;
    const std::array local{D::ContactLocalOccurrence{{wide, std::numeric_limits<std::uint64_t>::max()}, wide + 7}};
    const std::array remote{
        D::ContactRemoteOccurrence{wide + 9, {wide + 11, std::numeric_limits<std::uint64_t>::max()}}};
    const std::array input{D::ContactFanBlock::Input{local, remote}};
    const auto block = D::ContactFanBlock::build(input, allocation);
    ASSERT_TRUE(block->wide());
    EXPECT_EQ(block->local(0, 0), std::make_pair(wide, wide + 7));
    EXPECT_EQ(block->remote(0, 0), std::make_pair(wide + 9, wide + 11));
    EXPECT_EQ(block->storage_bytes(), allocation.bytes());
}
TEST(ContactPointGroups, CompleteAdmissionRejectsExactlyOneByteBelowObservedPeak)
{
    const auto run = [](std::shared_ptr<D::Budget> arena)
    {
        D::ContactPointGroups<> groups(arena);
        const auto group = groups.intern(Bytes(129, char(0x80)), point(0));
        auto owner = row(5, 17);
        owner->domain = Bytes(129, char(0x80));
        const auto binding = groups.catalog()->bind(owner, 5);
        const auto physical = groups.catalog()->points.retain(point(99));
        const std::array local{D::ContactLocalOccurrence{binding, 0}};
        const std::array remote{D::ContactRemoteOccurrence{9, physical}};
        const std::array edit{D::ContactPointGroups<>::Edit{group, {}, local, remote}};
        groups.replace_fans(edit);
        groups.catalog()->release(binding);
        groups.catalog()->points.release(physical);
    };
    auto measured = std::make_shared<D::Budget>(D::Budget{ample});
    run(measured);
    const auto peak = measured->peak;
    RecordProperty("admitted_operation_peak_bytes", std::to_string(peak));
    EXPECT_EQ(measured->used, 0u);
    auto admitted = std::make_shared<D::Budget>(D::Budget{peak});
    EXPECT_NO_THROW(run(admitted));
    EXPECT_EQ(admitted->used, 0u);
    auto rejected = std::make_shared<D::Budget>(D::Budget{peak - 1});
    EXPECT_THROW(run(rejected), std::invalid_argument);
    EXPECT_EQ(rejected->used, 0u);
}
TEST(ContactPointGroups, FailedForkAdmissionPreservesSourceAndReleasesEveryCandidateCredit)
{
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> source(source_arena);
    const auto group = source.intern({}, point(0));
    const auto binding = source.catalog()->bind(row(10, 101), 10);
    const std::array local{D::ContactLocalOccurrence{binding, 0}};
    const std::array edit{D::ContactPointGroups<>::Edit{group, {}, local, {}}};
    source.replace_fans(edit);
    source.catalog()->release(binding);
    const auto before = source_arena->used;
    auto measured = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::ContactPointGroups<> candidate(source, measured);
        EXPECT_EQ(candidate.fan(group).local(0).binding.record, 10u);
    }
    EXPECT_EQ(measured->used, 0u);
    RecordProperty("candidate_fork_peak_bytes", std::to_string(measured->peak));
    auto exact = std::make_shared<D::Budget>(D::Budget{measured->peak});
    EXPECT_NO_THROW({ D::ContactPointGroups<> candidate(source, exact); });
    EXPECT_EQ(exact->used, 0u);
    auto rejected = std::make_shared<D::Budget>(D::Budget{measured->peak - 1});
    EXPECT_THROW({ D::ContactPointGroups<> candidate(source, rejected); }, std::invalid_argument);
    EXPECT_EQ(rejected->used, 0u);
    EXPECT_EQ(source_arena->used, before);
    EXPECT_EQ(source.fan(group).local(0).binding.record, 10u);
    EXPECT_EQ(source.size(), 1u);
}

TEST(ContactPointGroups, FallbackPagesPackReclaimAndForkPartialTailsIndependently)
{
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto fork_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto source = std::make_unique<D::PointFactDirectory>(source_arena);
    std::vector<D::ContactPointRef> refs;
    for (unsigned i = 0; i < 49; ++i)
        refs.push_back(source->retain(point(i + 1024)));
    ASSERT_EQ(source->fallback_pages(), 4u);
    const auto stable = physical_location(*source, refs[0]);
    ASSERT_TRUE(stable);
    source->seal_fallback();
    D::PointFactDirectory fork(*source, fork_arena);
    EXPECT_EQ(physical_location(fork, refs[0]), stable);
    EXPECT_NE(fork.credit_identity(refs[0]), source->credit_identity(refs[0]));
    for (unsigned i = 0; i < 16; ++i)
        source->release(refs[i]);
    source->reclaim();
    EXPECT_EQ(source->fallback_pages(), 3u);
    EXPECT_EQ(fork.fallback_pages(), 4u);
    EXPECT_THROW((void) source->get(refs[0]), std::out_of_range);
    EXPECT_EQ(fork.get(refs[0]), point(1024));
    source.reset();
    EXPECT_EQ(source_arena->used, 0u);
    for (unsigned i = 0; i < refs.size(); ++i)
        EXPECT_EQ(fork.get(refs[i]), point(i + 1024));
    for (const auto ref : refs)
        fork.release(ref);
    fork.reclaim();
    EXPECT_EQ(fork.fallback_pages(), 0u);

    D::PointFactDirectory partial(source_arena);
    const auto first = partial.retain(point(1));
    (void) partial.retain(point(2));
    (void) partial.retain(point(3));
    D::PointFactDirectory other(partial, fork_arena);
    const auto partial_location = physical_location(partial, first), other_location = physical_location(other, first);
    ASSERT_TRUE(partial_location);
    ASSERT_TRUE(other_location);
    EXPECT_NE(other_location->first, partial_location->first);
    EXPECT_EQ(other_location->second, partial_location->second);
    const auto fourth = other.retain(point(99));
    EXPECT_EQ(other.fallback_pages(), 1u);
    EXPECT_EQ(partial.fallback_pages(), 1u);
    EXPECT_EQ(other.get(fourth), point(99));
    EXPECT_EQ(partial.get(first), point(1));
    EXPECT_EQ(partial.live_points(), 3u);
}
TEST(ContactPointGroups, PeriodicRemotePhysicalValuesPackAndSurviveOldFanSnapshots)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> groups(arena);
    const auto group = groups.intern({}, point(0));
    auto catalog = groups.catalog();
    std::vector<D::ContactRemoteOccurrence> remote;
    for (unsigned i = 0; i < 49; ++i)
        remote.push_back({i + 1, catalog->points.retain(point(i + 1024))});
    EXPECT_EQ(catalog->points.fallback_pages(), 4u);
    const std::array edit{D::ContactPointGroups<>::Edit{group, {}, {}, remote}};
    groups.replace_fans(edit);
    for (const auto occurrence : remote)
        catalog->points.release(occurrence.physical);
    auto old = groups.fan(group);
    const std::array empty{D::ContactPointGroups<>::Edit{group, {}, {}, {}}};
    groups.replace_fans(empty);
    groups.reclaim();
    EXPECT_EQ(catalog->points.fallback_pages(), 4u);
    for (unsigned i = 0; i < 49; ++i)
        EXPECT_EQ(old.remote(i).physical, point(i + 1024));
    old = {};
    groups.reclaim();
    EXPECT_EQ(catalog->points.live_points(), 1u);
    EXPECT_EQ(catalog->points.fallback_pages(), 1u);
    groups.retire(group);
    groups.reclaim();
    EXPECT_EQ(catalog->points.fallback_pages(), 0u);
}
TEST(ContactPointGroups, FramedEmissionMatchesIndependentNestedWriterOracle)
{
    const std::array domains{Bytes{}, Bytes{char(0x80)}, Bytes(256, char(0xff))};
    const auto value = point(1.1L, -7, std::nextafter(1.L, 2.L));
    for (const auto& domain : domains)
    {
        D::Writer oracle;
        oracle.bytes(old_key(domain, value));
        D::Writer actual;
        const D::ContactPointKeyView key{domain, &value};
        actual.data.reserve(D::framed_contact_point_key_size(key));
        std::size_t allocations = 0;
        allocation_count = &allocations;
        D::append_framed_contact_point_key(actual, key);
        allocation_count = nullptr;
        EXPECT_EQ(allocations, 0u);
        EXPECT_EQ(actual.data, oracle.data);
        EXPECT_EQ(D::contact_point_key_size(key), old_key(domain, value).size());
        EXPECT_EQ(D::framed_contact_point_key_size(key), oracle.data.size());
    }
}
TEST(ContactPointGroups, IdentityPacketAppendMatchesLegacyBytesAndFloatingEnvironmentWithoutHeapScratch)
{
    struct Restore
    {
        std::fenv_t value;
        Restore() { std::feholdexcept(&value); }
        ~Restore() { std::fesetenv(&value); }
    } restore;
    const auto maximum = std::numeric_limits<long double>::max();
    const auto minimum = std::numeric_limits<long double>::min();
    const auto tiny = std::numeric_limits<long double>::denorm_min();
    const std::array points{point(0, -0.L, 0), point(1.1L, -2.3L, 4.7L),
        point(0x1p40L + 1.1L, -0x1p40L + 2.3L, 0x1p40L + 4.7L), point(maximum, -maximum, minimum),
        point(tiny, -tiny, minimum), point(std::nextafter(1.L, 2.L), -1, 0)};
    std::size_t supported_modes = 0;
    for (const auto rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        if (std::fesetround(rounding) != 0)
            continue; // Only supported rounding modes form part of this oracle.
        ++supported_modes;
        for (const auto initial_flags : {0, FE_DIVBYZERO | FE_INEXACT})
        {
            for (const auto& value : points)
            {
                const auto body = P::canonical_key(value);
                using Header = P::GeometryPacketHeader;
                // Independent original v3 envelope oracle: zero all natural padding,
                // then set frozen metadata and little-endian lengths explicitly.
                Bytes expected(8 + sizeof(Header) + body.size(), 0);
                const auto word = [&](std::size_t offset, std::uint64_t n)
                {
                    for (unsigned i = 0; i < 8; ++i)
                        expected[offset + i] = static_cast<char>(n >> (8 * i));
                };
                word(0, sizeof(Header) + body.size());
                expected[8 + offsetof(Header, family)] = static_cast<char>(P::GeometryPacketFamily::Geometry);
                expected[8 + offsetof(Header, version)] = 3;
                expected[8 + offsetof(Header, kind)] = static_cast<char>(P::GeometryPacketKind::PointIdentity);
                expected[8 + offsetof(Header, mode)] = static_cast<char>(SimpleFluid::ArithmeticMode::Ordinary);
                using Profile = P::GeometryNumericProfile;
                constexpr auto profile = std::numeric_limits<long double>::digits == 53   ? Profile::Binary64
                                         : std::numeric_limits<long double>::digits == 64 ? Profile::X87Binary80
                                                                                          : Profile::Binary128;
                expected[8 + offsetof(Header, profile)] = static_cast<char>(profile);
                word(8 + offsetof(Header, body_length), body.size());
                std::copy(body.begin(), body.end(), expected.begin() + 8 + sizeof(Header));
                D::Writer legacy, actual;
                actual.data.reserve(expected.size());
                const auto capacity = actual.data.capacity();
                const auto* storage = actual.data.data();
                const auto reset_environment = [&]
                {
                    std::feclearexcept(FE_ALL_EXCEPT);
                    std::feraiseexcept(initial_flags);
#if defined(__GLIBC__)
                    feenableexcept(FE_INVALID);
#endif
                };
                reset_environment();
                legacy.bytes(P::encode_identity(value));
                const auto legacy_flags = std::fetestexcept(FE_ALL_EXCEPT);
                const auto legacy_rounding = std::fegetround();
#if defined(__GLIBC__)
                const auto legacy_traps = fegetexcept();
#endif
                reset_environment();
                std::size_t allocations = 0;
                struct CountScope
                {
                    explicit CountScope(std::size_t& count) { allocation_count = &count; }
                    ~CountScope() { allocation_count = nullptr; }
                };
                {
                    CountScope monitor(allocations);
                    D::append_framed_contact_point_identity(actual, value);
                }
                const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
                const auto mode = std::fegetround();
#if defined(__GLIBC__)
                const auto traps = fegetexcept();
#endif
                EXPECT_EQ(allocations, 0u);
                EXPECT_EQ(actual.data.data(), storage);
                EXPECT_EQ(actual.data.capacity(), capacity);
                EXPECT_EQ(actual.data, expected);
                EXPECT_EQ(actual.data, legacy.data);
                EXPECT_EQ(flags, legacy_flags);
                EXPECT_EQ(mode, legacy_rounding);
#if defined(__GLIBC__)
                EXPECT_EQ(traps, legacy_traps);
#endif
            }
        }
    }
    EXPECT_GT(supported_modes, 0u);
    RecordProperty("identity_packet_supported_rounding_modes", std::to_string(supported_modes));
}
TEST(ContactPointGroups, IdentityPacketAppendRejectsOneByteShortBeforeMutationAndReusesExactCapacity)
{
    const auto value = point(1.1L, -2.3L, 4.7L);
    D::Writer legacy;
    legacy.bytes(P::encode_identity(value));
    const auto required = legacy.data.size();
    D::Writer actual;
    actual.data.reserve(required + 7);
    const auto capacity = actual.data.capacity();
    // Use actual capacity, including any implementation-specific reserve growth.
    actual.data.resize(capacity - required + 1, char(0x83));
    const auto before = actual.data;
    const auto* storage = actual.data.data();
    EXPECT_THROW(D::append_framed_contact_point_identity(actual, value), std::invalid_argument);
    EXPECT_EQ(actual.data, before);
    EXPECT_EQ(actual.data.data(), storage);
    EXPECT_EQ(actual.data.capacity(), capacity);
    actual.data.pop_back();
    const auto prefix = actual.data.size();
    ASSERT_EQ(actual.data.capacity() - actual.data.size(), required);
    std::size_t allocations = 0;
    allocation_count = &allocations;
    try
    {
        D::append_framed_contact_point_identity(actual, value);
    }
    catch (...)
    {
        allocation_count = nullptr;
        throw;
    }
    allocation_count = nullptr;
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(actual.data.data(), storage);
    EXPECT_EQ(actual.data.capacity(), capacity);
    EXPECT_EQ(actual.data.size(), capacity);
    EXPECT_TRUE(std::equal(actual.data.begin(), actual.data.begin() + prefix, before.begin()));
    EXPECT_TRUE(std::equal(actual.data.begin() + prefix, actual.data.end(), legacy.data.begin(), legacy.data.end()));
}
TEST(ContactPointGroups, PartialBlockFailureLeavesSurvivingCallerLedgerUnchanged)
{
    const std::array local{D::ContactLocalOccurrence{{0, 1}, 0}};
    const std::array remote{D::ContactRemoteOccurrence{0, {0, 1}}};
    const std::array input{D::ContactFanBlock::Input{local, remote}};
    std::size_t body_bytes;
    {
        auto arena = std::make_shared<D::Budget>(D::Budget{ample});
        D::ContactGroupAllocation ledger(arena);
        const auto block = D::ContactFanBlock::build(input, ledger);
        body_bytes = ledger.bytes();
    }
    auto arena = std::make_shared<D::Budget>(D::Budget{37 + body_bytes - 1});
    D::ContactGroupAllocation ledger(arena);
    ledger.add(37);
    EXPECT_THROW((void) D::ContactFanBlock::build(input, ledger), std::invalid_argument);
    EXPECT_EQ(ledger.bytes(), 37u);
    EXPECT_EQ(arena->used, 37u);
    arena->limit = 37 + body_bytes;
    auto block = D::ContactFanBlock::build(input, ledger);
    EXPECT_EQ(ledger.bytes(), 37 + body_bytes);
    block.reset();
    ledger.release(body_bytes);
    EXPECT_EQ(arena->used, 37u);
}
struct ConstantOwnerHash
{
    std::uint64_t operator()(const void*) const noexcept { return 3; }
};
TEST(ContactPointGroups, PhysicalOwnerCollisionsRetainFullOwnerIdentity)
{
    D::Budget arena{ample};
    std::array<int, 64> owners{};
    {
        D::PhysicalOwnerIndex<std::size_t, ConstantOwnerHash> index(arena);
        for (std::size_t i = 0; i < owners.size(); ++i)
            index.insert(&owners[i], i + 1);
        for (std::size_t i = 0; i < owners.size(); ++i)
            ASSERT_EQ(*index.find(&owners[i]), i + 1);
        for (std::size_t i = 0; i < owners.size(); i += 2)
            EXPECT_TRUE(index.erase(&owners[i]));
        for (std::size_t i = 0; i < owners.size(); i += 2)
            index.insert(&owners[i], i + 100);
        for (std::size_t i = 0; i < owners.size(); ++i)
            EXPECT_EQ(*index.find(&owners[i]), i % 2 ? i + 1 : i + 100);
    }
    EXPECT_EQ(arena.used, 0u);
}
TEST(ContactPointGroups, LargeBatchUsesSortedDuplicateChecksAndIndexedFanOwners)
{
    constexpr std::size_t count = 4096, batch_size = 16;
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto target = std::make_shared<D::Budget>(D::Budget{ample});
    D::ContactPointGroups<> groups(arena);
    std::vector<D::ContactGroupId> ids;
    std::vector<D::ContactRemoteOccurrence> remote;
    std::vector<D::ContactPointGroups<>::Edit> edits;
    ids.reserve(count);
    remote.reserve(count);
    edits.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        ids.push_back(groups.intern({}, point(i)));
    for (std::size_t i = 0; i < count; ++i)
        remote.push_back({i, groups.catalog()->points.retain(point(i + count + 1024))});
    for (std::size_t i = count; i-- > 0;)
        edits.push_back({ids[i], {}, {}, std::span<const D::ContactRemoteOccurrence>(&remote[i], 1)});
    groups.replace_fans(edits);
    const auto work = groups.last_duplicate_check_work();
    EXPECT_LT(work, 64 * count);
    const auto* initial_owner = groups.fan(ids[0]).owner().get();
    EXPECT_EQ(initial_owner->block().groups(), count);
    for (const auto id : ids)
        EXPECT_EQ(groups.fan(id).owner().get(), initial_owner);
    for (const auto occurrence : remote)
        groups.catalog()->points.release(occurrence.physical);
    for (std::size_t first = 0; first < count; first += batch_size)
        groups.replace_fans(std::span<const D::ContactPointGroups<>::Edit>(edits).subspan(first, batch_size));
    groups.reclaim();
    D::ContactPointGroups<> fork(groups, target);
    EXPECT_LT(fork.fan_owner_lookup_work(), 64 * count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto fan = fork.fan(ids[i]);
        EXPECT_EQ(fan.owner()->block().groups(), batch_size);
        EXPECT_EQ(fan.remote(0).record, i);
        EXPECT_EQ(fan.remote(0).physical, point(i + count + 1024));
    }
    const auto before = groups.fan(ids[0]).owner().get();
    const std::array duplicate{edits[0], edits[0]};
    EXPECT_THROW(groups.replace_fans(duplicate), std::invalid_argument);
    EXPECT_EQ(groups.fan(ids[0]).owner().get(), before);
    RecordProperty("large_edit_duplicate_comparisons", std::to_string(work));
    RecordProperty("large_fork_owner_probes", std::to_string(fork.fan_owner_lookup_work()));
}
TEST(ContactPointGroups, TargetPoolPageCreditLookupHasLinearWorkForLargeBatch)
{
    constexpr std::size_t pages = 1024;
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto target_arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::PointFactDirectory::PointPool source(*source_arena, source_arena);
    for (std::size_t i = 0; i < pages * 16; ++i)
        (void) source.intern(point(i));
    auto view = source.page_view();
    std::vector<D::CapturedPointReference> refs;
    for (std::size_t i = 0; i < pages; ++i)
        refs.push_back(view->capture(source, 16 * i));
    view->freeze();
    D::PointFactDirectory::PointPool target(source, *target_arena, target_arena);
    const auto before = target.physical_pool().page_owner_lookup_probes();
    D::PointFactDirectory directory(target_arena);
    for (std::size_t i = 0; i < pages; ++i)
    {
        const auto ref = directory.retain({view.get(), refs[i]}, &target);
        EXPECT_EQ(directory.get(ref), point(16 * i));
    }
    const auto work = target.physical_pool().page_owner_lookup_probes() - before;
    EXPECT_LT(work, 64 * pages);
    EXPECT_EQ(directory.live_pages(), pages);
    EXPECT_EQ(directory.fallback_pages(), 0u);
    RecordProperty("large_target_page_credit_probes", std::to_string(work));
}
struct CensusHash
{
    inline static std::size_t calls = 0;
    std::uint64_t operator()(D::ContactPointKeyView value) const
    {
        ++calls;
        return D::ContactPointWireHasher{}(value);
    }
    std::uint64_t operator()(std::span<const char> value) const
    {
        ++calls;
        return D::ContactPointWireHasher{}(value);
    }
};
TEST(ContactPointGroups, ActualPayloadCensusMatchesIndependentHeapRequestsAndPreservesState)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    const Bytes domain{char(0x80)};
    auto row_owner = row(10, 101), retired_row = row(12, 102);
    row_owner->domain = domain;
    retired_row->domain = domain;
    HeapAudit audit;
    heap_audit = &audit;
    D::ContactPointGroups<CensusHash> groups(arena);
    const auto id = groups.intern(domain, point(0));
    const auto retired_group = groups.intern(domain, point(2));
    auto catalog = groups.catalog();
    const auto binding = catalog->bind(row_owner, 10), retired_binding = catalog->bind(retired_row, 12);
    const auto physical = catalog->points.retain(point(99));
    const std::array local{D::ContactLocalOccurrence{binding, 0}};
    const std::array remote{D::ContactRemoteOccurrence{11, physical}};
    const std::array edit{D::ContactPointGroups<CensusHash>::Edit{id, {}, local, remote}};
    groups.replace_fans(edit);
    catalog->release(binding);
    catalog->release(retired_binding);
    catalog->points.release(physical);
    groups.retire(retired_group);
    groups.reclaim();
    const auto hash_before = CensusHash::calls, work_before = groups.fan_owner_lookup_work();
    const auto used_before = arena->used, cumulative_before = arena->cumulative;
    const auto row_refs = row_owner.use_count();
    struct Restore
    {
        std::fenv_t environment;
        Restore() { std::fegetenv(&environment); }
        ~Restore() { std::fesetenv(&environment); }
    } restore;
    std::fesetround(FE_DOWNWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
#if defined(__GLIBC__)
    const auto traps_before = fegetexcept();
    feenableexcept(FE_INVALID);
#endif
    const auto flags_before = std::fetestexcept(FE_ALL_EXCEPT);
    std::array<D::ContactPayloadAllocation, 128> payload{};
    std::size_t count = 0, allocations = 0, facts = 0, bindings = 0, group_count = 0, flat_records = 0;
    allocation_count = &allocations;
    groups.visit_borrowed_payload([&](const D::ContactPayloadAllocation& value) { payload.at(count++) = value; });
    groups.visit_borrowed_groups(
        [&](const auto& group)
        {
            ++group_count;
            group.fan->owner()->block().visit_borrowed_arrays(
                [&](auto local, std::size_t, auto remote, std::size_t, auto slices, std::size_t)
                { flat_records += local.size() + remote.size() + slices.size(); });
        });
    groups.borrowed_catalog().visit_borrowed_bindings([&](const auto&) { ++bindings; });
    groups.borrowed_catalog().points.visit_borrowed_facts([&](const auto&) { ++facts; });
    allocation_count = nullptr;
    heap_audit = nullptr;
    EXPECT_FALSE(audit.overflow);
    EXPECT_EQ(allocations, 0u);
    std::size_t payload_bytes = 0, live_requested = 0, controls_excluded = 0, nodes_excluded = 0;
    std::set<D::ContactPayloadRole> roles;
    for (std::size_t i = 0; i < audit.count; ++i)
        if (audit.allocations[i].live)
            live_requested += audit.allocations[i].bytes;
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto& value = payload[i];
        roles.insert(value.identity.role);
        payload_bytes += value.payload_bytes;
        controls_excluded += value.control_overhead_excluded;
        nodes_excluded += value.node_overhead_excluded;
        EXPECT_TRUE(value.allocator_overhead_excluded);
        EXPECT_LE(value.live_count, value.capacity);
        EXPECT_EQ(value.payload_bytes, value.element_size * value.capacity);
        const auto address = reinterpret_cast<std::uintptr_t>(value.identity.address);
        const HeapAllocation* containing = nullptr;
        for (std::size_t j = 0; j < audit.count; ++j)
        {
            const auto& allocation = audit.allocations[j];
            const auto start = reinterpret_cast<std::uintptr_t>(allocation.address);
            if (allocation.live && address >= start && address + value.payload_bytes <= start + allocation.bytes)
                containing = &allocation;
        }
        ASSERT_NE(containing, nullptr);
        if (value.kind == D::ContactPayloadKind::Buffer)
        {
            EXPECT_EQ(value.identity.address, containing->address);
            EXPECT_EQ(value.payload_bytes, containing->bytes);
        }
        for (std::size_t j = 0; j < i; ++j)
            EXPECT_FALSE(value.identity == payload[j].identity);
    }
    const std::array<D::ContactPayloadRole, 24> expected_roles{D::ParallelSupportPayloadRole::GroupSlots,
        D::ParallelSupportPayloadRole::GroupFreeSlots, D::ParallelSupportPayloadRole::GroupBuckets,
        D::ParallelSupportPayloadRole::FanOwnerIndexBuckets, D::BasicContactPayloadRole::IdentityBySlot,
        D::BasicContactPayloadRole::IdentityPageEntries, D::BasicContactPayloadRole::IdentityHashBuckets,
        D::BasicContactPayloadRole::IdentityPageOwnerBuckets, D::BasicContactPayloadRole::IdentityPageValues,
        D::BasicContactPayloadRole::IdentityDynamicBytes, D::BasicContactPayloadRole::FactPages,
        D::BasicContactPayloadRole::FactEntries, D::BasicContactPayloadRole::FactFreeEntries,
        D::BasicContactPayloadRole::FactFreePages, D::BasicContactPayloadRole::FactBuckets,
        D::BasicContactPayloadRole::FactPageIndexValues, D::BasicContactPayloadRole::FallbackValues,
        D::ParallelSupportPayloadRole::BindingEntries, D::ParallelSupportPayloadRole::BindingFreeSlots,
        D::ParallelSupportPayloadRole::BindingBuckets, D::ParallelSupportPayloadRole::FanViews,
        D::ParallelSupportPayloadRole::FanLocal, D::ParallelSupportPayloadRole::FanRemote,
        D::ParallelSupportPayloadRole::FanSlices};
    for (const auto role : expected_roles)
        EXPECT_TRUE(roles.contains(role));
    EXPECT_FALSE(roles.contains(D::ParallelSupportPayloadRole::GroupObject));
    EXPECT_FALSE(roles.contains(D::BasicContactPayloadRole::FactDirectoryObject));
    EXPECT_FALSE(roles.contains(D::BasicContactPayloadRole::IdentityPoolObject));
    EXPECT_FALSE(roles.contains(D::ParallelSupportPayloadRole::PhysicalOwnerIndexObject));
    EXPECT_GT(controls_excluded, 0u);
    EXPECT_GT(nodes_excluded, 0u);
    EXPECT_LT(payload_bytes, live_requested);
    const auto admission = groups.storage_breakdown();
    EXPECT_GT(admission.private_metadata_bytes() + admission.shared_body_bytes(), payload_bytes);
    RecordProperty("known_payload_bytes", std::to_string(payload_bytes));
    RecordProperty("independent_live_operator_new_requested_bytes", std::to_string(live_requested));
    RecordProperty("admission_minus_known_payload_bytes",
        std::to_string(admission.private_metadata_bytes() + admission.shared_body_bytes() - payload_bytes));
    EXPECT_EQ(group_count, 1u);
    EXPECT_EQ(bindings, 1u);
    EXPECT_EQ(facts, 2u);
    EXPECT_EQ(flat_records, 3u);
    EXPECT_EQ(CensusHash::calls, hash_before);
    EXPECT_EQ(groups.fan_owner_lookup_work(), work_before);
    EXPECT_EQ(arena->used, used_before);
    EXPECT_EQ(arena->cumulative, cumulative_before);
    EXPECT_EQ(row_owner.use_count(), row_refs);
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags_before);
#if defined(__GLIBC__)
    EXPECT_EQ(fegetexcept(), traps_before | FE_INVALID);
#endif
}
TEST(ContactPointGroups, ExplicitCreditIdentityUnionAvoidsTargetPoolBodyAndMetadataDoubleCharge)
{
    auto first = std::make_shared<D::Budget>(D::Budget{ample});
    auto second = std::make_shared<D::Budget>(D::Budget{ample});
    D::PointFactDirectory::PointPool source(*first, first);
    for (unsigned i = 0; i < 32; ++i)
        (void) source.intern(point(i));
    auto view = source.page_view();
    const auto captured = view->capture(source, 0);
    view->freeze();
    D::PointFactDirectory::PointPool target(source, *second, second);
    D::PointFactDirectory directory(second);
    (void) directory.retain({view.get(), captured}, &target);
    std::map<const void*, std::size_t> metadata, bodies;
    const auto add = [&](const void* owner, std::size_t bytes, auto& entries)
    {
        const auto [found, inserted] = entries.emplace(owner, bytes);
        EXPECT_TRUE(inserted || found->second == bytes);
    };
    add(target.metadata_identity(), target.metadata_bytes(), metadata);
    target.visit_page_admissions(
        [&](const void* credit, const void* body, std::size_t owner, std::size_t bytes)
        {
            add(credit, owner, metadata);
            add(body, bytes, bodies);
        });
    directory.visit_admissions(
        [&](const D::ContactStorageAdmission& admission)
        {
            if (admission.private_metadata)
                add(admission.metadata_owner, admission.private_metadata, metadata);
            if (admission.shared_body)
                add(admission.body_owner, admission.shared_body, bodies);
        });
    std::size_t total = 0;
    for (const auto& [owner, bytes] : metadata)
        total += bytes;
    for (const auto& [owner, bytes] : bodies)
        total += bytes;
    EXPECT_EQ(total, second->used);
    EXPECT_EQ(bodies.size(), 2u);
}

TEST(ContactPointGroups, FullGroupAdmissionUnionDistinguishesCatalogAndEmbeddedFactLedgers)
{
    auto first = std::make_shared<D::Budget>(D::Budget{ample});
    auto second = std::make_shared<D::Budget>(D::Budget{ample});
    D::PointFactDirectory::PointPool source(*first, first);
    for (unsigned i = 0; i < 32; ++i)
        (void) source.intern(point(i));
    auto source_view = source.page_view();
    const auto normalized = source_view->capture(source, 0), physical = source_view->capture(source, 16);
    source_view->freeze();
    D::PointFactDirectory::PointPool target(source, *second, second);
    D::ContactPointGroups<> groups(second);
    const auto id = groups.intern({}, point(0), D::ContactPointArray::Locator{source_view.get(), normalized}, &target);
    auto catalog = groups.catalog();
    const auto binding = catalog->bind(row(0, 101), 10);
    const auto remote_point = catalog->points.retain({source_view.get(), physical}, &target);
    const std::array local{D::ContactLocalOccurrence{binding, 0}};
    const std::array remote{D::ContactRemoteOccurrence{11, remote_point}};
    const std::array edit{D::ContactPointGroups<>::Edit{id, {}, local, remote}};
    groups.replace_fans(edit);
    catalog->release(binding);
    catalog->points.release(remote_point);
    EXPECT_NE(catalog->metadata_identity(), catalog->points.metadata_identity());
    std::map<const void*, std::size_t> metadata, bodies;
    const auto add = [&](const void* owner, std::size_t bytes, auto& entries)
    {
        const auto [found, inserted] = entries.emplace(owner, bytes);
        EXPECT_TRUE(inserted || found->second == bytes);
    };
    add(target.metadata_identity(), target.metadata_bytes(), metadata);
    target.visit_page_admissions(
        [&](const void* credit, const void* body, std::size_t owner, std::size_t bytes)
        {
            add(credit, owner, metadata);
            add(body, bytes, bodies);
        });
    groups.visit_admissions(
        [&](const D::ContactStorageAdmission& admission)
        {
            if (admission.private_metadata)
                add(admission.metadata_owner, admission.private_metadata, metadata);
            if (admission.shared_body)
                add(admission.body_owner, admission.shared_body, bodies);
        });
    std::size_t total = 0;
    for (const auto& [owner, bytes] : metadata)
        total += bytes;
    for (const auto& [owner, bytes] : bodies)
        total += bytes;
    EXPECT_EQ(total, second->used);
    EXPECT_TRUE(metadata.contains(catalog->metadata_identity()));
    EXPECT_TRUE(metadata.contains(catalog->points.metadata_identity()));
    EXPECT_EQ(groups.fan(id).local(0).binding.record, 10u);
    EXPECT_EQ(groups.fan(id).remote(0).physical, point(16));
    EXPECT_EQ(catalog->points.fallback_pages(), 0u);
}
TEST(ContactPointGroups, PointArrayPayloadReportsBothOwningAndReferenceBuffersWithoutEmbeddedObjects)
{
    D::ContactPointArray owning;
    owning.reserve(4);
    owning.push_back({point(0), point(0)});
    owning.push_back({point(1), point(33)});
    std::size_t owning_buffers = 0;
    owning.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& value)
        {
            EXPECT_EQ(value.kind, D::ContactPayloadKind::Buffer);
            EXPECT_TRUE(value.identity.role == D::BasicContactPayloadRole::PointArrayPhysical ||
                        value.identity.role == D::BasicContactPayloadRole::PointArrayNormalized);
            EXPECT_EQ(value.element_size, sizeof(D::ContactPointIdentity));
            EXPECT_EQ(value.live_count, 2u);
            EXPECT_EQ(value.capacity, 4u);
            EXPECT_EQ(value.payload_bytes, 4 * sizeof(D::ContactPointIdentity));
            ++owning_buffers;
        });
    EXPECT_EQ(owning_buffers, 2u);
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::PointFactDirectory::PointPool pool(*arena, arena);
        const auto physical = pool.intern(point(0)), normalized = pool.intern(point(1));
        D::ContactPointBatch batch(pool);
        std::size_t reference_bytes = 0;
        {
            auto array = D::ContactPointArray::pooled(batch, 4, *arena);
            array.push_reference(batch.capture(physical), batch.capture(normalized), *arena);
            std::size_t references = 0, capture_values = 0, allocations = 0;
            allocation_count = &allocations;
            std::array<D::ContactPayloadAllocation, 32> payload{};
            std::size_t count = 0;
            array.visit_borrowed_payload([&](const auto& value) { payload.at(count++) = value; });
            allocation_count = nullptr;
            EXPECT_EQ(allocations, 0u);
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto& value = payload[i];
                if (value.identity.role == D::BasicContactPayloadRole::PointArrayPhysicalRefs ||
                    value.identity.role == D::BasicContactPayloadRole::PointArrayNormalizedRefs)
                {
                    ++references;
                    EXPECT_EQ(value.element_size, sizeof(std::uint32_t));
                    EXPECT_EQ(value.live_count, 1u);
                    EXPECT_EQ(value.capacity, 4u);
                    EXPECT_EQ(value.payload_bytes, 4 * sizeof(std::uint32_t));
                }
                if (value.identity.role == D::BasicContactPayloadRole::PageViewCaptureValues)
                {
                    ++capture_values;
                    EXPECT_TRUE(value.node_overhead_excluded);
                    EXPECT_EQ(value.payload_bytes, sizeof(std::pair<const void* const, std::size_t>));
                }
                EXPECT_NE(value.identity.role, D::BasicContactPayloadRole::PointArrayObject);
            }
            EXPECT_EQ(references, 2u);
            EXPECT_EQ(capture_values, 1u);
            reference_bytes = array.storage_bytes();
            batch.freeze();
        }
        arena->used -= reference_bytes;
    }
    EXPECT_EQ(arena->used, 0u);
}
TEST(ContactPointGroups, GenericPoolPayloadExposesOrderedNodeValuesAndOwnedAncestryPaths)
{
    using Key = SimpleFluid::Meshes::AMR::AdaptiveCellKey;
    using Ancestry = SimpleFluid::Meshes::AMR::ConvexCellAncestry;
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityPool<Key, D::NoIdentityPoolHash> pool(*arena, arena);
    Key key;
    Ancestry ancestry;
    ancestry.root = 7;
    ancestry.path = {1, 2, 3};
    ancestry.path.reserve(9);
    key.ancestry = std::move(ancestry);
    const auto slot = pool.intern(key);
    const auto& path = std::get<Ancestry>(pool.get(slot).ancestry).path;
    std::size_t ordered_values = 0, paths = 0, page_buffers = 0, hashes = 0;
    pool.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& value)
        {
            if (value.identity.role == D::BasicContactPayloadRole::IdentityOrderedValues)
            {
                ++ordered_values;
                EXPECT_TRUE(value.node_overhead_excluded);
                EXPECT_EQ(value.payload_bytes, sizeof(std::pair<const Key* const, std::size_t>));
            }
            if (value.identity.role == D::BasicContactPayloadRole::IdentityCellPath)
            {
                ++paths;
                EXPECT_EQ(value.identity.address, path.data());
                EXPECT_EQ(value.capacity, path.capacity());
                EXPECT_EQ(value.element_size, sizeof(std::uint64_t));
                EXPECT_EQ(value.payload_bytes, path.capacity() * sizeof(std::uint64_t));
            }
            if (value.identity.role == D::BasicContactPayloadRole::IdentityPageValues)
            {
                ++page_buffers;
                EXPECT_EQ(value.element_size, sizeof(Key));
                EXPECT_EQ(value.capacity, 16u);
                EXPECT_EQ(value.payload_bytes, 16 * sizeof(Key));
                EXPECT_FALSE(value.opaque_element_payload_excluded);
            }
            hashes += value.identity.role == D::BasicContactPayloadRole::IdentityHashBuckets;
            EXPECT_NE(value.identity.role, D::BasicContactPayloadRole::IdentityPoolObject);
        });
    EXPECT_EQ(ordered_values, 1u);
    EXPECT_EQ(paths, 1u);
    EXPECT_EQ(page_buffers, 1u);
    EXPECT_EQ(hashes, 0u);
}
} // namespace
