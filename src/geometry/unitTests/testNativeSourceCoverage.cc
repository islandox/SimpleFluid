#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "native_constructor_allocation_operators.inc"
#include <gtest/gtest.h>
#include <algorithm>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace G = SimpleFluid::geometry_detail;
using Record = SimpleFluid::PLIC::details::geometry_demand::OwnerRecord;
constexpr std::size_t ample = 8 * 1024 * 1024;

struct Fixture
{
    std::shared_ptr<D::Budget> arena = std::make_shared<D::Budget>(D::Budget{ample});
    std::shared_ptr<SimpleFluid::Meshes::NativeFragmentDescriptor> descriptor =
        std::make_shared<SimpleFluid::Meshes::NativeFragmentDescriptor>();
    A::AdaptiveLeaf leaf;
    D::AdaptiveLeafBodyUsage body;
    std::vector<Record> owners;
    std::size_t expected = 0;
    Fixture()
    {
        descriptor->view.assign(23, 'v');
        descriptor->parts.push_back(std::make_shared<const A::AdaptiveBytes>(43, 'p'));
        expected = sizeof(*descriptor) + descriptor->view.capacity() +
            descriptor->parts.capacity() * sizeof(descriptor->parts[0]) +
            sizeof(*descriptor->parts[0]) + descriptor->parts[0]->capacity();
        leaf.key = {0, A::ConvexCellAncestry{1, {}}};
        leaf.native_recipe = descriptor;
        D::AdaptiveLeafBodyDelta delta(arena);
        delta.record(std::span(&leaf, 1), true);
        body = delta.apply({}).usage;
        body.owners().owners[0].visit([&](const auto& owner) { owners.push_back(owner); });
    }
};

struct Floor
{
    std::shared_ptr<D::Budget> arena = std::make_shared<D::Budget>(D::Budget{ample});
    std::shared_ptr<std::size_t> total = std::make_shared<std::size_t>(0);
    std::shared_ptr<D::ContactExternalResidencyLease> lease =
        std::make_shared<D::ContactExternalResidencyLease>(arena, total);
    explicit Floor(std::size_t bytes) { lease->resize(bytes); }
};
G::NativeConstructionAdmission admission()
{ return G::NativeConstructionAdmission(G::NativeConstructionCounterAccess::create<D::Budget>(ample)); }

TEST(NativeSourceCoverage, NewArenaReusesImmutableMembershipWithoutOwnerRemint)
{
    Fixture source;
    ASSERT_EQ(source.body.payload_bytes(), source.expected);
    Floor first(source.expected + 17), second(source.expected + 17);
    auto native = admission();
    const auto a = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        first.lease, native, 1);
    const auto b = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        second.lease, native, 2);
    EXPECT_TRUE(a->same_membership(*b));
    EXPECT_TRUE(D::NativeSourceOwnerStore::coverage_group(a).same_membership(
        D::NativeSourceOwnerStore::coverage_group(b)));
    EXPECT_NE(a->floor()->arena_identity(), b->floor()->arena_identity());
    for (const auto& row : source.owners)
        EXPECT_TRUE(b->contains(row.key.kind, row.owner, row.bytes));
    std::size_t changes = 0;
    b->visit_delta(a.get(), {&changes, [](void* raw, auto, const auto&, auto, bool)
        { ++*static_cast<std::size_t*>(raw); }});
    EXPECT_EQ(changes, 0);
    auto fake = std::shared_ptr<const void>(source.descriptor.get(), [](const void*) {});
    EXPECT_THROW((void) b->contains(source.owners.front().key.kind, fake, source.owners.front().bytes), std::exception);
}

TEST(NativeSourceCoverage, DetachedSubsetSurvivesSourceFloorAndGroupExpiry)
{
    Fixture source;
    Floor floor(source.expected + 17);
    auto native = admission();
    auto group = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        floor.lease, native, 1);
    std::weak_ptr<const D::NativeSourceCoverage> old_group = group;
    std::weak_ptr<D::ContactExternalResidencyLease> old_floor = floor.lease;
    D::NativeSourceCoverageBroker broker;
    std::shared_ptr<const D::NativeSourceSubsetCoverage> subset;
    const auto selected = source.owners.front();
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        subset = transaction.collect(group, std::span(&selected, 1));
        ASSERT_EQ(subset->bytes(), selected.bytes);
        transaction.prepare();
        EXPECT_EQ(floor.lease->bytes(), source.expected + 17);
        transaction.commit();
        EXPECT_EQ(floor.lease->bytes(), source.expected + 17 - selected.bytes);
        EXPECT_EQ(floor.arena->used, source.expected + 17);
    }
    group.reset();
    floor.lease.reset();
    EXPECT_TRUE(old_group.expired());
    EXPECT_TRUE(old_floor.expired());
    EXPECT_EQ(floor.arena->used, selected.bytes);
    EXPECT_EQ(*floor.total, selected.bytes);
    subset.reset();
    EXPECT_EQ(floor.arena->used, 0);
    EXPECT_EQ(*floor.total, 0);
}

TEST(NativeSourceCoverage, OutputFirstReturnsSubsetAndLateFailurePreservesSource)
{
    Fixture source;
    Floor floor(source.expected + 17);
    auto native = admission();
    auto group = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        floor.lease, native, 1);
    D::NativeSourceCoverageBroker broker;
    const auto selected = source.owners.front();
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        auto subset = transaction.collect(group, std::span(&selected, 1));
        transaction.prepare();
        // Simulate the final collective rejecting another rank.
        EXPECT_EQ(floor.lease->bytes(), source.expected + 17);
    }
    EXPECT_EQ(floor.lease->bytes(), source.expected + 17);
    EXPECT_EQ(floor.arena->used, source.expected + 17);
    {
        std::shared_ptr<const D::NativeSourceSubsetCoverage> subset;
        {
            D::NativeSourceCoverageTransaction transaction(broker, native);
            subset = transaction.collect(group, std::span(&selected, 1));
            transaction.prepare();
            transaction.commit();
        }
        EXPECT_EQ(floor.lease->bytes(), source.expected + 17 - selected.bytes);
    }
    EXPECT_EQ(floor.lease->bytes(), source.expected + 17);
    EXPECT_EQ(floor.arena->used, source.expected + 17);
    EXPECT_NO_THROW(group->require_current());
}

TEST(NativeSourceCoverage, PublishedSubsetCanReturnWhileAnotherSubsetTransferIsPrepared)
{
    for (const bool commit : {false, true})
    {
        SCOPED_TRACE(commit);
        Fixture source;
        ASSERT_EQ(source.owners.size(), 2u);
        const auto& first = source.owners.front();
        const auto& second = source.owners.back();
        const auto original = source.expected;
        Floor floor(original);
        auto native = admission();
        const auto group = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
            floor.lease, native, 1);
        D::NativeSourceCoverageBroker broker;
        std::shared_ptr<const D::NativeSourceSubsetCoverage> first_output, second_output;
        {
            D::NativeSourceCoverageTransaction transaction(broker, native);
            first_output = transaction.collect(group, std::span(&first, 1));
            transaction.prepare();
            transaction.commit();
        }
        ASSERT_EQ(first_output->bytes(), first.bytes);
        {
            D::NativeSourceCoverageTransaction transaction(broker, native);
            second_output = transaction.collect(group, std::span(&second, 1));
            ASSERT_EQ(second_output->bytes(), second.bytes);
            transaction.prepare();
            EXPECT_EQ(floor.lease->bytes(), original - first.bytes);
            first_output.reset();
            EXPECT_EQ(floor.lease->bytes(), original);
            EXPECT_EQ(second_output->owners().front()->lease().bytes(), 0u);
            EXPECT_NO_THROW(group->require_current());
            if (commit) transaction.commit();
        }
        EXPECT_EQ(floor.lease->bytes(), original - (commit ? second.bytes : 0));
        EXPECT_EQ(second_output->owners().front()->lease().bytes(), commit ? second.bytes : 0);
        EXPECT_EQ(floor.arena->used, original);
        EXPECT_EQ(*floor.total, original);
        EXPECT_NO_THROW(group->require_current());
        second_output.reset();
        EXPECT_EQ(floor.lease->bytes(), original);
        EXPECT_NO_THROW(group->require_current());
    }
}

TEST(NativeSourceCoverage, EachCollectAllocationFailurePoisonsStagingAndAllowsFreshExactRetry)
{
    Fixture source;
    ASSERT_EQ(source.owners.size(), 2u);
    const auto& first = source.owners.front();
    const auto& second = source.owners.back();
    const auto original = source.expected;
    Floor floor(original);
    auto native = admission();
    const auto group = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        floor.lease, native, 1);
    D::NativeSourceCoverageBroker broker;
    std::shared_ptr<const D::NativeSourceSubsetCoverage> accepted;
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        accepted = transaction.collect(group, std::span(&first, 1));
        transaction.prepare();
        transaction.commit();
    }
    const auto accepted_broker_bytes = broker.storage_bytes();
    const auto accepted_native_bytes = native.statistics().actual_live;
    std::size_t requests = 0;
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        native_constructor_test::Capture capture(*native.state()->used_counter);
        const auto candidate = transaction.collect(group, std::span(&second, 1));
        requests = capture.finish().phase.front().allocations;
        ASSERT_EQ(candidate->bytes(), second.bytes);
        ASSERT_FALSE(pipeline_trace::overflow.load());
    }
    ASSERT_GT(requests, 0u);
    for (std::size_t request = 0; request < requests; ++request)
    {
        SCOPED_TRACE(request);
        {
            D::NativeSourceCoverageTransaction transaction(broker, native);
            bool rejected = false;
            {
                native_constructor_test::Failure failure(request);
                try
                {
                    (void) transaction.collect(group, std::span(&second, 1));
                }
                catch (const std::bad_alloc&)
                {
                    rejected = true;
                }
            }
            ASSERT_TRUE(rejected);
            EXPECT_THROW((void) transaction.collect(group, std::span(&second, 1)), std::logic_error);
            EXPECT_THROW(transaction.prepare(), std::logic_error);
        }
        EXPECT_EQ(broker.storage_bytes(), accepted_broker_bytes);
        EXPECT_EQ(native.statistics().actual_live, accepted_native_bytes);
        EXPECT_EQ(floor.lease->bytes(), original - first.bytes);
        EXPECT_EQ(floor.arena->used, original);
        EXPECT_EQ(*floor.total, original);
        EXPECT_NO_THROW(group->require_current());
    }
    std::shared_ptr<const D::NativeSourceSubsetCoverage> committed;
    {
        D::NativeSourceCoverageTransaction retry(broker, native);
        const auto reused = retry.collect(group, std::span(&first, 1));
        EXPECT_EQ(reused->owners().front(), accepted->owners().front());
        committed = retry.collect(group, std::span(&second, 1));
        retry.prepare();
        retry.commit();
    }
    EXPECT_EQ(floor.lease->bytes(), 0u);
    EXPECT_EQ(accepted->owners().front()->lease().bytes(), first.bytes);
    EXPECT_EQ(committed->owners().front()->lease().bytes(), second.bytes);
    EXPECT_EQ(floor.arena->used, original);
    EXPECT_EQ(*floor.total, original);
    EXPECT_NO_THROW(group->require_current());
    committed.reset();
    accepted.reset();
    EXPECT_EQ(floor.lease->bytes(), original);
    EXPECT_NO_THROW(group->require_current());
}

TEST(NativeSourceCoverage, ShortFloorIssuanceCanRetryAndCannotChangeEqualSizedMemberControls)
{
    for (const bool manifest : {false, true})
    {
        SCOPED_TRACE(manifest);
        Fixture original, foreign;
        ASSERT_EQ(original.expected, foreign.expected);
        ASSERT_NE(original.descriptor, foreign.descriptor);
        Floor floor(original.expected - 1);
        auto native = admission();
        const auto issue = [&](const Fixture& source)
        {
            if (manifest)
                return D::NativeSourceOwnerStore::manifest_coverage(source.body.owners(), floor.lease, native, 1);
            return D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
                floor.lease, native, 1);
        };
        const auto initial_native_bytes = native.statistics().actual_live;
        EXPECT_THROW((void) issue(original), std::invalid_argument);
        EXPECT_EQ(native.statistics().actual_live, initial_native_bytes);
        EXPECT_EQ(floor.lease->bytes(), original.expected - 1);
        EXPECT_EQ(floor.arena->used, original.expected - 1);
        EXPECT_EQ(*floor.total, original.expected - 1);
        floor.lease->resize(original.expected);
        const auto accepted = issue(original);
        ASSERT_EQ(accepted->covered_bytes(), original.expected);
        const auto accepted_native_bytes = native.statistics().actual_live;
        EXPECT_THROW((void) issue(foreign), std::invalid_argument);
        EXPECT_EQ(native.statistics().actual_live, accepted_native_bytes);
        EXPECT_EQ(floor.lease->bytes(), original.expected);
        EXPECT_EQ(floor.arena->used, original.expected);
        EXPECT_EQ(*floor.total, original.expected);
        EXPECT_NO_THROW(accepted->require_current());
        const auto repeated = issue(original);
        EXPECT_TRUE(repeated->same_membership(*accepted));
    }
}

TEST(NativeSourceCoverage, SameBodyMembershipCanReissueAfterCoverageExpiryWithALiveSubset)
{
    Fixture original, foreign;
    ASSERT_EQ(original.expected, foreign.expected);
    Floor floor(original.expected);
    auto native = admission();
    const auto issue = [&](const Fixture& source)
    {
        return D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
            floor.lease, native, 1);
    };
    auto group = issue(original);
    std::weak_ptr<const D::NativeSourceCoverage> old_group = group;
    D::NativeSourceCoverageBroker broker;
    const auto& selected = original.owners.front();
    std::shared_ptr<const D::NativeSourceSubsetCoverage> output, repeat;
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        output = transaction.collect(group, std::span(&selected, 1));
        transaction.prepare();
        transaction.commit();
    }
    group.reset();
    EXPECT_TRUE(old_group.expired());
    old_group.reset();
    EXPECT_EQ(floor.lease->bytes(), original.expected - selected.bytes);
    EXPECT_THROW((void) issue(foreign), std::invalid_argument);
    group = issue(original);
    {
        D::NativeSourceCoverageTransaction transaction(broker, native);
        repeat = transaction.collect(group, std::span(&selected, 1));
        EXPECT_EQ(repeat->owners().front(), output->owners().front());
        transaction.prepare();
        transaction.commit();
    }
    output.reset();
    EXPECT_EQ(floor.lease->bytes(), original.expected - selected.bytes);
    repeat.reset();
    EXPECT_EQ(floor.lease->bytes(), original.expected);
    EXPECT_EQ(floor.arena->used, original.expected);
    EXPECT_EQ(*floor.total, original.expected);
    EXPECT_NO_THROW(group->require_current());
}

TEST(NativeSourceCoverage, EmptyOptionalScopePreservesTheRemainingScopeMembership)
{
    Fixture initial, removed, retained;
    auto native = admission();
    const auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    const auto total = std::make_shared<std::size_t>(0);
    const auto issuer = [&](const Fixture& body)
    {
        auto floor = std::make_shared<D::ContactExternalResidencyLease>(arena, total);
        floor->resize(body.expected);
        return D::NativeSourceOwnerStore::body_coverage(body.body, D::AdaptiveLeafBodyRole::Leaf, floor, native, 1);
    };
    const std::array old_parts{issuer(initial), issuer(removed), issuer(retained)};
    const std::array new_parts{old_parts[0], decltype(old_parts)::value_type{}, old_parts[2]};
    const auto previous = D::NativeSourceOwnerStore::composite_coverage({}, old_parts, native, 1);
    const auto next = D::NativeSourceOwnerStore::composite_coverage(previous, new_parts, native, 2);
    EXPECT_EQ(next->covered_bytes(), initial.expected + retained.expected);
    std::vector<const void*> retired;
    next->visit_delta(previous.get(), {&retired, [](void* raw, auto, const auto& owner, auto, bool retain)
        {
            EXPECT_FALSE(retain);
            static_cast<std::vector<const void*>*>(raw)->push_back(owner.get());
        }});
    ASSERT_EQ(retired.size(), removed.owners.size());
    for (const auto& owner : removed.owners)
        EXPECT_TRUE(std::ranges::find(retired, owner.owner.get()) != retired.end());
    for (const auto& owner : retained.owners)
        EXPECT_TRUE(next->contains(owner.key.kind, owner.owner, owner.bytes));
}

TEST(NativeSourceCoverage, DefaultPreviousGroupAndChangedPhysicalMembershipUseExactControls)
{
    Fixture source;
    Floor floor(source.expected + 17);
    auto native = admission();
    const auto first = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        floor.lease, native, 1);
    const auto first_group = D::NativeSourceOwnerStore::coverage_group(first);
    std::size_t initial = 0;
    first_group.visit_delta({}, {&initial, [](void* raw, auto, const auto&, auto, bool retain)
        { EXPECT_TRUE(retain); ++*static_cast<std::size_t*>(raw); }});
    EXPECT_EQ(initial, source.owners.size());

    auto changed_leaf = source.leaf;
    auto changed_descriptor = std::make_shared<SimpleFluid::Meshes::NativeFragmentDescriptor>(*source.descriptor);
    changed_leaf.native_recipe = changed_descriptor;
    D::AdaptiveLeafBodyUsage changed_body;
    {
        D::AdaptiveLeafBodyDelta delta(source.arena);
        delta.record(std::span(&source.leaf, 1), false);
        delta.record(std::span(&changed_leaf, 1), true);
        changed_body = delta.apply(source.body, D::AdaptiveLeafBodyRole::Leaf, &source.body).usage;
    }
    Floor changed_floor(changed_body.payload_bytes() + 17);
    const auto second = D::NativeSourceOwnerStore::body_coverage(changed_body, D::AdaptiveLeafBodyRole::Leaf,
        changed_floor.lease, native, 2);
    EXPECT_FALSE(second->same_membership(*first));
    struct Changes { const void* added = nullptr; const void* retired = nullptr; std::size_t count = 0; } changes;
    second->visit_delta(first.get(), {&changes, [](void* raw, auto, const auto& owner, auto, bool retain)
        {
            auto& value = *static_cast<Changes*>(raw);
            (retain ? value.added : value.retired) = owner.get();
            ++value.count;
        }});
    // Descriptor copies share the part control. Only the actual descriptor
    // allocation changes; equal bytes never establish the original control.
    EXPECT_EQ(changes.count, 2);
    EXPECT_EQ(changes.added, changed_descriptor.get());
    EXPECT_EQ(changes.retired, source.descriptor.get());
    EXPECT_THROW((void) second->contains(source.owners.front().key.kind,
        std::shared_ptr<const void>(changed_descriptor.get(), [](const void*) {}),
        source.owners.front().bytes), std::exception);
}

TEST(NativeSourceCoverage, ClosedSourceRootAndUnionCreditCanonicalPhysicalMembersOnce)
{
    Fixture source;
    Floor floor(source.expected);
    auto native = admission();
    const auto coverage = D::NativeSourceOwnerStore::body_coverage(source.body, D::AdaptiveLeafBodyRole::Leaf,
        floor.lease, native, 1);
    std::array<SimpleFluid::PLIC::details::OwnerManifestPair, 4> components;
    components[0] = source.body.owners();
    const auto store = D::NativeSourceOwnerStore::assemble({}, std::move(components), native, nullptr, {},
        {source.body.administration(), {}}, coverage);
    EXPECT_NO_THROW(store.require_current());
    for (const auto& row : source.owners)
        EXPECT_TRUE(store.owners().owners[0].admits(0, row.key.kind, row.owner, row.bytes));

    SimpleFluid::PLIC::GeometryAllocationLedger ordinary, bounded;
    std::size_t cursor = 0;
    SimpleFluid::PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
        ordinary, native, cursor, ample);
    SimpleFluid::PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
        bounded, native, cursor, ample);
    const auto owners = store.replace_in({}, {}, ordinary, bounded, cursor, ample);
    EXPECT_EQ(owners.native_group_covered_bytes(floor.arena), source.expected);
    EXPECT_EQ(owners.native_paid_bytes(floor.arena), source.expected);
    EXPECT_GE(owners.resident_bytes(), source.expected);
    const auto foreign = std::shared_ptr<const void>(floor.arena.get(), [](const void*) {});
    EXPECT_THROW((void) owners.native_paid_bytes(foreign), std::exception);
}
} // namespace
