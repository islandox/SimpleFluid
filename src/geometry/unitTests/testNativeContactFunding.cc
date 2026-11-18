/** Real concrete contact funding, with no test-installed financial authority. */
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "geometry/mesh/AMR/ContactPhysicalResidency.hh"
#include "geometry/mesh/AMR/IdentityPool.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "pipeline_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
namespace
{
using namespace SimpleFluid;
namespace D = Meshes::AMR::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const D::SharedSubfaceRing> ring(const std::shared_ptr<D::Budget>& arena)
{
    constexpr std::size_t count = 11;
    arena->add(count * sizeof(std::size_t));
    std::size_t former = count * sizeof(std::size_t);
    std::vector<std::size_t> values;
    try
    {
        values.reserve(count);
    }
    catch (...)
    {
        arena->used -= former;
        throw;
    }
    if (values.capacity() != count)
    {
        arena->used -= former;
        throw std::invalid_argument("Unsupported exact ring vector capacity.");
    }
    for (std::size_t i = 0; i < count; ++i)
        values.push_back(i);
    auto result = D::SharedSubfaceRing::transfer(values, *arena, arena, former);
    if (former || !values.empty())
        throw std::invalid_argument("Ring did not transfer its actual precharged buffer.");
    return result;
}
TEST(NativeContactFunding, ActualRingControlTailStaysFundedUntilLastWeakControlDies)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{1024 * 1024});
    pipeline_trace::begin();
    auto owner = ring(arena);
    std::weak_ptr<const D::SharedSubfaceRing> weak = owner;
    const auto full = arena->used;
    owner.reset();
    const auto tail = pipeline_trace::summary();
    {
        const pipeline_trace::Pause pause;
        EXPECT_TRUE(weak.expired());
        EXPECT_GT(tail.all_live, 0u);
        EXPECT_GE(arena->used, tail.all_live);
        EXPECT_LT(arena->used, full);
    }
    weak.reset(); // Release diagnostic weak control before asking for raw zero.
    pipeline_trace::recording.store(false);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
    EXPECT_EQ(arena->used, 0u);
}
TEST(NativeContactFunding, AColdForeignControlCannotIssueTheOriginalRingsIntrinsicReceipt)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    const auto owner = ring(arena);
    auto foreign = std::shared_ptr<const D::SharedSubfaceRing>(owner.get(), [](const auto*) {});
    ASSERT_TRUE(owner.owner_before(foreign) || foreign.owner_before(owner));
    D::ContactPhysicalResidency residency(arena);
    const auto before = arena->used;
    EXPECT_THROW(residency.retain(
                     D::ContactPhysicalResidency::Role::SubfaceRing, foreign, owner->charged_bytes(), owner->origin()),
        std::invalid_argument);
    EXPECT_EQ(arena->used, before);
    EXPECT_EQ(residency.source_owners().owner_count(), 0u);
    EXPECT_TRUE(residency.source_paid_receipts().empty());
    EXPECT_NO_THROW(residency.retain(
        D::ContactPhysicalResidency::Role::SubfaceRing, owner, owner->charged_bytes(), owner->origin()));
    EXPECT_FALSE(residency.source_paid_receipts().empty());
    EXPECT_NO_THROW(residency.release(D::ContactPhysicalResidency::Role::SubfaceRing, owner));
    EXPECT_TRUE(residency.source_paid_receipts().empty());
}
TEST(NativeContactFunding, ReceiptOnlyRetentionKeepsTheRealRingControlWithoutItsSlots)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    pipeline_trace::begin();
    auto owner = ring(arena);
    const auto body = sizeof(D::SharedSubfaceRing) + owner->capacity() * sizeof(std::size_t);
    const auto complete = pipeline_trace::summary().all_live;
    std::weak_ptr<const D::SharedSubfaceRing> weak = owner;
    PLIC::details::NativeOwnerReceipts receipts;
    {
        const pipeline_trace::Pause pause;
        D::ContactPhysicalResidency residency(arena);
        residency.retain(D::ContactPhysicalResidency::Role::SubfaceRing, owner, owner->charged_bytes(), owner->origin());
        receipts = residency.source_paid_receipts();
    }
    owner.reset();
    const auto tail = pipeline_trace::summary().all_live;
    weak.reset();
    const auto receipt_tail = pipeline_trace::summary().all_live;
    const auto control_allowance = std::size_t{128};
    receipts = {};
    pipeline_trace::recording.store(false);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(complete - tail, body);
    EXPECT_GT(tail, 0u);
    EXPECT_LE(tail, control_allowance);
    EXPECT_EQ(receipt_tail, tail);
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
    EXPECT_EQ(arena->used, 0u);
}
TEST(NativeContactFunding, PlanePoolDetachesOnlyTheFirstLegacyCanonicalBacking)
{
    using Plane = PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>;
    using Access = PLIC::details::GeometryIdentityAccess;
    auto arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    const geometry_detail::NativeConstructionAdmission admission(
        geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(
            arena->limit, 0, D::NativeSourceOwnerStore::mirror(arena)));
    D::IdentityPool<Plane> planes(*arena, arena, {}, {}, D::IdentityPoolStorage::Persistent);
    planes.bind_native_plane_admission(admission.state());
    const auto legacy = Access::support_axis(1, 1.25L);
    ASSERT_FALSE(PLIC::details::NativeIdentityOwnerAccess::admitted(Access::backing(legacy)));
    const auto slot = planes.intern(legacy);
    const auto& captured = planes.get(slot);
    EXPECT_EQ(captured, legacy);
    EXPECT_NE(captured.storage_identity(), legacy.storage_identity());
    const auto* native = PLIC::details::NativeIdentityOwnerAccess::backing(Access::backing(captured));
    ASSERT_TRUE(native);
    EXPECT_EQ(native->envelope()->state().get(), admission.state().get());
    const auto before = arena->cumulative;
    EXPECT_EQ(planes.intern(legacy), slot);
    EXPECT_EQ(arena->cumulative, before);
    auto view = planes.page_view();
    const auto reference = view->capture(planes, slot);
    view->freeze();
    EXPECT_EQ(view->get(reference).storage_identity(), captured.storage_identity());
}
TEST(NativeContactFunding, PrivateForkMovesOnlyTheInheritedRingPrecharge)
{
    auto parent_arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    auto candidate_arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    auto parent_total = std::make_shared<std::size_t>();
    auto candidate_total = std::make_shared<std::size_t>();
    D::ContactPhysicalResidency parent(parent_arena, true);
    const auto source_ring = ring(parent_arena);
    constexpr auto role = D::ContactPhysicalResidency::Role::SubfaceRing;
    parent.retain(role, source_ring, source_ring->charged_bytes(), source_ring->origin());
    const auto parent_live = parent_arena->used;
    ASSERT_EQ(parent.legacy_members().payload_bytes(), source_ring->charged_bytes());
    EXPECT_EQ(parent.inherited_members().payload_bytes(), 0u);
    EXPECT_EQ(parent.foreign_source_owners(), 0u);

    D::ContactExternalResidencyLease old_floor(parent_arena, parent_total);
    D::ContactExternalResidencyLease new_floor(candidate_arena, candidate_total);
    {
        D::ContactPhysicalResidency candidate(parent, candidate_arena);
        const auto local_ring = ring(candidate_arena);
        candidate.retain(role, local_ring, local_ring->charged_bytes(), local_ring->origin());
        ASSERT_EQ(candidate.inherited_members().payload_bytes(), source_ring->charged_bytes());
        EXPECT_EQ(candidate.foreign_source_owners(), 1u);
        EXPECT_FALSE(parent.legacy_members().same_root(candidate.legacy_members()));
        const auto before = candidate_arena->used;
        EXPECT_THROW(candidate.transfer_inherited_members(old_floor), std::invalid_argument);
        EXPECT_EQ(old_floor.bytes(), 0u);
        EXPECT_EQ(candidate_arena->used, before);
        candidate.transfer_inherited_members(new_floor);
        EXPECT_EQ(candidate_arena->used, before);
        EXPECT_EQ(new_floor.bytes(), source_ring->charged_bytes());
        EXPECT_EQ(*candidate_total, new_floor.bytes());
        EXPECT_EQ(parent_arena->used, parent_live);
        EXPECT_EQ(parent.inherited_members().payload_bytes(), 0u);
    }
    EXPECT_EQ(candidate_arena->used, new_floor.bytes());
    new_floor.resize(0);
    EXPECT_EQ(candidate_arena->used, 0u);
    EXPECT_EQ(parent_arena->used, parent_live);
}
TEST(NativeContactFunding, InheritedPointPagesUseTheirActualPoolPrecharge)
{
    using Point = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
    auto original = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    D::IdentityPool<Point> points(*original, original, {}, {}, D::IdentityPoolStorage::Persistent);
    const auto slot = points.intern(PLIC::details::GeometryIdentityAccess::point(
        PLIC::details::OrdinaryIdentityPoint{1, 2, 3}));
    auto view = points.page_view();
    const auto reference = view->capture(points, slot);
    view->freeze();
    D::ContactPhysicalResidency parent(original, true);
    parent.bind_funded_point_scope(points.scope_id());
    view->visit_credit_owners([&](const auto& credit, std::size_t bytes) { parent.observe_point_credit(credit, bytes, true); });

    auto next = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    const auto total = std::make_shared<std::size_t>();
    D::ContactExternalResidencyLease floor(next, total);
    {
        D::IdentityPool<Point> inherited(points, *next, next);
        D::ContactPhysicalResidency residency(parent, next);
        const auto precharged = inherited.inherited_page_admission();
        const auto covered = residency.inherited_point_floor_bytes();
        ASSERT_EQ(residency.funded_point_scope(), inherited.scope_id());
        ASSERT_EQ(covered, residency.inherited_members().payload_bytes());
        ASSERT_GT(covered, 0u);
        ASSERT_GE(precharged, covered);
        // This pool still reserves its original separate 128-byte Page
        // allowance. The real Page/Credit fixed group enters the union once.
        EXPECT_EQ(precharged - covered, 128u);
        const auto before = next->used;
        EXPECT_THROW(residency.transfer_inherited_members(floor), std::invalid_argument);
        EXPECT_EQ(next->used, before);
        residency.transfer_inherited_members(floor, covered);
        inherited.transfer_inherited_page_bytes(covered);
        EXPECT_EQ(next->used, before);
        EXPECT_EQ(floor.bytes(), covered);
        EXPECT_EQ(inherited.inherited_page_admission(), 128u);
        EXPECT_EQ(view->get(reference), inherited.get(slot));
    }
    EXPECT_EQ(next->used, floor.bytes());
    floor.resize(0);
    EXPECT_EQ(next->used, 0u);
}
TEST(NativeContactFunding, SideIndexesAccountForSharedAndReplacedActualMetadata)
{
    using Point = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
    using Graph = PLIC::details::GeometryOwnerGraphEditor;
    using Manifest = PLIC::details::geometry_demand::GeometryOwnerManifest;
    using Metadata = PLIC::details::geometry_demand::NativeMetadataDelete;
    auto original = std::make_shared<D::Budget>(D::Budget{16 * 1024 * 1024});
    auto next = std::make_shared<D::Budget>(D::Budget{16 * 1024 * 1024});
    pipeline_trace::begin();
    {
        D::IdentityPool<Point> points(*original, original, {}, {}, D::IdentityPoolStorage::Persistent);
        const auto slot = points.intern(PLIC::details::GeometryIdentityAccess::point(
            PLIC::details::OrdinaryIdentityPoint{1, 2, 3}));
        auto view = points.page_view();
        (void) view->capture(points, slot);
        view->freeze();
        auto source = std::make_unique<D::ContactPhysicalResidency>(original, true);
        source->bind_funded_point_scope(points.scope_id());
        view->visit_credit_owners([&](const auto& owner, auto bytes) { source->observe_point_credit(owner, bytes, true); });
        constexpr auto role = D::ContactPhysicalResidency::Role::SubfaceRing;
        const auto retired = ring(original);
        source->retain(role, retired, retired->charged_bytes(), retired->origin());
        auto previous = source->source_side_indexes();

        D::IdentityPool<Point> inherited(points, *next, next);
        D::ContactPhysicalResidency candidate(*source, next);
        candidate.release(role, retired);
        const auto replacement = ring(next);
        candidate.retain(role, replacement, replacement->charged_bytes(), replacement->origin());
        const auto current = candidate.source_side_indexes();
        using Counts = std::map<const void*, std::size_t>;
        const auto physical_metadata = [](const auto& roots, Counts expected)
        {
            for (const auto& root : roots)
                root.visit_metadata([&](const auto* address, auto payload) { expected.emplace(address, payload); });
            return expected;
        };
        const auto project = [](Manifest manifest, const auto& roots, bool retain)
        {
            std::size_t cursor = 0;
            Graph graph(manifest, 0, cursor, 16 * 1024 * 1024);
            for (const auto& root : roots) D::project_contact_side_index(root, graph, retain);
            return graph.finish();
        };
        const auto verify = [](const Manifest& manifest, Counts expected)
        {
            pipeline_trace::prepare_census();
            manifest.visit([&](const auto& row)
            {
                const auto entry = expected.find(row.owner.get());
                if (entry == expected.end()) return;
                EXPECT_NE(pipeline_trace::mark(row.owner.get(), 1), nullptr);
                const auto* actual = std::get_deleter<Metadata>(row.owner);
                ASSERT_NE(actual, nullptr);
                EXPECT_EQ(row.bytes, entry->second + actual->control_bytes);
                EXPECT_TRUE(manifest.admits(0, Graph::external_kind(), actual->state,
                    actual->state->allocation_admission_bytes));
                EXPECT_TRUE(manifest.admits(0, Graph::external_kind(), actual->state->counter_owner,
                    actual->state->counter_allocations.object.bytes + actual->state->counter_allocations.control.bytes));
                EXPECT_TRUE(manifest.admits(0, Graph::external_kind(), actual->state->mirror.owner,
                    actual->state->mirror.owner_proof.bytes()));
                expected.erase(entry);
            });
            EXPECT_TRUE(expected.empty());
        };
        const pipeline_trace::Pause pause;
        const auto old_metadata = physical_metadata(previous, {});
        const auto new_metadata = physical_metadata(current, {});
        const auto union_metadata = physical_metadata(current, old_metadata);
        ASSERT_LT(union_metadata.size(), old_metadata.size() + new_metadata.size());
        ASSERT_GT(union_metadata.size(), new_metadata.size());
        {
            const auto side = std::find_if(previous.begin(), previous.end(), [](const auto& root)
                { return bool(PLIC::details::geometry_demand::metadata_state(root.root_control())); });
            ASSERT_NE(side, previous.end());
            const auto state = PLIC::details::geometry_demand::metadata_state(side->root_control());
            const auto envelope = geometry_detail::NativeAllocationEnvelopeAccess::make<unsigned char>(
                geometry_detail::NativeConstructionAdmission::from_state(state));
            const auto fixed = [&](const Manifest& accepted, bool retain)
            {
                std::size_t cursor = 0;
                Graph graph(accepted, 0, cursor, 16 * 1024 * 1024);
                graph.native_fixed(envelope.envelope, retain);
                return graph.finish();
            };
            const std::array side_only{*side};
            const auto mixed = fixed(project({}, side_only, true), true);
            const auto counter_edges = [&](const Manifest& graph)
            { return graph.references(0, Graph::external_kind(), state->counter_owner); };
            // Both roots reach the same typed State and its two actual counter
            // fields. Either retirement order must leave those fields until last.
            EXPECT_EQ(counter_edges(mixed), 2u);
            const auto after_fixed = fixed(mixed, false);
            const auto after_index = project(mixed, side_only, false);
            EXPECT_EQ(counter_edges(after_fixed), 2u);
            EXPECT_EQ(counter_edges(after_index), 2u);
            EXPECT_EQ(project(after_fixed, side_only, false).owner_count(), 0u);
            EXPECT_EQ(fixed(after_index, false).owner_count(), 0u);
        }

        auto graph = project(project({}, previous, true), current, true);
        verify(graph, union_metadata);
        graph = project(std::move(graph), previous, false);
        verify(graph, new_metadata);
        previous = {};
        source.reset();
        pipeline_trace::prepare_census();
        for (const auto& entry : old_metadata)
            if (!new_metadata.contains(entry.first)) EXPECT_EQ(pipeline_trace::mark(entry.first, 1), nullptr);
        graph = project(std::move(graph), current, false);
        EXPECT_EQ(graph.owner_count(), 0u);
    }
    pipeline_trace::recording.store(false);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
TEST(NativeContactFunding, ContactReplayPreAdmitsItsActualCompactKernel)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    const geometry_detail::NativeConstructionAdmission admission(
        geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(
            arena->limit, 0, D::NativeSourceOwnerStore::mirror(arena)));
    const auto original = arena->used;
    pipeline_trace::begin();
    {
        const auto replay = D::cartesian_incidence_replay_kernel(Meshes::AMR::AMRGrid{}, admission);
        const pipeline_trace::Pause pause;
        const auto actual = pipeline_trace::summary().all_live;
        EXPECT_TRUE(replay.envelope->shares_control(replay.owner));
        EXPECT_FALSE(D::NativeKernelAuthority::qualified(replay.owner));
        EXPECT_GT(actual, 0u);
        EXPECT_EQ(actual, replay.envelope->fixed_allocation_bytes());
        EXPECT_EQ(arena->used - original, actual);
    }
    pipeline_trace::recording.store(false);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
    EXPECT_EQ(arena->used, original);
}
} // namespace
