#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/NativePublicationOperation.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"
#include <gtest/gtest.h>
#include <memory>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace G = SimpleFluid::geometry_detail;
namespace A = SimpleFluid::Meshes::AMR;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(NativePublicationOperation, PreparedLeaseTransferPreservesAcceptedBytesUntilCommit)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{512});
    auto total = std::make_shared<std::size_t>(0);
    D::ContactExternalResidencyLease accepted(arena, total);
    accepted.resize(200);
    {
        auto growth = accepted.prepare_mirror(80);
        {
            auto transfer = accepted.prepare_transfer(growth, 80);
            EXPECT_EQ(accepted.bytes(), 200);
            EXPECT_EQ(growth.bytes(), 80);
            EXPECT_EQ(arena->used, 280);
            EXPECT_EQ(*total, 280);
            EXPECT_THROW(accepted.resize(201), std::logic_error);
            // A later rank-local failure drops this token without publication.
        }
        EXPECT_EQ(accepted.bytes(), 200);
        EXPECT_EQ(growth.bytes(), 80);
    }
    EXPECT_EQ(arena->used, 200);
    EXPECT_EQ(*total, 200);
    {
        auto growth = accepted.prepare_mirror(80);
        auto transfer = accepted.prepare_transfer(growth, 80);
        transfer.commit();
        transfer.commit(); // Consumed token is harmless; it cannot spend twice.
        EXPECT_EQ(accepted.bytes(), 280);
        EXPECT_EQ(growth.bytes(), 0);
        EXPECT_EQ(arena->used, 280);
        EXPECT_EQ(*total, 280);
    }
    accepted.resize(160);
    EXPECT_EQ(arena->used, 160);
    EXPECT_EQ(*total, 160);
}

TEST(NativePublicationOperation, ForeignArenaCannotFundAcceptedLeaseAndShortMirrorRollsBack)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{279});
    auto total = std::make_shared<std::size_t>(0);
    D::ContactExternalResidencyLease accepted(arena, total);
    accepted.resize(200);
    EXPECT_THROW((void) accepted.prepare_mirror(80), std::invalid_argument);
    EXPECT_EQ(accepted.bytes(), 200);
    EXPECT_EQ(arena->used, 200);
    EXPECT_EQ(*total, 200);
    auto foreign_arena = std::make_shared<D::Budget>(D::Budget{279});
    auto foreign_total = std::make_shared<std::size_t>(0);
    D::ContactExternalResidencyLease foreign(foreign_arena, foreign_total);
    foreign.resize(79);
    EXPECT_THROW((void) accepted.prepare_transfer(foreign, 79), std::invalid_argument);
    EXPECT_EQ(accepted.bytes(), 200);
    EXPECT_EQ(foreign.bytes(), 79);
    EXPECT_EQ(arena->used, 200);
    EXPECT_EQ(foreign_arena->used, 79);
    {
        auto exact = accepted.prepare_mirror(79);
        auto transfer = accepted.prepare_transfer(exact, 79);
        transfer.commit();
    }
    EXPECT_EQ(accepted.bytes(), 279);
    EXPECT_EQ(arena->used, 279);
}

TEST(NativePublicationOperation, DisjointSourcePartitionRejectsInventedOverlap)
{
    const auto partition = D::SourceAccountingPartition::split(317, 101, 211, 67);
    EXPECT_EQ(partition.exclusive_source_admission, 216);
    EXPECT_EQ(partition.exclusive_contact_admission, 144);
    EXPECT_EQ(partition.residual_bytes(), 360);
    EXPECT_THROW((void) D::SourceAccountingPartition::split(100, 101, 211, 67), std::logic_error);
    EXPECT_THROW((void) D::SourceAccountingPartition::split(317, 101, 66, 67), std::logic_error);
}

struct Row { int key; };
struct Less { bool operator()(Row a, Row b) const noexcept { return a.key < b.key; } };
TEST(NativePublicationOperation, PersistentNodeReceiptsSurviveActualAllocationRetirementBetweenEdits)
{
    using Pages = D::persistent_ordered_detail::OrderedPages<Row, Less>;
    G::NativeConstructionAdmission admission(G::NativeConstructionCounterAccess::create<D::Budget>(1024 * 1024));
    auto arena = std::static_pointer_cast<D::Budget>(std::const_pointer_cast<void>(admission.state()->counter_owner));
    Pages editor(arena, nullptr, {}, admission.state());
    const Row first{1}, second{2};
    auto root = editor.build(std::span(&first, 1));
    ASSERT_TRUE(root);
    ASSERT_TRUE(root->charge.native);
    EXPECT_TRUE(root->charge.native->node->matches(root));
    EXPECT_EQ(root->charge.bytes, root->charge.native->bytes());
    const auto before = admission.statistics();
    {
        auto temporary = admission.unique<std::array<std::byte, 4096>>();
        EXPECT_EQ(admission.statistics().actual_live, before.actual_live + sizeof(*temporary));
    }
    EXPECT_EQ(admission.statistics().actual_live, before.actual_live);
    EXPECT_GE(admission.statistics().actual_peak, before.actual_live + 4096);
    auto next = editor.edit(root, second, &second, false);
    EXPECT_EQ(next->size, 2);
    EXPECT_TRUE(next->charge.native->node->matches(next));
    EXPECT_EQ(next->charge.bytes, next->charge.native->bytes());
    EXPECT_EQ(root->size, 1);
}

TEST(NativePublicationOperation, ActualBuiltInSourceStagingFailurePreservesRootsLeasesAndMirroredArena)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The local rollback gate uses one actual owner; registered producer MPI gates are separate.";
    A::AMRGrid config;
    config.block_cells = {2, 2, 2};
    config.max_level = 1;
    A::CartesianAMR original(config);
    const std::array<std::uint64_t, 8> selection{0, 1, 2, 3, 4, 5, 6, 7};
    auto detached = original.detach_owned(selection, {'r', 'o', 'l', 'l', 'b', 'a', 'c', 'k'});
    const auto source = std::make_shared<const A::DistributedAdaptiveHierarchy>(std::move(detached.owned),
        std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
    const auto before = D::GeometrySnapshotAccess::cache_admission_statistics(*source);
    const auto input = D::NativePublicationSession::capture(*source);
    const auto accepted_indexes = input.cache->indexes;
    const auto accepted_revision = input.cache->revision;
    const auto accepted_coverage = input.source_coverage;
    const auto arena = D::GeometrySnapshotAccess::contact_cache(*source).allocation_arena();
    const auto admitted_before = arena->used;
    ASSERT_TRUE(accepted_coverage);
    ASSERT_GT(before.resident_bytes, 0);
    {
        D::NativePublicationOperation operation(*source);
        EXPECT_EQ(operation.admission().state()->mirror.owner.get(), arena.get());
        D::NativePublicationSession staged(operation.inputs(), operation.admission(), &operation.geometry_accounting(),
            &operation.source_coverage_transaction());
        const auto proof = staged.source(*source->owned_leaves().begin());
        ASSERT_TRUE(proof);
        EXPECT_DOUBLE_EQ(proof->metrics().volume, .125);
        EXPECT_EQ(input.cache->indexes, accepted_indexes);
        EXPECT_EQ(input.cache->revision, accepted_revision);
        // A later construction/collective failure abandons the staged source.
        // Destruction must retire every real mirrored allocation, without
        // installing the new index or consuming a source-floor transfer.
    }
    const auto after = D::GeometrySnapshotAccess::cache_admission_statistics(*source);
    EXPECT_EQ(after.resident_bytes, before.resident_bytes);
    EXPECT_EQ(after.lease_bytes, before.lease_bytes);
    EXPECT_EQ(input.cache->indexes, accepted_indexes);
    EXPECT_EQ(input.cache->revision, accepted_revision);
    EXPECT_EQ(D::NativePublicationSession::capture(*source).source_coverage, accepted_coverage);
    EXPECT_EQ(arena->used, admitted_before);
    EXPECT_NO_THROW(accepted_coverage->require_current());
    // The rejected operation did not poison a subsequent actual admitted
    // first-source capture through the same closed built-in kernel control.
    D::NativePublicationOperation retry(*source);
    D::NativePublicationSession staged(retry.inputs(), retry.admission(), &retry.geometry_accounting(),
        &retry.source_coverage_transaction());
    EXPECT_DOUBLE_EQ(staged.source(*source->owned_leaves().begin())->metrics().volume, .125);
}

TEST(NativePublicationOperation, UnionAdmissionRejectsCollectivelyAndPreservesAcceptedPublication)
{
    const auto comm = Tpetra::getDefaultComm();
    A::AMRGrid config;
    config.block_cells = {2, 2, 2};
    A::CartesianAMR original(config);
    const std::array<std::uint64_t, 8> cells{0, 1, 2, 3, 4, 5, 6, 7};
    const auto owned = comm->getRank() == 0 ? std::span<const std::uint64_t>(cells) : std::span<const std::uint64_t>{};
    auto detached = original.detach_owned(owned, {'u', 'n', 'i', 'o', 'n'});
    const A::DistributedAdaptiveHierarchy source(std::move(detached.owned), std::move(detached.families),
        std::move(detached.kernel), comm);
    const auto schema = A::native_publication_schema(source);
    const auto first = A::publish_adaptive_native(source, schema);
    const auto before = D::NativePublicationSession::capture(source);
    const auto revision = before.cache->revision;
    const auto receipt = before.cache->receipt;
    const auto indexes = before.cache->indexes;
    const auto totals = D::GeometrySnapshotAccess::cache_admission_statistics(source);
    const auto arena = D::GeometrySnapshotAccess::contact_cache(source).allocation_arena();
    const auto live = arena->used, spent = arena->cumulative, original_limit = arena->limit;
    struct Probe
    {
        D::Budget& arena;
        std::size_t limit;
        bool reject;
        bool entered = false;
    } probe{*arena, original_limit, comm->getRank() == comm->getSize() - 1};
    D::PipelineObserver observer;
    observer.context = &probe;
    observer.native_owner = [](void* raw, D::PipelineNativeOwnerStage stage, bool entering,
                               D::PipelineNativeOwnerAdmission) noexcept
    {
        if (stage != D::PipelineNativeOwnerStage::Union)
            return;
        auto& probe = *static_cast<Probe*>(raw);
        probe.entered |= entering;
        if (probe.reject)
            probe.arena.limit = entering ? probe.arena.used : probe.limit;
    };
    {
        const struct RestoreObserver
        {
            D::PipelineObserver previous;
            ~RestoreObserver() { (void) D::set_pipeline_observer(previous); }
        } restore{D::set_pipeline_observer(observer)};
        EXPECT_THROW((void) A::publish_adaptive_native(source, schema), std::invalid_argument);
    }
    EXPECT_TRUE(probe.entered);
    EXPECT_EQ(arena->limit, original_limit);
    EXPECT_EQ(arena->used, live);
    EXPECT_GT(arena->cumulative, spent);
    const auto after = D::NativePublicationSession::capture(source);
    EXPECT_EQ(after.cache->revision, revision);
    EXPECT_EQ(after.cache->receipt, receipt);
    EXPECT_EQ(after.cache->indexes, indexes);
    EXPECT_EQ(after.source_coverage, before.source_coverage);
    EXPECT_TRUE(D::NativePublicationSession::receipt_matches(after, first.partition(), first.mesh()->owned_cell_map()));
    EXPECT_EQ(D::GeometrySnapshotAccess::cache_admission_statistics(source).resident_bytes, totals.resident_bytes);
    const auto retried = A::publish_adaptive_native(source, schema);
    EXPECT_TRUE(D::NativePublicationSession::receipt_matches(
        D::NativePublicationSession::capture(source), retried.partition(), retried.mesh()->owned_cell_map()));
}
} // namespace
