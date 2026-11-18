#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/NativePublicationOperation.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
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
} // namespace
