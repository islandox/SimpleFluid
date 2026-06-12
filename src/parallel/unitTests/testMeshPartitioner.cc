/**
 * @file testMetisPartitioner.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Multi-rank unit tests for MetisPartitioner domain decomposition.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 * Tests are run with mpiexec; single-rank runs skip automatically.
 */

#include <gtest/gtest.h>

#include "parallel/MeshPartitioner.hh"

#include "geometry/Mesh.hh"
#include "fields/CellField.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cstddef>
#include <memory>
#include <numeric>
#include <set>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_4x4x4_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_4x4x4_database());
}

SimpleFluid::SP<MeshType> make_2x2x2_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

/**
 * @brief Gather all owned cell GIDs from all ranks and return the union.
 */
std::set<SimpleFluid::global_index_t>
gather_all_owned_gids(const MeshType& mesh)
{
    using GO = SimpleFluid::global_index_t;
    const auto comm = mesh.owned_cell_map()->getComm();
    const int nranks = comm->getSize();
    const int myrank = comm->getRank();

    const auto& owned = mesh.owned_cell_global_ids();
    int my_count = static_cast<int>(owned.size());

    std::vector<int> all_counts(static_cast<size_t>(nranks));
    my_mpi::allgather(&my_count, 1, all_counts.data(), 1);

    std::vector<int> displs(static_cast<size_t>(nranks), 0);
    for (int r = 1; r < nranks; ++r)
        displs[static_cast<size_t>(r)] =
            displs[static_cast<size_t>(r - 1)]
            + all_counts[static_cast<size_t>(r - 1)];

    std::vector<GO> my_gids(owned.begin(), owned.end());
    std::vector<GO> all_gids(
        static_cast<size_t>(displs.back() + all_counts.back()));

    my_mpi::allgatherv(my_gids.data(), my_count,
                   all_gids.data(), all_counts.data(), displs.data());

    return std::set<GO>(all_gids.begin(), all_gids.end());
}

} // namespace

// ---------------------------------------------------------------------------
//  Test 1: Partition produces a valid distribution
// ---------------------------------------------------------------------------
/**
 * @brief Verifies mesh partitioning produces a valid distribution with each rank owning at least one cell.
 */
TEST(MeshPartitionerTest, ProducesValidDistribution)
{
    auto mesh = make_4x4x4_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();

    SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, mesh->owned_cell_map()->getComm());

    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    // Every rank must have at least one owned cell (for a 64-cell mesh)
    EXPECT_GT(mesh->num_owned_cells(), 0u)
        << "Rank " << comm->getRank() << " has zero owned cells.";

    // Owned cells should be a subset of local cells
    EXPECT_LE(mesh->num_owned_cells(), mesh->num_local_cells());

    // Total owned cells across all ranks should equal the global cell count.
    // For a 4x4x4 mesh, there are 64 cells.
    const auto all_gids = gather_all_owned_gids(*mesh);
    EXPECT_EQ(all_gids.size(), 64u)
        << "Total unique owned GIDs across all ranks is not 64.";

    // No rank should own the same cell as another rank (GIDs unique)
    size_t total_owned = 0;
    {
        int my_count = static_cast<int>(mesh->num_owned_cells());
        int global_total = 0;
        my_mpi::reduce(&my_count, &global_total, 1, MPI_SUM, 0);
        if (comm->getRank() == 0)
            EXPECT_EQ(static_cast<size_t>(global_total), 64u);
    }
}

// ---------------------------------------------------------------------------
//  Test 2: Ghost cells are correctly identified
// ---------------------------------------------------------------------------
/**
 * @brief Ensures ghost cell GIDs do not appear in the owned cell GIDs on the same rank.
 */
TEST(MeshPartitionerTest, GhostCellsAreCorrect)
{
    auto mesh = make_2x2x2_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    // Ghost cell GIDs should not appear in owned cell GIDs
    for (size_t i = 0; i < mesh->num_local_cells(); ++i)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(i);
        if (mesh->is_owned_cell(lid)) continue;

        // This is a ghost cell — its GID should NOT be in owned list
        const auto gid = mesh->cell_global_id(lid);
        bool found = false;
        for (size_t o = 0; o < mesh->num_owned_cells(); ++o)
        {
            if (mesh->cell_global_id(
                    static_cast<Pack::local_ordinal_type>(o)) == gid)
            {
                found = true;
                break;
            }
        }
        EXPECT_FALSE(found)
            << "Ghost cell GID " << gid << " found in owned cells on rank "
            << comm->getRank();
    }
}

// ---------------------------------------------------------------------------
//  Test 3: Face connectivity is preserved
// ---------------------------------------------------------------------------
/**
 * @brief Checks that interior face owner/neighbor connectivity is preserved after partitioning.
 */
TEST(MeshPartitionerTest, FaceConnectivityPreserved)
{
    auto mesh = make_4x4x4_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    // Check every interior face has both owner and neighbor present locally
    for (size_t fid = 0; fid < mesh->num_faces(); ++fid)
    {
        const auto face_lid =
            static_cast<Pack::local_ordinal_type>(fid);
        const auto owner = mesh->owner_cell(face_lid);
        const auto neighbor = mesh->neighbor_cell(face_lid);

        // Owner must always be valid
        EXPECT_GE(owner, 0);
        EXPECT_LT(static_cast<size_t>(owner), mesh->num_local_cells());

        // Interior faces must have a valid neighbor
        if (!mesh->is_exterior_face(face_lid))
        {
            EXPECT_GE(neighbor, 0);
            EXPECT_LT(static_cast<size_t>(neighbor),
                      mesh->num_local_cells());
        }
    }
}

TEST(MeshPartitionerTest, PreservesBoundaryAndFaceGeometry)
{
    auto mesh = make_4x4x4_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    int local_ymax_faces = 0;
    for (const auto& [patch_id, patch] : mesh->boundary_patches())
    {
        if (mesh->boundary_patch_name(patch_id) != "ymax")
        {
            continue;
        }
        for (const auto face_lid : patch.face_lids)
        {
            if (!mesh->is_owned_face(face_lid))
            {
                continue;
            }
            ++local_ymax_faces;
            EXPECT_GT(mesh->face_area(face_lid), 0.0);
        }
    }

    int global_ymax_faces = 0;
    my_mpi::allreduce(
        &local_ymax_faces, &global_ymax_faces, 1, MPI_SUM);
    EXPECT_GT(global_ymax_faces, 0);

    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<Pack::local_ordinal_type>(face);
        if (!mesh->is_owned_face(face_lid)
            || !mesh->is_interior_face(face_lid))
        {
            continue;
        }
        EXPECT_GT(mesh->face_area(face_lid), 0.0);
        EXPECT_GT(mesh->face_cell_center_distance(face_lid), 0.0);
    }
}

// ---------------------------------------------------------------------------
//  Test 4: Field sync works after partitioning
// ---------------------------------------------------------------------------
/**
 * @brief Verifies ghost cell values are correctly synchronized after partitioning via sync_ghosts.
 */
TEST(MeshPartitionerTest, FieldSyncAfterPartitioning)
{
    auto mesh = make_2x2x2_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType field(mesh, "sync_test", false);

    // Set each owned cell's value to its global ID
    for (size_t o = 0; o < mesh->num_owned_cells(); ++o)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(o);
        const auto gid = mesh->cell_global_id(lid);
        field.set_owned_value(
            lid, static_cast<Pack::scalar_type>(gid));
    }

    // Sync to ghost cells
    field.sync_ghosts();

    // Verify every local cell (including ghosts) has correct GID-based value
    for (size_t c = 0; c < mesh->num_local_cells(); ++c)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(c);
        const auto gid = mesh->cell_global_id(lid);
        EXPECT_DOUBLE_EQ(
            field.local_value(lid),
            static_cast<Pack::scalar_type>(gid))
            << "Mismatch for cell LID " << c << " GID " << gid
            << " on rank " << comm->getRank();
    }
}

// ---------------------------------------------------------------------------
//  Test 5: Partitioning is deterministic
// ---------------------------------------------------------------------------
/**
 * @brief Confirms two identical meshes produce the same owned-cell partition.
 */
TEST(MeshPartitionerTest, PartitioningIsDeterministic)
{
    // Build two identical meshes and verify same owned-cell counts
    auto mesh1 = make_4x4x4_box_mesh();
    auto mesh2 = make_4x4x4_box_mesh();

    const auto comm = mesh1->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    EXPECT_EQ(mesh1->num_owned_cells(), mesh2->num_owned_cells());

    // Verify owned GIDs are the same set on both meshes
    std::set<SimpleFluid::global_index_t> gids1, gids2;
    for (size_t i = 0; i < mesh1->num_owned_cells(); ++i)
        gids1.insert(mesh1->cell_global_id(
            static_cast<Pack::local_ordinal_type>(i)));
    for (size_t i = 0; i < mesh2->num_owned_cells(); ++i)
        gids2.insert(mesh2->cell_global_id(
            static_cast<Pack::local_ordinal_type>(i)));

    EXPECT_EQ(gids1, gids2);
}

// ---------------------------------------------------------------------------
//  Test 6: Partitioning handles single-rank case gracefully
// ---------------------------------------------------------------------------
TEST(MeshPartitionerTest, SingleRankCase)
{
    auto mesh = make_2x2x2_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        GTEST_SKIP() << "This test is only for single-rank runs.";
    }
    // Partitioning should succeed without error and not change the mesh
    SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm);
    EXPECT_EQ(mesh->num_owned_cells(), 8u);
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
    {
        EXPECT_TRUE(mesh->is_owned_cell(
            static_cast<Pack::local_ordinal_type>(i)));
    }
}

// ---------------------------------------------------------------------------
//  Test 7: Repeated partitioning does not change the mesh
// ---------------------------------------------------------------------------
TEST(MeshPartitionerTest, RepeatedPartitioning)
{
    auto mesh = make_4x4x4_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
    // First partitioning
    SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm);
    const auto owned_after_first = mesh->num_owned_cells();
    const auto all_gids_after_first = gather_all_owned_gids(*mesh);
    // Second partitioning should not change anything
    SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm);
    EXPECT_EQ(mesh->num_owned_cells(), owned_after_first);
    const auto all_gids_after_second = gather_all_owned_gids(*mesh);
    EXPECT_EQ(all_gids_after_second, all_gids_after_first);
}
