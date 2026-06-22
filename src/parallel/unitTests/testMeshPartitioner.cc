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
#include "geometry/MeshHandle.hh"
#include "fields/CellField.hh"
#include "fields/FieldStored.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cstddef>
#include <memory>
#include <numeric>
#include <set>
#include <type_traits>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using MeshHandle = SimpleFluid::MeshHandle<Pack>;
using UnstructuredMesh = SimpleFluid::Meshes::UnstructuredMesh;
using PartitionedUnstructured =
    SimpleFluid::Meshes::PartitionedMesh<UnstructuredMesh, Pack>;
using PartitionedCellField =
    SimpleFluid::ScalarCellFieldStored<Pack, PartitionedUnstructured>;
using PartitionIndexer =
    SimpleFluid::MeshPartitioner<Pack>::indexer_type;
using MeshIndexer =
    UnstructuredMesh::local_global_indexer_t<
        Pack::local_ordinal_type,
        Pack::global_ordinal_type>;

static_assert(std::is_same_v<PartitionIndexer, MeshIndexer>);

static_assert(std::is_same_v<
    typename PartitionIndexer::cell_id_t,
    UnstructuredMesh::cell_id_t>);
static_assert(std::is_same_v<
    typename PartitionIndexer::face_id_t,
    UnstructuredMesh::face_id_t>);
static_assert(std::is_same_v<
    typename PartitionIndexer::node_id_t,
    UnstructuredMesh::node_id_t>);

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

SimpleFluid::SP<UnstructuredMesh>
make_unstructured_hex_line(unsigned cells)
{
    using Vec3 = UnstructuredMesh::Vec3;
    using CellDefinition = UnstructuredMesh::CellDefinition;
    using BoundaryFaceDefinition =
        UnstructuredMesh::BoundaryFaceDefinition;
    const auto nodes_per_x = cells + 1U;
    auto node_id = [nodes_per_x](unsigned i,
                                 unsigned j,
                                 unsigned k)
    {
        return i + nodes_per_x * (j + 2U * k);
    };

    SimpleFluid::Arr<Vec3> nodes;
    nodes.reserve(static_cast<size_t>(nodes_per_x) * 4U);
    for (unsigned k = 0; k < 2U; ++k)
    {
        for (unsigned j = 0; j < 2U; ++j)
        {
            for (unsigned i = 0; i <= cells; ++i)
            {
                nodes.push_back({
                    static_cast<SimpleFluid::real_t>(i),
                    static_cast<SimpleFluid::real_t>(j),
                    static_cast<SimpleFluid::real_t>(k)});
            }
        }
    }

    SimpleFluid::Arr<CellDefinition> cell_defs;
    cell_defs.reserve(cells);
    for (unsigned cell = 0; cell < cells; ++cell)
    {
        cell_defs.push_back({
            UnstructuredMesh::CellType::HEXAHEDRON,
            {
                node_id(cell, 0, 0),
                node_id(cell + 1U, 0, 0),
                node_id(cell + 1U, 1, 0),
                node_id(cell, 1, 0),
                node_id(cell, 0, 1),
                node_id(cell + 1U, 0, 1),
                node_id(cell + 1U, 1, 1),
                node_id(cell, 1, 1),
            }});
    }

    SimpleFluid::Arr<BoundaryFaceDefinition> boundaries{
        {{node_id(0, 1, 0),
          node_id(0, 1, 1),
          node_id(0, 0, 1),
          node_id(0, 0, 0)}, 1, "xmin"},
        {{node_id(cells, 0, 0),
          node_id(cells, 0, 1),
          node_id(cells, 1, 1),
          node_id(cells, 1, 0)}, 2, "xmax"}};

    return std::make_shared<UnstructuredMesh>(
        nodes,
        cell_defs,
        boundaries);
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

std::set<SimpleFluid::global_index_t>
gather_all_owned_gids(const MeshHandle& mesh)
{
    using GO = SimpleFluid::global_index_t;
    const auto comm = mesh.owned_cell_map()->getComm();
    const int nranks = comm->getSize();

    int my_count = static_cast<int>(mesh.num_owned_cells());
    std::vector<int> all_counts(static_cast<size_t>(nranks));
    my_mpi::allgather(&my_count, 1, all_counts.data(), 1);

    std::vector<int> displs(static_cast<size_t>(nranks), 0);
    for (int r = 1; r < nranks; ++r)
    {
        displs[static_cast<size_t>(r)] =
            displs[static_cast<size_t>(r - 1)]
            + all_counts[static_cast<size_t>(r - 1)];
    }

    std::vector<GO> my_gids;
    my_gids.reserve(mesh.num_owned_cells());
    for (size_t lid = 0; lid < mesh.num_owned_cells(); ++lid)
    {
        my_gids.push_back(mesh.cell_global_id(
            static_cast<Pack::local_ordinal_type>(lid)));
    }

    std::vector<GO> all_gids(
        static_cast<size_t>(displs.back() + all_counts.back()));

    my_mpi::allgatherv(
        my_gids.data(),
        my_count,
        all_gids.data(),
        all_counts.data(),
        displs.data());

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
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "ymax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
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

TEST(MeshPartitionerTest, MeshHandlePartitionsUnstructuredMesh)
{
    constexpr unsigned cell_count = 8;
    auto mesh = make_unstructured_hex_line(cell_count);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<double> source_cell_volumes;
    source_cell_volumes.reserve(mesh->num_cells());
    for (size_t cell = 0; cell < mesh->num_cells(); ++cell)
    {
        source_cell_volumes.push_back(
            mesh->cell_volume(mesh->cell_id(cell)));
    }
    std::vector<double> source_face_areas;
    std::vector<bool> source_interior_faces;
    source_face_areas.reserve(mesh->num_faces());
    source_interior_faces.reserve(mesh->num_faces());
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        source_face_areas.push_back(
            mesh->face_area(mesh->face_id(face)));
        source_interior_faces.push_back(
            mesh->is_interior_face(mesh->face_id(face)));
    }
    const auto source_node_count = mesh->num_nodes();

    const auto partition =
        SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm);
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto& indexer = partition.indexer;
    EXPECT_EQ(mesh->num_cells(), indexer.num_local_cells());
    EXPECT_EQ(mesh->num_owned_cells(), indexer.num_owned_cells());
    EXPECT_EQ(mesh->num_faces(), indexer.num_local_faces());
    EXPECT_EQ(mesh->num_owned_faces(), indexer.num_owned_faces());
    EXPECT_EQ(mesh->num_nodes(), indexer.num_local_nodes());
    EXPECT_LT(mesh->num_cells(), static_cast<size_t>(cell_count));
    EXPECT_GT(indexer.num_owned_cells(), 0U);
    EXPECT_GT(indexer.num_local_nodes(), 0U);
    EXPECT_EQ(
        partition.cell_owner_ranks.size(),
        static_cast<size_t>(cell_count));
    for (size_t local = 0; local < indexer.num_local_cells(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        const auto global_id = indexer.cell_global_id(local_id);
        EXPECT_EQ(indexer.cell_local_id(global_id), local_id);
        EXPECT_LT(static_cast<size_t>(global_id), source_cell_volumes.size());
        const auto local_geometry_id = mesh->cell_id(local);
        const auto global_ordinal =
            static_cast<Pack::global_ordinal_type>(global_id);
        EXPECT_EQ(
            indexer.local_to_global_cell_id(local_geometry_id),
            global_id);
        EXPECT_EQ(
            indexer.global_to_local_cell_id(global_id),
            local_geometry_id);
        EXPECT_EQ(
            indexer.local_to_global_cell_ordinal(local_id),
            global_ordinal);
        EXPECT_EQ(
            indexer.global_to_local_cell_ordinal(global_ordinal),
            local_id);
        EXPECT_EQ(
            mesh->is_owned_cell(local_geometry_id),
            indexer.is_owned_cell(local_id));
        EXPECT_DOUBLE_EQ(
            mesh->cell_volume(local_geometry_id),
            source_cell_volumes[static_cast<size_t>(global_id)]);
        if (indexer.is_owned_cell(local_id))
        {
            EXPECT_EQ(
                partition.cell_owner_ranks[static_cast<size_t>(global_id)],
                comm->getRank());
        }
    }
    for (size_t local = 0; local < indexer.num_local_faces(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        const auto global_id = indexer.face_global_id(local_id);
        EXPECT_EQ(indexer.face_local_id(global_id), local_id);
        EXPECT_LT(static_cast<size_t>(global_id), source_face_areas.size());
        const auto local_geometry_id = mesh->face_id(local);
        const auto global_ordinal =
            static_cast<Pack::global_ordinal_type>(global_id);
        EXPECT_EQ(
            indexer.local_to_global_face_id(local_geometry_id),
            global_id);
        EXPECT_EQ(
            indexer.global_to_local_face_id(global_id),
            local_geometry_id);
        EXPECT_EQ(
            indexer.local_to_global_face_ordinal(local_id),
            global_ordinal);
        EXPECT_EQ(
            indexer.global_to_local_face_ordinal(global_ordinal),
            local_id);
        EXPECT_EQ(
            mesh->is_owned_face(local_geometry_id),
            indexer.is_owned_face(local_id));
        EXPECT_DOUBLE_EQ(
            mesh->face_area(local_geometry_id),
            source_face_areas[static_cast<size_t>(global_id)]);
    }
    for (size_t local = 0; local < indexer.num_local_nodes(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        const auto global_id = indexer.node_global_id(local_id);
        EXPECT_EQ(indexer.node_local_id(global_id), local_id);
        EXPECT_LT(static_cast<size_t>(global_id), source_node_count);
        const auto local_geometry_id = mesh->node_id(local);
        const auto global_ordinal =
            static_cast<Pack::global_ordinal_type>(global_id);
        EXPECT_EQ(
            indexer.local_to_global_node_id(local_geometry_id),
            global_id);
        EXPECT_EQ(
            indexer.global_to_local_node_id(global_id),
            local_geometry_id);
        EXPECT_EQ(
            indexer.local_to_global_node_ordinal(local_id),
            global_ordinal);
        EXPECT_EQ(
            indexer.global_to_local_node_ordinal(global_ordinal),
            local_id);
    }

    EXPECT_THROW(MeshHandle{mesh}, std::invalid_argument);
    const auto partitioned = std::make_shared<PartitionedUnstructured>(
        mesh, partition.indexer, comm);
    EXPECT_EQ(partitioned->num_global_cells(),
              static_cast<size_t>(cell_count));
    EXPECT_EQ(partitioned->num_global_faces(), source_face_areas.size());
    EXPECT_EQ(partitioned->num_global_nodes(), source_node_count);
    EXPECT_EQ(partitioned->num_local_cells(), indexer.num_local_cells());
    EXPECT_EQ(partitioned->num_owned_cells(), indexer.num_owned_cells());
    EXPECT_EQ(partitioned->num_local_faces(), indexer.num_local_faces());
    EXPECT_EQ(partitioned->num_owned_faces(), indexer.num_owned_faces());
    EXPECT_EQ(partitioned->num_local_nodes(), indexer.num_local_nodes());
    EXPECT_EQ(partitioned->num_owned_nodes(), indexer.num_owned_nodes());
    EXPECT_EQ(
        partitioned->cell_global_id(0),
        indexer.local_to_global_cell_ordinal(0));
    EXPECT_EQ(
        partitioned->cell_local_id(indexer.cell_global_id(0)), 0);
    EXPECT_TRUE(partitioned->is_owned_cell(0));
    EXPECT_TRUE(partitioned->is_owned_node(0));
    EXPECT_DOUBLE_EQ(partitioned->cell_volume(0), mesh->cell_volume(0));

    PartitionedCellField stored(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("partitioned_value"),
        partitioned);
    for (size_t local = 0; local < partitioned->num_owned_cells(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        stored.set_owned_value(
            local_id,
            static_cast<double>(partitioned->cell_global_id(local_id)));
    }
    stored.sync_ghosts();
    for (size_t local = 0; local < partitioned->num_local_cells(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(
            stored.local_value(local_id),
            static_cast<double>(partitioned->cell_global_id(local_id)));
    }

    const MeshHandle handle(partitioned);
    const auto& handle_mesh =
        std::get<MeshHandle::UnstructuredPtr>(handle.variant());
    ASSERT_TRUE(handle_mesh);
    EXPECT_EQ(handle_mesh.get(), mesh.get());
    EXPECT_EQ(handle_mesh->num_cells(), handle.num_local_cells());
    EXPECT_EQ(handle_mesh->num_faces(), handle.num_faces());
    EXPECT_EQ(handle_mesh->num_owned_cells(), handle.num_owned_cells());
    EXPECT_LT(handle_mesh->num_cells(), static_cast<size_t>(cell_count));

    EXPECT_GT(handle.num_owned_cells(), 0u)
        << "Rank " << comm->getRank()
        << " has zero unstructured owned cells.";
    EXPECT_LE(handle.num_owned_cells(), handle.num_local_cells());

    const auto all_gids = gather_all_owned_gids(handle);
    EXPECT_EQ(all_gids.size(), static_cast<size_t>(cell_count));

    int my_owned_count = static_cast<int>(handle.num_owned_cells());
    int global_owned_count = 0;
    my_mpi::reduce(
        &my_owned_count,
        &global_owned_count,
        1,
        MPI_SUM,
        0);
    if (comm->getRank() == 0)
    {
        EXPECT_EQ(global_owned_count, static_cast<int>(cell_count));
    }

    const int my_ghost_count = static_cast<int>(
        handle.num_local_cells() - handle.num_owned_cells());
    int global_ghost_count = 0;
    my_mpi::allreduce(
        &my_ghost_count,
        &global_ghost_count,
        1,
        MPI_SUM);
    EXPECT_GT(global_ghost_count, 0);

    for (size_t cell_lid = 0;
         cell_lid < handle.num_owned_cells();
         ++cell_lid)
    {
        const auto local_cell =
            static_cast<Pack::local_ordinal_type>(cell_lid);
        for (const auto face_lid : handle.faces(local_cell))
        {
            const auto global_face = static_cast<size_t>(
                handle.face_global_id(face_lid));
            ASSERT_LT(global_face, source_interior_faces.size());
            if (source_interior_faces[global_face])
            {
                EXPECT_NE(
                    handle.opposite_cell(face_lid, local_cell),
                    MeshHandle::invalid_local_id());
            }
        }
    }
}
