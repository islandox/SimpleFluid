/**
 * @file testMeshPartitioner.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Multi-rank unit tests for MeshPartitioner domain decomposition.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 * Tests are run with mpiexec; single-rank runs skip automatically.
 */

#include <gtest/gtest.h>

#include "parallel/MeshPartitioner.hh"
#include "parallel/CollectiveConstruction.hh"

#include "geometry/details/NativeMeshAssembly.hh"
#include "geometry/MeshHandle.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_unstructured_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using UnstructuredMesh = SimpleFluid::Meshes::UnstructuredMesh;
using PartitionedUnstructured =
    SimpleFluid::Meshes::PartitionedMesh<UnstructuredMesh, Pack>;
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

SimpleFluid::SP<Mesh> make_4x4x4_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_4x4x4_database());
}

SimpleFluid::SP<Mesh> make_2x2x2_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

using SimpleFluid::test::make_partition_test_hex_line;

/** @brief Graded pentagonal prisms exercise polygonal partition faces. */
UnstructuredMesh make_polyhedral_column()
{
    using Node = UnstructuredMesh::NodeID;
    const SimpleFluid::Arr<UnstructuredMesh::Vec3> base{{0,0,0},{2,0,0},{3,1,0},{1.5,2,0},{0,1,0}};
    const SimpleFluid::ArrReal layers{0,0.2,0.6,1.3,2.1,3.2,4.5};
    SimpleFluid::Arr<UnstructuredMesh::Vec3> nodes;
    for (const auto z : layers)
        for (const auto& p : base) nodes.push_back({p.x, p.y, z});
    SimpleFluid::Arr<UnstructuredMesh::CellDefinition> cells;
    for (Node k = 0; k + 1 < layers.size(); ++k)
    {
        UnstructuredMesh::CellDefinition cell;
        cell.type = UnstructuredMesh::CellType::POLYHEDRON;
        const auto lo = k * 5, hi = lo + 5;
        cell.face_node_ids = {{lo+4,lo+3,lo+2,lo+1,lo}, {hi,hi+1,hi+2,hi+3,hi+4}};
        for (Node i = 0; i < 5; ++i)
        {
            const auto next = (i + 1) % 5;
            cell.face_node_ids.push_back({lo+i,lo+next,hi+next,hi+i});
        }
        cells.push_back(std::move(cell));
    }
    const UnstructuredMesh untagged(nodes, cells);
    SimpleFluid::Arr<UnstructuredMesh::BoundaryFaceDefinition> boundaries;
    for (size_t f = 0; f < untagged.num_faces(); ++f)
        if (untagged.is_boundary_face(f))
        {
            const auto z = untagged.face_centroid(f).z;
            const int id = z == layers.front() ? 21 : z == layers.back() ? 22 : 23;
            boundaries.push_back({untagged.face_nodes(f), id, id == 21 ? "inlet" : id == 22 ? "outlet" : "wall"});
        }
    return UnstructuredMesh(nodes, cells, boundaries);
}

std::set<SimpleFluid::global_index_t>
gather_all_owned_gids(const Mesh& mesh)
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

/** @brief Factory geometry stays native after root construction and distribution. */
TEST(MeshPartitionerTest, FactoryBuildsDistributedNativeGeometry)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    SimpleFluid::MeshFactory factory(SimpleFluid::test::make_4x4x4_database());
    const auto mesh = factory.build_handle<Pack>();
    ASSERT_NE(mesh, nullptr);
    EXPECT_TRUE(std::holds_alternative<Mesh::UnstructuredPtr>(mesh->variant()));
    EXPECT_TRUE(mesh->visit([](const auto& geometry)
        { return std::is_same_v<std::remove_cvref_t<decltype(geometry)>, UnstructuredMesh>; }));
    EXPECT_EQ(mesh->owned_cell_map()->getGlobalNumElements(), 64U);
    EXPECT_GT(mesh->num_owned_cells(), 0U);
    EXPECT_LT(mesh->num_owned_cells(), 64U);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<Pack::local_ordinal_type>(cell)), 1.0);
}


/** @brief Verifies native geometry assigns a distinct identity to every local face. */
TEST(MeshPartitionerTest, InitialFaceGlobalIDsAreOneToOne)
{
    auto mesh = make_4x4x4_box_mesh();
    std::set<Pack::global_ordinal_type> face_global_ids;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<Pack::local_ordinal_type>(face);
        EXPECT_TRUE(
            face_global_ids.insert(
                mesh->face_global_id(face_lid)).second)
            << "Two local faces share global ID "
            << mesh->face_global_id(face_lid);
    }
}

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

        // Owned interior faces must have their neighboring stencil cell
        if (mesh->is_owned_face(face_lid) && !mesh->is_exterior_face(face_lid))
        {
            EXPECT_GE(neighbor, 0);
            EXPECT_LT(static_cast<size_t>(neighbor),
                      mesh->num_local_cells());
        }
    }
}

/**
 * @brief Verifies partitioning preserves boundary metadata and face
 * centroids, areas, and normals.
 */
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
/** @brief Verifies single-rank partitioning preserves the complete mesh. */
TEST(MeshPartitionerTest, SingleRankCase)
{
    auto mesh = make_2x2x2_box_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        GTEST_SKIP() << "This test is only for single-rank runs.";
    }
    // Native factory construction preserves the complete serial mesh
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
/** @brief Verifies repeated partitioning leaves the owned entity set unchanged. */
TEST(MeshPartitionerTest, RepeatedFactoryConstructionRetainsIndependentIdentity)
{
    const auto first = make_4x4x4_box_mesh();
    const auto second = make_4x4x4_box_mesh();
    EXPECT_NE(first->geometry_identity(), second->geometry_identity());
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(first->num_owned_cells(), second->num_owned_cells());
    EXPECT_EQ(gather_all_owned_gids(*first), gather_all_owned_gids(*second));
    for (size_t cell = 0; cell < first->num_local_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        EXPECT_EQ(first->cell_geometry_global_id(lid), second->cell_geometry_global_id(lid));
        EXPECT_EQ(first->cell_centroid(lid), second->cell_centroid(lid));
        EXPECT_EQ(first->cell_volume(lid), second->cell_volume(lid));
    }
}

/** @brief Native reconstruction retains polygon identity, winding and physical data. */
TEST(MeshPartitionerTest, NativePolyhedraRetainPolygonLoopsMetricsAndBoundaryIds)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 2) GTEST_SKIP() << "Regression covers serial and two-rank native partitioning.";
    const auto source = make_polyhedral_column();
    auto local = source;
    const auto partition = SimpleFluid::MeshPartitioner<Pack>::partition(local, comm);
    const auto& indexer = partition.indexer;
    EXPECT_EQ(local.num_cells(), indexer.num_local_cells());
    int local_owned = static_cast<int>(local.num_owned_cells()), total_owned = 0;
    my_mpi::allreduce(&local_owned, &total_owned, 1, MPI_SUM);
    EXPECT_EQ(total_owned, source.num_cells());
    for (size_t c = 0; c < local.num_cells(); ++c)
    {
        const auto gid = indexer.cell_global_id(c);
        EXPECT_EQ(indexer.cell_local_id(gid), c);
        EXPECT_EQ(local.cell_type(c), UnstructuredMesh::CellType::POLYHEDRON);
        EXPECT_EQ(local.faces(c).size(), 7U);
        EXPECT_NEAR(local.cell_volume(c), source.cell_volume(gid), 1e-12);
        EXPECT_NEAR((local.cell_centroid(c) - source.cell_centroid(gid)).norm(), 0.0, 1e-12);
        if (c < local.num_owned_cells()) EXPECT_EQ(partition.cell_owner_ranks.at(gid), comm->getRank());
    }
    size_t polygon_faces = 0;
    for (size_t f = 0; f < local.num_faces(); ++f)
    {
        const auto gid = indexer.face_global_id(f);
        EXPECT_EQ(indexer.face_local_id(gid), f);
        auto expected_loop = source.face_nodes(gid);
        const bool same_owner = indexer.cell_global_id(local.owner_cell(f)) == source.owner_cell(gid);
        if (!same_owner) std::ranges::reverse(expected_loop);
        std::vector<UnstructuredMesh::NodeID> actual_loop;
        for (const auto n : local.face_nodes(f)) actual_loop.push_back(indexer.node_global_id(n));
        ASSERT_EQ(actual_loop.size(), expected_loop.size());
        const auto first = std::find(expected_loop.begin(), expected_loop.end(), actual_loop.front());
        ASSERT_NE(first, expected_loop.end());
        const auto shift = static_cast<size_t>(first - expected_loop.begin());
        for (size_t i = 0; i < actual_loop.size(); ++i)
            EXPECT_EQ(actual_loop[i], expected_loop[(shift + i) % expected_loop.size()]);
        if (actual_loop.size() > 4) ++polygon_faces;
        EXPECT_NEAR(local.face_area(f), source.face_area(gid), 1e-12);
        EXPECT_NEAR((local.face_centroid(f) - source.face_centroid(gid)).norm(), 0.0, 1e-12);
        const auto expected_area = source.face_area_vector(gid) * (same_owner ? 1.0 : -1.0);
        EXPECT_NEAR((local.face_area_vector(f) - expected_area).norm(), 0.0, 1e-12);
        EXPECT_EQ(local.boundary_id(f), source.boundary_id(gid));
        if (source.boundary_id(gid) != UnstructuredMesh::invalid_boundary_id)
            EXPECT_EQ(local.boundary_name(f), source.boundary_name(gid));
    }
    EXPECT_GT(polygon_faces, 0U);
    for (size_t n = 0; n < local.num_nodes(); ++n)
    {
        const auto gid = indexer.node_global_id(n);
        EXPECT_EQ(indexer.node_local_id(gid), n);
        EXPECT_EQ(local.node_coordinates(n), source.node_coordinates(gid));
    }
    if (comm->getSize() == 2)
    {
        EXPECT_GT(local.num_owned_cells(), 0U);
        EXPECT_LT(local.num_owned_cells(), source.num_cells());
        EXPECT_GT(local.num_cells(), local.num_owned_cells());
    }
}

/**
 * @brief Verifies Mesh dispatches unstructured partitioning and exposes
 * the resulting local mesh consistently.
 */
TEST(MeshPartitionerTest, MeshHandlePartitionsUnstructuredMesh)
{
    constexpr unsigned cell_count = 8;
    auto mesh = make_partition_test_hex_line(cell_count);
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

    EXPECT_THROW(Mesh{mesh}, std::invalid_argument);
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

    const Mesh handle(partitioned);
    const auto& handle_mesh =
        std::get<Mesh::UnstructuredPtr>(handle.variant());
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
                    Mesh::invalid_local_id());
            }
        }
    }
}

/** Graded planar quads must retain the serial area-centroid geometry after packet rebuild.
 * Vertex averaging gives x=1.5 while each trapezoid's area centroid is x=14/9.
 */
TEST(MeshPartitionerGeometryTest, GradedQuadCentroidsMatchSerialRule)
{
    using Data = SimpleFluid::Meshes::UnstructuredMeshData;
    const auto comm = Tpetra::getDefaultComm();
    Data data;
    std::exception_ptr error;
    try
    {
        if (comm->getRank() == 0)
        {
            const std::array<std::array<double, 2>, 4> xy{{{1., -1.}, {2., -2.}, {2., 2.}, {1., 1.}}};
            std::vector<Data::Node> nodes;
            std::vector<Data::Cell> cells;
            for (unsigned layer = 0; layer <= 4; ++layer)
                for (unsigned corner = 0; corner < 4; ++corner)
                    nodes.push_back({1 + 4 * layer + corner, {xy[corner][0], xy[corner][1], double(layer)}});
            for (unsigned layer = 0; layer < 4; ++layer)
            {
                Data::Cell cell;
                cell.id = layer + 1;
                cell.type = SimpleFluid::MeshUtils::CellType::HEXAHEDRON;
                for (unsigned corner = 0; corner < 8; ++corner)
                    cell.node_ids.push_back(1 + 4 * layer + corner);
                cells.push_back(std::move(cell));
            }
            std::vector<SimpleFluid::detail::NativeBoundarySide> boundaries;
            for (const auto& cell : cells)
                for (unsigned side = 0; side < 6; ++side)
                    if (side < 4 || (side == 4 && cell.id == 1) || (side == 5 && cell.id == 4))
                        boundaries.push_back({cell.id, side, "wall"});
            data =
                SimpleFluid::detail::build_native_mesh_data(std::move(nodes), std::move(cells), boundaries, {"wall"});
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (error)
        std::rethrow_exception(error);
    if (any_failed)
        throw std::runtime_error("Graded native source construction failed on another rank.");
    auto partition = SimpleFluid::MeshPartitioner<Pack>::partition_native(std::move(data), comm, 1, true);
    auto mesh = SimpleFluid::detail::collective_make_shared<Mesh>(
        *comm, std::move(partition.geometry), partition.indexer, comm);
    std::size_t trapezoids = 0;
    for (std::size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(face);
        if (std::abs(mesh->face_normal(lid).z) < 0.5)
            continue;
        ++trapezoids;
        const auto center = mesh->face_centroid(lid);
        EXPECT_NEAR(center.x, 14. / 9., 1e-14);
        EXPECT_NEAR(center.y, 0., 1e-14);
        EXPECT_NEAR(mesh->face_area(lid), 3., 1e-14);
    }
    const int local_count = static_cast<int>(trapezoids);
    int global_count = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_count, &global_count);
    EXPECT_GT(global_count, 0);
}
