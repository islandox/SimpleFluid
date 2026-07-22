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

#include "geometry/Mesh.hh"
#include "geometry/MeshHandle.hh"
#include "fields/CellField.hh"
#include "fields/FieldStored.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cstddef>
#include <map>
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

/** @brief Minimal mesh exposing contiguous-ID assignment for MPI tests. */
class ContiguousGidProbeMesh final : public MeshType
{
public:
    using GO = Pack::global_ordinal_type;

    void initialize(bool owns_cell,
                    GO owned_gid,
                    bool has_ghost,
                    GO ghost_gid)
    {
        if (owns_cell)
        {
            d_owned_cell_ids.push_back(0);
            d_owned_cell_global_ids.push_back(owned_gid);
        }
        if (has_ghost)
        {
            d_ghost_cell_global_ids.push_back(ghost_gid);
        }
    }

    void assign_ids()
    {
        assign_contiguous_tpetra_gids();
    }

    const ArrGO& ghost_tpetra_gids() const noexcept
    {
        return d_ghost_cell_tpetra_gids;
    }

    void assemble() override {}
    void export_vtu(const std::string&) const override {}
};

/**
 * @brief Replicated legacy mesh assembled from CRTP geometry without maps.
 *
 * Unlike MeshFactory's STK mesh, this fixture remains unpartitioned until a
 * test explicitly calls MeshPartitioner::partition().
 */
class ReplicatedLegacyMesh final : public MeshType
{
public:
    explicit ReplicatedLegacyMesh(const UnstructuredMesh& source)
    {
        d_spatial_dim = static_cast<int>(source.spatial_dimension());
        d_node_coords = source.nodes();
        for (size_t node = 0; node < source.num_nodes(); ++node)
        {
            const auto node_gid = static_cast<GO>(source.node_id(node));
            d_node_gid_to_lid.emplace(
                node_gid, static_cast<LO>(node));
        }

        size_t total_cell_nodes = 0;
        size_t total_cell_faces = 0;
        for (size_t cell = 0; cell < source.num_cells(); ++cell)
        {
            const auto cell_id = source.cell_id(cell);
            total_cell_nodes += source.cell_nodes(cell_id).size();
            total_cell_faces += source.faces(cell_id).size();
        }
        d_cells.reserve(source.num_cells());
        d_owned_cell_ids.reserve(source.num_cells());
        d_owned_cell_global_ids.reserve(source.num_cells());
        d_cell_owned_node_global_ids.reserve(total_cell_nodes);
        d_cell_owned_face_ids.reserve(total_cell_faces);

        for (size_t cell = 0; cell < source.num_cells(); ++cell)
        {
            const auto cell_id = source.cell_id(cell);
            const auto cell_lid = static_cast<LO>(cell);
            CellInfo info;
            info.owned = true;
            info.type = source.cell_type(cell_id);
            info.center = source.cell_centroid(cell_id);
            info.volume = source.cell_volume(cell_id);

            const auto node_offset = d_cell_owned_node_global_ids.size();
            for (const auto node_id : source.cell_nodes(cell_id))
            {
                d_cell_owned_node_global_ids.push_back(
                    static_cast<GO>(node_id));
            }
            info.node_gids = ViewGO(
                d_cell_owned_node_global_ids.data() + node_offset,
                source.cell_nodes(cell_id).size());

            const auto face_offset = d_cell_owned_face_ids.size();
            for (const auto face_id : source.faces(cell_id))
            {
                d_cell_owned_face_ids.push_back(
                    static_cast<LO>(source.face_local_id(face_id)));
            }
            info.faces = ViewLO(
                d_cell_owned_face_ids.data() + face_offset,
                source.faces(cell_id).size());

            const auto cell_gid = static_cast<GO>(cell_id);
            d_cell_gid_to_lid.emplace(cell_gid, cell_lid);
            d_owned_cell_ids.push_back(cell_lid);
            d_owned_cell_global_ids.push_back(cell_gid);
            d_cells.push_back(std::move(info));
        }

        size_t total_face_nodes = 0;
        for (size_t face = 0; face < source.num_faces(); ++face)
        {
            total_face_nodes +=
                source.face_nodes(source.face_id(face)).size();
        }
        d_faces.reserve(source.num_faces());
        d_owned_face_global_ids.reserve(source.num_faces());
        d_face_owned_node_global_ids.reserve(total_face_nodes);
        for (size_t face = 0; face < source.num_faces(); ++face)
        {
            const auto face_id = source.face_id(face);
            const auto& node_ids = source.face_nodes(face_id);
            FaceInfo info;
            info.type = node_ids.size() == 3
                      ? FaceType::TRIANGLE
                      : FaceType::QUAD;
            info.boundary_id = source.boundary_id(face_id);
            info.owner = static_cast<LO>(
                source.cell_local_id(source.owner_cell(face_id)));
            const auto neighbor = source.neighbor_cell(face_id);
            info.neighbor = neighbor == UnstructuredMesh::invalid_cell_id()
                          ? SimpleFluid::invalid_id<LO>()
                          : static_cast<LO>(source.cell_local_id(neighbor));

            const auto node_offset = d_face_owned_node_global_ids.size();
            for (const auto node_id : node_ids)
            {
                d_face_owned_node_global_ids.push_back(
                    static_cast<GO>(node_id));
            }
            info.node_gids = ViewGO(
                d_face_owned_node_global_ids.data() + node_offset,
                node_ids.size());

            d_faces.push_back(std::move(info));
            d_owned_face_global_ids.push_back(
                static_cast<GO>(10000 + 17 * face));
        }
    }

    void assemble() override {}
    void export_vtu(const std::string&) const override {}

private:
    using GO = Pack::global_ordinal_type;
    using LO = Pack::local_ordinal_type;
};

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

/** @brief Verifies face identity survives legacy cell-packet transport. */
TEST(MeshPartitionerPacketTest, FaceGlobalIDsSurviveSerialization)
{
    using Packet = SimpleFluid::partition_detail::CellPacket<Pack>;
    Packet packet;
    packet.gid = 17;
    packet.face_node_keys = {{8, 3, 5, 1}, {9, 4, 6, 2}};
    packet.face_global_ids = {101, 307};
    packet.face_boundary_ids = {
        MeshType::invalid_boundary_id, 4};

    const auto bytes = Packet::serialize_packets({packet});
    const auto restored =
        Packet::deserialize_packets(bytes.data(), bytes.size());

    ASSERT_EQ(restored.size(), 1U);
    EXPECT_EQ(restored.front().face_node_keys, packet.face_node_keys);
    EXPECT_EQ(restored.front().face_global_ids, packet.face_global_ids);
    EXPECT_EQ(restored.front().face_boundary_ids, packet.face_boundary_ids);
}

/** @brief Keeps owner resolution collective when only some ranks have ghosts. */
TEST(MeshContiguousGidTest, MixedZeroAndNonzeroGhostRanksRemainCoherent)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    const int rank = comm->getRank();
    const bool has_ghost = rank == 1;
    const GO owned_gid = static_cast<GO>(1000 + rank);
    constexpr GO ghost_gid = 1000;

    ContiguousGidProbeMesh mesh;
    mesh.initialize(true, owned_gid, has_ghost, ghost_gid);
    mesh.assign_ids();

    const GO expected_owned_tpetra_gid = static_cast<GO>(rank);
    EXPECT_EQ(
        mesh.mesh_gid_to_tpetra_gid(owned_gid),
        expected_owned_tpetra_gid);
    EXPECT_EQ(
        mesh.tpetra_gid_to_mesh_gid(expected_owned_tpetra_gid),
        owned_gid);
    if (has_ghost)
    {
        ASSERT_EQ(mesh.ghost_tpetra_gids().size(), 1U);
        EXPECT_EQ(mesh.ghost_tpetra_gids().front(), 0);
        EXPECT_EQ(mesh.mesh_gid_to_tpetra_gid(ghost_gid), 0);
    }
    else
    {
        EXPECT_TRUE(mesh.ghost_tpetra_gids().empty());
    }
}

/** @brief Repeated assignment is rank coherent when some ranks own no cells. */
TEST(MeshContiguousGidTest, ZeroOwnedRanksRemainIdempotent)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    const int rank = comm->getRank();
    constexpr GO owned_gid = 2000;

    ContiguousGidProbeMesh mesh;
    mesh.initialize(rank == 0, owned_gid, rank == 1, owned_gid);
    mesh.assign_ids();
    mesh.assign_ids();

    if (rank == 0)
    {
        EXPECT_EQ(mesh.mesh_gid_to_tpetra_gid(owned_gid), 0);
        EXPECT_EQ(mesh.tpetra_gid_to_mesh_gid(0), owned_gid);
    }
    else if (rank == 1)
    {
        ASSERT_EQ(mesh.ghost_tpetra_gids().size(), 1U);
        EXPECT_EQ(mesh.ghost_tpetra_gids().front(), 0);
        EXPECT_EQ(mesh.mesh_gid_to_tpetra_gid(owned_gid), 0);
    }
    else
    {
        EXPECT_TRUE(mesh.ghost_tpetra_gids().empty());
    }
}

/** @brief Verifies STK assigns a distinct identity to every local face. */
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

/**
 * @brief Verifies rebuilt local faces retain a rank-independent global ID.
 *
 * A face may be present on multiple ranks as owned or overlap geometry. Its
 * sorted global-node key and global face ID must remain a one-to-one mapping
 * after the legacy packet-based rebuild reorders local faces.
 */
TEST(MeshPartitionerTest, PreservesGlobalFaceIdentity)
{
    constexpr unsigned cell_count = 8;
    const auto source = make_unstructured_hex_line(cell_count);
    auto mesh = std::make_shared<ReplicatedLegacyMesh>(*source);
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    constexpr size_t nodes_per_face = 4;
    constexpr size_t record_size = nodes_per_face + 1;
    std::map<std::vector<GO>, GO> original_global_id_by_node_key;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        std::vector<GO> node_key(
            mesh->face(face_lid).node_gids.begin(),
            mesh->face(face_lid).node_gids.end());
        std::sort(node_key.begin(), node_key.end());
        original_global_id_by_node_key.emplace(
            std::move(node_key), mesh->face_global_id(face_lid));
    }

    EXPECT_TRUE(SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm));
    EXPECT_LT(mesh->num_owned_cells(), static_cast<size_t>(cell_count));

    std::vector<GO> local_records;
    local_records.reserve(mesh->num_faces() * record_size);
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        std::vector<GO> node_key(
            mesh->face(face_lid).node_gids.begin(),
            mesh->face(face_lid).node_gids.end());
        EXPECT_EQ(node_key.size(), nodes_per_face);
        if (node_key.size() != nodes_per_face)
        {
            continue;
        }
        std::sort(node_key.begin(), node_key.end());

        const auto original_id =
            original_global_id_by_node_key.find(node_key);
        EXPECT_NE(original_id, original_global_id_by_node_key.end());
        if (original_id == original_global_id_by_node_key.end())
        {
            continue;
        }
        const auto rebuilt_global_id = mesh->face_global_id(face_lid);
        EXPECT_EQ(rebuilt_global_id, original_id->second);
        local_records.push_back(rebuilt_global_id);
        local_records.insert(
            local_records.end(), node_key.begin(), node_key.end());
    }

    const int local_count = static_cast<int>(local_records.size());
    std::vector<int> counts(static_cast<size_t>(comm->getSize()), 0);
    my_mpi::allgather(&local_count, 1, counts.data(), 1);
    std::vector<int> displacements(counts.size(), 0);
    for (size_t rank = 1; rank < counts.size(); ++rank)
    {
        displacements[rank] = displacements[rank - 1] + counts[rank - 1];
    }

    std::vector<GO> global_records(static_cast<size_t>(
        displacements.back() + counts.back()));
    my_mpi::allgatherv(
        local_records.data(),
        local_count,
        global_records.data(),
        counts.data(),
        displacements.data());

    ASSERT_EQ(global_records.size() % record_size, 0U);
    std::map<std::vector<GO>, GO> global_id_by_node_key;
    std::map<GO, std::vector<GO>> node_key_by_global_id;
    for (size_t offset = 0;
         offset < global_records.size();
         offset += record_size)
    {
        const GO global_id = global_records[offset];
        std::vector<GO> node_key(
            global_records.begin() + static_cast<std::ptrdiff_t>(offset + 1),
            global_records.begin()
                + static_cast<std::ptrdiff_t>(offset + record_size));

        const auto [key_iter, key_inserted] =
            global_id_by_node_key.emplace(node_key, global_id);
        EXPECT_EQ(key_iter->second, global_id)
            << "A physical face has different global IDs across ranks.";

        const auto [id_iter, id_inserted] =
            node_key_by_global_id.emplace(global_id, node_key);
        EXPECT_EQ(id_iter->second, node_key)
            << "A global face ID identifies different physical faces.";
        (void)key_inserted;
        (void)id_inserted;
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
/** @brief Verifies single-rank partitioning preserves the complete mesh. */
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
/** @brief Verifies repeated partitioning leaves the owned entity set unchanged. */
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

/**
 * @brief Verifies MeshHandle dispatches unstructured partitioning and exposes
 * the resulting local mesh consistently.
 */
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
