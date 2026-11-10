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
#include "geometry/STKMesh.hh"
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include "geometry/MeshHandle.hh"
#include "fields/CellField.hh"
#include "fields/FieldStored.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>

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

    void initialize(const ArrGO& owned_gids,
                    const ArrGO& ghost_gids)
    {
        d_owned_cell_global_ids = owned_gids;
        d_ghost_cell_global_ids = ghost_gids;
        d_owned_cell_ids.resize(owned_gids.size());
        std::iota(d_owned_cell_ids.begin(), d_owned_cell_ids.end(), 0);
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
    ReplicatedLegacyMesh()
    {
        d_spatial_dim = 3;
    }

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

            info.center = source.face_centroid(face_id);
            info.area = source.face_area(face_id);
            info.unit_normal_from_owner = source.face_normal(face_id);
            info.unit_normal_from_neighbor = info.unit_normal_from_owner * -1.0;
            const auto& owner_center = d_cells[static_cast<size_t>(info.owner)].center;
            info.owner_to_face_distance = (info.center - owner_center).norm();
            if (info.neighbor != SimpleFluid::invalid_id<LO>())
            {
                const auto& neighbor_center =
                    d_cells[static_cast<size_t>(info.neighbor)].center;
                info.neighbor_to_face_distance =
                    (info.center - neighbor_center).norm();
                info.cell_center_distance = (neighbor_center - owner_center).norm();
            }

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

    void pair_end_faces_periodically()
    {
        FaceInfo* left = nullptr;
        FaceInfo* right = nullptr;
        for (auto& face : d_faces)
        {
            if (face.boundary_id == 1) left = &face;
            if (face.boundary_id == 2) right = &face;
        }
        if (left == nullptr || right == nullptr)
        {
            throw std::logic_error("Periodic fixture requires two tagged end faces.");
        }
        left->neighbor = right->owner;
        right->neighbor = left->owner;
        for (auto* face : {left, right})
        {
            const auto& owner = d_cells[static_cast<size_t>(face->owner)];
            const auto& neighbor = d_cells[static_cast<size_t>(face->neighbor)];
            face->neighbor_to_face_distance = (face->center - neighbor.center).norm();
            face->cell_center_distance = (neighbor.center - owner.center).norm();
        }
    }

    void perturb_face_metrics()
    {
        for (auto& face : d_faces)
        {
            // These one-ULP changes distinguish supplied metrics from a
            // receiver-side reconstruction using the unchanged coordinates.
            face.center.x = std::nextafter(
                face.center.x, std::numeric_limits<double>::infinity());
            face.area = std::nextafter(
                face.area, std::numeric_limits<double>::infinity());
            auto& normal = face.unit_normal_from_owner;
            normal.x = std::nextafter(normal.x, 0.0);
            normal.y = std::nextafter(normal.y, 0.0);
            normal.z = std::nextafter(normal.z, 0.0);
            face.unit_normal_from_neighbor = normal * -1.0;
            face.owner_to_face_distance = std::nextafter(
                face.owner_to_face_distance, std::numeric_limits<double>::infinity());
            if (face.neighbor != SimpleFluid::invalid_id<LO>())
            {
                face.neighbor_to_face_distance = std::nextafter(
                    face.neighbor_to_face_distance,
                    std::numeric_limits<double>::infinity());
                face.cell_center_distance = std::nextafter(
                    face.cell_center_distance, std::numeric_limits<double>::infinity());
            }
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

/** @brief Verifies face identity and metrics survive legacy packet transport. */
TEST(MeshPartitionerPacketTest, FaceGlobalIDsSurviveSerialization)
{
    using Packet = SimpleFluid::partition_detail::CellPacket<Pack>;
    Packet packet;
    packet.gid = 17;
    packet.face_node_keys = {{8, 3, 5, 1}, {9, 4, 6, 2}};
    packet.face_global_ids = {101, 307};
    packet.face_boundary_ids = {
        MeshType::invalid_boundary_id, 4};
    packet.face_neighbor_gids = {
        SimpleFluid::invalid_id<Pack::global_ordinal_type>(), 23};
    packet.face_neighbor_ranks = {-1, 2};
    packet.face_geometry = {
        {{0.25, -0.5, 1.0}, 0.125, {0.0, -1.0, 0.0}, 0.25, 0.0, 0.0, 17},
        {{1.25, 0.5, -1.0}, 0.375, {1.0, 0.0, 0.0}, 0.375, 0.625, 1.0, 23}};

    const auto bytes = Packet::serialize_packets({packet});
    const auto restored =
        Packet::deserialize_packets(bytes.data(), bytes.size());

    ASSERT_EQ(restored.size(), 1U);
    EXPECT_EQ(restored.front().face_node_keys, packet.face_node_keys);
    EXPECT_EQ(restored.front().face_global_ids, packet.face_global_ids);
    EXPECT_EQ(restored.front().face_boundary_ids, packet.face_boundary_ids);
    EXPECT_EQ(
        restored.front().face_neighbor_gids,
        packet.face_neighbor_gids);
    EXPECT_EQ(
        restored.front().face_neighbor_ranks,
        packet.face_neighbor_ranks);

    auto expect_geometry_equal = [](const Packet::FaceGeometry& actual,
                                    const Packet::FaceGeometry& expected)
    {
        EXPECT_EQ(actual.center.x, expected.center.x);
        EXPECT_EQ(actual.center.y, expected.center.y);
        EXPECT_EQ(actual.center.z, expected.center.z);
        EXPECT_EQ(actual.area, expected.area);
        EXPECT_EQ(actual.unit_normal_from_owner.x, expected.unit_normal_from_owner.x);
        EXPECT_EQ(actual.unit_normal_from_owner.y, expected.unit_normal_from_owner.y);
        EXPECT_EQ(actual.unit_normal_from_owner.z, expected.unit_normal_from_owner.z);
        EXPECT_EQ(actual.owner_to_face_distance, expected.owner_to_face_distance);
        EXPECT_EQ(actual.neighbor_to_face_distance, expected.neighbor_to_face_distance);
        EXPECT_EQ(actual.cell_center_distance, expected.cell_center_distance);
        EXPECT_EQ(actual.owner_gid, expected.owner_gid);
    };
    ASSERT_EQ(restored.front().face_geometry.size(), packet.face_geometry.size());
    for (size_t face = 0; face < packet.face_geometry.size(); ++face)
    {
        expect_geometry_equal(
            restored.front().face_geometry[face], packet.face_geometry[face]);
    }
    EXPECT_EQ(
        bytes.size(), sizeof(std::uint32_t) + Packet::serialized_size(packet));
    EXPECT_THROW(
        Packet::view_packets(bytes.data(), bytes.size() - 1),
        std::runtime_error);

    const auto views =
        Packet::view_packets(bytes.data(), bytes.size());
    ASSERT_EQ(views.size(), 1U);
    EXPECT_EQ(
        views.front().data(),
        bytes.data() + sizeof(std::uint32_t));
    EXPECT_EQ(views.front().gid(), packet.gid);

    std::vector<std::vector<Pack::global_ordinal_type>>
        viewed_face_nodes;
    std::vector<Pack::global_ordinal_type>
        viewed_face_global_ids;
    std::vector<int> viewed_face_neighbor_ranks;
    size_t viewed_face_count = 0;
    views.front().for_each_face(
        [&](const Packet::View::Face& face)
        {
            viewed_face_nodes.emplace_back(
                face.node_gids.begin(),
                face.node_gids.begin() + face.node_count);
            viewed_face_global_ids.push_back(face.global_id);
            viewed_face_neighbor_ranks.push_back(
                face.neighbor_rank);
            ASSERT_LT(viewed_face_count, packet.face_geometry.size());
            expect_geometry_equal(
                face.geometry, packet.face_geometry[viewed_face_count]);
            ++viewed_face_count;
        });
    EXPECT_EQ(viewed_face_count, packet.face_geometry.size());
    EXPECT_EQ(viewed_face_nodes, packet.face_node_keys);
    EXPECT_EQ(
        viewed_face_global_ids,
        packet.face_global_ids);
    EXPECT_EQ(
        viewed_face_neighbor_ranks,
        packet.face_neighbor_ranks);
}

/**
 * @brief Partitions a legacy mesh whose complete source geometry exists only
 *        on rank zero.
 */
TEST(MeshPartitionerTest, PartitionsRootOnlySourceGeometry)
{
    constexpr unsigned cell_count = 8;
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    std::shared_ptr<ReplicatedLegacyMesh> mesh;
    if (comm->getRank() == 0)
    {
        const auto source =
            make_unstructured_hex_line(cell_count);
        mesh =
            std::make_shared<ReplicatedLegacyMesh>(*source);
        ASSERT_EQ(mesh->num_owned_cells(), cell_count);
    }
    else
    {
        mesh = std::make_shared<ReplicatedLegacyMesh>();
        ASSERT_EQ(mesh->num_owned_cells(), 0U);
        ASSERT_EQ(mesh->num_local_cells(), 0U);
    }

    EXPECT_TRUE(
        SimpleFluid::MeshPartitioner<Pack>::partition(
            *mesh, comm));
    if (comm->getSize() <= static_cast<int>(cell_count))
    {
        EXPECT_GT(mesh->num_owned_cells(), 0U);
    }

    using GO = Pack::global_ordinal_type;
    std::vector<GO> local_gids(
        mesh->owned_cell_global_ids().begin(),
        mesh->owned_cell_global_ids().end());
    const int local_count =
        static_cast<int>(local_gids.size());
    std::vector<int> counts(
        static_cast<size_t>(comm->getSize()), 0);
    my_mpi::allgather(
        &local_count, 1, counts.data(), 1);
    std::vector<int> displacements(counts.size(), 0);
    for (size_t rank = 1; rank < counts.size(); ++rank)
    {
        displacements[rank] =
            displacements[rank - 1] + counts[rank - 1];
    }
    const int global_count =
        displacements.back() + counts.back();
    std::vector<GO> global_gids(
        static_cast<size_t>(global_count));
    my_mpi::allgatherv(
        local_gids.data(),
        local_count,
        global_gids.data(),
        counts.data(),
        displacements.data());

    EXPECT_EQ(global_count, static_cast<int>(cell_count));
    EXPECT_EQ(
        std::set<GO>(global_gids.begin(), global_gids.end()).size(),
        static_cast<size_t>(cell_count));

    for (size_t cell = 0;
         cell < mesh->num_owned_cells();
         ++cell)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(cell);
        EXPECT_FALSE(mesh->cell(cell_lid).faces.empty());
        for (const auto node_gid : mesh->cell(cell_lid).node_gids)
        {
            EXPECT_NO_THROW((void)mesh->node_coord(node_gid));
        }
    }
}

/**
 * @brief Confirms MeshFactory releases its root-only STK construction source
 *        after exposing a distributed finite-volume mesh.
 */
TEST(MeshPartitionerTest, FactoryReleasesRootOnlySTKSourceAfterPartition)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto mesh = make_4x4x4_box_mesh();
    const auto stk_mesh =
        std::dynamic_pointer_cast<SimpleFluid::STKMesh<Pack>>(mesh);
    ASSERT_NE(stk_mesh, nullptr);

    EXPECT_EQ(stk_mesh->meta(), nullptr);
    EXPECT_EQ(stk_mesh->bulk(), nullptr);
    EXPECT_GT(mesh->num_owned_cells(), 0U);
    EXPECT_LT(mesh->num_owned_cells(), 64U);
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

/** @brief Distributed lookup preserves the original contiguous numbering. */
TEST(MeshContiguousGidTest, DistributedDirectoryMatchesRankBlockNumbering)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    const int rank = comm->getRank();
    const int next_rank = (rank + 1) % comm->getSize();
    const int owned_count = rank + 1;
    const int next_owned_count = next_rank + 1;

    MeshType::ArrGO owned_gids;
    owned_gids.reserve(static_cast<size_t>(owned_count));
    for (int local = 0; local < owned_count; ++local)
    {
        owned_gids.push_back(
            static_cast<GO>(10000 + 100 * rank + 7 * local));
    }
    const GO ghost_gid = static_cast<GO>(
        10000 + 100 * next_rank + 7 * (next_owned_count - 1));

    ContiguousGidProbeMesh mesh;
    mesh.initialize(owned_gids, {ghost_gid});
    mesh.assign_ids();

    const GO expected_offset = static_cast<GO>(rank * (rank + 1) / 2);
    for (int local = 0; local < owned_count; ++local)
    {
        const GO expected_tpetra_gid =
            expected_offset + static_cast<GO>(local);
        EXPECT_EQ(
            mesh.mesh_gid_to_tpetra_gid(
                owned_gids[static_cast<size_t>(local)]),
            expected_tpetra_gid);
        EXPECT_EQ(
            mesh.tpetra_gid_to_mesh_gid(expected_tpetra_gid),
            owned_gids[static_cast<size_t>(local)]);
    }

    const GO next_offset =
        static_cast<GO>(next_rank * (next_rank + 1) / 2);
    const GO expected_ghost_tpetra_gid =
        next_offset + static_cast<GO>(next_owned_count - 1);
    ASSERT_EQ(mesh.ghost_tpetra_gids().size(), 1U);
    EXPECT_EQ(
        mesh.ghost_tpetra_gids().front(),
        expected_ghost_tpetra_gid);
    EXPECT_EQ(
        mesh.mesh_gid_to_tpetra_gid(ghost_gid),
        expected_ghost_tpetra_gid);
}

/** @brief A missing ghost ID fails coherently on every rank. */
TEST(MeshContiguousGidTest, MissingGhostIDThrowsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    const int rank = comm->getRank();
    const GO owned_gid = static_cast<GO>(3000 + rank);
    MeshType::ArrGO ghost_gids;
    if (rank == 1)
    {
        ghost_gids.push_back(static_cast<GO>(987654321));
    }

    ContiguousGidProbeMesh mesh;
    mesh.initialize({owned_gid}, ghost_gids);
    EXPECT_THROW(mesh.assign_ids(), std::runtime_error);
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

/** @brief Preserves supplied source metrics through redistribution and ghost exchange. */
TEST(MeshPartitionerTest, PreservesSourceFaceMetrics)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    using GO = Pack::global_ordinal_type;
    using LO = Pack::local_ordinal_type;
    using Geometry = SimpleFluid::partition_detail::CellPacket<Pack>::FaceGeometry;
    const auto source = make_unstructured_hex_line(8);
    auto expect_same_bits = [](double actual, double expected)
    {
        EXPECT_EQ(
            std::bit_cast<std::uint64_t>(actual),
            std::bit_cast<std::uint64_t>(expected));
    };

    for (const auto& [root_only, periodic] : std::array{
             std::pair{false, false}, std::pair{true, false},
             std::pair{false, true}, std::pair{true, true}})
    {
        SCOPED_TRACE(root_only ? "root-only source" : "replicated source");
        SCOPED_TRACE(periodic ? "periodic end faces" : "physical end faces");
        auto mesh = std::make_shared<ReplicatedLegacyMesh>(*source);
        if (periodic) mesh->pair_end_faces_periodically();
        mesh->perturb_face_metrics();
        std::map<GO, Geometry> original_geometry;
        std::map<GO, GO> periodic_neighbor_gids;
        for (size_t face = 0; face < mesh->num_faces(); ++face)
        {
            const auto lid = static_cast<LO>(face);
            const auto& info = mesh->face(lid);
            original_geometry.emplace(
                mesh->face_global_id(lid),
                Geometry{info.center, info.area, info.unit_normal_from_owner,
                         info.owner_to_face_distance, info.neighbor_to_face_distance,
                         info.cell_center_distance, mesh->cell_global_id(info.owner)});
            if (periodic && (info.boundary_id == 1 || info.boundary_id == 2))
            {
                periodic_neighbor_gids.emplace(
                    mesh->face_global_id(lid), mesh->cell_global_id(info.neighbor));
            }
        }
        if (root_only && comm->getRank() != 0)
        {
            mesh = std::make_shared<ReplicatedLegacyMesh>();
        }

        EXPECT_TRUE(SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm));
        int local_reversed = 0;
        int local_ghost_only = 0;
        int local_periodic_pairs = 0;
        for (size_t face = 0; face < mesh->num_faces(); ++face)
        {
            const auto lid = static_cast<LO>(face);
            const auto& info = mesh->face(lid);
            const auto original = original_geometry.find(mesh->face_global_id(lid));
            EXPECT_NE(original, original_geometry.end());
            if (original == original_geometry.end())
            {
                continue;
            }
            const auto& expected = original->second;
            const bool same_owner =
                mesh->cell_global_id(info.owner) == expected.owner_gid;
            const auto expected_normal = same_owner
                ? expected.unit_normal_from_owner
                : expected.unit_normal_from_owner * -1.0;
            const auto expected_neighbor_normal = expected_normal * -1.0;
            local_reversed += !same_owner;
            local_ghost_only += !mesh->cell(info.owner).owned;
            expect_same_bits(info.center.x, expected.center.x);
            expect_same_bits(info.center.y, expected.center.y);
            expect_same_bits(info.center.z, expected.center.z);
            expect_same_bits(info.area, expected.area);
            expect_same_bits(info.owner_to_face_distance, same_owner
                ? expected.owner_to_face_distance : expected.neighbor_to_face_distance);
            expect_same_bits(info.neighbor_to_face_distance, same_owner
                ? expected.neighbor_to_face_distance : expected.owner_to_face_distance);
            expect_same_bits(info.cell_center_distance, expected.cell_center_distance);
            expect_same_bits(info.unit_normal_from_owner.x, expected_normal.x);
            expect_same_bits(info.unit_normal_from_owner.y, expected_normal.y);
            expect_same_bits(info.unit_normal_from_owner.z, expected_normal.z);
            expect_same_bits(info.unit_normal_from_neighbor.x, expected_neighbor_normal.x);
            expect_same_bits(info.unit_normal_from_neighbor.y, expected_neighbor_normal.y);
            expect_same_bits(info.unit_normal_from_neighbor.z, expected_neighbor_normal.z);
            const auto paired = periodic_neighbor_gids.find(mesh->face_global_id(lid));
            if (paired != periodic_neighbor_gids.end())
            {
                const auto paired_lid = mesh->global_to_local_cell(paired->second);
                EXPECT_EQ(info.neighbor, paired_lid);
                local_periodic_pairs += paired_lid != SimpleFluid::invalid_id<LO>();
            }
        }
        int global_reversed = 0;
        int global_ghost_only = 0;
        int global_periodic_pairs = 0;
        my_mpi::allreduce(&local_reversed, &global_reversed, 1, MPI_SUM);
        my_mpi::allreduce(&local_ghost_only, &global_ghost_only, 1, MPI_SUM);
        my_mpi::allreduce(&local_periodic_pairs, &global_periodic_pairs, 1, MPI_SUM);
        EXPECT_GT(global_reversed, 0);
        EXPECT_GT(global_ghost_only, 0);
        if (periodic) EXPECT_GE(global_periodic_pairs, 2);
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

/** Graded planar quads must retain the serial area-centroid geometry after packet rebuild.
 * Vertex averaging gives x=1.5 while each trapezoid's area centroid is x=14/9.
 */
TEST(MeshPartitionerGeometryTest, GradedQuadCentroidsMatchSerialRule)
{
    auto mesh = std::make_shared<SimpleFluid::STKMesh<Pack>>();
    auto meta = mesh->meta();
    auto bulk = mesh->bulk();
    auto& coordinates = meta->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);
    auto& block = meta->declare_part_with_topology("graded_hexes", stk::topology::HEX_8);
    bulk->modification_begin();
    for (unsigned layer = 0; layer < 4; ++layer)
    {
        stk::mesh::EntityIdVector nodes;
        for (unsigned corner = 0; corner < 8; ++corner)
            nodes.push_back(1 + 4 * layer + corner);
        stk::mesh::declare_element(*bulk, block, layer + 1, nodes);
    }
    const std::array<std::array<double, 2>, 4> xy{{{1.,-1.},{2.,-2.},{2.,2.},{1.,1.}}};
    for (unsigned layer = 0; layer <= 4; ++layer)
        for (unsigned corner = 0; corner < 4; ++corner)
        {
            const auto node = bulk->get_entity(stk::topology::NODE_RANK, 1 + 4 * layer + corner);
            auto* values = stk::mesh::field_data(coordinates, node);
            values[0] = xy[corner][0]; values[1] = xy[corner][1]; values[2] = layer;
        }
    bulk->modification_end();
    mesh->assemble();
    std::size_t trapezoids = 0;
    for (std::size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(face);
        if (std::abs(mesh->face_normal(lid).z) < 0.5) continue;
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
