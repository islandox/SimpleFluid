/** @file testNativeMeshDistribution.cc
 *  @brief Native sparse distribution, identity, geometry, and collective failures.
 */
#include "geometry/MeshHandle.hh"
#include "parallel/MeshPartitioner.hh"
#include "parallel/NativeMeshDirectory.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>

namespace
{
using namespace SimpleFluid;
using MeshData = Meshes::UnstructuredMeshData;
using Geometry = Meshes::UnstructuredMesh;
using Partitioner = MeshPartitioner<DefaultTpetraTypes>;
using ID = MeshData::ID;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

// Sparse source IDs exceed local ordinals, while explicit algebraic ordinals
// are compact and independent. Geometry uses a native serial fixture only.
constexpr ID source_base = ID{1} << 40;
ID cell_id(size_t local)
{
    return source_base + 10001 + 41 * local;
}
ID face_id(size_t local)
{
    return source_base + 20001 + 43 * local;
}
ID node_id(size_t local)
{
    return source_base + 30001 + 47 * local;
}

MeshData chain(size_t count)
{
    Arr<Geometry::Vec3> points;
    for (size_t x = 0; x <= count; ++x)
    {
        points.push_back({static_cast<real_t>(x), 0, 0});
        points.push_back({static_cast<real_t>(x), 1, 0});
        points.push_back({static_cast<real_t>(x), 0, 1});
        points.push_back({static_cast<real_t>(x), 1, 1});
    }
    Arr<Geometry::CellDefinition> cells;
    for (size_t x = 0; x < count; ++x)
        cells.push_back({Geometry::CellType::HEXAHEDRON,
            {4 * x, 4 * x + 4, 4 * x + 5, 4 * x + 1, 4 * x + 2, 4 * x + 6, 4 * x + 7, 4 * x + 3}});
    const Geometry native(points, cells);
    MeshData data;
    data.boundary_names = {{7, "wall"}, {19, "catalog_without_local_faces"}};
    for (size_t n = 0; n < native.num_nodes(); ++n)
        data.nodes.push_back({node_id(n), points[n], 700 + n});
    for (size_t c = 0; c < native.num_cells(); ++c)
    {
        MeshData::Cell cell;
        cell.id = cell_id(c);
        cell.type = native.cell_type(c);
        cell.algebraic_id = 100 + c;
        cell.center = native.cell_centroid(c);
        cell.volume = std::nextafter(1.0, 2.0);
        for (auto node : native.cell_nodes(c))
            cell.node_ids.push_back(node_id(node));
        for (auto face : native.faces(c))
            cell.face_ids.push_back(face_id(face));
        // Deliberately retain source face-incidence order distinct from face storage.
        std::reverse(cell.face_ids.begin(), cell.face_ids.end());
        data.cells.push_back(std::move(cell));
    }
    for (size_t f = 0; f < native.num_faces(); ++f)
    {
        MeshData::Face face;
        face.id = face_id(f);
        face.algebraic_id = 400 + f;
        for (auto node : native.face_nodes(f))
            face.node_ids.push_back(node_id(node));
        face.owner = cell_id(native.owner_cell(f));
        if (native.neighbor_cell(f) != Geometry::invalid_cell_id())
            face.neighbor = cell_id(native.neighbor_cell(f));
        else
            face.boundary_id = 7;
        face.center = native.face_centroid(f);
        face.center.x = std::nextafter(face.center.x, 100.0);
        face.normal_from_owner = native.face_normal(f);
        face.area = std::nextafter(1.0, 2.0);
        data.faces.push_back(std::move(face));
    }
    return data;
}

MeshData shard(const MeshData& full, int rank)
{
    MeshData local = full;
    if (rank < 2)
    {
        std::swap(local.cells[0], local.cells[rank]);
        local.num_owned_cells = 1;
    }
    else
        local.num_owned_cells = 0;
    return local;
}

void expect_geometry(const Partitioner::NativeUnstructuredPartition& partition, const MeshData& reference)
{
    const auto& mesh = *partition.geometry;
    EXPECT_EQ(mesh.boundary_batch_name(7), "wall");
    EXPECT_EQ(mesh.boundary_batch_name(19), "catalog_without_local_faces");
    EXPECT_TRUE(mesh.boundary_face_batch(19).face_lids.empty());
    EXPECT_EQ(partition.cell_owner_ranks.size(), mesh.num_cells());
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        const auto id = partition.indexer.cell_global_id(c);
        const auto source = std::find_if(
            reference.cells.begin(), reference.cells.end(), [&](const auto& cell) { return cell.id == id; });
        ASSERT_NE(source, reference.cells.end());
        EXPECT_EQ(partition.indexer.local_to_global_cell_ordinal(c), source->algebraic_id);
        EXPECT_EQ(mesh.cell_volume(c), source->volume);
        EXPECT_EQ(mesh.cell_centroid(c), source->center);
        std::vector<ID> face_ids;
        for (auto face : mesh.faces(c))
            face_ids.push_back(partition.indexer.face_global_id(face));
        EXPECT_EQ(face_ids, source->face_ids);
    }
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto id = partition.indexer.face_global_id(f);
        const auto source = std::find_if(
            reference.faces.begin(), reference.faces.end(), [&](const auto& face) { return face.id == id; });
        ASSERT_NE(source, reference.faces.end());
        EXPECT_EQ(partition.indexer.local_to_global_face_ordinal(f), source->algebraic_id);
        EXPECT_EQ(mesh.face_area(f), source->area);
        EXPECT_EQ(mesh.face_centroid(f), source->center);
        const bool source_owner = partition.indexer.cell_global_id(mesh.owner_cell(f)) == source->owner;
        EXPECT_EQ(mesh.face_normal(f), source_owner ? source->normal_from_owner : source->normal_from_owner * -1.0);
        std::vector<ID> loop;
        for (auto node : mesh.face_nodes(f))
            loop.push_back(partition.indexer.node_global_id(node));
        EXPECT_EQ(loop, source->node_ids);
        if (source->neighbor != MeshData::invalid_id)
            EXPECT_TRUE(mesh.is_interior_face(f));
    }
}
} // namespace

TEST(NativeMeshDistributionTest, RootOnlyTwoCellsRetainMetricsIdentityInterfaceAndEmptyCatalog)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto reference = chain(2);
    auto source = comm->getRank() == 0 ? reference : MeshData{};
    const auto partition = Partitioner::partition_native(std::move(source), comm);
    expect_geometry(partition, reference);
    const auto& mesh = *partition.geometry;
    int owned = static_cast<int>(mesh.num_owned_cells()), global_owned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &owned, &global_owned);
    EXPECT_EQ(global_owned, 2);
    if (comm->getSize() == 2)
    {
        EXPECT_EQ(owned, 1);
        EXPECT_EQ(mesh.num_cells(), 2U);
    }
    if (comm->getSize() == 4)
    {
        int empty = owned == 0, global_empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &global_empty);
        EXPECT_EQ(global_empty, 2);
        if (!owned)
            EXPECT_EQ(mesh.num_cells(), 0U);
    }
    const auto handle = std::make_shared<MeshHandle<>>(partition.geometry, partition.indexer, comm);
    EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(), 2U);
    const Meshes::PartitionedMesh<Geometry> wrapped(partition.geometry, partition.indexer, comm);
    EXPECT_EQ(wrapped.boundary_batch_name(19), "catalog_without_local_faces");
    EXPECT_TRUE(wrapped.boundary_face_batch(19).face_lids.empty());
}

TEST(NativeMeshDistributionTest, DistributedOwnedShardsIgnoreAuraAsUniqueSources)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    const auto reference = chain(2);
    const auto partition = Partitioner::partition_native(shard(reference, comm->getRank()), comm);
    expect_geometry(partition, reference);
}

TEST(NativeMeshDistributionTest, OuterHaloFaceFlipsOwnerWithoutChangingMetricsOrLoops)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    const auto reference = chain(8);
    const auto partition = Partitioner::partition_native(comm->getRank() == 0 ? reference : MeshData{}, comm);
    expect_geometry(partition, reference);
    const auto& mesh = *partition.geometry;
    const Meshes::PartitionedMesh<Geometry> wrapped(partition.geometry, partition.indexer, comm);
    int flipped = 0;
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto id = partition.indexer.face_global_id(f);
        const auto source = std::find_if(
            reference.faces.begin(), reference.faces.end(), [&](const auto& face) { return face.id == id; });
        if (partition.indexer.cell_global_id(mesh.owner_cell(f)) != source->owner)
        {
            ++flipped;
            EXPECT_TRUE(mesh.is_interior_face(f));
            EXPECT_FALSE(mesh.is_boundary_face(f));
            EXPECT_EQ(mesh.neighbor_cell(f), Geometry::invalid_cell_id());
            EXPECT_TRUE(wrapped.is_interior_face(f));
            EXPECT_FALSE(wrapped.is_exterior_face(f));
            EXPECT_FALSE(wrapped.is_boundary_face(f));
        }
    }
    int global_flipped = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &flipped, &global_flipped);
    EXPECT_GT(global_flipped, 0);
}

TEST(NativeMeshDistributionTest, OneRankInvalidSourceFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshData source;
    if (comm->getRank() == 0)
    {
        source = chain(2);
        source.cells[0].face_ids[0] = 999999;
    }
    EXPECT_THROW(Partitioner::partition_native(std::move(source), comm), std::runtime_error);
}

TEST(NativeMeshDistributionTest, DuplicateOwnedSourcesFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto source = chain(2);
    source.num_owned_cells = comm->getRank() < 2 ? 1 : 0;
    EXPECT_THROW(Partitioner::partition_native(std::move(source), comm), std::runtime_error);
}

TEST(NativeMeshDistributionTest, SharedFaceDisagreementFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto source = shard(chain(2), comm->getRank());
    if (comm->getRank() == 1)
        for (auto& face : source.faces)
            if (face.neighbor != MeshData::invalid_id)
                face.area = 2;
    EXPECT_THROW(Partitioner::partition_native(std::move(source), comm), std::runtime_error);
}

TEST(NativeMeshDistributionTest, InconsistentHaloOptionsFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const unsigned depth = comm->getRank() == 0 ? 0 : 1;
    EXPECT_THROW(Partitioner::partition_native(MeshData{}, comm, depth), std::runtime_error);
}

TEST(NativeMeshDistributionTest, ExplicitFactoryPolicyAssignsRankContiguousCellOrdinals)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto reference = chain(2);
    const auto partition = Partitioner::partition_native(comm->getRank() == 0 ? reference : MeshData{}, comm, 1, true);
    const auto& mesh = *partition.geometry;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    GO count = mesh.num_owned_cells();
    std::vector<GO> counts(comm->getSize());
    Teuchos::gatherAll(*comm, 1, &count, comm->getSize(), counts.data());
    GO first = 0;
    for (int owner = 0; owner < comm->getRank(); ++owner)
        first += counts[owner];
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
        EXPECT_EQ(partition.indexer.local_to_global_cell_ordinal(c), first + static_cast<GO>(c));
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        EXPECT_LT(partition.indexer.local_to_global_cell_ordinal(c), 2);
        EXPECT_GE(partition.indexer.cell_global_id(c), source_base);
    }
}

TEST(NativeMeshDistributionTest, DuplicateAlgebraicOrdinalsFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshData source;
    if (comm->getRank() == 0)
    {
        source = chain(2);
        source.cells[1].algebraic_id = source.cells[0].algebraic_id;
    }
    EXPECT_THROW(Partitioner::partition_native(std::move(source), comm), std::runtime_error);
}

TEST(NativeMeshDistributionTest, SubcommunicatorDistributionUsesOnlyItsParticipants)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() < 2)
        GTEST_SKIP();
    const auto comm = world->split(world->getRank() % 2, world->getRank());
    const auto reference = chain(2);
    const auto partition = Partitioner::partition_native(comm->getRank() == 0 ? reference : MeshData{}, comm);
    expect_geometry(partition, reference);
    const auto handle = std::make_shared<MeshHandle<>>(partition.geometry, partition.indexer, comm);
    EXPECT_EQ(handle->owned_cell_map()->getComm()->getSize(), comm->getSize());
    EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(), 2U);
}

TEST(NativeMeshDistributionTest, SourceIdOutsideMappedGlobalOrdinalRangeFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshData source;
    if (comm->getRank() == 0)
    {
        source = chain(2);
        const auto original = source.nodes[0].id;
        const auto outside = static_cast<ID>(std::numeric_limits<DefaultTpetraTypes::global_ordinal_type>::max()) + 1;
        source.nodes[0].id = outside;
        for (auto& cell : source.cells)
            std::replace(cell.node_ids.begin(), cell.node_ids.end(), original, outside);
        for (auto& face : source.faces)
            std::replace(face.node_ids.begin(), face.node_ids.end(), original, outside);
    }
    EXPECT_THROW(Partitioner::partition_native(std::move(source), comm), std::runtime_error);
}

TEST(NativeMeshDistributionTest, DirectoryRoutingMixesStrideIdsDeterministically)
{
    // Raw id % 4 routes this entire sparse sequence to rank zero. The
    // fixed mixer must exercise all four directory ranks for this fixture.
    std::set<int> destinations;
    for (ID id = 4; id <= 256; id += 4)
    {
        const auto destination = native_partition_detail::directory_rank(id, 4);
        EXPECT_EQ(destination, native_partition_detail::directory_rank(id, 4));
        EXPECT_GE(destination, 0);
        EXPECT_LT(destination, 4);
        destinations.insert(destination);
    }
    EXPECT_EQ(destinations, (std::set<int>{0, 1, 2, 3}));
}
