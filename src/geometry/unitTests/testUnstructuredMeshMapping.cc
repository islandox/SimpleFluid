/** @file testUnstructuredMeshMapping.cc
 * @brief Native source/algebraic associations survive handle construction and reordering.
 */
#include "geometry/MeshHandle.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "parallel/CollectiveConstruction.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <sstream>

namespace
{
using Mesh = SimpleFluid::Meshes::UnstructuredMesh;
using Handle = SimpleFluid::MeshHandle<>;
using Factory = SimpleFluid::MeshReorderingFactory<>;
using Indexer = Handle::unstructured_indexer_type;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Mesh> two_cells()
{
    return std::make_shared<Mesh>(SimpleFluid::Arr<Mesh::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1},
                                      {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}},
        SimpleFluid::Arr<Mesh::CellDefinition>{{Mesh::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}},
            {Mesh::CellType::HEXAHEDRON, {1, 8, 9, 2, 5, 10, 11, 6}}});
}

Indexer mapping(const Mesh& mesh)
{
    std::vector<Indexer::CellMapping> cells{{17, 4000000001ULL, 41}, {23, 8000000003ULL, 97}};
    std::vector<Indexer::FaceMapping> faces;
    std::vector<Indexer::NodeMapping> owned_nodes, overlap_nodes;
    for (size_t f = 0; f < mesh.num_faces(); ++f)
        faces.push_back({100 + f, 16000000001ULL + 101 * f, static_cast<long long>(1001 + 13 * f)});
    for (size_t n = 0; n < mesh.num_nodes(); ++n)
        (n < 7 ? owned_nodes : overlap_nodes)
            .push_back({200 + n, 32000000001ULL + 103 * n, static_cast<long long>(2001 + 17 * n)});
    return Indexer(std::move(cells), {}, std::move(faces), {}, std::move(owned_nodes), std::move(overlap_nodes));
}
} // namespace

TEST(UnstructuredMeshMappingTest, ReorderingPreservesSourceAlgebraicAndNodeAssociations)
{
    const auto comm = Tpetra::getDefaultComm()->split(Tpetra::getDefaultComm()->getRank(), 0);
    const auto geometry = two_cells();
    const auto indexer = mapping(*geometry);
    auto handle = std::make_shared<Handle>(geometry, indexer, comm);
    ASSERT_EQ(handle->owned_cell_map()->getComm().getRawPtr(), comm.getRawPtr());
    ASSERT_EQ(handle->cell_geometry_global_id(0), 4000000001LL);
    ASSERT_EQ(handle->cell_global_id(0), 41);
    ASSERT_EQ(handle->cell_global_id(1), 97);
    ASSERT_EQ(handle->indexer().num_owned_nodes(), 7U);

    auto layout = Factory::selected_cells_first(
        std::move(handle), [](auto source, const auto&) { return source == 8000000003LL; });
    EXPECT_EQ(layout.selected_owned_cells, 1U);
    EXPECT_EQ(layout.mesh->cell_geometry_global_id(0), 8000000003LL);
    EXPECT_EQ(layout.mesh->cell_geometry_global_id(1), 4000000001LL);
    EXPECT_EQ(layout.mesh->cell_global_id(0), 97);
    EXPECT_EQ(layout.mesh->cell_global_id(1), 41);
    EXPECT_EQ(layout.mesh->owned_cell_map()->getGlobalElement(0), 97);
    EXPECT_EQ(layout.mesh->overlap_cell_map()->getGlobalElement(1), 41);
    EXPECT_DOUBLE_EQ(layout.mesh->cell_centroid(0).x, 1.5);
    EXPECT_DOUBLE_EQ(layout.mesh->cell_volume(0), 1);
    EXPECT_EQ(layout.mesh->indexer().local_to_global_cell_ordinal(0), 97);
    EXPECT_EQ(layout.mesh->indexer().global_to_local_cell_id(8000000003LL), 23);
    EXPECT_EQ(layout.mesh->indexer().num_owned_nodes(), 7U);
    for (size_t n = 0; n < geometry->num_nodes(); ++n)
    {
        EXPECT_EQ(layout.mesh->indexer().node_global_id(static_cast<int>(n)), 32000000001LL + 103 * n);
        EXPECT_EQ(layout.mesh->indexer().local_to_global_node_ordinal(static_cast<int>(n)), 2001 + 17 * n);
    }
    for (size_t f = 0; f < geometry->num_faces(); ++f)
    {
        EXPECT_EQ(layout.mesh->face_geometry_global_id(static_cast<int>(f)), 16000000001LL + 101 * f);
        EXPECT_EQ(layout.mesh->face_global_id(static_cast<int>(f)), 1001 + 13 * f);
        EXPECT_EQ(layout.mesh->overlap_face_map()->getGlobalElement(static_cast<int>(f)), 1001 + 13 * f);
    }

    auto reordered = std::const_pointer_cast<Handle>(std::move(layout.mesh));
    const auto again = Factory::selected_cells_first(
        std::move(reordered), [](auto source, const auto&) { return source == 4000000001LL; });
    EXPECT_EQ(again.mesh->cell_geometry_global_id(0), 4000000001LL);
    EXPECT_EQ(again.mesh->cell_global_id(0), 41);
    EXPECT_EQ(again.mesh->cell_global_id(1), 97);
    EXPECT_DOUBLE_EQ(again.mesh->cell_centroid(0).x, .5);
}

TEST(UnstructuredMeshMappingTest, EmptyPartitionWrapperRetainsGlobalBoundaryCatalog)
{
    const auto comm = Tpetra::getDefaultComm()->split(Tpetra::getDefaultComm()->getRank(), 0);
    SimpleFluid::Meshes::UnstructuredMeshData data;
    data.boundary_names = {{7, "wall"}, {19, "remote_patch"}};
    const auto geometry = std::make_shared<Mesh>(Mesh::from_data(std::move(data)));
    const SimpleFluid::Meshes::PartitionedMesh<Mesh> wrapper(geometry, Indexer{}, comm);
    EXPECT_EQ(wrapper.num_cells(), 0U);
    EXPECT_EQ(wrapper.boundary_batches().size(), 2U);
    EXPECT_EQ(wrapper.boundary_batch_name(19), "remote_patch");
    EXPECT_TRUE(wrapper.boundary_face_batch(7).face_lids.empty());
    EXPECT_TRUE(wrapper.boundary_face_batch(19).face_lids.empty());
    EXPECT_EQ(wrapper.boundary_face_map()->getLocalNumElements(), 0U);
}

TEST(UnstructuredMeshMappingTest, IndexedLayoutMismatchFailsCollectivelyBeforeMaps)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto geometry = two_cells();
    const auto indexer = comm->getRank() == 0 ? Indexer{} : mapping(*geometry);
    EXPECT_THROW((Handle(geometry, indexer, comm)), std::exception);
}

TEST(UnstructuredMeshMappingTest, NullIndexedGeometryFailsCollectivelyBeforeMaps)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto geometry = two_cells();
    const auto indexer = mapping(*geometry);
    Handle::UnstructuredPtr supplied = comm->getRank() == 0 ? nullptr : geometry;
    EXPECT_THROW((Handle(std::move(supplied), indexer, comm)), std::exception);
}

TEST(UnstructuredMeshMappingTest, OutputRetainsSourceIdsAndUsesMeshCommunicatorAfterReordering)
{
    const auto world = Tpetra::getDefaultComm();
    const auto comm = world->split(world->getRank(), 0);
    const auto geometry = two_cells();
    auto handle = SimpleFluid::detail::collective_make_shared<Handle>(*comm, geometry, mapping(*geometry), comm);
    auto layout = Factory::selected_cells_first(
        std::move(handle), [](auto source, const auto&) { return source == 8000000003LL; });
    const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("native_source_ids_" + std::to_string(world->getRank()) + "_" + std::to_string(token) + ".vtu");
    layout.mesh->export_vtu(path.string());
    // This mesh is serial even if the enclosing test runs on several ranks.
    // A world-communicator suffix would make this exact path absent.
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto values = [&](const std::string& name)
    {
        const auto tag = xml.find("Name=\"" + name + "\"");
        EXPECT_NE(tag, std::string::npos);
        std::vector<long long> result;
        if (tag == std::string::npos)
            return result;
        const auto first = xml.find('>', tag);
        const auto last = xml.find("</DataArray>", first);
        std::istringstream data(xml.substr(first + 1, last - first - 1));
        for (long long id; data >> id;)
            result.push_back(id);
        return result;
    };
    EXPECT_EQ(values("cell_gid"), (std::vector<long long>{8000000003LL, 4000000001LL}));
    EXPECT_EQ(values("algebraic_cell_gid"), (std::vector<long long>{97, 41}));
    input.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
