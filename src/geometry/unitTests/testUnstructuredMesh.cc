/**
 * @file testUnstructuredMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the STK-free unstructured CRTP mesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/MeshHandle.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace
{

using Mesh = SimpleFluid::Meshes::UnstructuredMesh;
using CellType = Mesh::CellType;
using Handle = SimpleFluid::MeshHandle<>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

Mesh make_two_hex_mesh()
{
    return Mesh(
        {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0},
            {2.0, 0.0, 0.0},
            {2.0, 1.0, 0.0},
            {2.0, 0.0, 1.0},
            {2.0, 1.0, 1.0},
        },
        {
            {CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}},
            {CellType::HEXAHEDRON, {1, 8, 9, 2, 5, 10, 11, 6}},
        },
        {
            {{3, 7, 4, 0}, 7, "inlet"},
            {{8, 10, 11, 9}, 8, "outlet"},
        });
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

Mesh::PolyhedralTopology prism_topology(size_t sides)
{
    Mesh::PolyhedralTopology topology{1, {}};
    SimpleFluid::Arr<Mesh::NodeID> bottom, top;
    for (size_t i = 0; i < sides; ++i)
    {
        bottom.push_back(sides - 1 - i);
        top.push_back(sides + i);
    }
    topology.faces.push_back({bottom, 0, Mesh::invalid_ordinal, 21, "bottom"});
    topology.faces.push_back({top, 0, Mesh::invalid_ordinal, 22, "top"});
    for (size_t i = 0; i < sides; ++i)
        topology.faces.push_back({{i, (i + 1) % sides, (i + 1) % sides + sides, i + sides}, 0});
    return topology;
}

SimpleFluid::Arr<Mesh::Vec3> concave_prism_nodes()
{
    return {{0, 0, 0}, {2, 0, 0}, {2, 1, 0}, {1, 1, 0}, {1, 2, 0}, {0, 2, 0},
            {0, 0, 2}, {2, 0, 2}, {2, 1, 2}, {1, 1, 2}, {1, 2, 2}, {0, 2, 2}};
}

} // namespace

/** @brief Verifies the unstructured mesh and indexer satisfy the mesh concepts. */
TEST(UnstructuredMeshTest, SatisfiesMeshBaseConcept)
{
    static_assert(SimpleFluid::MeshIndexer<Mesh::Indexer>);
    static_assert(SimpleFluid::MeshClass<Mesh>);
}

/**
 * @brief Verifies an unstructured mesh builds connectivity, geometry, local
 * indexing, and boundary batches from explicit definitions.
 */
TEST(UnstructuredMeshTest, BuildsConnectivityGeometryAndBoundaryBatches)
{
    const auto mesh = make_two_hex_mesh();

    EXPECT_EQ(mesh.num_cells(), 2U);
    EXPECT_EQ(mesh.num_owned_cells(), 2U);
    EXPECT_EQ(mesh.num_faces(), 11U);
    EXPECT_EQ(mesh.num_nodes(), 12U);
    EXPECT_DOUBLE_EQ(mesh.cell_volume(0U), 1.0);
    EXPECT_DOUBLE_EQ(mesh.cell_volume(1U), 1.0);

    const auto interior_face = mesh.faces(0U)[3];
    EXPECT_EQ(mesh.owner_cell(interior_face), 0U);
    EXPECT_EQ(mesh.neighbor_cell(interior_face), 1U);
    EXPECT_TRUE(mesh.is_interior_face(interior_face));
    EXPECT_EQ(mesh.face_normal(interior_face), Mesh::Vec3(1.0, 0.0, 0.0));

    const auto inlet_face = mesh.faces(0U)[5];
    EXPECT_TRUE(mesh.is_boundary_face(inlet_face));
    EXPECT_EQ(mesh.boundary_id(inlet_face), 7);
    EXPECT_EQ(mesh.boundary_name(inlet_face), "inlet");
    EXPECT_EQ(mesh.boundary_face_batch(7).face_lids.size(), 1U);
}

/** @brief Verifies boundary tags cannot be assigned to interior faces. */
TEST(UnstructuredMeshTest, RejectsBoundaryTagsOnInteriorFaces)
{
    EXPECT_THROW(
        Mesh(
            {
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0},
                {1.0, 1.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0},
                {1.0, 0.0, 1.0},
                {1.0, 1.0, 1.0},
                {0.0, 1.0, 1.0},
                {2.0, 0.0, 0.0},
                {2.0, 1.0, 0.0},
                {2.0, 0.0, 1.0},
                {2.0, 1.0, 1.0},
            },
            {
                {CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}},
                {CellType::HEXAHEDRON, {1, 8, 9, 2, 5, 10, 11, 6}},
            },
            {
                {{1, 5, 6, 2}, 9, "bad_interior"},
            }),
        std::invalid_argument);
}

/**
 * @brief Verifies MeshHandle wraps an unstructured mesh and exports its
 * geometry and fields to VTU.
 */
TEST(UnstructuredMeshTest, MeshHandleWrapsAndExportsUnstructuredMesh)
{
    const auto mesh = std::make_shared<Mesh>(make_two_hex_mesh());
    const Handle handle(mesh);

    EXPECT_FALSE(handle.is_stk());
    EXPECT_EQ(handle.num_owned_cells(), 2U);
    EXPECT_EQ(handle.num_faces(), 11U);
    EXPECT_EQ(handle.owned_cell_map()->getLocalNumElements(), 2U);
    EXPECT_EQ(handle.overlap_face_map()->getLocalNumElements(), 11U);
    EXPECT_EQ(handle.boundary_batch_name(7), "inlet");
    EXPECT_EQ(handle.faces(0).size(), 6U);

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_file = std::filesystem::temp_directory_path()
        / ("SimpleFluid_testUnstructuredMesh_export_"
           + std::to_string(unique_id) + ".vtu");

    handle.export_vtu(output_file.string());
    ASSERT_TRUE(std::filesystem::exists(output_file));

    const auto contents = read_file(output_file);
    EXPECT_NE(
        contents.find("<VTKFile type=\"UnstructuredGrid\""),
        std::string::npos);
    EXPECT_NE(
        contents.find("NumberOfPoints=\"12\" NumberOfCells=\"2\""),
        std::string::npos);
    EXPECT_NE(
        contents.find("<DataArray type=\"UInt8\" Name=\"types\""),
        std::string::npos);

    std::filesystem::remove(output_file);
}

TEST(UnstructuredMeshTest, ConcavePolyhedronUsesSignedAreaAndVolumeMoments)
{
    const Mesh mesh(concave_prism_nodes(), prism_topology(6));
    EXPECT_EQ(mesh.cell_type(0), CellType::POLYHEDRON);
    EXPECT_EQ(mesh.num_faces(), 8U);
    EXPECT_EQ(mesh.face_nodes(0).size(), 6U);
    EXPECT_NEAR(mesh.cell_volume(0), 6.0, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).x, 5.0 / 6.0, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).y, 5.0 / 6.0, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).z, 1.0, 1e-13);
    EXPECT_NEAR(mesh.face_area(0), 3.0, 1e-13);
    EXPECT_NEAR(mesh.face_centroid(0).x, 5.0 / 6.0, 1e-13);
    EXPECT_NEAR(mesh.face_centroid(0).y, 5.0 / 6.0, 1e-13);
    EXPECT_EQ(mesh.face_normal(0), Mesh::Vec3(0, 0, -1));
    EXPECT_EQ(mesh.boundary_name(0), "bottom");
    EXPECT_EQ(mesh.cell_face_orientation(0, 0), 1);

    auto shifted = concave_prism_nodes();
    for (auto& node : shifted) node = node + Mesh::Vec3(1e8, -1e8, 1e8);
    const Mesh translated(shifted, prism_topology(6));
    EXPECT_NEAR(translated.cell_volume(0), 6.0, 1e-13);
    EXPECT_NEAR(translated.cell_centroid(0).x - 1e8, 5.0 / 6.0, 1e-8);
}

TEST(UnstructuredMeshTest, PolyhedronSupportsLargeFacesAndCellStencils)
{
    constexpr size_t n = 12;
    constexpr double pi = 3.14159265358979323846;
    SimpleFluid::Arr<Mesh::Vec3> nodes;
    for (int z = 0; z < 2; ++z)
        for (size_t i = 0; i < n; ++i)
            nodes.push_back({std::cos(2 * pi * i / n), std::sin(2 * pi * i / n), static_cast<double>(z)});
    const Mesh mesh(nodes, prism_topology(n));
    EXPECT_EQ(mesh.faces(0).size(), n + 2);
    EXPECT_EQ(mesh.face_nodes(0).size(), n);
    EXPECT_NEAR(mesh.cell_volume(0), n * std::sin(2 * pi / n) / 2, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).x, 0.0, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).y, 0.0, 1e-13);
    EXPECT_NEAR(mesh.cell_centroid(0).z, 0.5, 1e-13);
}

TEST(UnstructuredMeshTest, ExplicitFacesRetainOwnerOrderAndNeighborIncidence)
{
    const auto source = make_two_hex_mesh();
    Mesh::PolyhedralTopology topology{source.num_cells(), {}};
    for (size_t face = 0; face < source.num_faces(); ++face)
        topology.faces.push_back({source.face_nodes(face), source.owner_cell(face), source.neighbor_cell(face)});
    const Mesh mesh(source.nodes(), topology, 1, 6);
    EXPECT_EQ(mesh.num_owned_cells(), 1U);
    EXPECT_EQ(mesh.num_owned_faces(), 6U);
    for (size_t face = 0; face < source.num_faces(); ++face)
    {
        EXPECT_EQ(mesh.face_nodes(face), source.face_nodes(face));
        EXPECT_EQ(mesh.owner_cell(face), source.owner_cell(face));
        EXPECT_EQ(mesh.neighbor_cell(face), source.neighbor_cell(face));
        if (mesh.is_interior_face(face))
        {
            EXPECT_EQ(mesh.cell_face_orientation(0, face), 1);
            EXPECT_EQ(mesh.cell_face_orientation(1, face), -1);
        }
    }
    EXPECT_NEAR(mesh.cell_volume(0), 1.0, 1e-14);
    EXPECT_NEAR(mesh.cell_volume(1), 1.0, 1e-14);

    SimpleFluid::Arr<Mesh::CellDefinition> cells(2);
    for (size_t cell = 0; cell < 2; ++cell)
    {
        cells[cell].type = CellType::POLYHEDRON;
        for (const auto face : mesh.faces(cell))
        {
            auto loop = mesh.face_nodes(face);
            if (mesh.cell_face_orientation(cell, face) < 0) std::ranges::reverse(loop);
            cells[cell].face_node_ids.push_back(std::move(loop));
        }
    }
    const Mesh rebuilt(source.nodes(), cells);
    EXPECT_EQ(rebuilt.num_faces(), 11U);
    EXPECT_NEAR(rebuilt.cell_volume(1), 1.0, 1e-14);
    EXPECT_THROW(mesh.cell_face_orientation(1, 0), std::invalid_argument);
}

TEST(UnstructuredMeshTest, PolyhedronOutputRetainsConcaveLoopsAndOutwardSharedFaces)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Serial topology export regression.";
    const auto source = make_two_hex_mesh();
    Mesh::PolyhedralTopology cubes{source.num_cells(), {}};
    for (size_t f = 0; f < source.num_faces(); ++f)
        cubes.faces.push_back({source.face_nodes(f), source.owner_cell(f), source.neighbor_cell(f)});
    for (const auto& mesh : {std::make_shared<Mesh>(source.nodes(), cubes),
             std::make_shared<Mesh>(concave_prism_nodes(), prism_topology(6))})
    {
        const Handle handle(mesh);
        const auto output = handle.vtu_topology();
        ASSERT_EQ(output->face_offsets.size(), mesh->num_cells());
        size_t cursor = 0;
        for (size_t c = 0; c < mesh->num_cells(); ++c)
        {
            EXPECT_EQ(output->cell_types[c], 42);
            ASSERT_EQ(output->faces.at(cursor++), mesh->faces(c).size());
            double volume = 0;
            for (const auto f : mesh->faces(c))
            {
                const auto count = static_cast<size_t>(output->faces.at(cursor++));
                ASSERT_EQ(count, mesh->face_nodes(f).size());
                SimpleFluid::Arr<Mesh::Vec3> loop;
                for (size_t i = 0; i < count; ++i)
                    loop.push_back(output->points.at(output->faces.at(cursor++)));
                const auto area = SimpleFluid::MeshUtils::face_area_vector(loop);
                const auto outward = mesh->face_area_vector(f) * mesh->cell_face_orientation(c, f);
                EXPECT_NEAR((area - outward).norm(), 0.0, 1e-13);
                for (size_t i = 1; i + 1 < loop.size(); ++i)
                    volume += loop[0].dot(loop[i].cross(loop[i + 1])) / 6;
            }
            EXPECT_EQ(output->face_offsets[c], cursor);
            EXPECT_NEAR(volume, mesh->cell_volume(c), 1e-13);
        }
        EXPECT_EQ(cursor, output->faces.size());
        const auto path = std::filesystem::temp_directory_path() / "simplefluid_polyhedron_handle.vtu";
        handle.export_vtu(path.string());
        EXPECT_NE(read_file(path).find("Name=\"faces\""), std::string::npos);
        EXPECT_NE(read_file(path).find("Name=\"faceoffsets\""), std::string::npos);
        std::filesystem::remove(path);
    }
}

TEST(UnstructuredMeshTest, RejectsOpenMisOrientedAndDegeneratePolyhedra)
{
    auto nodes = concave_prism_nodes();
    auto topology = prism_topology(6);
    topology.faces.pop_back();
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    topology = prism_topology(6);
    std::ranges::reverse(topology.faces[0].node_ids);
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    topology = prism_topology(6);
    for (auto& face : topology.faces) std::ranges::reverse(face.node_ids);
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    topology = prism_topology(6);
    topology.faces.push_back(topology.faces[0]);
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    topology = prism_topology(6);
    topology.faces[0].node_ids[1] = topology.faces[0].node_ids[0];
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    topology = prism_topology(6);
    nodes[0].z = 0.2;
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    nodes = concave_prism_nodes();
    nodes[0] = nodes[1];
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    nodes = concave_prism_nodes();
    std::swap(topology.faces[0].node_ids[1], topology.faces[0].node_ids[3]);
    EXPECT_THROW(Mesh(nodes, topology), std::invalid_argument);
    EXPECT_NO_THROW(Mesh(SimpleFluid::Arr<Mesh::Vec3>{}, Mesh::PolyhedralTopology{}));
    EXPECT_THROW(Mesh(nodes, prism_topology(6), 2, 0), std::invalid_argument);
}
