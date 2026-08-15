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
