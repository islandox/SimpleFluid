/**
 * @file testSTKMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for STKMesh class
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include "geometry/STKMesh.hh"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{

using MeshType = SimpleFluid::STKMesh<SimpleFluid::TpetraTypes<>>;

/**
 * @brief Global Google Test environment to initialize MPI and Kokkos.
 */
class KokkosEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized)
        {
            MPI_Init(nullptr, nullptr);
            d_initialized_mpi = true;
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
            d_initialized_kokkos = true;
        }
    }

    void TearDown() override
    {
        if (d_initialized_kokkos && Kokkos::is_initialized())
        {
            Kokkos::finalize();
        }

        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (d_initialized_mpi && !mpi_finalized)
        {
            MPI_Finalize();
        }
    }

private:
    bool d_initialized_mpi = false;
    bool d_initialized_kokkos = false;
};

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Declare the STK coordinate field for a mesh.
 *
 * @param mesh STK mesh instance that will own the coordinate field.
 * @return Reference to the declared coordinate field.
 */
stk::mesh::Field<double>& declare_coordinate_field(MeshType& mesh)
{
    auto meta = mesh.meta();
    auto& coord_field = meta->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);
    return coord_field;
}

/**
 * @brief Set the coordinates of a STK node.
 *
 * @param coord_field Coordinate field attached to the mesh.
 * @param node STK node entity.
 * @param coord Coordinate values to assign.
 */
void set_node_coord(stk::mesh::Field<double>& coord_field,
                    stk::mesh::Entity node,
                    const std::array<double, 3>& coord)
{
    auto* data = stk::mesh::field_data(coord_field, node);
    std::copy(coord.begin(), coord.end(), data);
}

/**
 * @brief Populate the mesh with a single hexahedral element.
 *
 * @param mesh STK mesh to populate.
 * @param elem_id Global ID for the new element.
 */
void populate_single_hex(MeshType& mesh, stk::mesh::EntityId elem_id)
{
    auto& coord_field = declare_coordinate_field(mesh);
    auto meta = mesh.meta();
    auto& hex_part = meta->declare_part_with_topology("hexes", stk::topology::HEX_8);

    auto bulk = mesh.bulk();
    bulk->modification_begin();

    const stk::mesh::EntityIdVector node_ids{1, 2, 3, 4, 5, 6, 7, 8};
    stk::mesh::declare_element(*bulk, hex_part, elem_id, node_ids);

    const std::array<std::array<double, 3>, 8> coords{{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {0.0, 1.0, 1.0},
    }};

    for (std::size_t i = 0; i < node_ids.size(); ++i)
    {
        const auto node = bulk->get_entity(stk::topology::NODE_RANK, node_ids[i]);
        set_node_coord(coord_field, node, coords[i]);
    }

    bulk->modification_end();
}

/**
 * @brief Populate the mesh with two adjacent hexahedral elements.
 *
 * The function constructs a shared face between the two elements.
 *
 * @param mesh STK mesh to populate.
 */
void populate_two_adjacent_hexes(MeshType& mesh)
{
    auto& coord_field = declare_coordinate_field(mesh);
    auto meta = mesh.meta();
    auto& hex_part = meta->declare_part_with_topology("hexes", stk::topology::HEX_8);
    auto bulk = mesh.bulk();

    bulk->modification_begin();

    stk::mesh::declare_element(*bulk, hex_part, 1, {1, 2, 3, 4, 5, 6, 7, 8});
    stk::mesh::declare_element(*bulk, hex_part, 2, {2, 9, 10, 3, 6, 11, 12, 7});

    const std::vector<std::pair<stk::mesh::EntityId, std::array<double, 3>>> coords{
        {1, {0.0, 0.0, 0.0}},
        {2, {1.0, 0.0, 0.0}},
        {3, {1.0, 1.0, 0.0}},
        {4, {0.0, 1.0, 0.0}},
        {5, {0.0, 0.0, 1.0}},
        {6, {1.0, 0.0, 1.0}},
        {7, {1.0, 1.0, 1.0}},
        {8, {0.0, 1.0, 1.0}},
        {9, {2.0, 0.0, 0.0}},
        {10, {2.0, 1.0, 0.0}},
        {11, {2.0, 0.0, 1.0}},
        {12, {2.0, 1.0, 1.0}},
    };

    for (const auto& [node_id, coord] : coords)
    {
        const auto node = bulk->get_entity(stk::topology::NODE_RANK, node_id);
        set_node_coord(coord_field, node, coord);
    }

    bulk->modification_end();
}

} // namespace

TEST(STKMeshTest, AssembleMeshWithoutThrowing)
{
    MeshType mesh;
    populate_two_adjacent_hexes(mesh);

    mesh.assemble();

    EXPECT_EQ(mesh.num_local_cells(), 2u);
    EXPECT_EQ(mesh.num_owned_cells(), 2u);
    EXPECT_EQ(mesh.num_faces(), 11u);
    EXPECT_DOUBLE_EQ(mesh.cell_volume(0), 1.0);
    EXPECT_DOUBLE_EQ(mesh.cell_volume(1), 1.0);
    EXPECT_EQ(mesh.global_to_local_cell(1), 0);
    EXPECT_EQ(mesh.global_to_local_cell(2), 1);
}

TEST(STKMeshTest, ExportVtuWritesUnstructuredGrid)
{
    MeshType mesh;
    populate_single_hex(mesh, 42);
    mesh.assemble();

    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_file = std::filesystem::temp_directory_path()
                           / ("SimpleFluid_testSTKMesh_export_" + std::to_string(unique_id) + ".vtu");

    mesh.export_vtu(output_file.string());

    std::ifstream input(output_file);
    ASSERT_TRUE(input.good());

    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());

    EXPECT_NE(contents.find("<VTKFile type=\"UnstructuredGrid\""), std::string::npos);
    EXPECT_NE(contents.find("NumberOfPoints=\"8\" NumberOfCells=\"1\""), std::string::npos);
    EXPECT_NE(contents.find("<DataArray type=\"Int64\" Name=\"cell_gid\" format=\"ascii\">"), std::string::npos);
    EXPECT_NE(contents.find("<DataArray type=\"Float64\" Name=\"cell_volume\" format=\"ascii\">"), std::string::npos);
    EXPECT_NE(contents.find("0 1 2 3 4 5 6 7"), std::string::npos);
    EXPECT_NE(contents.find("<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          8\n"), std::string::npos);
    EXPECT_NE(contents.find("<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          12\n"), std::string::npos);

    std::filesystem::remove(output_file);
}
