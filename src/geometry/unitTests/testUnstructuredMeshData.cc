/** @file testUnstructuredMeshData.cc
 * @brief Detached native topology and authoritative metric contracts.
 */
#include "geometry/mesh/UnstructuredMesh.hh"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
using Mesh = SimpleFluid::Meshes::UnstructuredMesh;
using MeshData = SimpleFluid::Meshes::UnstructuredMeshData;
using Vec3 = Mesh::Vec3;

MeshData hex_data()
{
    MeshData data;
    const std::vector<Vec3> points{
        {0, 0, 0}, {2, 0, 0}, {2, 3, 0}, {0, 3, 0}, {0, 0, 4}, {2, 0, 4}, {2, 3, 4}, {0, 3, 4}};
    std::vector<MeshData::ID> node_ids;
    for (size_t n = 0; n < points.size(); ++n)
    {
        node_ids.push_back(101 + 17 * n);
        data.nodes.push_back({node_ids.back(), points[n]});
    }
    const std::vector<std::vector<size_t>> loops{
        {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {3, 7, 6, 2}, {0, 4, 7, 3}};
    const std::vector<Vec3> centers{{1, 1.5, 0}, {1, 1.5, 4}, {1, 0, 2}, {2, 1.5, 2}, {1, 3, 2}, {0, 1.5, 2}};
    const std::vector<Vec3> normals{{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}};
    const std::vector<double> areas{6, 6, 8, 12, 8, 12};
    for (size_t f = 0; f < loops.size(); ++f)
    {
        MeshData::Face face;
        face.id = 701 + 19 * f;
        for (auto n : loops[f])
            face.node_ids.push_back(node_ids[n]);
        face.owner = 4001;
        face.boundary_id = 7;
        face.center = centers[f];
        face.normal_from_owner = normals[f];
        face.area = areas[f];
        data.faces.push_back(std::move(face));
    }
    data.cells.push_back({4001, Mesh::CellType::HEXAHEDRON, node_ids, {777, 720, 796, 701, 758, 739}, {1, 1.5, 2}, 24});
    data.boundary_names = {{7, "wall"}, {11, "remote_patch"}};
    return data;
}

void expect_bits(double actual, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}
} // namespace

TEST(UnstructuredMeshDataTest, RetainsAnalyticHexMetricsTypeAndOrderedConnectivity)
{
    const auto data = hex_data();
    const auto mesh = Mesh::from_data(data);
    EXPECT_EQ(mesh.cell_type(0), Mesh::CellType::HEXAHEDRON);
    EXPECT_EQ(mesh.cell_nodes(0), (std::vector<Mesh::NodeID>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(mesh.faces(0), (std::vector<Mesh::FaceID>{4, 1, 5, 0, 3, 2}));
    EXPECT_EQ(mesh.face_nodes(0), (std::vector<Mesh::NodeID>{0, 3, 2, 1}));
    EXPECT_EQ(mesh.cell_centroid(0), Vec3(1, 1.5, 2));
    expect_bits(mesh.cell_volume(0), 24);
    for (size_t f = 0; f < data.faces.size(); ++f)
    {
        expect_bits(mesh.face_area(f), data.faces[f].area);
        EXPECT_EQ(mesh.face_centroid(f), data.faces[f].center);
        EXPECT_EQ(mesh.face_normal(f), data.faces[f].normal_from_owner);
        EXPECT_EQ(mesh.face_normal_outward(f, 0), data.faces[f].normal_from_owner);
    }
    EXPECT_TRUE(mesh.boundary_face_batch(11).face_lids.empty());
}

TEST(UnstructuredMeshDataTest, RetainsWarpedFaceAndImportedMetricBitPatterns)
{
    auto data = hex_data();
    data.nodes[7].coordinates.z = 4.25;
    // Sum of the two triangles (4,5,6) and (4,6,7), independent of the native metric builder.
    const double warped_area = 3.0 + std::sqrt(589.0) / 8.0;
    data.faces[1].area = warped_area;
    data.faces[1].center.x = std::nextafter(1.0, 2.0);
    data.faces[1].center.y = -0.0;
    const Vec3 integrated_normal(.75, -.5, 12);
    data.faces[1].normal_from_owner = integrated_normal / integrated_normal.norm();
    data.cells[0].volume = std::nextafter(24.0, 25.0);
    const auto mesh = Mesh::from_data(data);
    expect_bits(mesh.face_area(1), warped_area);
    expect_bits(mesh.cell_volume(0), data.cells[0].volume);
    for (size_t i = 0; i < 3; ++i)
    {
        expect_bits(mesh.face_centroid(1).component(i), data.faces[1].center.component(i));
        expect_bits(mesh.face_normal(1).component(i), data.faces[1].normal_from_owner.component(i));
    }
}

TEST(UnstructuredMeshDataTest, RetainsWedgeTypeAndSuppliedMetrics)
{
    MeshData data;
    const std::vector<Vec3> nodes{{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {2, 0, 4}, {0, 3, 4}};
    for (size_t i = 0; i < nodes.size(); ++i)
        data.nodes.push_back({10 + i, nodes[i]});
    data.cells.push_back(
        {100, Mesh::CellType::TRIPRISM, {10, 11, 12, 13, 14, 15}, {200, 201, 202, 203, 204}, {2.0 / 3, 1, 2}, 12});
    data.faces = {{200, {10, 12, 11}, 100, MeshData::invalid_id, -1, {2.0 / 3, 1, 0}, {0, 0, -1}, 3},
        {201, {13, 14, 15}, 100, MeshData::invalid_id, -1, {2.0 / 3, 1, 4}, {0, 0, 1}, 3},
        {202, {10, 11, 14, 13}, 100, MeshData::invalid_id, -1, {1, 0, 2}, {0, -1, 0}, 8},
        {203, {11, 12, 15, 14}, 100, MeshData::invalid_id, -1, {1, 1.5, 2}, Vec3(3, 2, 0) / std::sqrt(13.0),
            4 * std::sqrt(13.0)},
        {204, {12, 10, 13, 15}, 100, MeshData::invalid_id, -1, {0, 1.5, 2}, {-1, 0, 0}, 12}};
    const auto mesh = Mesh::from_data(std::move(data));
    EXPECT_EQ(mesh.cell_type(0), Mesh::CellType::TRIPRISM);
    expect_bits(mesh.cell_volume(0), 12);
    EXPECT_EQ(mesh.faces(0).size(), 5U);
}

TEST(UnstructuredMeshDataTest, EmptyPartitionRetainsFullBoundaryCatalog)
{
    MeshData data;
    data.boundary_names = {{2, "inlet"}, {19, "outlet"}};
    const auto mesh = Mesh::from_data(std::move(data));
    EXPECT_EQ(mesh.num_cells(), 0U);
    EXPECT_EQ(mesh.num_owned_cells(), 0U);
    EXPECT_EQ(mesh.num_boundary_batches(), 2);
    EXPECT_EQ(mesh.boundary_batch_name(19), "outlet");
    EXPECT_TRUE(mesh.boundary_face_batch(2).face_lids.empty());
    EXPECT_TRUE(mesh.boundary_face_batch(19).face_lids.empty());
}

TEST(UnstructuredMeshDataTest, RemoteOuterHaloNeighborRemainsInterior)
{
    auto data = hex_data();
    data.num_owned_cells = 0;
    data.num_owned_faces = 0;
    data.faces[1].neighbor = 9001;
    data.faces[1].boundary_id = -1;
    const auto mesh = Mesh::from_data(data);
    EXPECT_EQ(mesh.neighbor_cell(1), Mesh::invalid_cell_id());
    EXPECT_TRUE(mesh.is_interior_face(1));
    EXPECT_FALSE(mesh.is_exterior_face(1));
    EXPECT_FALSE(mesh.is_boundary_face(1));
    EXPECT_EQ(mesh.cell_face_orientation(0, 1), 1);
    data.num_owned_cells = 1;
    EXPECT_THROW(Mesh::from_data(data), std::invalid_argument);
}

TEST(UnstructuredMeshDataTest, RejectsInvalidReferencesAdjacencyCountsAndBoundaries)
{
    auto bad = hex_data();
    bad.faces[0].node_ids[0] = 999999;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.faces[0].owner = 999999;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.cells[0].face_ids[0] = 999999;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.cells[0].node_ids.pop_back();
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.faces[0].node_ids.pop_back();
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.faces[0].neighbor = bad.cells[0].id;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.faces[0].boundary_id = 123;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.nodes[1].id = bad.nodes[0].id;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.num_owned_faces = 7;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.cells[0].type = Mesh::CellType::TETRAHEDRON;
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
}

TEST(UnstructuredMeshDataTest, RejectsNonfiniteAndNonpositiveMetrics)
{
    for (const auto invalid :
        {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto bad = hex_data();
        bad.cells[0].volume = invalid;
        EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
        bad = hex_data();
        bad.faces[0].area = invalid;
        EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    }
    auto bad = hex_data();
    bad.faces[0].normal_from_owner = Vec3{};
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.cells[0].center.x = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
    bad = hex_data();
    bad.nodes[0].coordinates.z = std::numeric_limits<double>::infinity();
    EXPECT_THROW(Mesh::from_data(bad), std::invalid_argument);
}

TEST(UnstructuredMeshDataTest, ExistingNodeCellConstructionRetainsComputedGeometry)
{
    const auto data = hex_data();
    std::vector<Vec3> nodes;
    for (const auto& node : data.nodes)
        nodes.push_back(node.coordinates);
    const Mesh mesh(nodes, {{Mesh::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}}});
    EXPECT_EQ(mesh.cell_type(0), Mesh::CellType::HEXAHEDRON);
    EXPECT_DOUBLE_EQ(mesh.cell_volume(0), 24);
    EXPECT_EQ(mesh.cell_centroid(0), Vec3(1, 1.5, 2));
}
