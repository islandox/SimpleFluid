/**
 * @file testSemiStructuredXY_Z.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for polygonal XY meshes extruded through Z.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace
{

using Mesh = SimpleFluid::Meshes::SemiStructuredXY_Z;
using CellID = Mesh::CellID;
using FaceID = Mesh::FaceID;
using NodeID = Mesh::NodeID;
using Vec3 = Mesh::Vec3;

Mesh make_mesh()
{
    return Mesh(
        {
            {0.0, 0.0, 0.0},
            {2.0, 0.0, 0.0},
            {2.0, 1.0, 0.0},
            {0.0, 1.0, 0.0}},
        {
            {0, 1, 3},
            {1, 2, 3}},
        {0.0, 1.0, 3.0},
        {
            {0, 1, "ymin"},
            {1, 2, "xmax"},
            {2, 3, "ymax"},
            {3, 0, "xmin"}});
}

} // namespace

static_assert(SimpleFluid::MeshClass<Mesh>);
static_assert(std::is_same_v<SimpleFluid::Meshes::SemiStructuredXYZ3D, Mesh>);

TEST(SemiStructuredXY_ZTest, RejectsInvalidInput)
{
    EXPECT_THROW(
        Mesh(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            {{0, 1, 2}},
            {0.0}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            {{0, 2, 1}},
            {0.0, 1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            {{0, 1, 2}},
            {0.0, 1.0},
            {{0, 1, "zmin"}}),
        std::invalid_argument);
}

TEST(SemiStructuredXY_ZTest, AssignsDefaultSideBatch)
{
    const Mesh mesh(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        {{0, 1, 2}},
        {0.0, 1.0});

    EXPECT_EQ(mesh.num_boundary_batches(), 3);
    EXPECT_EQ(mesh.boundary_batch_name(2), "side");
    EXPECT_EQ(mesh.boundary_face_batch(2).face_lids.size(), 3U);
}

TEST(SemiStructuredXY_ZTest, ReportsCountsAndCellGeometry)
{
    const auto mesh = make_mesh();

    EXPECT_EQ(mesh.spatial_dimension(), 3U);
    EXPECT_EQ(mesh.num_cells(), 4U);
    EXPECT_EQ(mesh.num_local_cells(), 4U);
    EXPECT_EQ(mesh.num_owned_cells(), 4U);
    EXPECT_EQ(mesh.num_faces(), 16U);
    EXPECT_EQ(mesh.num_owned_faces(), 16U);
    EXPECT_EQ(mesh.num_nodes(), 12U);
    EXPECT_EQ(mesh.topology().side_faces().size(), 5U);

    const CellID cell{0, 1};
    EXPECT_TRUE(mesh.is_owned_cell(cell));
    EXPECT_DOUBLE_EQ(mesh.cell_volume(cell), 2.0);
    EXPECT_EQ(mesh.cell_centroid(cell), (Vec3{2.0 / 3.0, 1.0 / 3.0, 2.0}));

    const auto faces = mesh.faces(cell);
    EXPECT_EQ(faces.size(), 5U);
    EXPECT_EQ(faces[0], (FaceID{0, 1, Mesh::Z_FACE}));
    EXPECT_EQ(faces[1], (FaceID{0, 2, Mesh::Z_FACE}));
    EXPECT_EQ(faces[2], (FaceID{0, 1, Mesh::SIDE_FACE}));
    EXPECT_EQ(faces[3], (FaceID{1, 1, Mesh::SIDE_FACE}));
    EXPECT_EQ(faces[4], (FaceID{2, 1, Mesh::SIDE_FACE}));
}

TEST(SemiStructuredXY_ZTest, ComputesInteriorFaceTopologyAndGeometry)
{
    const auto mesh = make_mesh();
    const FaceID face{1, 0, Mesh::SIDE_FACE};
    const CellID owner{0, 0};
    const CellID neighbor{1, 0};
    const auto inverse_sqrt5 = 1.0 / std::sqrt(5.0);

    EXPECT_TRUE(mesh.is_owned_face(face));
    EXPECT_TRUE(mesh.is_interior_face(face));
    EXPECT_EQ(mesh.owner_cell(face), owner);
    EXPECT_EQ(mesh.neighbor_cell(face), neighbor);
    EXPECT_EQ(mesh.opposite_cell(face, owner), neighbor);

    EXPECT_DOUBLE_EQ(mesh.face_area(face), std::sqrt(5.0));
    EXPECT_EQ(mesh.face_centroid(face), (Vec3{1.0, 0.5, 0.5}));
    EXPECT_NEAR(mesh.face_normal(face).x, inverse_sqrt5, 1.0e-12);
    EXPECT_NEAR(mesh.face_normal(face).y, 2.0 * inverse_sqrt5, 1.0e-12);
    EXPECT_DOUBLE_EQ(mesh.face_normal(face).z, 0.0);
    EXPECT_EQ(mesh.boundary_id(face), Mesh::invalid_boundary_id);

    const FaceID axial{0, 1, Mesh::Z_FACE};
    EXPECT_EQ(mesh.owner_cell(axial), (CellID{0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(axial), (CellID{0, 1}));
    EXPECT_DOUBLE_EQ(mesh.face_area(axial), 1.0);
    EXPECT_EQ(mesh.face_centroid(axial),
              (Vec3{2.0 / 3.0, 1.0 / 3.0, 1.0}));
    EXPECT_EQ(mesh.face_normal(axial), (Vec3{0.0, 0.0, 1.0}));
}

TEST(SemiStructuredXY_ZTest, BuildsBoundaryBatches)
{
    const auto mesh = make_mesh();
    const FaceID bottom{0, 0, Mesh::SIDE_FACE};

    EXPECT_TRUE(mesh.is_boundary_face(bottom));
    EXPECT_EQ(mesh.owner_cell(bottom), (CellID{0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(bottom), Mesh::invalid_cell_id());
    EXPECT_DOUBLE_EQ(mesh.face_area(bottom), 2.0);
    EXPECT_EQ(mesh.face_centroid(bottom), (Vec3{1.0, 0.0, 0.5}));
    EXPECT_EQ(mesh.face_normal(bottom), (Vec3{0.0, -1.0, 0.0}));
    EXPECT_EQ(mesh.boundary_name(bottom), "ymin");

    EXPECT_EQ(mesh.num_boundary_batches(), 6);
    for (int batch = 0; batch < 6; ++batch)
    {
        EXPECT_EQ(mesh.boundary_face_batch(batch).face_lids.size(), 2U);
    }
    EXPECT_EQ(mesh.boundary_batch_name(0), "zmin");
    EXPECT_EQ(mesh.boundary_batch_name(1), "zmax");
}

TEST(SemiStructuredXY_ZTest, MapsIdentifiersAndNodeCoordinates)
{
    const auto mesh = make_mesh();
    const auto& indexer = mesh.indexer();

    for (size_t id = 0; id < mesh.num_cells(); ++id)
    {
        EXPECT_EQ(indexer.cell_local_id(indexer.cell_id(id)), id);
    }
    for (size_t id = 0; id < mesh.num_faces(); ++id)
    {
        EXPECT_EQ(indexer.face_local_id(indexer.face_id(id)), id);
    }
    for (size_t id = 0; id < mesh.num_nodes(); ++id)
    {
        EXPECT_EQ(indexer.node_local_id(indexer.node_id(id)), id);
    }

    EXPECT_EQ(mesh.node_coordinates(NodeID{2, 2}), (Vec3{2.0, 1.0, 3.0}));
    EXPECT_THROW_WHEN_DEBUG(
        mesh.face_area(FaceID{5, 0, Mesh::SIDE_FACE}),
        std::out_of_range);
}
