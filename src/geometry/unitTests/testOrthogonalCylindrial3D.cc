/**
 * @file testOrthogonalCylindrial3D.cc
 * @brief Unit tests for the structured cylindrical mesh.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthogonalCylindrial3D.hh"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace
{

using Mesh = SimpleFluid::Mesh::OrthogonalCylindrial3D;
using CellID = Mesh::CellID;
using FaceID = Mesh::FaceID;
using NodeID = Mesh::NodeID;
using Vec3 = Mesh::Vec3;

constexpr auto pi = std::numbers::pi_v<SimpleFluid::real_t>;

Mesh make_sector_mesh()
{
    return Mesh({{
        {1.0, 2.0, 4.0},
        {0.0, 0.5 * pi, pi},
        {-1.0, 2.0}}});
}

Mesh make_periodic_mesh()
{
    return Mesh({{
        {1.0, 2.0},
        {0.0, pi, 2.0 * pi},
        {0.0, 1.0}}});
}

} // namespace

static_assert(std::is_same_v<
              SimpleFluid::Mesh::OrthogonalCylindrical3D,
              SimpleFluid::Mesh::OrthogonalCylindrial3D>);

TEST(OrthogonalCylindrial3DTest, RejectsInvalidCoordinates)
{
    EXPECT_THROW(
        Mesh({{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{1.0, 2.0}, {0.0, 1.0, 0.5}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{1.0, 2.0}, {0.0, 2.0 * pi + 1.0e-3}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{1.0, 2.0}, {0.0, 2.0 * pi}, {0.0, 1.0}}}),
        std::invalid_argument);
}

TEST(OrthogonalCylindrial3DTest, ReportsSectorCountsAndCellGeometry)
{
    const auto mesh = make_sector_mesh();

    EXPECT_FALSE(mesh.is_theta_periodic());
    EXPECT_EQ(mesh.num_cells_per_dimension(),
              (SimpleFluid::Vec3D<unsigned>{2, 2, 1}));
    EXPECT_EQ(mesh.num_cells(), 4U);
    EXPECT_EQ(mesh.num_local_cells(), 4U);
    EXPECT_EQ(mesh.num_owned_cells(), 4U);
    EXPECT_EQ(mesh.num_faces(), 20U);
    EXPECT_EQ(mesh.num_owned_faces(), 20U);
    EXPECT_EQ(mesh.num_nodes(), 18U);

    const CellID cell{0, 0, 0};
    EXPECT_NEAR(mesh.cell_volume(cell), 9.0 * pi / 4.0, 1.0e-12);
    EXPECT_NEAR(mesh.cell_centroid(cell).x, 28.0 / (9.0 * pi), 1.0e-12);
    EXPECT_NEAR(mesh.cell_centroid(cell).y, 28.0 / (9.0 * pi), 1.0e-12);
    EXPECT_DOUBLE_EQ(mesh.cell_centroid(cell).z, 0.5);

    const auto faces = mesh.faces(cell);
    EXPECT_EQ(faces[0], (FaceID{0, 0, 0, Mesh::R_FACE}));
    EXPECT_EQ(faces[1], (FaceID{1, 0, 0, Mesh::R_FACE}));
    EXPECT_EQ(faces[2], (FaceID{0, 0, 0, Mesh::THETA_FACE}));
    EXPECT_EQ(faces[3], (FaceID{0, 1, 0, Mesh::THETA_FACE}));
}

TEST(OrthogonalCylindrial3DTest, ComputesFaceGeometryAndTopology)
{
    const auto mesh = make_sector_mesh();

    const FaceID radial{1, 0, 0, Mesh::R_FACE};
    EXPECT_TRUE(mesh.is_interior_face(radial));
    EXPECT_EQ(mesh.owner_cell(radial), (CellID{0, 0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(radial), (CellID{1, 0, 0}));
    EXPECT_NEAR(mesh.face_area(radial), 3.0 * pi, 1.0e-12);
    EXPECT_NEAR(mesh.face_centroid(radial).x, 4.0 / pi, 1.0e-12);
    EXPECT_NEAR(mesh.face_centroid(radial).y, 4.0 / pi, 1.0e-12);
    EXPECT_NEAR(mesh.face_normal(radial).x, std::sqrt(0.5), 1.0e-12);
    EXPECT_NEAR(mesh.face_normal(radial).y, std::sqrt(0.5), 1.0e-12);

    const FaceID theta{0, 1, 0, Mesh::THETA_FACE};
    EXPECT_TRUE(mesh.is_interior_face(theta));
    EXPECT_EQ(mesh.owner_cell(theta), (CellID{0, 0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(theta), (CellID{0, 1, 0}));
    EXPECT_DOUBLE_EQ(mesh.face_area(theta), 3.0);
    EXPECT_NEAR(mesh.face_centroid(theta).x, 0.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(mesh.face_centroid(theta).y, 1.5);
    EXPECT_NEAR(mesh.face_normal(theta).x, -1.0, 1.0e-12);
    EXPECT_NEAR(mesh.face_normal(theta).y, 0.0, 1.0e-12);

    const FaceID axial{0, 0, 1, Mesh::Z_FACE};
    EXPECT_TRUE(mesh.is_boundary_face(axial));
    EXPECT_NEAR(mesh.face_area(axial), 3.0 * pi / 4.0, 1.0e-12);
    EXPECT_EQ(mesh.face_normal(axial), (Vec3{0.0, 0.0, 1.0}));
    EXPECT_DOUBLE_EQ(mesh.face_centroid(axial).z, 2.0);
}

TEST(OrthogonalCylindrial3DTest, BuildsSectorBoundaryPatches)
{
    const auto mesh = make_sector_mesh();

    EXPECT_EQ(mesh.boundary_patches().size(), 6U);
    EXPECT_EQ(mesh.boundary_patch_name(0), "rmin");
    EXPECT_EQ(mesh.boundary_patch_name(1), "rmax");
    EXPECT_EQ(mesh.boundary_patch_name(2), "thetamin");
    EXPECT_EQ(mesh.boundary_patch_name(3), "thetamax");
    EXPECT_EQ(mesh.boundary_patch_name(4), "zmin");
    EXPECT_EQ(mesh.boundary_patch_name(5), "zmax");

    EXPECT_EQ(mesh.boundary_face_patch(0).face_lids.size(), 2U);
    EXPECT_EQ(mesh.boundary_face_patch(1).face_lids.size(), 2U);
    EXPECT_EQ(mesh.boundary_face_patch(2).face_lids.size(), 2U);
    EXPECT_EQ(mesh.boundary_face_patch(3).face_lids.size(), 2U);
    EXPECT_EQ(mesh.boundary_face_patch(4).face_lids.size(), 4U);
    EXPECT_EQ(mesh.boundary_face_patch(5).face_lids.size(), 4U);

    const FaceID theta_min{0, 0, 0, Mesh::THETA_FACE};
    EXPECT_EQ(mesh.boundary_name(theta_min), "thetamin");
    EXPECT_EQ(mesh.face_normal(theta_min), (Vec3{0.0, -1.0, 0.0}));
}

TEST(OrthogonalCylindrial3DTest, ConnectsFullCirclePeriodically)
{
    const auto mesh = make_periodic_mesh();
    const FaceID seam{0, 0, 0, Mesh::THETA_FACE};

    EXPECT_TRUE(mesh.is_theta_periodic());
    EXPECT_EQ(mesh.num_cells(), 2U);
    EXPECT_EQ(mesh.num_faces(), 10U);
    EXPECT_EQ(mesh.num_nodes(), 8U);
    EXPECT_EQ(mesh.boundary_patches().size(), 4U);
    EXPECT_THROW(mesh.boundary_face_patch(2), std::out_of_range);

    EXPECT_TRUE(mesh.is_interior_face(seam));
    EXPECT_FALSE(mesh.is_boundary_face(seam));
    EXPECT_EQ(mesh.owner_cell(seam), (CellID{0, 1, 0}));
    EXPECT_EQ(mesh.neighbor_cell(seam), (CellID{0, 0, 0}));
    EXPECT_EQ(mesh.opposite_cell(seam, CellID{0, 1, 0}),
              (CellID{0, 0, 0}));
    EXPECT_EQ(mesh.faces(CellID{0, 1, 0})[3], seam);
    EXPECT_EQ(mesh.face_normal(seam), (Vec3{0.0, 1.0, 0.0}));
}

TEST(OrthogonalCylindrial3DTest, MapsStructuredAndLocalIdentifiers)
{
    const auto sector = make_sector_mesh();
    const auto& sector_indexer = sector.indexer();
    for (size_t id = 0; id < sector.num_cells(); ++id)
    {
        EXPECT_EQ(
            sector_indexer.cell_local_id(sector_indexer.cell_id(id)),
            id);
    }
    for (size_t id = 0; id < sector.num_faces(); ++id)
    {
        EXPECT_EQ(
            sector_indexer.face_local_id(sector_indexer.face_id(id)),
            id);
    }
    for (size_t id = 0; id < sector.num_nodes(); ++id)
    {
        EXPECT_EQ(
            sector_indexer.node_local_id(sector_indexer.node_id(id)),
            id);
    }

    const auto periodic = make_periodic_mesh();
    const auto& periodic_indexer = periodic.indexer();
    for (size_t id = 0; id < periodic.num_cells(); ++id)
    {
        EXPECT_EQ(
            periodic_indexer.cell_local_id(periodic_indexer.cell_id(id)),
            id);
    }
    for (size_t id = 0; id < periodic.num_faces(); ++id)
    {
        EXPECT_EQ(
            periodic_indexer.face_local_id(periodic_indexer.face_id(id)),
            id);
    }
    for (size_t id = 0; id < periodic.num_nodes(); ++id)
    {
        EXPECT_EQ(
            periodic_indexer.node_local_id(periodic_indexer.node_id(id)),
            id);
    }

    const auto node = sector.node_coordinates(NodeID{1, 1, 1});
    EXPECT_NEAR(node.x, 0.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(node.y, 2.0);
    EXPECT_DOUBLE_EQ(node.z, 2.0);
}
