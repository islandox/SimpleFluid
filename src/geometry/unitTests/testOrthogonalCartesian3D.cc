/**
 * @file testOrthogonalCartesian3D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for the structured Cartesian mesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "utils/testing_environment.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"

#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace
{

using Mesh = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Indexer = Mesh::Indexer;
using CellID = Mesh::CellID;
using FaceID = Mesh::FaceID;
using NodeID = Mesh::NodeID;
using Vec3 = Mesh::Vec3;

static_assert(SimpleFluid::MeshIndexer<Indexer>);

Mesh make_mesh()
{
    return Mesh({{
        {0.0, 1.0, 3.0},
        {-1.0, 2.0},
        {0.0, 2.0, 5.0}}});
}

} // namespace

/** @brief Verifies consecutive-coordinate differences and midpoints. */
TEST(MeshUtilsConsecutiveTest, ComputesDifferencesAndMidpoints)
{
    const SimpleFluid::Arr<SimpleFluid::real_t> values{-1.0, 2.0, 6.0};
    EXPECT_EQ(SimpleFluid::MeshUtils::consec_diff(values),
              (SimpleFluid::Arr<SimpleFluid::real_t>{3.0, 4.0}));
    EXPECT_EQ(SimpleFluid::MeshUtils::consec_mid(values),
              (SimpleFluid::Arr<SimpleFluid::real_t>{0.5, 4.0}));
    EXPECT_TRUE(SimpleFluid::MeshUtils::consec_diff(
        SimpleFluid::Arr<SimpleFluid::real_t>{}).empty());
}

/** @brief Verifies malformed Cartesian edge arrays are rejected. */
TEST(OrthogonalCartesian3DTest, RejectsInvalidEdges)
{
    EXPECT_THROW(
        Mesh({{{0.0}, {0.0, 1.0}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{0.0, 1.0}, {0.0, 0.0}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}}}),
        std::invalid_argument);
    EXPECT_THROW(
        Mesh({{{0.0, std::numeric_limits<double>::infinity()},
               {0.0, 1.0},
               {0.0, 1.0}}}),
        std::invalid_argument);
}

/**
 * @brief Verifies entity counts, cell centroids, volumes, and cell-face lists
 * for an orthogonal Cartesian mesh.
 */
TEST(OrthogonalCartesian3DTest, ReportsCountsAndCellGeometry)
{
    const auto mesh = make_mesh();

    EXPECT_EQ(mesh.spatial_dimension(), 3U);
    EXPECT_EQ(mesh.num_cells_per_dimension(),
              (SimpleFluid::Vec3D<unsigned>{2, 1, 2}));
    EXPECT_EQ(mesh.num_cells(), 4U);
    EXPECT_EQ(mesh.num_local_cells(), 4U);
    EXPECT_EQ(mesh.num_owned_cells(), 4U);
    EXPECT_EQ(mesh.num_faces(), 20U);
    EXPECT_EQ(mesh.num_owned_faces(), 20U);
    EXPECT_EQ(mesh.num_nodes(), 18U);

    const CellID cell{1, 0, 1};
    EXPECT_TRUE(mesh.is_owned_cell(cell));
    EXPECT_DOUBLE_EQ(mesh.cell_volume(cell), 18.0);
    EXPECT_EQ(mesh.cell_centroid(cell), (Vec3{2.0, 0.5, 3.5}));

    const auto faces = mesh.faces(cell);
    EXPECT_EQ(faces[0], (FaceID{1, 0, 1, Mesh::X_FACE}));
    EXPECT_EQ(faces[1], (FaceID{2, 0, 1, Mesh::X_FACE}));
    EXPECT_EQ(faces[5], (FaceID{1, 0, 2, Mesh::Z_FACE}));
    EXPECT_EQ(mesh.face_distances(cell).size(), 6U);
}

/**
 * @brief Verifies interior-face connectivity, centroids, areas, normals, and
 * center-to-center vectors.
 */
TEST(OrthogonalCartesian3DTest, ComputesInteriorFaceTopologyAndGeometry)
{
    const auto mesh = make_mesh();
    const FaceID face{1, 0, 0, Mesh::X_FACE};
    const CellID owner{0, 0, 0};
    const CellID neighbor{1, 0, 0};

    EXPECT_TRUE(mesh.is_owned_face(face));
    EXPECT_TRUE(mesh.is_interior_face(face));
    EXPECT_FALSE(mesh.is_exterior_face(face));
    EXPECT_FALSE(mesh.is_boundary_face(face));
    EXPECT_EQ(mesh.owner_cell(face), owner);
    EXPECT_EQ(mesh.neighbor_cell(face), neighbor);
    EXPECT_EQ(mesh.opposite_cell(face, owner), neighbor);
    EXPECT_EQ(mesh.opposite_cell(face, neighbor), owner);

    EXPECT_DOUBLE_EQ(mesh.face_area(face), 6.0);
    EXPECT_EQ(mesh.face_centroid(face), (Vec3{1.0, 0.5, 1.0}));
    EXPECT_EQ(mesh.face_normal(face), (Vec3{1.0, 0.0, 0.0}));
    EXPECT_EQ(mesh.face_area_vector(face), (Vec3{6.0, 0.0, 0.0}));
    EXPECT_EQ(mesh.face_normal_outward(face, neighbor),
              (Vec3{-1.0, 0.0, 0.0}));
    EXPECT_EQ(mesh.face_area_vector_outward(face, neighbor),
              (Vec3{-6.0, 0.0, 0.0}));

    EXPECT_DOUBLE_EQ(mesh.cell_to_face_distance(face, owner), 0.5);
    EXPECT_DOUBLE_EQ(mesh.cell_to_face_distance(face, neighbor), 1.0);
    EXPECT_DOUBLE_EQ(mesh.face_cell_center_distance(face), 1.5);
    EXPECT_EQ(mesh.cell_center_vector(face, owner),
              (Vec3{1.5, 0.0, 0.0}));
}

/**
 * @brief Verifies indexed face queries across the X, Y, and Z face
 * orientations.
 */
TEST(OrthogonalCartesian3DTest, ComputesIndexedQueriesAcrossOrientations)
{
    const auto mesh = make_mesh();

    const FaceID ymin{1, 0, 1, Mesh::Y_FACE};
    EXPECT_EQ(mesh.owner_cell(ymin), (CellID{1, 0, 1}));
    EXPECT_EQ(mesh.neighbor_cell(ymin), Mesh::invalid_cell_id());
    EXPECT_DOUBLE_EQ(mesh.face_area(ymin), 6.0);
    EXPECT_EQ(mesh.face_centroid(ymin), (Vec3{2.0, -1.0, 3.5}));
    EXPECT_EQ(mesh.face_normal(ymin), (Vec3{0.0, -1.0, 0.0}));
    EXPECT_EQ(mesh.boundary_id(ymin), 2);

    const FaceID zmid{1, 0, 1, Mesh::Z_FACE};
    EXPECT_EQ(mesh.owner_cell(zmid), (CellID{1, 0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(zmid), (CellID{1, 0, 1}));
    EXPECT_DOUBLE_EQ(mesh.face_area(zmid), 6.0);
    EXPECT_EQ(mesh.face_centroid(zmid), (Vec3{2.0, 0.5, 2.0}));
    EXPECT_EQ(mesh.face_normal(zmid), (Vec3{0.0, 0.0, 1.0}));
    EXPECT_EQ(mesh.boundary_id(zmid), Mesh::invalid_boundary_id);

    const FaceID zmax{1, 0, 2, Mesh::Z_FACE};
    EXPECT_EQ(mesh.owner_cell(zmax), (CellID{1, 0, 1}));
    EXPECT_EQ(mesh.neighbor_cell(zmax), Mesh::invalid_cell_id());
    EXPECT_EQ(mesh.boundary_id(zmax), 5);
}

/**
 * @brief Verifies physical boundary faces, IDs, names, normals, and batches
 * on all six Cartesian sides.
 */
TEST(OrthogonalCartesian3DTest, ComputesBoundaryFacesAndBatches)
{
    const auto mesh = make_mesh();
    const FaceID xmin{0, 0, 0, Mesh::X_FACE};

    EXPECT_TRUE(mesh.is_exterior_face(xmin));
    EXPECT_TRUE(mesh.is_boundary_face(xmin));
    EXPECT_EQ(mesh.owner_cell(xmin), (CellID{0, 0, 0}));
    EXPECT_EQ(mesh.neighbor_cell(xmin), Mesh::invalid_cell_id());
    EXPECT_EQ(mesh.face_normal(xmin), (Vec3{-1.0, 0.0, 0.0}));
    EXPECT_EQ(mesh.face_normal_outward(xmin, CellID{0, 0, 0}),
              (Vec3{-1.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(mesh.face_cell_center_distance(xmin), 0.0);
    EXPECT_EQ(mesh.boundary_id(xmin), 0);
    EXPECT_EQ(mesh.boundary_name(xmin), "xmin");

    EXPECT_EQ(mesh.num_boundary_batches(), 6);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(0)), 2);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(1)), 2);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(2)), 4);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(3)), 4);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(4)), 2);
    EXPECT_EQ(std::ranges::distance(mesh.boundary_face_batch(5)), 2);
    EXPECT_EQ(mesh.boundary_batch_name(5), "zmax");

    EXPECT_THROW(
        mesh.boundary_name(FaceID{1, 0, 0, Mesh::X_FACE}),
        std::out_of_range);
    EXPECT_THROW(mesh.boundary_face_batch(6), std::out_of_range);
}

/**
 * @brief Verifies structured IDs map consistently to local cell, face, and
 * node ordinals and coordinates.
 */
TEST(OrthogonalCartesian3DTest, MapsStructuredAndLocalIdentifiers)
{
    const auto mesh = make_mesh();
    auto indexer = mesh.indexer();

    for (size_t id = 0; id < mesh.num_cells(); ++id)
    {
        EXPECT_EQ(indexer.cell_ordinal(indexer.cell_id(id)), id);
    }
    for (size_t id = 0; id < mesh.num_faces(); ++id)
    {
        EXPECT_EQ(indexer.face_ordinal(indexer.face_id(id)), id);
    }
    for (size_t id = 0; id < mesh.num_nodes(); ++id)
    {
        EXPECT_EQ(indexer.node_ordinal(indexer.node_id(id)), id);
    }

    EXPECT_EQ(mesh.node_coordinates(NodeID{2, 1, 2}),
              (Vec3{3.0, 2.0, 5.0}));
    EXPECT_THROW_WHEN_DEBUG(mesh.face_area(FaceID{3, 0, 0, Mesh::X_FACE}),
                 std::out_of_range);
    EXPECT_THROW_WHEN_DEBUG(mesh.node_coord(NodeID{(unsigned)-1, 0, 0}), std::out_of_range);
}

/** @brief Verifies distance queries reject cells that do not share a face. */
TEST(OrthogonalCartesian3DTest, RejectsNonAdjacentCellQueries)
{
    const auto mesh = make_mesh();
    const FaceID face{1, 0, 0, Mesh::X_FACE};
    const CellID non_adjacent{0, 0, 1};

    EXPECT_THROW(mesh.opposite_cell(face, non_adjacent),
                 std::invalid_argument);
    EXPECT_THROW(mesh.cell_to_face_distance(face, non_adjacent),
                 std::invalid_argument);
    EXPECT_THROW(mesh.face_normal_outward(face, non_adjacent),
                 std::invalid_argument);
    EXPECT_THROW(
        mesh.cell_center_vector(FaceID{0, 0, 0, Mesh::X_FACE},
                                CellID{0, 0, 0}),
        std::invalid_argument);
}
