/**
 * @file testSemiStructMeshTopo.cc
 * @brief Unit tests for layered semi-structured topology.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/SemiStructMeshTopo.hh"

#include <stdexcept>

namespace
{

using Topology = SimpleFluid::Meshes::SemiStructMeshTopo;
using Indexer = Topology::Indexer;
using CellID = Topology::CellID;
using FaceID = Topology::FaceID;

Topology make_topology(bool axial_periodic = false)
{
    return Topology(
        4,
        {{0, 1, 3}, {1, 2, 3}},
        2,
        {
            {0, 1, "ymin"},
            {1, 2, "xmax"},
            {2, 3, "ymax"},
            {3, 0, "xmin"}},
        axial_periodic);
}

Topology make_topology_with_interior_cell()
{
    return Topology(
        12,
        {
            {0, 1, 2, 3},
            {4, 5, 1, 0},
            {1, 6, 7, 2},
            {3, 2, 8, 9},
            {10, 0, 3, 11}},
        3);
}

} // namespace

TEST(SemiStructMeshTopoTest, BuildsBaseConnectivity)
{
    const auto topology = make_topology();
    const auto& indexer = topology.indexer();

    EXPECT_EQ(indexer.num_cells_per_layer, 2U);
    EXPECT_EQ(indexer.num_side_faces_per_layer, 5U);
    EXPECT_EQ(indexer.num_nodes_per_layer, 4U);
    EXPECT_EQ(indexer.num_layers, 2U);
    EXPECT_EQ(indexer.total_cells(), 4U);
    EXPECT_EQ(indexer.total_faces(), 16U);
    EXPECT_EQ(indexer.total_nodes(), 12U);

    ASSERT_EQ(topology.side_faces().size(), 5U);
    EXPECT_EQ(topology.side_face(1).nodes,
              (std::array<unsigned, 2>{1, 3}));
    EXPECT_EQ(topology.side_face(1).owner, 0U);
    EXPECT_EQ(topology.side_face(1).neighbor, 1U);
    EXPECT_EQ(
        topology.side_face(1).boundary_id,
        Topology::invalid_boundary_id);

    EXPECT_EQ(
        topology.cell_side_faces(0),
        (SimpleFluid::Arr<unsigned>{0, 1, 2}));
    EXPECT_EQ(
        topology.cell_side_faces(1),
        (SimpleFluid::Arr<unsigned>{3, 4, 1}));
}

TEST(SemiStructMeshTopoTest, QueriesCellFacesAndAdjacency)
{
    const auto topology = make_topology();

    const auto faces = topology.cell_faces(CellID{1, 1});
    ASSERT_EQ(faces.size(), 5U);
    EXPECT_EQ(faces[0], (FaceID{1, 1, Indexer::AXIAL}));
    EXPECT_EQ(faces[1], (FaceID{1, 2, Indexer::AXIAL}));
    EXPECT_EQ(faces[2], (FaceID{3, 1, Indexer::SIDE}));
    EXPECT_EQ(faces[4], (FaceID{1, 1, Indexer::SIDE}));

    const FaceID shared{1, 0, Indexer::SIDE};
    EXPECT_EQ(topology.owner_cell(shared), (CellID{0, 0}));
    EXPECT_EQ(topology.neighbor_cell(shared), (CellID{1, 0}));
    EXPECT_FALSE(topology.is_boundary_face(shared));

    const FaceID axial{0, 1, Indexer::AXIAL};
    EXPECT_EQ(topology.owner_cell(axial), (CellID{0, 0}));
    EXPECT_EQ(topology.neighbor_cell(axial), (CellID{0, 1}));
}

TEST(SemiStructMeshTopoTest, BuildsBoundaryPatches)
{
    const auto topology = make_topology();

    EXPECT_EQ(topology.boundary_patches().size(), 6U);
    EXPECT_EQ(topology.boundary_patch_name(0), "zmin");
    EXPECT_EQ(topology.boundary_patch_name(1), "zmax");
    for (int patch = 0; patch < 6; ++patch)
    {
        EXPECT_EQ(topology.boundary_face_patch(patch).face_lids.size(), 2U);
    }

    const FaceID ymin{0, 1, Indexer::SIDE};
    EXPECT_TRUE(topology.is_boundary_face(ymin));
    EXPECT_EQ(topology.boundary_patch_name(topology.boundary_id(ymin)),
              "ymin");
    EXPECT_EQ(topology.neighbor_cell(ymin), CellID{});
    EXPECT_THROW(topology.boundary_face_patch(6), std::out_of_range);
}

TEST(SemiStructMeshTopoTest, CachesInteriorPatchAndCellNeighbors)
{
    const auto topology = make_topology_with_interior_cell();

    ASSERT_EQ(topology.interior_cell_patch().size(), 1U);
    EXPECT_EQ(topology.interior_cell_patch()[0], (CellID{0, 1}));
    EXPECT_EQ(
        topology.neighbor_cells(CellID{0, 1}),
        (Topology::NeighborCells{
            {0, 0},
            {0, 2},
            {1, 1},
            {2, 1},
            {3, 1},
            {4, 1}}));

    EXPECT_EQ(
        topology.neighbor_cells(CellID{1, 0}),
        (Topology::NeighborCells{
            {1, 1},
            {0, 0}}));
}

TEST(SemiStructMeshTopoTest, WrapsAxialFacesPeriodically)
{
    const Topology topology(
        3,
        {{0, 1, 2}},
        2,
        {},
        true);
    const FaceID seam{0, 0, Indexer::AXIAL};

    EXPECT_TRUE(topology.indexer().axial_periodic);
    EXPECT_EQ(topology.indexer().num_node_layers, 2U);
    EXPECT_EQ(topology.owner_cell(seam), (CellID{0, 1}));
    EXPECT_EQ(topology.neighbor_cell(seam), (CellID{0, 0}));
    EXPECT_FALSE(topology.is_boundary_face(seam));
    EXPECT_EQ(topology.cell_faces(CellID{0, 1})[1], seam);

    EXPECT_EQ(topology.boundary_patches().size(), 1U);
    EXPECT_EQ(topology.boundary_face_patch(2).face_lids.size(), 6U);
    EXPECT_THROW(topology.boundary_face_patch(0), std::out_of_range);
    EXPECT_TRUE(topology.interior_cell_patch().empty());
    EXPECT_EQ(
        topology.neighbor_cells(CellID{0, 0}),
        (Topology::NeighborCells{{0, 1}, {0, 1}}));
}

TEST(SemiStructMeshTopoTest, RejectsInvalidBaseTopology)
{
    EXPECT_THROW(
        Topology(3, {{0, 1}}, 1),
        std::invalid_argument);
    EXPECT_THROW(
        Topology(3, {{0, 1, 2}, {0, 1, 2}}, 1),
        std::invalid_argument);
    EXPECT_THROW(
        Topology(
            4,
            {{0, 1, 3}, {1, 2, 3}},
            1,
            {{1, 3, "interior"}}),
        std::invalid_argument);
    EXPECT_THROW(
        Topology(
            3,
            {{0, 1, 2}},
            1,
            {{0, 1, "zmin"}}),
        std::invalid_argument);
}
