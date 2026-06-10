/**
 * @file testSemiStructuredIndexer.cc
 * @brief Unit tests for layered semi-structured mesh indexing.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/SemiStructuredIndexer.hh"

#include <cstddef>

namespace
{

using Indexer = SimpleFluid::Meshes::SemiStructuredIndexer;
using CellID = Indexer::CellID;
using FaceID = Indexer::FaceID;
using NodeID = Indexer::NodeID;

TEST(SemiStructuredIndexerTest, ComputesNonPeriodicCountsAndOffsets)
{
    const Indexer indexer(3, 5, 4, 2);

    EXPECT_EQ(indexer.num_cells_per_layer, 3U);
    EXPECT_EQ(indexer.num_side_faces_per_layer, 5U);
    EXPECT_EQ(indexer.num_nodes_per_layer, 4U);
    EXPECT_EQ(indexer.num_layers, 2U);
    EXPECT_EQ(indexer.num_node_layers, 3U);
    EXPECT_FALSE(indexer.axial_periodic);

    EXPECT_EQ(indexer.num_faces_per_orientation[Indexer::AXIAL], 9U);
    EXPECT_EQ(indexer.num_faces_per_orientation[Indexer::SIDE], 10U);
    EXPECT_EQ(indexer.face_offsets[Indexer::AXIAL], 0U);
    EXPECT_EQ(indexer.face_offsets[Indexer::SIDE], 9U);

    EXPECT_EQ(indexer.face_strides[Indexer::AXIAL][Indexer::IJ], 3U);
    EXPECT_EQ(indexer.face_strides[Indexer::AXIAL][Indexer::K], 1U);
    EXPECT_EQ(indexer.face_strides[Indexer::SIDE][Indexer::IJ], 1U);
    EXPECT_EQ(indexer.face_strides[Indexer::SIDE][Indexer::K], 5U);

    EXPECT_EQ(indexer.total_cells(), 6U);
    EXPECT_EQ(indexer.total_faces(), 19U);
    EXPECT_EQ(indexer.total_nodes(), 12U);
}

TEST(SemiStructuredIndexerTest, MapsNonPeriodicIdentifiers)
{
    const Indexer indexer(3, 5, 4, 2);

    EXPECT_EQ(indexer.cell_local_id(CellID{2, 1}), 5U);
    EXPECT_EQ(indexer.cell_id(5), (CellID{2, 1}));

    EXPECT_EQ(
        indexer.face_local_id(FaceID{2, 2, Indexer::AXIAL}),
        8U);
    EXPECT_EQ(
        indexer.face_id(8),
        (FaceID{2, 2, Indexer::AXIAL}));

    EXPECT_EQ(
        indexer.face_local_id(FaceID{4, 1, Indexer::SIDE}),
        18U);
    EXPECT_EQ(
        indexer.face_id(18),
        (FaceID{4, 1, Indexer::SIDE}));

    EXPECT_EQ(indexer.node_local_id(NodeID{3, 2}), 11U);
    EXPECT_EQ(indexer.node_id(11), (NodeID{3, 2}));

    for (size_t id = 0; id < indexer.total_cells(); ++id)
    {
        EXPECT_EQ(indexer.cell_local_id(indexer.cell_id(id)), id);
    }
    for (size_t id = 0; id < indexer.total_faces(); ++id)
    {
        EXPECT_EQ(indexer.face_local_id(indexer.face_id(id)), id);
    }
    for (size_t id = 0; id < indexer.total_nodes(); ++id)
    {
        EXPECT_EQ(indexer.node_local_id(indexer.node_id(id)), id);
    }
}

TEST(SemiStructuredIndexerTest, SupportsPeriodicAxialDirection)
{
    const Indexer indexer(3, 5, 4, 2, true);

    EXPECT_TRUE(indexer.axial_periodic);
    EXPECT_EQ(indexer.num_node_layers, 2U);
    EXPECT_EQ(indexer.num_faces_per_orientation[Indexer::AXIAL], 6U);
    EXPECT_EQ(indexer.num_faces_per_orientation[Indexer::SIDE], 10U);
    EXPECT_EQ(indexer.face_offsets[Indexer::SIDE], 6U);
    EXPECT_EQ(indexer.total_cells(), 6U);
    EXPECT_EQ(indexer.total_faces(), 16U);
    EXPECT_EQ(indexer.total_nodes(), 8U);

    for (size_t id = 0; id < indexer.total_cells(); ++id)
    {
        EXPECT_EQ(indexer.cell_local_id(indexer.cell_id(id)), id);
    }
    for (size_t id = 0; id < indexer.total_faces(); ++id)
    {
        EXPECT_EQ(indexer.face_local_id(indexer.face_id(id)), id);
    }
    for (size_t id = 0; id < indexer.total_nodes(); ++id)
    {
        EXPECT_EQ(indexer.node_local_id(indexer.node_id(id)), id);
    }
}

} // namespace
