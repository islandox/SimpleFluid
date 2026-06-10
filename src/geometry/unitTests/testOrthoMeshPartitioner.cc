/**
 * @file testOrthoMeshPartitioner.cc
 * @brief Unit tests for coordinate-based orthogonal mesh partitioning.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthoMeshPartitioner.hh"

#include <stdexcept>

namespace
{

using Partitioner = SimpleFluid::Mesh::OrthoMeshPartitioner;
using Topology = Partitioner::Topology;
using Indexer = Partitioner::Indexer;
using CellID = Partitioner::CellID;
using FaceID = Partitioner::FaceID;
using Range = Partitioner::CoordinateRange;

Topology make_topology(
    unsigned ni,
    unsigned nj,
    unsigned nk,
    bool periodic_i = false,
    bool periodic_j = false,
    bool periodic_k = false)
{
    return Topology(
        Indexer(
            ni,
            nj,
            nk,
            periodic_i,
            periodic_j,
            periodic_k),
        {{"imin", "imax", "jmin", "jmax", "kmin", "kmax"}});
}

std::vector<size_t> local_ids(
    const Indexer& indexer,
    const std::vector<CellID>& cells)
{
    std::vector<size_t> ids;
    ids.reserve(cells.size());
    for (const auto cell : cells)
    {
        ids.push_back(indexer.cell_local_id(cell));
    }
    return ids;
}

} // namespace

TEST(OrthoMeshPartitionerTest, BalancesCoordinateSlabs)
{
    const auto topology = make_topology(10, 2, 1);
    const Partitioner partitioner(topology, Indexer::I, 3);

    EXPECT_EQ(partitioner.coordinate(), Indexer::I);
    EXPECT_EQ(partitioner.num_partitions(), 3U);
    EXPECT_EQ(partitioner.owned_coordinate_range(0), (Range{0, 4}));
    EXPECT_EQ(partitioner.owned_coordinate_range(1), (Range{4, 7}));
    EXPECT_EQ(partitioner.owned_coordinate_range(2), (Range{7, 10}));

    EXPECT_EQ(partitioner.num_owned_cells(0), 8U);
    EXPECT_EQ(partitioner.num_owned_cells(1), 6U);
    EXPECT_EQ(partitioner.num_owned_cells(2), 6U);

    EXPECT_EQ(partitioner.owner_partition(CellID{0, 1, 0}), 0U);
    EXPECT_EQ(partitioner.owner_partition(CellID{4, 0, 0}), 1U);
    EXPECT_EQ(partitioner.owner_partition(CellID{9, 1, 0}), 2U);

    const auto owned = partitioner.owned_cells(1);
    EXPECT_EQ(
        local_ids(topology.indexer(), owned),
        (std::vector<size_t>{4, 5, 6, 14, 15, 16}));
}

TEST(OrthoMeshPartitionerTest, BuildsNonPeriodicGhostLayers)
{
    const auto topology = make_topology(2, 7, 1);
    const Partitioner partitioner(topology, Indexer::J, 3);

    EXPECT_EQ(partitioner.owned_coordinate_range(0), (Range{0, 3}));
    EXPECT_EQ(partitioner.owned_coordinate_range(1), (Range{3, 5}));
    EXPECT_EQ(partitioner.owned_coordinate_range(2), (Range{5, 7}));

    EXPECT_EQ(partitioner.num_ghost_cells(0), 2U);
    EXPECT_EQ(partitioner.num_ghost_cells(1), 4U);
    EXPECT_EQ(partitioner.num_ghost_cells(2), 2U);
    EXPECT_EQ(
        local_ids(topology.indexer(), partitioner.ghost_cells(1)),
        (std::vector<size_t>{4, 5, 10, 11}));

    EXPECT_TRUE(partitioner.is_owned_cell(1, CellID{0, 3, 0}));
    EXPECT_TRUE(partitioner.is_ghost_cell(1, CellID{0, 2, 0}));
    EXPECT_TRUE(partitioner.is_ghost_cell(1, CellID{0, 5, 0}));
    EXPECT_FALSE(partitioner.is_local_cell(1, CellID{0, 0, 0}));
}

TEST(OrthoMeshPartitionerTest, SupportsMultipleGhostLayers)
{
    const auto topology = make_topology(1, 1, 8);
    const Partitioner partitioner(topology, Indexer::K, 2, 2);

    EXPECT_EQ(partitioner.owned_coordinate_range(0), (Range{0, 4}));
    EXPECT_EQ(partitioner.owned_coordinate_range(1), (Range{4, 8}));
    EXPECT_EQ(partitioner.num_ghost_cells(0), 2U);
    EXPECT_EQ(partitioner.num_ghost_cells(1), 2U);
    EXPECT_EQ(
        local_ids(topology.indexer(), partitioner.ghost_cells(0)),
        (std::vector<size_t>{4, 5}));
    EXPECT_EQ(
        local_ids(topology.indexer(), partitioner.ghost_cells(1)),
        (std::vector<size_t>{2, 3}));
}

TEST(OrthoMeshPartitionerTest, WrapsGhostsAndFaceOwnershipPeriodically)
{
    const auto topology = make_topology(6, 1, 1, true);
    const Partitioner partitioner(topology, Indexer::I, 3);

    EXPECT_EQ(
        local_ids(topology.indexer(), partitioner.ghost_cells(0)),
        (std::vector<size_t>{2, 5}));
    EXPECT_TRUE(partitioner.is_ghost_cell(0, CellID{5, 0, 0}));

    const FaceID periodic_seam{0, 0, 0, Indexer::I_FACE};
    EXPECT_EQ(partitioner.owner_partition(periodic_seam), 2U);
    EXPECT_TRUE(partitioner.is_owned_face(2, periodic_seam));

    const FaceID first_partition_interface{
        2, 0, 0, Indexer::I_FACE};
    EXPECT_EQ(partitioner.owner_partition(first_partition_interface), 0U);
}

TEST(OrthoMeshPartitionerTest, AssignsEveryCellAndFaceExactlyOnce)
{
    const auto topology = make_topology(5, 4, 3, false, true, false);
    const Partitioner partitioner(topology, Indexer::J, 3);
    const auto& indexer = topology.indexer();

    size_t total_owned_cells = 0;
    for (size_t partition = 0;
         partition < partitioner.num_partitions();
         ++partition)
    {
        total_owned_cells += partitioner.num_owned_cells(partition);
    }
    EXPECT_EQ(total_owned_cells, indexer.total_cells());

    for (size_t local_id = 0;
         local_id < indexer.total_cells();
         ++local_id)
    {
        const auto cell_id = indexer.cell_id(local_id);
        size_t owning_partitions = 0;
        for (size_t partition = 0;
             partition < partitioner.num_partitions();
             ++partition)
        {
            owning_partitions +=
                partitioner.is_owned_cell(partition, cell_id) ? 1 : 0;
        }
        EXPECT_EQ(owning_partitions, 1U);
    }

    for (size_t local_id = 0;
         local_id < indexer.total_faces();
         ++local_id)
    {
        const auto face_id = indexer.face_id(local_id);
        size_t owning_partitions = 0;
        for (size_t partition = 0;
             partition < partitioner.num_partitions();
             ++partition)
        {
            owning_partitions +=
                partitioner.is_owned_face(partition, face_id) ? 1 : 0;
        }
        EXPECT_EQ(owning_partitions, 1U);
    }
}

TEST(OrthoMeshPartitionerTest, RejectsInvalidRequests)
{
    const auto topology = make_topology(3, 2, 1);

    EXPECT_THROW(
        Partitioner(topology, Indexer::I, 0),
        std::invalid_argument);
    EXPECT_THROW(
        Partitioner(topology, Indexer::K, 2),
        std::invalid_argument);
    EXPECT_THROW(
        Partitioner(
            topology,
            static_cast<Indexer::Dimension>(3),
            1),
        std::invalid_argument);

    const Partitioner partitioner(topology, Indexer::I, 2, 0);
    EXPECT_EQ(partitioner.num_ghost_cells(0), 0U);
    EXPECT_THROW(
        partitioner.owned_coordinate_range(2),
        std::out_of_range);
    EXPECT_THROW(
        partitioner.owner_partition(CellID{3, 0, 0}),
        std::out_of_range);
    EXPECT_THROW(
        partitioner.owner_partition(
            FaceID{0, 0, 0, static_cast<uint8_t>(3)}),
        std::out_of_range);
}
