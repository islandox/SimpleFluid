/**
 * @file testOrthoMeshTopo.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for shared orthogonal-mesh topology queries.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <algorithm>
#include <initializer_list>
#include <ranges>
#include <stdexcept>

namespace
{

using Topology = SimpleFluid::Meshes::OrthoMeshTopo;
using Indexer = Topology::Indexer;
using CellID = Topology::CellID;
using FaceID = Topology::FaceID;

Topology::NeighborCells neighbor_cells(
    std::initializer_list<CellID> cells)
{
    Topology::NeighborCells result;
    result.num = static_cast<unsigned>(cells.size());
    std::ranges::copy(cells, result.neighbors.begin());
    return result;
}

Topology make_topology(
    bool i_periodic = false,
    bool j_periodic = false,
    bool k_periodic = false)
{
    return Topology(
        Indexer(2, 3, 4, i_periodic, j_periodic, k_periodic),
        {{"imin", "imax", "jmin", "jmax", "kmin", "kmax"}});
}

} // namespace

TEST(OrthoMeshTopoTest, QueriesNonPeriodicOwnerAndNeighborCells)
{
    const auto topology = make_topology();

    const FaceID i_lower{0, 1, 2, Indexer::I_FACE};
    EXPECT_EQ(topology.owner_cell(i_lower), (CellID{0, 1, 2}));
    EXPECT_EQ(topology.neighbor_cell(i_lower), CellID{});
    EXPECT_EQ(topology.boundary_id(i_lower), 0);

    const FaceID i_interior{1, 1, 2, Indexer::I_FACE};
    EXPECT_EQ(topology.owner_cell(i_interior), (CellID{0, 1, 2}));
    EXPECT_EQ(topology.neighbor_cell(i_interior), (CellID{1, 1, 2}));
    EXPECT_EQ(
        topology.boundary_id(i_interior),
        Topology::invalid_boundary_id);

    const FaceID j_upper{1, 3, 2, Indexer::J_FACE};
    EXPECT_EQ(topology.owner_cell(j_upper), (CellID{1, 2, 2}));
    EXPECT_EQ(topology.neighbor_cell(j_upper), CellID{});
    EXPECT_EQ(topology.boundary_id(j_upper), 3);

    const FaceID k_interior{1, 2, 3, Indexer::K_FACE};
    EXPECT_EQ(topology.owner_cell(k_interior), (CellID{1, 2, 2}));
    EXPECT_EQ(topology.neighbor_cell(k_interior), (CellID{1, 2, 3}));
    EXPECT_FALSE(topology.is_boundary_face(k_interior));
}

TEST(OrthoMeshTopoTest, BuildsBoundaryFaceBatches)
{
    const auto topology = make_topology();

    EXPECT_EQ(topology.num_boundary_batches(), 6);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(0)), 12);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(1)), 12);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(2)), 8);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(3)), 8);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(4)), 6);
    EXPECT_EQ(
        std::ranges::distance(topology.boundary_face_batch(5)), 6);

    EXPECT_EQ(topology.boundary_batch_name(0), "imin");
    EXPECT_EQ(topology.boundary_batch_name(5), "kmax");
    EXPECT_THROW(topology.boundary_face_batch(6), std::out_of_range);
    EXPECT_THROW(topology.boundary_batch_name(6), std::out_of_range);
}

TEST(OrthoMeshTopoTest, CachesInteriorBatchAndCellNeighbors)
{
    const Topology topology(
        Indexer(3, 3, 3),
        {{"imin", "imax", "jmin", "jmax", "kmin", "kmax"}});

    auto interior = topology.interior_cell_batch();
    ASSERT_EQ(std::ranges::distance(interior), 1);
    EXPECT_EQ(*std::ranges::begin(interior), (CellID{1, 1, 1}));
    EXPECT_EQ(
        topology.neighbor_cells(CellID{1, 1, 1}),
        neighbor_cells({
            {0, 1, 1},
            {2, 1, 1},
            {1, 0, 1},
            {1, 2, 1},
            {1, 1, 0},
            {1, 1, 2}}));

    EXPECT_EQ(
        topology.neighbor_cells(CellID{0, 0, 0}),
        neighbor_cells({
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}}));
}

TEST(OrthoMeshTopoTest, WrapsPeriodicDimensions)
{
    const auto topology = make_topology(true, true, true);

    const FaceID i_seam{0, 1, 2, Indexer::I_FACE};
    EXPECT_EQ(topology.owner_cell(i_seam), (CellID{1, 1, 2}));
    EXPECT_EQ(topology.neighbor_cell(i_seam), (CellID{0, 1, 2}));

    const FaceID j_seam{1, 0, 2, Indexer::J_FACE};
    EXPECT_EQ(topology.owner_cell(j_seam), (CellID{1, 2, 2}));
    EXPECT_EQ(topology.neighbor_cell(j_seam), (CellID{1, 0, 2}));

    const FaceID k_seam{1, 2, 0, Indexer::K_FACE};
    EXPECT_EQ(topology.owner_cell(k_seam), (CellID{1, 2, 3}));
    EXPECT_EQ(topology.neighbor_cell(k_seam), (CellID{1, 2, 0}));

    EXPECT_FALSE(topology.is_boundary_face(i_seam));
    EXPECT_FALSE(topology.is_boundary_face(j_seam));
    EXPECT_FALSE(topology.is_boundary_face(k_seam));
    EXPECT_EQ(topology.num_boundary_batches(), 0);
    EXPECT_EQ(
        std::ranges::distance(topology.interior_cell_batch()),
        topology.indexer().total_cells());
    EXPECT_EQ(
        topology.neighbor_cells(CellID{0, 0, 0}),
        neighbor_cells({
            {1, 0, 0},
            {1, 0, 0},
            {0, 2, 0},
            {0, 1, 0},
            {0, 0, 3},
            {0, 0, 1}}));
}

TEST(OrthoMeshTopoTest, HandlesSingleCellDimensions)
{
    const Topology non_periodic(
        Indexer(1, 2, 1),
        {{"imin", "imax", "jmin", "jmax", "kmin", "kmax"}});
    const FaceID i_lower{0, 0, 0, Indexer::I_FACE};
    const FaceID i_upper{1, 0, 0, Indexer::I_FACE};
    EXPECT_EQ(non_periodic.owner_cell(i_lower), (CellID{0, 0, 0}));
    EXPECT_EQ(non_periodic.owner_cell(i_upper), (CellID{0, 0, 0}));
    EXPECT_EQ(non_periodic.neighbor_cell(i_lower), CellID{});
    EXPECT_EQ(non_periodic.neighbor_cell(i_upper), CellID{});
    EXPECT_EQ(
        non_periodic.neighbor_cells(CellID{0, 0, 0}),
        neighbor_cells({{0, 1, 0}}));

    const Topology periodic(
        Indexer(1, 1, 1, true, true, true),
        {{"imin", "imax", "jmin", "jmax", "kmin", "kmax"}});
    const FaceID periodic_seam{0, 0, 0, Indexer::K_FACE};
    EXPECT_EQ(periodic.owner_cell(periodic_seam), (CellID{0, 0, 0}));
    EXPECT_EQ(
        periodic.neighbor_cell(periodic_seam),
        (CellID{0, 0, 0}));
    EXPECT_EQ(
        periodic.neighbor_cells(CellID{0, 0, 0}),
        neighbor_cells({
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}}));
}
