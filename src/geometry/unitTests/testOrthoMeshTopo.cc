/**
 * @file testOrthoMeshTopo.cc
 * @brief Unit tests for shared orthogonal-mesh topology queries.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <stdexcept>

namespace
{

using Topology = SimpleFluid::Mesh::OrthoMeshTopo;
using Indexer = Topology::Indexer;
using CellID = Topology::CellID;
using FaceID = Topology::FaceID;

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

TEST(OrthoMeshTopoTest, BuildsBoundaryFacePatches)
{
    const auto topology = make_topology();

    EXPECT_EQ(topology.boundary_patches().size(), 6U);
    EXPECT_EQ(topology.boundary_face_patch(0).face_lids.size(), 12U);
    EXPECT_EQ(topology.boundary_face_patch(1).face_lids.size(), 12U);
    EXPECT_EQ(topology.boundary_face_patch(2).face_lids.size(), 8U);
    EXPECT_EQ(topology.boundary_face_patch(3).face_lids.size(), 8U);
    EXPECT_EQ(topology.boundary_face_patch(4).face_lids.size(), 6U);
    EXPECT_EQ(topology.boundary_face_patch(5).face_lids.size(), 6U);

    EXPECT_EQ(topology.boundary_patch_name(0), "imin");
    EXPECT_EQ(topology.boundary_patch_name(5), "kmax");
    EXPECT_THROW(topology.boundary_face_patch(6), std::out_of_range);
    EXPECT_THROW(topology.boundary_patch_name(6), std::out_of_range);
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
    EXPECT_TRUE(topology.boundary_patches().empty());
}
