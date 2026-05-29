/**
 * @file testMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for Mesh class
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/Mesh.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/MeshUtils.hh"

#include "utils/testing_environment.hh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;
using Pack = SimpleFluid::TpetraTypes<>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

// ---------------------------------------------------------------------------
// Helper: build a small 2x2x2 hex mesh (8 cells)
// ---------------------------------------------------------------------------
SimpleFluid::SP<MeshType> make_2x2x2_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    SimpleFluid::MeshFactory factory(db);
    return factory.build<Pack>();
}

} // namespace

TEST(MeshTest, InvalidIdUsesNegativeOneForSignedTypes)
{
    EXPECT_EQ(SimpleFluid::invalid_id<int>(), -1);
    EXPECT_EQ(SimpleFluid::invalid_id<long>(), -1L);
}

TEST(MeshTest, InvalidIdUsesMaxForUnsignedTypes)
{
    EXPECT_EQ(SimpleFluid::invalid_id<unsigned>(),
              std::numeric_limits<unsigned>::max());
    EXPECT_EQ(SimpleFluid::invalid_id<std::size_t>(),
              std::numeric_limits<std::size_t>::max());
}

TEST(MeshTest, InvalidBoundaryIdMatchesInvalidInt)
{
    EXPECT_EQ(MeshType::invalid_boundary_id, SimpleFluid::invalid_id<int>());
}

TEST(MeshTest, VtuCellTypeCodeMapsSupportedCells)
{
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::HEXAHEDRON),
              12);
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::TRIPRISM),
              13);
}

TEST(MeshTest, VtuCellTypeCodeRejectsUnsupportedCells)
{
    EXPECT_THROW(SimpleFluid::MeshUtils::vtu_cell_type_code(
                     SimpleFluid::MeshUtils::CellType::INVALID),
                 std::runtime_error);
}

// ===========================================================================
// Mesh class method tests — uses MeshFactory to build a concrete mesh
// ===========================================================================

class MeshMethodTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mesh_ = make_2x2x2_mesh();
        ASSERT_NE(mesh_, nullptr);
    }

    SimpleFluid::SP<MeshType> mesh_;
};

TEST_F(MeshMethodTest, SpatialDimensionIsThree)
{
    EXPECT_EQ(mesh_->spatial_dimension(), 3UL);
}

TEST_F(MeshMethodTest, HasOwnedCells)
{
    EXPECT_GT(mesh_->num_owned_cells(), 0UL);
}

TEST_F(MeshMethodTest, OwnedCellsDoNotExceedLocalCells)
{
    EXPECT_LE(mesh_->num_owned_cells(), mesh_->num_local_cells());
}

TEST_F(MeshMethodTest, HasFaces)
{
    EXPECT_GT(mesh_->num_faces(), 0UL);
}

TEST_F(MeshMethodTest, CellAccessDoesNotThrowForValidLids)
{
    for (std::size_t i = 0; i < mesh_->num_local_cells(); ++i)
    {
        const auto lid = static_cast<SimpleFluid::local_index_t>(i);
        EXPECT_NO_THROW(mesh_->cell(lid));
    }
}

TEST_F(MeshMethodTest, CellAccessThrowsForOutOfBoundsLid)
{
    const auto bad_lid = static_cast<SimpleFluid::local_index_t>(
        mesh_->num_local_cells() + 100);
    EXPECT_THROW(mesh_->cell(bad_lid), std::out_of_range);
}

TEST_F(MeshMethodTest, CellAccessThrowsForNegativeLid)
{
    EXPECT_THROW(mesh_->cell(-1), std::out_of_range);
}

TEST_F(MeshMethodTest, FaceAccessDoesNotThrowForValidLids)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto lid = static_cast<SimpleFluid::local_index_t>(i);
        EXPECT_NO_THROW(mesh_->face(lid));
    }
}

TEST_F(MeshMethodTest, FaceAccessThrowsForOutOfBoundsLid)
{
    const auto bad_lid = static_cast<SimpleFluid::local_index_t>(
        mesh_->num_faces() + 100);
    EXPECT_THROW(mesh_->face(bad_lid), std::out_of_range);
}

TEST_F(MeshMethodTest, CellVolumeIsPositive)
{
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        EXPECT_GT(mesh_->cell_volume(static_cast<SimpleFluid::local_index_t>(i)),
                  0.0);
    }
}

TEST_F(MeshMethodTest, CellCentroidIsFinite)
{
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto& c = mesh_->cell_centroid(
            static_cast<SimpleFluid::local_index_t>(i));
        EXPECT_TRUE(std::isfinite(c.x));
        EXPECT_TRUE(std::isfinite(c.y));
        EXPECT_TRUE(std::isfinite(c.z));
    }
}

TEST_F(MeshMethodTest, FaceAreaIsPositiveForAllFaces)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        EXPECT_GT(mesh_->face_area(static_cast<SimpleFluid::local_index_t>(i)),
                  0.0);
    }
}

TEST_F(MeshMethodTest, FaceCentroidIsFinite)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto& c = mesh_->face_centroid(
            static_cast<SimpleFluid::local_index_t>(i));
        EXPECT_TRUE(std::isfinite(c.x));
        EXPECT_TRUE(std::isfinite(c.y));
        EXPECT_TRUE(std::isfinite(c.z));
    }
}

TEST_F(MeshMethodTest, FaceNormalIsUnitOrNearUnit)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto& n = mesh_->face_normal(
            static_cast<SimpleFluid::local_index_t>(i));
        const auto mag = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        EXPECT_NEAR(mag, 1.0, 1e-12);
    }
}

TEST_F(MeshMethodTest, OwnerCellIsValid)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        const auto owner = mesh_->owner_cell(fid);
        EXPECT_GE(owner, 0);
        EXPECT_LT(static_cast<std::size_t>(owner), mesh_->num_local_cells());
    }
}

TEST_F(MeshMethodTest, ExteriorFacesHaveInvalidNeighbor)
{
    std::size_t exterior_count = 0;
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_exterior_face(fid))
        {
            ++exterior_count;
            EXPECT_EQ(mesh_->neighbor_cell(fid),
                      SimpleFluid::invalid_id<SimpleFluid::local_index_t>());
        }
    }
    EXPECT_GT(exterior_count, 0UL)
        << "Expected at least some exterior faces on a box mesh";
}

TEST_F(MeshMethodTest, InteriorFacesHaveValidNeighbor)
{
    std::size_t interior_count = 0;
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_interior_face(fid))
        {
            ++interior_count;
            const auto neighbor = mesh_->neighbor_cell(fid);
            EXPECT_NE(neighbor,
                      SimpleFluid::invalid_id<SimpleFluid::local_index_t>());
        }
    }
    EXPECT_GT(interior_count, 0UL)
        << "Expected at least some interior faces on a 2x2x2 mesh";
}

TEST_F(MeshMethodTest, OppositeCellReturnsCorrectNeighbor)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_interior_face(fid))
        {
            const auto owner = mesh_->owner_cell(fid);
            const auto neighbor = mesh_->neighbor_cell(fid);
            EXPECT_EQ(mesh_->opposite_cell(fid, owner), neighbor);
            EXPECT_EQ(mesh_->opposite_cell(fid, neighbor), owner);
        }
    }
}

TEST_F(MeshMethodTest, OppositeCellThrowsForNonAdjacentCell)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_interior_face(fid))
        {
            const auto owner = mesh_->owner_cell(fid);
            const auto neighbor = mesh_->neighbor_cell(fid);
            for (std::size_t c = 0; c < mesh_->num_local_cells(); ++c)
            {
                const auto clid = static_cast<SimpleFluid::local_index_t>(c);
                if (clid != owner && clid != neighbor)
                {
                    EXPECT_THROW(mesh_->opposite_cell(fid, clid),
                                 std::invalid_argument);
                    return;
                }
            }
        }
    }
}

TEST_F(MeshMethodTest, BoundaryFacesHaveBoundaryIds)
{
    std::size_t boundary_count = 0;
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_boundary_face(fid))
        {
            ++boundary_count;
            EXPECT_NE(mesh_->boundary_id(fid),
                      MeshType::invalid_boundary_id);
            EXPECT_NO_THROW(mesh_->boundary_name(fid));
        }
    }
    EXPECT_GT(boundary_count, 0UL)
        << "Expected boundary faces on a box mesh";
}

TEST_F(MeshMethodTest, BoundaryNameThrowsForNonBoundaryFace)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (!mesh_->is_boundary_face(fid))
        {
            EXPECT_THROW(mesh_->boundary_name(fid), std::out_of_range);
            return;
        }
    }
}

TEST_F(MeshMethodTest, ExteriorFacesIncludeBoundaryFaces)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_boundary_face(fid))
        {
            EXPECT_TRUE(mesh_->is_exterior_face(fid));
        }
    }
}

TEST_F(MeshMethodTest, IsOwnedCellMatchesOwnedRange)
{
    for (std::size_t i = 0; i < mesh_->num_local_cells(); ++i)
    {
        const auto lid = static_cast<SimpleFluid::local_index_t>(i);
        if (i < mesh_->num_owned_cells())
        {
            EXPECT_TRUE(mesh_->is_owned_cell(lid));
        }
    }
}

TEST_F(MeshMethodTest, GlobalToLocalCellRoundtrips)
{
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto lid = static_cast<SimpleFluid::local_index_t>(i);
        const auto gid = mesh_->cell_global_id(lid);
        EXPECT_EQ(mesh_->global_to_local_cell(gid), lid);
    }
}

TEST_F(MeshMethodTest, GlobalToLocalCellReturnsInvalidForUnknownGid)
{
    EXPECT_EQ(mesh_->global_to_local_cell(999999999),
              SimpleFluid::invalid_id<SimpleFluid::local_index_t>());
}

TEST_F(MeshMethodTest, CellFaceDistanceIsPositive)
{
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto clid = static_cast<SimpleFluid::local_index_t>(i);
        const auto& cell_faces = mesh_->faces(clid);
        for (std::size_t j = 0; j < cell_faces.size(); ++j)
        {
            const auto fid = cell_faces[j];
            const auto d = mesh_->cell_to_face_distance(fid, clid);
            EXPECT_GT(d, 0.0);
        }
    }
}

TEST_F(MeshMethodTest, FaceCellCenterDistanceIsPositiveForInteriorFaces)
{
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<SimpleFluid::local_index_t>(i);
        if (mesh_->is_interior_face(fid))
        {
            EXPECT_GT(mesh_->face_cell_center_distance(fid), 0.0);
        }
    }
}

TEST_F(MeshMethodTest, OwnedCellMapIsNotNull)
{
    EXPECT_FALSE(mesh_->owned_cell_map().is_null());
}

TEST_F(MeshMethodTest, OverlapCellMapIsNotNull)
{
    EXPECT_FALSE(mesh_->overlap_cell_map().is_null());
}

TEST_F(MeshMethodTest, DeviceViewsAreAccessible)
{
    const auto views = mesh_->device_views();
    EXPECT_NO_THROW(static_cast<void>(views.cell_volume.extent(0)));
}

// ===========================================================================
// MeshUtils function tests
// ===========================================================================

TEST(MeshUtilsTest, AverageOfPointsComputesCentroid)
{
    std::vector<SimpleFluid::MeshUtils::Vec3> pts = {
        {0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 2, 0}
    };
    auto avg = SimpleFluid::MeshUtils::average(pts);
    EXPECT_DOUBLE_EQ(avg.x, 1.0);
    EXPECT_DOUBLE_EQ(avg.y, 1.0);
    EXPECT_DOUBLE_EQ(avg.z, 0.0);
}

TEST(MeshUtilsTest, AverageOfEmptyPointsReturnsZero)
{
    std::vector<SimpleFluid::MeshUtils::Vec3> pts;
    auto avg = SimpleFluid::MeshUtils::average(pts);
    EXPECT_DOUBLE_EQ(avg.x, 0.0);
    EXPECT_DOUBLE_EQ(avg.y, 0.0);
    EXPECT_DOUBLE_EQ(avg.z, 0.0);
}

TEST(MeshUtilsTest, TetraVolumeIsCorrectForUnitTetrahedron)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    auto vol = SimpleFluid::MeshUtils::tetra_volume(
        Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1});
    EXPECT_DOUBLE_EQ(vol, 1.0 / 6.0);
}

TEST(MeshUtilsTest, TetraVolumeIsZeroForDegenerateTetrahedron)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    auto vol = SimpleFluid::MeshUtils::tetra_volume(
        Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 1, 0});
    EXPECT_DOUBLE_EQ(vol, 0.0);
}

TEST(MeshUtilsTest, HexVolumeIsCorrectForUnitCube)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> cube = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    auto vol = SimpleFluid::MeshUtils::hex_volume(cube);
    EXPECT_DOUBLE_EQ(vol, 1.0);
}

TEST(MeshUtilsTest, WedgeVolumeIsCorrectForUnitWedge)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> wedge = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}
    };
    auto vol = SimpleFluid::MeshUtils::wedge_volume(wedge);
    EXPECT_DOUBLE_EQ(vol, 0.5); // area of right triangle = 0.5, height = 1
}

TEST(MeshUtilsTest, FaceAreaVectorTriangleIsCorrect)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> tri = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    auto av = SimpleFluid::MeshUtils::face_area_vector(tri);
    EXPECT_DOUBLE_EQ(av.x, 0.0);
    EXPECT_DOUBLE_EQ(av.y, 0.0);
    EXPECT_DOUBLE_EQ(av.z, 0.5); // area = 0.5, normal = +z
}

TEST(MeshUtilsTest, FaceAreaVectorQuadIsCorrect)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> quad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    auto av = SimpleFluid::MeshUtils::face_area_vector(quad);
    EXPECT_DOUBLE_EQ(av.x, 0.0);
    EXPECT_DOUBLE_EQ(av.y, 0.0);
    EXPECT_DOUBLE_EQ(av.z, 1.0); // area = 1.0, normal = +z
}

TEST(MeshUtilsTest, FaceAreaVectorThrowsForWrongSize)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> bad = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}};
    EXPECT_THROW(SimpleFluid::MeshUtils::face_area_vector(bad),
                 std::runtime_error); // from CHECK macro (default exception type)
}
