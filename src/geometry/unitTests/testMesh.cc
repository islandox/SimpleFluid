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

TEST(MeshTest, InvalidIdAndVtuCellType)
{
    // invalid_id
    EXPECT_EQ(SimpleFluid::invalid_id<int>(), -1);
    EXPECT_EQ(SimpleFluid::invalid_id<long>(), -1L);
    EXPECT_EQ(SimpleFluid::invalid_id<unsigned>(),
              std::numeric_limits<unsigned>::max());
    EXPECT_EQ(SimpleFluid::invalid_id<std::size_t>(),
              std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(MeshType::invalid_boundary_id, SimpleFluid::invalid_id<int>());

    // vtu_cell_type_code
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::HEXAHEDRON), 12);
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::TRIPRISM), 13);
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

TEST_F(MeshMethodTest, BasicMeshProperties)
{
    EXPECT_EQ(mesh_->spatial_dimension(), 3UL);
    EXPECT_GT(mesh_->num_owned_cells(), 0UL);
    EXPECT_LE(mesh_->num_owned_cells(), mesh_->num_local_cells());
    EXPECT_GT(mesh_->num_faces(), 0UL);
}

TEST_F(MeshMethodTest, CellAndFaceAccess)
{
    using lid_t = SimpleFluid::local_index_t;

    // Valid cell access
    for (std::size_t i = 0; i < mesh_->num_local_cells(); ++i)
        EXPECT_NO_THROW(mesh_->cell(static_cast<lid_t>(i)));

    // Out-of-bounds / negative cell access
    EXPECT_THROW(mesh_->cell(static_cast<lid_t>(mesh_->num_local_cells() + 100)),
                 std::out_of_range);
    EXPECT_THROW(mesh_->cell(-1), std::out_of_range);

    // Valid face access
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
        EXPECT_NO_THROW(mesh_->face(static_cast<lid_t>(i)));

    // Out-of-bounds face access
    EXPECT_THROW(mesh_->face(static_cast<lid_t>(mesh_->num_faces() + 100)),
                 std::out_of_range);

    // is_owned_cell matches owned range
    for (std::size_t i = 0; i < mesh_->num_local_cells(); ++i)
    {
        const auto lid = static_cast<lid_t>(i);
        if (i < mesh_->num_owned_cells())
            EXPECT_TRUE(mesh_->is_owned_cell(lid));
    }
}

TEST_F(MeshMethodTest, GeometryProperties)
{
    using lid_t = SimpleFluid::local_index_t;

    // Cell volumes and centroids
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto lid = static_cast<lid_t>(i);
        EXPECT_GT(mesh_->cell_volume(lid), 0.0);
        const auto& cc = mesh_->cell_centroid(lid);
        EXPECT_TRUE(std::isfinite(cc.x) && std::isfinite(cc.y) && std::isfinite(cc.z));
    }

    // Face areas, centroids, and normals
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<lid_t>(i);
        EXPECT_GT(mesh_->face_area(fid), 0.0);

        const auto& fc = mesh_->face_centroid(fid);
        EXPECT_TRUE(std::isfinite(fc.x) && std::isfinite(fc.y) && std::isfinite(fc.z));

        const auto& n = mesh_->face_normal(fid);
        EXPECT_NEAR(std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z), 1.0, 1e-12);
    }
}

TEST_F(MeshMethodTest, FaceTopology)
{
    using lid_t = SimpleFluid::local_index_t;
    const auto invalid = SimpleFluid::invalid_id<lid_t>();

    std::size_t exterior_count = 0, interior_count = 0, boundary_count = 0;

    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<lid_t>(i);

        // Owner cell always valid
        const auto owner = mesh_->owner_cell(fid);
        EXPECT_GE(owner, 0);
        EXPECT_LT(static_cast<std::size_t>(owner), mesh_->num_local_cells());

        if (mesh_->is_exterior_face(fid))
        {
            ++exterior_count;
            EXPECT_EQ(mesh_->neighbor_cell(fid), invalid);
        }

        if (mesh_->is_interior_face(fid))
        {
            ++interior_count;
            const auto neighbor = mesh_->neighbor_cell(fid);
            EXPECT_NE(neighbor, invalid);

            // opposite_cell correctness
            EXPECT_EQ(mesh_->opposite_cell(fid, owner), neighbor);
            EXPECT_EQ(mesh_->opposite_cell(fid, neighbor), owner);
        }

        if (mesh_->is_boundary_face(fid))
        {
            ++boundary_count;
            EXPECT_NE(mesh_->boundary_id(fid), MeshType::invalid_boundary_id);
            EXPECT_NO_THROW(mesh_->boundary_name(fid));
            EXPECT_TRUE(mesh_->is_exterior_face(fid));
        }
        else
        {
            EXPECT_THROW(mesh_->boundary_name(fid), std::out_of_range);
        }
    }

    EXPECT_GT(exterior_count, 0UL);
    EXPECT_GT(interior_count, 0UL);
    EXPECT_GT(boundary_count, 0UL);

    // opposite_cell throws for non-adjacent cell (spot-check one interior face)
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<lid_t>(i);
        if (mesh_->is_interior_face(fid))
        {
            const auto owner = mesh_->owner_cell(fid);
            const auto neighbor = mesh_->neighbor_cell(fid);
            for (std::size_t c = 0; c < mesh_->num_local_cells(); ++c)
            {
                const auto clid = static_cast<lid_t>(c);
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

TEST_F(MeshMethodTest, IdentifierMapping)
{
    using lid_t = SimpleFluid::local_index_t;

    // Global ↔ local roundtrip
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto lid = static_cast<lid_t>(i);
        EXPECT_EQ(mesh_->global_to_local_cell(mesh_->cell_global_id(lid)), lid);
    }

    // Unknown GID → invalid
    EXPECT_EQ(mesh_->global_to_local_cell(999999999),
              SimpleFluid::invalid_id<lid_t>());

    // Maps and device views
    EXPECT_FALSE(mesh_->owned_cell_map().is_null());
    EXPECT_FALSE(mesh_->overlap_cell_map().is_null());
    const auto views = mesh_->device_views();
    EXPECT_NO_THROW(static_cast<void>(views.cell_volume.extent(0)));
}

TEST_F(MeshMethodTest, DistanceComputations)
{
    using lid_t = SimpleFluid::local_index_t;

    // Cell-to-face distances
    for (std::size_t i = 0; i < mesh_->num_owned_cells(); ++i)
    {
        const auto clid = static_cast<lid_t>(i);
        for (const auto fid : mesh_->faces(clid))
            EXPECT_GT(mesh_->cell_to_face_distance(fid, clid), 0.0);
    }

    // Face-cell-center distances for interior faces
    for (std::size_t i = 0; i < mesh_->num_faces(); ++i)
    {
        const auto fid = static_cast<lid_t>(i);
        if (mesh_->is_interior_face(fid))
            EXPECT_GT(mesh_->face_cell_center_distance(fid), 0.0);
    }
}

// ===========================================================================
// MeshUtils function tests
// ===========================================================================

TEST(MeshUtilsTest, VolumeComputations)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;

    // Average of points
    {
        std::vector<Vec3> pts = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 2, 0}};
        auto avg = SimpleFluid::MeshUtils::average(pts);
        EXPECT_DOUBLE_EQ(avg.x, 1.0);
        EXPECT_DOUBLE_EQ(avg.y, 1.0);
        EXPECT_DOUBLE_EQ(avg.z, 0.0);
    }
    {
        std::vector<Vec3> empty;
        auto avg = SimpleFluid::MeshUtils::average(empty);
        EXPECT_DOUBLE_EQ(avg.x, 0.0);
        EXPECT_DOUBLE_EQ(avg.y, 0.0);
        EXPECT_DOUBLE_EQ(avg.z, 0.0);
    }

    // Tetrahedron volume
    EXPECT_DOUBLE_EQ(
        SimpleFluid::MeshUtils::tetra_volume(
            Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}),
        1.0 / 6.0);
    EXPECT_DOUBLE_EQ(
        SimpleFluid::MeshUtils::tetra_volume(
            Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 1, 0}),
        0.0);

    // Hexahedron volume (unit cube)
    {
        const std::vector<Vec3> cube = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        EXPECT_DOUBLE_EQ(SimpleFluid::MeshUtils::hex_volume(cube), 1.0);
    }

    // Wedge volume
    {
        const std::vector<Vec3> wedge = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
        EXPECT_DOUBLE_EQ(SimpleFluid::MeshUtils::wedge_volume(wedge), 0.5);
    }
}

TEST(MeshUtilsTest, FaceAreaVector)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;

    // Triangle: area = 0.5, normal = +z
    {
        const std::vector<Vec3> tri = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        auto av = SimpleFluid::MeshUtils::face_area_vector(tri);
        EXPECT_DOUBLE_EQ(av.x, 0.0);
        EXPECT_DOUBLE_EQ(av.y, 0.0);
        EXPECT_DOUBLE_EQ(av.z, 0.5);
    }
    // Quad: area = 1.0, normal = +z
    {
        const std::vector<Vec3> quad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        auto av = SimpleFluid::MeshUtils::face_area_vector(quad);
        EXPECT_DOUBLE_EQ(av.x, 0.0);
        EXPECT_DOUBLE_EQ(av.y, 0.0);
        EXPECT_DOUBLE_EQ(av.z, 1.0);
    }
}

TEST(MeshUtilsTest, FaceAreaVectorThrowsForWrongSize)
{
    using Vec3 = SimpleFluid::MeshUtils::Vec3;
    std::vector<Vec3> bad = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}};
    EXPECT_THROW(SimpleFluid::MeshUtils::face_area_vector(bad),
                 std::runtime_error); // from CHECK macro (default exception type)
}
