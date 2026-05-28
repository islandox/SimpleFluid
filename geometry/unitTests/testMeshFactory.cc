/**
 * @file testMeshFactory.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for MeshFactory class
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include "geometry/MeshFactory.hh"

#include "utils/testing_environment.hh"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Create a database for a 27-cell hexahedral box mesh.
 *
 * @return Shared pointer to a configured Database.
 */
SimpleFluid::SP<const SimpleFluid::Database> make_27_hex_box_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_boundary_layer_box_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.25});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"xmin", "xmax"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.1, 0.2});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_boundary_layer_cylinder_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.5});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("height", SimpleFluid::real_t{2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.1, 0.2, 0.2});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0, 1.0});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_boundary_layer_sphere_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.5});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::SPHERE));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"lower_surface", "upper_surface"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"lower_surface", "upper_surface"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.1, 0.1});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_cylinder_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("height", SimpleFluid::real_t{2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_sphere_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::SPHERE));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"surface"});

    return db;
}

SimpleFluid::SP<const SimpleFluid::Database> make_split_sphere_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::SPHERE));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"lower_surface", "upper_surface"});

    return db;
}

} // namespace

/**
 * @brief Verifies a structured 27-hex BOX mesh has correct cell counts, types, centroids, and boundary faces.
 */
TEST(MeshFactoryTest, BoxBuildsStructuredHex8STKMesh)
{
    auto db = make_27_hex_box_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_EQ(mesh->spatial_dimension(), 3u);
    EXPECT_EQ(mesh->num_local_cells(), 27u);
    EXPECT_EQ(mesh->num_owned_cells(), 27u);
    EXPECT_EQ(mesh->num_faces(), 108u);
    EXPECT_EQ(mesh->global_to_local_cell(1), 0);
    EXPECT_EQ(mesh->global_to_local_cell(2), 1);

    EXPECT_EQ(mesh->cell(0).type, MeshType::CellType::HEXAHEDRON);
    EXPECT_EQ(mesh->cell(1).type, MeshType::CellType::HEXAHEDRON);
    EXPECT_DOUBLE_EQ(mesh->cell_volume(0), 1.0);
    EXPECT_DOUBLE_EQ(mesh->cell_volume(1), 1.0);
    EXPECT_EQ(mesh->cell_centroid(0), (SimpleFluid::vec3{0.5, 0.5, 0.5}));
    EXPECT_EQ(mesh->cell_centroid(1), (SimpleFluid::vec3{1.5, 0.5, 0.5}));
    EXPECT_EQ(mesh->cell_centroid(26), (SimpleFluid::vec3{2.5, 2.5, 2.5}));

    auto has_face_centroid = [&](const MeshType::Vec3& expected)
    {
        for (MeshType::local_ordinal_type fid = 0;
             fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
             ++fid)
        {
            if (mesh->face_centroid(fid) == expected)
            {
                return true;
            }
        }
        return false;
    };

    EXPECT_TRUE(has_face_centroid(SimpleFluid::vec3{0.0, 0.5, 0.5}));
    EXPECT_TRUE(has_face_centroid(SimpleFluid::vec3{1.0, 0.5, 0.5}));
    EXPECT_TRUE(has_face_centroid(SimpleFluid::vec3{2.0, 0.5, 0.5}));

    std::unordered_map<std::string, std::size_t> boundary_counts;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            const auto& name = mesh->boundary_name(fid);
            const auto& center = mesh->face_centroid(fid);
            ++boundary_counts[name];

            if (name == "xmin") EXPECT_DOUBLE_EQ(center.x, 0.0);
            if (name == "xmax") EXPECT_DOUBLE_EQ(center.x, 3.0);
            if (name == "ymin") EXPECT_DOUBLE_EQ(center.y, 0.0);
            if (name == "ymax") EXPECT_DOUBLE_EQ(center.y, 3.0);
            if (name == "zmin") EXPECT_DOUBLE_EQ(center.z, 0.0);
            if (name == "zmax") EXPECT_DOUBLE_EQ(center.z, 3.0);
        }
    }

    EXPECT_EQ(boundary_counts["xmin"], 9u);
    EXPECT_EQ(boundary_counts["xmax"], 9u);
    EXPECT_EQ(boundary_counts["ymin"], 9u);
    EXPECT_EQ(boundary_counts["ymax"], 9u);
    EXPECT_EQ(boundary_counts["zmin"], 9u);
    EXPECT_EQ(boundary_counts["zmax"], 9u);
}

/**
 * @brief Verifies boundary layer refinement on a BOX mesh re-grids selected axis edges with graded spacing.
 */
TEST(MeshFactoryTest, BoxBoundaryLayersRegenerateSelectedAxisEdges)
{
    auto db = make_boundary_layer_box_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_EQ(mesh->num_local_cells(), 4u);
    EXPECT_EQ(mesh->cell_centroid(0), (SimpleFluid::vec3{0.05, 0.5, 0.5}));
    EXPECT_NEAR(mesh->cell_centroid(3).x, 0.9, 1.0e-12);
    EXPECT_DOUBLE_EQ(mesh->cell_centroid(3).y, 0.5);
    EXPECT_DOUBLE_EQ(mesh->cell_centroid(3).z, 0.5);
    EXPECT_DOUBLE_EQ(mesh->cell_volume(0), 0.1);
    EXPECT_DOUBLE_EQ(mesh->cell_volume(3), 0.2);
}

TEST(MeshFactoryTest, BoundaryLayersRejectOverlappingOppositeSides)
{
    auto db = std::const_pointer_cast<SimpleFluid::Database>(
        make_boundary_layer_box_database());
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{2, 2});

    SimpleFluid::MeshFactory factory(db);
    EXPECT_THROW(factory.template build<>(), std::runtime_error);
}

/**
 * @brief Verifies a wedge-mesh cylinder has correct cell counts, all cells are TRIPRISM, and boundary parts are properly assigned.
 */
TEST(MeshFactoryTest, CylinderBuildsWedgeMeshWithBoundaryParts)
{
    auto db = make_cylinder_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_EQ(mesh->spatial_dimension(), 3u);
    EXPECT_EQ(mesh->num_local_cells(), 16u);
    EXPECT_EQ(mesh->num_owned_cells(), 16u);

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_local_cells());
         ++lid)
    {
        EXPECT_EQ(mesh->cell(lid).type, MeshType::CellType::TRIPRISM);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
    }

    std::unordered_map<std::string, std::size_t> boundary_counts;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            ++boundary_counts[mesh->boundary_name(fid)];
        }
    }

    EXPECT_EQ(boundary_counts["radial"], 16u);
    EXPECT_EQ(boundary_counts["zmin"], 8u);
    EXPECT_EQ(boundary_counts["zmax"], 8u);
}

/**
 * @brief Verifies boundary layer refinement on a cylinder mesh produces all wedge cells with positive volumes.
 */
TEST(MeshFactoryTest, CylinderBoundaryLayersBuildPositiveWedgeMesh)
{
    auto db = make_boundary_layer_cylinder_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_GT(mesh->num_local_cells(), 0u);
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_local_cells());
         ++lid)
    {
        EXPECT_EQ(mesh->cell(lid).type, MeshType::CellType::TRIPRISM);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
    }
}

/**
 * @brief Verifies a spherified-cube mesh has correct cell count, type, volumes, and a single surface boundary patch.
 */
TEST(MeshFactoryTest, SphereBuildsSpherifiedHexMeshWithSurfacePatch)
{
    auto db = make_sphere_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_EQ(mesh->spatial_dimension(), 3u);
    EXPECT_EQ(mesh->num_local_cells(), 8u);
    EXPECT_EQ(mesh->num_owned_cells(), 8u);

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_local_cells());
         ++lid)
    {
        EXPECT_EQ(mesh->cell(lid).type, MeshType::CellType::HEXAHEDRON);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
    }

    std::size_t surface_faces = 0;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_EQ(mesh->boundary_name(fid), "surface");
            ++surface_faces;
        }
    }

    EXPECT_EQ(surface_faces, 24u);
}

/**
 * @brief Verifies a sphere can split its surface into lower and upper thermal boundary patches.
 */
TEST(MeshFactoryTest, SphereBuildsSplitSurfacePatches)
{
    auto db = make_split_sphere_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);

    std::unordered_map<std::string, std::size_t> boundary_counts;
    std::size_t boundary_faces = 0;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            const auto& name = mesh->boundary_name(fid);
            EXPECT_TRUE(name == "lower_surface" || name == "upper_surface");
            ++boundary_counts[name];
            ++boundary_faces;
        }
    }

    EXPECT_EQ(boundary_faces, 24u);
    EXPECT_EQ(boundary_counts["lower_surface"], 12u);
    EXPECT_EQ(boundary_counts["upper_surface"], 12u);
}

/**
 * @brief Verifies boundary layer refinement on a split-surface sphere produces the expected cell count.
 */
TEST(MeshFactoryTest, SphereBoundaryLayersBuildPositiveSplitSurfaceMesh)
{
    auto db = make_boundary_layer_sphere_database();
    SimpleFluid::MeshFactory factory(db);

    auto mesh = factory.template build<>();

    ASSERT_TRUE(mesh != nullptr);
    EXPECT_EQ(mesh->num_local_cells(), 64u);
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_local_cells());
         ++lid)
    {
        EXPECT_EQ(mesh->cell(lid).type, MeshType::CellType::HEXAHEDRON);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
    }
}
