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

} // namespace

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

    EXPECT_EQ(mesh->face_centroid(0), (SimpleFluid::vec3{0.0, 0.5, 0.5}));
    EXPECT_EQ(mesh->face_centroid(1), (SimpleFluid::vec3{1.0, 0.5, 0.5}));
    EXPECT_EQ(mesh->face_centroid(2), (SimpleFluid::vec3{2.0, 0.5, 0.5}));

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
