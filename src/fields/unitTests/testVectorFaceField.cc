/**
 * @file testVectorFaceField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for VectorFaceField.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "fields/VectorFaceField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::VectorFaceField<Pack>;
using Vec3 = MeshType::Vec3;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the assembled two-cell mesh shared by vector-face tests. */
SimpleFluid::SP<MeshType> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

/** @brief Minimal mesh fixture with one locally owned face. */
class MinimalFaceOwnershipMesh : public MeshType
{
public:
    void populate()
    {
        d_cells.resize(2);
        d_cells[0].owned = true;
        d_cells[1].owned = false;

        d_owned_cell_ids = {0};
        d_owned_cell_global_ids = {1};
        d_cell_gid_to_lid = {{1, 0}, {2, 1}};

        d_faces.resize(2);
        d_faces[0].owner = 0;
        d_faces[1].owner = 1;

        create_maps();
    }

    void assemble() override
    {
    }

    void export_vtu(const std::string& filename) const override
    {
        throw std::runtime_error(filename + " export not implemented");
    }
};

} // namespace

/** @brief Verifies vector components and global/local face access agree. */
TEST(VectorFaceFieldTest, StoresThreeComponentsOnOwnedFaceMap)
{
    auto mesh = make_two_hex_mesh();

    FieldType velocity(mesh, "boundary_velocity");

    EXPECT_EQ(velocity.name(), "boundary_velocity");
    EXPECT_EQ(velocity.mesh_ptr(), mesh);
    EXPECT_EQ(velocity.num_owned_faces(), mesh->num_faces());
    EXPECT_EQ(velocity.map()->getLocalNumElements(), mesh->num_faces());
    EXPECT_EQ(velocity.data().getNumVectors(), FieldType::num_components);

    velocity.set_value(0, Vec3{1.0, 2.0, 3.0});
    velocity.set_component_value(0, 2, 4.0);

    EXPECT_EQ(velocity.value(0), (Vec3{1.0, 2.0, 4.0}));
    EXPECT_DOUBLE_EQ(velocity.component_value(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(velocity.component_value(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(velocity.component_value(0, 2), 4.0);

    const auto gid = velocity.face_global_id(0);
    velocity.set_global_value(gid, Vec3{5.0, 6.0, 7.0});
    EXPECT_EQ(velocity.global_value(gid), (Vec3{5.0, 6.0, 7.0}));
}

/** @brief Confirms uniform construction fills all components on every face. */
TEST(VectorFaceFieldTest, InitialValueConstructorFillsAllComponents)
{
    auto mesh = make_two_hex_mesh();

    FieldType velocity(mesh, Vec3{1.0, -2.0, 3.5}, "initial_velocity");

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        EXPECT_EQ(velocity.value(fid), (Vec3{1.0, -2.0, 3.5}));
    }
}

/** @brief Verifies component-wise bulk host access on owned faces. */
TEST(VectorFaceFieldTest, ExposesComponentWiseHostViews)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, "face_velocity");

    {
        auto values = velocity.owned_write_view();
        const auto row = velocity.owned_row(0);
        for (size_t component = 0;
             component < FieldType::num_components; ++component)
        {
            values(row, component) =
                static_cast<double>(component + 1);
        }
    }

    const auto values = velocity.owned_read_view();
    const auto row = velocity.owned_row(0);
    EXPECT_DOUBLE_EQ(values(row, 0), 1.0);
    EXPECT_DOUBLE_EQ(values(row, 1), 2.0);
    EXPECT_DOUBLE_EQ(values(row, 2), 3.0);
}

/** @brief Checks that the face map follows owner-cell ownership. */
TEST(VectorFaceFieldTest, StoresOnlyFacesWhoseOwnerCellIsOwned)
{
    auto mesh = std::make_shared<MinimalFaceOwnershipMesh>();
    mesh->populate();

    FieldType velocity(mesh, "owned_velocity");

    ASSERT_EQ(velocity.num_owned_faces(), 1u);
    ASSERT_EQ(velocity.owned_face_ids().size(), 1u);
    EXPECT_EQ(velocity.owned_face_ids()[0], 0);
    EXPECT_TRUE(velocity.is_owned_face(0));
    EXPECT_FALSE(velocity.is_owned_face(1));

    velocity.set_value(0, Vec3{7.0, 8.0, 9.0});
    EXPECT_EQ(velocity.value(0), (Vec3{7.0, 8.0, 9.0}));
#ifndef NDEBUG
    EXPECT_THROW(velocity.value(1), std::out_of_range);
#endif
}

/** @brief Ensures construction rejects a mesh without assembled maps. */
TEST(VectorFaceFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
