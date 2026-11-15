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
#include <Teuchos_CommHelpers.hpp>

#include "fields/VectorFaceField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "NativeFieldOwnershipFixture.hh"
#include "utils/testing_environment.hh"

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::VectorFaceField<Pack>;
using Vec3 = Mesh::Vec3;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the assembled two-cell mesh shared by vector-face tests. */
SimpleFluid::SP<Mesh> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

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

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
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
    auto mesh = SimpleFluid::test::native_field_ownership_mesh<Pack>();
    FieldType velocity(mesh, "owned_velocity");
    size_t expected_owned = 0;
    size_t nonowned = 0;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto fid = static_cast<Pack::local_ordinal_type>(face);
        const bool owned = mesh->is_owned_cell(mesh->owner_cell(fid));
        EXPECT_EQ(velocity.is_owned_face(fid), owned);
        if (owned)
        {
            ++expected_owned;
            EXPECT_EQ(velocity.map()->getGlobalElement(velocity.owned_row(fid)), mesh->face_global_id(fid));
            velocity.set_value(fid, Vec3{7.0, 8.0, 9.0});
            EXPECT_EQ(velocity.value(fid), (Vec3{7.0, 8.0, 9.0}));
            EXPECT_EQ(velocity.local_value(fid), (Vec3{7.0, 8.0, 9.0}));
        }
        else
        {
            ++nonowned;
            EXPECT_THROW(velocity.value(fid), std::out_of_range);
        }
    }
    EXPECT_EQ(velocity.num_owned_faces(), expected_owned);
    EXPECT_EQ(velocity.owned_face_ids().size(), expected_owned);
    const auto comm = mesh->owned_cell_map()->getComm();
    int local_owned = static_cast<int>(expected_owned), global_owned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_owned, &global_owned);
    EXPECT_EQ(global_owned, 11);
    if (comm->getSize() > 1 && mesh->num_owned_cells() > 0) EXPECT_GT(nonowned, 0u);
}

/** @brief Ensures null mesh and empty descriptor names fail independently. */
TEST(VectorFaceFieldTest, RejectsNullMeshAndEmptyDescriptorName)
{
    EXPECT_THROW(FieldType(nullptr, "velocity"), std::invalid_argument);
    const auto mesh = make_two_hex_mesh();
    EXPECT_THROW(FieldType(mesh, ""), std::invalid_argument);
}
