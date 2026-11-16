/**
 * @file testFaceField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for FaceField class
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>
#include "utils/testing_environment.hh"
#include "fields/FaceField.hh"

#include "geometry/unitTests/test_mesh_helpers.hh"
#include "NativeFieldOwnershipFixture.hh"

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::FaceField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the assembled two-cell mesh shared by face-field tests. */
SimpleFluid::SP<Mesh> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

} // namespace

/**
 * @brief Validates storage, retrieval, and face ownership for a two-hex mesh face field.
 */
TEST(FaceFieldTest, StoresValuesOnOwnedFaceMap)
{
    auto mesh = make_two_hex_mesh();

    FieldType flux(mesh, "flux");

    EXPECT_EQ(flux.name(), "flux");
    EXPECT_EQ(flux.mesh_ptr(), mesh);
    EXPECT_EQ(flux.num_owned_faces(), 11u);
    EXPECT_EQ(flux.map()->getLocalNumElements(), mesh->num_faces());
    EXPECT_EQ(flux.owned_face_ids().size(), mesh->num_faces());

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        EXPECT_TRUE(flux.is_owned_face(fid));
        EXPECT_DOUBLE_EQ(flux.value(fid), 0.0);
    }

    const auto face0_gid = flux.face_global_id(0);
    const auto face10_gid = flux.face_global_id(10);

    flux.set_value(0, 2.5);
    flux.set_global_value(face10_gid, 4.0);
    flux.sum_into_value(0, 0.5);
    flux.sum_into_global_value(face10_gid, 1.0);

    EXPECT_DOUBLE_EQ(flux.value(0), 3.0);
    EXPECT_DOUBLE_EQ(flux.global_value(face0_gid), 3.0);
    EXPECT_DOUBLE_EQ(flux.value(10), 5.0);
    EXPECT_DOUBLE_EQ(flux.global_value(face10_gid), 5.0);
    EXPECT_TRUE(flux.is_owned_global_face(face10_gid));
    EXPECT_FALSE(flux.is_owned_global_face(77));
    EXPECT_THROW_WHEN_DEBUG(flux.value(11), std::out_of_range);
}

/**
 * @brief Confirms the initial-value constructor fills all owned face entries.
 */
TEST(FaceFieldTest, InitialValueConstructorFillsVector)
{
    auto mesh = make_two_hex_mesh();

    FieldType area_weight(mesh, 1.25, "area_weight");

    EXPECT_EQ(area_weight.name(), "area_weight");
    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        EXPECT_DOUBLE_EQ(area_weight.value(fid), 1.25);
    }
}

/** @brief Verifies checked face rows and bulk host-view access. */
TEST(FaceFieldTest, ExposesOwnedRowsAndBulkHostViews)
{
    auto mesh = make_two_hex_mesh();
    FieldType flux(mesh, "flux");

    {
        auto values = flux.owned_write_view();
        ASSERT_EQ(values.extent(0), flux.num_owned_faces());
        ASSERT_EQ(values.extent(1), 1u);
        for (const auto face_lid : flux.owned_face_ids())
        {
            const auto row = flux.owned_row(face_lid);
            EXPECT_EQ(row, face_lid);
            values(row, 0) = static_cast<double>(face_lid) + 0.5;
        }
    }

    const auto values = flux.owned_read_view();
    for (const auto face_lid : flux.owned_face_ids())
    {
        EXPECT_DOUBLE_EQ(
            values(flux.owned_row(face_lid), 0),
            static_cast<double>(face_lid) + 0.5);
    }
}

/**
 * @brief Ensures only faces whose owner cell is locally owned appear in the field map.
 */
TEST(FaceFieldTest, StoresOnlyFacesWhoseOwnerCellIsOwned)
{
    auto mesh = SimpleFluid::test::native_field_ownership_mesh<Pack>();
    FieldType flux(mesh, "owned_flux");
    size_t expected_owned = 0;
    size_t nonowned = 0;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto fid = static_cast<Pack::local_ordinal_type>(face);
        const bool owned = mesh->is_owned_cell(mesh->owner_cell(fid));
        EXPECT_EQ(flux.is_owned_face(fid), owned);
        EXPECT_TRUE(flux.is_local_face(fid));
        if (owned)
        {
            ++expected_owned;
            const auto row = flux.owned_row(fid);
            EXPECT_EQ(flux.map()->getGlobalElement(row), mesh->face_global_id(fid));
            flux.set_value(fid, 7.0);
            EXPECT_DOUBLE_EQ(flux.value(fid), 7.0);
            EXPECT_DOUBLE_EQ(flux.local_value(fid), 7.0);
        }
        else
        {
            ++nonowned;
            EXPECT_THROW(flux.owned_row(fid), std::out_of_range);
            EXPECT_THROW(flux.face_global_id(fid), std::out_of_range);
            EXPECT_THROW(flux.value(fid), std::out_of_range);
        }
    }
    EXPECT_EQ(flux.num_owned_faces(), expected_owned);
    EXPECT_EQ(flux.owned_face_ids().size(), expected_owned);
    EXPECT_EQ(flux.map()->getLocalNumElements(), expected_owned);
    const auto comm = mesh->owned_cell_map()->getComm();
    int local_owned = static_cast<int>(expected_owned), global_owned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_owned, &global_owned);
    EXPECT_EQ(global_owned, 11);
    if (comm->getSize() > 1 && mesh->num_owned_cells() > 0) EXPECT_GT(nonowned, 0u);
}

/** @brief Ensures null mesh and empty descriptor names fail independently. */
TEST(FaceFieldTest, RejectsNullMeshAndEmptyDescriptorName)
{
    EXPECT_THROW(FieldType(nullptr, "flux"), std::invalid_argument);
    const auto mesh = make_two_hex_mesh();
    EXPECT_THROW(FieldType(mesh, ""), std::invalid_argument);
}
