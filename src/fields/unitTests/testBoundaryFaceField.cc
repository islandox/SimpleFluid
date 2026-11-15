/**
 * @file testBoundaryFaceField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for BoundaryFaceField class
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>
#include "utils/testing_environment.hh"
#include "fields/BoundaryFaceField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "NativeFieldOwnershipFixture.hh"

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::BoundaryFaceField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the assembled two-cell mesh shared by field tests. */
SimpleFluid::SP<Mesh> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

} // namespace

/**
 * @brief Exercises all constructors, accessors, and value operations on a
 *        real native two-hex mesh.  Covers zero-out, initial-value, zero_out=false,
 *        name/mesh/map queries, per-face get/set, put_scalar, and
 *        boundary-vs-interior filtering.
 */
TEST(BoundaryFaceFieldTest, BasicOperationsOnRealMesh)
{
    auto mesh = make_two_hex_mesh();

    // --- zero-out constructor ---
    FieldType bc_field(mesh, "bc_flux");
    EXPECT_EQ(bc_field.name(), "bc_flux");
    EXPECT_EQ(bc_field.mesh_ptr(), mesh);
    EXPECT_EQ(&bc_field.mesh(), mesh.get());
    EXPECT_EQ(bc_field.mesh().num_faces(), mesh->num_faces());

    // Two-hex box: 11 faces total, 10 boundary faces (1 interior at x=1)
    EXPECT_EQ(bc_field.num_owned_boundary_faces(), 10u);
    EXPECT_EQ(bc_field.owned_boundary_face_ids().size(), 10u);
    EXPECT_EQ(bc_field.map()->getLocalNumElements(), 10u);

    for (const auto fid : bc_field.owned_boundary_face_ids())
    {
        EXPECT_TRUE(bc_field.is_owned_boundary_face(fid));
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 0.0);
    }

    // Interior face is NOT owned
    bool found_interior = false;
    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_boundary_face(fid))
        {
            EXPECT_FALSE(bc_field.is_owned_boundary_face(fid));
            found_interior = true;
            break;
        }
    }
    EXPECT_TRUE(found_interior);

    // Per-face set / get
    const auto first_fid = bc_field.owned_boundary_face_ids().front();
    bc_field.set_value(first_fid, 3.14);
    EXPECT_DOUBLE_EQ(bc_field.value(first_fid), 3.14);

    // Bulk put_scalar
    bc_field.put_scalar(2.0);
    for (const auto fid : bc_field.owned_boundary_face_ids())
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 2.0);

    // set_name round-trip
    bc_field.set_name("renamed");
    EXPECT_EQ(bc_field.name(), "renamed");

    // Non-boundary face throws in debug
#ifndef NDEBUG
    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_boundary_face(fid))
        {
            EXPECT_THROW(bc_field.value(fid), std::out_of_range);
            break;
        }
    }
#endif

    // --- initial-value constructor ---
    FieldType initial(mesh, -1.5, "initial");
    EXPECT_EQ(initial.name(), "initial");
    EXPECT_EQ(initial.num_owned_boundary_faces(), 10u);
    for (const auto fid : initial.owned_boundary_face_ids())
        EXPECT_DOUBLE_EQ(initial.value(fid), -1.5);

    // --- zero_out = false constructor ---
    FieldType no_zero(mesh, "no_zero", false);
    EXPECT_EQ(no_zero.name(), "no_zero");
    EXPECT_EQ(no_zero.num_owned_boundary_faces(), 10u);
    EXPECT_NO_THROW(no_zero.value(no_zero.owned_boundary_face_ids().front()));
}

/**
 * @brief Tests ownership filtering, map metadata, raw data access, value
 *        operations, and bounds checks on actual native boundary-map rows.
 */
TEST(BoundaryFaceFieldTest, OwnershipAndNativeMapRows)
{
    auto mesh = SimpleFluid::test::native_field_ownership_mesh<Pack>();
    FieldType bc_field(mesh, "owned_bc");
    const auto map = bc_field.map();
    EXPECT_EQ(map->getIndexBase(), 0);
    EXPECT_NE(map->getComm(), Teuchos::null);
    const auto& ids = bc_field.owned_boundary_face_ids();
    EXPECT_EQ(ids.size(), map->getLocalNumElements());
    size_t expected_owned = 0;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto fid = static_cast<Pack::local_ordinal_type>(face);
        const bool owned_boundary = mesh->is_boundary_face(fid) && mesh->is_owned_cell(mesh->owner_cell(fid));
        EXPECT_EQ(bc_field.is_owned_boundary_face(fid), owned_boundary);
        if (owned_boundary)
            ++expected_owned;
        else
            EXPECT_THROW(bc_field.value(fid), std::out_of_range);
    }
    EXPECT_EQ(bc_field.num_owned_boundary_faces(), expected_owned);
    EXPECT_EQ(ids.size(), expected_owned);
    for (size_t row = 0; row < ids.size(); ++row)
    {
        const auto fid = ids[row];
        EXPECT_EQ(map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row)), mesh->face_global_id(fid));
        bc_field.set_value(fid, 42.0);
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 42.0);
    }
    bc_field.put_scalar(-3.0);
    for (const auto fid : ids) EXPECT_DOUBLE_EQ(bc_field.value(fid), -3.0);
    if (!ids.empty())
    {
        const auto fid = ids.front();
        bc_field.set_value(fid, 99.0);
        EXPECT_DOUBLE_EQ(bc_field.data().getData()[0], 99.0);
        bc_field.data().replaceLocalValue(0, 77.0);
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 77.0);
    }
    const auto invalid_face = static_cast<Pack::local_ordinal_type>(mesh->num_faces());
    EXPECT_THROW(bc_field.is_owned_boundary_face(invalid_face), std::out_of_range);
    EXPECT_THROW(bc_field.set_value(invalid_face, 0.0), std::out_of_range);
    EXPECT_THROW(bc_field.value(invalid_face), std::out_of_range);
    if constexpr (std::is_signed_v<Mesh::local_ordinal_type>)
    {
        EXPECT_THROW(bc_field.is_owned_boundary_face(-1), std::out_of_range);
        EXPECT_THROW(bc_field.set_value(-1, 0.0), std::out_of_range);
        EXPECT_THROW(bc_field.value(-1), std::out_of_range);
    }
    int local_owned = static_cast<int>(expected_owned), global_owned = 0;
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_SUM, 1, &local_owned, &global_owned);
    EXPECT_EQ(global_owned, 10);
}

/**
 * @brief Verifies null mesh and empty descriptor names fail independently.
 */
TEST(BoundaryFaceFieldTest, ConstructionErrors)
{
    EXPECT_THROW(FieldType(nullptr, "boundary_flux"), std::invalid_argument);
    const auto mesh = make_two_hex_mesh();
    EXPECT_THROW(FieldType(mesh, ""), std::invalid_argument);
}
