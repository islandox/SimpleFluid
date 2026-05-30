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
#include "utils/testing_environment.hh"
#include "fields/BoundaryFaceField.hh"

#include "geometry/MeshFactory.hh"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::BoundaryFaceField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<const SimpleFluid::Database> make_two_hex_box_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    return db;
}

SimpleFluid::SP<MeshType> make_two_hex_mesh()
{
    auto db = make_two_hex_box_database();
    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

/**
 * @brief Minimal test mesh subclass that pre-populates connectivity and
 *        boundary information for boundary-face ownership tests.
 *
 * Two faces: face 0 is a boundary face owned by cell 0 (owned);
 * face 1 is an interior face with no boundary ID.
 */
class MinimalBoundaryFaceMesh : public MeshType
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
        // Face 0: boundary face owned by cell 0
        d_faces[0].owner = 0;
        d_faces[0].neighbor = SimpleFluid::invalid_id<MeshType::local_ordinal_type>();
        d_faces[0].boundary_id = 1;
        // Face 1: non-boundary face (neighbor exists, no boundary id)
        d_faces[1].owner = 0;
        d_faces[1].neighbor = 1;
        d_faces[1].boundary_id = MeshType::invalid_boundary_id;

        d_boundary_id_to_name = {{1, "wall"}};
        d_boundary_name_to_id = {{"wall", 1}};

        create_maps();
    }

    void assemble() override
    {
        // No-op since this mesh is already in an assembled state.
    }

    void export_vtu(const std::string& /*filename*/) const override
    {
        throw std::runtime_error(
            "export_vtu not implemented for MinimalBoundaryFaceMesh");
    }
};

} // namespace

/**
 * @brief Validates storage, retrieval, and boundary-face ownership filtering
 *        for a two-hex mesh.
 */
TEST(BoundaryFaceFieldTest, StoresValuesOnOwnedBoundaryFaces)
{
    auto mesh = make_two_hex_mesh();

    FieldType bc_field(mesh, "bc_flux");

    EXPECT_EQ(bc_field.name(), "bc_flux");
    EXPECT_EQ(bc_field.mesh_ptr(), mesh);

    // Two-hex box: 11 faces total, 10 boundary faces (1 interior at x=1)
    EXPECT_EQ(bc_field.num_owned_boundary_faces(), 10u);
    EXPECT_EQ(bc_field.owned_boundary_face_ids().size(), 10u);
    EXPECT_EQ(bc_field.map()->getLocalNumElements(), 10u);

    // All boundary faces should be owned and have initial value 0.0
    for (const auto fid : bc_field.owned_boundary_face_ids())
    {
        EXPECT_TRUE(bc_field.is_owned_boundary_face(fid));
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 0.0);
    }

    // Interior face should NOT be an owned boundary face
    // In the two-hex box the interior face is the shared face at x=1
    bool found_interior = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
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

    // Per-face value operations
    const auto first_fid = bc_field.owned_boundary_face_ids().front();
    bc_field.set_value(first_fid, 3.14);
    EXPECT_DOUBLE_EQ(bc_field.value(first_fid), 3.14);

    // Bulk put
    bc_field.put_scalar(2.0);
    for (const auto fid : bc_field.owned_boundary_face_ids())
    {
        EXPECT_DOUBLE_EQ(bc_field.value(fid), 2.0);
    }

    // Out-of-range on non-boundary face in debug mode
#ifndef NDEBUG
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_boundary_face(fid))
        {
            EXPECT_THROW(bc_field.value(fid), std::out_of_range);
            break;
        }
    }
#endif
}

/**
 * @brief Confirms the initial-value constructor fills all owned boundary-face
 *        entries.
 */
TEST(BoundaryFaceFieldTest, InitialValueConstructorFillsVector)
{
    auto mesh = make_two_hex_mesh();

    FieldType initial(mesh, -1.5, "initial");

    EXPECT_EQ(initial.name(), "initial");
    EXPECT_EQ(initial.num_owned_boundary_faces(), 10u);
    for (const auto fid : initial.owned_boundary_face_ids())
    {
        EXPECT_DOUBLE_EQ(initial.value(fid), -1.5);
    }
}

/**
 * @brief Ensures only boundary faces whose owner cell is locally owned are
 *        stored, using a minimal hand-populated mesh.
 */
TEST(BoundaryFaceFieldTest, StoresOnlyBoundaryFacesWhoseOwnerCellIsOwned)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "owned_bc");

    // Only face 0 is a boundary face owned by an owned cell
    ASSERT_EQ(bc_field.num_owned_boundary_faces(), 1u);
    ASSERT_EQ(bc_field.owned_boundary_face_ids().size(), 1u);
    EXPECT_EQ(bc_field.owned_boundary_face_ids()[0], 0);
    EXPECT_EQ(bc_field.map()->getLocalNumElements(), 1u);

    EXPECT_TRUE(bc_field.is_owned_boundary_face(0));
    EXPECT_FALSE(bc_field.is_owned_boundary_face(1));

    bc_field.set_value(0, 42.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), 42.0);

#ifndef NDEBUG
    EXPECT_THROW(bc_field.value(1), std::out_of_range);
#endif
}

/**
 * @brief Verifies that put_scalar overwrites all entries.
 */
TEST(BoundaryFaceFieldTest, PutScalarOverwritesAllValues)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "bc", true);

    EXPECT_DOUBLE_EQ(bc_field.value(0), 0.0);

    bc_field.set_value(0, 7.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), 7.0);

    bc_field.put_scalar(-3.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), -3.0);
}

/**
 * @brief Verifies that an out-of-range face LID throws in debug builds.
 */
#ifndef NDEBUG
TEST(BoundaryFaceFieldTest, ThrowsOnOutOfRangeFaceLid)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "bc");

    EXPECT_THROW(bc_field.is_owned_boundary_face(2), std::out_of_range);
    EXPECT_THROW(bc_field.set_value(2, 0.0), std::out_of_range);
    EXPECT_THROW(bc_field.value(2), std::out_of_range);
}
#endif

/**
 * @brief Verifies that a null mesh throws.
 */
TEST(BoundaryFaceFieldTest, RequiresNonNullMesh)
{
    EXPECT_THROW(FieldType field(nullptr), std::invalid_argument);
}

/**
 * @brief Verifies that an unassembled mesh throws.
 */
#include "geometry/STKMesh.hh"

TEST(BoundaryFaceFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}

/**
 * @brief Verifies that set_name updates the value returned by name().
 */
TEST(BoundaryFaceFieldTest, SetNameUpdatesName)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "original");

    EXPECT_EQ(bc_field.name(), "original");

    bc_field.set_name("updated");
    EXPECT_EQ(bc_field.name(), "updated");
}

/**
 * @brief Verifies that a negative face LID throws on signed ordinal types.
 */
TEST(BoundaryFaceFieldTest, ThrowsOnNegativeFaceLid)
{
    if constexpr (std::is_signed_v<MeshType::local_ordinal_type>)
    {
        auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
        mesh->populate();

        FieldType bc_field(mesh, "bc");

        EXPECT_THROW(bc_field.is_owned_boundary_face(-1), std::out_of_range);
        EXPECT_THROW(bc_field.set_value(-1, 0.0), std::out_of_range);
        EXPECT_THROW(bc_field.value(-1), std::out_of_range);
    }
}

/**
 * @brief Verifies mesh() returns the same mesh passed at construction.
 */
TEST(BoundaryFaceFieldTest, MeshAccessorReturnsInputMesh)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "bc");

    EXPECT_EQ(&bc_field.mesh(), mesh.get());
    EXPECT_EQ(bc_field.mesh_ptr(), mesh);
    EXPECT_EQ(bc_field.mesh().num_faces(), 2u);
}

/**
 * @brief Verifies the Tpetra map has the expected properties.
 */
TEST(BoundaryFaceFieldTest, MapHasExpectedProperties)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "bc");

    const auto map = bc_field.map();

    EXPECT_EQ(map->getLocalNumElements(), 1u);
    EXPECT_EQ(map->getIndexBase(), 0);
    EXPECT_NE(map->getComm(), Teuchos::null);
}

/**
 * @brief Verifies that data() gives access to the same underlying Tpetra
 *        vector used by value()/set_value().
 */
TEST(BoundaryFaceFieldTest, DataAccessorReflectsSetValue)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "bc");

    // Write via set_value, read via data()
    bc_field.set_value(0, 99.0);
    EXPECT_DOUBLE_EQ(bc_field.data().getData()[0], 99.0);

    // Write via data(), read via value()
    bc_field.data().replaceLocalValue(0, 77.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), 77.0);
}

/**
 * @brief Verifies that constructing with zero_out = false and then reading
 *        is well-defined (vector entries may be uninitialized, but the call
 *        itself must succeed).
 */
TEST(BoundaryFaceFieldTest, ConstructorZeroOutFalseDoesNotThrow)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    // Construct with zero_out = false — must not throw
    FieldType bc_field(mesh, "no_zero", false);

    EXPECT_EQ(bc_field.name(), "no_zero");
    EXPECT_EQ(bc_field.num_owned_boundary_faces(), 1u);

    // value() must succeed (returns whatever Tpetra initialized)
    EXPECT_NO_THROW(bc_field.value(0));
}
