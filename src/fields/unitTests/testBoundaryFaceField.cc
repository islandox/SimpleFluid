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
 * @brief Exercises all constructors, accessors, and value operations on a
 *        real two-hex STK mesh.  Covers zero-out, initial-value, zero_out=false,
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
 *        operations, and out-of-range/negative-LID throws on a hand-populated
 *        minimal mesh with one boundary and one interior face.
 */
TEST(BoundaryFaceFieldTest, OwnershipAndMinimalMesh)
{
    auto mesh = std::make_shared<MinimalBoundaryFaceMesh>();
    mesh->populate();

    FieldType bc_field(mesh, "owned_bc");

    // Ownership: only face 0 is a boundary face owned by an owned cell
    ASSERT_EQ(bc_field.num_owned_boundary_faces(), 1u);
    ASSERT_EQ(bc_field.owned_boundary_face_ids().size(), 1u);
    EXPECT_EQ(bc_field.owned_boundary_face_ids()[0], 0);
    EXPECT_TRUE(bc_field.is_owned_boundary_face(0));
    EXPECT_FALSE(bc_field.is_owned_boundary_face(1));

    // Map properties
    const auto map = bc_field.map();
    EXPECT_EQ(map->getLocalNumElements(), 1u);
    EXPECT_EQ(map->getIndexBase(), 0);
    EXPECT_NE(map->getComm(), Teuchos::null);

    // set_value / value
    bc_field.set_value(0, 42.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), 42.0);

    // put_scalar overwrites
    bc_field.put_scalar(-3.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), -3.0);

    // data() ↔ value()/set_value() bidirectional
    bc_field.set_value(0, 99.0);
    EXPECT_DOUBLE_EQ(bc_field.data().getData()[0], 99.0);
    bc_field.data().replaceLocalValue(0, 77.0);
    EXPECT_DOUBLE_EQ(bc_field.value(0), 77.0);

    // Non-owned boundary face throws in debug
#ifndef NDEBUG
    EXPECT_THROW(bc_field.value(1), std::out_of_range);
    EXPECT_THROW(bc_field.is_owned_boundary_face(2), std::out_of_range);
    EXPECT_THROW(bc_field.set_value(2, 0.0), std::out_of_range);
    EXPECT_THROW(bc_field.value(2), std::out_of_range);
#endif

    // Negative face LID (signed ordinals only)
    if constexpr (std::is_signed_v<MeshType::local_ordinal_type>)
    {
        EXPECT_THROW(bc_field.is_owned_boundary_face(-1), std::out_of_range);
        EXPECT_THROW(bc_field.set_value(-1, 0.0), std::out_of_range);
        EXPECT_THROW(bc_field.value(-1), std::out_of_range);
    }
}

#include "geometry/STKMesh.hh"

/**
 * @brief Verifies construction errors: null mesh and unassembled mesh.
 */
TEST(BoundaryFaceFieldTest, ConstructionErrors)
{
    EXPECT_THROW(FieldType(nullptr), std::invalid_argument);
    EXPECT_THROW(FieldType(std::make_shared<SimpleFluid::STKMesh<>>()),
                 std::runtime_error);
}
