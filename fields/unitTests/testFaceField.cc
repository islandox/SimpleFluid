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
#include "utils/testing_environment.hh"
#include "fields/FaceField.hh"

#include "geometry/MeshFactory.hh"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::FaceField<Pack>;

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
        // No-op since this mesh is already in an assembled state.
    }

    void export_vtu(const std::string& filename) const override
    {
        throw std::runtime_error("export_to_vtk not implemented for MinimalFaceOwnershipMesh");
    }
};

} // namespace

TEST(FaceFieldTest, StoresValuesOnOwnedFaceMap)
{
    auto mesh = make_two_hex_mesh();

    FieldType flux(mesh, "flux");

    EXPECT_EQ(flux.name(), "flux");
    EXPECT_EQ(flux.mesh_ptr(), mesh);
    EXPECT_EQ(flux.num_owned_faces(), 11u);
    EXPECT_EQ(flux.map()->getLocalNumElements(), mesh->num_faces());
    EXPECT_EQ(flux.owned_face_ids().size(), mesh->num_faces());

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
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
#ifndef NDEBUG
    EXPECT_THROW(flux.value(11), std::out_of_range);
#endif
}

TEST(FaceFieldTest, InitialValueConstructorFillsVector)
{
    auto mesh = make_two_hex_mesh();

    FieldType area_weight(mesh, 1.25, "area_weight");

    EXPECT_EQ(area_weight.name(), "area_weight");
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        EXPECT_DOUBLE_EQ(area_weight.value(fid), 1.25);
    }
}

TEST(FaceFieldTest, StoresOnlyFacesWhoseOwnerCellIsOwned)
{
    auto mesh = std::make_shared<MinimalFaceOwnershipMesh>();
    mesh->populate();

    FieldType flux(mesh, "owned_flux");

    ASSERT_EQ(flux.num_owned_faces(), 1u);
    ASSERT_EQ(flux.owned_face_ids().size(), 1u);
    EXPECT_EQ(flux.owned_face_ids()[0], 0);
    EXPECT_TRUE(flux.is_owned_face(0));
    EXPECT_FALSE(flux.is_owned_face(1));
    EXPECT_THROW(flux.face_global_id(1), std::out_of_range);
    EXPECT_EQ(flux.map()->getLocalNumElements(), 1u);

    flux.set_value(0, 7.0);

    EXPECT_DOUBLE_EQ(flux.value(0), 7.0);
#ifndef NDEBUG
    EXPECT_THROW(flux.value(1), std::out_of_range);
#endif
}

#include "geometry/STKMesh.hh"

TEST(FaceFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh = std::make_shared<SimpleFluid::STKMesh<>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
