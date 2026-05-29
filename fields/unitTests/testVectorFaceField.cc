/**
 * @file testVectorFaceField.cc
 * @brief Unit tests for VectorFaceField.
 */

#include <gtest/gtest.h>

#include "fields/VectorFaceField.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/STKMesh.hh"
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
    }

    void export_vtu(const std::string& filename) const override
    {
        throw std::runtime_error(filename + " export not implemented");
    }
};

} // namespace

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

TEST(VectorFaceFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
