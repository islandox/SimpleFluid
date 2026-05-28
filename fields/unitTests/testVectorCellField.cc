/**
 * @file testVectorCellField.cc
 * @brief Unit tests for MultiVector-backed vector cell fields.
 */

#include <gtest/gtest.h>

#include "fields/VectorCellField.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/STKMesh.hh"
#include "utils/testing_environment.hh"

#include <memory>
#include <stdexcept>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_two_hex_mesh()
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

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

} // namespace

TEST(VectorCellFieldTest, StoresThreeComponentsInMultiVector)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, "velocity");

    EXPECT_EQ(velocity.name(), "velocity");
    EXPECT_EQ(velocity.mesh_ptr(), mesh);
    EXPECT_EQ(velocity.owned_data().getNumVectors(), 3u);
    EXPECT_EQ(velocity.overlap_data().getNumVectors(), 3u);

    velocity.set_value(0, {1.0, 2.0, 3.0});
    velocity.set_component_value(1, 2, 9.0);

    EXPECT_EQ(velocity.value(0), (SimpleFluid::vec3{1.0, 2.0, 3.0}));
    EXPECT_DOUBLE_EQ(velocity.component_value(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(velocity.component_value(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(velocity.component_value(1, 2), 9.0);
}

TEST(VectorCellFieldTest, InitialValueConstructorFillsOwnedAndOverlapData)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, SimpleFluid::vec3{4.0, 5.0, 6.0}, "velocity");

    EXPECT_EQ(velocity.value(0), (SimpleFluid::vec3{4.0, 5.0, 6.0}));
    EXPECT_EQ(velocity.local_value(1), (SimpleFluid::vec3{4.0, 5.0, 6.0}));
}

TEST(VectorCellFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<Pack>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
