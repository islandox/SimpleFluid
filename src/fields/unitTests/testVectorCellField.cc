/**
 * @file testVectorCellField.cc
 * @brief Unit tests for MultiVector-backed vector cell fields.
 */

#include <gtest/gtest.h>

#include "fields/VectorCellField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
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
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
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

TEST(VectorCellFieldTest, SyncPeriodicBoundariesSynchronizesOverlapStorage)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");

    velocity.set_owned_value(0, {1.0, 2.0, 3.0});
    velocity.set_owned_value(1, {4.0, 5.0, 6.0});

    mesh->sync_periodic_boundaries(velocity);

    EXPECT_EQ(velocity.local_value(0), (SimpleFluid::vec3{1.0, 2.0, 3.0}));
    EXPECT_EQ(velocity.local_value(1), (SimpleFluid::vec3{4.0, 5.0, 6.0}));
}

TEST(VectorCellFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<Pack>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
