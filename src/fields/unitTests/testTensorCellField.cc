/**
 * @file testTensorCellField.cc
 * @brief Unit tests for MultiVector-backed tensor cell fields.
 */

#include <gtest/gtest.h>

#include "fields/TensorCellField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <memory>
#include <stdexcept>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::TensorCellField<Pack>;
using TensorType = FieldType::tensor_type;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

TensorType sample_tensor(double offset = 0.0)
{
    return {
        SimpleFluid::vec3{offset + 1.0, offset + 2.0, offset + 3.0},
        SimpleFluid::vec3{offset + 4.0, offset + 5.0, offset + 6.0},
        SimpleFluid::vec3{offset + 7.0, offset + 8.0, offset + 9.0}
    };
}

} // namespace

TEST(TensorCellFieldTest, StoresNineComponentsInRowMajorOrder)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, "gradient");

    EXPECT_EQ(gradient.name(), "gradient");
    EXPECT_EQ(gradient.mesh_ptr(), mesh);
    EXPECT_EQ(gradient.owned_data().getNumVectors(), 9u);
    EXPECT_EQ(gradient.overlap_data().getNumVectors(), 9u);

    gradient.set_value(0, sample_tensor());
    gradient.set_component_value(1, 2, 1, 42.0);

    EXPECT_EQ(gradient.value(0), sample_tensor());
    EXPECT_DOUBLE_EQ(gradient.component_value(0, 5), 6.0);
    EXPECT_DOUBLE_EQ(gradient.component_value(1, 2, 1), 42.0);
}

TEST(TensorCellFieldTest, InitialValueConstructorFillsOwnedAndOverlapData)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, sample_tensor(10.0), "gradient");

    EXPECT_EQ(gradient.value(0), sample_tensor(10.0));
    EXPECT_EQ(gradient.local_value(1), sample_tensor(10.0));
}

TEST(TensorCellFieldTest, SyncPeriodicBoundariesSynchronizesOverlapStorage)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, "gradient");

    gradient.set_owned_value(0, sample_tensor());
    gradient.set_owned_value(1, sample_tensor(10.0));

    mesh->sync_periodic_boundaries(gradient);

    EXPECT_EQ(gradient.local_value(0), sample_tensor());
    EXPECT_EQ(gradient.local_value(1), sample_tensor(10.0));
}

TEST(TensorCellFieldTest, RejectsOutOfRangeComponents)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, "gradient");

    EXPECT_THROW(gradient.component_value(0, 9), std::out_of_range);
    EXPECT_THROW(gradient.component_value(0, 3, 0), std::out_of_range);
    EXPECT_THROW(gradient.set_component_value(0, 0, 3, 1.0),
                 std::out_of_range);
}

TEST(TensorCellFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<Pack>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
