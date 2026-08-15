/**
 * @file testTensorCellField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for MultiVector-backed tensor cell fields.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
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

/** @brief Build the assembled two-cell mesh shared by tensor-field tests. */
SimpleFluid::SP<MeshType> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

/**
 * @brief Create a row-major tensor with distinct, optionally shifted values.
 *
 * @param offset Value added to all nine components.
 * @return Tensor containing consecutive component values.
 */
TensorType sample_tensor(double offset = 0.0)
{
    return {
        SimpleFluid::vec3{offset + 1.0, offset + 2.0, offset + 3.0},
        SimpleFluid::vec3{offset + 4.0, offset + 5.0, offset + 6.0},
        SimpleFluid::vec3{offset + 7.0, offset + 8.0, offset + 9.0}
    };
}

} // namespace

/** @brief Verifies row-major component storage and indexed updates. */
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

/** @brief Confirms uniform construction fills owned and overlap entries. */
TEST(TensorCellFieldTest, InitialValueConstructorFillsOwnedAndOverlapData)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, sample_tensor(10.0), "gradient");

    EXPECT_EQ(gradient.value(0), sample_tensor(10.0));
    EXPECT_EQ(gradient.local_value(1), sample_tensor(10.0));
}

/** @brief Verifies row-major tensor components through bulk host views. */
TEST(TensorCellFieldTest, ExposesRowMajorHostViews)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, "gradient");

    {
        auto owned = gradient.owned_write_view();
        ASSERT_EQ(owned.extent(1), FieldType::num_components);
        for (size_t component = 0;
             component < FieldType::num_components; ++component)
        {
            owned(0, component) =
                static_cast<double>(component + 1);
        }
    }

    gradient.sync_ghosts();
    const auto local = gradient.local_read_view();
    EXPECT_DOUBLE_EQ(local(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(local(0, 5), 6.0);
    EXPECT_DOUBLE_EQ(local(0, 8), 9.0);
}

/** @brief Verifies periodic synchronization copies all tensor components. */
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

/** @brief Ensures flat and row-column indices are bounds checked. */
TEST(TensorCellFieldTest, RejectsOutOfRangeComponents)
{
    auto mesh = make_two_hex_mesh();
    FieldType gradient(mesh, "gradient");

    EXPECT_THROW(gradient.component_value(0, 9), std::out_of_range);
    EXPECT_THROW(gradient.component_value(0, 3, 0), std::out_of_range);
    EXPECT_THROW(gradient.set_component_value(0, 0, 3, 1.0),
                 std::out_of_range);
}

/** @brief Ensures construction rejects a mesh without assembled maps. */
TEST(TensorCellFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh =
        std::make_shared<SimpleFluid::STKMesh<Pack>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
