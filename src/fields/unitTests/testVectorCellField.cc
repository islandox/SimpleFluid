/**
 * @file testVectorCellField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for MultiVector-backed vector cell fields.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
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
using MeshType = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the assembled two-cell mesh shared by vector-field tests. */
SimpleFluid::SP<MeshType> make_two_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
}

} // namespace

/** @brief Verifies three-component storage and component-wise updates. */
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

/** @brief Confirms uniform construction fills owned and overlap entries. */
TEST(VectorCellFieldTest, InitialValueConstructorFillsOwnedAndOverlapData)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, SimpleFluid::vec3{4.0, 5.0, 6.0}, "velocity");

    EXPECT_EQ(velocity.value(0), (SimpleFluid::vec3{4.0, 5.0, 6.0}));
    EXPECT_EQ(velocity.local_value(1), (SimpleFluid::vec3{4.0, 5.0, 6.0}));
}

/** @brief Verifies component-wise bulk host access without per-value Tpetra calls. */
TEST(VectorCellFieldTest, ExposesComponentWiseHostViews)
{
    auto mesh = make_two_hex_mesh();
    FieldType velocity(mesh, "velocity");

    {
        auto owned = velocity.owned_write_view();
        ASSERT_EQ(owned.extent(0), 2u);
        ASSERT_EQ(owned.extent(1), FieldType::num_components);
        for (size_t component = 0;
             component < FieldType::num_components; ++component)
        {
            owned(1, component) =
                10.0 + static_cast<double>(component);
        }
    }

    velocity.sync_ghosts();
    const auto local = velocity.local_read_view();
    EXPECT_DOUBLE_EQ(local(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(local(1, 1), 11.0);
    EXPECT_DOUBLE_EQ(local(1, 2), 12.0);
}

/** @brief Verifies periodic synchronization copies every vector component. */
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

/** @brief Ensures null mesh and empty descriptor names fail independently. */
TEST(VectorCellFieldTest, RejectsNullMeshAndEmptyDescriptorName)
{
    EXPECT_THROW(FieldType(nullptr, "velocity"), std::invalid_argument);
    const auto mesh = make_two_hex_mesh();
    EXPECT_THROW(FieldType(mesh, ""), std::invalid_argument);
}
