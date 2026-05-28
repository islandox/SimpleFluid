/**
 * @file testCellField.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for CellField class
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include "utils/testing_environment.hh"
#include "fields/CellField.hh"

#include "geometry/MeshFactory.hh"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

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

} // namespace

/**
 * @brief Validates storage, retrieval, and ownership queries for a cell field on a two-hex mesh.
 */
TEST(CellFieldTest, StoresValuesOnOwnedCellMap)
{
    auto mesh = make_two_hex_mesh();

    FieldType temperature(mesh, "temperature");

    EXPECT_EQ(temperature.name(), "temperature");
    EXPECT_EQ(temperature.mesh_ptr(), mesh);
    EXPECT_EQ(temperature.num_owned_cells(), 2u);
    EXPECT_EQ(temperature.map()->getLocalNumElements(), mesh->num_owned_cells());
    EXPECT_DOUBLE_EQ(temperature.value(0), 0.0);
    EXPECT_DOUBLE_EQ(temperature.value(1), 0.0);

    temperature.set_value(0, 2.5);
    temperature.set_global_value(2, 4.0);
    temperature.sum_into_value(0, 0.5);
    temperature.sum_into_global_value(2, 1.0);

    EXPECT_DOUBLE_EQ(temperature.value(0), 3.0);
    EXPECT_DOUBLE_EQ(temperature.global_value(1), 3.0);
    EXPECT_DOUBLE_EQ(temperature.value(1), 5.0);
    EXPECT_DOUBLE_EQ(temperature.global_value(2), 5.0);
    EXPECT_TRUE(temperature.is_owned_cell(0));
    EXPECT_TRUE(temperature.is_owned_global_cell(2));
    EXPECT_FALSE(temperature.is_owned_global_cell(77));
}

/**
 * @brief Confirms the initial-value constructor fills both owned and overlap data.
 */
TEST(CellFieldTest, InitialValueConstructorFillsVector)
{
    auto mesh = make_two_hex_mesh();

    FieldType pressure(mesh, 101325.0, "pressure");

    EXPECT_EQ(pressure.name(), "pressure");
    EXPECT_DOUBLE_EQ(pressure.value(0), 101325.0);
    EXPECT_DOUBLE_EQ(pressure.value(1), 101325.0);
    EXPECT_DOUBLE_EQ(pressure.local_value(0), 101325.0);
    EXPECT_DOUBLE_EQ(pressure.local_value(1), 101325.0);
    EXPECT_EQ(pressure.overlap_map()->getLocalNumElements(), mesh->num_local_cells());
}

/**
 * @brief Ensures sync_ghosts() propagates owned values into the overlap storage.
 */
TEST(CellFieldTest, SynchronizesOwnedValuesIntoOverlapStorage)
{
    auto mesh = make_two_hex_mesh();

    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 10.0);
    temperature.set_value(1, 20.0);
    temperature.sync_ghosts();

    EXPECT_TRUE(temperature.is_local_cell(0));
    EXPECT_TRUE(temperature.is_local_global_cell(1));
    EXPECT_DOUBLE_EQ(temperature.local_value(0), 10.0);
    EXPECT_DOUBLE_EQ(temperature.local_value(1), 20.0);
}

#include "geometry/STKMesh.hh"

TEST(CellFieldTest, RequiresAssembledMesh)
{
    SimpleFluid::SP<MeshType> unassembled_mesh = std::make_shared<SimpleFluid::STKMesh<Pack>>();

    EXPECT_THROW(FieldType field(unassembled_mesh), std::runtime_error);
}
