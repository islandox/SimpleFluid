/**
 * @file testFieldStored.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for typed field descriptors and mesh-aware storage.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "fields/FieldStored.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "problems/Problem.hh"
#include "utils/testing_environment.hh"

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using PartitionedCartesian =
    SimpleFluid::Meshes::PartitionedMesh<Cartesian, Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build a runtime mesh handle over a small Cartesian mesh. */
SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>> make_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(mesh);
}

/** @brief Build the equivalent statically typed partitioned Cartesian mesh. */
SimpleFluid::SP<const PartitionedCartesian> make_partitioned_cartesian()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});

    PartitionedCartesian::indexer_type indexer(mesh->indexer());
    return std::make_shared<PartitionedCartesian>(
        std::move(mesh), std::move(indexer));
}

} // namespace

/** @brief Verifies scalar and vector descriptors use typed cell storage. */
TEST(FieldStoredTest, StoresScalarAndVectorCellValues)
{
    auto mesh = make_handle();
    SimpleFluid::ScalarCellFieldStored<Pack> scalar(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"),
        mesh);
    SimpleFluid::VectorCellFieldStored<Pack> vector(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"),
        mesh);

    scalar.set_value(0, 2.5);
    vector.set_value(1, {1.0, 2.0, 3.0});
    scalar.sync_ghosts();
    vector.sync_ghosts();

    EXPECT_DOUBLE_EQ(scalar.value(0), 2.5);
    EXPECT_EQ(vector.value(1),
              (SimpleFluid::vec3<double>{1.0, 2.0, 3.0}));
    EXPECT_DOUBLE_EQ(vector.component_value(1, 2), 3.0);
    EXPECT_THROW(vector.component_value(1, 3), std::out_of_range);
    EXPECT_EQ(scalar.num_owned_entries(), mesh->num_owned_cells());
    EXPECT_EQ(scalar.num_owned_cells(), mesh->num_owned_cells());
    EXPECT_EQ(scalar.num_local_cells(), mesh->num_local_cells());
    EXPECT_TRUE(scalar.is_owned_cell(0));
    EXPECT_TRUE(scalar.is_local_cell(1));

    scalar.put_scalar(3.0);
    scalar.sum_into_value(0, 0.5);
    vector.set_owned_component_value(1, 2, 8.0);
    EXPECT_DOUBLE_EQ(vector.local_component_value(1, 2), 3.0);
    vector.sync_ghosts();
    EXPECT_DOUBLE_EQ(vector.local_component_value(1, 2), 8.0);
    EXPECT_DOUBLE_EQ(scalar.value(0), 3.5);

    {
        auto values = scalar.owned_write_view();
        values(0, 0) = 9.0;
    }
    scalar.sync_ghosts();
    {
        const auto& const_scalar = scalar;
        const auto owned = const_scalar.owned_read_view();
        const auto local = const_scalar.local_read_view();
        EXPECT_DOUBLE_EQ(owned(0, 0), 9.0);
        EXPECT_DOUBLE_EQ(local(0, 0), 9.0);
    }
    {
        auto local = scalar.local_write_view();
        local(1, 0) = -2.0;
    }
    EXPECT_DOUBLE_EQ(scalar.local_value(1), -2.0);
    EXPECT_DOUBLE_EQ(scalar.value(1), 3.0);
    scalar.sync_ghosts();
    EXPECT_DOUBLE_EQ(scalar.local_value(1), 3.0);
}

/** @brief Exercises face and boundary-face location-specific map selection. */
TEST(FieldStoredTest, SupportsFaceAndBoundaryFaceLocations)
{
    auto mesh = make_handle();
    SimpleFluid::ScalarFaceFieldStored<Pack> faces(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("flux"),
        mesh,
        1.0);
    SimpleFluid::ScalarBoundaryFaceFieldStored<Pack> boundary(
        SimpleFluid::ScalarBoundaryFaceFieldDescriptor<Pack>(
            "boundary_temperature"),
        mesh);

    const auto boundary_face =
        mesh->boundary_batches().begin()->second.face_lids.front();
    boundary.set_value(boundary_face, 7.0);

    EXPECT_DOUBLE_EQ(faces.value(0), 1.0);
    EXPECT_DOUBLE_EQ(boundary.value(boundary_face), 7.0);
    EXPECT_TRUE(boundary.is_owned(boundary_face));
    EXPECT_TRUE(boundary.is_owned_boundary_face(boundary_face));
    EXPECT_EQ(
        boundary.num_owned_boundary_faces(),
        mesh->boundary_face_map()->getLocalNumElements());
    EXPECT_EQ(faces.num_owned_faces(), mesh->num_owned_faces());
    EXPECT_EQ(faces.num_local_faces(), mesh->num_faces());
    ASSERT_EQ(faces.owned_face_ids().size(), mesh->num_owned_faces());
    for (size_t row = 0; row < faces.owned_face_ids().size(); ++row)
    {
        const auto face_lid = faces.owned_face_ids()[row];
        EXPECT_EQ(face_lid, static_cast<Pack::local_ordinal_type>(row));
        EXPECT_EQ(faces.owned_row(face_lid),
                  static_cast<Pack::local_ordinal_type>(row));
        EXPECT_TRUE(faces.is_owned_face(face_lid));
        EXPECT_TRUE(faces.is_local_face(face_lid));
    }
}

/** @brief Confirms runtime STK-backed fields preserve mesh global IDs. */
TEST(FieldStoredTest, UsesSTKStorageGlobalIds)
{
    auto legacy = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    auto mesh =
        std::make_shared<SimpleFluid::MeshHandle<Pack>>(legacy);
    SimpleFluid::ScalarCellFieldStored<Pack> cells(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"),
        mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack> faces(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("flux"),
        mesh);

    cells.set_value(0, 3.0);
    faces.set_value(0, 4.0);

    EXPECT_DOUBLE_EQ(cells.value(0), 3.0);
    EXPECT_DOUBLE_EQ(faces.value(0), 4.0);
    EXPECT_EQ(cells.map()->getGlobalElement(0),
              mesh->cell_global_id(0));
    EXPECT_EQ(faces.map()->getGlobalElement(0),
              mesh->face_global_id(0));
}

/** @brief Confirms storage also accepts a statically typed partitioned mesh. */
TEST(FieldStoredTest, AcceptsPartitionedCRTPMesh)
{
    auto mesh = make_partitioned_cartesian();
    SimpleFluid::ScalarCellFieldStored<Pack, PartitionedCartesian> cells(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"),
        mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, PartitionedCartesian> faces(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("flux"),
        mesh);

    const auto structured_cell = mesh->mesh().cell_id(1);
    cells.set_value(structured_cell, 4.5);
    faces.set_value(0, 2.0);

    EXPECT_DOUBLE_EQ(cells.value(structured_cell), 4.5);
    EXPECT_DOUBLE_EQ(faces.value(0), 2.0);
    EXPECT_EQ(mesh->num_global_cells(), mesh->num_local_cells());
    EXPECT_EQ(mesh->num_global_faces(), mesh->num_local_faces());
    EXPECT_EQ(mesh->num_global_nodes(), mesh->num_local_nodes());
    EXPECT_EQ(mesh->cell_global_id(1), 1);
    EXPECT_TRUE(mesh->is_owned_cell(1));
    EXPECT_TRUE(mesh->is_owned_face(0));
    EXPECT_TRUE(mesh->is_owned_node(0));
}

/** @brief Locks down Problem registry name uniqueness and checked type access. */
TEST(FieldStoredTest, ProblemRejectsDuplicateNamesAndWrongTypes)
{
    auto mesh = make_handle();
    SimpleFluid::Problem<Pack> problem(mesh);
    auto& field = problem.add_field(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"));

    ASSERT_TRUE(problem.fields().contains("temperature"));
    const auto& registered =
        std::get<SimpleFluid::SP<SimpleFluid::ScalarCellFieldStored<Pack>>>(
            problem.fields().at("temperature"));
    EXPECT_EQ(registered.get(), &field);
    EXPECT_EQ(registered->mesh_ptr(), problem.mesh_ptr());

    EXPECT_THROW(
        problem.add_field(
            SimpleFluid::VectorCellFieldDescriptor<Pack>("temperature")),
        std::invalid_argument);
    EXPECT_THROW(
        problem.object<SimpleFluid::VectorCellFieldStored<Pack>>(
            "temperature"),
        std::invalid_argument);
}
