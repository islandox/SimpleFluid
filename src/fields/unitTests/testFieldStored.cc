/**
 * @file testFieldStored.cc
 * @brief Tests for typed field descriptors and mesh-aware storage.
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

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>> make_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(mesh);
}

} // namespace

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
}

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
        mesh->boundary_patches().begin()->second.face_lids.front();
    boundary.set_value(boundary_face, 7.0);

    EXPECT_DOUBLE_EQ(faces.value(0), 1.0);
    EXPECT_DOUBLE_EQ(boundary.value(boundary_face), 7.0);
    EXPECT_TRUE(boundary.is_owned(boundary_face));
}

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

TEST(FieldStoredTest, ProblemRejectsDuplicateNamesAndWrongTypes)
{
    auto mesh = make_handle();
    SimpleFluid::Problem<Pack> problem(mesh);
    problem.add_field(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"));

    EXPECT_THROW(
        problem.add_field(
            SimpleFluid::VectorCellFieldDescriptor<Pack>("temperature")),
        std::invalid_argument);
    EXPECT_THROW(
        problem.object<SimpleFluid::VectorCellFieldStored<Pack>>(
            "temperature"),
        std::invalid_argument);
}
