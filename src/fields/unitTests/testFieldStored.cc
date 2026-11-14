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
#include "solvers/FieldStateTransaction.hh"
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

TEST(FieldStoredTest, SnapshotRestoresOwnedStorageAndRejectsForeignField)
{
    auto mesh = make_handle();
    SimpleFluid::ScalarCellFieldStored<Pack> scalar(mesh, "scalar");
    SimpleFluid::VectorCellFieldStored<Pack> vector(mesh, "vector");
    SimpleFluid::ScalarFaceFieldStored<Pack> faces(mesh, "faces");
    SimpleFluid::ScalarFaceFieldStored<Pack> other_faces(mesh, "other_faces");
    // Scalar and vector descriptors share immutable face IDs, not values.
    SimpleFluid::VectorFaceFieldStored<Pack> vector_faces(mesh, "vector_faces");
    EXPECT_EQ(&faces.owned_face_ids(), &vector_faces.owned_face_ids());
    EXPECT_EQ(&faces.owned_face_ids(), &other_faces.owned_face_ids());
    scalar.set_owned_value(0, 2.5);
    vector.set_owned_value(1, {1.0, 2.0, 3.0});
    const auto face = faces.owned_face_ids().back();
    faces.set_owned_value(face, 7.5);
    SimpleFluid::FieldStateSnapshot scalar_snapshot(scalar);
    SimpleFluid::FieldStateSnapshot vector_snapshot(vector);
    SimpleFluid::FieldStateSnapshot face_snapshot(faces);
    scalar.put_scalar(-1.0);
    vector.put_value({-2.0, -2.0, -2.0});
    faces.put_scalar(-3.0);
    scalar_snapshot.restore(scalar);
    vector_snapshot.restore(vector);
    face_snapshot.restore(faces);
    EXPECT_DOUBLE_EQ(scalar.local_value(0), 2.5);
    EXPECT_EQ(vector.local_value(1), (SimpleFluid::vec3<double>{1.0, 2.0, 3.0}));
    EXPECT_DOUBLE_EQ(faces.local_value(face), 7.5);
    EXPECT_DOUBLE_EQ(other_faces.value(face), 0.0);
    EXPECT_THROW(face_snapshot.restore(other_faces), std::invalid_argument);
}

TEST(FieldStoredTest, SnapshotRestoresLogicalNonconstantStrideColumns)
{
    auto mesh = make_handle();
    SimpleFluid::VectorCellFieldStored<Pack> field(mesh, "rebound");
    Pack::multi_vector_type backing(mesh->owned_cell_map(), 5);
    backing.putScalar(19.0);
    Teuchos::Array<size_t> columns{0, 2, 4};
    field.owned_data() = *backing.subViewNonConst(columns());
    ASSERT_FALSE(field.owned_data().isConstantStride());
    for (size_t row = 0; row < field.num_owned_entries(); ++row)
        field.set_owned_value(static_cast<Pack::local_ordinal_type>(row),
                              {1.0 + row, 2.0 + row, 3.0 + row});
    SimpleFluid::FieldStateSnapshot snapshot(field);
    field.put_value({-1.0, -2.0, -3.0});
    snapshot.restore(field);
    for (size_t row = 0; row < field.num_owned_entries(); ++row)
    {
        EXPECT_EQ(field.value(static_cast<Pack::local_ordinal_type>(row)),
                  (SimpleFluid::vec3<double>{1.0 + row, 2.0 + row, 3.0 + row}));
        EXPECT_EQ(field.local_value(static_cast<Pack::local_ordinal_type>(row)),
                  (SimpleFluid::vec3<double>{1.0 + row, 2.0 + row, 3.0 + row}));
        EXPECT_DOUBLE_EQ(backing.getData(1)[row], 19.0);
        EXPECT_DOUBLE_EQ(backing.getData(3)[row], 19.0);
    }
}

namespace
{
template<class Stored>
void expect_compiled_field_constructors(
    const SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>& mesh, const typename Stored::value_type& initial,
    const std::array<double, Stored::num_components>& expected)
{
    using Descriptor = typename Stored::descriptor_type;
    Stored descriptor_zero(Descriptor{"descriptor_zero"}, mesh, true);
    Stored mesh_zero(mesh, "mesh_zero", true);
    Stored descriptor_initial(Descriptor{"descriptor_initial"}, mesh, initial);
    Stored mesh_initial(mesh, initial, "mesh_initial");
    EXPECT_EQ(descriptor_zero.mesh_ptr().get(), mesh.get());
    EXPECT_EQ(mesh_initial.mesh_ptr().get(), mesh.get());
    EXPECT_NE(descriptor_zero.id(), mesh_zero.id());
    EXPECT_NE(descriptor_initial.id(), mesh_initial.id());
    for (const auto* field : {&descriptor_zero, &mesh_zero})
    {
        const auto owned = field->owned_read_view();
        const auto overlap = field->local_read_view();
        for (size_t component = 0; component < Stored::num_components; ++component)
        {
            for (size_t row = 0; row < field->num_owned_entries(); ++row)
                EXPECT_DOUBLE_EQ(owned(row, component), 0.0);
            for (size_t row = 0; row < field->num_local_entries(); ++row)
                EXPECT_DOUBLE_EQ(overlap(row, component), 0.0);
        }
    }
    descriptor_zero.put_value(initial);
    mesh_zero.put_scalar(initial);
    for (const auto* field : {&descriptor_zero, &mesh_zero, &descriptor_initial, &mesh_initial})
    {
        const auto owned = field->owned_read_view();
        const auto overlap = field->local_read_view();
        for (size_t component = 0; component < Stored::num_components; ++component)
        {
            for (size_t row = 0; row < field->num_owned_entries(); ++row)
                EXPECT_DOUBLE_EQ(owned(row, component), expected[component]);
            for (size_t row = 0; row < field->num_local_entries(); ++row)
                EXPECT_DOUBLE_EQ(overlap(row, component), expected[component]);
        }
    }
}
} // namespace

TEST(FieldStoredTest, CompiledProvidersCoverEveryDefaultAliasAndConstructor)
{
    const auto mesh = make_handle();
    using namespace SimpleFluid;
    expect_compiled_field_constructors<ScalarCellFieldStored<>>(mesh, 5.0, {5.0});
    expect_compiled_field_constructors<VectorCellFieldStored<>>(mesh, {2.0, 3.0, 5.0}, {2.0, 3.0, 5.0});
    expect_compiled_field_constructors<TensorCellFieldStored<>>(mesh,
        {{{2.0, 3.0, 5.0}, {7.0, 11.0, 13.0}, {17.0, 19.0, 23.0}}},
        {2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0});
    expect_compiled_field_constructors<ScalarFaceFieldStored<>>(mesh, 7.0, {7.0});
    expect_compiled_field_constructors<VectorFaceFieldStored<>>(mesh, {7.0, 11.0, 13.0}, {7.0, 11.0, 13.0});
    expect_compiled_field_constructors<ScalarBoundaryFaceFieldStored<>>(mesh, 11.0, {11.0});
    expect_compiled_field_constructors<VectorBoundaryFaceFieldStored<>>(
        mesh, {17.0, 19.0, 23.0}, {17.0, 19.0, 23.0});
}
