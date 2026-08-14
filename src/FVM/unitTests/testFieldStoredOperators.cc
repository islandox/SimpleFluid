/**
 * @file testFieldStoredOperators.cc
 * @brief Tests FVM operator overloads for mesh-aware stored fields.
 */

#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>
#include <optional>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;
using PartitionedCartesian = SimpleFluid::Meshes::PartitionedMesh<Cartesian, Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<const Handle> make_cartesian_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<const Handle> make_semi_structured_handle()
{
    auto mesh = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 3}, {1, 2, 3}}, SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<const Handle> make_cylindrical_handle()
{
    auto mesh = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{1.0, 2.0, 3.0}, {0.0, 0.5, 1.0}, {0.0, 1.0, 2.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<const PartitionedCartesian> make_partitioned_cartesian()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}});
    PartitionedCartesian::indexer_type indexer(mesh->indexer());
    return std::make_shared<PartitionedCartesian>(std::move(mesh), std::move(indexer));
}

template<class MeshType> void expect_linear_stored_gradients(const SimpleFluid::SP<const MeshType>& mesh)
{
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> scalar(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("phi"), mesh);
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> scalar_gradient(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("grad_phi"), mesh);
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> vector(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh);
    SimpleFluid::TensorCellFieldStored<Pack, MeshType> vector_gradient(
        SimpleFluid::TensorCellFieldDescriptor<Pack>("grad_velocity"), mesh);

    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        const auto center = mesh->cell_centroid(lid);
        scalar.set_owned_value(lid, 2.0 * center.x - 3.0 * center.y + 4.0 * center.z);
        vector.set_owned_value(lid, {center.x, 2.0 * center.y, 3.0 * center.z});
    }
    scalar.sync_ghosts();
    vector.sync_ghosts();

    auto expect_gradients = [&](const char* stage)
    {
        SCOPED_TRACE(stage);
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto lid = static_cast<Pack::local_ordinal_type>(cell);
            const auto gradient = scalar_gradient.value(lid);
            EXPECT_NEAR(gradient.x, 2.0, 1.0e-12);
            EXPECT_NEAR(gradient.y, -3.0, 1.0e-12);
            EXPECT_NEAR(gradient.z, 4.0, 1.0e-12);

            const auto tensor = vector_gradient.value(lid);
            EXPECT_NEAR(tensor[0].x, 1.0, 1.0e-12);
            EXPECT_NEAR(tensor[0].y, 0.0, 1.0e-12);
            EXPECT_NEAR(tensor[1].x, 0.0, 1.0e-12);
            EXPECT_NEAR(tensor[1].y, 2.0, 1.0e-12);
            EXPECT_NEAR(tensor[2].z, 3.0, 1.0e-12);
        }
    };

    SimpleFluid::FVM::cell_gradient(scalar, scalar_gradient);
    SimpleFluid::FVM::cell_gradient(vector, vector_gradient);
    expect_gradients("interior least squares");

    SimpleFluid::FVM::CellGradientCache<Pack, MeshType> cache(mesh);
    SimpleFluid::FVM::cell_gradient(scalar, scalar_gradient, cache);
    SimpleFluid::FVM::cell_gradient(vector, vector_gradient, cache);
    expect_gradients("cached interior least squares");

    auto boundary_center = [&](int batch_id, size_t in_batch_id)
    {
        const auto& batch = mesh->boundary_face_batch(batch_id);
        return mesh->face_centroid(batch.face_lids.at(in_batch_id));
    };
    auto scalar_boundary = [&](int batch_id, size_t in_batch_id)
    {
        const auto center = boundary_center(batch_id, in_batch_id);
        return 2.0 * center.x - 3.0 * center.y + 4.0 * center.z;
    };
    auto scalar_condition = [&](int batch_id, size_t in_batch_id)
    {
        return SimpleFluid::BoundaryCondition{
            SimpleFluid::BoundaryConditionType::Dirichlet, scalar_boundary(batch_id, in_batch_id)};
    };
    auto vector_boundary = [&](int batch_id, size_t in_batch_id)
    {
        const auto center = boundary_center(batch_id, in_batch_id);
        return SimpleFluid::vec3<double>{center.x, 2.0 * center.y, 3.0 * center.z};
    };

    SimpleFluid::FVM::cell_gradient(scalar, scalar_condition, scalar_boundary, scalar_gradient);
    SimpleFluid::FVM::cell_gradient(vector, vector_boundary, vector_gradient);
    expect_gradients("boundary least squares");

    SimpleFluid::FVM::cell_gradient(scalar, scalar_condition, scalar_boundary, scalar_gradient, cache);
    SimpleFluid::FVM::cell_gradient(vector, vector_boundary, vector_gradient, cache);
    expect_gradients("cached boundary least squares");

    SimpleFluid::FVM::gauss_linear_cell_gradient(scalar, scalar_condition, scalar_boundary, scalar_gradient);
    SimpleFluid::FVM::gauss_linear_cell_gradient(vector, vector_boundary, vector_gradient);
    expect_gradients("Gauss linear");
}

template<class MeshType>
void expect_constant_velocity_fluxes(
    const SimpleFluid::SP<const MeshType>& mesh, SimpleFluid::vec3<double> velocity_value = {1.0, -0.5, 0.25})
{
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, velocity_value);
    SimpleFluid::VectorFaceFieldStored<Pack, MeshType> face_velocity(
        SimpleFluid::VectorFaceFieldDescriptor<Pack>("face_velocity"), mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> projected_flux(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("projected_flux"), mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> direct_flux(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("direct_flux"), mesh);

    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        (void) batch;
        conditions.velocity[mesh->boundary_batch_name(batch_id)] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, velocity_value};
    }
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    SimpleFluid::FVM::face_velocities(velocity, cache, face_velocity);
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, projected_flux);
    SimpleFluid::FVM::face_fluxes(velocity, cache, direct_flux);

    for (size_t face = 0; face < mesh->num_owned_faces(); ++face)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(face);
        EXPECT_NEAR(direct_flux.value(lid), projected_flux.value(lid), 1.0e-12);
    }
    const auto divergence = SimpleFluid::FVM::cell_divergence_from_fluxes(*mesh, direct_flux);
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 0.0, 1.0e-12);
    }
}

template<class MeshType> void expect_stored_matrix_operators(const SimpleFluid::SP<const MeshType>& mesh)
{
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("fluxes"), mesh, 0.0);
    const auto diffusion = SimpleFluid::FVM::diffusion_matrix<Pack>(*mesh, 1.0);
    const auto convection = SimpleFluid::FVM::upwind_convection_matrix(*mesh, fluxes);
    const auto poisson =
        SimpleFluid::FVM::pressure_poisson_matrix<Pack>(*mesh, mesh->owned_cell_map()->getGlobalElement(0));
    const auto boundary_poisson = SimpleFluid::FVM::pressure_poisson_matrix<Pack>(*mesh,
        std::optional<Pack::global_ordinal_type>{},
        [](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0}; });
    const auto system = SimpleFluid::FVM::diffusion_system<Pack>(
        *mesh, 1.0,
        [](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0}; },
        [](Pack::local_ordinal_type) { return 1.0; });
    const auto vector_system = SimpleFluid::FVM::vector_diffusion_system<Pack>(
        *mesh, 1.0,
        [](int, size_t)
        {
            return SimpleFluid::VectorBoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {0.0, 0.0, 0.0}};
        },
        [](Pack::local_ordinal_type) { return SimpleFluid::vec3<double>{1.0, 2.0, 3.0}; });

    EXPECT_EQ(diffusion->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(convection->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(poisson->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(boundary_poisson->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(system.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(system.rhs->getLocalLength(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.rhs->getLocalLength(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.rhs->getNumVectors(), 3U);
}

} // namespace

/** @brief Covers the runtime orthogonal mesh/FieldStored specialization. */
TEST(FieldStoredOperatorsTest, SupportsOrthogonalMeshHandle)
{
    const auto mesh = make_cartesian_handle();
    expect_linear_stored_gradients(mesh);
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);
}

/** @brief Covers static dispatch through PartitionedMesh and FieldStored. */
TEST(FieldStoredOperatorsTest, SupportsStaticOrthogonalMesh)
{
    const auto mesh = make_partitioned_cartesian();
    expect_linear_stored_gradients(mesh);
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);
}

/** @brief Covers the cylindrical branch of the runtime orthogonal handle. */
TEST(FieldStoredOperatorsTest, SupportsCylindricalOrthogonalMeshHandle)
{
    const auto mesh = make_cylindrical_handle();
    expect_constant_velocity_fluxes(mesh, {0.0, 0.0, 1.0});
    expect_stored_matrix_operators(mesh);
}

/** @brief Covers semi-structured mesh-handle fields and matrix operators. */
TEST(FieldStoredOperatorsTest, SupportsSemiStructuredMeshHandle)
{
    const auto mesh = make_semi_structured_handle();
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);

    SimpleFluid::ScalarCellFieldStored<Pack> scalar(SimpleFluid::ScalarCellFieldDescriptor<Pack>("z"), mesh);
    SimpleFluid::VectorCellFieldStored<Pack> gradient(SimpleFluid::VectorCellFieldDescriptor<Pack>("grad_z"), mesh);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        scalar.set_owned_value(lid, mesh->cell_centroid(lid).z);
    }
    scalar.sync_ghosts();
    SimpleFluid::FVM::cell_gradient(scalar, gradient);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto value = gradient.value(static_cast<Pack::local_ordinal_type>(cell));
        EXPECT_NEAR(value.x, 0.0, 1.0e-12);
        EXPECT_NEAR(value.y, 0.0, 1.0e-12);
        EXPECT_NEAR(value.z, 1.0, 1.0e-12);
    }
}
