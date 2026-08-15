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

#include <Teuchos_Array.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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

SimpleFluid::SP<const Handle> make_skewed_semi_structured_handle()
{
    auto mesh =
        std::make_shared<SemiStructured>(SimpleFluid::Arr<SemiStructured::Vec3>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                             {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.35, 0.40, 0.0}},
            SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}},
            SimpleFluid::ArrReal{0.0, 1.0});
    return std::make_shared<Handle>(std::move(mesh));
}

double stored_matrix_entry(
    const Pack::matrix_type& matrix, Pack::local_ordinal_type row, Pack::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    Pack::matrix_type::nonconst_local_inds_host_view_type columns("stored_columns", row_entries);
    Pack::matrix_type::nonconst_values_host_view_type values("stored_values", row_entries);
    size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    double result = 0.0;
    for (size_t entry = 0; entry < num_entries; ++entry)
    {
        if (columns(entry) == column)
        {
            result += values(entry);
        }
    }
    return result;
}

template<class MeshType>
std::vector<double> stored_matrix_action(
    const Pack::matrix_type& matrix, const SimpleFluid::VectorCellFieldStored<Pack, MeshType>& field, size_t component)
{
    std::vector<double> result(field.num_owned_cells(), 0.0);
    for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
    {
        const auto row = static_cast<Pack::local_ordinal_type>(owned);
        const auto row_entries = matrix.getNumEntriesInLocalRow(row);
        Pack::matrix_type::nonconst_local_inds_host_view_type columns("stored_action_columns", row_entries);
        Pack::matrix_type::nonconst_values_host_view_type values("stored_action_values", row_entries);
        size_t num_entries = 0;
        matrix.getLocalRowCopy(row, columns, values, num_entries);
        for (size_t entry = 0; entry < num_entries; ++entry)
        {
            result[owned] += values(entry) * field.local_value(columns(entry)).component(component);
        }
    }
    return result;
}

template<class MeshType>
std::vector<double> stored_matrix_action(
    const Pack::matrix_type& matrix, const SimpleFluid::ScalarCellFieldStored<Pack, MeshType>& field)
{
    std::vector<double> result(field.num_owned_cells(), 0.0);
    for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
    {
        const auto row = static_cast<Pack::local_ordinal_type>(owned);
        const auto row_entries = matrix.getNumEntriesInLocalRow(row);
        Pack::matrix_type::nonconst_local_inds_host_view_type columns("stored_scalar_action_columns", row_entries);
        Pack::matrix_type::nonconst_values_host_view_type values("stored_scalar_action_values", row_entries);
        size_t num_entries = 0;
        matrix.getLocalRowCopy(row, columns, values, num_entries);
        for (size_t entry = 0; entry < num_entries; ++entry)
        {
            result[owned] += values(entry) * field.local_value(columns(entry));
        }
    }
    return result;
}

double skewed_linear_scalar(const SimpleFluid::vec3<double>& point)
{
    return 1.5 * point.x - 0.75 * point.y + 0.4 * point.z;
}

SimpleFluid::vec3<double> skewed_linear_velocity(const SimpleFluid::vec3<double>& point)
{
    return {point.x + 0.25 * point.y, -0.5 * point.x + 2.0 * point.y, point.x - point.y};
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
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> old_scalar(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("old_scalar"), mesh, 1.0);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> storage_weight(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("storage_weight"), mesh, 6.0);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> advection_weight(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("advection_weight"), mesh, 6.0);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> density(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("density"), mesh, 2.0);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> specific_heat_capacity(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("specific_heat_capacity"), mesh, 3.0);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> thermal_conductivity(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("thermal_conductivity"), mesh, 0.25);
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> old_velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("old_velocity"), mesh, {1.0, 2.0, 3.0});
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> dynamic_viscosity(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("dynamic_viscosity"), mesh, 0.25);
    SimpleFluid::FVM::TransportGeometryCache<MeshType> geometry_cache(*mesh);
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
    const auto scalar_transport = SimpleFluid::FVM::transport_system<Pack>(
        old_scalar, fluxes, 0.5, 0.1,
        [](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 1.0}; },
        [](int, size_t) { return 1.0; }, [](Pack::local_ordinal_type) { return 0.5; });
    const auto weighted_scalar = SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
        old_scalar, fluxes, 0.5, storage_weight, advection_weight, thermal_conductivity,
        [](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 1.0}; },
        [](int, size_t) { return 1.0; }, [](Pack::local_ordinal_type) { return 0.5; },
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    const auto physical_temperature = SimpleFluid::FVM::physical_temperature_transport_system<Pack>(
        old_scalar, fluxes, 0.5, density, specific_heat_capacity, thermal_conductivity,
        [](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 1.0}; },
        [](int, size_t) { return 1.0; }, [](Pack::local_ordinal_type) { return 0.5; },
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    const auto vector_transport = SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(
        old_velocity, fluxes, 0.5, 0.1, [](int, size_t) { return SimpleFluid::vec3<double>{1.0, 2.0, 3.0}; },
        [](Pack::local_ordinal_type) { return SimpleFluid::vec3<double>{0.5, 0.25, -0.5}; },
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, Teuchos::null,
        SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, &geometry_cache);
    const auto physical_momentum = SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
        old_velocity, fluxes, 0.5, dynamic_viscosity, 1.5,
        [](int, size_t) { return SimpleFluid::vec3<double>{1.0, 2.0, 3.0}; }, [](Pack::local_ordinal_type)
        { return SimpleFluid::vec3<double>{0.5, 0.25, -0.5}; }, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
        nullptr, Teuchos::null, SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, nullptr, &geometry_cache);

    EXPECT_EQ(diffusion->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(convection->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(poisson->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(boundary_poisson->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(system.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(system.rhs->getLocalLength(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.rhs->getLocalLength(), mesh->num_owned_cells());
    EXPECT_EQ(vector_system.rhs->getNumVectors(), 3U);
    EXPECT_EQ(scalar_transport.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(scalar_transport.rhs->getLocalLength(), mesh->num_owned_cells());
    EXPECT_EQ(weighted_scalar.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(physical_temperature.matrix->getLocalNumRows(), mesh->num_owned_cells());
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            EXPECT_NEAR(stored_matrix_entry(*physical_temperature.matrix, row_lid, column_lid),
                stored_matrix_entry(*weighted_scalar.matrix, row_lid, column_lid), 1.0e-12);
        }
        EXPECT_NEAR(physical_temperature.rhs->getData()[row], weighted_scalar.rhs->getData()[row], 1.0e-12);
    }
    EXPECT_EQ(vector_transport.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(vector_transport.rhs->getNumVectors(), 3U);
    EXPECT_EQ(physical_momentum.matrix->getLocalNumRows(), mesh->num_owned_cells());
    EXPECT_EQ(physical_momentum.rhs->getNumVectors(), 3U);
}

template<class MeshType> void expect_pressure_weighted_stored_fluxes(const SimpleFluid::SP<const MeshType>& mesh)
{
    using local_ordinal_type = Pack::local_ordinal_type;
    using scalar_type = Pack::scalar_type;

    const SimpleFluid::vec3<scalar_type> velocity_value{0.25, -0.5, 0.75};
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, velocity_value);
    SimpleFluid::ScalarCellFieldStored<Pack, MeshType> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> baseline(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("baseline_flux"), mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> automatic(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("automatic_flux"), mesh);
    SimpleFluid::ScalarFaceFieldStored<Pack, MeshType> precomputed(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("precomputed_flux"), mesh);
    SimpleFluid::VectorCellFieldStored<Pack, MeshType> pressure_gradient(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("pressure_gradient"), mesh);

    SimpleFluid::BoundaryConditionSet conditions;
    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        (void) batch;
        const auto& name = mesh->boundary_batch_name(batch_id);
        conditions.velocity[name] = {SimpleFluid::BoundaryConditionType::Dirichlet, velocity_value};
        pressure_boundaries[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    const auto boundary_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);

    SimpleFluid::FVM::face_fluxes(velocity, boundary_cache, baseline);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(velocity, pressure, scalar_type{}, boundary_cache, automatic);
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (baseline.is_owned(face_lid))
        {
            EXPECT_DOUBLE_EQ(automatic.value(face_lid), baseline.value(face_lid));
        }
    }

    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        pressure.set_owned_value(cell_lid, cell % 3 == 0   ? scalar_type{-1}
                                           : cell % 3 == 1 ? scalar_type{2}
                                                           : scalar_type{0.5});
    }
    pressure.sync_ghosts();

    SimpleFluid::FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType> workspace(mesh);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, scalar_type{0.2}, boundary_cache, pressure_boundaries, workspace, automatic);
    SimpleFluid::FVM::cell_gradient(pressure, pressure_boundaries, pressure_gradient, workspace.gradient_cache(),
        SimpleFluid::FVM::CellGradientScheme::LeastSquares);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(velocity, pressure, pressure_gradient, scalar_type{0.2},
        boundary_cache, pressure_boundaries, workspace, precomputed);

    bool observed_pressure_correction = false;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!automatic.is_owned(face_lid))
        {
            continue;
        }
        EXPECT_TRUE(std::isfinite(automatic.value(face_lid)));
        EXPECT_NEAR(automatic.value(face_lid), precomputed.value(face_lid), 1.0e-12);
        observed_pressure_correction =
            observed_pressure_correction || std::abs(automatic.value(face_lid) - baseline.value(face_lid)) > 1.0e-12;
    }
    EXPECT_TRUE(observed_pressure_correction);

    EXPECT_THROW(SimpleFluid::FVM::pressure_weighted_face_fluxes(
                     velocity, pressure, scalar_type{-1}, boundary_cache, workspace, automatic),
        std::invalid_argument);

    SimpleFluid::BoundaryConditionSet open_conditions;
    SimpleFluid::BoundaryConditionMap open_pressure_boundaries;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        (void) batch;
        const auto& name = mesh->boundary_batch_name(batch_id);
        open_conditions.velocity[name] = {SimpleFluid::BoundaryConditionType::Neumann, {}};
        open_pressure_boundaries[name] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.25};
    }
    const auto open_boundary_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, open_conditions);
    EXPECT_NO_THROW(
        SimpleFluid::FVM::pressure_weighted_face_fluxes(velocity, pressure, scalar_type{0.2}, open_boundary_cache,
            open_pressure_boundaries, workspace, automatic, SimpleFluid::FVM::CellGradientScheme::GaussLinear));
    EXPECT_THROW(SimpleFluid::FVM::pressure_weighted_face_fluxes(velocity, pressure, scalar_type{0.2}, boundary_cache,
                     open_pressure_boundaries, workspace, automatic),
        std::invalid_argument);
}

} // namespace

/** @brief Covers the runtime orthogonal mesh/FieldStored specialization. */
TEST(FieldStoredOperatorsTest, SupportsOrthogonalMeshHandle)
{
    const auto mesh = make_cartesian_handle();
    expect_linear_stored_gradients(mesh);
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);
    expect_pressure_weighted_stored_fluxes(mesh);
}

/** @brief Covers static dispatch through PartitionedMesh and FieldStored. */
TEST(FieldStoredOperatorsTest, SupportsStaticOrthogonalMesh)
{
    const auto mesh = make_partitioned_cartesian();
    expect_linear_stored_gradients(mesh);
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);
    expect_pressure_weighted_stored_fluxes(mesh);
}

/** @brief Rank-local mapped transport failures reject coherently without mutating matrix caches. */
TEST(FieldStoredOperatorsTest, MappedTransportValidationAndCachedGraphFailuresAreCollective)
{
    const auto mesh = make_cartesian_handle();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() != 2)
    {
        GTEST_SKIP() << "This regression requires exactly two MPI ranks.";
    }
    const auto rank = communicator->getRank();

    SimpleFluid::ScalarFaceFieldStored<Pack> fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("collective_fluxes"), mesh, 0.0);
    SimpleFluid::VectorCellFieldStored<Pack> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("collective_velocity"), mesh, {1.0, 0.5, -0.25});
    SimpleFluid::ScalarCellFieldStored<Pack> scalar(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("collective_scalar"), mesh, 1.0);
    SimpleFluid::ScalarCellFieldStored<Pack> viscosity(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("collective_viscosity"), mesh, 0.4);
    const auto other_mesh = make_cartesian_handle();
    SimpleFluid::ScalarFaceFieldStored<Pack> other_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("other_collective_fluxes"), other_mesh, 0.0);
    velocity.sync_ghosts();
    scalar.sync_ghosts();
    viscosity.sync_ghosts();

    SimpleFluid::FVM::TransportGeometryCache<Handle> geometry_cache(*mesh);
    SimpleFluid::FVM::TransportGeometryCache<Handle> other_geometry_cache(*other_mesh);
    auto scalar_boundary_condition = [](int, size_t)
    { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0}; };
    auto scalar_boundary_value = [](int, size_t) { return 0.0; };
    auto scalar_zero_source = [](Pack::local_ordinal_type) { return 0.0; };
    auto assemble_orthogonal_scalar =
        [&](const SimpleFluid::ScalarFaceFieldStored<Pack>& selected_fluxes, double time_step, double diffusivity)
    {
        return SimpleFluid::FVM::transport_system<Pack>(scalar, selected_fluxes, time_step, diffusivity,
            scalar_boundary_condition, scalar_boundary_value, scalar_zero_source);
    };
    auto boundary_value = [](int, size_t) { return SimpleFluid::vec3<double>{}; };
    auto zero_source = [](Pack::local_ordinal_type) { return SimpleFluid::vec3<double>{}; };
    auto assemble = [&](double time_step, Teuchos::RCP<Pack::matrix_type> cached,
                        const SimpleFluid::FVM::TransportGeometryCache<Handle>* geometry)
    {
        return SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(velocity, fluxes, time_step, 0.2, boundary_value,
            zero_source, SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, std::move(cached),
            SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, geometry);
    };

    EXPECT_THROW(assemble_orthogonal_scalar(rank == 0 ? other_fluxes : fluxes, 0.5, 0.2), std::invalid_argument);
    EXPECT_THROW(assemble_orthogonal_scalar(fluxes, rank == 0 ? -0.5 : 0.5, 0.2), std::invalid_argument);
    EXPECT_THROW(assemble_orthogonal_scalar(fluxes, 0.5, rank == 0 ? -0.2 : 0.2), std::invalid_argument);
    EXPECT_THROW(assemble(rank == 0 ? -0.5 : 0.5, Teuchos::null, &geometry_cache), std::invalid_argument);
    EXPECT_THROW(
        assemble(0.5, Teuchos::null, rank == 0 ? &other_geometry_cache : &geometry_cache), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(scalar, fluxes, 0.5, rank == 0 ? -0.2 : 0.2,
                     scalar_boundary_condition, scalar_boundary_value, scalar_zero_source,
                     SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, Teuchos::null, &geometry_cache),
        std::invalid_argument);

    SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, Handle> viscosity_cache;
    viscosity_cache.mesh = mesh;
    SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, Handle> other_viscosity_cache;
    other_viscosity_cache.mesh = other_mesh;
    EXPECT_THROW(SimpleFluid::FVM::physical_momentum_transport_system<Pack>(velocity, fluxes, 0.5, viscosity, 1.0,
                     boundary_value, zero_source, SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr,
                     Teuchos::null, SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{},
                     rank == 0 ? &other_viscosity_cache : &viscosity_cache, &geometry_cache),
        std::invalid_argument);

    const auto good = assemble(0.5, Teuchos::null, &geometry_cache);
    auto incompatible = Teuchos::rcp(new Pack::matrix_type(
        mesh->owned_cell_map(), mesh->overlap_cell_map(), good.matrix->getLocalMaxNumRowEntries()));
    for (size_t row = 0; row < good.matrix->getLocalNumRows(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        good.matrix->getLocalRowView(static_cast<Pack::local_ordinal_type>(row), columns, values);
        Teuchos::Array<Pack::local_ordinal_type> copied_columns;
        Teuchos::Array<Pack::scalar_type> copied_values;
        copied_columns.reserve(columns.extent(0));
        copied_values.reserve(values.extent(0));
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            if (rank == 0 && row == 0 && entry == 0)
            {
                continue;
            }
            copied_columns.push_back(columns[entry]);
            copied_values.push_back(values[entry] + 7.0);
        }
        incompatible->insertLocalValues(static_cast<Pack::local_ordinal_type>(row), copied_columns(), copied_values());
    }
    incompatible->fillComplete();

    struct RowSnapshot
    {
        std::vector<Pack::local_ordinal_type> columns;
        std::vector<Pack::scalar_type> values;
    };
    std::vector<RowSnapshot> before(incompatible->getLocalNumRows());
    for (size_t row = 0; row < before.size(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        incompatible->getLocalRowView(static_cast<Pack::local_ordinal_type>(row), columns, values);
        before[row].columns.assign(columns.data(), columns.data() + columns.extent(0));
        before[row].values.assign(values.data(), values.data() + values.extent(0));
    }

    EXPECT_THROW(assemble(0.25, incompatible, &geometry_cache), std::invalid_argument);
    EXPECT_TRUE(incompatible->isFillComplete());
    for (size_t row = 0; row < before.size(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        incompatible->getLocalRowView(static_cast<Pack::local_ordinal_type>(row), columns, values);
        ASSERT_EQ(columns.extent(0), before[row].columns.size());
        ASSERT_EQ(values.extent(0), before[row].values.size());
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            EXPECT_EQ(columns[entry], before[row].columns[entry]);
            EXPECT_DOUBLE_EQ(values[entry], before[row].values[entry]);
        }
    }
}

/** @brief Covers the cylindrical branch of the runtime orthogonal handle. */
TEST(FieldStoredOperatorsTest, SupportsCylindricalOrthogonalMeshHandle)
{
    const auto mesh = make_cylindrical_handle();
    expect_constant_velocity_fluxes(mesh, {0.0, 0.0, 1.0});
    expect_stored_matrix_operators(mesh);
    expect_pressure_weighted_stored_fluxes(mesh);
}

/** @brief Covers semi-structured mesh-handle fields and matrix operators. */
TEST(FieldStoredOperatorsTest, SupportsSemiStructuredMeshHandle)
{
    const auto mesh = make_semi_structured_handle();
    expect_constant_velocity_fluxes(mesh);
    expect_stored_matrix_operators(mesh);
    expect_pressure_weighted_stored_fluxes(mesh);

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

/** @brief Mapped scalar transport matches explicit/implicit/hybrid residuals. */
TEST(FieldStoredOperatorsTest, SemiStructuredScalarTransportHonorsNonOrthogonalTreatmentAndCaches)
{
    const auto mesh = make_skewed_semi_structured_handle();
    SimpleFluid::ScalarFaceFieldStored<Pack> fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("zero_scalar_fluxes"), mesh, 0.0);
    SimpleFluid::ScalarCellFieldStored<Pack> scalar(SimpleFluid::ScalarCellFieldDescriptor<Pack>("scalar"), mesh);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        scalar.set_owned_value(lid, skewed_linear_scalar(mesh->cell_centroid(lid)));
    }
    scalar.sync_ghosts();

    auto boundary_value = [&](int batch_id, size_t in_batch)
    {
        const auto face_lid = mesh->boundary_face_batch(batch_id).face_lids.at(in_batch);
        return skewed_linear_scalar(mesh->face_centroid(face_lid));
    };
    auto boundary_condition = [&](int batch_id, size_t in_batch)
    {
        return SimpleFluid::BoundaryCondition{
            SimpleFluid::BoundaryConditionType::Dirichlet, boundary_value(batch_id, in_batch)};
    };
    auto source = [](Pack::local_ordinal_type) { return 0.125; };
    SimpleFluid::FVM::TransportGeometryCache<Handle> geometry_cache(*mesh);
    auto assemble = [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
                        const SimpleFluid::ScalarCellFieldStored<Pack>* correction, double time_step,
                        double diffusivity, Teuchos::RCP<Pack::matrix_type> cached,
                        const SimpleFluid::FVM::TransportGeometryCache<Handle>* cache)
    {
        return SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(scalar, fluxes, time_step, diffusivity,
            boundary_condition, boundary_value, source, treatment, correction, std::move(cached), cache);
    };

    const auto orthogonal =
        assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, 0.5, 0.7, Teuchos::null, &geometry_cache);
    const auto explicit_system =
        assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, &scalar, 0.5, 0.7, Teuchos::null, &geometry_cache);
    auto implicit_system =
        assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.5, 0.7, Teuchos::null, &geometry_cache);
    const auto hybrid_system =
        assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, &scalar, 0.5, 0.7, Teuchos::null, &geometry_cache);
    const auto uncached_geometry =
        assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.5, 0.7, Teuchos::null, nullptr);

    double explicit_rhs_delta = 0.0;
    double implicit_matrix_delta = 0.0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            const auto baseline = stored_matrix_entry(*orthogonal.matrix, row_lid, column_lid);
            const auto explicit_value = stored_matrix_entry(*explicit_system.matrix, row_lid, column_lid);
            const auto implicit_value = stored_matrix_entry(*implicit_system.matrix, row_lid, column_lid);
            const auto hybrid_value = stored_matrix_entry(*hybrid_system.matrix, row_lid, column_lid);
            EXPECT_NEAR(explicit_value, baseline, 1.0e-12);
            EXPECT_NEAR(hybrid_value, 0.5 * (baseline + implicit_value), 1.0e-11);
            EXPECT_NEAR(stored_matrix_entry(*uncached_geometry.matrix, row_lid, column_lid), implicit_value, 1.0e-12);
            implicit_matrix_delta += std::abs(implicit_value - baseline);
        }
        const auto baseline = orthogonal.rhs->getData()[row];
        const auto explicit_value = explicit_system.rhs->getData()[row];
        const auto implicit_value = implicit_system.rhs->getData()[row];
        const auto hybrid_value = hybrid_system.rhs->getData()[row];
        EXPECT_NEAR(hybrid_value, 0.5 * (explicit_value + implicit_value), 1.0e-10);
        EXPECT_NEAR(uncached_geometry.rhs->getData()[row], implicit_value, 1.0e-12);
        explicit_rhs_delta += std::abs(explicit_value - baseline);
    }
    EXPECT_GT(explicit_rhs_delta, 1.0e-8);
    EXPECT_GT(implicit_matrix_delta, 1.0e-8);

    const auto explicit_action = stored_matrix_action(*explicit_system.matrix, scalar);
    const auto implicit_action = stored_matrix_action(*implicit_system.matrix, scalar);
    const auto hybrid_action = stored_matrix_action(*hybrid_system.matrix, scalar);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto explicit_residual = explicit_action[row] - explicit_system.rhs->getData()[row];
        const auto implicit_residual = implicit_action[row] - implicit_system.rhs->getData()[row];
        const auto hybrid_residual = hybrid_action[row] - hybrid_system.rhs->getData()[row];
        EXPECT_NEAR(implicit_residual, explicit_residual, 1.0e-10);
        EXPECT_NEAR(hybrid_residual, explicit_residual, 1.0e-10);
    }

    const auto* const matrix_storage = implicit_system.matrix.getRawPtr();
    const auto reused = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.25, 0.3,
        implicit_system.matrix, &geometry_cache);
    const auto fresh = assemble(
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.25, 0.3, Teuchos::null, &geometry_cache);
    EXPECT_EQ(reused.matrix.getRawPtr(), matrix_storage);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            EXPECT_NEAR(stored_matrix_entry(*reused.matrix, row_lid, column_lid),
                stored_matrix_entry(*fresh.matrix, row_lid, column_lid), 1.0e-12);
        }
        EXPECT_NEAR(reused.rhs->getData()[row], fresh.rhs->getData()[row], 1.0e-12);
    }
}

/** @brief Distinguishes every mapped constant-viscosity correction branch. */
TEST(FieldStoredOperatorsTest, SemiStructuredTransportHonorsNonOrthogonalTreatmentAndMatrixReuse)
{
    const auto mesh = make_skewed_semi_structured_handle();
    SimpleFluid::ScalarFaceFieldStored<Pack> fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("zero_fluxes"), mesh, 0.0);
    SimpleFluid::VectorCellFieldStored<Pack> velocity(SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        velocity.set_owned_value(lid, skewed_linear_velocity(mesh->cell_centroid(lid)));
    }
    velocity.sync_ghosts();

    double maximum_tangential_area = 0.0;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (!mesh->is_interior_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        maximum_tangential_area = std::max(maximum_tangential_area,
            SimpleFluid::FVM::detail::non_orthogonal_area_vector(
                mesh->face_area_vector_outward(face_lid, owner), mesh->cell_center_vector(face_lid, owner))
                .norm());
    }
    ASSERT_GT(maximum_tangential_area, 0.1);

    auto boundary_value = [&](int batch_id, size_t in_batch)
    {
        const auto face_lid = mesh->boundary_face_batch(batch_id).face_lids.at(in_batch);
        return skewed_linear_velocity(mesh->face_centroid(face_lid));
    };
    auto zero_source = [](Pack::local_ordinal_type) { return SimpleFluid::vec3<double>{}; };
    auto no_boundary_diffusion = [](int, size_t) { return false; };
    SimpleFluid::FVM::TransportGeometryCache<Handle> geometry_cache(*mesh);
    auto assemble = [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
                        const SimpleFluid::VectorCellFieldStored<Pack>* correction, double time_step,
                        double diffusivity, Teuchos::RCP<Pack::matrix_type> cached = Teuchos::null)
    {
        return SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(velocity, fluxes, time_step, diffusivity,
            boundary_value, zero_source, treatment, correction, std::move(cached), no_boundary_diffusion,
            &geometry_cache);
    };

    const auto orthogonal = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, 0.5, 0.7);
    const auto explicit_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, &velocity, 0.5, 0.7);
    const auto implicit_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.5, 0.7);
    const auto hybrid_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, &velocity, 0.5, 0.7);

    double explicit_rhs_delta = 0.0;
    double implicit_matrix_delta = 0.0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            const auto orthogonal_value = stored_matrix_entry(*orthogonal.matrix, row_lid, column_lid);
            const auto explicit_value = stored_matrix_entry(*explicit_system.matrix, row_lid, column_lid);
            const auto implicit_value = stored_matrix_entry(*implicit_system.matrix, row_lid, column_lid);
            const auto hybrid_value = stored_matrix_entry(*hybrid_system.matrix, row_lid, column_lid);
            EXPECT_NEAR(explicit_value, orthogonal_value, 1.0e-12);
            EXPECT_NEAR(hybrid_value, 0.5 * (orthogonal_value + implicit_value), 1.0e-11);
            implicit_matrix_delta += std::abs(implicit_value - orthogonal_value);
        }
        for (size_t component = 0; component < 3; ++component)
        {
            const auto baseline = orthogonal.rhs->getData(component)[row];
            const auto explicit_value = explicit_system.rhs->getData(component)[row];
            const auto implicit_value = implicit_system.rhs->getData(component)[row];
            const auto hybrid_value = hybrid_system.rhs->getData(component)[row];
            EXPECT_NEAR(implicit_value, baseline, 1.0e-12);
            EXPECT_NEAR(hybrid_value, 0.5 * (baseline + explicit_value), 1.0e-11);
            explicit_rhs_delta += std::abs(explicit_value - baseline);
        }
    }
    EXPECT_GT(explicit_rhs_delta, 1.0e-8);
    EXPECT_GT(implicit_matrix_delta, 1.0e-8);

    for (size_t component = 0; component < 3; ++component)
    {
        const auto explicit_action = stored_matrix_action(*explicit_system.matrix, velocity, component);
        const auto implicit_action = stored_matrix_action(*implicit_system.matrix, velocity, component);
        const auto hybrid_action = stored_matrix_action(*hybrid_system.matrix, velocity, component);
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto explicit_residual = explicit_action[row] - explicit_system.rhs->getData(component)[row];
            const auto implicit_residual = implicit_action[row] - implicit_system.rhs->getData(component)[row];
            const auto hybrid_residual = hybrid_action[row] - hybrid_system.rhs->getData(component)[row];
            EXPECT_NEAR(implicit_residual, explicit_residual, 1.0e-10);
            EXPECT_NEAR(hybrid_residual, explicit_residual, 1.0e-10);
        }
    }

    auto first = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.5, 0.7);
    const auto* const matrix_storage = first.matrix.getRawPtr();
    const auto reused = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.25, 0.3, first.matrix);
    const auto fresh = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr, 0.25, 0.3);
    EXPECT_EQ(reused.matrix.getRawPtr(), matrix_storage);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            EXPECT_NEAR(stored_matrix_entry(*reused.matrix, row_lid, column_lid),
                stored_matrix_entry(*fresh.matrix, row_lid, column_lid), 1.0e-12);
        }
        for (size_t component = 0; component < 3; ++component)
        {
            EXPECT_NEAR(reused.rhs->getData(component)[row], fresh.rhs->getData(component)[row], 1.0e-12);
        }
    }
}

/** @brief Verifies mapped physical momentum uses every viscosity input. */
TEST(FieldStoredOperatorsTest, SemiStructuredPhysicalMomentumMatchesTreatmentAndViscositySemantics)
{
    const auto mesh = make_skewed_semi_structured_handle();
    SimpleFluid::ScalarFaceFieldStored<Pack> fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("zero_fluxes"), mesh, 0.0);
    SimpleFluid::VectorCellFieldStored<Pack> velocity(SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh);
    SimpleFluid::ScalarCellFieldStored<Pack> viscosity(SimpleFluid::ScalarCellFieldDescriptor<Pack>("viscosity"), mesh);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        const auto center = mesh->cell_centroid(lid);
        velocity.set_owned_value(lid, skewed_linear_velocity(center));
        viscosity.set_owned_value(lid, 1.0 + 3.0 * center.x);
    }
    velocity.sync_ghosts();
    viscosity.sync_ghosts();

    auto boundary_value = [&](int batch_id, size_t in_batch)
    {
        const auto face_lid = mesh->boundary_face_batch(batch_id).face_lids.at(in_batch);
        return skewed_linear_velocity(mesh->face_centroid(face_lid));
    };
    auto zero_source = [](Pack::local_ordinal_type) { return SimpleFluid::vec3<double>{}; };
    auto no_boundary_diffusion = [](int, size_t) { return false; };
    SimpleFluid::FVM::TransportGeometryCache<Handle> geometry_cache(*mesh);
    auto assemble = [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
                        const SimpleFluid::VectorCellFieldStored<Pack>* correction,
                        SimpleFluid::FVM::FaceCoefficientInterpolation interpolation,
                        Teuchos::RCP<Pack::matrix_type> cached = Teuchos::null)
    {
        return SimpleFluid::FVM::physical_momentum_transport_system<Pack>(velocity, fluxes, 0.5, viscosity, 2.0,
            boundary_value, zero_source, treatment, correction, std::move(cached), no_boundary_diffusion, nullptr,
            &geometry_cache, interpolation);
    };

    const auto orthogonal = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);
    const auto explicit_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, &velocity,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);
    auto implicit_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);
    const auto hybrid_system = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, &velocity,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);

    double explicit_rhs_delta = 0.0;
    double implicit_matrix_delta = 0.0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            const auto baseline = stored_matrix_entry(*orthogonal.matrix, row_lid, column_lid);
            const auto explicit_value = stored_matrix_entry(*explicit_system.matrix, row_lid, column_lid);
            const auto implicit_value = stored_matrix_entry(*implicit_system.matrix, row_lid, column_lid);
            const auto hybrid_value = stored_matrix_entry(*hybrid_system.matrix, row_lid, column_lid);
            EXPECT_NEAR(explicit_value, baseline, 1.0e-12);
            EXPECT_NEAR(hybrid_value, 0.5 * (baseline + implicit_value), 1.0e-11);
            implicit_matrix_delta += std::abs(implicit_value - baseline);
        }
        for (size_t component = 0; component < 3; ++component)
        {
            const auto baseline = orthogonal.rhs->getData(component)[row];
            const auto explicit_value = explicit_system.rhs->getData(component)[row];
            const auto implicit_value = implicit_system.rhs->getData(component)[row];
            const auto hybrid_value = hybrid_system.rhs->getData(component)[row];
            EXPECT_NEAR(hybrid_value, 0.5 * (explicit_value + implicit_value), 1.0e-10);
            explicit_rhs_delta += std::abs(explicit_value - baseline);
        }
    }
    EXPECT_GT(explicit_rhs_delta, 1.0e-8);
    EXPECT_GT(implicit_matrix_delta, 1.0e-8);

    for (size_t component = 0; component < 3; ++component)
    {
        const auto explicit_action = stored_matrix_action(*explicit_system.matrix, velocity, component);
        const auto implicit_action = stored_matrix_action(*implicit_system.matrix, velocity, component);
        const auto hybrid_action = stored_matrix_action(*hybrid_system.matrix, velocity, component);
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto explicit_residual = explicit_action[row] - explicit_system.rhs->getData(component)[row];
            const auto implicit_residual = implicit_action[row] - implicit_system.rhs->getData(component)[row];
            const auto hybrid_residual = hybrid_action[row] - hybrid_system.rhs->getData(component)[row];
            EXPECT_NEAR(implicit_residual, explicit_residual, 1.0e-10);
            EXPECT_NEAR(hybrid_residual, explicit_residual, 1.0e-10);
        }
    }

    const auto linear = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Linear);
    double interpolation_delta = 0.0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            interpolation_delta +=
                std::abs(stored_matrix_entry(*linear.matrix, static_cast<Pack::local_ordinal_type>(row),
                             static_cast<Pack::local_ordinal_type>(column)) -
                         stored_matrix_entry(*orthogonal.matrix, static_cast<Pack::local_ordinal_type>(row),
                             static_cast<Pack::local_ordinal_type>(column)));
        }
    }
    EXPECT_GT(interpolation_delta, 1.0e-8);

    const auto* const matrix_storage = implicit_system.matrix.getRawPtr();
    const auto reused = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Linear, implicit_system.matrix);
    const auto fresh = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Linear);
    EXPECT_EQ(reused.matrix.getRawPtr(), matrix_storage);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        for (size_t column = 0; column < mesh->num_local_cells(); ++column)
        {
            EXPECT_NEAR(stored_matrix_entry(*reused.matrix, static_cast<Pack::local_ordinal_type>(row),
                            static_cast<Pack::local_ordinal_type>(column)),
                stored_matrix_entry(*fresh.matrix, static_cast<Pack::local_ordinal_type>(row),
                    static_cast<Pack::local_ordinal_type>(column)),
                1.0e-12);
        }
        for (size_t component = 0; component < 3; ++component)
        {
            EXPECT_NEAR(reused.rhs->getData(component)[row], fresh.rhs->getData(component)[row], 1.0e-12);
        }
    }

    SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, Handle> boundary_viscosity;
    boundary_viscosity.mesh = mesh;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        boundary_viscosity.value[batch_id] = SimpleFluid::Arr<double>(batch.face_lids.size(), 8.0);
    }
    auto all_boundaries = [](int, size_t) { return true; };
    auto assemble_boundary = [&](const SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, Handle>* cache)
    {
        return SimpleFluid::FVM::physical_momentum_transport_system<Pack>(velocity, fluxes, 0.5, viscosity, 2.0,
            boundary_value, zero_source, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, Teuchos::null,
            all_boundaries, cache, &geometry_cache, SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);
    };
    const auto without_boundary_cache = assemble_boundary(nullptr);
    const auto with_boundary_cache = assemble_boundary(&boundary_viscosity);
    double boundary_matrix_delta = 0.0;
    double boundary_rhs_delta = 0.0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        boundary_matrix_delta += std::abs(stored_matrix_entry(*with_boundary_cache.matrix, row_lid, row_lid) -
                                          stored_matrix_entry(*without_boundary_cache.matrix, row_lid, row_lid));
        for (size_t component = 0; component < 3; ++component)
        {
            boundary_rhs_delta += std::abs(
                with_boundary_cache.rhs->getData(component)[row] - without_boundary_cache.rhs->getData(component)[row]);
        }
    }
    EXPECT_GT(boundary_matrix_delta, 1.0e-8);
    EXPECT_GT(boundary_rhs_delta, 1.0e-8);

    auto orthogonal_graph = assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic);
    EXPECT_THROW(assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr,
                     SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, orthogonal_graph.matrix),
        std::invalid_argument);

    const auto other_mesh = make_cartesian_handle();
    auto incompatible_matrix =
        Teuchos::rcp(new Pack::matrix_type(other_mesh->owned_cell_map(), other_mesh->overlap_cell_map(), 1));
    incompatible_matrix->fillComplete();
    EXPECT_THROW(assemble(SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr,
                     SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, incompatible_matrix),
        std::invalid_argument);
}
