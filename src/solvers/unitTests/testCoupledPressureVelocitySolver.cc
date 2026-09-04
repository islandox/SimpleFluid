/**
 * @file testCoupledPressureVelocitySolver.cc
 * @brief Direct tests for mesh-generic coupled pressure-velocity assembly.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include "utils/testing_environment.hh"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using NativeSolver = SimpleFluid::CoupledPressureVelocitySolver<Pack, Handle>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

static_assert(std::is_same_v<SimpleFluid::CoupledPressureVelocitySolver<Pack>::mesh_type, SimpleFluid::Mesh<Pack>>);
static_assert(std::is_same_v<NativeSolver::field_type, SimpleFluid::ScalarCellFieldStored<Pack, Handle>>);

SimpleFluid::SP<const Handle> make_native_mesh()
{
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 0.5, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<Handle>(std::move(cartesian));
}

} // namespace

TEST(CoupledPressureVelocitySolverTest, AssemblesDirectlyOnNativeMeshHandleFields)
{
    const auto mesh = make_native_mesh();
    ASSERT_FALSE(mesh->legacy_mesh());

    SimpleFluid::VectorCellFieldStored<Pack, Handle> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, SimpleFluid::vec3<double>{});
    SimpleFluid::ScalarCellFieldStored<Pack, Handle> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0);
    SimpleFluid::ScalarFaceFieldStored<Pack, Handle> face_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("face_fluxes"), mesh, 0.0);

    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    const SimpleFluid::TimeStepperOptions time_options;
    const SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> momentum_equation(mesh);
    const NativeSolver solver(mesh);

    const auto system = solver.assemble(
        momentum_equation, velocity, pressure, face_fluxes, velocity_boundary_cache, boundary_conditions, time_options);

    ASSERT_FALSE(system.matrix.is_null());
    ASSERT_FALSE(system.rhs.is_null());
    EXPECT_EQ(system.map->getGlobalNumElements(), 4 * mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(system.matrix->getGlobalNumRows(), system.map->getGlobalNumElements());
    EXPECT_EQ(system.momentum->getGlobalNumRows(), mesh->owned_cell_map()->getGlobalNumElements());
}

TEST(CoupledPressureVelocitySolverTest, LegacyZeroTargetRetainsArithmeticInteriorInterpolation)
{
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 0.25, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    SimpleFluid::SP<const Handle> mesh = std::make_shared<Handle>(std::move(geometry));
    SimpleFluid::VectorCellFieldStored<Pack, Handle> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, SimpleFluid::vec3<double>{});
    SimpleFluid::ScalarCellFieldStored<Pack, Handle> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0);
    SimpleFluid::ScalarFaceFieldStored<Pack, Handle> face_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("face_fluxes"), mesh, 0.0);
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    const SimpleFluid::TimeStepperOptions time_options;
    const SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> momentum_equation(mesh);
    const NativeSolver solver(mesh);
    const auto system = solver.assemble(
        momentum_equation, velocity, pressure, face_fluxes, velocity_boundary_cache, boundary_conditions, time_options);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        for (const auto face : mesh->faces(cell))
        {
            if (!mesh->is_interior_face(face))
            {
                continue;
            }
            const auto other = mesh->opposite_or_periodic_neighbor_cell(face, cell);
            const auto other_gid = mesh->cell_global_id(other);
            const auto other_column = system.divergence[0]->getColMap()->getLocalElement(other_gid);
            typename Pack::matrix_type::local_inds_host_view_type columns;
            typename Pack::matrix_type::values_host_view_type values;
            system.divergence[0]->getLocalRowView(cell, columns, values);
            bool found = false;
            for (size_t entry = 0; entry < columns.extent(0); ++entry)
            {
                if (columns(entry) == other_column)
                {
                    const auto expected = 0.5 * mesh->face_area_vector_outward(face, cell).x;
                    EXPECT_DOUBLE_EQ(values(entry), expected);
                    found = true;
                }
            }
            EXPECT_TRUE(found);
        }
    }
}

TEST(CoupledPressureVelocitySolverTest, AddsIntegratedVolumeTargetToContinuityRhs)
{
    const auto mesh = make_native_mesh();
    SimpleFluid::VectorCellFieldStored<Pack, Handle> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, SimpleFluid::vec3<double>{});
    SimpleFluid::ScalarCellFieldStored<Pack, Handle> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0);
    SimpleFluid::ScalarFaceFieldStored<Pack, Handle> face_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("face_fluxes"), mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.pressure["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    const SimpleFluid::TimeStepperOptions time_options;
    const SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> momentum_equation(mesh);
    const NativeSolver zero_solver(mesh);
    const NativeSolver target_solver(mesh);
    const std::vector<Pack::scalar_type> target_values{0.01, 0.02};
    const SimpleFluid::VolumeContinuityTarget<Pack, Handle> target(mesh, target_values, 31);

    const auto zero_system = zero_solver.assemble(
        momentum_equation, velocity, pressure, face_fluxes, velocity_boundary_cache, boundary_conditions, time_options);
    const auto target_system = target_solver.assemble(momentum_equation, velocity, pressure, face_fluxes,
        velocity_boundary_cache, boundary_conditions, time_options, target);
    const auto zero_rhs = zero_system.rhs->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto target_rhs = target_system.rhs->getLocalViewHost(Tpetra::Access::ReadOnly);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_NEAR(target_rhs(4 * owned + 3, 0) - zero_rhs(4 * owned + 3, 0), target_values[owned], 1.0e-14);
    }
    EXPECT_EQ(target_system.continuity_target_generation, 31U);
    EXPECT_EQ(target_system.geometry_epoch, target.geometry_epoch());
}

TEST(CoupledPressureVelocitySolverTest, RejectsGloballyIncompatibleClosedVolumeTarget)
{
    const auto mesh = make_native_mesh();
    SimpleFluid::VectorCellFieldStored<Pack, Handle> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, SimpleFluid::vec3<double>{});
    SimpleFluid::ScalarCellFieldStored<Pack, Handle> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0);
    SimpleFluid::ScalarFaceFieldStored<Pack, Handle> face_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("face_fluxes"), mesh, 0.0);
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    const SimpleFluid::TimeStepperOptions time_options;
    const SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> momentum_equation(mesh);
    const NativeSolver solver(mesh);
    const SimpleFluid::VolumeContinuityTarget<Pack, Handle> target(mesh, std::vector<Pack::scalar_type>{0.01, 0.0}, 4);

    EXPECT_THROW(solver.assemble(momentum_equation, velocity, pressure, face_fluxes, velocity_boundary_cache,
                     boundary_conditions, time_options, target),
        std::invalid_argument);
}

TEST(CoupledPressureVelocitySolverTest, FixedBoundaryFluxIsExactAndUsesCorrectionGauge)
{
    const auto mesh = make_native_mesh();
    SimpleFluid::VectorCellFieldStored<Pack, Handle> velocity(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, SimpleFluid::vec3<double>{});
    SimpleFluid::ScalarCellFieldStored<Pack, Handle> pressure(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0);
    SimpleFluid::ScalarFaceFieldStored<Pack, Handle> face_fluxes(
        SimpleFluid::ScalarFaceFieldDescriptor<Pack>("face_fluxes"), mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundary_conditions.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, SimpleFluid::vec3<double>{0.0, 0.0, 8.0}};
    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    const SimpleFluid::TimeStepperOptions time_options;
    const SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> momentum_equation(mesh);
    NativeSolver solver(mesh);
    constexpr Pack::scalar_type fixed_flux = 0.05;
    solver.set_fixed_boundary_flux_provider(
        {"zmax"}, [](int, size_t, Pack::local_ordinal_type) { return Pack::scalar_type{0.05}; }, 9);
    const SimpleFluid::VolumeContinuityTarget<Pack, Handle> target(
        mesh, std::vector<Pack::scalar_type>(mesh->num_owned_cells(), fixed_flux), 9);

    const auto system = solver.assemble(momentum_equation, velocity, pressure, face_fluxes, velocity_boundary_cache,
        boundary_conditions, time_options, target);

    EXPECT_TRUE(system.pressure_gauge_gid.has_value());
    EXPECT_EQ(system.fixed_boundary_flux_revision, 1U);
    solver.apply_fixed_boundary_fluxes(face_fluxes);
    size_t fixed_faces = 0;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (mesh->is_owned_face(face_lid))
            {
                EXPECT_DOUBLE_EQ(face_fluxes.value(face_lid), fixed_flux);
                ++fixed_faces;
            }
        }
    }
    EXPECT_EQ(fixed_faces, mesh->num_owned_cells());

    const auto relaxed_cache = solver.pressure_flux_boundary_cache(velocity_boundary_cache);
    EXPECT_EQ(relaxed_cache.type_by_name.at("zmax"), SimpleFluid::BoundaryConditionType::Neumann);

    solver.clear_fixed_boundary_flux_provider();
    EXPECT_THROW(solver.solve(system, velocity, pressure, SimpleFluid::LinearSolverOptions{}), std::invalid_argument);
}
