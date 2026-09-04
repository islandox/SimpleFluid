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
#include <type_traits>

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
