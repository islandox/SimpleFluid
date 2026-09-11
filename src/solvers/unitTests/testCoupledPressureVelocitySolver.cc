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
#include <tuple>
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
    std::vector<Pack::scalar_type> target_values(mesh->num_owned_cells());
    for (size_t owned = 0; owned < target_values.size(); ++owned)
    {
        const auto gid = mesh->owned_cell_map()->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
        target_values[owned] = 0.01 * (gid + 1);
    }
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
    std::vector<Pack::scalar_type> target_values(mesh->num_owned_cells(), 0.0);
    for (size_t owned = 0; owned < target_values.size(); ++owned)
    {
        const auto gid = mesh->owned_cell_map()->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
        target_values[owned] = gid == 0 ? 0.01 : 0.0;
    }
    const SimpleFluid::VolumeContinuityTarget<Pack, Handle> target(mesh, target_values, 4);
    ASSERT_NO_THROW(target.validate(*mesh, "Test continuity target"));

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

namespace
{
using AssemblyPurpose = SimpleFluid::CoupledAssemblyPurpose;
using Backend = SimpleFluid::CoupledOperatorBackend;
using WorkspacePolicy = SimpleFluid::CoupledWorkspacePolicy;
using Equation = SimpleFluid::IncompressibleMomentumEquation<Pack, Handle>;
using Vector = Pack::vector_type;

void expect_vector_near(const Vector& actual, const Vector& expected)
{
    Vector difference(actual, Teuchos::Copy);
    difference.update(-1.0, expected, 1.0);
    EXPECT_LE(difference.norm2(), 1.0e-12 * (1.0 + expected.norm2()));
}

class CoupledResidualAssemblyTest : public testing::TestWithParam<std::tuple<Backend, WorkspacePolicy>>
{
protected:
    const SimpleFluid::SP<const Handle> mesh =
        std::make_shared<Handle>(std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 0.15, 0.4, 0.7, 1.0}, {0.0, 0.4, 1.0}, {0.0, 0.6, 1.0}}}));
    NativeSolver::velocity_field_type velocity{SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"), mesh, {}};
    NativeSolver::field_type pressure{SimpleFluid::ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.0};
    NativeSolver::face_flux_field_type flux{SimpleFluid::ScalarFaceFieldDescriptor<Pack>("flux"), mesh, 0.015};
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::TimeStepperOptions options;

    void SetUp() override
    {
        options.coupled_operator_backend = std::get<0>(GetParam());
        options.coupled_workspace_policy = std::get<1>(GetParam());
        options.time_step = 0.04;
        options.kinematic_viscosity = 0.025;
        for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        {
            boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
            boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        }
        boundaries.velocity["ymax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, {0.3, 0.0, 0.0}};
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const auto gid = mesh->owned_cell_map()->getGlobalElement(cell);
            velocity.set_owned_value(cell, {0.03 * (gid + 1), 0.02 * (gid + 2), -0.01 * (gid + 3)});
            pressure.set_owned_value(cell, 0.04 * (gid + 1));
        }
        velocity.sync_ghosts();
        pressure.sync_ghosts();
    }

    NativeSolver::system_type assemble(NativeSolver& solver, const Equation& equation, AssemblyPurpose purpose)
    {
        const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        std::vector<double> targets(mesh->num_owned_cells());
        for (size_t owned = 0; owned < targets.size(); ++owned)
        {
            const auto gid = mesh->owned_cell_map()->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            targets[owned] = gid % 2 == 0 ? 0.01 : -0.01;
        }
        const NativeSolver::continuity_target_type target(mesh, targets, 17);
        return solver.assemble(
            equation, velocity, pressure, flux, cache, boundaries, options, target, 2.0, nullptr, purpose);
    }

    Vector direction(const NativeSolver::system_type& system) const
    {
        Vector result(system.map);
        const auto entries = result.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t row = 0; row < entries.extent(0); ++row)
        {
            const auto gid = system.map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
            entries(row, 0) = 0.007 * (gid + 1) * (gid + 2);
        }
        return result;
    }

    void fill_pressure_direction(Vector& direction) const
    {
        const auto entries = direction.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t row = 0; row < entries.extent(0); ++row)
        {
            const auto gid = direction.getMap()->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
            entries(row, 0) = 0.031 * (gid + 1) * (gid + 2);
        }
    }

    void expect_same_affine_system(
        const NativeSolver::system_type& actual, const NativeSolver::system_type& expected) const
    {
        ASSERT_FALSE(actual.linear_operator.is_null());
        EXPECT_EQ(actual.pressure_gauge_gid, expected.pressure_gauge_gid);
        EXPECT_EQ(actual.continuity_target_generation, expected.continuity_target_generation);
        EXPECT_DOUBLE_EQ(actual.reference_density, expected.reference_density);
        const auto x = direction(actual);
        Vector actual_action(actual.map), expected_action(expected.map);
        actual.linear_operator->apply(x, actual_action);
        expected.linear_operator->apply(x, expected_action);
        expect_vector_near(actual_action, expected_action);
        expect_vector_near(*actual.rhs, *expected.rhs);
        actual_action.update(-1.0, *actual.rhs, 1.0);
        expected_action.update(-1.0, *expected.rhs, 1.0);
        expect_vector_near(actual_action, expected_action);
        if (actual.pressure_gauge_gid)
        {
            const auto gauge = actual.map->getLocalElement(4 * *actual.pressure_gauge_gid + 3);
            if (gauge != Teuchos::OrdinalTraits<Pack::local_ordinal_type>::invalid())
                EXPECT_DOUBLE_EQ(actual_action.getData()[gauge], x.getData()[gauge]);
        }
    }
};

TEST_P(CoupledResidualAssemblyTest, MatchesFullAffineOperatorWithGaugeAndPressureOutlet)
{
    for (const bool pressure_outlet : {false, true})
    {
        if (pressure_outlet)
        {
            boundaries.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Neumann, {}};
            boundaries.pressure["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.15};
        }
        NativeSolver residual_solver(mesh), full_solver(mesh);
        const Equation residual_equation(mesh), full_equation(mesh);
        const auto residual = assemble(residual_solver, residual_equation, AssemblyPurpose::ResidualOnly);
        const auto full = assemble(full_solver, full_equation, AssemblyPurpose::LinearSolve);
        EXPECT_EQ(residual.pressure_gauge_gid.has_value(), !pressure_outlet);
        EXPECT_TRUE(residual.schur.is_null());
        ASSERT_FALSE(full.schur.is_null());
        expect_same_affine_system(residual, full);
        EXPECT_EQ(residual_solver.cache_statistics().schur_builds, 0U);
        EXPECT_EQ(residual_solver.cache_statistics().schur_product_builds, 0U);
        EXPECT_EQ(residual_solver.cache_statistics().preconditioner_builds, 0U);
        EXPECT_EQ(full_solver.cache_statistics().schur_builds, 1U);
        EXPECT_EQ(full_solver.cache_statistics().schur_product_builds, 3U);
        EXPECT_THROW(residual_solver.right_preconditioner(residual), std::invalid_argument);
    }
}

TEST_P(CoupledResidualAssemblyTest, RefreshesSchurAfterSwitchingAssemblyPurpose)
{
    NativeSolver solver(mesh);
    const Equation equation(mesh);
    size_t expected_schur_builds = 0;
    for (const auto purpose : {AssemblyPurpose::ResidualOnly, AssemblyPurpose::LinearSolve,
             AssemblyPurpose::ResidualOnly, AssemblyPurpose::LinearSolve})
    {
        options.time_step += 0.03;
        const auto system = assemble(solver, equation, purpose);
        NativeSolver fresh_solver(mesh);
        const Equation fresh_equation(mesh);
        const auto fresh = assemble(fresh_solver, fresh_equation, AssemblyPurpose::LinearSolve);
        expect_same_affine_system(system, fresh);
        if (purpose == AssemblyPurpose::LinearSolve)
        {
            ++expected_schur_builds;
            ASSERT_FALSE(system.schur.is_null());
            Vector x(mesh->owned_cell_map()), actual(x.getMap()), expected(x.getMap());
            fill_pressure_direction(x);
            system.schur->apply(x, actual);
            fresh.schur->apply(x, expected);
            expect_vector_near(actual, expected);
        }
        else
            EXPECT_TRUE(system.schur.is_null());
        EXPECT_EQ(solver.cache_statistics().schur_builds, expected_schur_builds);
        EXPECT_EQ(solver.cache_statistics().schur_product_builds, 3 * expected_schur_builds);
    }
    EXPECT_EQ(solver.cache_statistics().matrix_graph_reuses, 3U);
}

TEST_P(CoupledResidualAssemblyTest, PreservesRetainedGenerationsAcrossAssemblyPurposes)
{
    NativeSolver solver(mesh);
    const Equation equation(mesh);
    const auto full = assemble(solver, equation, AssemblyPurpose::LinearSolve);
    const auto x = direction(full);
    Vector full_action(full.map), full_after(full.map);
    full.linear_operator->apply(x, full_action);
    Vector pressure_direction(mesh->owned_cell_map()), schur_action(mesh->owned_cell_map()),
        schur_after(mesh->owned_cell_map());
    fill_pressure_direction(pressure_direction);
    full.schur->apply(pressure_direction, schur_action);

    options.time_step = 0.11;
    const auto residual = assemble(solver, equation, AssemblyPurpose::ResidualOnly);
    Vector residual_action(residual.map), residual_after(residual.map);
    residual.linear_operator->apply(x, residual_action);
    options.time_step = 0.19;
    const auto next_full = assemble(solver, equation, AssemblyPurpose::LinearSolve);
    ASSERT_FALSE(next_full.schur.is_null());
    full.linear_operator->apply(x, full_after);
    full.schur->apply(pressure_direction, schur_after);
    residual.linear_operator->apply(x, residual_after);
    expect_vector_near(full_after, full_action);
    expect_vector_near(schur_after, schur_action);
    expect_vector_near(residual_after, residual_action);
    EXPECT_EQ(solver.cache_statistics().schur_builds, 2U);
    EXPECT_EQ(solver.cache_statistics().schur_product_builds, 6U);
}

TEST_P(CoupledResidualAssemblyTest, RejectsInvalidOrRankDivergentPurposeBeforeNumericMutation)
{
    NativeSolver solver(mesh);
    const Equation equation(mesh);
    const auto system = assemble(solver, equation, AssemblyPurpose::ResidualOnly);
    const auto x = direction(system);
    Vector before(system.map), after(system.map);
    system.linear_operator->apply(x, before);
    options.time_step = 0.19;
    EXPECT_THROW(assemble(solver, equation, static_cast<AssemblyPurpose>(255)), std::invalid_argument);
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        const auto purpose = comm->getRank() == 0 ? AssemblyPurpose::LinearSolve : AssemblyPurpose::ResidualOnly;
        EXPECT_THROW(assemble(solver, equation, purpose), std::invalid_argument);
    }
    system.linear_operator->apply(x, after);
    expect_vector_near(after, before);
    EXPECT_EQ(solver.cache_statistics().schur_builds, 0U);
}

INSTANTIATE_TEST_SUITE_P(Native, CoupledResidualAssemblyTest,
    testing::Combine(testing::Values(Backend::Assembled, Backend::BlockComposite),
        testing::Values(WorkspacePolicy::CachedProducts, WorkspacePolicy::StreamedProducts)));
} // namespace
