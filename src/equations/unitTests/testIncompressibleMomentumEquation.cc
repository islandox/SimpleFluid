/**
 * @file testIncompressibleMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for generic incompressible momentum transport.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoundaryConditions.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <stdexcept>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

SimpleFluid::SP<MeshType> make_2x2x2_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

Pack::scalar_type local_matrix_entry(
    const Pack::matrix_type& matrix,
    MeshType::local_ordinal_type row,
    MeshType::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
        "columns", row_entries);
    typename Pack::matrix_type::nonconst_values_host_view_type values(
        "values", row_entries);
    size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    Pack::scalar_type entry = 0.0;
    for (size_t i = 0; i < num_entries; ++i)
    {
        if (columns(i) == column)
        {
            entry += values(i);
        }
    }
    return entry;
}

} // namespace

/**
 * @brief Verify that a constant body-force source advances velocity
 *        linearly on a single-hex mesh with zero viscosity.
 */
TEST(IncompressibleMomentumEquationTest, AdvancesVelocityFromSource)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(mesh, "velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;

    auto source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {1.0, 2.0, 3.0};
    };

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity(
        velocity, zero_fluxes, boundary_cache, options, velocity, source);

    EXPECT_NEAR(velocity.value(0).x, 0.1, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 0.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 0.3, 1.0e-12);
}

/**
 * @brief Verify the physical momentum advance with explicit density,
 *        dynamic viscosity, and acceleration on a single-hex mesh.
 */
TEST(IncompressibleMomentumEquationTest, AdvancesPhysicalMomentum)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType dynamic_viscosity(mesh, 0.0, "dynamic_viscosity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.2;

    auto acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {-2.0, 1.0, 0.5};
    };

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity_physical(
        velocity, zero_fluxes, boundary_cache, options, dynamic_viscosity,
        10.0, velocity, acceleration);

    EXPECT_NEAR(velocity.value(0).x, -0.4, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 0.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 0.1, 1.0e-12);
}

/**
 * @brief An accepted physical velocity that already solves the new system is
 *        passed to Belos as the initial guess.
 */
TEST(IncompressibleMomentumEquationTest,
     WarmStartsPhysicalMomentumFromAcceptedVelocity)
{
    auto mesh = make_single_hex_mesh();
    const VectorFieldType::vec_type accepted{1.0, 2.0, 3.0};
    VectorFieldType old_velocity(mesh, accepted, "old_velocity");
    VectorFieldType velocity(
        mesh, VectorFieldType::vec_type{9.0, 8.0, 7.0}, "velocity");
    FieldType dynamic_viscosity(mesh, 0.0, "dynamic_viscosity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.2;
    auto zero_acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto summary = equation.advance_velocity_physical(
        old_velocity, zero_fluxes, boundary_cache, options,
        dynamic_viscosity, 10.0, velocity, zero_acceleration,
        linear_options);

    EXPECT_TRUE(summary.converged);
    EXPECT_EQ(summary.iterations, 0);
    EXPECT_DOUBLE_EQ(velocity.value(0).x, accepted.x);
    EXPECT_DOUBLE_EQ(velocity.value(0).y, accepted.y);
    EXPECT_DOUBLE_EQ(velocity.value(0).z, accepted.z);
}

/**
 * @brief A zero RHS column remains solvable when its warm start is nonzero,
 *        alongside nonzero RHS columns in the same pseudo-block solve.
 */
TEST(IncompressibleMomentumEquationTest,
     WarmStartHandlesMixedZeroAndNonzeroMomentumRhsColumns)
{
    auto mesh = make_single_hex_mesh();
    const VectorFieldType::vec_type accepted{1.0, 2.0, 3.0};
    VectorFieldType velocity(mesh, accepted, "velocity");
    FieldType dynamic_viscosity(mesh, 0.0, "dynamic_viscosity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.2;
    auto acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        // Cancels the transient RHS in X, advances Y, and leaves Z at its
        // exact warm start.
        return {-5.0, 1.0, 0.0};
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 5;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto summary = equation.advance_velocity_physical(
        velocity, zero_fluxes, boundary_cache, options,
        dynamic_viscosity, 10.0, velocity, acceleration,
        linear_options);

    EXPECT_TRUE(summary.converged);
    EXPECT_NEAR(velocity.value(0).x, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 2.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 3.0, 1.0e-12);
}

/**
 * @brief A rejected physical momentum solve leaves aliased accepted velocity
 *        unchanged.
 */
TEST(IncompressibleMomentumEquationTest,
     PhysicalSolveRejectionPreservesAliasedAcceptedVelocity)
{
    auto mesh = make_2x2x2_mesh();
    VectorFieldType velocity(mesh, "accepted_velocity");
    std::vector<VectorFieldType::vec_type> accepted(
        mesh->num_local_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto index = static_cast<double>(owned + 1);
        velocity.set_owned_value(
            cell_lid, {1.0 + index * index,
                       2.0 + index * index * index,
                       3.0 + index * index * index * index});
    }
    velocity.sync_ghosts();
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        accepted[local] = velocity.local_value(
            static_cast<MeshType::local_ordinal_type>(local));
    }

    FieldType dynamic_viscosity(mesh, 1.0, "dynamic_viscosity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0;
    auto zero_acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-14;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    EXPECT_THROW(
        equation.advance_velocity_physical(
            velocity, zero_fluxes, boundary_cache, options,
            dynamic_viscosity, 1.0, velocity, zero_acceleration,
            linear_options),
        std::runtime_error);

    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto actual = velocity.local_value(
            static_cast<MeshType::local_ordinal_type>(local));
        EXPECT_DOUBLE_EQ(actual.x, accepted[local].x);
        EXPECT_DOUBLE_EQ(actual.y, accepted[local].y);
        EXPECT_DOUBLE_EQ(actual.z, accepted[local].z);
    }
}

/** @brief Verifies that slip boundaries add no diffusive momentum diagonal. */
TEST(IncompressibleMomentumEquationTest,
     SlipBoundariesDoNotAddDiffusiveMomentumDiagonal)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(
        mesh,
        VectorFieldType::vec_type{1.0, 2.0, 3.0},
        "slip_velocity");
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundary_conditions.velocity[name] =
            {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.25;
    options.kinematic_viscosity = 10.0;
    const auto transient = mesh->cell_volume(0) / options.time_step;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto constant_viscosity_system = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options);
    EXPECT_NEAR(
        local_matrix_entry(*constant_viscosity_system.matrix, 0, 0),
        transient,
        1.0e-12);

    FieldType dynamic_viscosity(mesh, 10.0, "dynamic_viscosity");
    auto zero_acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    const auto physical_viscosity_system =
        equation.assemble_physical_system(
            velocity, zero_fluxes, boundary_cache, options,
            dynamic_viscosity, 1.0, zero_acceleration);
    EXPECT_NEAR(
        local_matrix_entry(*physical_viscosity_system.matrix, 0, 0),
        transient,
        1.0e-12);
}

/**
 * @brief Explicit-to-hybrid transition rebuilds the graph, then reuses it
 *        without changing assembled values.
 */
TEST(IncompressibleMomentumEquationTest,
     RebuildsAndReusesNonOrthogonalAssemblyGraph)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    VectorFieldType velocity(mesh, "cached_momentum_velocity");
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_value(
            cell_lid,
            {0.5 + center.x * center.y,
             -0.25 + center.y * center.z,
             center.x - 0.5 * center.z});
    }
    velocity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        (void)batch;
        boundary_conditions.velocity[mesh->boundary_batch_name(batch_id)] =
            {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.2;
    options.kinematic_viscosity = 0.7;
    options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto orthogonal = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, &velocity);
    const auto* const orthogonal_storage = orthogonal.matrix.get();

    options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid;
    const auto first = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, &velocity);
    EXPECT_NE(first.matrix.get(), orthogonal_storage);
    std::vector<std::array<Pack::scalar_type, 3>> first_rhs(
        mesh->num_owned_cells());
    for (size_t component = 0; component < 3; ++component)
    {
        const auto values = first.rhs->getData(component);
        for (size_t row = 0; row < first_rhs.size(); ++row)
        {
            first_rhs[row][component] = values[row];
        }
    }
    std::vector<Pack::scalar_type> first_matrix(
        mesh->num_owned_cells() * mesh->num_local_cells());
    for (MeshType::local_ordinal_type row = 0;
         row < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++row)
    {
        for (MeshType::local_ordinal_type column = 0;
             column < static_cast<MeshType::local_ordinal_type>(
                 mesh->num_local_cells());
             ++column)
        {
            first_matrix[
                static_cast<size_t>(row) * mesh->num_local_cells()
                + static_cast<size_t>(column)] =
                    local_matrix_entry(*first.matrix, row, column);
        }
    }
    const auto* const matrix_storage = first.matrix.get();

    const auto second = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, &velocity);
    EXPECT_EQ(second.matrix.get(), matrix_storage);
    for (MeshType::local_ordinal_type row = 0;
         row < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++row)
    {
        for (size_t component = 0; component < 3; ++component)
        {
            EXPECT_NEAR(
                second.rhs->getData(component)[row],
                first_rhs[static_cast<size_t>(row)][component],
                1.0e-12);
        }
        for (MeshType::local_ordinal_type column = 0;
             column < static_cast<MeshType::local_ordinal_type>(
                 mesh->num_local_cells());
             ++column)
        {
            EXPECT_NEAR(
                local_matrix_entry(*second.matrix, row, column),
                first_matrix[
                    static_cast<size_t>(row) * mesh->num_local_cells()
                    + static_cast<size_t>(column)],
                1.0e-12);
        }
    }

    // A zero-diffusivity implicit assembly has only the compact graph.  A
    // later positive coefficient must therefore rebuild before it can cache
    // and reuse the expanded non-orthogonal graph.
    SimpleFluid::IncompressibleMomentumEquation<Pack>
        coefficient_transition(mesh);
    options.kinematic_viscosity = 0.0;
    const auto zero_diffusion = coefficient_transition.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, &velocity);
    options.kinematic_viscosity = 0.7;
    const auto positive_diffusion =
        coefficient_transition.assemble_system(
            velocity, zero_fluxes, boundary_cache, options, &velocity);
    EXPECT_NE(
        positive_diffusion.matrix.get(),
        zero_diffusion.matrix.get());
}

/**
 * @brief A failed cached assembly is discarded so the following call can
 *        rebuild a complete transport matrix.
 */
TEST(IncompressibleMomentumEquationTest,
     RecoversAfterCachedTransportAssemblyThrows)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(mesh, "recovery_velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.5;
    options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit;

    auto zero_source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    auto throwing_source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        throw std::runtime_error("intentional assembly failure");
    };

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto first = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, zero_source);
    ASSERT_TRUE(first.matrix->isFillComplete());
    const auto* const first_storage = first.matrix.get();

    EXPECT_THROW(
        equation.assemble_system(
            velocity, zero_fluxes, boundary_cache, options,
            throwing_source),
        std::runtime_error);

    const auto third = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options, zero_source);
    EXPECT_TRUE(third.matrix->isFillComplete());
    EXPECT_NE(third.matrix.get(), first_storage);
}
