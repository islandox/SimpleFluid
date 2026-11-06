/**
 * @file testPressureProjectionMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Distributed pressure-projection gauge and residual tests.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/PressureProjectionEquation.hh"
#include "equations/PressureProjectionOperator.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/Operators.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::detail
{

template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess
{
    static Teuchos::RCP<const typename Pack::matrix_type> pressure_matrix(
        const PressureProjectionEquation<Pack>& equation)
    {
        return equation.d_cached_pressure_matrix;
    }

    static auto reuse_predictor(PressureProjectionEquation<Pack>& equation,
        CellField<Pack>& pressure, CellField<Pack>& correction,
        typename Pack::scalar_type time_step, typename Pack::scalar_type density,
        const FVM::VelocityBoundaryCache<Pack>& boundaries, VectorCellField<Pack>& velocity)
    {
        return equation.project_reusing_cached_predictor(
            pressure, correction, time_step, density, boundaries, velocity);
    }
};

} // namespace SimpleFluid::detail

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
using global_ordinal_type = Pack::global_ordinal_type;
using scalar_type = Pack::scalar_type;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

template<class T>
T global_sum(const Teuchos::Comm<int>& comm, T local_value)
{
    T global_value{};
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_SUM, 1, &local_value, &global_value);
    return global_value;
}

template<class T>
T global_min(const Teuchos::Comm<int>& comm, T local_value)
{
    T global_value{};
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_MIN, 1, &local_value, &global_value);
    return global_value;
}

void expect_replicated(const Teuchos::Comm<int>& comm, scalar_type value)
{
    scalar_type minimum{};
    scalar_type maximum{};
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_MIN, 1, &value, &minimum);
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_MAX, 1, &value, &maximum);

    EXPECT_NEAR(
        minimum,
        maximum,
        1.0e-12 * std::max<scalar_type>(1.0, std::abs(maximum)));
}

} // namespace

/** @brief CG removes the gauge column even on its neighboring MPI rank. */
TEST(PressureProjectionMultiRankTest, CgEliminatesRemoteGaugeColumnAndMatchesGmres)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1, 0.5));
    const auto row_map = mesh->owned_cell_map();
    const auto comm = row_map->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }
    ASSERT_EQ(global_min(*comm, static_cast<int>(mesh->num_owned_cells())), 1);
    const auto gauge = row_map->getMinAllGlobalIndex();
    const auto row_gid = row_map->getGlobalElement(0);
    const std::vector<scalar_type> rates{row_gid == gauge ? scalar_type{0.125} : scalar_type{-0.125}};
    const SimpleFluid::VolumeContinuityTarget<Pack> target(mesh, rates, 1);
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    FieldType gmres_pressure(mesh, "gmres_pressure");
    VectorFieldType gmres_velocity(mesh, SimpleFluid::vec3{}, "gmres_velocity");
    const auto gmres = equation.project(gmres_pressure, 0.1, 1.0, cache, gmres_velocity, target);
    ASSERT_TRUE(gmres.linear_solve.converged);

    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    equation.set_linear_solver_options(options);
    FieldType cg_pressure(mesh, "cg_pressure");
    VectorFieldType cg_velocity(mesh, SimpleFluid::vec3{}, "cg_velocity");
    const auto cg = equation.project(cg_pressure, 0.1, 1.0, cache, cg_velocity, target);
    ASSERT_TRUE(cg.linear_solve.converged);
    EXPECT_NEAR(cg_pressure.value(0), gmres_pressure.value(0), 1.0e-12);
    EXPECT_NEAR(cg.continuity, gmres.continuity, 1.0e-12);
    EXPECT_NEAR(cg.continuity, 0.0, 1.0e-12);

    Pack::vector_type unit_gauge(row_map, true);
    Pack::vector_type gauge_column(row_map, true);
    unit_gauge.replaceLocalValue(0, row_gid == gauge ? scalar_type{1} : scalar_type{});
    const auto matrix = SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::pressure_matrix(equation);
    matrix->apply(unit_gauge, gauge_column);
    gauge_column.update(-1.0, unit_gauge, 1.0);
    EXPECT_NEAR(gauge_column.norm2(), 0.0, 1.0e-14);
}

/**
 * @brief A two-rank projection uses one global gauge and reports global
 *        pressure and continuity norms on every rank.
 *
 * The only source is placed on the minimum owned Tpetra row of the rank
 * that does not own the global-minimum row.  A per-rank gauge therefore
 * erases the source, while a single global gauge produces a nonzero solve.
 */
TEST(PressureProjectionMultiRankTest,
     UsesSingleGlobalGaugeAndReportsGlobalResidualNorms)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto row_map = mesh->owned_cell_map();
    const auto comm = row_map->getComm();

    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    const int every_rank_has_owned_cells = global_min(
        *comm, mesh->num_owned_cells() > 0 ? 1 : 0);
    ASSERT_EQ(every_rank_has_owned_cells, 1);

    const auto gauge_row_gid = row_map->getMinAllGlobalIndex();
    const bool owns_gauge = row_map->isNodeGlobalElement(gauge_row_gid);
    const auto no_source_candidate =
        std::numeric_limits<global_ordinal_type>::max();
    const auto local_source_row_gid = owns_gauge
                                    ? no_source_candidate
                                    : row_map->getMinGlobalIndex();
    const auto source_row_gid = global_min(*comm, local_source_row_gid);

    ASSERT_NE(source_row_gid, no_source_candidate);
    ASSERT_NE(source_row_gid, gauge_row_gid);

    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(
        mesh, SimpleFluid::vec3{}, "velocity");
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 400;
    linear_options.tolerance = 1.0e-12;
    linear_options.preconditioner =
        SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> equation(
        mesh, linear_options);

    auto source =
        [&](MeshType::local_ordinal_type cell_lid) -> scalar_type
    {
        const auto row_gid = row_map->getGlobalElement(cell_lid);
        return row_gid == source_row_gid
             ? scalar_type{1} / mesh->cell_volume(cell_lid)
             : scalar_type{};
    };

    constexpr scalar_type time_step = 1.0;
    const auto result = equation.project(
        pressure,
        time_step,
        scalar_type{1},
        boundary_cache,
        velocity,
        source);

    EXPECT_TRUE(result.linear_solve.converged);

    scalar_type local_pressure_norm_squared{};
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto value = pressure.value(cell_lid);
        local_pressure_norm_squared +=
            value * value * mesh->cell_volume(cell_lid);

        if (row_map->getGlobalElement(cell_lid) == gauge_row_gid)
        {
            EXPECT_NEAR(value, 0.0, 1.0e-11);
        }
    }
    const auto global_pressure_norm_squared =
        global_sum(*comm, local_pressure_norm_squared);
    const auto expected_pressure_norm =
        std::sqrt(global_pressure_norm_squared);

    EXPECT_GT(expected_pressure_norm, 1.0e-6);
    EXPECT_NEAR(
        result.pressure_correction,
        expected_pressure_norm,
        1.0e-11 * std::max<scalar_type>(1.0, expected_pressure_norm));
    expect_replicated(*comm, result.pressure_correction);

    SimpleFluid::FaceField<Pack> face_fluxes(
        mesh, "independent_projected_face_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        time_step,
        boundary_cache,
        boundary_conditions.pressure,
        face_fluxes);

    scalar_type local_continuity_norm_squared{};
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto balance = SimpleFluid::FVM::cell_flux_balance<Pack>(
            *mesh, face_fluxes, cell_lid);
        // The legacy source overload is converted to the same integrated
        // target contract: Q = dt*V*S, which is one on source_row_gid.
        const auto target =
            row_map->getGlobalElement(cell_lid) == source_row_gid
                ? scalar_type{1}
                : scalar_type{};
        const auto residual = balance - target;
        local_continuity_norm_squared += residual * residual;
    }
    const auto global_continuity_norm_squared =
        global_sum(*comm, local_continuity_norm_squared);
    const auto expected_continuity_norm =
        std::sqrt(global_continuity_norm_squared);

    EXPECT_GT(expected_continuity_norm, 1.0e-6);
    EXPECT_NEAR(
        result.continuity,
        expected_continuity_norm,
        1.0e-11 * std::max<scalar_type>(1.0, expected_continuity_norm));
    expect_replicated(*comm, result.continuity);
}

/** @brief Two ranks satisfy one conservative integrated-volume target. */
TEST(PressureProjectionMultiRankTest,
     SatisfiesDistributedIntegratedVolumeTargetAndReportsGlobalNorms)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto row_map = mesh->owned_cell_map();
    const auto comm = row_map->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    const auto first_gid = row_map->getMinAllGlobalIndex();
    const auto last_gid = row_map->getMaxAllGlobalIndex();
    std::vector<scalar_type> target_values(mesh->num_owned_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto gid = row_map->getGlobalElement(cell_lid);
        target_values[owned] = gid == first_gid
            ? scalar_type{0.125}
            : gid == last_gid ? scalar_type{-0.125} : scalar_type{};
    }

    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 400;
    linear_options.tolerance = 1.0e-12;
    linear_options.preconditioner =
        SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> equation(
        mesh, linear_options);
    const SimpleFluid::VolumeContinuityTarget<Pack> target(
        mesh, target_values, 23);

    const auto result = equation.project(
        pressure, scalar_type{0.1}, scalar_type{1}, boundary_cache,
        velocity, target);

    scalar_type local_norm_squared{};
    scalar_type local_scale_squared{};
    scalar_type local_maximum{};
    const auto& fluxes = equation.corrected_face_fluxes();
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto balance = SimpleFluid::FVM::cell_flux_balance<Pack>(
            *mesh, fluxes, cell_lid);
        const auto residual = balance - target_values[owned];
        local_norm_squared += residual * residual;
        local_scale_squared += std::max(
            balance * balance,
            target_values[owned] * target_values[owned]);
        local_maximum = std::max(local_maximum, std::abs(residual));
    }
    const auto expected_l2 =
        std::sqrt(global_sum(*comm, local_norm_squared));
    const auto expected_scale =
        std::sqrt(global_sum(*comm, local_scale_squared));
    scalar_type expected_maximum{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1,
        &local_maximum, &expected_maximum);

    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_NEAR(expected_l2, 0.0, 1.0e-10);
    EXPECT_NEAR(result.continuity, expected_l2, 1.0e-13);
    EXPECT_NEAR(result.continuity_residuals.l2, expected_l2, 1.0e-13);
    EXPECT_NEAR(result.continuity_residuals.maximum,
        expected_maximum, 1.0e-13);
    EXPECT_NEAR(result.continuity_residuals.normalization,
        expected_scale, 1.0e-13);
    EXPECT_NEAR(result.continuity_residuals.normalized_l2,
        expected_l2 / expected_scale, 1.0e-13);
    expect_replicated(*comm, result.continuity_residuals.l2);
    expect_replicated(*comm, result.continuity_residuals.maximum);
}

/** @brief Target generation disagreement is rejected collectively. */
TEST(PressureProjectionMultiRankTest,
     EveryRankRejectsDivergentContinuityTargetGeneration)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh);
    const SimpleFluid::VolumeContinuityTarget<Pack> target(
        mesh,
        std::vector<scalar_type>(mesh->num_owned_cells(), scalar_type{}),
        static_cast<std::uint64_t>(comm->getRank()));

    EXPECT_THROW(
        equation.project(pressure, scalar_type{0.1}, scalar_type{1},
            boundary_cache, velocity, target),
        std::invalid_argument);
}

/** @brief An exact fixed outlet flux remains invariant on two ranks. */
TEST(PressureProjectionMultiRankTest,
     PreservesDistributedFixedBoundaryFluxAgainstMatchingTarget)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto row_map = mesh->owned_cell_map();
    const auto comm = row_map->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    constexpr scalar_type fixed_flux = 0.125;
    const auto last_gid = row_map->getMaxAllGlobalIndex();
    std::vector<scalar_type> target_values(mesh->num_owned_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        if (row_map->getGlobalElement(cell_lid) == last_gid)
        {
            target_values[owned] = fixed_flux;
        }
    }

    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundary_conditions.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        SimpleFluid::vec3{9.0, 0.0, 0.0}};
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 400;
    linear_options.tolerance = 1.0e-12;
    linear_options.preconditioner =
        SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> equation(
        mesh, linear_options, boundary_conditions.pressure);
    equation.set_fixed_boundary_flux_provider(
        {"xmax"},
        [](int, size_t, MeshType::local_ordinal_type)
        {
            return scalar_type{0.125};
        },
        29);
    const SimpleFluid::VolumeContinuityTarget<Pack> target(
        mesh, target_values, 29);

    const auto result = equation.project(
        pressure, scalar_type{0.1}, scalar_type{1}, boundary_cache,
        velocity, target);

    int local_fixed_faces = 0;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "xmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (mesh->is_owned_face(face_lid))
            {
                EXPECT_DOUBLE_EQ(
                    equation.corrected_face_fluxes().value(face_lid),
                    fixed_flux);
                ++local_fixed_faces;
            }
        }
    }
    EXPECT_EQ(global_sum(*comm, local_fixed_faces), 1);
    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_NEAR(result.continuity_residuals.l2, 0.0, 1.0e-10);
    EXPECT_NEAR(result.continuity_residuals.maximum, 0.0, 1.0e-10);
    expect_replicated(*comm, result.continuity_residuals.l2);
}

/** @brief Verifies coherent rejection of incompatible pressure-velocity boundaries on all ranks. */
TEST(PressureProjectionMultiRankTest,
     EveryRankRejectsIncompatiblePressureVelocityBoundaryPair)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    boundary_conditions.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        {1.0, 0.0, 0.0}};
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");

    int local_rejected = 0;
    try
    {
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure,
            scalar_type{1},
            boundary_cache,
            boundary_conditions.pressure,
            face_fluxes);
    }
    catch (const std::invalid_argument&)
    {
        local_rejected = 1;
    }

    EXPECT_EQ(global_min(*comm, local_rejected), 1);
}

/** @brief PCG/DIC retains cross-rank pressure coupling after gauge elimination. */
TEST(PressureProjectionMultiRankTest,
     ParallelDicMatchesGmresPressureAndConservativeCorrectedFlux)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 2, 1, 0.25));
    const auto row_map = mesh->owned_cell_map();
    const auto comm = row_map->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }
    ASSERT_GT(global_min(*comm, static_cast<int>(mesh->num_owned_cells())), 1);
    const auto gauge = row_map->getMinAllGlobalIndex();
    const auto last_gid = row_map->getMaxAllGlobalIndex();
    std::vector<scalar_type> rates(mesh->num_owned_cells(), scalar_type{});
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto gid = row_map->getGlobalElement(
            static_cast<Pack::local_ordinal_type>(owned));
        rates[owned] = gid == gauge ? scalar_type{0.125}
            : gid == last_gid ? scalar_type{-0.125} : scalar_type{};
    }
    const SimpleFluid::VolumeContinuityTarget<Pack> target(mesh, rates, 31);
    const SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.max_iterations = 100;
    options.preconditioner = SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> gmres_equation(mesh, options);
    FieldType gmres_pressure(mesh, "distributed_gmres_pressure");
    VectorFieldType gmres_velocity(mesh, SimpleFluid::vec3{}, "distributed_gmres_velocity");
    const auto gmres = gmres_equation.project(
        gmres_pressure, 0.1, 1.0, boundary_cache, gmres_velocity, target);
    ASSERT_TRUE(gmres.linear_solve.converged);

    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::DIC;
    SimpleFluid::PressureProjectionEquation<Pack> dic_equation(mesh, options);
    FieldType dic_pressure(mesh, "distributed_dic_pressure");
    VectorFieldType dic_velocity(mesh, SimpleFluid::vec3{}, "distributed_dic_velocity");
    const auto dic = dic_equation.project(
        dic_pressure, 0.1, 1.0, boundary_cache, dic_velocity, target);
    ASSERT_TRUE(dic.linear_solve.converged);
    EXPECT_LE(dic.linear_solve.achieved_tolerance, options.tolerance);
    EXPECT_GT(dic.pressure_correction, 1.0e-6);
    EXPECT_NEAR(dic.pressure_correction, gmres.pressure_correction, 1.0e-11);
    EXPECT_NEAR(dic.continuity, 0.0, 1.0e-11);
    EXPECT_NEAR(dic.continuity, gmres.continuity, 1.0e-11);

    const auto matrix =
        SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::pressure_matrix(dic_equation);
    int local_remote_entries = 0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto row = static_cast<Pack::local_ordinal_type>(owned);
        const auto row_gid = row_map->getGlobalElement(row);
        EXPECT_NEAR(dic_pressure.value(row), gmres_pressure.value(row), 1.0e-10);
        const auto velocity_difference = dic_velocity.value(row) - gmres_velocity.value(row);
        EXPECT_NEAR(velocity_difference.dot(velocity_difference), 0.0, 1.0e-20);
        const auto balance = SimpleFluid::FVM::cell_flux_balance<Pack>(
            *mesh, dic_equation.corrected_face_fluxes(), row);
        EXPECT_NEAR(balance, rates[owned], 1.0e-11);

        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type values;
        matrix->getLocalRowView(row, columns, values);
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            const auto column_gid = matrix->getColMap()->getGlobalElement(columns[entry]);
            if (column_gid == row_gid)
                EXPECT_GT(values[entry], 0.0);
            if (column_gid == gauge)
                EXPECT_DOUBLE_EQ(values[entry], row_gid == gauge ? 1.0 : 0.0);
            if (!row_map->isNodeGlobalElement(column_gid) && values[entry] != 0.0)
                ++local_remote_entries;
        }
    }
    // A two-cell fixture loses its only partition edge when the gauge is
    // removed. This physical matrix must retain nonzero remote dependencies.
    EXPECT_GT(global_sum(*comm, local_remote_entries), 0);
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (mesh->is_owned_face(face_lid))
            EXPECT_NEAR(dic_equation.corrected_face_fluxes().value(face_lid),
                        gmres_equation.corrected_face_fluxes().value(face_lid), 1.0e-11);
    }
}


/** @brief A nontrivial skew projection solves the operator used by final fluxes. */
TEST(PressureProjectionMultiRankTest, SkewProjectionMatchesReconstructedFluxAndContinuity)
{
    if (Tpetra::getDefaultComm()->getSize()!=1)
        GTEST_SKIP() << "Serial manufactured skew projection.";
    auto mesh=SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    for (const auto method : {SimpleFluid::PressureProjectionMethod::Anderson,
             SimpleFluid::PressureProjectionMethod::FullOperator})
    {
        SCOPED_TRACE(static_cast<int>(method));
        FieldType pressure(mesh,"skew_projection_pressure");
        VectorFieldType velocity(mesh,"skew_projection_velocity");
        for (size_t owned=0; owned<mesh->num_owned_cells(); ++owned)
        {
            const auto cell=static_cast<MeshType::local_ordinal_type>(owned);
            const auto x=mesh->cell_centroid(cell);
            velocity.set_value(cell,{x.x*x.x+0.2*x.y,-0.3*x.y*x.y,0.1*x.z});
        }
        velocity.sync_ghosts();
        const SimpleFluid::BoundaryConditionSet conditions;
        const auto boundaries=SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh,conditions);
        SimpleFluid::FaceField<Pack> initial_flux(mesh,"skew_initial_flux");
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,pressure,0.1,boundaries,conditions.pressure,initial_flux);
        double initial_squared=0;
        for (size_t owned=0; owned<mesh->num_owned_cells(); ++owned)
        {
            const auto cell=static_cast<MeshType::local_ordinal_type>(owned);
            const auto balance=SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh,initial_flux,cell);
            initial_squared+=balance*balance;
        }
        ASSERT_GT(std::sqrt(initial_squared),1e-3);
        SimpleFluid::LinearSolverOptions options;
        options.tolerance=1e-11;
        options.max_iterations=500;
        options.preconditioner=SimpleFluid::LinearPreconditioner::None;
        if (method == SimpleFluid::PressureProjectionMethod::FullOperator)
        {
            options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
            options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
        }
        SimpleFluid::PressureProjectionEquation<Pack> equation(mesh,options);
        equation.set_projection_method(method);
        const auto result=equation.project(pressure,0.1,1.0,boundaries,velocity);
        ASSERT_TRUE(result.linear_solve.converged);
        EXPECT_GT(result.pressure_correction,1e-4);
        SimpleFluid::FaceField<Pack> reconstructed(mesh,"skew_independent_final_flux");
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,pressure,0.1,boundaries,conditions.pressure,reconstructed);
        double squared=0;
        double reconstructed_squared=0;
        for (size_t owned=0; owned<mesh->num_owned_cells(); ++owned)
        {
            const auto cell=static_cast<MeshType::local_ordinal_type>(owned);
            const auto balance=SimpleFluid::FVM::cell_flux_balance<Pack>(
                *mesh,equation.corrected_face_fluxes(),cell);
            const auto independent=SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh,reconstructed,cell);
            squared+=balance*balance;
            reconstructed_squared+=independent*independent;
        }
        EXPECT_LT(std::sqrt(squared),1e-8);
        EXPECT_LT(std::sqrt(reconstructed_squared),1e-8);
        EXPECT_NEAR(result.continuity,std::sqrt(squared),1e-13);
        for (size_t local=0; local<mesh->num_faces(); ++local)
        {
            const auto face=static_cast<MeshType::local_ordinal_type>(local);
            if (mesh->is_owned_face(face))
                EXPECT_NEAR(equation.corrected_face_fluxes().value(face),reconstructed.value(face),1e-12);
        }
    }
}


TEST(PressureProjectionMultiRankTest, NativeSkewOutletProjectionIsConservativeAcrossRanks)
{
    using namespace SimpleFluid;
    const auto comm=Tpetra::getDefaultComm();
    if (comm->getSize()!=2) GTEST_SKIP() << "Two-rank native skew outlet regression.";
    using Geometry=Meshes::UnstructuredMesh;
    using Partitioned=Meshes::PartitionedMesh<Geometry,Pack>;
    using Native=MeshHandle<Pack>;
    // A genuinely two-dimensional grid retains transverse pressure gradients.
    constexpr size_t nx=4, ny=3;
    const auto node=[](size_t i,size_t j,size_t k) {
        return static_cast<Geometry::NodeID>(i+(nx+1)*(j+(ny+1)*k));
    };
    Arr<Geometry::Vec3> nodes;
    for(size_t k=0;k<2;++k) for(size_t j=0;j<=ny;++j) for(size_t i=0;i<=nx;++i)
    {
        const double x=double(i)/nx,y=double(j)/ny;
        nodes.push_back({x+0.3*y,y,double(k)});
    }
    Arr<Geometry::CellDefinition> cells;
    Arr<Geometry::BoundaryFaceDefinition> exterior;
    for(size_t j=0;j<ny;++j) for(size_t i=0;i<nx;++i)
    {
        const auto a=node(i,j,0),b=node(i+1,j,0),c=node(i+1,j+1,0),d=node(i,j+1,0);
        const auto e=node(i,j,1),f=node(i+1,j,1),g=node(i+1,j+1,1),h=node(i,j+1,1);
        cells.push_back({Geometry::CellType::HEXAHEDRON,{a,b,c,d,e,f,g,h}});
        if(i==0) exterior.push_back({{d,h,e,a},1,"xmin"});
        if(i+1==nx) exterior.push_back({{b,f,g,c},2,"xmax"});
        if(j==0) exterior.push_back({{a,e,f,b},3,"ymin"});
        if(j+1==ny) exterior.push_back({{d,c,g,h},4,"ymax"});
        exterior.push_back({{a,b,c,d},5,"zmin"});
        exterior.push_back({{e,h,g,f},6,"zmax"});
    }
    auto geometry=std::make_shared<Geometry>(nodes,cells,exterior);
    auto partition=MeshPartitioner<Pack>::partition(*geometry,comm);
    auto distributed=std::make_shared<Partitioned>(geometry,std::move(partition.indexer),comm);
    SP<const Native> mesh=std::make_shared<Native>(distributed);
    ASSERT_GT(global_min(*comm,int(mesh->num_owned_cells())),0);
    ASSERT_FALSE(mesh->legacy_mesh());
    for (const auto method : {SimpleFluid::PressureProjectionMethod::Anderson,
             SimpleFluid::PressureProjectionMethod::FullOperator})
    {
        SCOPED_TRACE(static_cast<int>(method));
        ScalarCellFieldStored<Pack> pressure(mesh,"native_skew_pressure");
        VectorCellFieldStored<Pack> velocity(mesh,"native_skew_velocity");
        for(size_t owned=0;owned<mesh->num_owned_cells();++owned)
        {
            const auto cell=static_cast<Pack::local_ordinal_type>(owned);
            const auto x=mesh->cell_centroid(cell);
            velocity.set_owned_value(cell,{x.x*x.x+0.3*x.y,x.x*x.y-0.2*x.y*x.y,0.0});
        }
        velocity.sync_ghosts();
        BoundaryConditionSet conditions;
        conditions.velocity["xmax"]={BoundaryConditionType::Neumann,{}};
        conditions.pressure["xmax"]={BoundaryConditionType::Dirichlet,0.0};
        const auto cache=FVM::cache_velocity_boundary_conditions<Pack>(mesh,conditions);
        ScalarFaceFieldStored<Pack> initial(mesh,"native_skew_initial_flux");
        FVM::pressure_weighted_face_fluxes(velocity,pressure,0.1,cache,conditions.pressure,initial);
        double initial_squared=0;
        for(size_t owned=0;owned<mesh->num_owned_cells();++owned)
        {
            const auto cell=static_cast<Pack::local_ordinal_type>(owned);
            const auto q=FVM::cell_flux_balance<Pack>(*mesh,initial,cell);
            initial_squared+=q*q;
        }
        ASSERT_GT(std::sqrt(global_sum(*comm,initial_squared)),1e-3);
        LinearSolverOptions options;
        options.tolerance=1e-11; options.max_iterations=500;
        options.preconditioner=LinearPreconditioner::None;
        if (method == SimpleFluid::PressureProjectionMethod::FullOperator)
        {
            options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
            options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
        }
        PressureProjectionEquation<Pack,Native> equation(mesh,options,conditions.pressure);
        equation.set_projection_method(method);
        const auto result=equation.project(pressure,0.1,1.0,cache,velocity);
        ASSERT_TRUE(result.linear_solve.converged);
        EXPECT_GT(result.pressure_correction,1e-4);
        ScalarFaceFieldStored<Pack> independent(mesh,"native_skew_reconstructed_flux");
        FVM::pressure_weighted_face_fluxes(velocity,pressure,0.1,cache,conditions.pressure,independent);
        double squared=0,reconstructed_squared=0;
        for(size_t owned=0;owned<mesh->num_owned_cells();++owned)
        {
            const auto cell=static_cast<Pack::local_ordinal_type>(owned);
            const auto q=FVM::cell_flux_balance<Pack>(*mesh,equation.corrected_face_fluxes(),cell);
            const auto r=FVM::cell_flux_balance<Pack>(*mesh,independent,cell);
            squared+=q*q; reconstructed_squared+=r*r;
        }
        const auto norm=std::sqrt(global_sum(*comm,squared));
        EXPECT_LT(norm,1e-8);
        EXPECT_LT(std::sqrt(global_sum(*comm,reconstructed_squared)),1e-8);
        EXPECT_NEAR(result.continuity,norm,1e-13);
        expect_replicated(*comm,result.continuity);
        for(const auto face:independent.owned_face_ids())
            EXPECT_NEAR(equation.corrected_face_fluxes().local_value(face),independent.local_value(face),1e-12);
    }
}


/** Check the pressure/flux identity away from a converged solution. */
TEST(PressureProjectionMultiRankTest, NonAffineCorrectionMatchesFluxAndFullPressureDefect)
{
    using namespace SimpleFluid;
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial manufactured non-affine correction.";
    const auto mesh = test::make_skewed_prism_mesh<Pack>();
    BoundaryConditionSet physical;
    for (const auto& [batch, faces] : mesh->boundary_batches())
    {
        const auto name = mesh->boundary_batch_name(batch);
        physical.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        physical.pressure[name] = {BoundaryConditionType::Neumann, 2.0};
    }
    physical.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
    physical.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 17.0};
    auto correction_boundaries = physical.pressure;
    for (auto& [name, condition] : correction_boundaries) condition.value = 0.0;
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, physical);
    constexpr double dt = 0.13, rho = 13.0;
    for (const auto scheme : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear})
    {
        FieldType pressure(mesh, "identity_p"), correction(mesh, "identity_q"), updated_pressure(mesh, "identity_p1");
        VectorFieldType velocity(mesh, "identity_u"), updated_velocity(mesh, "identity_u1"), gradient(mesh, "identity_gq");
        FaceField<Pack> initial_flux(mesh, "identity_phi0"), final_flux(mesh, "identity_phi1"), explicit_flux(mesh, "identity_C");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const auto x = mesh->cell_centroid(cell);
            pressure.set_value(cell, 0.2*x.x*x.x - 0.1*x.y*x.y + 0.3*x.x*x.z);
            correction.set_value(cell, std::sin(0.8*x.x + 0.4*x.y - 0.3*x.z));
            velocity.set_value(cell, {0.7 + 0.2*x.y, -0.1*x.x, 0.05*x.z});
        }
        pressure.sync_ghosts(); correction.sync_ghosts(); velocity.sync_ghosts();
        FVM::PressureWeightedFaceFluxWorkspace<Pack> workspace(mesh);
        FVM::CellGradientCache<Pack> gradient_cache(mesh);
        FVM::cell_gradient(correction, correction_boundaries, gradient, gradient_cache, scheme);
        gradient.sync_ghosts();
        FVM::pressure_weighted_face_fluxes(velocity, pressure, dt/rho,
            boundaries, physical.pressure, workspace, initial_flux, scheme);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            updated_velocity.set_value(cell, velocity.value(cell) - gradient.value(cell)*dt);
            updated_pressure.set_value(cell, pressure.value(cell) + rho*correction.value(cell));
        }
        updated_velocity.sync_ghosts(); updated_pressure.sync_ghosts();
        FVM::pressure_weighted_face_fluxes(updated_velocity, updated_pressure, dt/rho,
            boundaries, physical.pressure, workspace, final_flux, scheme);
        const auto locations = FVM::detail::boundary_face_locations(*mesh);
        size_t non_affine_faces = 0;
        for (size_t local = 0; local < mesh->num_faces(); ++local)
        {
            const auto face = static_cast<Pack::local_ordinal_type>(local);
            if (!mesh->is_owned_face(face)) continue;
            const auto owner = mesh->owner_cell(face);
            vec3<double> d{}, interpolated{};
            double jump = 0.0;
            bool active = mesh->is_interior_face(face);
            if (active)
            {
                const auto neighbor = mesh->opposite_or_periodic_neighbor_cell(face, owner);
                d = mesh->cell_center_vector(face, owner);
                const auto weights = FVM::detail::interior_face_linear_weights(*mesh, face, owner, neighbor);
                interpolated = gradient.local_value(owner)*weights.first + gradient.local_value(neighbor)*weights.second;
                jump = correction.local_value(neighbor) - correction.local_value(owner);
            }
            else if (mesh->is_boundary_face(face) && locations.at(local).active)
            {
                const auto& location = locations.at(local);
                const auto name = mesh->boundary_batch_name(location.batch_id);
                active = correction_boundaries.at(name).type == BoundaryConditionType::Dirichlet;
                d = mesh->face_centroid(face) - mesh->cell_centroid(owner);
                interpolated = gradient.local_value(owner);
                jump = -correction.local_value(owner);
            }
            double correction_flux = 0.0, full_flux = 0.0;
            if (active)
            {
                const auto area = mesh->face_area_vector_outward(face, owner);
                ASSERT_GT(area.dot(d), 0.0);
                const auto alpha = area.dot(area)/area.dot(d);
                correction_flux = (area - d*alpha).dot(interpolated);
                full_flux = alpha*jump + correction_flux;
                if (std::abs(jump - interpolated.dot(d)) > 1e-3) ++non_affine_faces;
            }
            explicit_flux.set_value(face, correction_flux);
            EXPECT_NEAR(final_flux.value(face) - initial_flux.value(face), -dt*full_flux, 2e-12);
        }
        ASSERT_GT(non_affine_faces, 0U);
        const auto matrix = FVM::pressure_poisson_matrix<Pack>(*mesh,
            std::optional<Pack::global_ordinal_type>{}, [&](int batch, size_t)
            { return correction_boundaries.at(mesh->boundary_batch_name(batch)); });
        Pack::vector_type action(mesh->owned_cell_map(), false);
        matrix->apply(correction.vector(), action);
        double maximum_defect = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const auto target = 0.01*(1.0 + mesh->owned_cell_map()->getGlobalElement(cell));
            const auto before = FVM::cell_flux_balance<Pack>(*mesh, initial_flux, cell);
            const auto after = FVM::cell_flux_balance<Pack>(*mesh, final_flux, cell);
            const auto explicit_balance = FVM::cell_flux_balance<Pack>(*mesh, explicit_flux, cell);
            const auto defect = action.getData(0)[cell] - (-before + target)/dt - explicit_balance;
            maximum_defect = std::max(maximum_defect, std::abs(defect));
            EXPECT_NEAR(after - target, dt*defect, 5e-12);
        }
        EXPECT_GT(maximum_defect, 1e-3);
    }
}


/** A fixed pressure outlet keeps its physical gradient map during projection. */
TEST(PressureProjectionMultiRankTest, AccumulatedPressureFixedFluxMatchesPhysicalReconstruction)
{
    using namespace SimpleFluid;
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial accumulated-pressure boundary regression.";
    const auto mesh = test::make_skewed_prism_mesh<Pack>();
    BoundaryConditionSet conditions;
    for (const auto& [batch, faces] : mesh->boundary_batches())
    {
        const auto name = mesh->boundary_batch_name(batch);
        conditions.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        conditions.pressure[name] = {BoundaryConditionType::Neumann, 0.0};
    }
    conditions.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
    conditions.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 0.0};
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    for (const auto method : {SimpleFluid::PressureProjectionMethod::Anderson,
             SimpleFluid::PressureProjectionMethod::FullOperator})
    {
        SCOPED_TRACE(static_cast<int>(method));
        for (const auto scheme : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear})
        {
            FieldType pressure(mesh, "fixed_physical_p"), correction(mesh, "fixed_q");
            VectorFieldType velocity(mesh, "fixed_u");
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                const auto x = mesh->cell_centroid(cell);
                pressure.set_value(cell, 0.04*std::sin(x.x + 0.3*x.y));
                velocity.set_value(cell, {0.1*x.x*x.x, -0.03*x.y, 0.02*x.z});
            }
            pressure.sync_ghosts(); velocity.sync_ghosts();
            LinearSolverOptions linear;
            linear.tolerance = 1e-11;
            linear.max_iterations = 500;
            linear.preconditioner = LinearPreconditioner::None;
            if (method == SimpleFluid::PressureProjectionMethod::FullOperator)
            {
                linear.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
                linear.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
            }
            PressureProjectionEquation<Pack> equation(mesh, linear, conditions.pressure, scheme);
            equation.set_projection_method(method);
            equation.set_fixed_boundary_flux_provider({"xmax"},
                [](int, size_t, Pack::local_ordinal_type) { return 0.0; }, 1);
            const auto override_outlet = [&](FaceField<Pack>& flux)
            {
                for (const auto& [batch, faces] : mesh->boundary_batches())
                    if (mesh->boundary_batch_name(batch) == "xmax")
                        for (const auto face : faces.face_lids)
                            if (mesh->is_owned_face(face)) flux.set_value(face, 0.0);
            };
            FVM::PressureWeightedFaceFluxWorkspace<Pack> workspace(mesh);
            FaceField<Pack> independent(mesh, "fixed_independent_phi");
            FVM::pressure_weighted_face_fluxes(velocity, pressure, 0.13/7.0,
                boundaries, conditions.pressure, workspace, independent, scheme);
            override_outlet(independent);
            double initial_square = 0.0;
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto balance = FVM::cell_flux_balance<Pack>(*mesh, independent,
                    static_cast<Pack::local_ordinal_type>(owned));
                initial_square += balance*balance;
            }
            ASSERT_GT(initial_square, 1e-6);
            const auto projected = equation.project(pressure, correction, 0.13, 7.0, boundaries, velocity);
            ASSERT_TRUE(projected.linear_solve.converged);
            EXPECT_GT(projected.pressure_correction, 1e-7);
            const auto check = [&]
            {
                FVM::pressure_weighted_face_fluxes(velocity, pressure, 0.13/7.0,
                    boundaries, conditions.pressure, workspace, independent, scheme);
                override_outlet(independent);
                for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
                {
                    const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                    EXPECT_NEAR(FVM::cell_flux_balance<Pack>(*mesh, independent, cell), 0.0, 1e-9);
                }
                for (size_t local = 0; local < mesh->num_faces(); ++local)
                {
                    const auto face = static_cast<Pack::local_ordinal_type>(local);
                    if (mesh->is_owned_face(face))
                        EXPECT_NEAR(equation.corrected_face_fluxes().value(face), independent.value(face), 2e-12);
                }
                EXPECT_NEAR(correction.value(0), 0.0, 1e-14);
            };
            check();
            std::vector<double> previous_pressure;
            std::vector<vec3<double>> previous_velocity;
            double remaining_rhs_squared = 0.0;
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                previous_pressure.push_back(pressure.value(cell));
                previous_velocity.push_back(velocity.value(cell));
                if (mesh->owned_cell_map()->getGlobalElement(cell) == 0) continue;
                const auto rhs = FVM::cell_flux_balance<Pack>(*mesh, equation.corrected_face_fluxes(), cell) / 0.13;
                remaining_rhs_squared += rhs * rhs;
            }
            const auto reused = detail::PressureProjectionEquationTestAccess<Pack>::reuse_predictor(
                equation, pressure, correction, 0.13, 7.0, boundaries, velocity);
            EXPECT_TRUE(reused.linear_solve.converged);
            if (method == PressureProjectionMethod::FullOperator)
            {
                // This positive residual is accepted against the original RHS floor.
                EXPECT_GT(remaining_rhs_squared, 0.0);
                EXPECT_LE(std::sqrt(remaining_rhs_squared), linear.tolerance * projected.linear_solve.rhs_norm);
                EXPECT_EQ(reused.linear_solve.iterations, 0);
                EXPECT_DOUBLE_EQ(correction.owned_data().norm2(), 0.0);
                for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
                {
                    const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                    EXPECT_DOUBLE_EQ(pressure.value(cell), previous_pressure[owned]);
                    const auto actual = velocity.value(cell);
                    EXPECT_DOUBLE_EQ(actual.x, previous_velocity[owned].x);
                    EXPECT_DOUBLE_EQ(actual.y, previous_velocity[owned].y);
                    EXPECT_DOUBLE_EQ(actual.z, previous_velocity[owned].z);
                }
            }
            check();
        }
    }
}


/** The full action handles strided columns, gauge coupling and Tpetra apply algebra. */
TEST(PressureProjectionMultiRankTest, FullOperatorApplyMatchesIndependentMatrix)
{
    using Operator = SimpleFluid::detail::PressureProjectionOperator<Pack>;
    const auto comm = Tpetra::getDefaultComm();
    const auto map = Teuchos::rcp(new Pack::map_type(2, 0, comm));
    const auto make_matrix = [&](const std::array<std::array<double, 2>, 2>& coefficients)
    {
        auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 2));
        const Teuchos::Array<global_ordinal_type> columns{0, 1};
        for (size_t owned = 0; owned < map->getLocalNumElements(); ++owned)
        {
            const auto row = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            const Teuchos::Array<double> values{coefficients[row][0], coefficients[row][1]};
            matrix->insertGlobalValues(row, columns(), values());
        }
        matrix->fillComplete();
        return matrix;
    };
    // Row-only gauge retains an arbitrary gauge input's remote contribution.
    const auto compact = make_matrix({{{1.0, 0.0}, {-0.5, 0.5}}});
    const auto correction = make_matrix({{{0.0, 0.0}, {0.2, -0.2}}});
    Operator full(compact, [correction](const Pack::vector_type& input, Pack::vector_type& output)
        { correction->apply(input, output); });
    Pack::multi_vector_type storage(map, 5, true), output_storage(map, 5, true);
    const Teuchos::Array<size_t> selected{0, 2, 4};
    auto input = storage.subViewNonConst(selected());
    auto output = output_storage.subViewNonConst(selected());
    for (size_t column = 0; column < input->getNumVectors(); ++column)
    {
        auto x = input->getDataNonConst(column);
        auto y = output->getDataNonConst(column);
        for (size_t owned = 0; owned < map->getLocalNumElements(); ++owned)
        {
            const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            x[owned] = 1.0 + gid + 3.0 * column;
            y[owned] = 4.0 - gid + 2.0 * column;
        }
    }
    full.apply(*input, *output, Teuchos::NO_TRANS, -0.7, 0.3);
    for (size_t column = 0; column < input->getNumVectors(); ++column)
    {
        const auto x = input->getData(column);
        const auto y = output->getData(column);
        for (size_t owned = 0; owned < map->getLocalNumElements(); ++owned)
        {
            const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            const auto expected_action = gid == 0 ? 1.0 + 3.0 * column : 0.7;
            const auto initial_output = 4.0 - gid + 2.0 * column;
            EXPECT_NEAR(y[owned], -0.7 * expected_action + 0.3 * initial_output, 2e-14);
            EXPECT_DOUBLE_EQ(x[owned], 1.0 + gid + 3.0 * column);
        }
    }
    // All input columns are read before overlapping output is changed.
    full.apply(*input, *input);
    for (size_t column = 0; column < input->getNumVectors(); ++column)
    {
        const auto values = input->getData(column);
        for (size_t owned = 0; owned < map->getLocalNumElements(); ++owned)
        {
            const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            EXPECT_NEAR(values[owned], gid == 0 ? 1.0 + 3.0 * column : 0.7, 2e-14);
        }
    }
    input->putScalar(0.0);
    output->putScalar(std::numeric_limits<double>::quiet_NaN());
    full.apply(*input, *output);
    Teuchos::Array<double> norms(3);
    output->norm2(norms());
    for (const auto norm : norms) EXPECT_DOUBLE_EQ(norm, 0.0);
    output->putScalar(std::numeric_limits<double>::quiet_NaN());
    const auto applications = full.applications();
    full.apply(*input, *output, Teuchos::NO_TRANS, 0.0, 0.0);
    EXPECT_EQ(full.applications(), applications);
    output->norm2(norms());
    for (const auto norm : norms) EXPECT_DOUBLE_EQ(norm, 0.0);
    EXPECT_THROW(full.apply(*input, *output, Teuchos::TRANS), std::invalid_argument);
    Pack::multi_vector_type wrong_columns(map, 1, true);
    EXPECT_THROW(full.apply(*input, wrong_columns), std::invalid_argument);
    const auto other_map = Teuchos::rcp(new Pack::map_type(4, 0, comm));
    Pack::multi_vector_type wrong_map(other_map, 3, true);
    EXPECT_THROW(full.apply(*input, wrong_map), std::invalid_argument);
    full.deactivate();
    EXPECT_TRUE(full.getDomainMap()->isSameAs(*map));
    EXPECT_THROW(full.apply(*input, *output), std::invalid_argument);
}


/** Reject incompatible or divergent policies before changing the selected mode. */
TEST(PressureProjectionMultiRankTest, FullOperatorSelectionValidatesCollectively)
{
    using namespace SimpleFluid;
    const auto mesh = test::build_mesh<Pack>(test::make_box_database(4, 1, 1, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    LinearSolverOptions options;
    PressureProjectionEquation<Pack> equation(mesh, options);
    equation.set_projection_method(PressureProjectionMethod::FullOperator);
    options.backend = LinearSolverBackend::Cg;
    EXPECT_THROW(equation.set_linear_solver_options(options), std::invalid_argument);
    EXPECT_EQ(equation.linear_solver_options().backend, LinearSolverBackend::Gmres);
    EXPECT_EQ(equation.projection_method(), PressureProjectionMethod::FullOperator);
    equation.set_projection_method(PressureProjectionMethod::Anderson);
    equation.set_linear_solver_options(options);
    EXPECT_THROW(equation.set_projection_method(PressureProjectionMethod::FullOperator), std::invalid_argument);
    EXPECT_EQ(equation.projection_method(), PressureProjectionMethod::Anderson);
    options.backend = LinearSolverBackend::BiCGStab;
    options.preconditioner = LinearPreconditioner::ILUT;
    equation.set_linear_solver_options(options);
    const auto invalid = comm->getRank() == 0 ? static_cast<PressureProjectionMethod>(-1)
                                              : PressureProjectionMethod::Anderson;
    EXPECT_THROW(equation.set_projection_method(invalid), std::invalid_argument);
    EXPECT_EQ(equation.projection_method(), PressureProjectionMethod::Anderson);
    if (comm->getSize() > 1)
    {
        const auto divergent = comm->getRank() == 0 ? PressureProjectionMethod::FullOperator
                                                   : PressureProjectionMethod::Anderson;
        EXPECT_THROW(equation.set_projection_method(divergent), std::invalid_argument);
        EXPECT_EQ(equation.projection_method(), PressureProjectionMethod::Anderson);
    }
}


/** A homogeneous zero RHS is accepted before preconditioning or Krylov iteration. */
TEST(PressureProjectionMultiRankTest, FullOperatorAcceptsZeroCorrectionAndDefersInvalidOptions)
{
    using namespace SimpleFluid;
    const auto mesh = test::build_mesh<Pack>(test::make_box_database(4, 1, 1, 0.25));
    LinearSolverOptions options;
    options.backend = LinearSolverBackend::BiCGStab;
    options.preconditioner = LinearPreconditioner::ILUT;
    PressureProjectionEquation<Pack> equation(mesh, options);
    equation.set_projection_method(PressureProjectionMethod::FullOperator);
    FieldType pressure(mesh, "zero_full_pressure");
    VectorFieldType velocity(mesh, vec3{}, "zero_full_velocity");
    const BoundaryConditionSet conditions;
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    const auto result = equation.project(pressure, 0.1, 1.0, boundaries, velocity);
    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_EQ(result.linear_solve.iterations, 0);
    EXPECT_DOUBLE_EQ(result.linear_solve.achieved_tolerance, 0.0);
    EXPECT_DOUBLE_EQ(result.continuity, 0.0);
    options.max_iterations = 0;
    EXPECT_NO_THROW(equation.set_linear_solver_options(options));
    EXPECT_THROW(equation.project(pressure, 0.1, 1.0, boundaries, velocity), std::invalid_argument);
}
