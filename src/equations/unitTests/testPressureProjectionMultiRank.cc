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
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh,options);
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
    PressureProjectionEquation<Pack,Native> equation(mesh,options,conditions.pressure);
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
