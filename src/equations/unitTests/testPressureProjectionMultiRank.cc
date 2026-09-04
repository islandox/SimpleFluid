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
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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
