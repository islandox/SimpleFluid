/**
 * @file testPressureProjectionMultiRank.cc
 * @brief Distributed pressure-projection gauge and residual tests.
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
#include <limits>

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
        pressure, time_step, boundary_cache, velocity, source);

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
        face_fluxes);

    scalar_type local_continuity_norm_squared{};
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto balance = SimpleFluid::FVM::cell_flux_balance<Pack>(
            *mesh, face_fluxes, cell_lid);
        local_continuity_norm_squared += balance * balance;
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
