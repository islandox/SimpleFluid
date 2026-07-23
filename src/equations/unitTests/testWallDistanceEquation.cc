/**
 * @file testWallDistanceEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit and MPI tests for distributed Poisson wall distance.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/turbulence/WallDistanceEquation.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::Mesh<Pack>;
using Field = SimpleFluid::CellField<Pack>;
using Equation = SimpleFluid::PoissonWallDistanceEquation<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<Mesh> make_line_mesh(size_t cell_count)
{
    const auto spacing = 1.0 / static_cast<double>(cell_count);
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            cell_count, 1, 1, spacing));
}

double global_maximum(const Mesh& mesh, double local_value)
{
    double global_value = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
        &local_value, &global_value);
    return global_value;
}

double solve_maximum_error(size_t cell_count,
                           const SimpleFluid::ArrString& walls)
{
    auto mesh = make_line_mesh(cell_count);
    Field distance(mesh, -7.0, "wall_distance");
    Equation equation(mesh);
    SimpleFluid::WallDistanceEquationOptions options;
    options.linear_solver.tolerance = 1.0e-13;
    options.linear_solver.max_iterations = 500;
    equation.solve(walls, distance, options);

    double local_error = 0.0;
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(local);
        const auto x = mesh->cell_centroid(cell_lid).x;
        const auto exact =
            walls.size() == 1 ? x : std::min(x, 1.0 - x);
        const auto value = distance.local_value(cell_lid);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, 0.0);
        if (distance.is_owned_cell(cell_lid))
        {
            local_error = std::max(
                local_error, std::abs(value - exact));
        }
    }
    return global_maximum(*mesh, local_error);
}

/** @brief A one-wall Poisson reconstruction converges to geometric distance. */
TEST(WallDistanceEquationTest, SingleWallDistanceConvergesUnderRefinement)
{
    const auto coarse_error =
        solve_maximum_error(8, {"xmin"});
    const auto fine_error =
        solve_maximum_error(32, {"xmin"});

    EXPECT_LT(fine_error, coarse_error);
    EXPECT_LT(fine_error, 4.0e-2);
}

/** @brief Opposed Dirichlet walls recover distance to their nearest member. */
TEST(WallDistanceEquationTest, OpposedWallsRecoverNearestDistance)
{
    const auto error =
        solve_maximum_error(32, {"xmin", "xmax"});
    EXPECT_LT(error, 4.0e-2);
}

/**
 * @brief Invalid selections fail collectively without modifying the output.
 */
TEST(WallDistanceEquationTest, RejectsInvalidSelectionsWithoutPublishing)
{
    auto mesh = make_line_mesh(8);
    Field distance(mesh, 17.0, "wall_distance");
    Equation equation(mesh);

    EXPECT_THROW(equation.solve({}, distance), std::invalid_argument);
    EXPECT_THROW(
        equation.solve({"xmin", "xmin"}, distance),
        std::invalid_argument);
    EXPECT_THROW(
        equation.solve({"not_a_boundary"}, distance),
        std::invalid_argument);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_DOUBLE_EQ(
            distance.value(
                static_cast<Pack::local_ordinal_type>(owned)),
            17.0);
    }
}

/** @brief The output field must use the equation's mesh. */
TEST(WallDistanceEquationTest, RejectsOutputOnAnotherMesh)
{
    auto mesh = make_line_mesh(4);
    auto other_mesh = make_line_mesh(4);
    Field other_distance(other_mesh, 1.0, "other_wall_distance");
    Equation equation(mesh);

    EXPECT_THROW(
        equation.solve({"xmin"}, other_distance),
        std::invalid_argument);
}

/**
 * @brief Invalid caller storage is rejected before either field view changes.
 *
 * Replacing the overlap map models a late output-import failure without
 * corrupting the underlying values. The equation must validate that commit
 * precondition before solving instead of updating owned values first.
 */
TEST(WallDistanceEquationTest,
     RejectsInvalidOutputStorageWithoutPartialPublication)
{
    auto mesh = make_line_mesh(8);
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This regression requires at least two MPI ranks.";
    }

    Field distance(mesh, 23.0, "wall_distance");
    Equation equation(mesh);
    const auto expected_overlap_map =
        distance.overlap_data().getMap();

    // Tpetra permits relabeling a MultiVector without moving its data. This
    // leaves the values intact while making the caller's ghost import invalid.
    distance.overlap_data().replaceMap(mesh->owned_cell_map());
    EXPECT_THROW(
        equation.solve({"xmin"}, distance),
        std::invalid_argument);
    distance.overlap_data().replaceMap(expected_overlap_map);

    for (size_t owned = 0;
         owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(distance.value(cell_lid), 23.0);
    }
    for (size_t local = 0;
         local < mesh->num_local_cells(); ++local)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(distance.local_value(cell_lid), 23.0);
    }
}

/**
 * @brief A partition with no local selected face still receives distance.
 */
TEST(WallDistanceEquationTest, DistributedSolveSupportsRankWithoutLocalWall)
{
    auto mesh = make_line_mesh(8);
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    int local_has_wall = 0;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "xmin")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (mesh->is_owned_face(face_lid)
                && mesh->is_boundary_face(face_lid))
            {
                local_has_wall = 1;
            }
        }
    }
    int minimum_has_wall = 0;
    int maximum_has_wall = 0;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1,
        &local_has_wall, &minimum_has_wall);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1,
        &local_has_wall, &maximum_has_wall);
    ASSERT_EQ(minimum_has_wall, 0);
    ASSERT_EQ(maximum_has_wall, 1);

    Field distance(mesh, "wall_distance");
    Equation equation(mesh);
    SimpleFluid::WallDistanceEquationOptions options;
    options.linear_solver.tolerance = 1.0e-13;
    options.linear_solver.max_iterations = 500;
    ASSERT_NO_THROW(equation.solve({"xmin"}, distance, options));

    double local_error = 0.0;
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(local);
        const auto value = distance.local_value(cell_lid);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, 0.0);
        if (distance.is_owned_cell(cell_lid))
        {
            local_error = std::max(
                local_error,
                std::abs(value - mesh->cell_centroid(cell_lid).x));
        }
    }
    EXPECT_LT(global_maximum(*mesh, local_error), 1.5e-1);
}

/** @brief Rank-dependent wall selections are rejected before assembly. */
TEST(WallDistanceEquationTest, RejectsRankInconsistentWallSelection)
{
    auto mesh = make_line_mesh(8);
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    Field distance(mesh, 3.0, "wall_distance");
    Equation equation(mesh);
    const SimpleFluid::ArrString walls =
        communicator->getRank() == 0
            ? SimpleFluid::ArrString{"xmin"}
            : SimpleFluid::ArrString{"xmax"};
    EXPECT_THROW(
        equation.solve(walls, distance),
        std::invalid_argument);
}

} // namespace
