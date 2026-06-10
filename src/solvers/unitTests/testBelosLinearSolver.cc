/**
 * @file testBelosLinearSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for the Belos linear solver wrapper
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

/**
 * @brief Solves an identity system with GMRES and verifies the solution matches the RHS.
 */
TEST(BelosLinearSolverTest, SolvesIdentitySystem)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(invalid_global_size,
                                               3,
                                               0,
                                               Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row),
                              static_cast<double>(gid + 1));
    }

    auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-14;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(op, rhs, solution, options));

    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
        EXPECT_DOUBLE_EQ(solution.getData()[row], static_cast<double>(gid + 1));
    }
}

/**
 * @brief Solves a three-column identity system and verifies all RHS columns.
 */
TEST(BelosLinearSolverTest, SolvesMultiVectorIdentitySystem)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(invalid_global_size,
                                               3,
                                               0,
                                               Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);

    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type solution(map, 3, true);
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(row);
        const auto gid = map->getGlobalElement(lid);
        for (size_t component = 0; component < 3; ++component)
        {
            rhs.replaceLocalValue(
                lid, component,
                static_cast<double>(gid + 1 + 10 * component));
        }
    }

    auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-14;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(op, rhs, solution, options));

    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(row);
        const auto gid = map->getGlobalElement(lid);
        for (size_t component = 0; component < 3; ++component)
        {
            EXPECT_DOUBLE_EQ(
                solution.getData(component)[row],
                static_cast<double>(gid + 1 + 10 * component));
        }
    }
}

TEST(BelosLinearSolverTest, ReusesSolverForChangedRightHandSide)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    SimpleFluid::BelosLinearSolver<Pack> solver;
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-14;

    rhs.putScalar(2.0);
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));
    for (const auto value : solution.getData())
    {
        EXPECT_DOUBLE_EQ(value, 2.0);
    }

    rhs.putScalar(5.0);
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));
    for (const auto value : solution.getData())
    {
        EXPECT_DOUBLE_EQ(value, 5.0);
    }
}
