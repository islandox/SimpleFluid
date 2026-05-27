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

#include "operators/FvmOperators.hh"
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
    auto matrix = SimpleFluid::FvmOperators::identity_matrix<Pack>(map);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row),
                              static_cast<double>(gid + 1));
    }

    auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-14;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(op, rhs, solution, options));

    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
        EXPECT_DOUBLE_EQ(solution.getData()[row], static_cast<double>(gid + 1));
    }
}
