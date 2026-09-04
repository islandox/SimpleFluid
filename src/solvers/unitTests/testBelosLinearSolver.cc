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

#include <array>
#include <limits>

namespace SimpleFluid::detail
{

/**
 * @brief Expose retained preconditioner state to focused unit tests.
 *
 * @tparam Pack Tpetra type pack used by the solver under test.
 */
template<TpetraTypePack Pack>
struct BelosLinearSolverTestAccess
{
    static std::size_t preconditioner_setup_count(
        const BelosLinearSolver<Pack>& solver) noexcept
    {
        return solver.d_preconditioner_setup_count;
    }

    static real_t true_relative_residual(
        const BelosLinearSolver<Pack>& solver,
        const Teuchos::RCP<const typename Pack::operator_type>& matrix,
        const typename Pack::multi_vector_type& rhs,
        const typename Pack::multi_vector_type& solution,
        LinearResidualScaling residual_scaling = {})
    {
        return solver.true_relative_residual(
            matrix, rhs, solution, residual_scaling);
    }
};

} // namespace SimpleFluid::detail

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Read the number of preconditioner constructions from a solver.
 *
 * @param solver Solver instance under test.
 * @return Number of completed preconditioner setups.
 */
std::size_t preconditioner_setup_count(
    const SimpleFluid::BelosLinearSolver<Pack>& solver)
{
    using Access =
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    return Access::preconditioner_setup_count(solver);
}

double true_relative_residual(
    const SimpleFluid::BelosLinearSolver<Pack>& solver,
    const Teuchos::RCP<const Pack::operator_type>& matrix,
    const Pack::multi_vector_type& rhs,
    const Pack::multi_vector_type& solution,
    SimpleFluid::LinearResidualScaling residual_scaling = {})
{
    using Access =
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    return Access::true_relative_residual(
        solver, matrix, rhs, solution, residual_scaling);
}

/** @brief Minimal identity operator used to test non-CRS Belos inputs. */
class IdentityOperator final : public Pack::operator_type
{
public:
    explicit IdentityOperator(Teuchos::RCP<const Pack::map_type> map)
        : d_map(std::move(map))
    {
    }

    Teuchos::RCP<const Pack::map_type> getDomainMap() const override
    {
        return d_map;
    }

    Teuchos::RCP<const Pack::map_type> getRangeMap() const override
    {
        return d_map;
    }

    void apply(
        const Pack::multi_vector_type& input,
        Pack::multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = 1.0,
        double beta = 0.0) const override
    {
        ASSERT_EQ(mode, Teuchos::NO_TRANS);
        output.update(alpha, input, beta);
    }

private:
    Teuchos::RCP<const Pack::map_type> d_map;
};

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

/** @brief Verify stable names and accepted configuration aliases. */
TEST(BelosLinearSolverTest, ParsesBackendAndPreconditionerNames)
{
    using SimpleFluid::LinearPreconditioner;
    using SimpleFluid::LinearSolverBackend;

    EXPECT_EQ(
        SimpleFluid::parse_linear_solver_backend("gmres"),
        LinearSolverBackend::Gmres);
    EXPECT_EQ(
        SimpleFluid::parse_linear_solver_backend("CG"),
        LinearSolverBackend::Cg);
    EXPECT_EQ(
        SimpleFluid::parse_linear_solver_backend("BiCGStab"),
        LinearSolverBackend::BiCGStab);
    EXPECT_EQ(
        SimpleFluid::parse_linear_preconditioner("none"),
        LinearPreconditioner::None);
    EXPECT_EQ(
        SimpleFluid::parse_linear_preconditioner("jacobi"),
        LinearPreconditioner::Jacobi);
    EXPECT_EQ(
        SimpleFluid::parse_linear_preconditioner("RILUK"),
        LinearPreconditioner::ILU0);
    EXPECT_EQ(
        SimpleFluid::parse_linear_preconditioner("ILUT"),
        LinearPreconditioner::ILUT);
    EXPECT_EQ(
        SimpleFluid::parse_linear_preconditioner("MueLu"),
        LinearPreconditioner::MueLu);

    EXPECT_EQ(
        SimpleFluid::to_string(LinearSolverBackend::BiCGStab),
        "bicgstab");
    EXPECT_EQ(
        SimpleFluid::to_string(LinearPreconditioner::ILU0),
        "ilu0");
    EXPECT_THROW(
        SimpleFluid::parse_linear_solver_backend("direct"),
        std::invalid_argument);
    EXPECT_THROW(
        SimpleFluid::parse_linear_preconditioner("DILU"),
        std::invalid_argument);
}

/** @brief Verify every exposed backend solves scalar and multi-RHS systems. */
TEST(BelosLinearSolverTest, SupportsSelectableBackendsForMultipleRightHandSides)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 4, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);

    Pack::multi_vector_type rhs(map, 3, true);
    for (size_t column = 0; column < rhs.getNumVectors(); ++column)
    {
        auto values = rhs.getDataNonConst(column);
        std::fill(
            values.begin(), values.end(),
            static_cast<double>(column + 1));
    }

    constexpr std::array backends{
        SimpleFluid::LinearSolverBackend::Gmres,
        SimpleFluid::LinearSolverBackend::Cg,
        SimpleFluid::LinearSolverBackend::BiCGStab};
    for (const auto backend : backends)
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        Pack::multi_vector_type solution(map, 3, true);
        SimpleFluid::LinearSolverOptions options;
        options.backend = backend;
        options.tolerance = 1.0e-12;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto statistics =
            solver.solve_with_statistics(op, rhs, solution, options);
        ASSERT_TRUE(statistics.converged);
        for (size_t column = 0; column < rhs.getNumVectors(); ++column)
        {
            const auto expected = static_cast<double>(column + 1);
            for (const auto value : solution.getData(column))
            {
                EXPECT_NEAR(value, expected, 1.0e-12);
            }
        }
    }
}

/** @brief Verify changing the backend recreates compatible retained state. */
TEST(BelosLinearSolverTest, SwitchesBackendWithCompatibleMaps)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 4, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    SimpleFluid::BelosLinearSolver<Pack> solver;
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;

    constexpr std::array backends{
        SimpleFluid::LinearSolverBackend::Gmres,
        SimpleFluid::LinearSolverBackend::BiCGStab,
        SimpleFluid::LinearSolverBackend::Cg};
    for (size_t index = 0; index < backends.size(); ++index)
    {
        options.backend = backends[index];
        const auto expected = static_cast<double>(index + 2);
        rhs.putScalar(expected);
        solution.putScalar(0.0);
        ASSERT_TRUE(solver.solve(op, rhs, solution, options));
        for (const auto value : solution.getData())
        {
            EXPECT_NEAR(value, expected, 1.0e-12);
        }
    }
}

/**
 * @brief A cancellation-small but already acceptable warm residual is scaled
 *        by the RHS instead of by itself.
 */
TEST(BelosLinearSolverTest, UsesRhsScaledResidualForWarmStart)
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
    rhs.putScalar(1.0);
    solution.putScalar(1.0 + 1.0e-12);

    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = 1.0e-10;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics =
        solver.solve_with_statistics(op, rhs, solution, options);

    EXPECT_TRUE(statistics.converged);
    EXPECT_EQ(statistics.iterations, 0);
    EXPECT_LT(statistics.achieved_tolerance, options.tolerance);
}

/** @brief A retained reference norm bounds scaling for a tiny nonzero RHS. */
TEST(BelosLinearSolverTest, UsesReferenceNormFloorForTinyRhs)
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
    constexpr double rhs_value = 1.0e-15;
    constexpr double residual_value = 1.25e-24;
    rhs.putScalar(rhs_value);
    solution.putScalar(rhs_value - residual_value);

    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto relative =
        true_relative_residual(solver, op, rhs, solution);
    SimpleFluid::LinearResidualScaling scaling;
    scaling.rhs_norm_floor = 2.0 * rhs.norm2();
    const auto scaled =
        true_relative_residual(solver, op, rhs, solution, scaling);

    EXPECT_GT(relative, 1.0e-9);
    EXPECT_LT(scaled, 1.0e-9);
    EXPECT_NEAR(scaled, 0.5 * relative, 1.0e-15);
}

/** @brief Invalid explicit-residual norm floors are rejected before solving. */
TEST(BelosLinearSolverTest, RejectsInvalidResidualNormFloor)
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
    rhs.putScalar(1.0);
    SimpleFluid::BelosLinearSolver<Pack> solver;

    for (const auto invalid : {
             -1.0,
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()})
    {
        SimpleFluid::LinearResidualScaling scaling;
        scaling.rhs_norm_floor = invalid;
        EXPECT_THROW(
            solver.solve_with_statistics(
                op, rhs, solution, {}, scaling),
            std::invalid_argument);
    }
}

/**
 * @brief Zero-RHS columns use absolute scaling and discard a harmful nonzero
 *        guess without disturbing helpful columns.
 */
TEST(BelosLinearSolverTest, HandlesMixedZeroRhsColumns)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    Pack::multi_vector_type rhs(map, 2, true);
    Pack::multi_vector_type solution(map, 2, true);
    {
        auto values = rhs.getDataNonConst(0);
        std::fill(values.begin(), values.end(), 2.0);
    }
    {
        auto values = solution.getDataNonConst(0);
        std::fill(values.begin(), values.end(), 2.0);
    }
    {
        auto values = solution.getDataNonConst(1);
        std::fill(values.begin(), values.end(), 7.0);
    }

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics =
        solver.solve_with_statistics(op, rhs, solution, options);

    ASSERT_TRUE(statistics.converged);
    for (const auto value : solution.getData(0))
        EXPECT_DOUBLE_EQ(value, 2.0);
    for (const auto value : solution.getData(1))
        EXPECT_DOUBLE_EQ(value, 0.0);
}

/** @brief A non-finite residual column can never report convergence. */
TEST(BelosLinearSolverTest, RejectsNonFiniteTrueResidual)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    Pack::multi_vector_type rhs(map, 1, true);
    Pack::multi_vector_type solution(map, 1, true);
    rhs.putScalar(std::numeric_limits<double>::quiet_NaN());

    SimpleFluid::BelosLinearSolver<Pack> solver;
    EXPECT_FALSE(std::isfinite(
        true_relative_residual(solver, op, rhs, solution)));
}

/** @brief Verify retained solver state handles a changed right-hand side. */
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

/** @brief Verify one solve reports convergence iterations and achieved tolerance. */
TEST(BelosLinearSolverTest, ReportsIterationsAndAchievedTolerance)
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
    rhs.putScalar(2.0);

    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics =
        solver.solve_with_statistics(op, rhs, solution);

    EXPECT_TRUE(statistics.converged);
    EXPECT_GE(statistics.iterations, 0);
    EXPECT_TRUE(std::isfinite(statistics.achieved_tolerance));
    EXPECT_GE(statistics.achieved_tolerance, 0.0);
    EXPECT_NEAR(statistics.rhs_norm, rhs.norm2(), 1.0e-14);
}

/** @brief Verify CRS matrices can be solved with a MueLu preconditioner. */
TEST(BelosLinearSolverTest, SupportsMueLuForCrsMatrices)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(3.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner =
        SimpleFluid::LinearPreconditioner::MueLu;

    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics =
        solver.solve_with_statistics(op, rhs, solution, options);

    EXPECT_TRUE(statistics.converged);
    EXPECT_GE(statistics.iterations, 0);
}

/** @brief Verify all exposed Ifpack2 preconditioners solve CRS systems. */
TEST(BelosLinearSolverTest, SupportsIfpack2PreconditionersForCrsMatrices)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
    Pack::vector_type rhs(map, true);
    rhs.putScalar(3.0);

    constexpr std::array preconditioners{
        SimpleFluid::LinearPreconditioner::Jacobi,
        SimpleFluid::LinearPreconditioner::ILU0,
        SimpleFluid::LinearPreconditioner::ILUT};
    for (const auto preconditioner : preconditioners)
    {
        SCOPED_TRACE(SimpleFluid::to_string(preconditioner));
        Pack::vector_type solution(map, true);
        SimpleFluid::LinearSolverOptions options;
        options.backend =
            SimpleFluid::LinearSolverBackend::BiCGStab;
        options.preconditioner = preconditioner;
        options.tolerance = 1.0e-12;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto statistics =
            solver.solve_with_statistics(op, rhs, solution, options);
        ASSERT_TRUE(statistics.converged);
        for (const auto value : solution.getData())
            EXPECT_NEAR(value, 3.0, 1.0e-12);
    }
}

/** @brief Verify MueLu is rebuilt between solves unless reuse is enabled. */
TEST(BelosLinearSolverTest, RebuildsMueLuByDefault)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(3.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner =
        SimpleFluid::LinearPreconditioner::MueLu;
    ASSERT_FALSE(options.reuse_preconditioner);

    SimpleFluid::BelosLinearSolver<Pack> solver;
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));

    EXPECT_EQ(preconditioner_setup_count(solver), 2U);
}

/** @brief Verify MueLu reuse is tied to operator identity and enablement. */
TEST(BelosLinearSolverTest,
     ReusesMueLuOnlyForSameOperatorAndInvalidatesWhenDisabled)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    auto first_matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto second_matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto first_op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(first_matrix);
    auto second_op =
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(second_matrix);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(3.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner =
        SimpleFluid::LinearPreconditioner::MueLu;
    options.reuse_preconditioner = true;

    SimpleFluid::BelosLinearSolver<Pack> solver;
    ASSERT_TRUE(solver.solve(first_op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 1U);

    rhs.putScalar(5.0);
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(first_op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 1U);
    for (const auto value : solution.getData())
    {
        EXPECT_DOUBLE_EQ(value, 5.0);
    }

    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(second_op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 2U);

    options.preconditioner = SimpleFluid::LinearPreconditioner::None;
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(second_op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 2U);

    options.preconditioner =
        SimpleFluid::LinearPreconditioner::MueLu;
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(second_op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 3U);
}

/** @brief Explicit reset invalidates same-operator numeric solver state. */
TEST(BelosLinearSolverTest, ResetForcesSameOperatorPreconditionerRebuild)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(3.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
    options.reuse_preconditioner = true;

    SimpleFluid::BelosLinearSolver<Pack> solver;
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 1U);

    solver.reset();
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve(op, rhs, solution, options));
    EXPECT_EQ(preconditioner_setup_count(solver), 2U);
}

/** @brief Verify MueLu rejects operators that are not Tpetra CRS matrices. */
TEST(BelosLinearSolverTest, RejectsMueLuForNonCrsOperators)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    Teuchos::RCP<const Pack::operator_type> op =
        Teuchos::rcp(new IdentityOperator(map));

    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner =
        SimpleFluid::LinearPreconditioner::MueLu;

    SimpleFluid::BelosLinearSolver<Pack> solver;
    EXPECT_THROW(
        solver.solve_with_statistics(op, rhs, solution, options),
        std::invalid_argument);
}

/** @brief Verify Ifpack2 preconditioners reject non-CRS operators. */
TEST(BelosLinearSolverTest, RejectsIfpack2ForNonCrsOperators)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    Teuchos::RCP<const Pack::operator_type> op =
        Teuchos::rcp(new IdentityOperator(map));
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);

    constexpr std::array preconditioners{
        SimpleFluid::LinearPreconditioner::Jacobi,
        SimpleFluid::LinearPreconditioner::ILU0,
        SimpleFluid::LinearPreconditioner::ILUT};
    for (const auto preconditioner : preconditioners)
    {
        SCOPED_TRACE(SimpleFluid::to_string(preconditioner));
        SimpleFluid::LinearSolverOptions options;
        options.preconditioner = preconditioner;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        EXPECT_THROW(
            solver.solve_with_statistics(op, rhs, solution, options),
            std::invalid_argument);
    }
}
