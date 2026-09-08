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
    template<class Solver, class... Arguments>
    static void replace_solver(
        BelosLinearSolver<Pack>& solver, Arguments&&... arguments)
    {
        solver.d_solver = Teuchos::rcp(new Solver(
            solver.d_problem, solver.d_parameters,
            std::forward<Arguments>(arguments)...));
    }

    static std::size_t preconditioner_setup_count(
        const BelosLinearSolver<Pack>& solver) noexcept
    {
        return solver.d_preconditioner_setup_count;
    }

    static Teuchos::RCP<const typename Pack::operator_type> preconditioner(
        const BelosLinearSolver<Pack>& solver)
    {
        return solver.d_preconditioner;
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

/** @brief Build one small independent dense block on each participating rank. */
template<std::size_t Count>
Teuchos::RCP<Pack::matrix_type> block_matrix(
    const std::array<std::array<double, Count>, Count>& entries)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, Count, 0, Tpetra::getDefaultComm()));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, Count));
    for (std::size_t row = 0; row < Count; ++row)
    {
        Teuchos::Array<Pack::global_ordinal_type> columns;
        Teuchos::Array<double> values;
        for (std::size_t column = 0; column < Count; ++column)
        {
            if (entries[row][column] != 0.0)
            {
                columns.push_back(map->getGlobalElement(column));
                values.push_back(entries[row][column]);
            }
        }
        matrix->insertGlobalValues(
            map->getGlobalElement(row), columns(), values());
    }
    matrix->fillComplete();
    return matrix;
}

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
        ++d_application_count;
        output.update(alpha, input, beta);
    }

    std::size_t application_count() const { return d_application_count; }
    void reset_application_count() const { d_application_count = 0; }

private:
    Teuchos::RCP<const Pack::map_type> d_map;
    mutable std::size_t d_application_count = 0;
};

/** @brief Reproduce recurrence/true-residual disagreement deterministically. */
struct ResidualGap
{
    double relative_error;
    int remaining_injections = 1;
    int solve_calls = 0;
};

/**
 * @brief Perturb a converged identity solution after the real Belos solve.
 *
 * This models a residual gap without relying on a particular MPI reduction
 * order to trigger roundoff. Refinement still uses the real Krylov manager
 * and the wrapper's independently recomputed residual and original RHS.
 */
template<class Manager>
class ResidualGapSolver final : public Manager
{
public:
    ResidualGapSolver(
        const Teuchos::RCP<SimpleFluid::BelosLinearSolver<Pack>::problem_type>& problem,
        const Teuchos::RCP<Teuchos::ParameterList>& parameters,
        ResidualGap& gap)
        : Manager(problem, parameters), d_gap(gap)
    {
    }

    Belos::ReturnType solve() override
    {
        const auto status = Manager::solve();
        ++d_gap.solve_calls;
        if (status == Belos::Converged && d_gap.remaining_injections > 0)
        {
            --d_gap.remaining_injections;
            this->getProblem().getLHS()->scale(1.0 - d_gap.relative_error);
        }
        return status;
    }

private:
    ResidualGap& d_gap;
};

void inject_residual_gap(
    SimpleFluid::BelosLinearSolver<Pack>& solver,
    SimpleFluid::LinearSolverBackend backend, ResidualGap& gap)
{
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    if (backend == SimpleFluid::LinearSolverBackend::Cg)
    {
        using Manager = Belos::PseudoBlockCGSolMgr<
            Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
        Access::replace_solver<ResidualGapSolver<Manager>>(solver, gap);
    }
    else
    {
        using Manager = Belos::BiCGStabSolMgr<
            Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
        Access::replace_solver<ResidualGapSolver<Manager>>(solver, gap);
    }
}

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
        SimpleFluid::parse_linear_solver_backend("PCG"),
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
    EXPECT_EQ(SimpleFluid::parse_linear_preconditioner("DIC"),
        LinearPreconditioner::DIC);
    EXPECT_EQ(SimpleFluid::parse_linear_preconditioner("GS"),
        LinearPreconditioner::GaussSeidel);
    EXPECT_EQ(SimpleFluid::parse_linear_preconditioner("symGaussSeidel"),
        LinearPreconditioner::SymmetricGaussSeidel);
    for (const auto preconditioner : {
             LinearPreconditioner::DIC, LinearPreconditioner::GaussSeidel,
             LinearPreconditioner::SymmetricGaussSeidel})
    {
        EXPECT_EQ(SimpleFluid::parse_linear_preconditioner(
            SimpleFluid::to_string(preconditioner)), preconditioner);
    }

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

/** @brief Count actual operator calls while exercising real Krylov managers. */
TEST(BelosLinearSolverTest, ReusesPreparedInitialResidualAndSkipsVerifiedZeroScreening)
{
    const auto map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        3, 0, Tpetra::getDefaultComm()));
    const auto op = Teuchos::rcp(new IdentityOperator(map));
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Gmres,
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        SimpleFluid::LinearSolverOptions options;
        options.backend = backend;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        solution.putScalar(0.0);
        op->reset_application_count();
        const auto screened = solver.solve_with_statistics(op, rhs, solution, options);
        ASSERT_TRUE(screened.converged);
        const auto screened_calls = op->application_count();

        // A poisoned incoming guess proves the explicit API establishes zero.
        solution.putScalar(std::numeric_limits<double>::quiet_NaN());
        op->reset_application_count();
        const auto zero = solver.solve_from_zero_with_statistics(op, rhs, solution, options);
        ASSERT_TRUE(zero.converged);
        EXPECT_EQ(zero.iterations, screened.iterations);
        EXPECT_EQ(op->application_count() + 1, screened_calls);
        EXPECT_GE(op->application_count(), 1U); // The final true residual remains.
        for (const auto value : solution.getData())
            EXPECT_NEAR(value, 1.0, 1.0e-14);

        if (backend == SimpleFluid::LinearSolverBackend::Cg)
        {
            solution.putScalar(1.0);
            op->reset_application_count();
            const auto warm = solver.solve_with_statistics(op, rhs, solution, options);
            EXPECT_TRUE(warm.converged);
            EXPECT_EQ(warm.iterations, 0);
            // One screening apply and one final check; setProblem adds none.
            EXPECT_EQ(op->application_count(), 2U);
        }
    }
}

TEST(BelosLinearSolverTest, RefreshesPreparedResidualForChangedMapsBackendsRhsAndColumnCounts)
{
    SimpleFluid::BelosLinearSolver<Pack> solver;
    for (const auto rows : {3U, 5U, 3U})
    {
        const auto map = Teuchos::rcp(new Pack::map_type(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            rows, 0, Tpetra::getDefaultComm()));
        const auto op = Teuchos::rcp(new IdentityOperator(map));
        for (const auto backend : {
                 SimpleFluid::LinearSolverBackend::Gmres,
                 SimpleFluid::LinearSolverBackend::Cg,
                 SimpleFluid::LinearSolverBackend::BiCGStab})
        {
            SCOPED_TRACE(SimpleFluid::to_string(backend));
            SimpleFluid::LinearSolverOptions options;
            options.backend = backend;
            for (const auto columns : {1U, 3U, 2U})
            {
                SCOPED_TRACE(columns);
                Pack::multi_vector_type rhs(map, columns, true), solution(map, columns, true);
                for (const double value : {2.0, -3.0})
                {
                    rhs.putScalar(value);
                    auto statistics = solver.solve_from_zero_with_statistics(
                        op, rhs, solution, options);
                    ASSERT_TRUE(statistics.converged);
                    statistics = solver.solve_with_statistics(op, rhs, solution, options);
                    ASSERT_TRUE(statistics.converged);
                    EXPECT_EQ(statistics.iterations, 0);

                    // Retain the already exact first column while rejecting
                    // nonfinite and catastrophically bad remaining guesses.
                    for (std::size_t column = 1; column < columns; ++column)
                    {
                        auto values = solution.getDataNonConst(column);
                        std::fill(values.begin(), values.end(), column == 1
                            ? std::numeric_limits<double>::quiet_NaN() : -1.0e9 * value);
                    }
                    statistics = solver.solve_with_statistics(op, rhs, solution, options);
                    ASSERT_TRUE(statistics.converged);
                    for (std::size_t column = 0; column < columns; ++column)
                        for (const auto actual : solution.getData(column))
                            EXPECT_NEAR(actual, value, 1.0e-13);
                }
            }
        }
    }
}

TEST(BelosLinearSolverTest, VerifiedZeroBiCGStabHandlesStridedInactiveColumns)
{
    const auto matrix = block_matrix<3>({{
        {{4.0, -0.5, 0.0}}, {{-1.5, 5.0, -0.5}}, {{0.0, -1.5, 6.0}}}});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type rhs_storage(map, 5, true), solution_storage(map, 5, true);
    solution_storage.putScalar(55.0);
    const Teuchos::Array<std::size_t> rhs_columns{4, 0, 2}, solution_columns{3, 1, 4};
    auto rhs = rhs_storage.subViewNonConst(rhs_columns());
    auto solution = solution_storage.subViewNonConst(solution_columns());
    Pack::multi_vector_type exact(map, 3, true);
    for (const auto column : {0U, 2U})
    {
        auto values = exact.getDataNonConst(column);
        for (std::size_t row = 0; row < 3; ++row)
            values[row] = (column + 1.0) * (row + 1.0);
    }
    matrix->apply(exact, *rhs);
    solution->putScalar(std::numeric_limits<double>::quiet_NaN());
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    options.tolerance = 1.0e-12;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics = solver.solve_from_zero_with_statistics(matrix, *rhs, *solution, options);
    ASSERT_TRUE(statistics.converged);
    for (std::size_t column = 0; column < 3; ++column)
    {
        const auto expected = exact.getData(column);
        const auto actual = solution->getData(column);
        for (std::size_t row = 0; row < 3; ++row)
            EXPECT_NEAR(actual[row], expected[row], 1.0e-10);
    }
    for (const auto column : {0U, 2U})
        for (const auto value : solution_storage.getData(column))
            EXPECT_DOUBLE_EQ(value, 55.0);
}

TEST(BelosLinearSolverTest, RefinementReusesTheCheckedTrueResidualWithoutAnotherApply)
{
    const auto map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        3, 0, Tpetra::getDefaultComm()));
    const auto op = Teuchos::rcp(new IdentityOperator(map));
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        SimpleFluid::LinearSolverOptions options;
        options.backend = backend;
        options.tolerance = 1.0e-11;
        options.max_iterations = 8;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        op->reset_application_count();
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(op, rhs, solution, options).converged);
        const auto one_solve_calls = op->application_count();
        ResidualGap gap{1.0002 * options.tolerance};
        inject_residual_gap(solver, backend, gap);
        op->reset_application_count();
        const auto statistics = solver.solve_from_zero_with_statistics(op, rhs, solution, options);
        EXPECT_TRUE(statistics.converged);
        EXPECT_EQ(gap.solve_calls, 2);
        EXPECT_EQ(statistics.iterations, 2);
        // Each real Krylov solve has one final check. There is no extra
        // initial SpMV before either solve or between the two solves.
        EXPECT_EQ(op->application_count(), 2 * one_solve_calls);
    }
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

/** @brief Near-threshold recurrence drift is corrected at the original norm. */
TEST(BelosLinearSolverTest, RefinesNearThresholdTrueResidual)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0e-15);
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        for (const double norm_multiplier : {1.0, 2.0})
        {
            SCOPED_TRACE(norm_multiplier);
            SimpleFluid::LinearSolverOptions options;
            options.backend = backend;
            options.preconditioner = backend == SimpleFluid::LinearSolverBackend::Cg
                ? SimpleFluid::LinearPreconditioner::DIC
                : SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
            options.reuse_preconditioner = true;
            options.tolerance = 1.0e-11;
            options.max_iterations = 8;
            SimpleFluid::LinearResidualScaling scaling;
            scaling.rhs_norm_floor = norm_multiplier * rhs.norm2();
            SimpleFluid::BelosLinearSolver<Pack> solver;
            solution.putScalar(0.0);
            ASSERT_TRUE(solver.solve(matrix, rhs, solution, options));
            ResidualGap gap{norm_multiplier * 1.0002 * options.tolerance};
            inject_residual_gap(solver, backend, gap);
            solution.putScalar(0.0);

            const auto statistics = solver.solve_with_statistics(
                matrix, rhs, solution, options, scaling);

            EXPECT_TRUE(statistics.converged);
            EXPECT_EQ(gap.solve_calls, 2);
            EXPECT_EQ(statistics.iterations, 2);
            EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
            EXPECT_DOUBLE_EQ(statistics.rhs_norm, rhs.norm2());
            EXPECT_DOUBLE_EQ(statistics.achieved_tolerance,
                true_relative_residual(solver, matrix, rhs, solution, scaling));
            EXPECT_EQ(preconditioner_setup_count(solver), 1U);

            // A subsequent ordinary solve restores its original parameters.
            solution.putScalar(0.0);
            const auto next = solver.solve_with_statistics(
                matrix, rhs, solution, options, scaling);
            EXPECT_TRUE(next.converged);
            EXPECT_EQ(next.iterations, 1);
        }
    }
}

/** @brief Refinement cannot extend the iteration budget or retry indefinitely. */
TEST(BelosLinearSolverTest, BoundsTrueResidualRefinementWork)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        for (const int budget : {1, 2, 8})
        {
            SCOPED_TRACE(budget);
            SimpleFluid::LinearSolverOptions options;
            options.backend = backend;
            options.tolerance = 1.0e-11;
            options.max_iterations = budget;
            SimpleFluid::BelosLinearSolver<Pack> solver;
            solution.putScalar(0.0);
            ASSERT_TRUE(solver.solve(matrix, rhs, solution, options));
            ResidualGap gap{1.0002 * options.tolerance, 100};
            inject_residual_gap(solver, backend, gap);
            solution.putScalar(0.0);

            const auto statistics = solver.solve_with_statistics(
                matrix, rhs, solution, options);

            EXPECT_FALSE(statistics.converged);
            EXPECT_GT(statistics.achieved_tolerance, options.tolerance);
            EXPECT_EQ(gap.solve_calls, std::min(budget, 3));
            EXPECT_EQ(statistics.iterations, gap.solve_calls);
            EXPECT_LE(statistics.iterations, options.max_iterations);
        }
    }
}

/** @brief Large or non-finite residual failures retain their immediate exit. */
TEST(BelosLinearSolverTest, DoesNotRefineInvalidOrLargeResidualGap)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        for (const double error : {
                 1.0e-4, std::numeric_limits<double>::quiet_NaN()})
        {
            SimpleFluid::LinearSolverOptions options;
            options.backend = backend;
            options.tolerance = 1.0e-11;
            SimpleFluid::BelosLinearSolver<Pack> solver;
            solution.putScalar(0.0);
            ASSERT_TRUE(solver.solve(matrix, rhs, solution, options));
            ResidualGap gap{error};
            inject_residual_gap(solver, backend, gap);
            solution.putScalar(0.0);

            const auto statistics = solver.solve_with_statistics(
                matrix, rhs, solution, options);

            EXPECT_FALSE(statistics.converged);
            EXPECT_EQ(gap.solve_calls, 1);
            EXPECT_EQ(statistics.iterations, 1);
            EXPECT_GT(statistics.achieved_tolerance, options.tolerance);
        }
    }
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
        SimpleFluid::LinearPreconditioner::ILUT,
        SimpleFluid::LinearPreconditioner::GaussSeidel,
        SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel};
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
        SimpleFluid::LinearPreconditioner::ILUT,
        SimpleFluid::LinearPreconditioner::DIC,
        SimpleFluid::LinearPreconditioner::GaussSeidel,
        SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel};
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

/** @brief DIC preserves off-diagonals instead of computing general ILU factors. */
TEST(BelosLinearSolverTest, DICAppliesDiagonalIncompleteCholeskyInverse)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial local-order check; distributed DIC has separate MPI coverage.";

    const auto matrix = block_matrix<3>({{
        {{4.0, -1.0, -1.0}}, {{-1.0, 4.0, -1.0}}, {{-1.0, -1.0, 4.0}}}});
    SimpleFluid::detail::DICPreconditioner<Pack> inverse(*matrix);
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type rhs(map, 2, true);
    Pack::multi_vector_type result(map, 2, true);
    // The DIC factor product has (1,2) and (2,1) = -3/4, while A has -1.
    // Its product with [1,2,3] is [-1,19/4,19/2]. General ILU(0) gives a
    // different inverse for this dense graph, so this checks the distinction.
    for (std::size_t column = 0; column < 2; ++column)
    {
        auto values = rhs.getDataNonConst(column);
        const auto scale = static_cast<double>(column + 1);
        values[0] = -scale;
        values[1] = 4.75 * scale;
        values[2] = 9.5 * scale;
    }
    result.putScalar(std::numeric_limits<double>::quiet_NaN());
    inverse.apply(rhs, result);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto values = result.getData(column);
        for (std::size_t row = 0; row < 3; ++row)
            EXPECT_NEAR(values[row], (column + 1) * (row + 1), 1.0e-13);
    }
    inverse.apply(rhs, result, Teuchos::NO_TRANS, 2.0, 3.0);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto values = result.getData(column);
        for (std::size_t row = 0; row < 3; ++row)
            EXPECT_NEAR(values[row], 5.0 * (column + 1) * (row + 1), 1.0e-13);
    }
    inverse.apply(rhs, rhs);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto values = rhs.getData(column);
        for (std::size_t row = 0; row < 3; ++row)
            EXPECT_NEAR(values[row], (column + 1) * (row + 1), 1.0e-13);
    }
}

/** @brief Incorrect matrix symmetry and failed incomplete pivots fail closed. */
TEST(BelosLinearSolverTest, DICRejectsNonsymmetricAndNonpositiveMatrices)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial local-order check; distributed DIC has separate MPI coverage.";

    for (const auto entries : {
             std::array<std::array<double, 2>, 2>{{{{2.0, -1.0}}, {{0.0, 2.0}}}},
             std::array<std::array<double, 2>, 2>{{{{1.0, -2.0}}, {{-2.0, 1.0}}}},
             std::array<std::array<double, 2>, 2>{{{{0.0, 0.0}}, {{0.0, 1.0}}}}})
    {
        const auto matrix = block_matrix<2>(entries);
        EXPECT_THROW(
            SimpleFluid::detail::DICPreconditioner<Pack> inverse(*matrix),
            std::invalid_argument);
    }
}

/** @brief The existing Belos selector also supports distributed DIC factors. */
TEST(BelosLinearSolverTest, DICSolvesDistributedBlockMatrices)
{
    const auto matrix = block_matrix<2>({{{{2.0, -1.0}}, {{-1.0, 2.0}}}});
    Pack::vector_type rhs(matrix->getRowMap(), true);
    Pack::vector_type solution(matrix->getRowMap(), true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::DIC;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    EXPECT_TRUE(solver.solve(matrix, rhs, solution, options));
    EXPECT_LT(true_relative_residual(solver, matrix, rhs, solution), options.tolerance);
    EXPECT_EQ(preconditioner_setup_count(solver), 1U);
}

/** @brief A noncommuting SPD factor exercises the actual PCG recurrence. */
TEST(BelosLinearSolverTest, PCGWithDICSolvesSymmetricSystem)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial local-order check; distributed DIC has separate MPI coverage.";

    const auto matrix = block_matrix<4>({{
        {{4.0, -1.0, 0.0, -1.0}}, {{-1.0, 5.0, -1.0, 0.0}},
        {{0.0, -1.0, 6.0, -1.0}}, {{-1.0, 0.0, -1.0, 7.0}}}});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type exact(map, 2, true);
    for (std::size_t column = 0; column < 2; ++column)
    {
        auto values = exact.getDataNonConst(column);
        for (std::size_t row = 0; row < 4; ++row)
            values[row] = (column + 1) * (row + 1);
    }
    Pack::multi_vector_type rhs(map, 2, true);
    Pack::multi_vector_type solution(map, 2, true);
    matrix->apply(exact, rhs);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::DIC;
    options.max_iterations = 8;
    options.tolerance = 1.0e-12;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(statistics.converged);
    EXPECT_LE(statistics.iterations, 4);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto actual = solution.getData(column);
        const auto expected = exact.getData(column);
        for (std::size_t row = 0; row < 4; ++row)
            EXPECT_NEAR(actual[row], expected[row], 1.0e-11);
    }
}

/** @brief BiCGStab with GS handles nonsymmetric coupled rows and inactive RHS. */
TEST(BelosLinearSolverTest, BiCGStabGaussSeidelSolvesNonsymmetricMultipleRhs)
{
    const auto matrix = block_matrix<4>({{
        {{4.0, -0.5, 0.0, -0.25}}, {{-1.5, 5.0, -0.5, 0.0}},
        {{0.0, -1.5, 6.0, -0.5}}, {{-0.75, 0.0, -1.5, 7.0}}}});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type exact(map, 3, true);
    for (const std::size_t column : {0U, 2U})
    {
        auto values = exact.getDataNonConst(column);
        for (std::size_t row = 0; row < 4; ++row)
            values[row] = column == 0
                ? 1.0e-12 * (row + 1) : (row + 1) * (row + 1);
    }
    Pack::multi_vector_type rhs(map, 3, true);
    matrix->apply(exact, rhs);
    for (const auto preconditioner : {
             SimpleFluid::LinearPreconditioner::GaussSeidel,
             SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        SCOPED_TRACE(SimpleFluid::to_string(preconditioner));
        Pack::multi_vector_type solution(map, 3, true);
        solution.putScalar(7.0);
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        options.preconditioner = preconditioner;
        options.tolerance = 1.0e-12;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
        ASSERT_TRUE(statistics.converged);
        EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        for (std::size_t column = 0; column < 3; ++column)
        {
            const auto actual = solution.getData(column);
            const auto expected = exact.getData(column);
            for (std::size_t row = 0; row < 4; ++row)
                EXPECT_NEAR(actual[row], expected[row], column == 0 ? 1.0e-22 : 1.0e-10);
        }
    }
}

/** @brief Reused status tests refresh norms across many orders of RHS magnitude. */
TEST(BelosLinearSolverTest, ReusesBiCGStabForChangingMultipleRhsNorms)
{
    const auto matrix = block_matrix<4>({{
        {{4.0, -0.5, 0.0, -0.25}}, {{-1.5, 5.0, -0.5, 0.0}},
        {{0.0, -1.5, 6.0, -0.5}}, {{-0.75, 0.0, -1.5, 7.0}}}});
    const auto map = matrix->getRowMap();
    for (const auto preconditioner : {
             SimpleFluid::LinearPreconditioner::GaussSeidel,
             SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        options.preconditioner = preconditioner;
        options.tolerance = 1.0e-12;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        Pack::multi_vector_type exact(map, 3, true);
        Pack::multi_vector_type rhs(map, 3, true);
        Pack::multi_vector_type solution(map, 3, true);
        for (const double scale : {1.0, 1.0e-8, 1.0e8})
        {
            SCOPED_TRACE(scale);
            for (const std::size_t column : {0U, 2U})
            {
                auto values = exact.getDataNonConst(column);
                for (std::size_t row = 0; row < 4; ++row)
                    values[row] = scale * (column == 0
                        ? 1.0e-12 * (row + 1) : (row + 1) * (row + 1));
            }
            matrix->apply(exact, rhs);
            const auto statistics =
                solver.solve_with_statistics(matrix, rhs, solution, options);
            ASSERT_TRUE(statistics.converged);
            EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
        }
        EXPECT_EQ(preconditioner_setup_count(solver), 3U);
    }
}

/** @brief Verify that the relaxation names select one actual GS or SGS sweep. */
TEST(BelosLinearSolverTest, GaussSeidelAppliesSelectedSweep)
{
    const auto matrix = block_matrix<2>({{{{4.0, -1.0}}, {{-2.0, 3.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    for (const auto preconditioner : {
             SimpleFluid::LinearPreconditioner::GaussSeidel,
             SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        options.preconditioner = preconditioner;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        ASSERT_TRUE(solver.solve(matrix, rhs, solution, options));
        solution.putScalar(27.0);
        Access::preconditioner(solver)->apply(rhs, solution);
        const auto values = solution.getData();
        const bool symmetric = preconditioner
            == SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
        EXPECT_NEAR(values[0], symmetric ? 0.375 : 0.25, 1.0e-14);
        EXPECT_NEAR(values[1], 0.5, 1.0e-14);
    }
}

/** @brief Explicitly nonsymmetric inverse choices cannot be used with CG. */
TEST(BelosLinearSolverTest, CGRejectsNonsymmetricPreconditioners)
{
    const auto matrix = block_matrix<2>({{{{2.0, -1.0}}, {{-1.0, 2.0}}}});
    Pack::vector_type rhs(matrix->getRowMap(), true);
    Pack::vector_type solution(matrix->getRowMap(), true);
    rhs.putScalar(1.0);
    for (const auto preconditioner : {
             SimpleFluid::LinearPreconditioner::ILU0,
             SimpleFluid::LinearPreconditioner::ILUT,
             SimpleFluid::LinearPreconditioner::GaussSeidel})
    {
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::Cg;
        options.preconditioner = preconditioner;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        EXPECT_THROW(solver.solve(matrix, rhs, solution, options), std::invalid_argument);
        EXPECT_EQ(preconditioner_setup_count(solver), 0U);
    }
}

/** @brief New factors obey opt-in reuse, identity, kind, and explicit reset. */
TEST(BelosLinearSolverTest, ReusesAndInvalidatesDICAndGaussSeidelFactors)
{
    const auto first = block_matrix<2>({{{{4.0, -1.0}}, {{-1.0, 3.0}}}});
    const auto second = block_matrix<2>({{{{8.0, -1.0}}, {{-1.0, 6.0}}}});
    const auto map = first->getRowMap();
    Pack::vector_type rhs(map, true);
    Pack::vector_type solution(map, true);
    rhs.putScalar(1.0);
    for (const auto preconditioner : {
             SimpleFluid::LinearPreconditioner::DIC,
             SimpleFluid::LinearPreconditioner::GaussSeidel,
             SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        if (preconditioner == SimpleFluid::LinearPreconditioner::DIC
            && Tpetra::getDefaultComm()->getSize() != 1)
            continue;
        SCOPED_TRACE(SimpleFluid::to_string(preconditioner));
        SimpleFluid::LinearSolverOptions options;
        options.preconditioner = preconditioner;
        options.reuse_preconditioner = true;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        ASSERT_TRUE(solver.solve(first, rhs, solution, options));
        ASSERT_TRUE(solver.solve(first, rhs, solution, options));
        EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        ASSERT_TRUE(solver.solve(second, rhs, solution, options));
        EXPECT_EQ(preconditioner_setup_count(solver), 2U);
        solver.reset();
        ASSERT_TRUE(solver.solve(second, rhs, solution, options));
        EXPECT_EQ(preconditioner_setup_count(solver), 3U);
        options.reuse_preconditioner = false;
        ASSERT_TRUE(solver.solve(second, rhs, solution, options));
        EXPECT_EQ(preconditioner_setup_count(solver), 4U);
        options.preconditioner = SimpleFluid::LinearPreconditioner::None;
        ASSERT_TRUE(solver.solve(second, rhs, solution, options));
        options.preconditioner = preconditioner;
        options.reuse_preconditioner = true;
        ASSERT_TRUE(solver.solve(second, rhs, solution, options));
        EXPECT_EQ(preconditioner_setup_count(solver), 5U);
    }
}
