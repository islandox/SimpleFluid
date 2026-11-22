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
#include "solvers/DICPreconditioner.hh"
#include "solvers/ScaledLinearOperator.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

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

    static std::size_t configuration_count(const BelosLinearSolver<Pack>& solver)
    {
        return solver.d_muelu_configuration_count;
    }

    static Teuchos::RCP<const typename Pack::operator_type> preconditioner(
        const BelosLinearSolver<Pack>& solver)
    {
        return solver.d_preconditioner;
    }

    static bool has_original_problem_vectors(const BelosLinearSolver<Pack>& solver,
        const typename Pack::multi_vector_type& rhs, const typename Pack::multi_vector_type& solution)
    {
        return solver.d_problem->getLHS().get() == &solution
            && solver.d_problem->getRHS().get() == &rhs
            && solver.d_problem->getInitResVec().get() == solver.d_residual_workspace.get()
            && solver.d_problem->getInitPrecResVec().get() == solver.d_residual_workspace.get();
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

    static real_t accurate_relative_residual(
        const BelosLinearSolver<Pack>& solver,
        const Teuchos::RCP<const typename Pack::operator_type>& matrix,
        const typename Pack::multi_vector_type& rhs,
        const typename Pack::multi_vector_type& solution,
        LinearResidualScaling residual_scaling = {},
        real_t* maximum_rhs_norm = nullptr)
    {
        if (!solver.accurate_crs_residual(matrix, rhs, solution))
            throw std::logic_error("Accurate residual fixture requires a supported CRS operator.");
        return solver.residual_relative_norm(rhs, residual_scaling, maximum_rhs_norm);
    }

    static void set_batched_post_solve_hook(
        BelosLinearSolver<Pack>& solver,
        std::function<void(typename Pack::multi_vector_type&, std::size_t)> hook)
    {
        solver.d_batched_post_solve_hook = std::move(hook);
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

/** @brief Expose a CRS matrix only through its general operator interface. */
class ForwardingOperator final : public Pack::operator_type
{
public:
    explicit ForwardingOperator(Teuchos::RCP<const Pack::matrix_type> matrix)
        : d_matrix(std::move(matrix))
    {
    }

    Teuchos::RCP<const Pack::map_type> getDomainMap() const override
    {
        return d_matrix->getDomainMap();
    }

    Teuchos::RCP<const Pack::map_type> getRangeMap() const override
    {
        return d_matrix->getRangeMap();
    }

    void apply(
        const Pack::multi_vector_type& input,
        Pack::multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = 1.0,
        double beta = 0.0) const override
    {
        d_matrix->apply(input, output, mode, alpha, beta);
    }

private:
    Teuchos::RCP<const Pack::matrix_type> d_matrix;
};

/** @brief Reproduce recurrence/true-residual disagreement deterministically. */
struct ResidualGap
{
    double relative_error;
    int remaining_injections = 1;
    int solve_calls = 0;
    std::vector<double> rhs_norms{};
    int throw_on_call = 0;
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

/** @brief Reproduce GMRES LOA after a real Krylov solve, without relying on roundoff. */
class GmresResidualGapSolver final : public Belos::PseudoBlockGmresSolMgr<
    Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>
{
    using Manager = Belos::PseudoBlockGmresSolMgr<
        Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
public:
    GmresResidualGapSolver(
        const Teuchos::RCP<SimpleFluid::BelosLinearSolver<Pack>::problem_type>& problem,
        const Teuchos::RCP<Teuchos::ParameterList>& parameters,
        ResidualGap& gap, bool report_unconverged, bool report_loa)
        : Manager(problem, parameters), d_gap(gap),
          d_report_unconverged(report_unconverged), d_report_loa(report_loa) {}

    Belos::ReturnType solve() override
    {
        d_injected = false;
        Teuchos::Array<double> norms(this->getProblem().getRHS()->getNumVectors());
        this->getProblem().getRHS()->norm2(norms());
        d_gap.rhs_norms.push_back(*std::max_element(norms.begin(), norms.end()));
        if (d_gap.throw_on_call == d_gap.solve_calls + 1)
            throw std::runtime_error("Injected GMRES correction failure");
        const auto status = Manager::solve();
        ++d_gap.solve_calls;
        if (status == Belos::Converged && d_gap.remaining_injections > 0)
        {
            --d_gap.remaining_injections;
            if (d_reference_solution.is_null())
            {
                const auto solution = this->getProblem().getLHS();
                d_reference_solution = Teuchos::rcp(new Pack::multi_vector_type(
                    solution->getMap(), solution->getNumVectors()));
                d_reference_solution->update(1.0, *solution, 0.0);
            }
            // Keep the injected absolute error tied to the original problem.
            // Scaling a tiny correction itself would no longer model a
            // persistent gap in the final original-system residual.
            this->getProblem().getLHS()->update(-d_gap.relative_error, *d_reference_solution, 1.0);
            d_injected = true;
            return d_report_unconverged ? Belos::Unconverged : status;
        }
        return status;
    }

    bool isLOADetected() const override
    {
        return d_injected ? d_report_loa : Manager::isLOADetected();
    }

private:
    ResidualGap& d_gap;
    bool d_report_unconverged, d_report_loa, d_injected = false;
    Teuchos::RCP<Pack::multi_vector_type> d_reference_solution;
};

void inject_gmres_residual_gap(SimpleFluid::BelosLinearSolver<Pack>& solver,
                               ResidualGap& gap, bool unconverged = true, bool loa = true)
{
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    Access::replace_solver<GmresResidualGapSolver>(solver, gap, unconverged, loa);
}

/**
 * @brief Inject a scheduled perturbation after each converged GMRES solve.
 *
 * Each entry is added to the current problem LHS. In a batched refined pass the
 * LHS is the correction vector, so the next defect still models the scheduled
 * absolute gap; iteration counts remain independent of the injection mechanics.
 */
class ScheduledGmresGapSolver final : public Belos::PseudoBlockGmresSolMgr<
    Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>
{
    using Manager = Belos::PseudoBlockGmresSolMgr<
        Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
public:
    ScheduledGmresGapSolver(
        const Teuchos::RCP<SimpleFluid::BelosLinearSolver<Pack>::problem_type>& problem,
        const Teuchos::RCP<Teuchos::ParameterList>& parameters,
        std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors,
        int& solve_calls)
        : Manager(problem, parameters), d_errors(std::move(errors)),
          d_solve_calls(solve_calls)
    {
    }

    Belos::ReturnType solve() override
    {
        const auto status = Manager::solve();
        ++d_solve_calls;
        if (status == Belos::Converged && d_next < d_errors.size())
        {
            const auto& error = *d_errors[d_next++];
            auto lhs = this->getProblem().getLHS();
            if (lhs->getNumVectors() == error.getNumVectors())
            {
                lhs->update(1.0, error, 1.0);
            }
        }
        return status;
    }

private:
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> d_errors;
    std::size_t d_next = 0;
    int& d_solve_calls;
};

/** @brief Schedule gaps plus per-call loss-of-accuracy/failure reports. */
class ScheduledLoaGmresSolver final : public Belos::PseudoBlockGmresSolMgr<
    Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>
{
    using Manager = Belos::PseudoBlockGmresSolMgr<
        Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
public:
    ScheduledLoaGmresSolver(
        const Teuchos::RCP<SimpleFluid::BelosLinearSolver<Pack>::problem_type>& problem,
        const Teuchos::RCP<Teuchos::ParameterList>& parameters,
        std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors,
        std::vector<int> fail_calls,
        std::vector<int> loa_calls,
        int& solve_calls)
        : Manager(problem, parameters), d_errors(std::move(errors)),
          d_fail_calls(std::move(fail_calls)), d_loa_calls(std::move(loa_calls)),
          d_solve_calls(solve_calls)
    {
    }

    Belos::ReturnType solve() override
    {
        const auto status = Manager::solve();
        const int call = ++d_solve_calls;
        const bool fail = std::find(d_fail_calls.begin(), d_fail_calls.end(), call)
            != d_fail_calls.end();
        const bool loa = std::find(d_loa_calls.begin(), d_loa_calls.end(), call)
            != d_loa_calls.end();
        d_last_loa = loa;
        if (fail)
        {
            // Discard the correction so the affected column still needs repair.
            this->getProblem().getLHS()->putScalar(0.0);
            return Belos::Unconverged;
        }
        if (status == Belos::Converged && d_next < d_errors.size())
        {
            const auto& error = *d_errors[d_next++];
            auto lhs = this->getProblem().getLHS();
            if (lhs->getNumVectors() == error.getNumVectors())
            {
                lhs->update(1.0, error, 1.0);
            }
        }
        return status;
    }

    bool isLOADetected() const override { return d_last_loa; }

private:
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> d_errors;
    std::vector<int> d_fail_calls;
    std::vector<int> d_loa_calls;
    int& d_solve_calls;
    std::size_t d_next = 0;
    bool d_last_loa = false;
};

Teuchos::RCP<Pack::multi_vector_type> make_gap_errors(
    const Teuchos::RCP<const Pack::map_type>& map,
    const std::vector<std::vector<double>>& columns)
{
    auto errors = Teuchos::rcp(new Pack::multi_vector_type(map, columns.size(), true));
    for (std::size_t column = 0; column < columns.size(); ++column)
    {
        auto values = errors->getDataNonConst(column);
        std::copy(columns[column].begin(), columns[column].end(), values.begin());
    }
    return errors;
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

TEST(BelosLinearSolverTest, AccurateStoredCrsResidualPreservesCancellationAndOriginalScaling)
{
    // The exact product of the stored coefficients is 1, including the
    // middle coefficient lost by an ordinary left-to-right double sum.
    const auto matrix = block_matrix<3>({{
        {{1.0e16, 1.0, -1.0e16}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type rhs(map, 2, true), solution(map, 2, true);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const double scale = static_cast<double>(column + 1);
        auto x = solution.getDataNonConst(column);
        auto b = rhs.getDataNonConst(column);
        std::fill(x.begin(), x.end(), scale);
        std::fill(b.begin(), b.end(), scale);
    }
    SimpleFluid::BelosLinearSolver<Pack> solver;
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    EXPECT_DOUBLE_EQ(Access::accurate_relative_residual(solver, matrix, rhs, solution), 0.0);

    for (std::size_t column = 0; column < 2; ++column)
    {
        auto b = rhs.getDataNonConst(column);
        b[0] = 2.0 * static_cast<double>(column + 1);
    }
    // Per rank residuals are [1,0,0] and [2,0,0]; both RHS-relative
    // residuals equal 1/sqrt(6), independent of MPI rank count.
    double maximum_rhs_norm = 0.0;
    const double expected_rhs_norm = 2.0 * std::sqrt(6.0 * map->getComm()->getSize());
    EXPECT_NEAR(Access::accurate_relative_residual(
        solver, matrix, rhs, solution, {}, &maximum_rhs_norm), 1.0 / std::sqrt(6.0), 2.0e-15);
    EXPECT_NEAR(maximum_rhs_norm, expected_rhs_norm, 2.0e-14);

    SimpleFluid::LinearResidualScaling scaling;
    scaling.rhs_norm_floor = 2.0 * expected_rhs_norm;
    EXPECT_NEAR(Access::accurate_relative_residual(
        solver, matrix, rhs, solution, scaling), 0.5 / std::sqrt(6.0), 2.0e-15);

    // Reusing scratch storage must not retain the preceding nonzero residual.
    for (std::size_t column = 0; column < 2; ++column)
    {
        auto b = rhs.getDataNonConst(column);
        b[0] = static_cast<double>(column + 1);
    }
    EXPECT_DOUBLE_EQ(Access::accurate_relative_residual(solver, matrix, rhs, solution), 0.0);

    // Wrapping the same cancellation-sensitive CRS matrix in a scaled
    // operator must keep the compensated residual path active: the plain
    // matrix dynamic cast would silently disable it otherwise.
    for (const double crs_scale : {1.0, 1.0e-3})
    {
        SCOPED_TRACE(crs_scale);
        const auto scaled = Teuchos::rcp(
            new SimpleFluid::detail::ScaledLinearOperator<Pack>(matrix, crs_scale));
        // The executable must observe SimpleFluidSolvers' type information:
        // a locally emitted private copy would make this cast and the solver's
        // internal unwrapping disagree under hidden-visibility LTO.
        const Teuchos::RCP<const Pack::operator_type> scaled_as_operator = scaled;
        ASSERT_FALSE(Teuchos::rcp_dynamic_cast<
            const SimpleFluid::detail::ScaledLinearOperator<Pack>>(scaled_as_operator, false).is_null());
        Pack::multi_vector_type scaled_rhs(map, 2, true), scaled_solution(map, 2, true);
        for (std::size_t column = 0; column < 2; ++column)
        {
            const double column_scale = static_cast<double>(column + 1);
            auto x = scaled_solution.getDataNonConst(column);
            auto b = scaled_rhs.getDataNonConst(column);
            std::fill(x.begin(), x.end(), column_scale);
            std::fill(b.begin(), b.end(), crs_scale * column_scale);
        }
        EXPECT_DOUBLE_EQ(Access::accurate_relative_residual(
            solver, scaled, scaled_rhs, scaled_solution), 0.0);

        for (std::size_t column = 0; column < 2; ++column)
        {
            auto b = scaled_rhs.getDataNonConst(column);
            b[0] = 2.0 * crs_scale * static_cast<double>(column + 1);
        }
        // Residual and RHS norms both scale with crs_scale, so the
        // RHS-relative residual keeps the existing accuracy criterion.
        double scaled_maximum_rhs_norm = 0.0;
        const double scaled_expected_rhs_norm =
            2.0 * crs_scale * std::sqrt(6.0 * map->getComm()->getSize());
        EXPECT_NEAR(Access::accurate_relative_residual(
            solver, scaled, scaled_rhs, scaled_solution, {},
            &scaled_maximum_rhs_norm), 1.0 / std::sqrt(6.0), 2.0e-15);
        EXPECT_NEAR(scaled_maximum_rhs_norm, scaled_expected_rhs_norm,
            2.0e-14 * std::max(crs_scale, 1.0));

        SimpleFluid::LinearResidualScaling scaled_scaling;
        scaled_scaling.rhs_norm_floor = 2.0 * scaled_expected_rhs_norm;
        EXPECT_NEAR(Access::accurate_relative_residual(
            solver, scaled, scaled_rhs, scaled_solution, scaled_scaling),
            0.5 / std::sqrt(6.0), 2.0e-15);
    }
}

TEST(BelosLinearSolverTest, AccurateStoredCrsResidualImportsAndRefreshesRemoteColumnValues)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 3, 0, comm));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 3));
    const int next_rank = (comm->getRank() + 1) % comm->getSize();
    const auto next_begin = static_cast<Pack::global_ordinal_type>(3 * next_rank);
    const Teuchos::Array<Pack::global_ordinal_type> remote_columns{
        next_begin, next_begin + 1, next_begin + 2};
    const Teuchos::Array<double> cancellation_values{1.0e16, 1.0, -1.0e16};
    matrix->insertGlobalValues(map->getGlobalElement(0), remote_columns(), cancellation_values());
    for (const Pack::local_ordinal_type row : {1, 2})
    {
        const Teuchos::Array<Pack::global_ordinal_type> column{map->getGlobalElement(row)};
        const Teuchos::Array<double> value{1.0};
        matrix->insertGlobalValues(map->getGlobalElement(row), column(), value());
    }
    matrix->fillComplete();
    if (comm->getSize() > 1)
        ASSERT_GT(matrix->getColMap()->getLocalNumElements(), map->getLocalNumElements());

    Pack::vector_type rhs(map, true), solution(map, true);
    SimpleFluid::BelosLinearSolver<Pack> solver;
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    for (const double offset : {0.0, 4.0})
    {
        SCOPED_TRACE(offset);
        {
            auto x = solution.getDataNonConst(0);
            x[0] = x[2] = 1.0 + comm->getRank() + offset;
            x[1] = 2.0 + comm->getRank() + offset;
            auto b = rhs.getDataNonConst(0);
            b[0] = 2.0 + next_rank + offset;
            b[1] = x[1];
            b[2] = x[2];
        }
        EXPECT_DOUBLE_EQ(Access::accurate_relative_residual(solver, matrix, rhs, solution), 0.0);
        {
            auto b = rhs.getDataNonConst(0);
            b[0] += 1.0;
        }
        const double expected = std::sqrt(static_cast<double>(comm->getSize())) / rhs.norm2();
        EXPECT_NEAR(Access::accurate_relative_residual(solver, matrix, rhs, solution), expected, 2.0e-15);
    }
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

TEST(BelosLinearSolverTest, RefinesGmresLossOfAccuracyAgainstOriginalTrueResidualGate)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1e-15);
    for (const bool loa : {false, true})
        for (const double norm_multiplier : {1.0, 2.0})
        {
            SCOPED_TRACE(loa);
            SCOPED_TRACE(norm_multiplier);
            SimpleFluid::LinearSolverOptions options;
            options.backend = SimpleFluid::LinearSolverBackend::Gmres;
            options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
            options.reuse_preconditioner = true;
            options.tolerance = 1e-14;
            options.max_iterations = 8;
            SimpleFluid::LinearResidualScaling scaling{norm_multiplier * rhs.norm2()};
            SimpleFluid::BelosLinearSolver<Pack> solver;
            ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options, scaling).converged);
            const auto preconditioner =
                SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::preconditioner(solver);
            // Exceeds the old 2x eligibility bound. Exercise both a reported
            // convergence gap and an Unconverged result specifically due to LOA.
            ResidualGap gap{norm_multiplier * 2.705 * options.tolerance};
            inject_gmres_residual_gap(solver, gap, loa, loa);
            const auto statistics = solver.solve_from_zero_with_statistics(
                matrix, rhs, solution, options, scaling);
            EXPECT_TRUE(statistics.converged);
            EXPECT_EQ(gap.solve_calls, 2);
            EXPECT_EQ(statistics.iterations, 2);
            EXPECT_LE(statistics.iterations, options.max_iterations);
            ASSERT_EQ(gap.rhs_norms.size(), 2U);
            EXPECT_DOUBLE_EQ(gap.rhs_norms[0], rhs.norm2());
            EXPECT_GT(gap.rhs_norms[1], options.tolerance * scaling.rhs_norm_floor);
            EXPECT_LT(gap.rhs_norms[1], 8 * options.tolerance * scaling.rhs_norm_floor);
            EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
            EXPECT_DOUBLE_EQ(statistics.achieved_tolerance,
                true_relative_residual(solver, matrix, rhs, solution, scaling));
            EXPECT_DOUBLE_EQ(statistics.rhs_norm, rhs.norm2());
            EXPECT_EQ(preconditioner_setup_count(solver), 1U);
            EXPECT_EQ(SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::preconditioner(solver).get(),
                      preconditioner.get());
            EXPECT_TRUE(SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::has_original_problem_vectors(
                solver, rhs, solution));
            // The caller's normal parameters and RHS remain usable after the
            // stricter internal restart target and injected LOA have cleared.
            const auto next = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options, scaling);
            EXPECT_TRUE(next.converged);
            EXPECT_EQ(next.iterations, 1);
            EXPECT_LE(next.achieved_tolerance, options.tolerance);
            EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        }
}

TEST(BelosLinearSolverTest, GmresLossOfAccuracyRefinementKeepsIterationAndRestartBounds)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    for (const int budget : {1, 2, 8})
    {
        SCOPED_TRACE(budget);
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::Gmres;
        // Keep every deliberately persistent injection above roundoff;
        // this case tests the total iteration and four-correction bounds.
        options.tolerance = 1e-11;
        options.max_iterations = budget;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
        ResidualGap gap{2.705 * options.tolerance, 100};
        inject_gmres_residual_gap(solver, gap);
        const auto statistics = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);
        EXPECT_FALSE(statistics.converged);
        EXPECT_GT(statistics.achieved_tolerance, options.tolerance);
        EXPECT_EQ(gap.solve_calls, std::min(budget, 5));
        EXPECT_EQ(statistics.iterations, gap.solve_calls);
        EXPECT_LE(statistics.iterations, options.max_iterations);
        EXPECT_DOUBLE_EQ(statistics.achieved_tolerance,
            true_relative_residual(solver, matrix, rhs, solution));
    }
}

TEST(BelosLinearSolverTest, GmresDoesNotRetryOrdinaryFailureLargeGapOrNonfiniteResidual)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    for (const auto [factor, loa] : std::array<std::pair<double, bool>, 3>{{
             {2.705, false}, {1.0e12, true}, {std::numeric_limits<double>::quiet_NaN(), true}}})
    {
        SCOPED_TRACE(factor);
        SCOPED_TRACE(loa);
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::Gmres;
        options.tolerance = 1e-14;
        options.max_iterations = 8;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
        ResidualGap gap{factor * options.tolerance};
        inject_gmres_residual_gap(solver, gap, true, loa);
        const auto statistics = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);
        EXPECT_FALSE(statistics.converged);
        EXPECT_EQ(gap.solve_calls, 1);
        EXPECT_EQ(statistics.iterations, 1);
        EXPECT_GT(statistics.achieved_tolerance, options.tolerance);
    }
}

TEST(BelosLinearSolverTest, GmresDefectCorrectionRestoresOriginalVectorsAfterException)
{
    const auto matrix = block_matrix<3>({{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}});
    const auto map = matrix->getRowMap();
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = 1e-14;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
    ResidualGap gap{2.705 * options.tolerance};
    gap.throw_on_call = 2;
    inject_gmres_residual_gap(solver, gap);
    EXPECT_THROW((void)solver.solve_from_zero_with_statistics(matrix, rhs, solution, options), std::runtime_error);
    EXPECT_TRUE(SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::has_original_problem_vectors(
        solver, rhs, solution));
    gap.throw_on_call = 0;
    EXPECT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
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

/** @brief Krylov solves retain the full operator with a separate ILUT matrix. */
TEST(BelosLinearSolverTest, SeparatePreconditioningMatrixUsesOriginalOperator)
{
    const auto matrix = block_matrix<2>({{{{4.0, 1.0}}, {{2.0, 3.0}}}});
    const auto compact = block_matrix<2>({{{{4.0, 0.0}}, {{0.0, 3.0}}}});
    const auto op = Teuchos::rcp(new ForwardingOperator(matrix));
    const auto map = matrix->getRowMap();
    Pack::vector_type exact(map, true);
    {
        auto values = exact.getDataNonConst();
        values[0] = 1.0;
        values[1] = -2.0;
    }
    Pack::vector_type rhs(map, true);
    op->apply(exact, rhs);
    for (const auto backend : {SimpleFluid::LinearSolverBackend::BiCGStab,
             SimpleFluid::LinearSolverBackend::Gmres})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        SimpleFluid::LinearSolverOptions options;
        options.backend = backend;
        options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
        options.tolerance = 1.0e-12;
        Pack::vector_type solution(map, true);
        solution.putScalar(0.25);
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto statistics = solver.solve_with_statistics(
            op, rhs, solution, options, {}, compact);
        ASSERT_TRUE(statistics.converged);
        EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
        EXPECT_LE(true_relative_residual(solver, op, rhs, solution), options.tolerance);
        EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        const auto values = solution.getData();
        EXPECT_NEAR(values[0], 1.0, 1.0e-11);
        EXPECT_NEAR(values[1], -2.0, 1.0e-11);
    }
}

/** @brief Separate ILUT setup survives per-column solves and both initial-guess paths. */
TEST(BelosLinearSolverTest, SeparatePreconditioningMatrixSupportsBiCGStabMultipleRhs)
{
    const auto matrix = block_matrix<2>({{{{4.0, 1.0}}, {{2.0, 3.0}}}});
    const auto compact = block_matrix<2>({{{{4.0, 0.0}}, {{0.0, 3.0}}}});
    const auto op = Teuchos::rcp(new ForwardingOperator(matrix));
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type exact(map, 3, true);
    {
        auto first = exact.getDataNonConst(0);
        first[0] = 1.0;
        first[1] = -2.0;
        auto third = exact.getDataNonConst(2);
        third[0] = -3.0;
        third[1] = 0.5;
    }
    Pack::multi_vector_type rhs(map, 3, true);
    op->apply(exact, rhs);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
    options.tolerance = 1.0e-12;
    for (const bool zero_initial_guess : {false, true})
    {
        SCOPED_TRACE(zero_initial_guess);
        Pack::multi_vector_type solution(map, 3, true);
        solution.putScalar(0.25);
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto statistics = zero_initial_guess
            ? solver.solve_from_zero_with_statistics(op, rhs, solution, options, {}, compact)
            : solver.solve_with_statistics(op, rhs, solution, options, {}, compact);
        ASSERT_TRUE(statistics.converged);
        EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
        EXPECT_LE(true_relative_residual(solver, op, rhs, solution), options.tolerance);
        EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        for (std::size_t column = 0; column < 3; ++column)
        {
            const auto actual = solution.getData(column);
            const auto expected = exact.getData(column);
            for (std::size_t row = 0; row < 2; ++row)
                EXPECT_NEAR(actual[row], expected[row], 1.0e-11);
        }
    }
}

/** @brief Incompatible preconditioning maps are rejected before factor setup. */
TEST(BelosLinearSolverTest, SeparatePreconditioningMatrixRejectsIncompatibleMaps)
{
    const auto matrix = block_matrix<2>({{{{4.0, 1.0}}, {{2.0, 3.0}}}});
    const auto incompatible = block_matrix<3>({{
        {{4.0, 0.0, 0.0}}, {{0.0, 3.0, 0.0}}, {{0.0, 0.0, 2.0}}}});
    const auto op = Teuchos::rcp(new ForwardingOperator(matrix));
    Pack::vector_type rhs(matrix->getRowMap(), true);
    Pack::vector_type solution(matrix->getRowMap(), true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    EXPECT_THROW(solver.solve_with_statistics(op, rhs, solution, options, {}, incompatible),
        std::invalid_argument);
    EXPECT_EQ(preconditioner_setup_count(solver), 0U);
    const auto compact = block_matrix<2>({{{{4.0, 0.0}}, {{0.0, 3.0}}}});
    Pack::vector_type incompatible_vector(incompatible->getRowMap(), true);
    EXPECT_THROW(solver.solve_with_statistics(
        op, incompatible_vector, solution, options, {}, compact), std::invalid_argument);
    EXPECT_THROW(solver.solve_from_zero_with_statistics(
        op, rhs, incompatible_vector, options, {}, compact), std::invalid_argument);
    EXPECT_EQ(preconditioner_setup_count(solver), 0U);
}

/** @brief Reuse follows the supplied matrix identity, selection, and value invalidation. */
TEST(BelosLinearSolverTest, SeparatePreconditioningMatrixControlsReuse)
{
    const auto matrix = block_matrix<2>({{{{4.0, 1.0}}, {{2.0, 3.0}}}});
    const auto first_compact = block_matrix<2>({{{{4.0, 0.0}}, {{0.0, 3.0}}}});
    const auto second_compact = block_matrix<2>({{{{5.0, 0.0}}, {{0.0, 2.0}}}});
    const auto first_op = Teuchos::rcp(new ForwardingOperator(matrix));
    const auto second_op = Teuchos::rcp(new ForwardingOperator(matrix));
    Pack::vector_type rhs(matrix->getRowMap(), true);
    Pack::vector_type solution(matrix->getRowMap(), true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
    options.reuse_preconditioner = true;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    ASSERT_TRUE(solver.solve_with_statistics(
        first_op, rhs, solution, options, {}, first_compact).converged);
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve_with_statistics(
        second_op, rhs, solution, options, {}, first_compact).converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 1U);
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve_with_statistics(
        second_op, rhs, solution, options, {}, second_compact).converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 2U);
    options.preconditioner = SimpleFluid::LinearPreconditioner::Jacobi;
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve_with_statistics(
        second_op, rhs, solution, options, {}, second_compact).converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 3U);
    solver.notify_operator_values_changed();
    solution.putScalar(0.0);
    ASSERT_TRUE(solver.solve_with_statistics(
        second_op, rhs, solution, options, {}, second_compact).converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 4U);
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

/** @brief In-place numeric updates rebuild factors while maps remain compatible. */
TEST(BelosLinearSolverTest, ChangedOperatorValuesInvalidateReusedPreconditioners)
{
    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(invalid_global_size, 8, 0, Tpetra::getDefaultComm()));
    for (const auto kind : {SimpleFluid::LinearPreconditioner::MueLu,
             SimpleFluid::LinearPreconditioner::Jacobi, SimpleFluid::LinearPreconditioner::DIC})
    {
        auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
        auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix);
        Pack::vector_type rhs(map, true);
        Pack::vector_type solution(map, true);
        rhs.putScalar(3.0);
        SimpleFluid::LinearSolverOptions options;
        options.preconditioner = kind;
        options.reuse_preconditioner = true;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(op, rhs, solution, options).converged);
        EXPECT_EQ(preconditioner_setup_count(solver), 1U);
        matrix->scale(2.0);
        solver.notify_operator_values_changed();
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(op, rhs, solution, options).converged);
        EXPECT_EQ(preconditioner_setup_count(solver), 2U);
        const auto values = solution.getData(0);
        for (const auto value : values) EXPECT_NEAR(value, 1.5, 1.0e-12);
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(op, rhs, solution, options).converged);
        EXPECT_EQ(preconditioner_setup_count(solver), 2U);
        if (kind == SimpleFluid::LinearPreconditioner::MueLu)
        {
            EXPECT_EQ(SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::configuration_count(solver), 1U);
            solver.reset();
            ASSERT_TRUE(solver.solve_from_zero_with_statistics(op, rhs, solution, options).converged);
            EXPECT_EQ(SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::configuration_count(solver), 2U);
        }
    }
}

TEST(BelosLinearSolverTest, CachedMueLuConfigurationMatchesFreshMultilevelSetup)
{
    using GO = Pack::global_ordinal_type;
    auto map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 128, 0, Tpetra::getDefaultComm()));
    SimpleFluid::BelosLinearSolver<Pack> cached;
    SimpleFluid::LinearSolverOptions options;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
    options.tolerance = 1e-12;
    Pack::vector_type rhs(map), actual(map), expected(map), cached_action(map), fresh_action(map);
    rhs.putScalar(1.0);
    for (int generation = 0; generation < 3; ++generation)
    {
        SCOPED_TRACE(generation);
        auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 3));
        for (size_t row = 0; row < map->getLocalNumElements(); ++row)
        {
            const GO gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(row));
            Teuchos::Array<GO> columns{gid};
            Teuchos::Array<double> values{2.5 + 0.1 * generation + 0.001 * gid};
            if (gid > 0) { columns.push_back(gid - 1); values.push_back(-1.0); }
            if (gid + 1 < static_cast<GO>(map->getGlobalNumElements()) && (generation != 2 || gid % 3))
            { columns.push_back(gid + 1); values.push_back(-1.0); }
            matrix->insertGlobalValues(gid, columns(), values());
        }
        matrix->fillComplete();
        SimpleFluid::BelosLinearSolver<Pack> fresh;
        ASSERT_TRUE(cached.solve_from_zero_with_statistics(matrix, rhs, actual, options).converged);
        ASSERT_TRUE(fresh.solve_from_zero_with_statistics(matrix, rhs, expected, options).converged);
        using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
        Access::preconditioner(cached)->apply(rhs, cached_action);
        Access::preconditioner(fresh)->apply(rhs, fresh_action);
        const auto actual_values = actual.getData(), expected_values = expected.getData();
        const auto actual_action = cached_action.getData(), expected_action = fresh_action.getData();
        for (size_t row = 0; row < map->getLocalNumElements(); ++row)
        {
            EXPECT_DOUBLE_EQ(actual_values[row], expected_values[row]);
            EXPECT_DOUBLE_EQ(actual_action[row], expected_action[row]);
        }
        EXPECT_EQ(Access::configuration_count(cached), 1U);
        EXPECT_EQ(preconditioner_setup_count(cached), static_cast<size_t>(generation + 1));
    }
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
    EXPECT_EQ(statistics.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(statistics.max_batch_width, 2U);
    EXPECT_LT(statistics.operator_apply_calls, statistics.operator_column_applies);
    for (std::size_t column = 0; column < 2; ++column)
    {
        EXPECT_LE(statistics.column_statistics[column].iterations, 4);
    }
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto actual = solution.getData(column);
        const auto expected = exact.getData(column);
        for (std::size_t row = 0; row < 4; ++row)
            EXPECT_NEAR(actual[row], expected[row], 1.0e-11);
    }

    // ColumnSerial policy remains available and produces identical solution.
    options.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
    Pack::multi_vector_type serial_solution(map, 2, true);
    const auto serial_stats = solver.solve_with_statistics(matrix, rhs, serial_solution, options);
    ASSERT_TRUE(serial_stats.converged);
    EXPECT_EQ(serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
    EXPECT_EQ(serial_stats.max_batch_width, 1U);
    for (std::size_t column = 0; column < 2; ++column)
    {
        const auto actual = serial_solution.getData(column);
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

/** Exercise multiple AMG levels and cross-rank edges, not only a coarse solve. */
TEST(BelosLinearSolverTest, SymmetricSmoothedAmgIsPositiveSymmetricAndSupportsCg)
{
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    auto map = Teuchos::rcp(new Pack::map_type(512, 0, Tpetra::getDefaultComm()));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 3));
    for (size_t i = 0; i < map->getLocalNumElements(); ++i)
    {
        const auto gid = map->getGlobalElement(i);
        Teuchos::Array<Pack::global_ordinal_type> columns;
        Teuchos::Array<Pack::scalar_type> values;
        if (gid > 0) { columns.push_back(gid - 1); values.push_back(-1.0); }
        columns.push_back(gid); values.push_back(2.01 + .001 * (gid % 13));
        if (gid < 511) { columns.push_back(gid + 1); values.push_back(-1.0); }
        matrix->insertGlobalValues(gid, columns(), values());
    }
    matrix->fillComplete();
    Pack::vector_type x(map, true), y(map, true), bx(map, true), by(map, true);
    {
        auto xv = x.getDataNonConst(0), yv = y.getDataNonConst(0);
        for (size_t i = 0; i < map->getLocalNumElements(); ++i)
        {
            const double gid = map->getGlobalElement(i);
            xv[i] = std::sin(.13 * gid) + .3;
            yv[i] = std::cos(.07 * gid) - .2;
        }
    }
    Pack::vector_type rhs(map, true), solution(map, true);
    matrix->apply(x, rhs);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLuSymmetric;
    options.tolerance = 1e-12;
    options.reuse_preconditioner = true;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto stats = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(stats.converged);
    EXPECT_LE(stats.achieved_tolerance, options.tolerance);
    const auto inverse = Access::preconditioner(solver);
    using AMG = MueLu::TpetraOperator<Pack::scalar_type, Pack::local_ordinal_type,
        Pack::global_ordinal_type, Pack::node_type>;
    const auto amg = Teuchos::rcp_dynamic_cast<const AMG>(inverse, true);
    EXPECT_GT(amg->GetHierarchy()->GetNumLevels(), 1);
    inverse->apply(x, bx); inverse->apply(y, by);
    EXPECT_GT(x.dot(bx), 0.0);
    EXPECT_GT(y.dot(by), 0.0);
    EXPECT_NEAR(x.dot(by), y.dot(bx), 1e-11 * x.norm2() * by.norm2());
    Pack::vector_type sum(map, true), bsum(map, true);
    sum.update(1.0, x, 1.0, y, 0.0);
    inverse->apply(sum, bsum);
    bsum.update(-1.0, bx, -1.0, by, 1.0);
    EXPECT_LE(bsum.norm2(), 1e-12 * (bx.norm2() + by.norm2()));
    ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
    EXPECT_EQ(Access::preconditioner_setup_count(solver), 1U);
    EXPECT_EQ(Access::configuration_count(solver), 1U);
    // The same matrix/map must not retain the other MueLu policy's factory.
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
    ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
    EXPECT_EQ(Access::configuration_count(solver), 2U);
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLuSymmetric;
    ASSERT_TRUE(solver.solve_from_zero_with_statistics(matrix, rhs, solution, options).converged);
    EXPECT_EQ(Access::configuration_count(solver), 3U);
    EXPECT_EQ(SimpleFluid::parse_linear_preconditioner("MueLuSymmetric"), options.preconditioner);
}

TEST(BelosLinearSolverTest, SymmetricSmoothedAmgSupportsEmptyRanks)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2) GTEST_SKIP() << "Requires an empty MPI rank.";
    auto map = Teuchos::rcp(new Pack::map_type(1, 0, comm));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
    if (map->getLocalNumElements())
    {
        const Pack::global_ordinal_type column = 0;
        const Pack::scalar_type value = 2.0;
        matrix->insertGlobalValues(0, 1, &value, &column);
    }
    matrix->fillComplete();
    Pack::vector_type rhs(map, true), solution(map, true);
    rhs.putScalar(1.0);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLuSymmetric;
    options.tolerance = 1e-12;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto stats = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(stats.converged);
    EXPECT_LE(stats.achieved_tolerance, options.tolerance);
    EXPECT_NEAR(solution.norm2(), .5, 1e-12);
}

/**
 * @brief Absolute acceptance requires every RHS column to pass; a passing
 *        column must not mask a failing one.
 */
TEST(BelosLinearSolverTest, MixedColumnAbsoluteAcceptanceRequiresEveryColumn)
{
    // Each rank owns an independent three-row block. Column 0 has a zero
    // RHS and remains exactly converged, while column 1 has a large RHS
    // that one GMRES iteration cannot solve, so its true residual stays
    // well above the absolute tolerance.
    const auto matrix = block_matrix<3>({{
        {{4.0, -0.5, 0.0}}, {{-1.5, 5.0, -0.5}}, {{0.0, -1.5, 6.0}}}});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type rhs(map, 2, true);
    Pack::multi_vector_type solution(map, 2, true);
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(row);
        rhs.replaceLocalValue(lid, 0, 0.0);
        rhs.replaceLocalValue(lid, 1, 1.0e3);
    }

    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = 1.0e-12;
    options.absolute_tolerance = 1.0e-12;
    options.max_iterations = 1;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics =
        solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);

    // The exactly-converged column 0 must not mask the failing column 1.
    // The rank-zero failure report includes both columns' RHS and true
    // residual norms.
    EXPECT_FALSE(statistics.converged);
    EXPECT_GT(statistics.achieved_tolerance, options.tolerance);
}

/** @brief A nonfinite RHS column never takes the zero-guess shortcut. */
TEST(BelosLinearSolverTest, NonfiniteRhsNeverTakesZeroShortcutWithAbsoluteTolerance)
{
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto map = Teuchos::rcp(new Pack::map_type(
        invalid_global_size, 3, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    for (const double poison : {
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()})
    {
        SCOPED_TRACE(poison);
        Pack::multi_vector_type rhs(map, 2, true);
        Pack::multi_vector_type solution(map, 2, true);
        {
            auto values = rhs.getDataNonConst(1);
            std::fill(values.begin(), values.end(), poison);
        }
        solution.putScalar(std::numeric_limits<double>::quiet_NaN());

        SimpleFluid::LinearSolverOptions options;
        options.absolute_tolerance = 1.0;
        options.max_iterations = 8;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        // std::max(0.0, NaN) returns 0.0, so the old shortcut wrongly
        // reported convergence for a NaN RHS without entering Krylov. The
        // solve must now run bounded Krylov work and fail closed.
        const auto statistics = solver.solve_from_zero_with_statistics(
            matrix, rhs, solution, options);

        EXPECT_FALSE(statistics.converged);
    }
}

namespace
{

class WidthRecordingOperator final : public Pack::operator_type
{
public:
    explicit WidthRecordingOperator(Teuchos::RCP<const Pack::operator_type> target)
        : d_target(std::move(target))
    {
    }

    Teuchos::RCP<const Pack::map_type> getDomainMap() const override
    {
        return d_target->getDomainMap();
    }

    Teuchos::RCP<const Pack::map_type> getRangeMap() const override
    {
        return d_target->getRangeMap();
    }

    bool hasTransposeApply() const override
    {
        return d_target->hasTransposeApply();
    }

    void apply(
        const Pack::multi_vector_type& input,
        Pack::multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = 1.0,
        double beta = 0.0) const override
    {
        d_recorded_widths.push_back(static_cast<std::size_t>(input.getNumVectors()));
        d_target->apply(input, output, mode, alpha, beta);
    }

    const std::vector<std::size_t>& recorded_widths() const noexcept
    {
        return d_recorded_widths;
    }

    void clear_recorded_widths() const noexcept
    {
        d_recorded_widths.clear();
    }

private:
    Teuchos::RCP<const Pack::operator_type> d_target;
    mutable std::vector<std::size_t> d_recorded_widths;
};

Teuchos::RCP<Pack::matrix_type> distributed_diagonal_matrix(
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
    const std::vector<double>& diagonal)
{
    auto map = Teuchos::rcp(new Pack::map_type(diagonal.size(), 0, comm));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
    for (std::size_t i = 0; i < map->getLocalNumElements(); ++i)
    {
        const auto gid = map->getGlobalElement(i);
        const double value = diagonal.at(static_cast<std::size_t>(gid));
        matrix->insertGlobalValues(gid, Teuchos::arrayView(&gid, 1), Teuchos::arrayView(&value, 1));
    }
    matrix->fillComplete();
    return matrix;
}

Teuchos::RCP<Pack::matrix_type> distributed_tridiagonal_matrix(
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
    const Pack::global_ordinal_type global_rows,
    const bool symmetric)
{
    auto map = Teuchos::rcp(new Pack::map_type(global_rows, 0, comm));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 3));
    for (std::size_t i = 0; i < map->getLocalNumElements(); ++i)
    {
        const auto gid = map->getGlobalElement(i);
        Teuchos::Array<Pack::global_ordinal_type> cols;
        Teuchos::Array<double> vals;
        if (gid > 0)
        {
            cols.push_back(gid - 1);
            vals.push_back(symmetric ? -1.0 : -1.2);
        }
        cols.push_back(gid);
        vals.push_back(4.0 + 0.05 * static_cast<double>(gid % 7));
        if (gid + 1 < global_rows)
        {
            cols.push_back(gid + 1);
            vals.push_back(symmetric ? -1.0 : -0.6);
        }
        matrix->insertGlobalValues(gid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

} // namespace

/** @brief Prove PseudoBlockGmres genuinely batches multi-column apply and re-sets Block Size. */
TEST(BelosLinearSolverTest, GmresResetsBlockSizeAcrossConsecutiveSolvesAndRefinements)
{
    const auto matrix = distributed_tridiagonal_matrix(Tpetra::getDefaultComm(), 24, false);
    const auto op = Teuchos::rcp(new WidthRecordingOperator(matrix));
    const auto map = matrix->getRowMap();

    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = 1.0e-12;
    options.max_iterations = 100;
    SimpleFluid::BelosLinearSolver<Pack> solver;

    for (const std::size_t cols : {1U, 4U, 2U, 5U})
    {
        SCOPED_TRACE(cols);
        Pack::multi_vector_type exact(map, cols, true);
        Pack::multi_vector_type rhs(map, cols, true);
        Pack::multi_vector_type solution(map, cols, true);
        for (std::size_t c = 0; c < cols; ++c)
        {
            auto x = exact.getDataNonConst(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                const auto gid = map->getGlobalElement(r);
                x[r] = static_cast<double>(c + 1) * (1.0 + 0.1 * static_cast<double>(gid));
            }
        }
        matrix->apply(exact, rhs);

        op->clear_recorded_widths();
        const auto stats = solver.solve_from_zero_with_statistics(op, rhs, solution, options);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.column_statistics.size(), cols);
        EXPECT_GT(stats.operator_apply_calls, 0U);
        EXPECT_EQ(stats.max_batch_width, cols);
        EXPECT_GE(stats.operator_column_applies, cols * stats.operator_apply_calls);

        if (cols > 1)
        {
            EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
            EXPECT_EQ(stats.batch_krylov_iterations, stats.iterations);
            EXPECT_EQ(stats.column_serial_iterations, 0);
            const auto& widths = op->recorded_widths();
            EXPECT_TRUE(std::any_of(widths.begin(), widths.end(),
                [cols](std::size_t w) { return w == cols; }));
        }
        else
        {
            EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::SingleRhs);
        }

        for (std::size_t c = 0; c < cols; ++c)
        {
            EXPECT_TRUE(stats.column_statistics[c].converged);
            EXPECT_LE(stats.column_statistics[c].achieved_tolerance, options.tolerance);
            const auto actual = solution.getData(c);
            const auto expected = exact.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-10);
            }
        }
    }

    // Also prove Block Size > 1 is re-applied during GMRES defect-correction refinement.
    const std::size_t ref_cols = 3;
    Pack::multi_vector_type ref_rhs(map, ref_cols, true);
    Pack::multi_vector_type ref_sol(map, ref_cols, true);
    for (std::size_t c = 0; c < ref_cols; ++c)
    {
        ref_rhs.getVectorNonConst(c)->putScalar(static_cast<double>(c + 1));
    }
    auto id_matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    auto id_op = Teuchos::rcp(new WidthRecordingOperator(id_matrix));

    SimpleFluid::BelosLinearSolver<Pack> ref_solver;
    SimpleFluid::LinearSolverOptions ref_options;
    ref_options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    ref_options.tolerance = 1.0e-10;
    ref_options.max_iterations = 20;
    ASSERT_TRUE(ref_solver.solve_from_zero_with_statistics(id_op, ref_rhs, ref_sol, ref_options).converged);

    ResidualGap gap{2.0 * ref_options.tolerance, 1};
    inject_gmres_residual_gap(ref_solver, gap, true, true);
    id_op->clear_recorded_widths();
    const auto ref_stats = ref_solver.solve_from_zero_with_statistics(id_op, ref_rhs, ref_sol, ref_options);
    ASSERT_TRUE(ref_stats.converged);
    EXPECT_EQ(gap.solve_calls, 2);
    EXPECT_EQ(ref_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(ref_stats.max_batch_width, ref_cols);
    const std::size_t full_width_applies = std::count(
        id_op->recorded_widths().begin(), id_op->recorded_widths().end(), ref_cols);
    EXPECT_GE(full_width_applies, 4U);
    for (std::size_t c = 0; c < ref_cols; ++c)
    {
        EXPECT_GE(ref_stats.column_statistics[c].refinement_passes, 1);
    }
}

/** @brief Verify batch-width chunking invariance and BiCGStab ColumnSerial reporting. */
TEST(BelosLinearSolverTest, BatchWidthChunkingInvarianceAndExecutionModes)
{
    const auto matrix = distributed_tridiagonal_matrix(Tpetra::getDefaultComm(), 20, false);
    const auto map = matrix->getRowMap();
    const std::size_t cols = 5;

    Pack::multi_vector_type exact(map, cols, true);
    Pack::multi_vector_type rhs(map, cols, true);
    for (std::size_t c = 0; c < cols; ++c)
    {
        auto x = exact.getDataNonConst(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            x[r] = std::pow(10.0, static_cast<int>(c) - 2) * (1.0 + static_cast<double>(r));
        }
    }
    matrix->apply(exact, rhs);

    std::vector<double> baseline;
    for (const int batch_width : {0, 1, 2, 3, 5})
    {
        SCOPED_TRACE(batch_width);
        Pack::multi_vector_type solution(map, cols, true);
        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::Gmres;
        options.tolerance = 1.0e-12;
        options.max_iterations = 200;
        options.max_batch_width = batch_width;
        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_from_zero_with_statistics(matrix, rhs, solution, options);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.column_statistics.size(), cols);

        if (batch_width == 1)
        {
            EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
            EXPECT_EQ(stats.max_batch_width, 1U);
            EXPECT_EQ(stats.batch_krylov_iterations, 0);
            EXPECT_EQ(stats.column_serial_iterations, stats.iterations);
        }
        else
        {
            EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
            const std::size_t expected_max = batch_width == 0 ? cols : static_cast<std::size_t>(batch_width);
            EXPECT_EQ(stats.max_batch_width, expected_max);
            EXPECT_EQ(stats.batch_krylov_iterations, stats.iterations);
            EXPECT_EQ(stats.column_serial_iterations, 0);
        }

        if (baseline.empty())
        {
            for (std::size_t c = 0; c < cols; ++c)
            {
                const auto s = solution.getData(c);
                baseline.insert(baseline.end(), s.begin(), s.end());
            }
        }
        else
        {
            std::size_t idx = 0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                const auto s = solution.getData(c);
                for (std::size_t r = 0; r < map->getLocalNumElements(); ++r, ++idx)
                {
                    EXPECT_NEAR(s[r], baseline[idx], 1.0e-11 * std::max(1.0, std::abs(baseline[idx])));
                }
            }
        }
    }

    // BiCGStab now batches cleanly with max_batch_width > 1 while preserving
    // selectable ColumnSerial reference mode with matching solutions.
    Pack::multi_vector_type bicg_solution(map, cols, true);
    SimpleFluid::LinearSolverOptions bicg_options;
    bicg_options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    bicg_options.tolerance = 1.0e-12;
    bicg_options.max_iterations = 200;
    bicg_options.max_batch_width = 4;
    SimpleFluid::BelosLinearSolver<Pack> bicg_solver;
    const auto bicg_stats = bicg_solver.solve_from_zero_with_statistics(
        matrix, rhs, bicg_solution, bicg_options);
    ASSERT_TRUE(bicg_stats.converged);
    EXPECT_EQ(bicg_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(bicg_stats.max_batch_width, 4U);
    EXPECT_EQ(bicg_stats.batch_krylov_iterations, bicg_stats.iterations);
    EXPECT_EQ(bicg_stats.column_serial_iterations, 0);
    EXPECT_LT(bicg_stats.operator_apply_calls, bicg_stats.operator_column_applies);
    EXPECT_EQ(bicg_stats.column_statistics.size(), cols);

    Pack::multi_vector_type bicg_serial_solution(map, cols, true);
    bicg_options.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
    const auto bicg_serial_stats = bicg_solver.solve_from_zero_with_statistics(
        matrix, rhs, bicg_serial_solution, bicg_options);
    ASSERT_TRUE(bicg_serial_stats.converged);
    EXPECT_EQ(bicg_serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
    EXPECT_EQ(bicg_serial_stats.max_batch_width, 1U);
    EXPECT_EQ(bicg_serial_stats.batch_krylov_iterations, 0);
    EXPECT_EQ(bicg_serial_stats.column_serial_iterations, bicg_serial_stats.iterations);
    for (std::size_t c = 0; c < cols; ++c)
    {
        const auto batched_col = bicg_solution.getData(c);
        const auto serial_col = bicg_serial_solution.getData(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            EXPECT_NEAR(batched_col[r], serial_col[r], 1.0e-11 * std::max(1.0, std::abs(serial_col[r])));
        }
    }
}

/** @brief Batched per-column counters must equal the column-serial convergence work. */
TEST(BelosLinearSolverTest, BatchedColumnIterationsMatchColumnSerialConvergence)
{
    auto map = Teuchos::rcp(new Pack::map_type(3, 0, Tpetra::getDefaultComm()));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        const auto gid = map->getGlobalElement(r);
        const double value = static_cast<double>(gid + 1);
        matrix->insertGlobalValues(gid, Teuchos::arrayView(&gid, 1), Teuchos::arrayView(&value, 1));
    }
    matrix->fillComplete();

    Pack::multi_vector_type rhs(map, 2, true);
    rhs.putScalar(1.0);
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        if (map->getGlobalElement(r) != 0)
        {
            rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 0, 0.0);
        }
    }

    SimpleFluid::MultiRhsSolveOptions options;
    options.solver.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.solver.tolerance = 1.0e-12;
    options.solver.max_iterations = 10;

    Pack::multi_vector_type batched_solution(map, 2, true);
    SimpleFluid::BelosLinearSolver<Pack> batched_solver;
    const auto batched = batched_solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs, batched_solution, options);
    ASSERT_TRUE(batched.converged);
    ASSERT_EQ(batched.column_statistics.size(), 2U);
    EXPECT_EQ(batched.column_statistics[0].iterations, 1);
    EXPECT_EQ(batched.column_statistics[1].iterations, 3);

    options.solver.max_batch_width = 1;
    Pack::multi_vector_type serial_solution(map, 2, true);
    SimpleFluid::BelosLinearSolver<Pack> serial_solver;
    const auto serial = serial_solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs, serial_solution, options);
    ASSERT_TRUE(serial.converged);
    ASSERT_EQ(serial.column_statistics.size(), 2U);
    for (std::size_t col = 0; col < 2; ++col)
    {
        EXPECT_EQ(batched.column_statistics[col].iterations, serial.column_statistics[col].iterations);
    }

    // The same manager was first solved one column at a time; widening the batch
    // must keep the recorder installed and still report scalar-equivalent counts.
    options.solver.max_batch_width = 0;
    Pack::multi_vector_type widened_solution(map, 2, true);
    const auto widened = serial_solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs, widened_solution, options);
    ASSERT_TRUE(widened.converged);
    ASSERT_EQ(widened.column_statistics.size(), 2U);
    EXPECT_EQ(widened.column_statistics[0].iterations, 1);
    EXPECT_EQ(widened.column_statistics[1].iterations, 3);
}

/** @brief Pass records reset and accumulate against independent scalar runs. */
TEST(BelosLinearSolverTest, GmresRefinementPassesResetAndAccumulatePerColumn)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 2.0, 3.0});
    const auto map = matrix->getRowMap();
    const double tolerance = 1.0e-12;
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = tolerance;
    options.max_iterations = 20;

    Pack::multi_vector_type rhs(map, 2, true);
    rhs.putScalar(1.0);
    const double scale = 4.0 * tolerance * std::sqrt(3.0 / 14.0);
    const auto make_batch_error = [&](double sign)
    {
        auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 2, true));
        auto column0 = error->getDataNonConst(0);
        auto column1 = error->getDataNonConst(1);
        for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
        {
            const auto gid = map->getGlobalElement(row);
            column0[row] = sign * (gid == 0 ? scale : 0.0);
            column1[row] = sign * scale;
        }
        return error;
    };
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors{
        make_batch_error(1.0), make_batch_error(-1.0)};

    struct PolicyVariant
    {
        const char* name;
        bool use_floor;
        bool use_absolute;
    };
    const std::vector<PolicyVariant> variants{
        PolicyVariant{"none", false, false},
        PolicyVariant{"floor", true, false},
        PolicyVariant{"absolute", false, true}};
    std::array<int, 2> baseline_iterations{};
    for (std::size_t variant_index = 0; variant_index < variants.size(); ++variant_index)
    {
        const auto& variant = variants[variant_index];
        SCOPED_TRACE(variant.name);
        SimpleFluid::MultiRhsSolveOptions solve_options;
        solve_options.solver = options;
        if (variant.use_floor || variant.use_absolute)
        {
            solve_options.column_policies.resize(2);
            for (auto& policy : solve_options.column_policies)
            {
                if (variant.use_floor)
                {
                    policy.rhs_norm_floor = 1.0;
                }
                else
                {
                    policy.absolute_tolerance = 1.0e-12;
                }
            }
        }

        SimpleFluid::BelosLinearSolver<Pack> solver;
        Pack::multi_vector_type solution(map, 2, true);
        ASSERT_TRUE(solver.solve_multi_rhs_with_statistics(
            matrix, rhs, solution, solve_options).converged);
        solution.putScalar(0.0);
        int solve_calls = 0;
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
            ScheduledGmresGapSolver>(solver, errors, solve_calls);
        const auto batched = solver.solve_multi_rhs_with_statistics(
            matrix, rhs, solution, solve_options);
        ASSERT_TRUE(batched.converged);
        EXPECT_GE(solve_calls, 3);
        ASSERT_EQ(batched.column_statistics.size(), 2U);
        EXPECT_GE(batched.column_statistics[0].refinement_passes, 2);
        EXPECT_GE(batched.column_statistics[1].refinement_passes, 2);
        if (variant_index == 0)
        {
            baseline_iterations[0] = batched.column_statistics[0].iterations;
            baseline_iterations[1] = batched.column_statistics[1].iterations;
        }
        else
        {
            // Original floors/absolute tolerances govern final acceptance, not
            // the correction solve; they must not change recorded Krylov work.
            EXPECT_EQ(batched.column_statistics[0].iterations, baseline_iterations[0]);
            EXPECT_EQ(batched.column_statistics[1].iterations, baseline_iterations[1]);
        }

        // Each column solved alone with the same scheduled gaps must report
        // the same accumulated Krylov count and reset-pass count.
        for (std::size_t column = 0; column < 2; ++column)
        {
            SCOPED_TRACE(column);
            Pack::multi_vector_type single_rhs(map, 1, true);
            {
                const auto source = rhs.getData(column);
                auto destination = single_rhs.getDataNonConst(0);
                std::copy(source.begin(), source.end(), destination.begin());
            }
            std::vector<Teuchos::RCP<Pack::multi_vector_type>> single_errors;
            for (const double sign : {1.0, -1.0})
            {
                auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 1, true));
                auto values = error->getDataNonConst(0);
                for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
                {
                    const auto gid = map->getGlobalElement(row);
                    values[row] = sign * (column == 0
                        ? (gid == 0 ? scale : 0.0)
                        : scale);
                }
                single_errors.push_back(error);
            }

            auto scalar_options = solve_options;
            if (!scalar_options.column_policies.empty())
            {
                scalar_options.column_policies = {solve_options.column_policies[column]};
            }
            SimpleFluid::BelosLinearSolver<Pack> scalar_solver;
            Pack::multi_vector_type scalar_solution(map, 1, true);
            ASSERT_TRUE(scalar_solver.solve_multi_rhs_with_statistics(
                matrix, single_rhs, scalar_solution, scalar_options).converged);
            scalar_solution.putScalar(0.0);
            int scalar_calls = 0;
            SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
                ScheduledGmresGapSolver>(scalar_solver, single_errors, scalar_calls);
            const auto scalar = scalar_solver.solve_multi_rhs_with_statistics(
                matrix, single_rhs, scalar_solution, scalar_options);
            ASSERT_TRUE(scalar.converged);
            ASSERT_EQ(scalar.column_statistics.size(), 1U);
            EXPECT_EQ(scalar.column_statistics[0].refinement_passes,
                batched.column_statistics[column].refinement_passes);
            EXPECT_EQ(scalar.column_statistics[0].iterations,
                batched.column_statistics[column].iterations);
        }
    }
}

/** @brief A slow column cannot consume another column's repair allowance. */
TEST(BelosLinearSolverTest, GmresRefinementUsesEachColumnsRemainingBudget)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 2.0, 3.0});
    const auto map = matrix->getRowMap();
    const double tolerance = 1.0e-12;
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = tolerance;
    options.max_iterations = 4;

    Pack::multi_vector_type rhs(map, 2, true);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 0, gid == 0 ? 1.0 : 0.0);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 1, 1.0);
    }

    // Column 0's correction needs more than the aggregate remainder would
    // leave after column 1's slower initial solve.
    const double scale = 4.0 * tolerance / std::sqrt(14.0);
    auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 2, true));
    auto error_column = error->getDataNonConst(0);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        error_column[row] = scale;
    }

    SimpleFluid::BelosLinearSolver<Pack> solver;
    Pack::multi_vector_type solution(map, 2, true);
    ASSERT_TRUE(solver.solve_with_statistics(matrix, rhs, solution, options).converged);
    solution.putScalar(0.0);
    int solve_calls = 0;
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors{error};
    SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
        ScheduledGmresGapSolver>(solver, errors, solve_calls);
    const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(statistics.converged);
    EXPECT_EQ(solve_calls, 2);
    ASSERT_EQ(statistics.column_statistics.size(), 2U);

    EXPECT_EQ(statistics.column_statistics[1].iterations, 3);
    EXPECT_EQ(statistics.column_statistics[1].refinement_passes, 0);
    EXPECT_GT(statistics.column_statistics[0].iterations, 1);
    EXPECT_LE(statistics.column_statistics[0].iterations, options.max_iterations);
    EXPECT_EQ(statistics.column_statistics[0].refinement_passes, 1);

    // The accepted column remains the exact initial solution.
    const auto accepted = solution.getData(1);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        const double expected = gid == 0 ? 1.0 : (gid == 1 ? 0.5 : 1.0 / 3.0);
        EXPECT_NEAR(accepted[row], expected, 1.0e-12);
    }
}

/** @brief Different remaining allowances form separate refinement groups. */
TEST(BelosLinearSolverTest, GmresRefinementGroupsColumnsByRemainingBudget)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 2.0, 3.0});
    const auto map = matrix->getRowMap();
    const double tolerance = 1.0e-12;
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = tolerance;
    options.max_iterations = 6;

    Pack::multi_vector_type rhs(map, 2, true);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 0, gid == 0 ? 1.0 : 0.0);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 1, 1.0);
    }

    const double scale0 = 4.0 * tolerance / std::sqrt(14.0);
    const double scale1 = 4.0 * tolerance * std::sqrt(3.0);
    auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 2, true));
    auto error0 = error->getDataNonConst(0);
    auto error1 = error->getDataNonConst(1);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        error0[row] = scale0;
        error1[row] = gid == 0 ? scale1 : 0.0;
    }

    SimpleFluid::BelosLinearSolver<Pack> solver;
    Pack::multi_vector_type solution(map, 2, true);
    ASSERT_TRUE(solver.solve_with_statistics(matrix, rhs, solution, options).converged);
    solution.putScalar(0.0);
    int solve_calls = 0;
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors{error};
    SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
        ScheduledGmresGapSolver>(solver, errors, solve_calls);
    const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(statistics.converged);
    EXPECT_EQ(solve_calls, 3); // one initial solve plus two differently budgeted groups
    ASSERT_EQ(statistics.column_statistics.size(), 2U);
    EXPECT_EQ(statistics.column_statistics[0].refinement_passes, 1);
    EXPECT_EQ(statistics.column_statistics[1].refinement_passes, 1);
    EXPECT_LE(statistics.column_statistics[0].iterations, options.max_iterations);
    EXPECT_LE(statistics.column_statistics[1].iterations, options.max_iterations);

    const auto column0 = solution.getData(0);
    const auto column1 = solution.getData(1);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        const double expected1 = gid == 0 ? 1.0 : (gid == 1 ? 0.5 : 1.0 / 3.0);
        EXPECT_NEAR(column1[row], expected1, 1.0e-12);
        EXPECT_NEAR(column0[row], gid == 0 ? 1.0 : 0.0, 1.0e-12);
    }
}

/** @brief A later group must not erase an earlier group's LOA retry right. */
TEST(BelosLinearSolverTest, GmresRefinementRetriesLoaGroupAfterLaterGroupSucceeds)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 2.0, 3.0});
    const auto map = matrix->getRowMap();
    const double tolerance = 1.0e-12;
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = tolerance;
    options.max_iterations = 20;

    Pack::multi_vector_type rhs(map, 2, true);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 0, gid == 0 ? 1.0 : 0.0);
        rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 1, 1.0);
    }
    const double scale0 = 4.0 * tolerance / std::sqrt(14.0);
    const double scale1 = 4.0 * tolerance * std::sqrt(3.0);
    auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 2, true));
    auto error0 = error->getDataNonConst(0);
    auto error1 = error->getDataNonConst(1);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        error0[row] = scale0;
        error1[row] = gid == 0 ? scale1 : 0.0;
    }

    SimpleFluid::BelosLinearSolver<Pack> solver;
    Pack::multi_vector_type solution(map, 2, true);
    ASSERT_TRUE(solver.solve_with_statistics(matrix, rhs, solution, options).converged);
    solution.putScalar(0.0);
    int solve_calls = 0;
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors{error};
    SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
        ScheduledLoaGmresSolver>(solver, errors, std::vector<int>{2},
        std::vector<int>{2}, solve_calls);
    const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(statistics.converged);
    EXPECT_EQ(solve_calls, 4);
    ASSERT_EQ(statistics.column_statistics.size(), 2U);
    EXPECT_EQ(statistics.column_statistics[0].refinement_passes, 1);
    EXPECT_EQ(statistics.column_statistics[1].refinement_passes, 2);
    EXPECT_LE(statistics.column_statistics[0].iterations, options.max_iterations);
    EXPECT_LE(statistics.column_statistics[1].iterations, options.max_iterations);

    const auto column0 = solution.getData(0);
    const auto column1 = solution.getData(1);
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(row);
        const double expected1 = gid == 0 ? 1.0 : (gid == 1 ? 0.5 : 1.0 / 3.0);
        EXPECT_NEAR(column1[row], expected1, 1.0e-12);
        EXPECT_NEAR(column0[row], gid == 0 ? 1.0 : 0.0, 1.0e-12);
    }
}

/** @brief Asymmetric gaps must not lend one column's strict correction policy. */
TEST(BelosLinearSolverTest, GmresAsymmetricGapAccountingMatchesScalarColumns)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 2.0, 3.0});
    const auto map = matrix->getRowMap();
    const double tolerance = 1.0e-12;
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = tolerance;
    options.max_iterations = 6;

    Pack::multi_vector_type rhs(map, 2, true);
    rhs.putScalar(1.0);
    const double error_scale = std::sqrt(3.0 / 14.0);
    const double small_error = 2.0e-12 * error_scale;
    const double large_error = 1.0e-8 * error_scale;
    auto error = Teuchos::rcp(new Pack::multi_vector_type(map, 2, true));
    error->getVectorNonConst(0)->putScalar(small_error);
    error->getVectorNonConst(1)->putScalar(large_error);

    SimpleFluid::BelosLinearSolver<Pack> solver;
    Pack::multi_vector_type solution(map, 2, true);
    ASSERT_TRUE(solver.solve_with_statistics(matrix, rhs, solution, options).converged);
    solution.putScalar(0.0);
    int solve_calls = 0;
    std::vector<Teuchos::RCP<Pack::multi_vector_type>> errors{error};
    SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
        ScheduledGmresGapSolver>(solver, errors, solve_calls);
    const auto batched = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(batched.converged);
    ASSERT_EQ(batched.column_statistics.size(), 2U);

    for (std::size_t column = 0; column < 2; ++column)
    {
        SCOPED_TRACE(column);
        Pack::multi_vector_type single_rhs(map, 1, true);
        single_rhs.putScalar(1.0);
        auto single_error = Teuchos::rcp(new Pack::multi_vector_type(map, 1, true));
        single_error->putScalar(column == 0 ? small_error : large_error);
        std::vector<Teuchos::RCP<Pack::multi_vector_type>> single_errors{single_error};

        SimpleFluid::BelosLinearSolver<Pack> scalar_solver;
        Pack::multi_vector_type scalar_solution(map, 1, true);
        ASSERT_TRUE(scalar_solver.solve_with_statistics(
            matrix, single_rhs, scalar_solution, options).converged);
        scalar_solution.putScalar(0.0);
        int scalar_calls = 0;
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<
            ScheduledGmresGapSolver>(scalar_solver, single_errors, scalar_calls);
        const auto scalar = scalar_solver.solve_with_statistics(
            matrix, single_rhs, scalar_solution, options);
        ASSERT_TRUE(scalar.converged);
        ASSERT_EQ(scalar.column_statistics.size(), 1U);
        EXPECT_EQ(batched.column_statistics[column].iterations,
            scalar.column_statistics[0].iterations);
    }
    // The small-gap column needs two correction iterations at the loose
    // tolerance; the large-gap column needs the stricter tolerance.
    EXPECT_EQ(batched.column_statistics[0].iterations, 5);
    EXPECT_EQ(batched.column_statistics[1].iterations, 6);
}

/** @brief Recorder mapping remains valid when a rank owns no cells. */
TEST(BelosLinearSolverTest, GmresColumnIterationsOnEmptyRank)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {2.0});
    const auto map = matrix->getRowMap();
    Pack::multi_vector_type rhs(map, 2, true);
    rhs.putScalar(1.0);
    Pack::multi_vector_type solution(map, 2, true);

    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    options.tolerance = 1.0e-12;
    options.max_iterations = 10;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
    ASSERT_TRUE(statistics.converged);
    ASSERT_EQ(statistics.column_statistics.size(), 2U);
    EXPECT_EQ(statistics.column_statistics[0].iterations, 1);
    EXPECT_EQ(statistics.column_statistics[1].iterations, 1);
}

/** @brief Verify per-column true residual prevents Frobenius masking of trace species and honors active mask. */
TEST(BelosLinearSolverTest, PerColumnTrueResidualRejectsTraceMaskingAndHonorsActiveMask)
{
    const auto matrix = distributed_tridiagonal_matrix(Tpetra::getDefaultComm(), 16, false);
    const auto map = matrix->getRowMap();

    Pack::multi_vector_type exact(map, 4, true);
    Pack::multi_vector_type rhs(map, 4, true);
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        const auto gid = static_cast<double>(map->getGlobalElement(r));
        exact.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 0, 1.0e8 * (1.0 + gid));
        exact.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 1, 0.0);
        exact.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 2, 1.0e-10 * (1.0 + 0.5 * gid));
        exact.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 3, 1.0e6 * (2.0 + gid));
    }
    matrix->apply(exact, rhs);

    // Tight per-column tolerance on trace col 2 with only 1 iteration: col 0/3 can have loose
    // tolerance and pass, while col 2 fails — Frobenius norm must NOT mask col 2!
    {
        Pack::multi_vector_type solution(map, 4, true);
        SimpleFluid::MultiRhsSolveOptions opts;
        opts.solver.backend = SimpleFluid::LinearSolverBackend::Gmres;
        opts.solver.tolerance = 1.0e-1;
        opts.solver.max_iterations = 1;
        opts.column_policies.resize(4);
        opts.column_policies[0].tolerance = 1.0;
        opts.column_policies[1].tolerance = 1.0;
        opts.column_policies[2].tolerance = 1.0e-12;
        opts.column_policies[3].tolerance = 1.0;
        opts.active_columns = {true, false, true, true};

        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(matrix, rhs, solution, opts);
        EXPECT_FALSE(stats.converged);
        ASSERT_EQ(stats.column_statistics.size(), 4U);
        EXPECT_TRUE(stats.column_statistics[1].skipped_zero);
        EXPECT_TRUE(stats.column_statistics[1].converged);
        EXPECT_FALSE(stats.column_statistics[2].converged);
        EXPECT_GT(stats.column_statistics[2].achieved_tolerance, 1.0e-12);
    }

    // Full convergence with poisoned inactive col 1 zeroed and trace col 2 solved accurately.
    {
        Pack::multi_vector_type solution(map, 4, true);
        solution.getVectorNonConst(1)->putScalar(999.0);
        SimpleFluid::MultiRhsSolveOptions opts;
        opts.solver.backend = SimpleFluid::LinearSolverBackend::Gmres;
        opts.solver.tolerance = 1.0e-12;
        opts.solver.max_iterations = 200;
        opts.active_columns = {true, false, true, true};

        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(matrix, rhs, solution, opts);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        EXPECT_EQ(stats.max_batch_width, 3U);
        EXPECT_TRUE(stats.column_statistics[1].skipped_zero);
        for (const auto val : solution.getData(1))
        {
            EXPECT_DOUBLE_EQ(val, 0.0);
        }
        for (const std::size_t col : {0U, 2U, 3U})
        {
            EXPECT_FALSE(stats.column_statistics[col].skipped_zero);
            EXPECT_LE(stats.column_statistics[col].achieved_tolerance, 1.0e-12);
            const auto actual = solution.getData(col);
            const auto expected = exact.getData(col);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-10 * std::abs(expected[r]));
            }
        }

        if (Tpetra::getDefaultComm()->getSize() > 1)
        {
            opts.active_columns[1] = (Tpetra::getDefaultComm()->getRank() == 0);
            EXPECT_THROW(
                solver.solve_multi_rhs_from_zero_with_statistics(matrix, rhs, solution, opts),
                std::invalid_argument);
        }
    }
}

/** @brief Verify CG batches multi-RHS by default and preserves ColumnSerial parity. */
TEST(BelosLinearSolverTest, CgSolvesMultiRhsColumnSeriallyAndReportsExecutionMode)
{
    const auto matrix = distributed_tridiagonal_matrix(Tpetra::getDefaultComm(), 24, true);
    const auto op = Teuchos::rcp(new WidthRecordingOperator(matrix));
    const auto map = matrix->getRowMap();
    const std::size_t cols = 4;

    Pack::multi_vector_type exact(map, cols, true);
    Pack::multi_vector_type rhs(map, cols, true);
    Pack::multi_vector_type solution(map, cols, true);
    for (std::size_t c = 0; c < cols; ++c)
    {
        auto x = exact.getDataNonConst(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const auto gid = static_cast<double>(map->getGlobalElement(r));
            x[r] = static_cast<double>(c + 1) * (0.5 + 0.05 * gid);
        }
    }
    matrix->apply(exact, rhs);

    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::Jacobi;
    options.tolerance = 1.0e-12;
    options.max_iterations = 100;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto stats = solver.solve_from_zero_with_statistics(
        op, rhs, solution, options, {}, matrix);
    ASSERT_TRUE(stats.converged);
    EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(stats.max_batch_width, cols);
    EXPECT_EQ(stats.batch_krylov_iterations, stats.iterations);
    EXPECT_EQ(stats.column_serial_iterations, 0);
    EXPECT_LT(stats.operator_apply_calls, stats.operator_column_applies);
    EXPECT_TRUE(std::any_of(op->recorded_widths().begin(), op->recorded_widths().end(),
        [cols](std::size_t w) { return w == cols; }));
    for (std::size_t c = 0; c < cols; ++c)
    {
        EXPECT_LE(stats.column_statistics[c].achieved_tolerance, options.tolerance);
        const auto actual = solution.getData(c);
        const auto expected = exact.getData(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            EXPECT_NEAR(actual[r], expected[r], 1.0e-10);
        }
    }

    op->clear_recorded_widths();
    options.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
    Pack::multi_vector_type serial_solution(map, cols, true);
    const auto serial_stats = solver.solve_from_zero_with_statistics(
        op, rhs, serial_solution, options, {}, matrix);
    ASSERT_TRUE(serial_stats.converged);
    EXPECT_EQ(serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
    EXPECT_EQ(serial_stats.max_batch_width, 1U);
    EXPECT_EQ(serial_stats.batch_krylov_iterations, 0);
    EXPECT_EQ(serial_stats.column_serial_iterations, serial_stats.iterations);
    EXPECT_TRUE(std::all_of(op->recorded_widths().begin(), op->recorded_widths().end(),
        [](std::size_t w) { return w == 1; }));
    for (std::size_t c = 0; c < cols; ++c)
    {
        const auto actual = serial_solution.getData(c);
        const auto expected = solution.getData(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            EXPECT_NEAR(actual[r], expected[r], 1.0e-11);
        }
    }
}

/** @brief Unequal-speed CG columns retain exact per-column iteration counts in batched & serial modes. */
TEST(BelosLinearSolverTest, CgColumnsKeepIndependentIterationCounts)
{
    auto map = Teuchos::rcp(new Pack::map_type(3, 0, Tpetra::getDefaultComm()));
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        const auto gid = map->getGlobalElement(r);
        const double value = static_cast<double>(gid + 1);
        matrix->insertGlobalValues(gid, Teuchos::arrayView(&gid, 1), Teuchos::arrayView(&value, 1));
    }
    matrix->fillComplete();

    Pack::multi_vector_type rhs(map, 2, true);
    rhs.putScalar(1.0);
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        if (map->getGlobalElement(r) != 0)
        {
            rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(r), 0, 0.0);
        }
    }

    SimpleFluid::MultiRhsSolveOptions options;
    options.solver.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.solver.tolerance = 1.0e-12;
    options.solver.max_iterations = 10;

    Pack::multi_vector_type solution(map, 2, true);
    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs, solution, options);
    ASSERT_TRUE(stats.converged);
    ASSERT_EQ(stats.column_statistics.size(), 2U);
    EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(stats.column_statistics[0].iterations, 1);
    EXPECT_EQ(stats.column_statistics[1].iterations, 3);

    options.solver.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
    Pack::multi_vector_type serial_solution(map, 2, true);
    const auto serial_stats = solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs, serial_solution, options);
    ASSERT_TRUE(serial_stats.converged);
    EXPECT_EQ(serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
    EXPECT_EQ(serial_stats.column_statistics[0].iterations, 1);
    EXPECT_EQ(serial_stats.column_statistics[1].iterations, 3);
}

/**
 * @brief A tiny column accepted via ColumnResidualPolicy::absolute_tolerance or
 *        rhs_norm_floor must never inflate the gap gate and block GMRES refinement
 *        for another column.
 */
TEST(BelosLinearSolverTest, PerColumnPolicyTinyAcceptedColumnDoesNotBlockGmresRefinement)
{
    class TinyAcceptedGapSolver final : public Belos::PseudoBlockGmresSolMgr<
        Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>
    {
        using Manager = Belos::PseudoBlockGmresSolMgr<
            Pack::scalar_type, Pack::multi_vector_type, Pack::operator_type>;
    public:
        TinyAcceptedGapSolver(
            const Teuchos::RCP<SimpleFluid::BelosLinearSolver<Pack>::problem_type>& problem,
            const Teuchos::RCP<Teuchos::ParameterList>& parameters,
            int& solve_calls)
            : Manager(problem, parameters), d_solve_calls(solve_calls)
        {
        }

        Belos::ReturnType solve() override
        {
            d_injected = false;
            const auto status = Manager::solve();
            ++d_solve_calls;
            if (status == Belos::Converged && d_solve_calls == 1)
            {
                auto lhs = this->getProblem().getLHS();
                // Column 0 gets an error of 1e-15 (huge relative to ||b_0|| = 1e-18 * sqrt(N),
                // well above gmres_gap_ceiling, yet accepted by abs_tol=1e-12 and floor=1.0).
                // Column 1 gets a 2*tol gap needing defect-correction refinement.
                auto c0 = lhs->getDataNonConst(0);
                std::fill(c0.begin(), c0.end(), 1.0e-15);
                auto c1 = lhs->getDataNonConst(1);
                for (auto& v : c1)
                {
                    v *= (1.0 - 2.0e-10);
                }
                d_injected = true;
                return Belos::Unconverged;
            }
            return status;
        }

        bool isLOADetected() const override
        {
            return d_injected ? true : Manager::isLOADetected();
        }

    private:
        int& d_solve_calls;
        bool d_injected = false;
    };

    const auto map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        4, 0, Tpetra::getDefaultComm()));
    const auto id = SimpleFluid::FVM::identity_matrix<Pack>(map);

    Pack::multi_vector_type rhs(map, 2, true);
    rhs.getVectorNonConst(0)->putScalar(1.0e-18);
    rhs.getVectorNonConst(1)->putScalar(1.0);

    for (const bool use_floor : {false, true})
    {
        SCOPED_TRACE(use_floor);
        Pack::multi_vector_type solution(map, 2, true);
        SimpleFluid::BelosLinearSolver<Pack> solver;
        SimpleFluid::MultiRhsSolveOptions opts;
        opts.solver.backend = SimpleFluid::LinearSolverBackend::Gmres;
        opts.solver.tolerance = 1.0e-10;
        opts.solver.max_iterations = 20;
        opts.column_policies.resize(2);
        if (use_floor)
        {
            opts.column_policies[0].rhs_norm_floor = 1.0;
        }
        else
        {
            opts.column_policies[0].absolute_tolerance = 1.0e-12;
        }

        ASSERT_TRUE(solver.solve_multi_rhs_from_zero_with_statistics(id, rhs, solution, opts).converged);

        int solve_calls = 0;
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::replace_solver<TinyAcceptedGapSolver>(
            solver, solve_calls);
        const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(id, rhs, solution, opts);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(solve_calls, 2);
        EXPECT_EQ(stats.column_statistics[0].refinement_passes, 0);
        EXPECT_GE(stats.column_statistics[1].refinement_passes, 1);
        for (const auto v : solution.getData(1))
        {
            EXPECT_NEAR(v, 1.0, 1.0e-12);
        }
    }
}

TEST(BelosLinearSolverTest, NonReusablePreconditionerRebuildsWhenLeadingChunkIsSkipped)
{
    using scalar_type = Pack::scalar_type;
    using LO = Pack::local_ordinal_type;
    using GO = Pack::global_ordinal_type;

    const auto comm = Tpetra::getDefaultComm();
    auto matrix = distributed_tridiagonal_matrix(comm, 24, false);
    const auto map = matrix->getRowMap();
    const std::size_t local_rows = map->getLocalNumElements();

    SimpleFluid::BelosLinearSolver<Pack> solver;
    SimpleFluid::MultiRhsSolveOptions prime_opts;
    prime_opts.solver.backend = SimpleFluid::LinearSolverBackend::Gmres;
    prime_opts.solver.preconditioner = SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
    prime_opts.solver.reuse_preconditioner = true;
    prime_opts.solver.tolerance = 1.0e-12;
    prime_opts.solver.absolute_tolerance = 1.0e-12;

    Pack::multi_vector_type prime_rhs(map, 1);
    Pack::multi_vector_type prime_sol(map, 1);
    prime_rhs.putScalar(1.0);
    ASSERT_TRUE(solver.solve_multi_rhs_from_zero_with_statistics(matrix, prime_rhs, prime_sol, prime_opts).converged);
    ASSERT_EQ(preconditioner_setup_count(solver), 1U);

    // Mutate M1 -> M2 in place so reusing M1's cached preconditioner would be stale.
    matrix->resumeFill();
    for (std::size_t r = 0; r < local_rows; ++r)
    {
        const GO gid = map->getGlobalElement(static_cast<LO>(r));
        const scalar_type diag_shift = 6.0;
        matrix->sumIntoGlobalValues(
            gid, Teuchos::ArrayView<const GO>(&gid, 1), Teuchos::ArrayView<const scalar_type>(&diag_shift, 1));
    }
    matrix->fillComplete();

    Pack::multi_vector_type rhs_zero_first(map, 2);
    Pack::multi_vector_type rhs_nonzero_first(map, 2);
    rhs_zero_first.putScalar(0.0);
    rhs_nonzero_first.putScalar(0.0);
    for (std::size_t r = 0; r < local_rows; ++r)
    {
        const double val = 1.0 + 0.1 * static_cast<double>(map->getGlobalElement(static_cast<LO>(r)));
        rhs_zero_first.replaceLocalValue(static_cast<LO>(r), 1, val);
        rhs_nonzero_first.replaceLocalValue(static_cast<LO>(r), 0, val);
    }

    // Reference exact solve on M2.
    SimpleFluid::BelosLinearSolver<Pack> ref_solver;
    SimpleFluid::MultiRhsSolveOptions ref_opts = prime_opts;
    ref_opts.solver.reuse_preconditioner = false;
    Pack::multi_vector_type ref_sol(map, 2);
    ASSERT_TRUE(
        ref_solver.solve_multi_rhs_from_zero_with_statistics(matrix, rhs_zero_first, ref_sol, ref_opts).converged);
    ASSERT_EQ(preconditioner_setup_count(ref_solver), 1U);

    // Leading chunk is zero (skipped), trailing chunk is nonzero: reuse_preconditioner=false must rebuild for M2.
    SimpleFluid::MultiRhsSolveOptions chunked_opts = prime_opts;
    chunked_opts.solver.reuse_preconditioner = false;
    chunked_opts.solver.max_batch_width = 1;
    Pack::multi_vector_type sol_zero_first(map, 2);
    const auto stats_zero_first = solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs_zero_first, sol_zero_first, chunked_opts);
    ASSERT_TRUE(stats_zero_first.converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 2U);
    EXPECT_TRUE(stats_zero_first.column_statistics[0].skipped_zero);
    EXPECT_FALSE(stats_zero_first.column_statistics[1].skipped_zero);

    for (std::size_t r = 0; r < local_rows; ++r)
    {
        EXPECT_DOUBLE_EQ(sol_zero_first.getData(0)[r], 0.0);
        EXPECT_NEAR(sol_zero_first.getData(1)[r], ref_sol.getData(1)[r], 1.0e-11);
    }

    // Reversed column ordering [nonzero, zero] with max_batch_width=1 and batched max_batch_width=2 match M2 reference.
    Pack::multi_vector_type sol_nonzero_first(map, 2);
    const auto stats_nonzero_first = solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs_nonzero_first, sol_nonzero_first, chunked_opts);
    ASSERT_TRUE(stats_nonzero_first.converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 3U);

    chunked_opts.solver.max_batch_width = 2;
    Pack::multi_vector_type sol_batched(map, 2);
    const auto stats_batched = solver.solve_multi_rhs_from_zero_with_statistics(
        matrix, rhs_zero_first, sol_batched, chunked_opts);
    ASSERT_TRUE(stats_batched.converged);
    EXPECT_EQ(preconditioner_setup_count(solver), 4U);

    for (std::size_t r = 0; r < local_rows; ++r)
    {
        EXPECT_NEAR(sol_nonzero_first.getData(0)[r], ref_sol.getData(1)[r], 1.0e-11);
        EXPECT_DOUBLE_EQ(sol_nonzero_first.getData(1)[r], 0.0);
        EXPECT_DOUBLE_EQ(sol_batched.getData(0)[r], 0.0);
        EXPECT_NEAR(sol_batched.getData(1)[r], ref_sol.getData(1)[r], 1.0e-11);
    }
}

TEST(BelosLinearSolverTest, BatchedCgAndBiCGStabApplyWidthsPreconditionersAndParity)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto spd_matrix = distributed_tridiagonal_matrix(comm, 28, true);
    const auto nonsym_matrix = distributed_tridiagonal_matrix(comm, 28, false);
    const auto map = spd_matrix->getRowMap();
    const std::size_t cols = 5;

    Pack::multi_vector_type exact(map, cols, true);
    for (std::size_t c = 0; c < cols; ++c)
    {
        auto vals = exact.getDataNonConst(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const double gid = static_cast<double>(map->getGlobalElement(r));
            vals[r] = static_cast<double>(c + 1) * (0.4 + 0.03 * gid) * ((c % 2 == 0) ? 1.0 : -1.0);
        }
    }

    const std::vector<SimpleFluid::LinearPreconditioner> cg_precs = comm->getSize() == 1
        ? std::vector<SimpleFluid::LinearPreconditioner>{
              SimpleFluid::LinearPreconditioner::None,
              SimpleFluid::LinearPreconditioner::Jacobi,
              SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel,
              SimpleFluid::LinearPreconditioner::DIC,
              SimpleFluid::LinearPreconditioner::MueLuSymmetric}
        : std::vector<SimpleFluid::LinearPreconditioner>{
              SimpleFluid::LinearPreconditioner::None,
              SimpleFluid::LinearPreconditioner::Jacobi,
              SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel,
              SimpleFluid::LinearPreconditioner::MueLuSymmetric};

    for (const auto prec : cg_precs)
    {
        SCOPED_TRACE(SimpleFluid::to_string(prec));
        Pack::multi_vector_type rhs(map, cols, true);
        spd_matrix->apply(exact, rhs);
        auto op = Teuchos::rcp(new WidthRecordingOperator(spd_matrix));

        SimpleFluid::LinearSolverOptions opts;
        opts.backend = SimpleFluid::LinearSolverBackend::Cg;
        opts.preconditioner = prec;
        opts.tolerance = 1.0e-12;
        opts.max_iterations = 200;

        Pack::multi_vector_type batched_sol(map, cols, true);
        SimpleFluid::BelosLinearSolver<Pack> batched_solver;
        const auto stats = batched_solver.solve_from_zero_with_statistics(
            op, rhs, batched_sol, opts, {}, spd_matrix);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        EXPECT_EQ(stats.max_batch_width, cols);
        EXPECT_LT(stats.operator_apply_calls, stats.operator_column_applies);
        EXPECT_TRUE(std::any_of(op->recorded_widths().begin(), op->recorded_widths().end(),
            [cols](std::size_t w) { return w == cols; }));

        opts.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
        Pack::multi_vector_type serial_sol(map, cols, true);
        SimpleFluid::BelosLinearSolver<Pack> serial_solver;
        const auto serial_stats = serial_solver.solve_from_zero_with_statistics(
            spd_matrix, rhs, serial_sol, opts, {}, spd_matrix);
        ASSERT_TRUE(serial_stats.converged);
        EXPECT_EQ(serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
        EXPECT_EQ(serial_stats.max_batch_width, 1U);
        for (std::size_t c = 0; c < cols; ++c)
        {
            const auto actual = batched_sol.getData(c);
            const auto expected = serial_sol.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-10 * std::max(1.0, std::abs(expected[r])));
            }
        }
    }

    for (const auto prec : {
             SimpleFluid::LinearPreconditioner::None,
             SimpleFluid::LinearPreconditioner::Jacobi,
             SimpleFluid::LinearPreconditioner::ILU0,
             SimpleFluid::LinearPreconditioner::ILUT,
             SimpleFluid::LinearPreconditioner::GaussSeidel,
             SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        SCOPED_TRACE(SimpleFluid::to_string(prec));
        Pack::multi_vector_type rhs(map, cols, true);
        nonsym_matrix->apply(exact, rhs);
        auto op = Teuchos::rcp(new WidthRecordingOperator(nonsym_matrix));

        SimpleFluid::LinearSolverOptions opts;
        opts.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        opts.preconditioner = prec;
        opts.tolerance = 1.0e-12;
        opts.max_iterations = 200;

        Pack::multi_vector_type batched_sol(map, cols, true);
        SimpleFluid::BelosLinearSolver<Pack> batched_solver;
        const auto stats = batched_solver.solve_from_zero_with_statistics(
            op, rhs, batched_sol, opts, {}, nonsym_matrix);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        EXPECT_EQ(stats.max_batch_width, cols);
        EXPECT_LT(stats.operator_apply_calls, stats.operator_column_applies);
        EXPECT_TRUE(std::any_of(op->recorded_widths().begin(), op->recorded_widths().end(),
            [cols](std::size_t w) { return w == cols; }));

        opts.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
        Pack::multi_vector_type serial_sol(map, cols, true);
        SimpleFluid::BelosLinearSolver<Pack> serial_solver;
        const auto serial_stats = serial_solver.solve_from_zero_with_statistics(
            nonsym_matrix, rhs, serial_sol, opts, {}, nonsym_matrix);
        ASSERT_TRUE(serial_stats.converged);
        EXPECT_EQ(serial_stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);
        EXPECT_EQ(serial_stats.max_batch_width, 1U);
        for (std::size_t c = 0; c < cols; ++c)
        {
            const auto actual = batched_sol.getData(c);
            const auto expected = serial_sol.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-10 * std::max(1.0, std::abs(expected[r])));
            }
        }
    }
}

TEST(BelosLinearSolverTest, BatchedCgAndBiCGStabExtremeScalesAndCompaction)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {1.0, 3.0, 9.0, 27.0});
    const auto op = Teuchos::rcp(new WidthRecordingOperator(matrix));
    const auto map = matrix->getRowMap();

    // Col 0: normal single-mode (1 iter), Col 1: trace 1e-160 (4 iters), Col 2: huge 1e+12 (2 iters),
    // Col 3: exact zero RHS (0 iters), Col 4: trace 1e-120 (4 iters).
    Pack::multi_vector_type exact(map, 5, true);
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        const auto gid = map->getGlobalElement(r);
        const auto lid = static_cast<Pack::local_ordinal_type>(r);
        exact.replaceLocalValue(lid, 0, gid == 0 ? 1.0 : 0.0);
        exact.replaceLocalValue(lid, 1, 1.0e-160 * static_cast<double>(gid + 1));
        exact.replaceLocalValue(lid, 2, (gid < 2) ? 1.0e12 * static_cast<double>(gid + 1) : 0.0);
        exact.replaceLocalValue(lid, 3, 0.0);
        exact.replaceLocalValue(lid, 4, 1.0e-120 * static_cast<double>(4 - gid));
    }
    Pack::multi_vector_type rhs(map, 5, true);
    matrix->apply(exact, rhs);

    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        op->clear_recorded_widths();
        Pack::multi_vector_type solution(map, 5, true);
        SimpleFluid::MultiRhsSolveOptions opts;
        opts.solver.backend = backend;
        opts.solver.tolerance = 1.0e-12;
        opts.solver.max_iterations = 20;
        opts.column_policies.resize(5);
        opts.column_policies[1].tolerance = 1.0e-13;
        opts.column_policies[4].tolerance = 1.0e-13;

        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(op, rhs, solution, opts);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        ASSERT_EQ(stats.column_statistics.size(), 5U);
        EXPECT_EQ(stats.column_statistics[3].iterations, 0);
        EXPECT_LT(stats.column_statistics[0].iterations, stats.column_statistics[1].iterations);
        EXPECT_LT(stats.column_statistics[2].iterations, stats.column_statistics[1].iterations);
        // Active compaction must shrink apply width below 4 once cols 0 & 2 finish.
        const auto& widths = op->recorded_widths();
        EXPECT_TRUE(std::any_of(widths.begin(), widths.end(), [](std::size_t w) { return w == 4; }));
        EXPECT_TRUE(std::any_of(widths.begin(), widths.end(), [](std::size_t w) { return w < 4 && w > 0; }));
        for (std::size_t c = 0; c < 5; ++c)
        {
            const auto actual = solution.getData(c);
            const auto exp_vals = exact.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], exp_vals[r], 1.0e-11 * std::abs(exp_vals[r]) + 1.0e-300);
            }
        }
    }
}

TEST(BelosLinearSolverTest, BatchedBiCGStabHalfStepConvergenceAndZeroResidual)
{
    // A = diag(2, 2, 2, 3, 5, 11): Col 0 is supported only on {0,1,2} (eigenvalue 2 -> exact in 1 step
    // at BiCGStab half-step with s == 0), while Col 1 is supported on {0,3,4,5} (needs > 1 step).
    const auto comm = Tpetra::getDefaultComm();
    const auto matrix = distributed_diagonal_matrix(comm, {2.0, 2.0, 2.0, 3.0, 5.0, 11.0});
    const auto op = Teuchos::rcp(new WidthRecordingOperator(matrix));
    const auto map = matrix->getRowMap();

    Pack::multi_vector_type exact(map, 2, true);
    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
    {
        const auto gid = map->getGlobalElement(r);
        const auto lid = static_cast<Pack::local_ordinal_type>(r);
        exact.replaceLocalValue(lid, 0, gid < 3 ? static_cast<double>(gid + 1) : 0.0);
        exact.replaceLocalValue(lid, 1, static_cast<double>(gid + 1));
    }
    Pack::multi_vector_type rhs(map, 2, true);
    matrix->apply(exact, rhs);

    Pack::multi_vector_type solution(map, 2, true);
    SimpleFluid::LinearSolverOptions opts;
    opts.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    opts.tolerance = 1.0e-12;
    opts.max_iterations = 20;

    SimpleFluid::BelosLinearSolver<Pack> solver;
    const auto stats = solver.solve_from_zero_with_statistics(op, rhs, solution, opts);
    ASSERT_TRUE(stats.converged);
    EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
    ASSERT_EQ(stats.column_statistics.size(), 2U);
    EXPECT_EQ(stats.column_statistics[0].iterations, 1);
    EXPECT_GT(stats.column_statistics[1].iterations, 1);
    // Exact half-step retirement compacts out col 0 immediately before second half-step A*z.
    const auto& widths = op->recorded_widths();
    ASSERT_GE(widths.size(), 2U);
    EXPECT_EQ(widths[0], 2U);
    EXPECT_EQ(widths[1], 1U);
    for (std::size_t c = 0; c < 2; ++c)
    {
        const auto actual = solution.getData(c);
        const auto expected = exact.getData(c);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            EXPECT_NEAR(actual[r], expected[r], 1.0e-11);
        }
    }

    // Half-step retirement of a large-norm column (s_norms[0] ~ 2e8, accepted under loose policy)
    // must compact s_norms so surviving tiny-norm col 1 (||b_1|| ~ 1e-8) does not inherit col 0's
    // huge s_norms[0] in tT_floor / tS_floor and suffer a false breakdown.
    {
        Pack::multi_vector_type exact_mixed(map, 2, true);
        Pack::multi_vector_type rhs_mixed(map, 2, true);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const auto gid = map->getGlobalElement(r);
            const auto lid = static_cast<Pack::local_ordinal_type>(r);
            const double x0 = (gid == 0) ? 2.5e8 : ((gid == 3) ? (2.0e8 / 3.0) : 0.0);
            const double x1 = 1.0e-9 * static_cast<double>(gid + 1);
            exact_mixed.replaceLocalValue(lid, 0, x0);
            exact_mixed.replaceLocalValue(lid, 1, x1);
        }
        matrix->apply(exact_mixed, rhs_mixed);

        Pack::multi_vector_type sol_mixed(map, 2, true);
        SimpleFluid::MultiRhsSolveOptions mixed_opts;
        mixed_opts.solver.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        mixed_opts.solver.tolerance = 1.0e-12;
        mixed_opts.solver.max_iterations = 20;
        mixed_opts.column_policies.resize(2);
        mixed_opts.column_policies[0].tolerance = 0.25;
        mixed_opts.column_policies[1].tolerance = 1.0e-12;

        const auto mixed_stats = solver.solve_multi_rhs_from_zero_with_statistics(
            matrix, rhs_mixed, sol_mixed, mixed_opts);
        ASSERT_TRUE(mixed_stats.converged);
        EXPECT_EQ(mixed_stats.column_statistics[0].iterations, 1);
        EXPECT_GT(mixed_stats.column_statistics[1].iterations, 1);
        EXPECT_LE(mixed_stats.column_statistics[1].achieved_tolerance, 1.0e-12);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            EXPECT_NEAR(sol_mixed.getData(1)[r], exact_mixed.getData(1)[r], 1.0e-20);
        }
    }
}

TEST(BelosLinearSolverTest, BatchedCgAndBiCGStabBreakdownIsolation)
{
    const auto comm = Tpetra::getDefaultComm();

    // CG: row 0 has negative diagonal (-2.0 -> p^T A p < 0 breakdown on col 0),
    // rows 1..3 are SPD (cols 1 & 2 stay healthy and converge cleanly).
    {
        const auto cg_matrix = distributed_diagonal_matrix(comm, {-2.0, 4.0, 5.0, 7.0});
        const auto map = cg_matrix->getRowMap();
        Pack::multi_vector_type exact(map, 3, true);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const auto gid = map->getGlobalElement(r);
            const auto lid = static_cast<Pack::local_ordinal_type>(r);
            exact.replaceLocalValue(lid, 0, gid == 0 ? 1.0 : 0.0);
            exact.replaceLocalValue(lid, 1, gid >= 1 ? static_cast<double>(gid + 1) : 0.0);
            exact.replaceLocalValue(lid, 2, gid >= 1 ? -2.0 * static_cast<double>(gid) : 0.0);
        }
        Pack::multi_vector_type rhs(map, 3, true);
        cg_matrix->apply(exact, rhs);

        Pack::multi_vector_type solution(map, 3, true);
        SimpleFluid::LinearSolverOptions opts;
        opts.backend = SimpleFluid::LinearSolverBackend::Cg;
        opts.tolerance = 1.0e-12;
        opts.max_iterations = 20;

        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_from_zero_with_statistics(cg_matrix, rhs, solution, opts);
        EXPECT_FALSE(stats.converged);
        ASSERT_EQ(stats.column_statistics.size(), 3U);
        EXPECT_FALSE(stats.column_statistics[0].converged);
        EXPECT_TRUE(stats.column_statistics[1].converged);
        EXPECT_TRUE(stats.column_statistics[2].converged);
        for (std::size_t c : {1U, 2U})
        {
            const auto actual = solution.getData(c);
            const auto expected = exact.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-11);
            }
        }
    }

    // BiCGStab: row 0 is singular (0.0 -> A*y == 0 / <rhat, v> == 0 breakdown on col 0),
    // rows 1..3 are well-conditioned SPD/nonsymmetric (cols 1 & 2 stay healthy and converge).
    {
        const auto bicg_matrix = distributed_diagonal_matrix(comm, {0.0, 4.0, 5.0, 7.0});
        const auto map = bicg_matrix->getRowMap();
        Pack::multi_vector_type rhs(map, 3, true);
        Pack::multi_vector_type expected_healthy(map, 3, true);
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const auto gid = map->getGlobalElement(r);
            const auto lid = static_cast<Pack::local_ordinal_type>(r);
            rhs.replaceLocalValue(lid, 0, gid == 0 ? 1.0 : 0.0);
            expected_healthy.replaceLocalValue(lid, 1, gid >= 1 ? static_cast<double>(gid + 1) : 0.0);
            expected_healthy.replaceLocalValue(lid, 2, gid >= 1 ? -3.0 * static_cast<double>(gid) : 0.0);
        }
        // Populate cols 1 & 2 RHS cleanly using matrix apply on a scratch vector.
        Pack::multi_vector_type healthy_rhs(map, 3, true);
        bicg_matrix->apply(expected_healthy, healthy_rhs);
        for (std::size_t c : {1U, 2U})
        {
            const auto src = healthy_rhs.getData(c);
            auto dst = rhs.getDataNonConst(c);
            std::copy(src.begin(), src.end(), dst.begin());
        }

        Pack::multi_vector_type solution(map, 3, true);
        SimpleFluid::LinearSolverOptions opts;
        opts.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        opts.tolerance = 1.0e-12;
        opts.max_iterations = 20;

        SimpleFluid::BelosLinearSolver<Pack> solver;
        const auto stats = solver.solve_from_zero_with_statistics(bicg_matrix, rhs, solution, opts);
        EXPECT_FALSE(stats.converged);
        ASSERT_EQ(stats.column_statistics.size(), 3U);
        EXPECT_FALSE(stats.column_statistics[0].converged);
        EXPECT_TRUE(stats.column_statistics[1].converged);
        EXPECT_TRUE(stats.column_statistics[2].converged);
        for (std::size_t c : {1U, 2U})
        {
            const auto actual = solution.getData(c);
            const auto expected = expected_healthy.getData(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-11);
            }
        }
    }
}

TEST(BelosLinearSolverTest, BatchedBiCGStabOperatorScalingDoesNotFalseBreakdown)
{
    const auto comm = Tpetra::getDefaultComm();
    for (const double scale : {1.0, 1.0e-18, 1.0e-20})
    {
        SCOPED_TRACE(scale);
        const auto matrix = distributed_diagonal_matrix(comm, {scale, 2.0 * scale});
        const auto map = matrix->getRowMap();
        Pack::multi_vector_type exact(map, 2, true);
        Pack::multi_vector_type rhs(map, 2, true);
        for (std::size_t column = 0; column < 2; ++column)
        {
            exact.getVectorNonConst(column)->putScalar(1.0);
        }
        matrix->apply(exact, rhs);

        SimpleFluid::LinearSolverOptions options;
        options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        options.tolerance = 1.0e-12;
        options.max_iterations = 20;

        Pack::multi_vector_type batched_solution(map, 2, true);
        SimpleFluid::BelosLinearSolver<Pack> batched_solver;
        const auto batched = batched_solver.solve_from_zero_with_statistics(
            matrix, rhs, batched_solution, options);
        ASSERT_TRUE(batched.converged);
        EXPECT_EQ(batched.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        ASSERT_EQ(batched.column_statistics.size(), 2U);

        auto serial_options = options;
        serial_options.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
        Pack::multi_vector_type serial_solution(map, 2, true);
        SimpleFluid::BelosLinearSolver<Pack> serial_solver;
        const auto serial = serial_solver.solve_from_zero_with_statistics(
            matrix, rhs, serial_solution, serial_options);
        ASSERT_TRUE(serial.converged);
        EXPECT_EQ(serial.execution_mode, SimpleFluid::MultiRhsExecutionMode::ColumnSerial);

        for (std::size_t column = 0; column < 2; ++column)
        {
            EXPECT_TRUE(batched.column_statistics[column].converged);
            EXPECT_LE(batched.column_statistics[column].achieved_tolerance, options.tolerance);
            EXPECT_EQ(batched.column_statistics[column].iterations,
                serial.column_statistics[column].iterations);
            const auto actual = batched_solution.getData(column);
            const auto serial_actual = serial_solution.getData(column);
            for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
            {
                EXPECT_NEAR(actual[row], 1.0, 1.0e-10);
                EXPECT_NEAR(serial_actual[row], 1.0, 1.0e-10);
            }
        }
    }
}

TEST(BelosLinearSolverTest, BatchedHealthyColumnCorrectionIsolatedFromFailedNeighbor)
{
    const auto comm = Tpetra::getDefaultComm();
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        for (const int neighbor_mode : {0, 1, 2})
        {
            // 0 = zero neighbor, 1 = singular breakdown, 2 = nonfinite neighbor.
            SCOPED_TRACE(neighbor_mode);
            const auto matrix = distributed_diagonal_matrix(comm, {0.0, 1.0});
            const auto map = matrix->getRowMap();
            Pack::multi_vector_type rhs(map, 2, true);
            for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
            {
                const auto gid = map->getGlobalElement(row);
                const double neighbor_value = neighbor_mode == 0
                    ? 0.0
                    : (neighbor_mode == 1
                        ? (gid == 0 ? 1.0 : 0.0)
                        : std::numeric_limits<double>::quiet_NaN());
                rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 0, neighbor_value);
                rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(row), 1,
                    gid == 1 ? 1.0 : 0.0);
            }

            SimpleFluid::LinearSolverOptions options;
            options.backend = backend;
            options.tolerance = 1.0e-12;
            options.max_iterations = 20;
            Pack::multi_vector_type solution(map, 2, true);
            SimpleFluid::BelosLinearSolver<Pack> solver;
            SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::set_batched_post_solve_hook(
                solver,
                [](Pack::multi_vector_type& current, std::size_t pass)
                {
                    if (pass != 0)
                    {
                        return;
                    }
                    auto healthy = current.getDataNonConst(1);
                    for (auto& value : healthy)
                    {
                        value *= 1.0 - 1.5e-12;
                    }
                });
            const auto stats = solver.solve_from_zero_with_statistics(
                matrix, rhs, solution, options);
            ASSERT_EQ(stats.column_statistics.size(), 2U);
            EXPECT_EQ(stats.converged, neighbor_mode == 0);
            EXPECT_EQ(stats.column_statistics[0].converged, neighbor_mode == 0);

            // The healthy column's independent correction runs even though the
            // neighbor is broken down; overall batch acceptance remains gated.
            EXPECT_TRUE(stats.column_statistics[1].converged);
            EXPECT_EQ(stats.column_statistics[1].refinement_passes, 1);
            EXPECT_EQ(stats.column_statistics[1].iterations, 2);
            EXPECT_LE(stats.column_statistics[1].achieved_tolerance, options.tolerance);
            const auto healthy = solution.getData(1);
            for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
            {
                const auto gid = map->getGlobalElement(row);
                EXPECT_NEAR(healthy[row], gid == 1 ? 1.0 : 0.0, 1.0e-12);
            }
        }
    }
}

TEST(BelosLinearSolverTest, BatchedCgAndBiCGStabDefectCorrectionRefinement)
{
    const auto comm = Tpetra::getDefaultComm();

    // 1) Larger non-diagonal 24-row tridiagonal system at tol=1e-12 with 1.5x gap on col 1 only:
    // proves defect-correction solves A*dx = defect at gap-ratio tolerance (never tol^2!),
    // finishes in a few iterations well below max_iterations, and never touches already-converged col 0.
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        const bool sym = (backend == SimpleFluid::LinearSolverBackend::Cg);
        const auto tri = distributed_tridiagonal_matrix(comm, 24, sym);
        const auto tri_op = Teuchos::rcp(new WidthRecordingOperator(tri));
        const auto tri_map = tri->getRowMap();

        Pack::multi_vector_type exact(tri_map, 2, true);
        for (std::size_t r = 0; r < tri_map->getLocalNumElements(); ++r)
        {
            const auto gid = static_cast<double>(tri_map->getGlobalElement(r));
            const auto lid = static_cast<Pack::local_ordinal_type>(r);
            exact.replaceLocalValue(lid, 0, 1.0 + 0.05 * gid);
            exact.replaceLocalValue(lid, 1, 2.0 - 0.03 * gid);
        }
        Pack::multi_vector_type rhs(tri_map, 2, true);
        tri->apply(exact, rhs);

        SimpleFluid::LinearSolverOptions opts;
        opts.backend = backend;
        opts.preconditioner = SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
        opts.tolerance = 1.0e-12;
        opts.max_iterations = 80;

        SimpleFluid::BelosLinearSolver<Pack> solver;
        SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::set_batched_post_solve_hook(
            solver,
            [&](Pack::multi_vector_type& sol, std::size_t pass)
            {
                if (pass == 0)
                {
                    auto c1 = sol.getDataNonConst(1);
                    for (double& v : c1)
                    {
                        v *= (1.0 - 1.5 * opts.tolerance);
                    }
                }
            });

        Pack::multi_vector_type solution(tri_map, 2, true);
        tri_op->clear_recorded_widths();
        const auto stats = solver.solve_from_zero_with_statistics(tri_op, rhs, solution, opts, {}, tri);
        ASSERT_TRUE(stats.converged);
        EXPECT_EQ(stats.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
        ASSERT_EQ(stats.column_statistics.size(), 2U);
        EXPECT_EQ(stats.column_statistics[0].refinement_passes, 0);
        EXPECT_EQ(stats.column_statistics[1].refinement_passes, 1);
        EXPECT_LT(stats.column_statistics[1].iterations, 40);
        EXPECT_LE(stats.column_statistics[1].achieved_tolerance, opts.tolerance);
        // Refinement pass must pack only the single unconverged column (width=1 < 2).
        EXPECT_TRUE(std::any_of(tri_op->recorded_widths().begin(), tri_op->recorded_widths().end(),
            [](std::size_t w) { return w == 1U; }));
        for (std::size_t c = 0; c < 2; ++c)
        {
            const auto actual = solution.getData(c);
            const auto expected = exact.getData(c);
            for (std::size_t r = 0; r < tri_map->getLocalNumElements(); ++r)
            {
                EXPECT_NEAR(actual[r], expected[r], 1.0e-10);
            }
        }
    }

    // 2) Cancellation-sensitive CRS matrix triggers accurate_crs_residual for batched CG and BiCGStab.
    {
        const auto crs = block_matrix<3>({{
            {{1.0e16, 1.0, -1.0e16}}, {{0.0, 1.0e12, 0.0}}, {{0.0, 0.0, 1.0e12}}}});
        const auto sym_crs = block_matrix<3>({{
            {{1.0e16, 1.0, -1.0e16}}, {{1.0, 1.0e12, 0.0}}, {{-1.0e16, 0.0, 1.0e16 + 1.0e12}}}});
        for (const auto backend : {
                 SimpleFluid::LinearSolverBackend::Cg,
                 SimpleFluid::LinearSolverBackend::BiCGStab})
        {
            SCOPED_TRACE(SimpleFluid::to_string(backend));
            const auto matrix = (backend == SimpleFluid::LinearSolverBackend::Cg) ? sym_crs : crs;
            const auto map = matrix->getRowMap();
            Pack::multi_vector_type exact(map, 2, true);
            Pack::multi_vector_type rhs(map, 2, true);
            for (std::size_t c = 0; c < 2; ++c)
            {
                const double s = static_cast<double>(c + 1);
                auto x = exact.getDataNonConst(c);
                auto b = rhs.getDataNonConst(c);
                x[0] = s;
                x[1] = s;
                x[2] = s;
                b[0] = s;
                b[1] = (backend == SimpleFluid::LinearSolverBackend::Cg ? 1.0 + 1.0e12 : 1.0e12) * s;
                b[2] = 1.0e12 * s;
            }
            // Standard A*x drops the middle 1.0 term on row 0, leaving rel = 1/sqrt(2e24) ~ 7.07e-13
            // inside (tol, 2*tol] for tol=5e-13; accurate_crs_residual recovers exact 0.0!
            Pack::multi_vector_type solution(map, 2, true);
            SimpleFluid::LinearSolverOptions opts;
            opts.backend = backend;
            opts.tolerance = 5.0e-13;
            opts.max_iterations = 20;
            SimpleFluid::BelosLinearSolver<Pack> solver;
            SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>::set_batched_post_solve_hook(
                solver,
                [&](Pack::multi_vector_type& sol, std::size_t)
                {
                    sol.assign(exact);
                });
            const auto stats = solver.solve_from_zero_with_statistics(matrix, rhs, solution, opts);
            EXPECT_TRUE(stats.converged);
            EXPECT_DOUBLE_EQ(stats.column_statistics[0].achieved_tolerance, 0.0);
            EXPECT_DOUBLE_EQ(stats.column_statistics[1].achieved_tolerance, 0.0);
            EXPECT_EQ(stats.column_statistics[0].refinement_passes, 0);
            EXPECT_EQ(stats.column_statistics[1].refinement_passes, 0);
        }
    }
}

TEST(BelosLinearSolverTest, BatchedCgAndBiCGStabChunkingPrecondInvalidationAndMpiInvariants)
{
    const auto comm = Tpetra::getDefaultComm();

    // 1) BatchedKrylov + max_batch_width == 1 must throw; BatchedKrylov on 1-col with max_batch_width == 0 succeeds.
    {
        const auto matrix = distributed_tridiagonal_matrix(comm, 12, true);
        const auto map = matrix->getRowMap();
        Pack::multi_vector_type rhs(map, 1, true);
        Pack::multi_vector_type sol(map, 1, true);
        rhs.putScalar(1.0);
        SimpleFluid::BelosLinearSolver<Pack> solver;

        SimpleFluid::LinearSolverOptions invalid;
        invalid.backend = SimpleFluid::LinearSolverBackend::Cg;
        invalid.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::BatchedKrylov;
        invalid.max_batch_width = 1;
        EXPECT_THROW((void)solver.solve_from_zero_with_statistics(matrix, rhs, sol, invalid), std::invalid_argument);

        SimpleFluid::LinearSolverOptions valid_single;
        valid_single.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        valid_single.multi_rhs_policy = SimpleFluid::MultiRhsExecutionPolicy::BatchedKrylov;
        valid_single.max_batch_width = 0;
        const auto s = solver.solve_from_zero_with_statistics(matrix, rhs, sol, valid_single);
        ASSERT_TRUE(s.converged);
        EXPECT_EQ(s.execution_mode, SimpleFluid::MultiRhsExecutionMode::SingleRhs);
    }

    // 2) Chunking invariance across {0, 1, 2, 3} (incl. trailing width=1 chunk) + active mask + M1->M2 rebuild.
    for (const auto backend : {
             SimpleFluid::LinearSolverBackend::Cg,
             SimpleFluid::LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(SimpleFluid::to_string(backend));
        const bool sym = (backend == SimpleFluid::LinearSolverBackend::Cg);
        auto matrix = distributed_tridiagonal_matrix(comm, 20, sym);
        const auto map = matrix->getRowMap();
        const std::size_t cols = 5;

        Pack::multi_vector_type exact(map, cols, true);
        for (std::size_t c = 0; c < cols; ++c)
        {
            auto x = exact.getDataNonConst(c);
            for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
            {
                x[r] = (c == 1) ? 0.0 : static_cast<double>(c + 1) * (1.0 + 0.1 * static_cast<double>(r));
            }
        }
        Pack::multi_vector_type rhs(map, cols, true);
        matrix->apply(exact, rhs);

        Pack::multi_vector_type baseline(map, cols, true);
        for (const int bw : {0, 1, 2, 3})
        {
            SCOPED_TRACE(bw);
            Pack::multi_vector_type sol(map, cols, true);
            sol.getVectorNonConst(1)->putScalar(777.0); // Poisoned inactive col must be zeroed.
            SimpleFluid::MultiRhsSolveOptions opts;
            opts.solver.backend = backend;
            opts.solver.preconditioner = SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
            opts.solver.tolerance = 1.0e-12;
            opts.solver.max_iterations = 200;
            opts.solver.max_batch_width = bw;
            opts.active_columns = {true, false, true, true, true}; // 4 active cols -> bw=3 yields [3,1] chunks.

            SimpleFluid::BelosLinearSolver<Pack> solver;
            const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(matrix, rhs, sol, opts);
            ASSERT_TRUE(stats.converged);
            EXPECT_EQ(stats.execution_mode,
                bw == 1 ? SimpleFluid::MultiRhsExecutionMode::ColumnSerial
                        : SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
            EXPECT_EQ(stats.preconditioner_builds, 1U);
            EXPECT_TRUE(stats.column_statistics[1].skipped_zero);
            for (const auto v : sol.getData(1))
            {
                EXPECT_DOUBLE_EQ(v, 0.0);
            }
            if (bw == 0)
            {
                baseline.assign(sol);
            }
            else
            {
                for (std::size_t c : {0U, 2U, 3U, 4U})
                {
                    const auto a = sol.getData(c);
                    const auto b = baseline.getData(c);
                    for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
                    {
                        EXPECT_NEAR(a[r], b[r], 1.0e-10 * std::max(1.0, std::abs(b[r])));
                    }
                }
            }
        }

        // Reuse=false with leading inactive chunk on in-place M1 -> M2 must invalidate stale preconditioner.
        SimpleFluid::BelosLinearSolver<Pack> m1_m2_solver;
        SimpleFluid::MultiRhsSolveOptions prime_opts;
        prime_opts.solver.backend = backend;
        prime_opts.solver.preconditioner = SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
        prime_opts.solver.reuse_preconditioner = true;
        prime_opts.solver.tolerance = 1.0e-12;
        Pack::multi_vector_type prime_rhs(map, 1, true);
        Pack::multi_vector_type prime_sol(map, 1, true);
        prime_rhs.putScalar(1.0);
        ASSERT_TRUE(m1_m2_solver.solve_multi_rhs_from_zero_with_statistics(
            matrix, prime_rhs, prime_sol, prime_opts).converged);
        ASSERT_EQ(preconditioner_setup_count(m1_m2_solver), 1U);

        matrix->resumeFill();
        for (std::size_t r = 0; r < map->getLocalNumElements(); ++r)
        {
            const auto gid = map->getGlobalElement(static_cast<Pack::local_ordinal_type>(r));
            const double shift = 8.0;
            matrix->sumIntoGlobalValues(
                gid, Teuchos::ArrayView<const Pack::global_ordinal_type>(&gid, 1),
                Teuchos::ArrayView<const double>(&shift, 1));
        }
        matrix->fillComplete();

        Pack::multi_vector_type rhs2(map, 2, true);
        rhs2.getVectorNonConst(1)->putScalar(2.0);
        Pack::multi_vector_type sol2(map, 2, true);
        SimpleFluid::MultiRhsSolveOptions opts2 = prime_opts;
        opts2.solver.reuse_preconditioner = false;
        opts2.solver.max_batch_width = 2;
        opts2.active_columns = {false, true};
        const auto s2 = m1_m2_solver.solve_multi_rhs_from_zero_with_statistics(
            matrix, rhs2, sol2, opts2);
        ASSERT_TRUE(s2.converged);
        EXPECT_EQ(preconditioner_setup_count(m1_m2_solver), 2U);
    }

    // 3) Empty-owner MPI rank + cross-rank max_batch_width / multi_rhs_policy disagreement rejection.
    {
        const auto empty_matrix = distributed_diagonal_matrix(comm, {4.0});
        const auto empty_map = empty_matrix->getRowMap();
        Pack::multi_vector_type empty_rhs(empty_map, 3, true);
        for (std::size_t c = 0; c < 3; ++c)
        {
            empty_rhs.getVectorNonConst(c)->putScalar(static_cast<double>(c + 1));
        }
        for (const auto backend : {
                 SimpleFluid::LinearSolverBackend::Cg,
                 SimpleFluid::LinearSolverBackend::BiCGStab})
        {
            Pack::multi_vector_type empty_sol(empty_map, 3, true);
            SimpleFluid::MultiRhsSolveOptions opts;
            opts.solver.backend = backend;
            opts.solver.tolerance = 1.0e-12;
            SimpleFluid::BelosLinearSolver<Pack> solver;
            const auto s = solver.solve_multi_rhs_from_zero_with_statistics(
                empty_matrix, empty_rhs, empty_sol, opts);
            ASSERT_TRUE(s.converged);
            EXPECT_EQ(s.execution_mode, SimpleFluid::MultiRhsExecutionMode::BatchedKrylov);
            for (std::size_t c = 0; c < 3; ++c)
            {
                for (const auto v : empty_sol.getData(c))
                {
                    EXPECT_NEAR(v, 0.25 * static_cast<double>(c + 1), 1.0e-12);
                }
            }

            if (comm->getSize() > 1)
            {
                auto disagree_bw = opts;
                disagree_bw.solver.max_batch_width = comm->getRank() == 0 ? 0 : 2;
                EXPECT_THROW((void)solver.solve_multi_rhs_from_zero_with_statistics(
                    empty_matrix, empty_rhs, empty_sol, disagree_bw), std::invalid_argument);

                auto disagree_pol = opts;
                disagree_pol.solver.multi_rhs_policy = comm->getRank() == 0
                    ? SimpleFluid::MultiRhsExecutionPolicy::Auto
                    : SimpleFluid::MultiRhsExecutionPolicy::ColumnSerial;
                EXPECT_THROW((void)solver.solve_multi_rhs_from_zero_with_statistics(
                    empty_matrix, empty_rhs, empty_sol, disagree_pol), std::invalid_argument);
            }
        }
    }
}
