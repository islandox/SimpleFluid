#include "solvers/BelosLinearSolver.hh"
#include "solvers/NoxNonlinearSolver.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>
#include <stdexcept>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Vector = Pack::vector_type;
using MultiVector = Pack::multi_vector_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

#ifdef SIMPLEFLUID_ENABLE_NOX
class DiagonalOperator final : public Pack::operator_type
{
public:
    explicit DiagonalOperator(Teuchos::RCP<const Pack::map_type> map) : diagonal(std::move(map)) {}
    Teuchos::RCP<const Pack::map_type> getDomainMap() const override { return diagonal.getMap(); }
    Teuchos::RCP<const Pack::map_type> getRangeMap() const override { return diagonal.getMap(); }
    void apply(const MultiVector& x, MultiVector& y, Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.,
        double beta = 0.) const override
    {
        if (mode != Teuchos::NO_TRANS)
            throw std::invalid_argument("Diagonal test operator supports forward action only");
        const auto factors = diagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto output = y.getLocalViewHost(Tpetra::Access::ReadWrite);
        for (size_t column = 0; column < input.extent(1); ++column)
            for (size_t row = 0; row < input.extent(0); ++row)
            {
                const double value = alpha * factors(row, 0) * input(row, column);
                output(row, column) = beta == 0. ? value : value + beta * output(row, column);
            }
    }
    Vector diagonal;
};

/** A native cache that may update J/P only after every consumer releases them.
 * Keeping either operator requires a fresh pair, as with a coupled-system lease.
 */
struct NativeCache
{
    Teuchos::RCP<DiagonalOperator> jacobian, preconditioner;
    Teuchos::RCP<const Pack::operator_type> external;
    size_t builds = 0;
    size_t generations = 0;
    size_t trials_with_retained_generation = 0;
    int retain_first = 0; // 1: Jacobian, 2: preconditioner

    NonlinearCallbacks callbacks(Teuchos::RCP<const Pack::map_type> map)
    {
        NonlinearCallbacks result;
        result.map = std::move(map);
        result.residual = [this](const Vector& x, Vector& residual)
        {
            if (!jacobian.is_null() && jacobian.strong_count() > 1)
                ++trials_with_retained_generation;
            const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
            auto output = residual.getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t row = 0; row < input.extent(0); ++row)
                output(row, 0) = input(row, 0) * input(row, 0) - 1.;
            return true;
        };
        result.linearize = [this](const Vector& x)
        {
            if (jacobian.is_null() || jacobian.strong_count() != 1 || preconditioner.strong_count() != 1)
            {
                jacobian = Teuchos::rcp(new DiagonalOperator(x.getMap()));
                preconditioner = Teuchos::rcp(new DiagonalOperator(x.getMap()));
                ++builds;
            }
            const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
            auto j = jacobian->diagonal.getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto p = preconditioner->diagonal.getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t row = 0; row < input.extent(0); ++row)
            {
                j(row, 0) = 2. * input(row, 0);
                p(row, 0) = 1. / j(row, 0);
            }
            if (++generations == 1)
            {
                if (retain_first == 1)
                    external = jacobian;
                else if (retain_first == 2)
                    external = preconditioner;
            }
            return NonlinearLinearization{jacobian, preconditioner};
        };
        return result;
    }
};

Teuchos::RCP<const Pack::map_type> equation_map()
{
    const auto comm = Tpetra::getDefaultComm();
    return Teuchos::rcp(new Pack::map_type(4 * comm->getSize(), 4, 0, comm));
}

NonlinearSolverOptions nonlinear_options()
{
    NonlinearSolverOptions options;
    options.absolute_tolerance = 1.e-12;
    options.relative_tolerance = 1.e-11;
    return options;
}
#endif
} // namespace

#ifdef SIMPLEFLUID_ENABLE_NOX
TEST(NoxLinearizationRetirementTest, ReusesNativeCacheAfterBacktrackingAndAcrossSolves)
{
    for (const auto backend : {LinearSolverBackend::Gmres, LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        NativeCache cache;
        const auto map = equation_map();
        NOXNonlinearSolver solver(cache.callbacks(map));
        LinearSolverOptions linear;
        linear.backend = backend;
        Vector x(map);
        x.putScalar(.1); // Full Newton step overshoots: exercise retained Armijo J.
        const auto first = solver.solve(x, nonlinear_options(), linear);
        ASSERT_TRUE(first.converged) << first.reason;
        EXPECT_GT(first.line_search_rejections, 0);
        EXPECT_GT(cache.generations, 2U);
        EXPECT_GT(cache.trials_with_retained_generation, 0U);
        EXPECT_EQ(cache.builds, 1U);
        const auto first_statistics = solver.cache_statistics();
        EXPECT_EQ(first_statistics.linearization_generations, cache.generations);
        EXPECT_EQ(first_statistics.linearization_retirements, cache.generations - 1);

        x.putScalar(.2);
        const auto second = solver.solve(x, nonlinear_options(), linear);
        ASSERT_TRUE(second.converged) << second.reason;
        for (const auto value : x.getData(0))
            EXPECT_NEAR(value, 1., 1.e-10);
        EXPECT_EQ(cache.builds, 1U);
        const auto second_statistics = solver.cache_statistics();
        EXPECT_EQ(second_statistics.linearization_generations, cache.generations);
        EXPECT_EQ(second_statistics.linearization_retirements, cache.generations - 2);
        EXPECT_EQ(second_statistics.vector_cache_builds, first_statistics.vector_cache_builds);
        EXPECT_EQ(second_statistics.linear_solver_builds, first_statistics.linear_solver_builds);
    }
}

TEST(NoxLinearizationRetirementTest, ExternalJacobianOrPreconditionerRequiresFreshImmutableGeneration)
{
    for (const int retained : {1, 2})
    {
        SCOPED_TRACE(retained);
        NativeCache cache;
        cache.retain_first = retained;
        const auto map = equation_map();
        NOXNonlinearSolver solver(cache.callbacks(map));
        Vector x(map);
        x.putScalar(.1);
        const auto result = solver.solve(x, nonlinear_options(), LinearSolverOptions{});
        ASSERT_TRUE(result.converged) << result.reason;
        EXPECT_GT(cache.generations, 2U);
        // The externally owned first generation forces exactly one replacement.
        // Following generations reuse the latest cache after solver retirement.
        EXPECT_EQ(cache.builds, 2U);
        ASSERT_FALSE(cache.external.is_null());
        Vector ones(map), action(map);
        ones.putScalar(1.);
        cache.external->apply(ones, action);
        for (const auto value : action.getData(0))
            EXPECT_NEAR(value, retained == 1 ? .2 : 5., 1.e-13);
        EXPECT_EQ(solver.cache_statistics().linearization_retirements, cache.generations - 1);
    }
}
#endif
