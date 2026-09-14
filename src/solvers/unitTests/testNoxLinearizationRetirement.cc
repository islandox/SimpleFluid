#include "solvers/BelosLinearSolver.hh"
#include "solvers/NoxNonlinearSolver.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

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

/** Two nonzero rows expose component membership without depending on a mesh.
 * Nonidentity scales also exercise the scaled operator and physical-vector
 * paths while leaving the exact Newton correction a multiple of identity.
 */
NonlinearCallbacks component_callbacks(Teuchos::RCP<const Pack::map_type> map)
{
    NonlinearCallbacks callbacks;
    callbacks.map = std::move(map);
    auto state_scale = Teuchos::rcp(new Vector(callbacks.map));
    auto residual_scale = Teuchos::rcp(new Vector(callbacks.map));
    state_scale->putScalar(2.);
    residual_scale->putScalar(.5);
    callbacks.state_scale = state_scale;
    callbacks.residual_scale = residual_scale;
    callbacks.residual = [](const Vector& x, Vector& residual)
    {
        const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto output = residual.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t row = 0; row < input.extent(0); ++row)
            output(row, 0) = input(row, 0) - (row < 2 ? 4. : 0.);
        return true;
    };
    auto identity = Teuchos::rcp(new DiagonalOperator(callbacks.map));
    identity->diagonal.putScalar(1.);
    callbacks.linearize = [identity](const Vector&) { return NonlinearLinearization{identity, Teuchos::null}; };
    return callbacks;
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

TEST(NoxLinearizationRetirementTest, ComponentChecksFollowGlobalIdsAcrossRepeatedSolvesAndMapReplacement)
{
    const auto comm = Tpetra::getDefaultComm();
    // Components are [1,1,2,2,3,3,0,0], including negative and noncontiguous
    // global IDs. The first two residuals have RMS 2 within component 1;
    // assigning components by local row would incorrectly report sqrt(2).
    Teuchos::Array<Pack::global_ordinal_type> gids{-7, -3, 6, 10, 15, 19, 20, 24};
    for (auto& gid : gids)
        gid += 64 * comm->getRank();
    const auto map = Teuchos::rcp(new Pack::map_type(8 * comm->getSize(), gids(), -7, comm));
    const auto equivalent =
        Teuchos::rcp(new Pack::map_type(map->getGlobalNumElements(), gids(), map->getIndexBase(), comm->duplicate()));
    std::swap(gids[1], gids[2]);
    const auto reordered =
        Teuchos::rcp(new Pack::map_type(map->getGlobalNumElements(), gids(), map->getIndexBase(), comm));

    for (const auto backend : {LinearSolverBackend::Gmres, LinearSolverBackend::BiCGStab})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        NOXNonlinearSolver solver(component_callbacks(map));
        auto nonlinear = nonlinear_options();
        nonlinear.absolute_tolerance = 2.1;
        LinearSolverOptions linear;
        linear.backend = backend;
        Vector x(map);
        x.putScalar(0.);
        const auto first = solver.solve(x, nonlinear, linear);
        ASSERT_TRUE(first.converged) << first.reason;
        EXPECT_EQ(first.nonlinear_iterations, 0);
        EXPECT_DOUBLE_EQ(first.scaled_residual, 2.);
        const auto first_statistics = solver.cache_statistics();
        EXPECT_EQ(first_statistics.vector_cache_builds, 1U);

        nonlinear.absolute_tolerance = 1.5;
        const auto correction = solver.solve(x, nonlinear, linear);
        ASSERT_TRUE(correction.converged) << correction.reason;
        EXPECT_GT(correction.nonlinear_iterations, 0);
        {
            const auto corrected = x.getData(0);
            for (int row = 0; row < corrected.size(); ++row)
                EXPECT_NEAR(corrected[row], row < 2 ? 4. : 0., 1.e-12);
        }
        const auto correction_statistics = solver.cache_statistics();
        EXPECT_EQ(correction_statistics.vector_cache_builds, first_statistics.vector_cache_builds);

        // A new map instance on a congruent communicator preserves ordered IDs
        // and must reuse the vector/component cache, including after a solve.
        solver.set_callbacks(component_callbacks(equivalent));
        Vector equivalent_x(equivalent);
        equivalent_x.putScalar(0.);
        nonlinear.absolute_tolerance = 2.1;
        const auto compatible = solver.solve(equivalent_x, nonlinear, linear);
        ASSERT_TRUE(compatible.converged) << compatible.reason;
        EXPECT_EQ(compatible.nonlinear_iterations, 0);
        EXPECT_DOUBLE_EQ(compatible.scaled_residual, 2.);
        EXPECT_EQ(solver.cache_statistics().vector_cache_builds, first_statistics.vector_cache_builds);
        EXPECT_EQ(solver.cache_statistics().linear_solver_builds, correction_statistics.linear_solver_builds);

        // The same GID set reordered locally now splits the two nonzero rows
        // between two components. Stale membership would force a correction.
        solver.set_callbacks(component_callbacks(reordered));
        Vector reordered_x(reordered);
        reordered_x.putScalar(0.);
        nonlinear.absolute_tolerance = 1.5;
        const auto changed = solver.solve(reordered_x, nonlinear, linear);
        ASSERT_TRUE(changed.converged) << changed.reason;
        EXPECT_EQ(changed.nonlinear_iterations, 0);
        EXPECT_NEAR(changed.scaled_residual, std::sqrt(2.), 1.e-14);
        for (const auto value : reordered_x.getData(0))
            EXPECT_DOUBLE_EQ(value, 0.);
        EXPECT_EQ(solver.cache_statistics().vector_cache_builds, first_statistics.vector_cache_builds + 1);

        // Returning to the original map must restore its component groups.
        solver.set_callbacks(component_callbacks(map));
        x.putScalar(0.);
        const auto restored = solver.solve(x, nonlinear, linear);
        ASSERT_TRUE(restored.converged) << restored.reason;
        EXPECT_GT(restored.nonlinear_iterations, 0);
        EXPECT_EQ(solver.cache_statistics().vector_cache_builds, first_statistics.vector_cache_builds + 2);
    }
}
#endif
