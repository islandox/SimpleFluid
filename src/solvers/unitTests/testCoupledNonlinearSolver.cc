/** @file testCoupledNonlinearSolver.cc
 *  @brief Residual, derivative, transaction, and NOX integration contracts.
 */
#include "FVM/FaceFlux.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/CoupledNonlinearProblem.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "solvers/NoxNonlinearSolver.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using NativeMesh = MeshHandle<Pack>;
using Vector = Pack::vector_type;
using LO = Pack::local_ordinal_type;
using NonlinearProblem = CoupledNonlinearProblem;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const NativeMesh> cavity_mesh(bool reverse_partitions = false)
{
    NativeMesh::DistributionOptions distribution;
    if (reverse_partitions)
    {
        const auto comm = Tpetra::getDefaultComm();
        distribution.partition = static_cast<size_t>(comm->getSize() - comm->getRank() - 1);
        distribution.partitions = static_cast<size_t>(comm->getSize());
    }
    return std::make_shared<NativeMesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                            Vec3D<ArrReal>{{{0., .25, .5, .75, 1.}, {0., .3, .65, 1.}, {0., .5, 1.}}}),
        distribution);
}

SP<const NativeMesh> planar_cavity_mesh()
{
    return std::make_shared<NativeMesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0., .25, .5, .75, 1.}, {0., .3, .65, 1.}, {0., 1.}}}));
}

BoundaryConditionSet cavity_boundaries(const NativeMesh& mesh)
{
    BoundaryConditionSet result;
    std::vector<std::string> names;
    if (const auto composite = std::get_if<NativeMesh::MultiRegionPtr>(&mesh.variant()))
    {
        for (const auto batch : (*composite)->boundary_batch_ids())
            names.push_back((*composite)->boundary_batch_name(batch));
    }
    else
        names = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    // Configuration is replicated, including patches absent from a rank's halo.
    for (const auto& name : names)
    {
        result.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        result.pressure[name] = {BoundaryConditionType::Neumann, 0.};
    }
    if (result.velocity.contains("ymax"))
        result.velocity["ymax"] = {BoundaryConditionType::Dirichlet, {.3, 0., 0.}};
    return result;
}

std::vector<double> values(const Pack::multi_vector_type& field)
{
    const auto data = field.getLocalViewHost(Tpetra::Access::ReadOnly);
    std::vector<double> result;
    for (size_t row = 0; row < data.extent(0); ++row)
        for (size_t component = 0; component < data.extent(1); ++component)
            result.push_back(data(row, component));
    return result;
}

double difference(const Vector& a, const Vector& b)
{
    Vector delta(a, Teuchos::Copy);
    delta.update(-1., b, 1.);
    return delta.norm2();
}

void set_direction(Vector& direction)
{
    auto data = direction.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (size_t row = 0; row < data.extent(0); ++row)
    {
        const auto gid = direction.getMap()->getGlobalElement(static_cast<LO>(row));
        data(row, 0) = .02 * std::sin(.31 * (static_cast<double>(gid) + 1.));
    }
}

struct State
{
    SP<const NativeMesh> mesh;
    NonlinearProblem::velocity_field_type velocity{VectorCellFieldDescriptor<Pack>("velocity"), mesh, vec3<double>{}};
    NonlinearProblem::field_type pressure{ScalarCellFieldDescriptor<Pack>("pressure"), mesh, 0.};
    NonlinearProblem::face_flux_field_type flux{ScalarFaceFieldDescriptor<Pack>("flux"), mesh, 0.};
    BoundaryConditionSet boundaries = cavity_boundaries(*mesh);
    TimeStepperOptions time;
    NonlinearSolverOptions nonlinear;
    LinearSolverOptions linear;

    explicit State(SP<const NativeMesh> selected_mesh = cavity_mesh()) : mesh(std::move(selected_mesh))
    {
        time.time_step = .04;
        time.kinematic_viscosity = .04;
        time.pressure_velocity_coupling = PressureVelocityCoupling::CoupledNonlinear;
        nonlinear.maximum_iterations = 40;
        nonlinear.absolute_tolerance = 1.e-11;
        nonlinear.relative_tolerance = 1.e-9;
        nonlinear.continuity_tolerance = 1.e-10;
        nonlinear.forcing_initial = .02;
        nonlinear.forcing_maximum = .2;
        nonlinear.forcing_minimum = 1.e-9;
        linear.max_iterations = 250;
        linear.tolerance = 1.e-11;
        seed(velocity, pressure);
    }

    static void seed(NonlinearProblem::velocity_field_type& u, NonlinearProblem::field_type& p)
    {
        const auto& mesh = u.mesh();
        for (size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
        {
            const auto lid = static_cast<LO>(cell);
            const auto center = mesh.cell_centroid(lid);
            // Interior transport fluxes are safely away from upwind switches.
            u.set_owned_value(
                lid, {.16 + .03 * std::sin(2. * center.y), .09 + .02 * std::cos(center.x), .04 + .01 * center.z});
            p.set_owned_value(lid, .03 * (center.x + .3 * center.y));
        }
        u.sync_ghosts();
        p.sync_ghosts();
    }

    std::unique_ptr<NonlinearProblem> problem(
        double density = 1., const NonlinearProblem::continuity_target_type* target = nullptr)
    {
        return std::make_unique<NonlinearProblem>(
            mesh, velocity, pressure, boundaries, time, nonlinear, density, target);
    }
};

void expect_physical_continuity(const State& state, const NonlinearProblem& problem, double tolerance)
{
    EXPECT_LT(problem.continuity().maximum, tolerance);
    const auto flux = state.flux.owned_read_view();
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<LO>(cell);
        // Include the pressure-gauge cell, whose algebraic row is not continuity.
        EXPECT_LT(std::abs(FVM::cell_flux_balance<Pack>(*state.mesh, state.flux, flux, lid)), tolerance);
        if (state.mesh->cell_global_id(lid) == 0)
            EXPECT_NEAR(state.pressure.value(lid), 0., tolerance);
    }
}

#ifdef SIMPLEFLUID_ENABLE_NOX
NonlinearCallbacks affine_callbacks(Teuchos::RCP<const Pack::map_type> map, double coefficient, double root)
{
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(static_cast<LO>(row));
        const Teuchos::Array<Pack::global_ordinal_type> columns{gid};
        const Teuchos::Array<double> coefficients{coefficient};
        matrix->insertGlobalValues(gid, columns(), coefficients());
    }
    matrix->fillComplete(map, map);
    NonlinearCallbacks callbacks;
    callbacks.map = std::move(map);
    callbacks.residual = [coefficient, root](const Vector& x, Vector& residual)
    {
        const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto output = residual.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t row = 0; row < input.extent(0); ++row)
            output(row, 0) = coefficient * (input(row, 0) - root);
        return true;
    };
    callbacks.linearize = [matrix](const Vector&) { return NonlinearLinearization{matrix, Teuchos::null}; };
    return callbacks;
}
#endif
} // namespace

TEST(CoupledNonlinearProblemTest, ResidualTrialsPreserveAcceptedHistoryPressureAndTarget)
{
    State state;
    std::vector<double> rates(state.mesh->num_owned_cells(), 0.);
    for (size_t cell = 0; cell < rates.size(); ++cell)
    {
        const auto gid = state.mesh->cell_global_id(static_cast<LO>(cell));
        rates[cell] = gid == 0 ? .002 : (gid == 1 ? -.002 : 0.);
    }
    const NonlinearProblem::continuity_target_type target(state.mesh, rates, 17);
    auto problem = state.problem(1., &target);
    const auto callbacks = problem->callbacks();
    auto x = problem->pack_initial();
    Vector y(*x, Teuchos::Copy), direction(callbacks.map), first(callbacks.map), trial(callbacks.map),
        repeat(callbacks.map);
    set_direction(direction);
    y.update(.2, direction, 1.);
    const auto accepted_u = values(state.velocity.owned_data());
    const auto accepted_p = values(state.pressure.owned_data());

    ASSERT_TRUE(callbacks.residual(*x, first));
    ASSERT_TRUE(callbacks.residual(y, trial));
    ASSERT_TRUE(callbacks.residual(*x, repeat));
    EXPECT_GT(difference(first, trial), 1.e-5);
    EXPECT_LT(difference(first, repeat), 1.e-14);
    EXPECT_EQ(values(state.velocity.owned_data()), accepted_u);
    EXPECT_EQ(values(state.pressure.owned_data()), accepted_p);
    EXPECT_EQ(
        std::vector<double>(target.owned_integrated_rates().begin(), target.owned_integrated_rates().end()), rates);
    EXPECT_EQ(target.generation(), 17U);

    // The physical-step context owns its snapshot even if its caller changes fields.
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        state.velocity.set_owned_value(static_cast<LO>(cell), {5., -2., 1.});
        state.pressure.set_owned_value(static_cast<LO>(cell), 31.);
    }
    ASSERT_TRUE(callbacks.residual(*x, repeat));
    EXPECT_LT(difference(first, repeat), 1.e-14);
}

TEST(CoupledNonlinearProblemTest, ContinuityTargetChangesIntegratedRowsAndGaugeMetric)
{
    State state;
    std::vector<double> rates(state.mesh->num_owned_cells(), 0.);
    for (size_t cell = 0; cell < rates.size(); ++cell)
    {
        const auto gid = state.mesh->cell_global_id(static_cast<LO>(cell));
        rates[cell] = gid == 0 ? .002 : (gid == 1 ? -.002 : 0.);
    }
    const NonlinearProblem::continuity_target_type target(state.mesh, rates, 5);
    auto zero = state.problem();
    auto sourced = state.problem(1., &target);
    auto x = zero->pack_initial();
    Vector f0(x->getMap()), f1(x->getMap());
    ASSERT_TRUE(zero->callbacks().residual(*x, f0));
    ASSERT_TRUE(sourced->callbacks().residual(*x, f1));
    const auto a = f0.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto b = f1.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t cell = 0; cell < rates.size(); ++cell)
    {
        const auto gid = state.mesh->cell_global_id(static_cast<LO>(cell));
        for (size_t component = 0; component < 3; ++component)
            EXPECT_DOUBLE_EQ(a(4 * cell + component, 0), b(4 * cell + component, 0));
        EXPECT_NEAR(b(4 * cell + 3, 0) - a(4 * cell + 3, 0), gid == 0 ? 0. : -rates[cell], 1.e-14);
    }
    sourced->commit(*x, state.velocity, state.pressure, state.flux);
    const auto face_values = state.flux.owned_read_view();
    double local_max = 0., global_max = 0.;
    for (size_t cell = 0; cell < rates.size(); ++cell)
        local_max = std::max(local_max,
            std::abs(FVM::cell_flux_balance<Pack>(*state.mesh, state.flux, face_values, static_cast<LO>(cell)) -
                     rates[cell]));
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_max, &global_max);
    EXPECT_NEAR(sourced->continuity().maximum, global_max, 1.e-14);
}

TEST(CoupledNonlinearProblemTest, PressureUsesReferenceDensityAndCellInterleavedGlobalIds)
{
    State state(cavity_mesh(true));
    const auto accepted_u = values(state.velocity.owned_data());
    const auto accepted_p = values(state.pressure.owned_data());
    const auto cells = state.mesh->owned_cell_map();
    const auto gauge_gid = cells->getMinAllGlobalIndex();
    const double local_gauge_pressure =
        cells->isNodeGlobalElement(gauge_gid) ? state.pressure.value(cells->getLocalElement(gauge_gid)) : 0.;
    double gauge_pressure = 0.;
    Teuchos::reduceAll(*cells->getComm(), Teuchos::REDUCE_SUM, 1, &local_gauge_pressure, &gauge_pressure);
    auto problem = state.problem(7.);
    auto x = problem->pack_initial();
    const auto data = x->getLocalViewHost(Tpetra::Access::ReadOnly);
    EXPECT_EQ(x->getGlobalLength(), 4 * state.mesh->owned_cell_map()->getGlobalNumElements());
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        const auto gid = state.mesh->cell_global_id(static_cast<LO>(cell));
        for (int component = 0; component < 4; ++component)
            EXPECT_EQ(x->getMap()->getGlobalElement(static_cast<LO>(4 * cell + component)), 4 * gid + component);
        EXPECT_DOUBLE_EQ(data(4 * cell + 3, 0), (accepted_p[cell] - gauge_pressure) / 7.);
    }
    problem->commit(*x, state.velocity, state.pressure, state.flux);
    EXPECT_EQ(values(state.velocity.owned_data()), accepted_u);
    const auto roundtrip_p = values(state.pressure.owned_data());
    for (size_t cell = 0; cell < accepted_p.size(); ++cell)
        EXPECT_NEAR(roundtrip_p[cell], accepted_p[cell] - gauge_pressure, 1.e-16);
}

TEST(CoupledNonlinearProblemTest, InitialPressureGaugeRemovesLargeOffsetsWithinInputRoundoff)
{
    State state(cavity_mesh(true));
    const double density = 7.;
    const double offset = 7.e6;
    auto reference = state.problem(density);
    auto expected = reference->pack_initial();
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<LO>(cell);
        state.pressure.set_owned_value(lid, state.pressure.value(lid) + offset);
    }
    state.pressure.sync_ghosts();
    const auto accepted_pressure = values(state.pressure.owned_data());
    auto shifted = state.problem(density);
    auto actual = shifted->pack_initial();
    const auto a = expected->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto b = actual->getLocalViewHost(Tpetra::Access::ReadOnly);
    // The two rounded inputs p and p_g each carry at most one offset-sized ULP.
    const double input_roundoff =
        2. * (std::nextafter(offset, std::numeric_limits<double>::infinity()) - offset) / density;
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        for (size_t component = 0; component < 3; ++component)
            EXPECT_DOUBLE_EQ(a(4 * cell + component, 0), b(4 * cell + component, 0));
        EXPECT_NEAR(a(4 * cell + 3, 0), b(4 * cell + 3, 0), input_roundoff);
    }
    EXPECT_EQ(values(state.pressure.owned_data()), accepted_pressure);
}

TEST(CoupledNonlinearProblemTest, AnalyticJacobianMatchesDirectionalDifferencesForBothBackends)
{
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        auto problem = state.problem();
        const auto callbacks = problem->callbacks();
        auto x = problem->pack_initial();
        Vector direction(callbacks.map), analytic(callbacks.map);
        set_direction(direction);
        const auto linearization = callbacks.linearize(*x);
        ASSERT_FALSE(linearization.jacobian.is_null());
        linearization.jacobian->apply(direction, analytic);
        for (const double epsilon : {1.e-4, 1.e-5})
        {
            SCOPED_TRACE(epsilon);
            Vector plus(*x, Teuchos::Copy), minus(*x, Teuchos::Copy), fp(callbacks.map), fm(callbacks.map);
            plus.update(epsilon, direction, 1.);
            minus.update(-epsilon, direction, 1.);
            ASSERT_TRUE(callbacks.residual(plus, fp));
            ASSERT_TRUE(callbacks.residual(minus, fm));
            fp.update(-1., fm, 1.);
            fp.scale(.5 / epsilon);
            EXPECT_LT(difference(fp, analytic) / analytic.norm2(), 2.e-6);
        }
        Vector retained(callbacks.map);
        linearization.jacobian->apply(direction, retained);
        EXPECT_LT(difference(analytic, retained), 1.e-14);
    }
}

TEST(CoupledNonlinearProblemTest, PlanarSlipJacobianMatchesDirectionsAndTrialsPreserveAcceptedState)
{
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state(planar_cavity_mesh());
        state.boundaries.velocity["zmin"] = {BoundaryConditionType::Slip, {}};
        state.boundaries.velocity["zmax"] = {BoundaryConditionType::Slip, {}};
        state.time.coupled_operator_backend = backend;
        auto problem = state.problem();
        const auto callbacks = problem->callbacks();
        auto x = problem->pack_initial();
        const auto accepted_u = values(state.velocity.owned_data());
        const auto accepted_p = values(state.pressure.owned_data());
        const auto accepted_flux = values(state.flux.owned_data());
        Vector direction(callbacks.map), analytic(callbacks.map), first(callbacks.map), repeat(callbacks.map);
        // Include nonzero z components: projecting both the trial and its
        // direction must remove the normal flux on the one-layer slip faces.
        set_direction(direction);
        ASSERT_TRUE(callbacks.residual(*x, first));
        const auto linearization = callbacks.linearize(*x);
        linearization.jacobian->apply(direction, analytic);
        for (const double epsilon : {1.e-4, 1.e-5})
        {
            SCOPED_TRACE(epsilon);
            Vector plus(*x, Teuchos::Copy), minus(*x, Teuchos::Copy), fp(callbacks.map), fm(callbacks.map);
            plus.update(epsilon, direction, 1.);
            minus.update(-epsilon, direction, 1.);
            ASSERT_TRUE(callbacks.residual(plus, fp));
            ASSERT_TRUE(callbacks.residual(minus, fm));
            fp.update(-1., fm, 1.);
            fp.scale(.5 / epsilon);
            EXPECT_LT(difference(fp, analytic) / analytic.norm2(), 2.e-6);
        }
        ASSERT_TRUE(callbacks.residual(*x, repeat));
        EXPECT_LT(difference(first, repeat), 1.e-14);
        EXPECT_EQ(values(state.velocity.owned_data()), accepted_u);
        EXPECT_EQ(values(state.pressure.owned_data()), accepted_p);
        EXPECT_EQ(values(state.flux.owned_data()), accepted_flux);
    }
}

TEST(CoupledNonlinearProblemTest, PicardMatchesNewtonAtRestButOmitsConvectiveDerivatives)
{
    State state;
    state.nonlinear.linearization = CoupledLinearization::Picard;
    auto picard = state.problem();
    state.nonlinear.linearization = CoupledLinearization::AnalyticNewton;
    auto newton = state.problem();
    auto x = newton->pack_initial();
    Vector direction(x->getMap()), jp(x->getMap()), jn(x->getMap());
    set_direction(direction);
    picard->callbacks().linearize(*x).jacobian->apply(direction, jp);
    newton->callbacks().linearize(*x).jacobian->apply(direction, jn);
    EXPECT_GT(difference(jp, jn), 1.e-6);
    x->putScalar(0.);
    picard->callbacks().linearize(*x).jacobian->apply(direction, jp);
    newton->callbacks().linearize(*x).jacobian->apply(direction, jn);
    EXPECT_LT(difference(jp, jn), 1.e-13);
}

TEST(CoupledNonlinearProblemTest, ZeroFluxGeneralizedDerivativeConservesMomentumAcrossInteriorFaces)
{
    State state;
    for (auto& [name, boundary] : state.boundaries.velocity)
        boundary = {BoundaryConditionType::NoSlip, {}};
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<LO>(cell);
        // Every x-face has exactly zero flux but different tangential donors.
        state.velocity.set_owned_value(lid, {0., .1 + .03 * state.mesh->cell_centroid(lid).x, 0.});
        state.pressure.set_owned_value(lid, 0.);
    }
    state.velocity.sync_ghosts();
    state.pressure.sync_ghosts();
    state.nonlinear.linearization = CoupledLinearization::Picard;
    auto picard = state.problem();
    state.nonlinear.linearization = CoupledLinearization::AnalyticNewton;
    auto newton = state.problem();
    auto x = newton->pack_initial();
    Vector direction(x->getMap()), jp(x->getMap()), correction(x->getMap());
    set_direction(direction);
    picard->callbacks().linearize(*x).jacobian->apply(direction, jp);
    newton->callbacks().linearize(*x).jacobian->apply(direction, correction);
    correction.update(-1., jp, 1.);
    EXPECT_GT(correction.norm2(), 1.e-6);
    std::array<double, 3> local{}, global{};
    const auto data = correction.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
        for (size_t component = 0; component < 3; ++component)
            local[component] += data(4 * cell + component, 0);
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 3, local.data(), global.data());
    // A single donor choice per shared zero-flux face gives equal/opposite K actions.
    for (const auto balance : global)
        EXPECT_NEAR(balance, 0., 1.e-12);
}

TEST(CoupledNonlinearProblemTest, RetainedJacobianAndPreconditionerSurviveAnotherLinearization)
{
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        auto problem = state.problem();
        const auto callbacks = problem->callbacks();
        auto x = problem->pack_initial();
        Vector y(*x, Teuchos::Copy), direction(callbacks.map);
        set_direction(direction);
        y.update(.2, direction, 1.);
        const auto first = callbacks.linearize(*x);
        ASSERT_FALSE(first.right_preconditioner.is_null());
        Vector jacobian_before(callbacks.map), inverse_before(callbacks.map);
        first.jacobian->apply(direction, jacobian_before);
        first.right_preconditioner->apply(direction, inverse_before);
        const auto second = callbacks.linearize(y);
        Vector jacobian_after(callbacks.map), inverse_after(callbacks.map), new_action(callbacks.map);
        first.jacobian->apply(direction, jacobian_after);
        first.right_preconditioner->apply(direction, inverse_after);
        second.jacobian->apply(direction, new_action);
        EXPECT_GT(difference(jacobian_before, new_action), 1.e-6);
        EXPECT_LT(difference(jacobian_before, jacobian_after), 1.e-14);
        EXPECT_LT(difference(inverse_before, inverse_after), 1.e-12);
    }
}

TEST(CoupledNonlinearProblemTest, RetainedPreconditionerAlonePreservesItsNumericalGeneration)
{
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        auto problem = state.problem();
        const auto callbacks = problem->callbacks();
        auto x = problem->pack_initial();
        Vector y(*x, Teuchos::Copy), direction(callbacks.map), before(callbacks.map), after(callbacks.map);
        set_direction(direction);
        y.update(.2, direction, 1.);
        // Discard the Jacobian immediately; the preconditioner alone owns its generation.
        const auto retained = callbacks.linearize(*x).right_preconditioner;
        retained->apply(direction, before);
        const auto next = callbacks.linearize(y);
        retained->apply(direction, after);
        EXPECT_LT(difference(before, after), 1.e-12);
        ASSERT_FALSE(next.jacobian.is_null());
    }
}

TEST(CoupledNonlinearProblemTest, UniformTranslationIsAnExactDiscreteSolution)
{
    State state;
    const vec3<double> uniform{.2, .1, .05};
    for (auto& [name, boundary] : state.boundaries.velocity)
        boundary = {BoundaryConditionType::Dirichlet, uniform};
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        state.velocity.set_owned_value(static_cast<LO>(cell), uniform);
        state.pressure.set_owned_value(static_cast<LO>(cell), 0.);
    }
    state.velocity.sync_ghosts();
    state.pressure.sync_ghosts();
    auto problem = state.problem();
    auto x = problem->pack_initial();
    Vector residual(x->getMap());
    ASSERT_TRUE(problem->callbacks().residual(*x, residual));
    EXPECT_LT(residual.norm2(), 1.e-13);
    problem->commit(*x, state.velocity, state.pressure, state.flux);
    expect_physical_continuity(state, *problem, 1.e-13);
}

TEST(CoupledNonlinearProblemTest, PeriodicTranslationAndDerivativePreserveWrappedInterfaces)
{
    State state(std::make_shared<NativeMesh>(test::periodic_regions()));
    state.time.coupled_operator_backend = CoupledOperatorBackend::BlockComposite;
    const vec3<double> uniform{.2, .1, .05};
    for (auto& [name, boundary] : state.boundaries.velocity)
        boundary = {BoundaryConditionType::Dirichlet, uniform};
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
    {
        state.velocity.set_owned_value(static_cast<LO>(cell), uniform);
        state.pressure.set_owned_value(static_cast<LO>(cell), 0.);
    }
    state.velocity.sync_ghosts();
    state.pressure.sync_ghosts();
    auto problem = state.problem();
    const auto callbacks = problem->callbacks();
    auto x = problem->pack_initial();
    Vector residual(callbacks.map), direction(callbacks.map), analytic(callbacks.map);
    ASSERT_TRUE(callbacks.residual(*x, residual));
    EXPECT_LT(residual.norm2(), 1.e-13);
    problem->commit(*x, state.velocity, state.pressure, state.flux);
    expect_physical_continuity(state, *problem, 1.e-13);
    set_direction(direction);
    callbacks.linearize(*x).jacobian->apply(direction, analytic);
    const double epsilon = 1.e-5;
    Vector plus(*x, Teuchos::Copy), minus(*x, Teuchos::Copy), fp(callbacks.map), fm(callbacks.map);
    plus.update(epsilon, direction, 1.);
    minus.update(-epsilon, direction, 1.);
    ASSERT_TRUE(callbacks.residual(plus, fp));
    ASSERT_TRUE(callbacks.residual(minus, fm));
    fp.update(-1., fm, 1.);
    fp.scale(.5 / epsilon);
    EXPECT_LT(difference(fp, analytic) / analytic.norm2(), 2.e-6);
}

TEST(CoupledNonlinearProblemTest, RejectsUnsupportedPressureOutletAndVelocityExtrapolation)
{
    State state;
    state.boundaries.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 0.};
    EXPECT_THROW(state.problem(), std::invalid_argument);
    state.boundaries.pressure["xmax"] = {BoundaryConditionType::Neumann, 0.};
    state.boundaries.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
    EXPECT_THROW(state.problem(), std::invalid_argument);
}

TEST(CoupledNonlinearProblemTest, RejectsObliqueSlipEvenOnAnOrthogonalMesh)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The rotated SemiStructuredXY_Z fixture supports serial distribution only.";
    // Two orthogonal square prisms rotated 45 degrees in XY. Prescribed
    // velocity remains supported, but oblique Slip is outside the contract.
    auto mesh = std::make_shared<Meshes::SemiStructuredXY_Z>(
        Arr<MeshUtils::Vec3>{{0., 0., 0.}, {1., 1., 0.}, {2., 2., 0.}, {-1., 1., 0.}, {0., 2., 0.}, {1., 3., 0.}},
        Arr<Arr<unsigned>>{{0, 1, 4, 3}, {1, 2, 5, 4}}, ArrReal{0., 1.},
        Arr<Meshes::SemiStructuredXY_Z::BoundaryEdge>{{0, 1, "ymin"}, {1, 2, "ymin"}, {2, 5, "xmax"},
            {5, 4, "ymax"}, {4, 3, "ymax"}, {3, 0, "xmin"}});
    State state(std::make_shared<NativeMesh>(mesh));
    // A stationary prescribed wall avoids the rotated lid's net inflow.
    state.boundaries.velocity["ymax"] = {BoundaryConditionType::NoSlip, {}};
    EXPECT_NO_THROW(state.problem());
    state.boundaries.velocity["xmin"] = {BoundaryConditionType::Slip, {}};
    EXPECT_THROW(state.problem(), std::invalid_argument);
}

TEST(CoupledNonlinearProblemTest, RejectsInvalidPhysicalCoefficientsAndIncompatibleFlux)
{
    State state;
    EXPECT_THROW(state.problem(0.), std::invalid_argument);
    EXPECT_THROW(state.problem(std::numeric_limits<double>::infinity()), std::invalid_argument);
    state.time.time_step = 0.;
    EXPECT_THROW(state.problem(), std::invalid_argument);
    state.time.time_step = .04;
    state.time.kinematic_viscosity = -.01;
    EXPECT_THROW(state.problem(), std::invalid_argument);
    state.time.kinematic_viscosity = .04;
    state.boundaries.velocity["xmin"] = {BoundaryConditionType::Dirichlet, {.1, 0., 0.}};
    EXPECT_THROW(state.problem(), std::invalid_argument);
}

#ifdef SIMPLEFLUID_ENABLE_NOX
TEST(CoupledNonlinearSolverTest, OwningSolverResetsConvergenceBaselineAndReusesVectorAndLinearCaches)
{
    State state;
    const auto map = state.problem()->callbacks().map;
    NOXNonlinearSolver solver(affine_callbacks(map, 1., 1.));
    state.nonlinear.relative_tolerance = .1;
    Vector x(map);
    x.putScalar(1.e6);
    const auto first = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(first.converged) << first.reason;
    const auto first_cache = solver.cache_statistics();
    EXPECT_GT(first_cache.vector_cache_builds, 0U);
    EXPECT_GT(first_cache.linear_solver_builds, 0U);

    x.putScalar(2.);
    const auto second = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(second.converged) << second.reason;
    // Reusing the previous 1e6-scale baseline would accept x=2 without a solve.
    EXPECT_GT(second.nonlinear_iterations, 0);
    for (const auto value : values(x))
        EXPECT_NEAR(value, 1., 1.e-10);
    const auto second_cache = solver.cache_statistics();
    EXPECT_EQ(second_cache.vector_cache_builds, first_cache.vector_cache_builds);
    EXPECT_EQ(second_cache.linear_solver_builds, first_cache.linear_solver_builds);
    EXPECT_GT(second_cache.linearization_generations, first_cache.linearization_generations);
}

TEST(CoupledNonlinearSolverTest, OwningSolverReplacesCallbackContextAndJacobianOnTheSameMap)
{
    State state;
    const auto map = state.problem()->callbacks().map;
    NOXNonlinearSolver solver(affine_callbacks(map, 1., 1.));
    Vector x(map);
    x.putScalar(0.);
    const auto first = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(first.converged) << first.reason;
    const auto first_cache = solver.cache_statistics();

    solver.set_callbacks(affine_callbacks(map, 2., 3.));
    const auto second = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(second.converged) << second.reason;
    EXPECT_GT(second.nonlinear_iterations, 0);
    for (const auto value : values(x))
        EXPECT_NEAR(value, 3., 1.e-10);
    const auto second_cache = solver.cache_statistics();
    EXPECT_EQ(second_cache.vector_cache_builds, first_cache.vector_cache_builds);
    EXPECT_EQ(second_cache.linear_solver_builds, first_cache.linear_solver_builds);
    EXPECT_GT(second_cache.linearization_generations, first_cache.linearization_generations);
}

TEST(CoupledNonlinearSolverTest, OwningSolverKeepsPhysicalCallbackContextAliveAfterCallerDestruction)
{
    std::unique_ptr<NOXNonlinearSolver> solver;
    Teuchos::RCP<Vector> x;
    NonlinearSolverOptions nonlinear;
    LinearSolverOptions linear;
    {
        State state;
        nonlinear = state.nonlinear;
        linear = state.linear;
        auto problem = state.problem();
        x = problem->pack_initial();
        solver = std::make_unique<NOXNonlinearSolver>(problem->callbacks());
    }
    const auto result = solver->solve(*x, nonlinear, linear);
    ASSERT_TRUE(result.converged) << result.reason;
    EXPECT_GT(result.nonlinear_iterations, 0);
    const auto data = x->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t row = 0; row < x->getLocalLength(); ++row)
    {
        EXPECT_TRUE(std::isfinite(data(row, 0)));
        if (x->getMap()->getGlobalElement(static_cast<LO>(row)) == 3)
            EXPECT_LE(std::abs(data(row, 0)), nonlinear.pressure_scale * nonlinear.absolute_tolerance);
    }
}

TEST(CoupledNonlinearSolverTest, ArmijoRejectsInadmissibleTrialsAndExhaustionIsTransactional)
{
    State state;
    NonlinearCallbacks callbacks;
    callbacks.map = state.problem()->callbacks().map;
    callbacks.residual = [rank = callbacks.map->getComm()->getRank()](const Vector& x, Vector& residual)
    {
        const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto output = residual.getLocalViewHost(Tpetra::Access::OverwriteAll);
        bool admissible = true;
        for (size_t row = 0; row < input.extent(0); ++row)
        {
            const auto value = input(row, 0);
            admissible = admissible && value > 0. && value < 2.;
            output(row, 0) = value * value * value - 1.;
        }
        // Only one rank owns the domain restriction; all ranks must backtrack.
        return rank != 0 || admissible;
    };
    callbacks.linearize = [map = callbacks.map](const Vector& x)
    {
        auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 1));
        const auto input = x.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t row = 0; row < input.extent(0); ++row)
        {
            const auto gid = map->getGlobalElement(static_cast<LO>(row));
            const Teuchos::Array<Pack::global_ordinal_type> columns{gid};
            const Teuchos::Array<double> coefficients{3. * input(row, 0) * input(row, 0)};
            matrix->insertGlobalValues(gid, columns(), coefficients());
        }
        matrix->fillComplete(map, map);
        return NonlinearLinearization{matrix, Teuchos::null};
    };
    Vector x(callbacks.map);
    NOXNonlinearSolver solver(callbacks);
    x.putScalar(.1);
    const auto result = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(result.converged) << result.reason;
    EXPECT_GT(result.line_search_rejections, 0);
    for (const auto value : values(x))
        EXPECT_NEAR(value, 1., 1.e-9);

    x.putScalar(.1);
    const auto initial = values(x);
    state.nonlinear.maximum_backtracks = 0;
    const auto failure = solver.solve(x, state.nonlinear, state.linear);
    EXPECT_FALSE(failure.converged);
    EXPECT_GT(failure.line_search_rejections, 0);
    EXPECT_EQ(values(x), initial);

    state.nonlinear.maximum_backtracks = 12;
    const auto recovered = solver.solve(x, state.nonlinear, state.linear);
    ASSERT_TRUE(recovered.converged) << recovered.reason;
    for (const auto value : values(x))
        EXPECT_NEAR(value, 1., 1.e-9);
}

TEST(CoupledNonlinearSolverTest, RankDivergentControlsAndScalePresenceFailCollectively)
{
    State state;
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires at least two MPI ranks.";
    auto problem = state.problem();
    auto callbacks = problem->callbacks();
    auto x = problem->pack_initial();
    auto controls = state.nonlinear;
    controls.maximum_iterations += comm->getRank() == 0 ? 1 : 0;
    EXPECT_THROW(solve_nox(callbacks, *x, controls, state.linear), std::invalid_argument);
    controls.maximum_iterations = comm->getRank() == 0 ? 0 : state.nonlinear.maximum_iterations;
    EXPECT_THROW(solve_nox(callbacks, *x, controls, state.linear), std::invalid_argument);
    if (comm->getRank() == 0)
        callbacks.state_scale = Teuchos::null;
    EXPECT_THROW(solve_nox(callbacks, *x, state.nonlinear, state.linear), std::invalid_argument);
}

TEST(CoupledNonlinearSolverTest, HugeFiniteResidualCannotFalselyConvergeFromNormOverflow)
{
    State state;
    auto problem = state.problem();
    auto callbacks = problem->callbacks();
    callbacks.state_scale = Teuchos::null;
    callbacks.residual_scale = Teuchos::null;
    callbacks.residual = [](const Vector&, Vector& residual)
    {
        residual.putScalar(1.e200);
        return true;
    };
    auto x = problem->pack_initial();
    const auto before = values(*x);
    const auto result = solve_nox(callbacks, *x, state.nonlinear, state.linear);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(values(*x), before);
}

TEST(CoupledNonlinearSolverTest, RankLocalNonfiniteResidualFailsCollectivelyWithoutPublishing)
{
    State state;
    auto problem = state.problem();
    auto callbacks = problem->callbacks();
    callbacks.residual = [rank = callbacks.map->getComm()->getRank()](const Vector&, Vector& residual)
    {
        residual.putScalar(rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 1.);
        return true;
    };
    auto x = problem->pack_initial();
    const auto before = values(*x);
    const auto result = solve_nox(callbacks, *x, state.nonlinear, state.linear);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(values(*x), before);
}

TEST(CoupledNonlinearSolverTest, NoncontiguousOwnershipKeepsPressureGaugeOnItsOwningRank)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2)
        GTEST_SKIP() << "Requires exactly two MPI ranks.";
    State state(cavity_mesh(true));
    state.time.coupled_operator_backend = CoupledOperatorBackend::BlockComposite;
    const auto owned = state.mesh->owned_cell_map();
    bool has_gap = false;
    for (size_t cell = 1; cell < state.mesh->num_owned_cells(); ++cell)
        has_gap = has_gap || owned->getGlobalElement(static_cast<LO>(cell)) !=
                                 owned->getGlobalElement(static_cast<LO>(cell - 1)) + 1;
    EXPECT_TRUE(has_gap);
    EXPECT_EQ(owned->isNodeGlobalElement(0), comm->getRank() == 1);
    auto problem = state.problem(7.);
    auto x = problem->pack_initial();
    const auto result = solve_nox(problem->callbacks(), *x, state.nonlinear, state.linear);
    ASSERT_TRUE(result.converged) << result.reason;
    problem->commit(*x, state.velocity, state.pressure, state.flux);
    expect_physical_continuity(state, *problem, 1.e-9);
}

TEST(CoupledNonlinearSolverTest, NewtonAndPicardConvergeToSameConservativeSolutionAcrossBackends)
{
    std::vector<double> reference;
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
        for (auto method : {CoupledLinearization::AnalyticNewton, CoupledLinearization::Picard})
        {
            SCOPED_TRACE(static_cast<int>(backend));
            SCOPED_TRACE(static_cast<int>(method));
            State state;
            state.time.coupled_operator_backend = backend;
            state.nonlinear.linearization = method;
            auto problem = state.problem(7.);
            auto x = problem->pack_initial();
            const auto before = values(*x);
            const auto result = solve_nox(problem->callbacks(), *x, state.nonlinear, state.linear);
            ASSERT_TRUE(result.converged) << result.reason;
            EXPECT_NE(values(*x), before);
            problem->commit(*x, state.velocity, state.pressure, state.flux);
            expect_physical_continuity(state, *problem, 1.e-9);
            const auto current = values(*x);
            if (reference.empty())
                reference = current;
            else
            {
                ASSERT_EQ(reference.size(), current.size());
                for (size_t i = 0; i < current.size(); ++i)
                    EXPECT_NEAR(current[i], reference[i], 1.e-7);
            }
        }
}

TEST(CoupledNonlinearSolverTest, PlanarSlipCavityConvergesWithExactlyZeroNormalBoundaryFlux)
{
    std::vector<double> reference;
    for (auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
        for (auto method : {CoupledLinearization::AnalyticNewton, CoupledLinearization::Picard})
        {
            SCOPED_TRACE(static_cast<int>(backend));
            SCOPED_TRACE(static_cast<int>(method));
            State state(planar_cavity_mesh());
            state.boundaries.velocity["zmin"] = {BoundaryConditionType::Slip, {}};
            state.boundaries.velocity["zmax"] = {BoundaryConditionType::Slip, {}};
            state.time.coupled_operator_backend = backend;
            state.nonlinear.linearization = method;
            auto problem = state.problem();
            auto x = problem->pack_initial();
            const auto accepted_u = values(state.velocity.owned_data());
            const auto accepted_p = values(state.pressure.owned_data());
            const auto accepted_flux = values(state.flux.owned_data());
            const auto result = solve_nox(problem->callbacks(), *x, state.nonlinear, state.linear);
            ASSERT_TRUE(result.converged) << result.reason;
            EXPECT_EQ(values(state.velocity.owned_data()), accepted_u);
            EXPECT_EQ(values(state.pressure.owned_data()), accepted_p);
            EXPECT_EQ(values(state.flux.owned_data()), accepted_flux);
            problem->commit(*x, state.velocity, state.pressure, state.flux);
            expect_physical_continuity(state, *problem, 1.e-9);
            const auto flux = state.flux.local_read_view();
            size_t slip_faces = 0;
            for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
            {
                const auto lid = static_cast<LO>(cell);
                for (const auto face : state.mesh->faces(lid))
                    if (state.mesh->is_boundary_face(face) && state.mesh->face_normal_outward(face, lid).z != 0.)
                    {
                        ++slip_faces;
                        EXPECT_DOUBLE_EQ(flux(face, 0), 0.);
                    }
            }
            EXPECT_EQ(slip_faces, 2 * state.mesh->num_owned_cells());
            const auto current = values(*x);
            if (reference.empty())
                reference = current;
            else
            {
                ASSERT_EQ(reference.size(), current.size());
                for (size_t i = 0; i < current.size(); ++i)
                    EXPECT_NEAR(current[i], reference[i], 1.e-7);
            }
        }
}

TEST(CoupledNonlinearSolverTest, PositiveReferenceScalingPreservesThePhysicalSolution)
{
    State state;
    auto baseline = state.problem(7.);
    auto x = baseline->pack_initial();
    ASSERT_TRUE(solve_nox(baseline->callbacks(), *x, state.nonlinear, state.linear).converged);
    state.nonlinear.velocity_scale = .7;
    state.nonlinear.pressure_scale = 2.3;
    state.nonlinear.momentum_residual_scale = 1.9;
    state.nonlinear.continuity_residual_scale = .4;
    auto scaled = state.problem(7.);
    auto y = scaled->pack_initial();
    ASSERT_TRUE(solve_nox(scaled->callbacks(), *y, state.nonlinear, state.linear).converged);
    EXPECT_LT(difference(*x, *y), 1.e-7);
    scaled->commit(*y, state.velocity, state.pressure, state.flux);
    expect_physical_continuity(state, *scaled, 1.e-9);
}

TEST(CoupledNonlinearSolverTest, LargeInitialPressureOffsetStillRequiresAnAbsoluteGaugeAndMassGate)
{
    State state;
    state.nonlinear.relative_tolerance = .1;
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
        state.pressure.set_owned_value(static_cast<LO>(cell), 7.e6);
    state.pressure.sync_ghosts();
    auto problem = state.problem(7.);
    const auto callbacks = problem->callbacks();
    auto x = problem->pack_initial();
    ASSERT_TRUE(static_cast<bool>(callbacks.convergence_gate));
    EXPECT_FALSE(callbacks.convergence_gate(*x));
    const auto result = solve_nox(callbacks, *x, state.nonlinear, state.linear);
    ASSERT_TRUE(result.converged) << result.reason;
    EXPECT_TRUE(callbacks.convergence_gate(*x));
    problem->commit(*x, state.velocity, state.pressure, state.flux);
    expect_physical_continuity(state, *problem, 1.e-9);
    const auto data = x->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
        if (state.mesh->cell_global_id(static_cast<LO>(cell)) == 0)
            EXPECT_LE(
                std::abs(data(4 * cell + 3, 0)) / state.nonlinear.pressure_scale, state.nonlinear.absolute_tolerance);
}

TEST(CoupledNonlinearSolverTest, NativeDriverCommitsOnceAndFailedBudgetPreservesAcceptedState)
{
    State state;
    state.time.nonlinear = state.nonlinear;
    FluidSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear);
    State::seed(solver.velocity(), solver.pressure());
    solver.step();
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_DOUBLE_EQ(solver.time(), state.time.time_step);
    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_LT(solver.last_volume_continuity_residuals().maximum, 1.e-9);
    solver.step();
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_DOUBLE_EQ(solver.time(), 2. * state.time.time_step);

    state.time.nonlinear.maximum_iterations = 1;
    state.time.nonlinear.relative_tolerance = 1.e-14;
    state.time.nonlinear.absolute_tolerance = 1.e-15;
    FluidSolver<Pack> failing(state.mesh, state.boundaries, state.time, state.linear);
    State::seed(failing.velocity(), failing.pressure());
    const auto accepted_u = values(failing.velocity().owned_data());
    const auto accepted_p = values(failing.pressure().owned_data());
    const auto accepted_flux = values(failing.pressure_corrected_face_fluxes().owned_data());
    EXPECT_THROW(failing.step(), std::runtime_error);
    EXPECT_EQ(failing.step_index(), 0);
    EXPECT_DOUBLE_EQ(failing.time(), 0.);
    EXPECT_EQ(values(failing.velocity().owned_data()), accepted_u);
    EXPECT_EQ(values(failing.pressure().owned_data()), accepted_p);
    EXPECT_EQ(values(failing.pressure_corrected_face_fluxes().owned_data()), accepted_flux);
}

TEST(CoupledNonlinearSolverTest, NativeDriverLinearFailurePreservesAcceptedFields)
{
    State state;
    state.time.nonlinear = state.nonlinear;
    state.time.nonlinear.forcing_initial = 1.e-10;
    state.time.nonlinear.forcing_minimum = 1.e-10;
    state.linear.max_iterations = 1;
    FluidSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear);
    State::seed(solver.velocity(), solver.pressure());
    const auto accepted_u = values(solver.velocity().owned_data());
    const auto accepted_p = values(solver.pressure().owned_data());
    EXPECT_THROW(solver.step(), std::runtime_error);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_DOUBLE_EQ(solver.time(), 0.);
    EXPECT_EQ(values(solver.velocity().owned_data()), accepted_u);
    EXPECT_EQ(values(solver.pressure().owned_data()), accepted_p);
}

TEST(CoupledNonlinearSolverTest, DerivedPhysicalSolverRejectsUnsupportedModeBeforeAdvancing)
{
    State state;
    state.time.nonlinear = state.nonlinear;
    EXPECT_THROW(
        {
            IncompressibleIsothermalSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear, 7.);
            solver.step();
        },
        std::invalid_argument);
}

TEST(CoupledNonlinearSolverTest, CustomMomentumOverrideRejectsModeBeforeItsHookOrStateChanges)
{
    class CustomMomentumSolver final : public FluidSolver<Pack>
    {
    public:
        using FluidSolver<Pack>::FluidSolver;
        bool hook_called = false;

    protected:
        LinearSolveSummary advance_momentum() override
        {
            hook_called = true;
            return {};
        }
    };
    State state;
    CustomMomentumSolver solver(state.mesh, state.boundaries, state.time, state.linear);
    State::seed(solver.velocity(), solver.pressure());
    const auto accepted_u = values(solver.velocity().owned_data());
    EXPECT_THROW(solver.step(), std::invalid_argument);
    EXPECT_FALSE(solver.hook_called);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_DOUBLE_EQ(solver.time(), 0.);
    EXPECT_EQ(values(solver.velocity().owned_data()), accepted_u);
}
#else
TEST(CoupledNonlinearSolverTest, DisabledBackendRejectsModeBeforeChangingAcceptedState)
{
    State state;
    FluidSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear);
    State::seed(solver.velocity(), solver.pressure());
    const auto accepted_u = values(solver.velocity().owned_data());
    const auto accepted_p = values(solver.pressure().owned_data());
    EXPECT_THROW(solver.step(), std::invalid_argument);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_DOUBLE_EQ(solver.time(), 0.);
    EXPECT_EQ(values(solver.velocity().owned_data()), accepted_u);
    EXPECT_EQ(values(solver.pressure().owned_data()), accepted_p);
}
#endif
