/** @file testCoupledNonlinearSolver.cc
 *  @brief Residual, derivative, transaction, and NOX integration contracts.
 */
#include "FVM/FaceFlux.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/CoupledNonlinearProblem.hh"
#include "solvers/BoussinesqSolver.hh"
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

TEST(CoupledNonlinearProblemTest, FrozenPhysicalBoussinesqMatchesNativeStressSourcesAndDerivative)
{
    for (const auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
        for (const bool density_feedback : {false, true})
        {
            State state;
            state.time.coupled_operator_backend = backend;
            state.time.thermal_expansion = .03;
            state.time.reference_temperature = 2.;
            BoussinesqModelOptions options;
            options.reference_density = 7.;
            options.density = 6.8;
            options.dynamic_viscosity = .19;
            NonlinearProblem::material_type material(state.mesh, options, state.time);
            NonlinearProblem::field_type temperature(state.mesh, "frozen_test_temperature");
            NonlinearProblem::field_type viscosity(state.mesh, "frozen_test_viscosity");
            NonlinearProblem::velocity_field_type k_gradient(state.mesh, "frozen_test_k_gradient");
            for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
            {
                const auto lid = static_cast<LO>(cell);
                const auto center = state.mesh->cell_centroid(lid);
                temperature.set_owned_value(lid, 2. + .5 * center.z);
                material.density.set_owned_value(lid, 6.8 + .15 * center.x);
                viscosity.set_owned_value(lid, .3 + .2 * center.x);
                k_gradient.set_owned_value(lid, {.01 * center.x, -.02 * center.y, .03});
            }
            temperature.sync_ghosts();
            material.density.sync_ghosts();
            viscosity.sync_ghosts();
            k_gradient.sync_ghosts();
            auto velocity_cache = FVM::cache_velocity_boundary_conditions<Pack>(state.mesh, state.boundaries);
            NonlinearProblem::boundary_cache_type boundary_mu;
            boundary_mu.mesh = state.mesh;
            for (const auto& [batch, boundary_values] : velocity_cache.value)
                boundary_mu.value[batch].assign(boundary_values.size(), .43);
            const NonlinearProblem::FrozenBoussinesqInput physical{
                temperature, material, density_feedback, &viscosity, &k_gradient, &boundary_mu};
            NonlinearProblem problem(state.mesh, state.velocity, state.pressure, state.boundaries,
                state.time, state.nonlinear, 7., nullptr, &physical);
            auto x = problem.pack_initial();
            const auto callbacks = problem.callbacks();
            Vector residual(callbacks.map), expected(callbacks.map), repeated(callbacks.map);
            ASSERT_TRUE(callbacks.residual(*x, residual));
            FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, NativeMesh> workspace(state.mesh);
            FVM::pressure_weighted_face_fluxes(state.velocity, state.pressure, state.time.time_step / 7.,
                velocity_cache, state.boundaries.pressure, workspace, state.flux, state.time.pressure_gradient_scheme);
            BoussinesqMomentumEquation<Pack, NativeMesh> equation(state.mesh);
            CoupledPressureVelocitySolver<Pack, NativeMesh> native_solver(state.mesh);
            const NonlinearProblem::continuity_target_type target(state.mesh);
            const auto native = native_solver.assemble(equation, state.velocity, state.pressure, temperature,
                state.flux, velocity_cache, state.boundaries, state.time, target, &material, 7., density_feedback,
                &viscosity, &k_gradient, &boundary_mu);
            native.linear_operator->apply(*x, expected);
            expected.update(-1., *native.rhs, 1.);
            EXPECT_LT(difference(residual, expected), 1.e-12);

            Vector direction(callbacks.map), action(callbacks.map), plus(*x, Teuchos::Copy), minus(*x, Teuchos::Copy),
                fplus(callbacks.map), fminus(callbacks.map);
            set_direction(direction);
            auto linearization = callbacks.linearize(*x);
            linearization.jacobian->apply(direction, action);
            constexpr double epsilon = 1.e-5;
            plus.update(epsilon, direction, 1.);
            minus.update(-epsilon, direction, 1.);
            ASSERT_TRUE(callbacks.residual(plus, fplus));
            ASSERT_TRUE(callbacks.residual(minus, fminus));
            fplus.update(-1., fminus, 1.);
            fplus.scale(.5 / epsilon);
            EXPECT_LT(difference(action, fplus), 1.e-8);

            temperature.put_value(100.);
            material.density.put_value(700.);
            viscosity.put_value(30.);
            k_gradient.put_value({9., 8., 7.});
            for (auto& [batch, boundary_values] : boundary_mu.value)
                std::fill(boundary_values.begin(), boundary_values.end(), 43.);
            ASSERT_TRUE(callbacks.residual(*x, repeated));
            EXPECT_LT(difference(residual, repeated), 1.e-14);
            auto later_linearization = callbacks.linearize(*x);
            later_linearization.jacobian->apply(direction, repeated);
            EXPECT_LT(difference(action, repeated), 1.e-12);
        }
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

namespace
{
std::unique_ptr<NonlinearProblem> workspace_problem(State& state, CoupledNonlinearWorkspace* workspace,
    double density = 1., const NonlinearProblem::continuity_target_type* target = nullptr,
    const NonlinearProblem::FrozenBoussinesqInput* physical = nullptr)
{
    return std::make_unique<NonlinearProblem>(state.mesh, state.velocity, state.pressure, state.boundaries, state.time,
        state.nonlinear, density, target, physical, workspace);
}

void expect_same_problem(NonlinearProblem& actual, NonlinearProblem& expected)
{
    const auto actual_callbacks = actual.callbacks(), expected_callbacks = expected.callbacks();
    const auto x = actual.pack_initial(), expected_x = expected.pack_initial();
    EXPECT_LT(difference(*x, *expected_x), 1.e-14);
    Vector actual_value(x->getMap()), expected_value(x->getMap()), direction(x->getMap());
    ASSERT_TRUE(actual_callbacks.residual(*x, actual_value));
    ASSERT_TRUE(expected_callbacks.residual(*x, expected_value));
    EXPECT_LT(difference(actual_value, expected_value), 1.e-12 * (1. + expected_value.norm2()));
    EXPECT_NEAR(actual.continuity().maximum, expected.continuity().maximum, 1.e-14);
    EXPECT_NEAR(actual.continuity().l2, expected.continuity().l2, 1.e-14);
    set_direction(direction);
    const auto actual_linear = actual_callbacks.linearize(*x), expected_linear = expected_callbacks.linearize(*x);
    actual_linear.jacobian->apply(direction, actual_value);
    expected_linear.jacobian->apply(direction, expected_value);
    EXPECT_LT(difference(actual_value, expected_value), 1.e-12 * (1. + expected_value.norm2()));
    actual_linear.right_preconditioner->apply(direction, actual_value);
    expected_linear.right_preconditioner->apply(direction, expected_value);
    EXPECT_LT(difference(actual_value, expected_value), 1.e-10 * (1. + expected_value.norm2()));
}
} // namespace

TEST(CoupledNonlinearProblemTest, RecycledWorkspaceRefreshesHistoryCoefficientsAndTargetWithoutStaticSchur)
{
    for (const auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        CoupledNonlinearWorkspace workspace;
        {
            auto first = workspace_problem(state, &workspace);
            EXPECT_EQ(first->statistics().workspace_builds, 1U);
            EXPECT_EQ(first->statistics().geometry_builds, 1U);
            EXPECT_EQ(first->statistics().schur_builds, 0U);
            const auto x = first->pack_initial();
            Vector residual(x->getMap());
            ASSERT_TRUE(first->callbacks().residual(*x, residual));
            EXPECT_EQ(first->statistics().schur_builds, 0U);
        }
        state.time.time_step = .13;
        state.time.kinematic_viscosity = .075;
        state.velocity.owned_data().scale(1.4);
        state.pressure.owned_data().scale(2.3);
        state.velocity.sync_ghosts();
        state.pressure.sync_ghosts();
        std::vector<double> rates(state.mesh->num_owned_cells(), 0.);
        for (size_t cell = 0; cell < rates.size(); ++cell)
        {
            const auto gid = state.mesh->cell_global_id(static_cast<LO>(cell));
            rates[cell] = gid == 0 ? .003 : (gid == 1 ? -.003 : 0.);
        }
        const NonlinearProblem::continuity_target_type target(state.mesh, rates, 21);
        auto reused = workspace_problem(state, &workspace, 7., &target);
        auto fresh = workspace_problem(state, nullptr, 7., &target);
        EXPECT_EQ(reused->statistics().workspace_reuses, 1U);
        EXPECT_EQ(reused->statistics().geometry_builds, 0U);
        EXPECT_EQ(reused->statistics().schur_builds, 0U);
        EXPECT_GE(reused->statistics().graph_reuses, 1U);
        expect_same_problem(*reused, *fresh);
        EXPECT_EQ(reused->statistics().schur_builds, 1U);
    }
}

TEST(CoupledNonlinearProblemTest, RetainedCallbacksKeepSnapshotWhileWorkspaceUsesAnotherSlot)
{
    State state;
    CoupledNonlinearWorkspace workspace;
    NonlinearCallbacks retained;
    Teuchos::RCP<Vector> x;
    {
        auto first = workspace_problem(state, &workspace);
        retained = first->callbacks();
        x = first->pack_initial();
    }
    Vector before(x->getMap()), after(x->getMap()), current(x->getMap());
    ASSERT_TRUE(retained.residual(*x, before));
    state.time.time_step = .12;
    state.velocity.owned_data().scale(1.6);
    state.velocity.sync_ghosts();
    {
        auto second = workspace_problem(state, &workspace);
        EXPECT_EQ(second->statistics().workspace_builds, 1U);
        EXPECT_EQ(second->statistics().workspace_reuses, 0U);
        ASSERT_TRUE(second->callbacks().residual(*x, current));
        EXPECT_GT(difference(before, current), 1.e-4);
        ASSERT_TRUE(retained.residual(*x, after));
        EXPECT_LT(difference(before, after), 1.e-14);
    }
    retained = {};
    auto third = workspace_problem(state, &workspace);
    EXPECT_EQ(third->statistics().workspace_reuses, 1U);
    EXPECT_EQ(third->statistics().geometry_builds, 0U);
}

TEST(CoupledNonlinearProblemTest, RecycledWorkspacePreservesExternallyRetainedJacobianAndInverse)
{
    for (const auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        CoupledNonlinearWorkspace workspace;
        NonlinearLinearization retained;
        Teuchos::RCP<Vector> x;
        {
            auto first = workspace_problem(state, &workspace);
            x = first->pack_initial();
            retained = first->callbacks().linearize(*x);
        }
        Vector direction(x->getMap()), jacobian_before(x->getMap()), inverse_before(x->getMap()),
            jacobian_after(x->getMap()), inverse_after(x->getMap()), current(x->getMap());
        set_direction(direction);
        retained.jacobian->apply(direction, jacobian_before);
        retained.right_preconditioner->apply(direction, inverse_before);
        state.time.time_step = .13;
        state.time.kinematic_viscosity = .075;
        state.velocity.owned_data().scale(1.4);
        state.velocity.sync_ghosts();
        auto reused = workspace_problem(state, &workspace, 7.);
        EXPECT_EQ(reused->statistics().workspace_reuses, 1U);
        EXPECT_EQ(reused->statistics().geometry_builds, 0U);
        const auto next = reused->callbacks().linearize(*reused->pack_initial());
        next.jacobian->apply(direction, current);
        EXPECT_GT(difference(jacobian_before, current), 1.e-5);
        retained.jacobian->apply(direction, jacobian_after);
        retained.right_preconditioner->apply(direction, inverse_after);
        EXPECT_LT(difference(jacobian_before, jacobian_after), 1.e-14);
        EXPECT_LT(difference(inverse_before, inverse_after), 1.e-12);
    }
}

TEST(CoupledNonlinearProblemTest, WorkspaceRebuildsChangedBoundaryGeometryAndRecoversAfterInvalidConstruction)
{
    State state;
    CoupledNonlinearWorkspace workspace;
    {
        auto first = workspace_problem(state, &workspace);
        EXPECT_EQ(first->statistics().geometry_builds, 1U);
    }
    for (const bool change_gradient : {false, true})
    {
        if (change_gradient)
            state.time.pressure_gradient_scheme = FVM::CellGradientScheme::GaussLinear;
        else
            state.boundaries.velocity["ymax"].value.x = .45;
        auto reused = workspace_problem(state, &workspace);
        auto fresh = workspace_problem(state, nullptr);
        EXPECT_EQ(reused->statistics().workspace_reuses, 1U);
        EXPECT_EQ(reused->statistics().geometry_builds, 1U);
        expect_same_problem(*reused, *fresh);
    }
    state.time.time_step = 0.;
    EXPECT_THROW(workspace_problem(state, &workspace), std::invalid_argument);
    state.time.time_step = .08;
    auto recovered = workspace_problem(state, &workspace);
    auto fresh = workspace_problem(state, nullptr);
    EXPECT_EQ(recovered->statistics().workspace_builds, 1U);
    EXPECT_EQ(recovered->statistics().schur_builds, 0U);
    expect_same_problem(*recovered, *fresh);
}

TEST(CoupledNonlinearProblemTest, RecycledWorkspaceRefreshesPhysicalModeAndOptionalCoefficientSnapshots)
{
    State state;
    state.time.coupled_operator_backend = CoupledOperatorBackend::BlockComposite;
    state.time.thermal_expansion = .03;
    state.time.reference_temperature = 2.;
    BoussinesqModelOptions options;
    options.reference_density = 7.;
    options.density = 6.8;
    options.dynamic_viscosity = .19;
    NonlinearProblem::material_type material(state.mesh, options, state.time);
    NonlinearProblem::field_type temperature(state.mesh, "pooled_temperature");
    NonlinearProblem::field_type viscosity(state.mesh, "pooled_viscosity");
    NonlinearProblem::velocity_field_type k_gradient(state.mesh, "pooled_k_gradient");
    NonlinearProblem::boundary_cache_type boundary_mu;
    boundary_mu.mesh = state.mesh;
    const auto velocity_cache = FVM::cache_velocity_boundary_conditions<Pack>(state.mesh, state.boundaries);
    for (const auto& [batch, boundary_values] : velocity_cache.value)
        boundary_mu.value[batch].assign(boundary_values.size(), .43);
    CoupledNonlinearWorkspace workspace;
    for (int step = 0; step < 4; ++step)
    {
        SCOPED_TRACE(step);
        state.time.time_step = .04 + .02 * step;
        temperature.put_value(2.5 + .2 * step);
        material.density.put_value(6.8 + .1 * step);
        material.dynamic_viscosity.put_value(.19 + .03 * step);
        viscosity.put_value(.3 + .04 * step);
        k_gradient.put_value({.01, -.02 * step, .03});
        for (auto& [batch, boundary_values] : boundary_mu.value)
            std::fill(boundary_values.begin(), boundary_values.end(), .43 + .05 * step);
        const bool optional_fields = step == 0 || step == 3;
        const NonlinearProblem::FrozenBoussinesqInput physical{temperature, material, step == 2,
            optional_fields ? &viscosity : nullptr, optional_fields ? &k_gradient : nullptr,
            optional_fields ? &boundary_mu : nullptr};
        // Visit full physical input, isothermal input, then physical inputs
        // without and with optional stress/viscosity data on the same storage.
        const auto* selected = step == 1 ? nullptr : &physical;
        auto reused = workspace_problem(state, &workspace, 7., nullptr, selected);
        auto fresh = workspace_problem(state, nullptr, 7., nullptr, selected);
        EXPECT_EQ(reused->statistics().workspace_reuses, step == 0 ? 0U : 1U);
        EXPECT_EQ(reused->statistics().geometry_builds, step == 0 ? 1U : 0U);
        EXPECT_EQ(reused->statistics().schur_builds, 0U);
        expect_same_problem(*reused, *fresh);
    }
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

    solver.step();
    EXPECT_EQ(solver.step_index(), 3);
    EXPECT_EQ(solver.last_nonlinear_result().native_workspace_builds, 0U);
    EXPECT_EQ(solver.last_nonlinear_result().native_workspace_reuses, 1U);
    EXPECT_EQ(solver.last_nonlinear_result().native_geometry_builds, 0U);
    EXPECT_GT(solver.last_nonlinear_result().native_graph_reuses, 0U);
    EXPECT_GT(solver.last_nonlinear_result().native_preconditioner_refreshes, 0U);

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

TEST(CoupledNonlinearSolverTest, BoussinesqAdvancesHeatAndMaterialUpdaterOnceAfterAcceptedFlow)
{
    State state;
    state.time.nonlinear = state.nonlinear;
    state.time.nonlinear.linear_backend = LinearSolverBackend::Gmres;
    state.linear.backend = LinearSolverBackend::BiCGStab;
    state.time.reference_temperature = 300.;
    for (const auto& [name, velocity] : state.boundaries.velocity)
        state.boundaries.temperature[name] = {BoundaryConditionType::Neumann, 0.};
    BoussinesqModelOptions material;
    material.reference_density = material.density = 7.;
    material.dynamic_viscosity = .28;
    BoussinesqSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear, material);
    solver.temperature().put_value(300.);
    State::seed(solver.velocity(), solver.pressure());
    int updates = 0;
    solver.set_material_updater([&](const auto&, auto&) { ++updates; });
    solver.add_temperature_source("heat", 14.);
    for (int step = 1; step <= 2; ++step)
    {
        ASSERT_NO_THROW(solver.step());
        EXPECT_EQ(solver.step_index(), step);
        EXPECT_DOUBLE_EQ(solver.time(), step * state.time.time_step);
        EXPECT_EQ(updates, step);
        EXPECT_TRUE(solver.last_nonlinear_result().converged);
        EXPECT_LT(solver.last_volume_continuity_residuals().maximum, 1.e-9);
        for (size_t cell = 0; cell < state.mesh->num_owned_cells(); ++cell)
            EXPECT_NEAR(solver.temperature().value(static_cast<LO>(cell)), 300. + step * .08, 1.e-6);
    }
    EXPECT_EQ(solver.linear_solver_options().backend, LinearSolverBackend::BiCGStab);
}

TEST(CoupledNonlinearSolverTest, BoussinesqRejectedFlowRestoresGasMaterialAndSupportsRetry)
{
    State state;
    state.time.nonlinear = state.nonlinear;
    state.time.nonlinear.linear_backend = LinearSolverBackend::Gmres;
    state.time.nonlinear.forcing_initial = 1.e-10;
    state.time.nonlinear.forcing_minimum = 1.e-10;
    state.time.reference_temperature = 300.;
    state.linear.max_iterations = 1;
    for (const auto& [name, velocity] : state.boundaries.velocity)
        state.boundaries.temperature[name] = {BoundaryConditionType::Neumann, 0.};
    BoussinesqModelOptions material;
    material.reference_density = material.density = 7.;
    material.dynamic_viscosity = .28;
    BoussinesqSolver<Pack> solver(state.mesh, state.boundaries, state.time, state.linear, material);
    solver.temperature().put_value(300.);
    solver.add_fission_power_source().initialize_constant(0.);
    RadiolyticGasOptions gas_options;
    gas_options.mode = RadiolyticGasMode::Sheng2024TwoPopulation;
    gas_options.hydrogen_yield_mol_per_j = 2.e-7;
    gas_options.max_source_alpha_rate = 1.;
    gas_options.dissolved_transport = RadiolyticTransportMode::Advective;
    gas_options.henry_coefficient = 1.e-5;
    gas_options.surface_tension = .07;
    gas_options.hydrogen_diffusivity = 4.e-4;
    gas_options.initial_dissolved_hydrogen = 1.;
    gas_options.uranium_concentration_mol_per_m3 = 1000.;
    gas_options.hydrogen_yield_molecules_per_100_ev = 1.8;
    auto& gas = solver.configure_radiolytic_gas(gas_options);
    State::seed(solver.velocity(), solver.pressure());
    solver.set_material_updater([](const auto&, auto& properties) { properties.density.put_scalar(6.9); });
    const auto accepted_u = values(solver.velocity().owned_data());
    const auto accepted_p = values(solver.pressure().owned_data());
    const auto accepted_t = values(solver.temperature().owned_data());
    const auto accepted_rho = values(solver.material_properties().density.owned_data());
    const auto accepted_flux = values(solver.pressure_corrected_face_fluxes().owned_data());
    std::vector<std::vector<double>> accepted_gas;
    for (const auto& [name, field] : gas.output_fields())
        accepted_gas.push_back(values(field->owned_data()));
    const bool accepted_initialization = gas.initial_state_initialized();
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        EXPECT_THROW(solver.step(), std::runtime_error);
        EXPECT_EQ(solver.step_index(), 0);
        EXPECT_DOUBLE_EQ(solver.time(), 0.);
        EXPECT_EQ(values(solver.velocity().owned_data()), accepted_u);
        EXPECT_EQ(values(solver.pressure().owned_data()), accepted_p);
        EXPECT_EQ(values(solver.temperature().owned_data()), accepted_t);
        EXPECT_EQ(values(solver.material_properties().density.owned_data()), accepted_rho);
        EXPECT_EQ(values(solver.pressure_corrected_face_fluxes().owned_data()), accepted_flux);
        EXPECT_EQ(gas.initial_state_initialized(), accepted_initialization);
        size_t index = 0;
        for (const auto& [name, field] : gas.output_fields())
            EXPECT_EQ(values(field->owned_data()), accepted_gas.at(index++));
    }
    state.linear.max_iterations = 250;
    solver.set_linear_solver_options(state.linear);
    ASSERT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_DOUBLE_EQ(solver.time(), state.time.time_step);
    EXPECT_TRUE(solver.last_nonlinear_result().converged);
}

TEST(CoupledNonlinearSolverTest, RejectsPcgNonlinearCorrectionOverride)
{
    State state;
    auto problem = state.problem();
    auto x = problem->pack_initial();
    state.nonlinear.linear_backend = LinearSolverBackend::Cg;
    EXPECT_THROW(solve_nox(problem->callbacks(), *x, state.nonlinear, state.linear), std::invalid_argument);
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

#ifdef SIMPLEFLUID_ENABLE_NOX
TEST(CoupledNonlinearSolverTest, PerStepPreconditionerReducesSetupWithTheSamePhysicalSolution)
{
    for (const auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        auto reference_problem = state.problem();
        auto reference = reference_problem->pack_initial();
        const auto reference_result =
            solve_nox(reference_problem->callbacks(), *reference, state.nonlinear, state.linear);
        ASSERT_TRUE(reference_result.converged) << reference_result.reason;
        const auto reference_statistics = reference_problem->statistics();
        EXPECT_EQ(reference_statistics.schur_builds, reference_statistics.linearizations);

        state.nonlinear.preconditioner_update = CoupledPreconditionerUpdate::PerTimeStep;
        auto lagged_problem = state.problem();
        auto lagged = lagged_problem->pack_initial();
        const auto callbacks = lagged_problem->callbacks();
        const auto result = solve_nox(callbacks, *lagged, state.nonlinear, state.linear);
        ASSERT_TRUE(result.converged) << result.reason;
        EXPECT_LT(difference(*reference, *lagged), 2.e-8);
        ASSERT_TRUE(callbacks.convergence_gate(*lagged));
        EXPECT_LT(lagged_problem->continuity().maximum, state.nonlinear.continuity_tolerance);
        const auto statistics = lagged_problem->statistics();
        EXPECT_LT(statistics.schur_builds, statistics.linearizations);
        EXPECT_EQ(statistics.preconditioner_builds + statistics.preconditioner_refreshes, statistics.schur_builds);
    }
}

TEST(CoupledNonlinearSolverTest, PerStepPreconditionerRefreshesStaleOrStagnatingResidualsWithoutMutatingSnapshots)
{
    for (const auto backend : {CoupledOperatorBackend::Assembled, CoupledOperatorBackend::BlockComposite})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        State state;
        state.time.coupled_operator_backend = backend;
        state.nonlinear.preconditioner_update = CoupledPreconditionerUpdate::PerTimeStep;
        auto problem = state.problem();
        const auto callbacks = problem->callbacks();
        auto x = problem->pack_initial();
        Vector residual(callbacks.map), y(*x, Teuchos::Copy), direction(callbacks.map);
        set_direction(direction);
        // Only one rank changes its trial. Cache compatibility must be collective.
        if (callbacks.map->getComm()->getRank() == 0)
            y.update(.2, direction, 1.);
        ASSERT_TRUE(callbacks.residual(*x, residual));
        const auto first = callbacks.linearize(*x);
        Vector j_before(callbacks.map), p_before(callbacks.map), action(callbacks.map);
        first.jacobian->apply(direction, j_before);
        first.right_preconditioner->apply(direction, p_before);

        ASSERT_TRUE(callbacks.residual(y, residual));
        // The latest residual describes y, not x: force a refresh and invalidate
        // the previous-base norm instead of making a stale lagging decision.
        const auto second = callbacks.linearize(*x);
        EXPECT_EQ(problem->statistics().schur_builds, 2U);
        EXPECT_NE(second.right_preconditioner.getRawPtr(), first.right_preconditioner.getRawPtr());
        ASSERT_TRUE(callbacks.residual(y, residual));
        const auto third = callbacks.linearize(y);
        EXPECT_EQ(problem->statistics().schur_builds, 3U);
        ASSERT_TRUE(callbacks.residual(y, residual));
        const auto fourth = callbacks.linearize(y); // No reduction at all: refresh.
        EXPECT_EQ(problem->statistics().schur_builds, 4U);
        EXPECT_NE(fourth.right_preconditioner.getRawPtr(), third.right_preconditioner.getRawPtr());
        first.jacobian->apply(direction, action);
        EXPECT_LT(difference(j_before, action), 1.e-14);
        first.right_preconditioner->apply(direction, action);
        EXPECT_LT(difference(p_before, action), 1.e-12);
        third.jacobian->apply(direction, action);
        EXPECT_GT(difference(j_before, action), 1.e-7);
    }
}

TEST(CoupledNonlinearSolverTest, RejectsInvalidOrRankDivergentPreconditionerUpdateControls)
{
    State state;
    auto problem = state.problem();
    const auto callbacks = problem->callbacks();
    auto x = problem->pack_initial();
    for (const double ratio : {0., 1.1, std::numeric_limits<double>::quiet_NaN()})
    {
        state.nonlinear.preconditioner_stagnation_ratio = ratio;
        EXPECT_THROW(state.problem(), std::invalid_argument);
        EXPECT_THROW(solve_nox(callbacks, *x, state.nonlinear, state.linear), std::invalid_argument);
    }
    state.nonlinear.preconditioner_stagnation_ratio = .9;
    state.nonlinear.preconditioner_update = static_cast<CoupledPreconditionerUpdate>(-1);
    EXPECT_THROW(state.problem(), std::invalid_argument);
    EXPECT_THROW(solve_nox(callbacks, *x, state.nonlinear, state.linear), std::invalid_argument);
    if (callbacks.map->getComm()->getSize() > 1)
    {
        state.nonlinear.preconditioner_update = callbacks.map->getComm()->getRank() == 0
                                                  ? CoupledPreconditionerUpdate::PerTimeStep
                                                  : CoupledPreconditionerUpdate::EveryLinearization;
        EXPECT_THROW(state.problem(), std::invalid_argument);
        EXPECT_THROW(solve_nox(callbacks, *x, state.nonlinear, state.linear), std::invalid_argument);
    }
}
#endif
