#include <gtest/gtest.h>

#include "equations/PressureProjectionEquation.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string_view>

namespace SimpleFluid::detail
{
template<TpetraTypePack Pack>
struct BelosLinearSolverTestAccess
{
    static auto preconditioning_matrix(const BelosLinearSolver<Pack>& solver)
    {
        return solver.d_preconditioner_operator;
    }
    static auto backend(const BelosLinearSolver<Pack>& solver) { return solver.d_backend; }
    static auto preconditioner_kind(const BelosLinearSolver<Pack>& solver) { return solver.d_preconditioner_kind; }
    static auto problem(const BelosLinearSolver<Pack>& solver) { return solver.d_problem.getConst(); }
    static auto setups(const BelosLinearSolver<Pack>& solver) { return solver.d_preconditioner_setup_count; }
};

template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess
{
    using Equation = PressureProjectionEquation<Pack>;
    using BelosAccess = BelosLinearSolverTestAccess<Pack>;
    static auto assembled(const Equation& equation) { return equation.d_assembled_matrix.getConst(); }
    static void double_assembled_values(Equation& equation) { equation.d_assembled_matrix->scale(2.0); }
    static auto compact(const Equation& equation) { return equation.d_cached_pressure_matrix.getConst(); }
    static auto graphs(const Equation& equation) { return equation.d_assembled_graph_builds; }
    static auto values(const Equation& equation) { return equation.d_assembled_value_builds; }
    static auto backend(const Equation& equation) { return BelosAccess::backend(equation.d_linear_solver); }
    static auto preconditioner_kind(const Equation& equation)
    {
        return BelosAccess::preconditioner_kind(equation.d_linear_solver);
    }
    static auto problem(const Equation& equation) { return BelosAccess::problem(equation.d_linear_solver); }
    static auto setups(const Equation& equation) { return BelosAccess::setups(equation.d_linear_solver); }
    static auto preconditioning_matrix(const Equation& equation)
    {
        return BelosAccess::preconditioning_matrix(equation.d_linear_solver);
    }
};
}

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
using Equation = PressureProjectionEquation<Pack>;
using Access = detail::PressureProjectionEquationTestAccess<Pack>;
using Method = PressureProjectionMethod;
using Local = Pack::local_ordinal_type;
using Global = Pack::global_ordinal_type;
constexpr double dt = 0.125, rho = 7.0;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

auto make_mesh(size_t z_partitions = 1)
{
    using Geometry = Mesh::SemiStructured;
    constexpr unsigned width = 5, height = 4;
    Arr<Geometry::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    Arr<Geometry::BoundaryEdge> boundaries;
    for (unsigned row = 0; row <= height; ++row)
        for (unsigned col = 0; col <= width; ++col)
        {
            const double xi = double(col) / width, eta = double(row) / height;
            nodes.push_back({xi + (0.33 + 0.13 * xi * xi) * eta, eta + 0.10 * xi * eta * eta, 0.0});
        }
    for (unsigned row = 0; row < height; ++row)
        for (unsigned col = 0; col < width; ++col)
        {
            const unsigned lower = row * (width + 1) + col, upper = lower + width + 1;
            cells.push_back({lower, lower + 1, upper + 1, upper});
            if (row == 0) boundaries.push_back({lower, lower + 1, "wall"});
            if (col + 1 == width) boundaries.push_back({lower + 1, upper + 1, "outlet"});
            if (row + 1 == height) boundaries.push_back({upper + 1, upper, "wall"});
            if (col == 0) boundaries.push_back({upper, lower, "wall"});
        }
    auto geometry = std::make_shared<Geometry>(nodes, cells, ArrReal{0.0, 0.24, 0.6}, boundaries);
    Mesh::DistributionOptions distribution;
    distribution.z_partitions = z_partitions;
    return std::make_shared<Mesh>(geometry, distribution);
}

BoundaryConditionSet boundary_conditions()
{
    BoundaryConditionSet conditions;
    conditions.velocity["outlet"] = {BoundaryConditionType::Neumann, {}};
    conditions.pressure["outlet"] = {BoundaryConditionType::Dirichlet, 0.0};
    return conditions;
}

LinearSolverOptions linear_options()
{
    LinearSolverOptions options;
    options.backend = LinearSolverBackend::Gmres;
    options.preconditioner = LinearPreconditioner::ILUT;
    options.reuse_preconditioner = true;
    options.tolerance = 1e-11;
    return options;
}

double prescribed_flux(const Mesh& mesh, Local face, double amplitude)
{
    return mesh.face_centroid(face).z < 0.24 ? amplitude : -amplitude;
}

void overwrite_outlet(const Mesh& mesh, FaceField<Pack>& flux, double amplitude)
{
    for (const auto& [batch, faces] : mesh.boundary_batches())
        if (mesh.boundary_batch_name(batch) == "outlet")
            for (const auto face : faces.face_lids)
                if (mesh.is_owned_face(face)) flux.set_value(face, prescribed_flux(mesh, face, amplitude));
    flux.sync_ghosts();
}

void fix_outlet(Equation& equation, const SP<Mesh>& mesh, std::uint64_t generation, double amplitude)
{
    equation.set_fixed_boundary_flux_provider({"outlet"},
        [mesh, amplitude](int, size_t, Local face) { return prescribed_flux(*mesh, face, amplitude); }, generation);
}

double global_sum(const Mesh& mesh, double local)
{
    double total{};
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &total);
    return total;
}

void compare_projection(const SP<Mesh>& mesh, Equation& assembled, double forcing,
    bool accumulated = false, bool fixed_outlet = false, double outlet_amplitude = 0.0)
{
    const auto conditions = boundary_conditions();
    auto gradient_boundaries = conditions.pressure;
    if (fixed_outlet && !accumulated) gradient_boundaries["outlet"] = {BoundaryConditionType::Neumann, 0.0};
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    Equation reference(mesh, linear_options(), conditions.pressure);
    reference.set_projection_method(Method::FullOperator);
    if (fixed_outlet) fix_outlet(reference, mesh, 1, outlet_amplitude);

    Equation::field_type pressure(mesh, "p"), increment(mesh, "dp"), ref_pressure(mesh, "ref_p"),
        ref_increment(mesh, "ref_dp");
    Equation::velocity_field_type velocity(mesh, "u"), ref_velocity(mesh, "ref_u");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto center = mesh->cell_centroid(cell);
        velocity.set_value(cell, {center.x * center.y + forcing * center.z, center.x * center.x - forcing * center.y,
            center.z * center.y});
        if (accumulated) pressure.set_value(cell, 0.7 * center.x * center.x - 0.2 * center.y * center.z);
    }
    ref_pressure.owned_data().update(1.0, pressure.owned_data(), 0.0);
    ref_velocity.owned_data().update(1.0, velocity.owned_data(), 0.0);
    pressure.sync_ghosts(); ref_pressure.sync_ghosts();
    velocity.sync_ghosts(); ref_velocity.sync_ghosts();

    FVM::PressureWeightedFaceFluxWorkspace<Pack> workspace(mesh);
    FaceField<Pack> flux(mesh, "physical_flux");
    const auto reconstruct = [&]
    {
        FVM::pressure_weighted_face_fluxes(velocity, pressure, dt / rho, boundaries, gradient_boundaries,
            workspace, flux, FVM::CellGradientScheme::LeastSquares);
        if (fixed_outlet) overwrite_outlet(*mesh, flux, outlet_amplitude);
    };
    const auto norm_squared = [&](bool exclude_gauge)
    {
        const auto map = mesh->owned_cell_map();
        double local = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Local>(owned);
            if (exclude_gauge && fixed_outlet && map->getGlobalElement(cell) == map->getMinAllGlobalIndex()) continue;
            const auto balance = FVM::cell_flux_balance<Pack>(*mesh, flux, cell);
            local += balance * balance;
        }
        return global_sum(*mesh, local);
    };
    reconstruct();
    const double initial = std::sqrt(norm_squared(true));
    ASSERT_GT(initial, 1e-4);
    const auto project = [&](Equation& equation, auto& total, auto& correction, auto& predictor)
    {
        return accumulated ? equation.project(total, correction, dt, rho, boundaries, predictor)
                           : equation.project(total, dt, rho, boundaries, predictor);
    };
    const auto result = project(assembled, pressure, increment, velocity);
    const auto expected = project(reference, ref_pressure, ref_increment, ref_velocity);
    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_TRUE(expected.linear_solve.converged);
    EXPECT_GT(result.linear_solve.iterations, 0);
    // The retained Belos RHS is the integral predictor defect, including a zero on a pinned row.
    const auto problem = Access::problem(assembled);
    ASSERT_EQ(global_sum(*mesh, problem.is_null()), 0.0);
    const auto rhs = problem->getRHS();
    ASSERT_EQ(global_sum(*mesh, rhs.is_null()), 0.0);
    const auto rhs_values = rhs->getData(0);
    const auto map = mesh->owned_cell_map();
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto row = map->getGlobalElement(cell);
        const double defect = fixed_outlet && row == map->getMinAllGlobalIndex() ? 0.0
            : -FVM::cell_flux_balance<Pack>(*mesh, flux, cell);
        EXPECT_NEAR(rhs_values[owned], defect, 2e-13 * (1.0 + initial)) << "RHS row " << row;
    }
    reconstruct();
    const double relative_defect = std::sqrt(norm_squared(true)) / initial;
    EXPECT_LE(relative_defect, linear_options().tolerance + 1e-13);
    EXPECT_NEAR(result.linear_solve.achieved_tolerance, relative_defect, 1e-13);
    EXPECT_NEAR(result.linear_solve.rhs_norm, initial, 2e-13 * (1.0 + initial));
    const double continuity = std::sqrt(norm_squared(false));
    EXPECT_NEAR(result.continuity, continuity, 2e-13);
    // For these balanced boundaries the omitted gauge defect is minus the sum over the free rows.
    EXPECT_LE(continuity, std::sqrt(static_cast<double>(map->getGlobalNumElements())) *
        initial * linear_options().tolerance + 2e-13);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        EXPECT_NEAR(pressure.value(cell), ref_pressure.value(cell), 2e-8);
        if (accumulated) EXPECT_NEAR(increment.value(cell), ref_increment.value(cell), 2e-8);
        EXPECT_NEAR(velocity.value(cell).x, ref_velocity.value(cell).x, 2e-10);
        EXPECT_NEAR(velocity.value(cell).y, ref_velocity.value(cell).y, 2e-10);
        EXPECT_NEAR(velocity.value(cell).z, ref_velocity.value(cell).z, 2e-10);
    }
    for (const auto face : flux.owned_face_ids())
    {
        EXPECT_NEAR(assembled.corrected_face_fluxes().value(face), flux.value(face), 2e-13);
        EXPECT_NEAR(assembled.corrected_face_fluxes().value(face),
            reference.corrected_face_fluxes().value(face), 2e-10);
    }
    EXPECT_EQ(Access::preconditioning_matrix(reference).get(), Access::compact(reference).get());
    EXPECT_NE(Access::assembled(assembled).get(), Access::compact(assembled).get());
}

void check_basis_actions(const SP<Mesh>& mesh, const Equation& equation, bool fixed_outlet = false,
    bool accumulated = false)
{
    auto conditions = boundary_conditions();
    if (fixed_outlet && !accumulated) conditions.pressure["outlet"] = {BoundaryConditionType::Neumann, 0.0};
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    FVM::PressureWeightedFaceFluxWorkspace<Pack> workspace(mesh);
    FVM::CellGradientCache<Pack> gradients(mesh);
    Equation::field_type probe(mesh, "basis_pressure");
    Equation::velocity_field_type negative_gradient(mesh, "basis_velocity");
    FaceField<Pack> flux(mesh, "basis_flux");
    const auto map = mesh->owned_cell_map();
    Pack::vector_type action(map), scaled_action(map);
    const auto matrix = Access::assembled(equation);
    ASSERT_EQ(global_sum(*mesh, matrix.is_null()), 0.0);
    const auto problem = Access::problem(equation);
    ASSERT_EQ(global_sum(*mesh, problem.is_null()), 0.0);
    const auto solve_operator = problem->getOperator();
    ASSERT_EQ(global_sum(*mesh, solve_operator.is_null()), 0.0);
    double extended_remote_max = 0.0;
    for (Global column = map->getMinAllGlobalIndex(); column <= map->getMaxAllGlobalIndex(); ++column)
    {
        SCOPED_TRACE(column);
        probe.owned_data().putScalar(0.0);
        if (map->isNodeGlobalElement(column)) probe.set_value(map->getLocalElement(column), 1.0);
        probe.sync_ghosts();
        FVM::cell_gradient(probe, conditions.pressure, negative_gradient, gradients,
            FVM::CellGradientScheme::LeastSquares);
        negative_gradient.owned_data().scale(-1.0);
        negative_gradient.sync_ghosts();
        FVM::pressure_weighted_face_fluxes(negative_gradient, probe, 1.0, cache, conditions.pressure, workspace, flux);
        if (fixed_outlet) overwrite_outlet(*mesh, flux, 0.0);
        matrix->apply(probe.vector(), action);
        solve_operator->apply(probe.vector(), scaled_action);
        const auto actual = action.getData(0);
        const auto scaled = scaled_action.getData(0);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Local>(owned);
            const Global row = map->getGlobalElement(cell);
            const double expected = fixed_outlet && row == map->getMinAllGlobalIndex() ? double(column == row)
                : FVM::cell_flux_balance<Pack>(*mesh, flux, cell);
            EXPECT_NEAR(actual[owned], expected, 2e-12 * (1.0 + std::abs(expected))) << "row " << row;
            EXPECT_NEAR(scaled[owned], dt * expected, dt * 2e-12 * (1.0 + std::abs(expected))) << "row " << row;
            if (!mesh->overlap_cell_map()->isNodeGlobalElement(column))
                extended_remote_max = std::max(extended_remote_max, std::abs(expected));
        }
    }
    // Only an XY partition cuts through the skew stencil beyond the one-cell halo.
    if (mesh->semi_structured_decomposition().xy_partitions > 1)
        EXPECT_GT(global_sum(*mesh, extended_remote_max), 1e-8);
}

void check_muelu_compact_rows(const Mesh& mesh, const Equation& equation, bool pinned)
{
    EXPECT_EQ(Access::backend(equation), LinearSolverBackend::BiCGStab);
    EXPECT_EQ(Access::preconditioner_kind(equation), LinearPreconditioner::MueLuSymmetric);
    const auto selected = Teuchos::rcp_dynamic_cast<const Pack::matrix_type>(Access::preconditioning_matrix(equation));
    ASSERT_EQ(global_sum(mesh, selected.is_null()), 0.0);
    EXPECT_NE(selected.get(), Access::assembled(equation).get());
    const auto map = mesh.owned_cell_map();
    ASSERT_TRUE(selected->getRowMap()->isSameAs(*map));
    const auto gauge = map->getMinAllGlobalIndex();
    int remote_gauge_neighbors = 0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto row = map->getGlobalElement(cell);
        SCOPED_TRACE(row);
        std::map<Global, double> expected;
        if (pinned && row == gauge)
            expected[row] = 1.0;
        else
        {
            const auto center = mesh.cell_centroid(cell);
            for (const auto face : mesh.faces(cell))
            {
                const auto area = mesh.face_area_vector_outward(face, cell);
                if (mesh.is_interior_face(face))
                {
                    const auto neighbor = mesh.opposite_cell(face, cell);
                    const auto column = mesh.cell_global_id(neighbor);
                    const auto delta = mesh.cell_centroid(neighbor) - center;
                    const double weight = area.dot(area) / area.dot(delta);
                    expected[row] += weight;
                    if (!pinned || column != gauge)
                        expected[column] -= weight;
                    else if (!map->isNodeGlobalElement(gauge))
                        ++remote_gauge_neighbors;
                }
                else if (!pinned && mesh.is_boundary_face(face) &&
                         mesh.boundary_batch_name(mesh.boundary_id(face)) == "outlet")
                    expected[row] += area.dot(area) / area.dot(mesh.face_centroid(face) - center);
            }
        }
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        selected->getLocalRowView(cell, columns, values);
        for (size_t slot = 0; slot < columns.extent(0); ++slot)
            expected[selected->getColMap()->getGlobalElement(columns(slot))] -= values(slot);
        for (const auto& [column, difference] : expected)
            EXPECT_NEAR(difference, 0.0, 2e-12) << "column " << column;
    }
    if (pinned && map->getComm()->getSize() == 2)
        EXPECT_GT(global_sum(mesh, remote_gauge_neighbors), 0.0);
}

TEST(AssembledPressureProjectionTest, EmptyRankParticipatesInSingleCellProduct)
{
    if (Tpetra::getDefaultComm()->getSize() != 2) GTEST_SKIP() << "Requires exactly two MPI ranks.";
    const auto mesh = test::build_mesh<Pack>(test::make_single_hex_database());
    const auto map = mesh->owned_cell_map();
    ASSERT_EQ(map->getGlobalNumElements(), 1U);
    ASSERT_EQ(global_sum(*mesh, map->getLocalNumElements() == 0), 1.0);

    BoundaryConditionSet conditions;
    conditions.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
    conditions.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 0.0};
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    LinearSolverOptions options;
    options.preconditioner = LinearPreconditioner::None;
    Equation equation(mesh, options, conditions.pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    Teuchos::RCP<const Pack::graph_type> original_graph;
    Pack::vector_type probe(map), action(map);
    probe.putScalar(3.0);
    constexpr double outlet_coefficient = 1.0 / 0.5;
    for (size_t pass = 1; pass <= 2; ++pass)
    {
        SCOPED_TRACE(pass);
        if (pass == 2) equation.refresh_geometry();
        Equation::field_type pressure(mesh, "pressure");
        Equation::velocity_field_type velocity(mesh, "velocity");
        const auto result = equation.project(pressure, dt, rho, boundaries, velocity, [](Local) { return 1.0; });
        EXPECT_TRUE(result.linear_solve.converged);
        EXPECT_NEAR(result.linear_solve.rhs_norm, dt, 1e-13);

        const auto matrix = Access::assembled(equation);
        ASSERT_EQ(global_sum(*mesh, matrix.is_null()), 0.0);
        EXPECT_EQ(matrix->getLocalNumRows(), map->getLocalNumElements());
        EXPECT_EQ(Access::graphs(equation), 1U);
        EXPECT_EQ(Access::values(equation), pass);
        if (pass == 1) original_graph = matrix->getCrsGraph();
        else EXPECT_EQ(matrix->getCrsGraph(), original_graph);

        action.putScalar(0.0);
        matrix->apply(probe, action);
        if (map->getLocalNumElements() != 0)
        {
            EXPECT_NEAR(action.getData(0)[0], 3.0 * outlet_coefficient, 1e-13);
            EXPECT_NEAR(pressure.value(0), rho / outlet_coefficient, 1e-12);
        }
    }
}

TEST(AssembledPressureProjectionTest, FullOperatorsSkipRoundoffRhsAndSolveAboveFloor)
{
    const auto mesh = test::build_mesh<Pack>(test::make_two_hex_database());
    if (mesh->owned_cell_map()->getComm()->getSize() == 2)
        ASSERT_EQ(global_sum(*mesh, mesh->num_owned_cells() == 1), 2.0);

    // Two unit cubes. The prescribed outward flux exactly meets the left target.
    // The right cell starts at rest with a small, independently representable target.
    constexpr double left_flux = 0x1p-9;
    BoundaryConditionSet conditions;
    conditions.velocity["xmin"] = {BoundaryConditionType::Dirichlet, {-left_flux, 0.0, 0.0}};
    conditions.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 0.0};
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);

    LinearSolverOptions options;
    options.backend = LinearSolverBackend::BiCGStab;
    options.preconditioner = LinearPreconditioner::None;
    options.tolerance = 1e-10;
    // Predictor balances (left_flux, 0) give b = (0, right_rate). A fresh
    // projection scales the corrected limit by ||b||. Both ranks share the floor.
    constexpr double left_cell_floor = 64.0 * std::numeric_limits<double>::epsilon() * (2.0 * left_flux);
    for (const double right_rate : {0.0, left_cell_floor / 2, left_cell_floor * 2})
    {
        SCOPED_TRACE(right_rate);
        const bool skip = right_rate <= left_cell_floor;
        const auto target = VolumeContinuityTarget<Pack>::from_integrated_rate_provider(mesh,
            [&](Local cell) { return mesh->cell_centroid(cell).x < 1.0 ? left_flux : right_rate; }, 1);
        for (const auto method : {Method::FullOperator, Method::AssembledFullOperator})
        {
            SCOPED_TRACE(static_cast<int>(method));
            Equation equation(mesh, options, conditions.pressure);
            equation.set_projection_method(method);
            Equation::field_type pressure(mesh, "pressure");
            Equation::velocity_field_type velocity(mesh, "velocity");
            const auto result = equation.project(pressure, dt, rho, boundaries, velocity, target);
            EXPECT_TRUE(result.linear_solve.converged);
            EXPECT_DOUBLE_EQ(result.linear_solve.rhs_norm, right_rate);
            EXPECT_EQ(Access::values(equation), method == Method::AssembledFullOperator && !skip ? 1U : 0U);
            if (skip)
            {
                EXPECT_EQ(result.linear_solve.iterations, 0);
                EXPECT_DOUBLE_EQ(result.linear_solve.achieved_tolerance, right_rate == 0.0 ? 0.0 : 1.0);
            }
            else
            {
                EXPECT_GT(result.linear_solve.iterations, 0);
                EXPECT_LE(result.linear_solve.achieved_tolerance, options.tolerance);
            }

            // Zero interior correction flux requires equal pressures. The outlet has
            // unit area and half-cell distance, hence right_rate = 2 * dt * p / rho.
            const double expected_pressure = skip ? 0.0 : rho * right_rate / (2 * dt);
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<Local>(owned);
                EXPECT_NEAR(pressure.value(cell), expected_pressure, expected_pressure * options.tolerance);
                const double expected_balance = skip && mesh->cell_centroid(cell).x >= 1.0
                    ? 0.0 : target.integrated_rate(cell);
                const auto balance = FVM::cell_flux_balance<Pack>(*mesh, equation.corrected_face_fluxes(), cell);
                EXPECT_NEAR(balance, expected_balance, options.tolerance * right_rate);
            }
            EXPECT_LE(result.continuity, skip ? right_rate : options.tolerance * right_rate);
        }
    }
}

TEST(AssembledPressureProjectionTest, SkewStencilMatchesEveryBasisAndReusesRhsOnlySetup)
{
    const auto mesh = make_mesh();
    Equation equation(mesh, linear_options(), boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    compare_projection(mesh, equation, 0.3);
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), Access::assembled(equation).get());
    check_basis_actions(mesh, equation);
    const auto first = Access::assembled(equation);
    ASSERT_EQ(global_sum(*mesh, first.is_null()), 0.0);
    const auto graph = first->getCrsGraph();
    EXPECT_EQ(Access::graphs(equation), 1U);
    EXPECT_EQ(Access::values(equation), 1U);
    compare_projection(mesh, equation, -1.1);
    const auto second = Access::assembled(equation);
    ASSERT_EQ(global_sum(*mesh, second.is_null()), 0.0);
    EXPECT_EQ(second, first);
    EXPECT_EQ(second->getCrsGraph(), graph);
    EXPECT_EQ(Access::graphs(equation), 1U);
    EXPECT_EQ(Access::values(equation), 1U);
    EXPECT_EQ(Access::setups(equation), 1U);
}

TEST(AssembledPressureProjectionTest, BiCGStabWithCorrectedILUTMatchesMatrixFreeReference)
{
    const auto mesh = make_mesh();
    const auto map = mesh->owned_cell_map();
    ASSERT_EQ(global_sum(*mesh, map->getLocalNumElements() > 0), map->getComm()->getSize());

    auto options = linear_options();
    options.backend = LinearSolverBackend::BiCGStab;
    Equation equation(mesh, options, boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    compare_projection(mesh, equation, 0.3, /*accumulated=*/true);

    EXPECT_EQ(Access::backend(equation), LinearSolverBackend::BiCGStab);
    const auto corrected = Access::assembled(equation);
    EXPECT_FALSE(corrected.is_null());
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), corrected.get());
    EXPECT_EQ(Access::setups(equation), 1U);
}

TEST(AssembledPressureProjectionTest, BiCGStabMueLuSymmetricKeepsDirichletCompactRowsUnpinned)
{
    const auto mesh = make_mesh();
    auto options = linear_options();
    options.backend = LinearSolverBackend::BiCGStab;
    options.preconditioner = LinearPreconditioner::MueLuSymmetric;
    Equation equation(mesh, options, boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    compare_projection(mesh, equation, 0.3, /*accumulated=*/true);
    check_muelu_compact_rows(*mesh, equation, /*pinned=*/false);
    check_basis_actions(mesh, equation);
}

TEST(AssembledPressureProjectionTest, BiCGStabMueLuSymmetricEliminatesGaugeAcrossSetupReuseAndRefresh)
{
    // With two axial slabs every cell in the gauge layer has a neighbor on the other rank.
    const auto mesh = make_mesh(Tpetra::getDefaultComm()->getSize() == 2 ? 2 : 1);
    auto options = linear_options();
    options.backend = LinearSolverBackend::BiCGStab;
    options.preconditioner = LinearPreconditioner::MueLuSymmetric;
    Equation equation(mesh, options, boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    fix_outlet(equation, mesh, 1, 0.0);
    const auto project = [&](double forcing)
    {
        compare_projection(mesh, equation, forcing, /*accumulated=*/false, /*fixed_outlet=*/true);
    };
    project(0.3);
    check_muelu_compact_rows(*mesh, equation, /*pinned=*/true);
    check_basis_actions(mesh, equation, /*fixed_outlet=*/true);
    EXPECT_EQ(Access::setups(equation), 1U);
    project(-1.1);
    EXPECT_EQ(Access::setups(equation), 1U);

    PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(0.81, dt);
    equation.refresh_geometry();
    project(0.7);
    check_muelu_compact_rows(*mesh, equation, /*pinned=*/true);
    check_basis_actions(mesh, equation, /*fixed_outlet=*/true);
    EXPECT_EQ(Access::setups(equation), 2U);

    options.preconditioner = LinearPreconditioner::ILUT;
    equation.set_linear_solver_options(options);
    project(0.3);
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), Access::assembled(equation).get());
    options.preconditioner = LinearPreconditioner::MueLuSymmetric;
    equation.set_linear_solver_options(options);
    project(-1.1);
    check_muelu_compact_rows(*mesh, equation, /*pinned=*/true);
    EXPECT_EQ(Access::setups(equation), 4U);
}

TEST(AssembledPressureProjectionTest, MovingAndRollbackRefreshValuesOnTheSameGraph)
{
    const auto mesh = make_mesh();
    PlanarALEMeshMotion<Pack> motion(mesh);
    Equation equation(mesh, linear_options(), boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    compare_projection(mesh, equation, 0.7);
    const auto first = Access::assembled(equation);
    ASSERT_EQ(global_sum(*mesh, first.is_null()), 0.0);
    const auto graph = first->getCrsGraph();
    const auto map = mesh->owned_cell_map();
    Pack::vector_type probe(map), original(map), current(map);
    for (size_t owned = 0; owned < map->getLocalNumElements(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        probe.replaceLocalValue(cell, std::sin(0.73 * map->getGlobalElement(cell)));
    }
    first->apply(probe, original);
    const auto refresh = [&](size_t builds)
    {
        equation.refresh_geometry();
        compare_projection(mesh, equation, 0.7);
        check_basis_actions(mesh, equation);
        const auto matrix = Access::assembled(equation);
        ASSERT_EQ(global_sum(*mesh, matrix.is_null()), 0.0);
        EXPECT_EQ(matrix->getCrsGraph(), graph);
        EXPECT_EQ(Access::graphs(equation), 1U);
        EXPECT_EQ(Access::values(equation), builds);
        EXPECT_EQ(Access::setups(equation), builds);
        matrix->apply(probe, current);
        current.update(-1.0, original, 1.0);
    };
    motion.begin_trial(0.81, dt);
    refresh(2);
    EXPECT_GT(current.norm2(), 1e-3);
    motion.rollback_trial();
    refresh(3);
    EXPECT_LT(current.norm2(), 2e-12 * original.norm2());
}

TEST(AssembledPressureProjectionTest, FixedFluxSwitchesBetweenPhysicalAndCorrectionGradientMaps)
{
    const auto mesh = make_mesh();
    Equation equation(mesh, linear_options(), boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    fix_outlet(equation, mesh, 1, 0.0);
    std::size_t expected_setups = 0;
    for (const bool accumulated : {false, true, false})
    {
        SCOPED_TRACE(accumulated);
        compare_projection(mesh, equation, 0.7, accumulated, true);
        ++expected_setups;
        EXPECT_EQ(Access::setups(equation), expected_setups);
        EXPECT_EQ(Access::preconditioning_matrix(equation).get(), Access::assembled(equation).get());
        check_basis_actions(mesh, equation, true, accumulated);
    }
    const auto builds = Access::values(equation), graphs = Access::graphs(equation);
    fix_outlet(equation, mesh, 2, 0.015);
    compare_projection(mesh, equation, 0.7, false, true, 0.015);
    EXPECT_EQ(Access::values(equation), builds);
    EXPECT_EQ(Access::graphs(equation), graphs);
    EXPECT_EQ(Access::setups(equation), expected_setups);
}

TEST(AssembledPressureProjectionTest, SwitchingPreconditionersKeepsTheCorrectedOperator)
{
    const auto mesh = make_mesh();
    auto options = linear_options();
    Equation equation(mesh, options, boundary_conditions().pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    compare_projection(mesh, equation, 0.3);
    const auto corrected = Access::assembled(equation);
    ASSERT_EQ(global_sum(*mesh, corrected.is_null()), 0.0);
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), corrected.get());

    options.preconditioner = LinearPreconditioner::SymmetricGaussSeidel;
    equation.set_linear_solver_options(options);
    compare_projection(mesh, equation, 0.7);
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), Access::compact(equation).get());

    options.preconditioner = LinearPreconditioner::ILUT;
    equation.set_linear_solver_options(options);
    compare_projection(mesh, equation, -1.1);
    EXPECT_EQ(Access::preconditioning_matrix(equation).get(), corrected.get());
    EXPECT_EQ(Access::values(equation), 1U);
    EXPECT_EQ(Access::setups(equation), 3U);
}

TEST(AssembledPressureProjectionTest, IndependentlyRejectsAConvergedButMisScaledMatrix)
{
    const auto mesh = make_mesh();
    const auto conditions = boundary_conditions();
    Equation equation(mesh, linear_options(), conditions.pressure);
    equation.set_projection_method(Method::AssembledFullOperator);
    const auto boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, conditions);
    Equation::field_type pressure(mesh, "pressure");
    Equation::velocity_field_type velocity(mesh, "velocity");
    const auto source = [](Local) { return 1.0; };
    const auto baseline = equation.project(pressure, dt, rho, boundaries, velocity, source);
    ASSERT_TRUE(baseline.linear_solve.converged);
    ASSERT_GT(baseline.linear_solve.rhs_norm, 0.0);
    ASSERT_EQ(global_sum(*mesh, Access::assembled(equation).is_null()), 0.0);
    Access::double_assembled_values(equation);
    velocity.owned_data().putScalar(0.0);
    velocity.sync_ghosts();
    try
    {
        equation.project(pressure, dt, rho, boundaries, velocity, source);
        ADD_FAILURE() << "Krylov can solve 2*A but its solution has half the required physical correction";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string_view(error.what()).find("corrected pressure residual"), std::string_view::npos)
            << error.what();
    }
    EXPECT_EQ(Access::values(equation), 1U);
}

TEST(AssembledPressureProjectionTest, RejectsUnsupportedGradientAndCg)
{
    const auto mesh = make_mesh();
    auto options = linear_options();
    options.preconditioner = LinearPreconditioner::None;
    Equation gauss(mesh, options, {}, FVM::CellGradientScheme::GaussLinear);
    EXPECT_THROW(gauss.set_projection_method(Method::AssembledFullOperator), std::invalid_argument);
    EXPECT_EQ(gauss.projection_method(), Method::Anderson);
    Equation least_squares(mesh, options);
    least_squares.set_projection_method(Method::AssembledFullOperator);
    options.backend = LinearSolverBackend::Cg;
    EXPECT_THROW(least_squares.set_linear_solver_options(options), std::invalid_argument);
    EXPECT_EQ(least_squares.linear_solver_options().backend, LinearSolverBackend::Gmres);
    least_squares.set_projection_method(Method::Anderson);
    least_squares.set_linear_solver_options(options);
    EXPECT_THROW(least_squares.set_projection_method(Method::AssembledFullOperator), std::invalid_argument);
    EXPECT_EQ(least_squares.projection_method(), Method::Anderson);
}
}
