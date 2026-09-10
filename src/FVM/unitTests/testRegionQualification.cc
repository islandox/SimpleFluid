/** @file testRegionQualification.cc @brief Accuracy and decomposition qualification of compact regions. */
#include "FVM/FaceFlux.hh"
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Scalar = ScalarCellFieldStored<Pack>;
using Flux = ScalarFaceFieldStored<Pack>;
using Vec = MeshUtils::Vec3;
constexpr real_t pi = std::numbers::pi_v<real_t>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

ArrReal coordinates(int count, real_t lower = 0, real_t upper = 1)
{
    ArrReal result;
    for (int i = 0; i <= count; ++i) result.push_back(lower + (upper - lower) * i / count);
    return result;
}

SP<Meshes::MultiRegionMesh> decomposed_grid(int nx, int ny, int nz, int regions, bool periodic = false)
{
    using namespace Meshes;
    std::vector<MultiRegionMesh::Region> parts;
    std::vector<MultiRegionMesh::Interface> interfaces;
    for (int r = 0; r < regions; ++r)
    {
        ArrReal x;
        for (int i = r * nx / regions; i <= (r + 1) * nx / regions; ++i) x.push_back(real_t(i) / nx);
        parts.push_back(cartesian_region("r" + std::to_string(r), {{x, coordinates(ny), coordinates(nz)}}));
        if (r) interfaces.push_back(StructuredPatchInterface{{size_t(r - 1), 1}, {size_t(r), 0}});
    }
    if (periodic)
    {
        StructuredPatchInterface closure{{0, 0}, {size_t(regions - 1), 1}};
        closure.periodic_translation = Vec{1, 0, 0};
        interfaces.push_back(closure);
    }
    return std::make_shared<MultiRegionMesh>(std::move(parts), std::move(interfaces));
}

struct RefinementCase
{
    int ratio;
    real_t interface_x;
    int normal_refinement;
};

// Changing normal_refinement changes center-to-center nonorthogonality at the
// same planar coarse/fine tiling; no unsupported sheared provider is needed.
SP<Meshes::MultiRegionMesh> coarse_fine_grid(int n, RefinementCase test)
{
    using namespace Meshes;
    const int nx = n * test.normal_refinement;
    auto coarse = cartesian_region("coarse", {{coordinates(nx, 0, test.interface_x), coordinates(n), {0, 1}}});
    auto fine = cartesian_region("fine", {{coordinates(nx, test.interface_x, 1), coordinates(test.ratio * n), {0, 1}}});
    NonconformingInterface interface{0, 1, 1, 0, {}};
    std::vector<std::vector<uint64_t>> fine_rows(n);
    for (size_t f = 0; f < fine.layout().faces; ++f)
        if (fine.topology().boundary_id(f) == 0)
            fine_rows.at(std::min(n - 1, int(fine.geometry().face_centroid(f).y * n))).push_back(f);
    for (size_t f = 0; f < coarse.layout().faces; ++f)
        if (coarse.topology().boundary_id(f) == 1)
            interface.faces.push_back({f, fine_rows.at(std::min(n - 1, int(coarse.geometry().face_centroid(f).y * n)))});
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{std::move(coarse), std::move(fine)},
        std::vector<MultiRegionMesh::Interface>{std::move(interface)});
}

std::vector<real_t> global_sum(const std::vector<real_t>& local)
{
    std::vector<real_t> global(local.size());
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    return global;
}

std::vector<real_t> solve(const FVM::TransportSystem<Pack>& system, const SP<const Handle>& mesh)
{
    Pack::vector_type result(mesh->owned_cell_map(), true);
    BelosLinearSolver<> solver;
    LinearSolverOptions options;
    options.tolerance = 1e-12;
    options.max_iterations = 1000;
    EXPECT_TRUE(solver.solve(system.matrix, *system.rhs, result, options));
    const auto values = result.getData();
    return {values.begin(), values.end()};
}

void expect_compact(const Handle& mesh)
{
    EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(mesh.has_materialized_indexer());
}

real_t manufactured(Vec point) { return 1 + std::sin(pi * point.x) * std::sin(pi * point.y); }

struct ErrorNorms
{
    real_t global, interface_band, maximum_angle, relative_residual;
    int corrections;
};

ErrorNorms nonlinear_error(int n, RefinementCase test)
{
    SP<const Handle> mesh = std::make_shared<Handle>(coarse_fine_grid(n, test));
    Scalar exact(mesh, "manufactured"), correction(mesh, "partition_correction");
    Flux zero_flux(mesh, 0.0, "zero_flux");
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        exact.set_owned_value(c, manufactured(mesh->cell_centroid(c)));
        correction.set_owned_value(c, exact.value(c));
    }
    exact.sync_ghosts();
    correction.sync_ghosts();
    zero_flux.sync_ghosts();
    auto boundary = [&](int b, size_t i) { return manufactured(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    auto condition = [&](int b, size_t i)
    {
        const auto face = mesh->boundary_face_batch(b).face_lids[i];
        return std::abs(mesh->face_normal(face).z) > 0.5
            ? BoundaryCondition{BoundaryConditionType::Neumann, 0}
            : BoundaryCondition{BoundaryConditionType::Dirichlet, boundary(b, i)};
    };
    // A steady manufactured reaction-diffusion problem: (5 - Laplacian) phi
    // = 5 phi_exact + 2 pi^2 (phi_exact - 1), assembled through native transport.
    auto source = [&](int c) { return 2 * pi * pi * (manufactured(mesh->cell_centroid(c)) - 1); };
    // The existing implicit treatment places the remote half of partition-face
    // gradients on the RHS. Converge those lagged gradients while keeping the
    // manufactured transient/source terms fixed, so MPI and serial solve the
    // same reaction-diffusion problem rather than comparing one correction.
    FVM::TransportGeometryCache<Handle> geometry(*mesh);
    Teuchos::RCP<Pack::matrix_type> matrix;
    std::vector<real_t> values;
    bool converged = false;
    int corrections = 0;
    for (; corrections < 40; ++corrections)
    {
        const auto system = FVM::non_orthogonal_transport_system<Pack>(exact, zero_flux, 0.2, 1.0, condition, boundary,
            source, FVM::NonOrthogonalTreatment::Implicit, &correction, matrix, &geometry);
        matrix = system.matrix;
        values = solve(system, mesh);
        real_t local_update = 0, global_update = 0;
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            local_update = std::max(local_update, std::abs(values[c] - correction.value(c)));
            correction.set_owned_value(c, values[c]);
        }
        correction.sync_ghosts();
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_update, &global_update);
        if (global_update < 2e-12)
        {
            converged = true;
            ++corrections;
            break;
        }
    }
    EXPECT_TRUE(converged) << "Partition-gradient corrections must converge before measuring spatial accuracy.";
    // Rebuild the RHS from the final remote gradients before checking the
    // residual: the last linear solve alone only verifies a lagged equation.
    const auto final_system = FVM::non_orthogonal_transport_system<Pack>(exact, zero_flux, 0.2, 1.0, condition, boundary,
        source, FVM::NonOrthogonalTreatment::Implicit, &correction, matrix, &geometry);
    Pack::vector_type final_values(mesh->owned_cell_map(), true), residual(mesh->owned_cell_map(), true);
    {
        auto data = final_values.getDataNonConst();
        std::copy(values.begin(), values.end(), data.begin());
    }
    final_system.matrix->apply(final_values, residual);
    residual.update(-1.0, *final_system.rhs, 1.0);
    const auto relative_residual = residual.norm2() / final_system.rhs->norm2();
    EXPECT_LT(relative_residual, 2e-10);
    std::vector<real_t> local(4, 0.0);
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto p = mesh->cell_centroid(c);
        const auto error = values[c] - manufactured(p);
        const auto volume = mesh->cell_volume(c);
        local[0] += volume * error * error;
        local[1] += volume;
        const auto width = (p.x < test.interface_x ? test.interface_x : 1 - test.interface_x) / (n * test.normal_refinement);
        if (std::abs(p.x - test.interface_x) < width)
        {
            local[2] += volume * error * error;
            local[3] += volume;
        }
    }
    real_t local_angle = 0, global_angle = 0;
    for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (mesh->is_interior_face(f) && std::abs(mesh->face_centroid(f).x - test.interface_x) < 1e-12)
        {
            const auto direction = mesh->cell_center_vector(f, mesh->owner_cell(f));
            const auto cosine = std::abs(direction.dot(mesh->face_normal(f))) / direction.norm();
            local_angle = std::max(local_angle, std::acos(std::clamp(cosine, real_t(0), real_t(1))) * 180 / pi);
        }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_angle, &global_angle);
    const auto totals = global_sum(local);
    EXPECT_GT(totals[3], 0);
    expect_compact(*mesh);
    return {std::sqrt(totals[0] / totals[1]), std::sqrt(totals[2] / totals[3]), global_angle, relative_residual, corrections};
}

struct AdvectionErrors
{
    real_t amplitude, phase, l2;
};

AdvectionErrors periodic_advection(int n)
{
    SP<const Handle> mesh = std::make_shared<Handle>(decomposed_grid(n, 1, 1, 2, true));
    Scalar old(mesh, "sine");
    Flux flux(mesh, "uniform_velocity_flux");
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        old.set_owned_value(c, 1 + std::sin(2 * pi * mesh->cell_centroid(c).x));
    int local_remote_seams = 0, remote_seams = 0;
    for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        flux.set_owned_value(f, mesh->face_area_vector(f).x);
        if (std::abs(mesh->face_centroid(f).x) < 1e-12 && mesh->is_interior_face(f)
            && !mesh->is_owned_cell(mesh->neighbor_cell(f))) ++local_remote_seams;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_remote_seams, &remote_seams);
    if (Tpetra::getDefaultComm()->getSize() > 1) EXPECT_EQ(remote_seams, 1);
    old.sync_ghosts();
    flux.sync_ghosts();
    const real_t cfl = 0.5, dt = cfl / n;
    const int steps = 4 * n; // Two full periods; every feature crosses the periodic seam twice.
    Teuchos::RCP<Pack::matrix_type> cached_matrix;
    BelosLinearSolver<> solver;
    LinearSolverOptions options;
    options.tolerance = 1e-13;
    Pack::vector_type result(mesh->owned_cell_map(), true);
    for (int step = 0; step < steps; ++step)
    {
        const auto system = FVM::transport_system<Pack>(old, flux, dt, 0.0,
            [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; },
            [](int, size_t) { return 0.0; }, [](int) { return 0.0; }, cached_matrix);
        cached_matrix = system.matrix;
        EXPECT_TRUE(solver.solve(system.matrix, *system.rhs, result, options));
        const auto values = result.getData();
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c) old.set_owned_value(c, values[c]);
        old.sync_ghosts();
    }
    std::vector<real_t> local(4, 0.0);
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto theta = 2 * pi * mesh->cell_centroid(c).x;
        const auto value = old.value(c), volume = mesh->cell_volume(c);
        local[0] += volume * value;
        local[1] += 2 * volume * (value - 1) * std::sin(theta);
        local[2] += 2 * volume * (value - 1) * std::cos(theta);
        local[3] += volume * std::pow(value - 1 - std::sin(theta), 2);
    }
    const auto moments = global_sum(local);
    // Independent Fourier symbol of backward Euler + first-order upwind on a
    // uniform periodic grid. This distinguishes expected scheme damping from
    // seam-induced error without prescribing second-order transport accuracy.
    using Complex = std::complex<real_t>;
    const Complex symbol = real_t(1) / (real_t(1) + cfl * (real_t(1) - std::exp(Complex(0, -2 * pi / n))));
    const Complex predicted = std::pow(symbol, steps);
    EXPECT_NEAR(moments[0], 1, 2e-10);
    EXPECT_NEAR(moments[1], predicted.real(), 2e-9);
    EXPECT_NEAR(moments[2], predicted.imag(), 2e-9);
    const Complex measured(moments[1], moments[2]);
    const AdvectionErrors errors{std::abs(1 - std::abs(measured)), std::abs(std::arg(measured)), std::sqrt(moments[3])};
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << std::setprecision(10) << "REGION_ADVECTION ranks=" << Tpetra::getDefaultComm()->getSize()
                  << " cells=" << n << " periods=2 cfl=" << cfl << " amplitude=" << std::abs(measured)
                  << " amplitude_error=" << errors.amplitude << " phase_error=" << errors.phase
                  << " l2=" << errors.l2 << " inventory_error=" << std::abs(moments[0] - 1) << '\n';
    expect_compact(*mesh);
    return errors;
}

struct DecompositionResult
{
    std::vector<real_t> action, solution, flux, continuity;
};

DecompositionResult fixed_grid_result(int regions)
{
    constexpr int nx = 16, ny = 4, nz = 2;
    constexpr size_t cells = nx * ny * nz;
    constexpr size_t faces = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1);
    SP<const Handle> mesh = std::make_shared<Handle>(decomposed_grid(nx, ny, nz, regions));
    auto cell_key = [&](size_t c)
    {
        const auto p = mesh->cell_centroid(c);
        return size_t(int(p.x * nx) + nx * (int(p.y * ny) + ny * int(p.z * nz)));
    };
    auto boundary = [&](int b, size_t i) { return manufactured(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    auto condition = [&](int b, size_t i) { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary(b, i)}; };
    Scalar old(mesh, "manufactured"), pressure(mesh, "converged_scalar");
    Flux zero_flux(mesh, 0.0, "zero_flux"), flux(mesh, "pressure_weighted_flux");
    VectorCellFieldStored<Pack> velocity(mesh, "velocity");
    Pack::vector_type analytic(mesh->owned_cell_map(), true), action(mesh->owned_cell_map(), true);
    {
        auto values = analytic.getDataNonConst();
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto p = mesh->cell_centroid(c);
            old.set_owned_value(c, manufactured(p));
            values[c] = manufactured(p);
            velocity.set_owned_value(c, {std::sin(pi * p.y), std::sin(pi * p.z), std::sin(pi * p.x)});
        }
    }
    old.sync_ghosts();
    velocity.sync_ghosts();
    zero_flux.sync_ghosts();
    const auto system = FVM::transport_system<Pack>(old, zero_flux, 0.2, 1.0, condition, boundary,
        [&](int c) { return 2 * pi * pi * (manufactured(mesh->cell_centroid(c)) - 1); });
    system.matrix->apply(analytic, action);
    const auto solution = solve(system, mesh);
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c) pressure.set_owned_value(c, solution[c]);
    pressure.sync_ghosts();
    const auto velocity_boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, BoundaryConditionSet{});
    FVM::pressure_weighted_face_fluxes(velocity, pressure, 0.1, velocity_boundary, flux);
    flux.sync_ghosts();
    const auto divergence = FVM::cell_divergence_from_fluxes(*mesh, flux);
    DecompositionResult local{std::vector<real_t>(cells), std::vector<real_t>(cells),
        std::vector<real_t>(faces), std::vector<real_t>(cells)};
    const auto applied = action.getData();
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto key = cell_key(c);
        local.action[key] = applied[c];
        local.solution[key] = solution[c];
        local.continuity[key] = divergence[c];
    }
    for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto p = mesh->face_centroid(f), area = mesh->face_area_vector(f);
        size_t key;
        real_t orientation;
        if (std::abs(area.x) > 0)
        {
            key = size_t(std::lround(p.x * nx) + (nx + 1) * (int(p.y * ny) + ny * int(p.z * nz)));
            orientation = area.x;
        }
        else if (std::abs(area.y) > 0)
        {
            key = (nx + 1) * ny * nz + size_t(int(p.x * nx) + nx * (std::lround(p.y * ny) + (ny + 1) * int(p.z * nz)));
            orientation = area.y;
        }
        else
        {
            key = (nx + 1) * ny * nz + nx * (ny + 1) * nz
                + size_t(int(p.x * nx) + nx * (int(p.y * ny) + ny * std::lround(p.z * nz)));
            orientation = area.z;
        }
        local.flux.at(key) = std::copysign(real_t(1), orientation) * flux.value(f);
    }
    expect_compact(*mesh);
    return {global_sum(local.action), global_sum(local.solution), global_sum(local.flux), global_sum(local.continuity)};
}
} // namespace

TEST(RegionQualificationTest, NonlinearCoarseFineRefinementReportsGlobalAndInterfaceErrors)
{
    const std::array<RefinementCase, 4> cases{{{2, 0.5, 1}, {3, 0.5, 1}, {2, 0.25, 1}, {2, 0.5, 2}}};
    for (const auto test : cases)
    {
        ErrorNorms previous{};
        for (const int n : {4, 8, 16})
        {
            SCOPED_TRACE(testing::Message() << "n=" << n << " ratio=" << test.ratio
                << " interface=" << test.interface_x << " normal_refinement=" << test.normal_refinement);
            const auto error = nonlinear_error(n, test);
            const auto conforming = nonlinear_error(n, {1, test.interface_x, test.normal_refinement});
            if (n > 4)
            {
                EXPECT_LT(error.global, previous.global);
                EXPECT_LT(error.interface_band, previous.interface_band);
            }
            if (n == 16)
            {
                // These bounded smooth-field cases measure finest-level rates
                // near 1.92--2.01. A broad 1.7 regression floor detects degraded
                // interface convergence without asserting universal order two.
                EXPECT_GT(std::log2(previous.global / error.global), 1.7);
                EXPECT_GT(std::log2(previous.interface_band / error.interface_band), 1.7);
            }
            if (Tpetra::getDefaultComm()->getRank() == 0)
                std::cout << std::setprecision(10) << "REGION_REFINEMENT ranks=" << Tpetra::getDefaultComm()->getSize()
                          << " n=" << n << " ratio=" << test.ratio << " interface=" << test.interface_x
                          << " normal_refinement=" << test.normal_refinement << " max_angle=" << error.maximum_angle
                          << " corrections=" << error.corrections
                          << " relative_residual=" << error.relative_residual
                          << " global_l2=" << error.global << " interface_l2=" << error.interface_band
                          << " global_order=" << (n > 4 ? std::log2(previous.global / error.global) : 0)
                          << " interface_order=" << (n > 4 ? std::log2(previous.interface_band / error.interface_band) : 0)
                          << " conforming_global_l2=" << conforming.global
                          << " conforming_interface_l2=" << conforming.interface_band << '\n';
            previous = error;
        }
    }
}

TEST(RegionQualificationTest, PeriodicSineCrossesSeamTwiceAndMatchesUpwindFourierSymbol)
{
    AdvectionErrors previous{};
    for (const int n : {32, 64, 128})
    {
        SCOPED_TRACE(n);
        const auto error = periodic_advection(n);
        if (n > 32)
        {
            EXPECT_LT(error.amplitude, previous.amplitude);
            EXPECT_LT(error.phase, previous.phase);
            EXPECT_LT(error.l2, previous.l2);
        }
        previous = error;
    }
}

TEST(RegionQualificationTest, FixedPhysicalGridPreservesOperatorSolutionFluxAndContinuity)
{
    const auto reference = fixed_grid_result(1);
    for (const int regions : {2, 8})
    {
        SCOPED_TRACE(regions);
        const auto candidate = fixed_grid_result(regions);
        std::array<real_t, 4> maximum{};
        const std::array comparisons{std::pair{&reference.action, &candidate.action},
            std::pair{&reference.solution, &candidate.solution}, std::pair{&reference.flux, &candidate.flux},
            std::pair{&reference.continuity, &candidate.continuity}};
        for (size_t field = 0; field < comparisons.size(); ++field)
        {
            const auto [a, b] = comparisons[field];
            ASSERT_EQ(a->size(), b->size());
            for (size_t i = 0; i < a->size(); ++i) maximum[field] = std::max(maximum[field], std::abs((*a)[i] - (*b)[i]));
        }
        EXPECT_LT(maximum[0], 2e-12);
        EXPECT_LT(maximum[1], 2e-10);
        EXPECT_LT(maximum[2], 2e-10);
        EXPECT_LT(maximum[3], 2e-8);
        if (Tpetra::getDefaultComm()->getRank() == 0)
            std::cout << std::setprecision(10) << "REGION_DECOMPOSITION ranks=" << Tpetra::getDefaultComm()->getSize()
                      << " regions=" << regions << " operator_max=" << maximum[0] << " solution_max=" << maximum[1]
                      << " flux_max=" << maximum[2] << " continuity_max=" << maximum[3] << '\n';
    }
}
