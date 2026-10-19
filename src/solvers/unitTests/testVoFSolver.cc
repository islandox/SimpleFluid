/**
 * @file testVoFSolver.cc
 * @brief Independent conservative and physical checks for the uniform VoF core.
 */

#include "solvers/VoFSolver.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{

using SimpleFluid::VoFMaterial;
using SimpleFluid::VoFSolver;
using SimpleFluid::VoFSolverOptions;
using SimpleFluid::VoF::FaceField;
using SimpleFluid::VoF::UniformGrid;
using SimpleFluid::VoF::Vector;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double pi = 3.141592653589793238462643383279502884;

class VoFSolverTest : public testing::Test
{
protected:
    void SetUp() override
    {
        if (Tpetra::getDefaultComm()->getSize() != 1)
            GTEST_SKIP() << "The uniform Cartesian VoF baseline is serial.";
    }
};

UniformGrid cube(std::size_t n)
{
    UniformGrid grid;
    grid.cells = {n, n, n};
    grid.spacing = {1.0 / n, 1.0 / n, 1.0 / n};
    return grid;
}

VoFMaterial inviscid(double density_ratio = 10000.0, double sigma = 0.0)
{
    VoFMaterial material;
    material.liquid_density = 1.0;
    material.gas_density = 1.0 / density_ratio;
    material.liquid_viscosity = 0.0;
    material.gas_viscosity = 0.0;
    material.surface_tension = sigma;
    return material;
}

FaceField faces(const UniformGrid& grid, Vector value = {})
{
    FaceField field;
    for (std::size_t axis = 0; axis < 3; ++axis)
        field[axis].assign(grid.face_count(axis), value[axis]);
    return field;
}

double gas_volume(const UniformGrid& grid, const std::vector<double>& fraction)
{
    return grid.cell_volume() * std::accumulate(fraction.begin(), fraction.end(), 0.0);
}

Vector gas_centroid(const UniformGrid& grid, const std::vector<double>& fraction)
{
    Vector result{};
    const double total_fraction = std::accumulate(fraction.begin(), fraction.end(), 0.0);
    for (std::size_t c = 0; c < fraction.size(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            result[a] += fraction[c] * grid.cell_center(c)[a] / total_fraction;
    return result;
}

Vector momentum(const UniformGrid& grid, const VoFMaterial& material, const VoFSolver& solver)
{
    Vector result{};
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const double f = solver.gas_fraction()[c];
        const double mass = grid.cell_volume() * (material.gas_density * f + material.liquid_density * (1.0 - f));
        for (std::size_t a = 0; a < 3; ++a)
            result[a] += mass * solver.velocity()[c][a];
    }
    return result;
}

double maximum_divergence(const UniformGrid& grid, const FaceField& field)
{
    double result = 0.0;
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto low = grid.cell_coordinates(c);
        double divergence = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            auto high = low;
            ++high[a];
            divergence += (field[a][grid.face_index(a, high[0], high[1], high[2])] -
                              field[a][grid.face_index(a, low[0], low[1], low[2])]) /
                          grid.spacing[a];
        }
        result = std::max(result, std::abs(divergence));
    }
    return result;
}

struct QuadraturePoint
{
    double node;
    double weight;
};

const std::array<QuadraturePoint, 32>& sphere_quadrature()
{
    static const auto rule = []
    {
        constexpr std::size_t order = 64;
        std::array<QuadraturePoint, order / 2> result{};
        const auto polynomial = [](double x)
        {
            double previous = 1.0, current = x;
            for (std::size_t degree = 2; degree <= order; ++degree)
            {
                const double next = ((2.0 * degree - 1.0) * x * current - (degree - 1.0) * previous) / degree;
                previous = current;
                current = next;
            }
            return std::array<double, 2>{current, order * (x * current - previous) / (x * x - 1.0)};
        };
        for (std::size_t q = 0; q < result.size(); ++q)
        {
            double x = std::cos(pi * (q + 0.75) / (order + 0.5));
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                const auto [value, derivative] = polynomial(x);
                const double update = value / derivative;
                x -= update;
                if (std::abs(update) < 4.0 * std::numeric_limits<double>::epsilon())
                    break;
            }
            const double derivative = polynomial(x)[1];
            result[q] = {x, 2.0 / ((1.0 - x * x) * derivative * derivative)};
        }
        return result;
    }();
    return rule;
}

// Sixty-four-point Gauss-Legendre quadrature is independent of PLIC geometry.
// Integrate each sphere's exact x-directed chord over its y-z cell footprint.
// Lower quadrature orders introduce fraction noise amplified by curvature's
// second spatial derivatives; the 64-point rule reduces that input sensitivity.
std::vector<double> sphere_fraction(const UniformGrid& grid, Vector center, double radius)
{
    const auto& quadrature = sphere_quadrature();
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto x = grid.cell_center(c);
        double nearest2 = 0.0, farthest2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            const double distance = std::abs(x[a] - center[a]);
            const double nearest = std::max(0.0, distance - 0.5 * grid.spacing[a]);
            nearest2 += nearest * nearest;
            farthest2 += std::pow(distance + 0.5 * grid.spacing[a], 2);
        }
        if (nearest2 >= radius * radius)
            continue;
        if (farthest2 <= radius * radius)
        {
            fraction[c] = 1.0;
            continue;
        }
        const double xlo = x[0] - 0.5 * grid.spacing[0];
        const double xhi = x[0] + 0.5 * grid.spacing[0];
        for (std::size_t j = 0; j < quadrature.size(); ++j)
            for (int sj : {-1, 1})
                for (std::size_t k = 0; k < quadrature.size(); ++k)
                    for (int sk : {-1, 1})
                    {
                        const double dy = x[1] - center[1] + sj * 0.5 * grid.spacing[1] * quadrature[j].node;
                        const double dz = x[2] - center[2] + sk * 0.5 * grid.spacing[2] * quadrature[k].node;
                        const double chord2 = radius * radius - dy * dy - dz * dz;
                        if (chord2 <= 0.0)
                            continue;
                        const double chord = std::sqrt(chord2);
                        const double length =
                            std::max(0.0, std::min(xhi, center[0] + chord) - std::max(xlo, center[0] - chord));
                        fraction[c] += 0.25 * quadrature[j].weight * quadrature[k].weight * length / grid.spacing[0];
                    }
    }
    return fraction;
}

std::vector<double> wave_fraction(const UniformGrid& grid, double amplitude)
{
    // Subcell midpoint integration of an analytic graph; no production PLIC use.
    constexpr std::size_t samples = 256;
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto x = grid.cell_center(c);
        const double bottom = x[1] - 0.5 * grid.spacing[1];
        for (std::size_t q = 0; q < samples; ++q)
        {
            const double xq = x[0] + grid.spacing[0] * ((q + 0.5) / samples - 0.5);
            const double height = 0.5 + amplitude * std::cos(2.0 * pi * xq);
            fraction[c] += std::clamp((height - bottom) / grid.spacing[1], 0.0, 1.0) / samples;
        }
    }
    return fraction;
}

double wave_amplitude(const UniformGrid& grid, const std::vector<double>& fraction)
{
    double sum = 0.0;
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const double phase = 2.0 * pi * grid.cell_center(c)[0];
        sum += fraction[c] * grid.spacing[1] * std::cos(phase);
    }
    return 2.0 * sum / (grid.cells[0] * grid.cells[2]);
}

struct BubbleMeasurement
{
    double relative_pressure_error;
    double maximum_velocity;
    double relative_volume_error;
};

BubbleMeasurement static_bubble(std::size_t n)
{
    auto grid = cube(n);
    const auto material = inviscid(10000.0, 1.0);
    VoFSolverOptions options;
    options.pressure_relative_tolerance = 1e-12;
    options.divergence_tolerance = 1e-7;
    VoFSolver solver(grid, material, options);
    constexpr double radius = 0.25;
    auto initial = sphere_fraction(grid, {0.5, 0.5, 0.5}, radius);
    const double volume = gas_volume(grid, initial);
    solver.initialize(initial);
    const double dt = 0.5 * solver.stable_timestep();
    double maximum_speed = 0.0;
    for (int step = 0; step < 4; ++step)
    {
        (void) solver.step(dt);
        for (const auto& velocity : solver.velocity())
            maximum_speed = std::max(
                maximum_speed, std::sqrt(std::inner_product(velocity.begin(), velocity.end(), velocity.begin(), 0.0)));
    }
    double inside = 0.0, outside = 0.0;
    std::size_t count_inside = 0, count_outside = 0;
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto x = grid.cell_center(c);
        const double r2 = std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2) + std::pow(x[2] - 0.5, 2);
        if (r2 < std::pow(0.6 * radius, 2))
        {
            inside += solver.pressure()[c];
            ++count_inside;
        }
        if (r2 > std::pow(1.5 * radius, 2))
        {
            outside += solver.pressure()[c];
            ++count_outside;
        }
    }
    const double jump = inside / count_inside - outside / count_outside;
    const double expected_jump = 2.0 * material.surface_tension / radius;
    return {std::abs(jump / expected_jump - 1.0), maximum_speed,
        std::abs(gas_volume(grid, solver.gas_fraction()) / volume - 1.0)};
}

// The original 4x4x2 split-transport counterexample is divergence-free:
// transverse velocity varies in x/y only, so projection cannot bound it for us.
TEST_F(VoFSolverTest, SplitPhaseMassPreservesPassiveMomentumBounds)
{
    UniformGrid grid;
    grid.cells = {4, 4, 2};
    grid.spacing = {1, 1, 100};
    grid.periodic = {true, true, true};
    auto material = inviscid();
    material.liquid_density = 1000;
    material.gas_density = 1;
    VoFSolver flow(grid, material);
    std::vector<double> fraction(grid.cell_count());
    std::vector<Vector> velocity(grid.cell_count(), Vector{1, 1, 0});
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto ijk = grid.cell_coordinates(c);
        fraction[c] = ijk[0] == 0 ? 0 : 1;
        velocity[c][2] = ijk[0] == 1 && ijk[1] == 1 ? 1 : 0;
    }
    flow.initialize(fraction, velocity);
    const auto before = flow.diagnostics();
    // Three accepted steps exercise each cyclic first-sweep order.
    for (int step = 0; step < 3; ++step)
    {
        const auto after = flow.step(0.1);
        double lo = 1, hi = 0;
        for (const auto& u : flow.velocity())
        {
            lo = std::min(lo, u[2]);
            hi = std::max(hi, u[2]);
            EXPECT_NEAR(u[0], 1, 2e-11);
            EXPECT_NEAR(u[1], 1, 2e-11);
        }
        RecordProperty("minimum_passive_velocity_" + std::to_string(step), lo);
        EXPECT_GE(lo, -2e-11);
        EXPECT_LE(hi, 1 + 2e-11);
        EXPECT_NEAR(after.gas_volume, before.gas_volume, 2e-9);
        for (std::size_t a = 0; a < 3; ++a)
            EXPECT_NEAR(after.momentum[a], before.momentum[a], 2e-11 * std::max(1.0, std::abs(before.momentum[a])));
    }
}

} // namespace

TEST(VoFSolverParallelTest, RejectsDistributedConstruction)
{
    if (Tpetra::getDefaultComm()->getSize() == 1)
        GTEST_SKIP() << "Run with two ranks to check explicit serial rejection.";
    EXPECT_THROW(VoFSolver(cube(2)), std::exception);
}

TEST_F(VoFSolverTest, PublishesNativeFieldsAndConservativePhaseMasses)
{
    auto grid = cube(4);
    grid.origin = {-0.5, 0.25, 2.0};
    auto material = inviscid();
    VoFSolver solver(grid, material);
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t c = 0; c < fraction.size(); ++c)
        fraction[c] = static_cast<double>(c % 5) / 4.0;
    solver.initialize(fraction);
    ASSERT_EQ(solver.mesh()->num_owned_cells(), grid.cell_count());
    ASSERT_EQ(solver.fraction_field().num_owned_cells(), grid.cell_count());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto lid = static_cast<SimpleFluid::DefaultTpetraTypes::local_ordinal_type>(c);
        const auto center = solver.mesh()->cell_centroid(lid);
        const auto index =
            grid.cell_index(static_cast<std::size_t>(std::floor((center.x - grid.origin[0]) / grid.spacing[0])),
                static_cast<std::size_t>(std::floor((center.y - grid.origin[1]) / grid.spacing[1])),
                static_cast<std::size_t>(std::floor((center.z - grid.origin[2]) / grid.spacing[2])));
        EXPECT_DOUBLE_EQ(solver.fraction_field().value(lid), fraction[index]);
        EXPECT_DOUBLE_EQ(solver.pressure_field().value(lid), solver.pressure()[index]);
        const auto value = solver.velocity_field().value(lid);
        EXPECT_DOUBLE_EQ(value.x, 0.0);
        EXPECT_DOUBLE_EQ(value.y, 0.0);
        EXPECT_DOUBLE_EQ(value.z, 0.0);
    }
    const double volume = gas_volume(grid, fraction);
    EXPECT_NEAR(solver.diagnostics().gas_mass, volume * material.gas_density, 1e-15);
    EXPECT_NEAR(solver.diagnostics().liquid_mass, (1.0 - volume) * material.liquid_density, 1e-14);
}

TEST_F(VoFSolverTest, RejectsInvalidMaterialAndFractionWithoutPublishing)
{
    const auto grid = cube(3);
    auto invalid = inviscid();
    invalid.gas_density = 0.0;
    EXPECT_THROW(VoFSolver(grid, invalid), std::invalid_argument);
    invalid = inviscid();
    invalid.surface_tension = -1.0;
    EXPECT_THROW(VoFSolver(grid, invalid), std::invalid_argument);
    VoFSolver solver(grid, inviscid());
    const std::vector<double> accepted(grid.cell_count(), 0.25);
    solver.initialize(accepted);
    auto bad = accepted;
    bad[7] = 1.01;
    EXPECT_THROW(solver.initialize(bad), std::invalid_argument);
    EXPECT_EQ(solver.gas_fraction(), accepted);
    bad[7] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(solver.initialize(bad), std::invalid_argument);
    EXPECT_EQ(solver.gas_fraction(), accepted);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
}

TEST_F(VoFSolverTest, TranslatesDensityContrastWithConsistentMassAndMomentum)
{
    auto grid = cube(10);
    grid.periodic = {true, true, true};
    const auto material = inviscid();
    VoFSolver solver(grid, material);
    auto fraction = sphere_fraction(grid, {0.35, 0.45, 0.55}, 0.2);
    const Vector velocity{0.3, -0.2, 0.1};
    solver.initialize(fraction, faces(grid, velocity));
    const auto initial_momentum = momentum(grid, material, solver);
    const auto initial_center = gas_centroid(grid, fraction);
    const double volume = gas_volume(grid, fraction);
    const double dt = 0.7 * solver.stable_timestep();
    for (int step = 0; step < 12; ++step)
    {
        const auto report = solver.step(dt);
        EXPECT_LT(report.maximum_divergence, 1e-9);
        EXPECT_NEAR(gas_volume(grid, solver.gas_fraction()), volume, 2e-13);
        for (std::size_t c = 0; c < grid.cell_count(); ++c)
        {
            EXPECT_GE(solver.gas_fraction()[c], -2e-12);
            EXPECT_LE(solver.gas_fraction()[c], 1.0 + 2e-12);
            for (std::size_t a = 0; a < 3; ++a)
                EXPECT_NEAR(solver.velocity()[c][a], velocity[a], 2e-9);
        }
    }
    const auto final_momentum = momentum(grid, material, solver);
    const auto final_center = gas_centroid(grid, solver.gas_fraction());
    for (std::size_t a = 0; a < 3; ++a)
    {
        EXPECT_NEAR(final_momentum[a], initial_momentum[a], 2e-11);
        EXPECT_NEAR(final_center[a], initial_center[a] + 12.0 * dt * velocity[a], 0.02);
    }
    EXPECT_NE(solver.gas_fraction(), fraction);
    EXPECT_NEAR(solver.time(), 12.0 * dt, 1e-14);
}

TEST_F(VoFSolverTest, ProjectsNonSolenoidalFaceVelocityAtLargeDensityContrast)
{
    auto grid = cube(8);
    grid.periodic = {true, true, true};
    auto field = faces(grid);
    for (std::size_t k = 0; k < grid.cells[2]; ++k)
        for (std::size_t j = 0; j < grid.cells[1]; ++j)
            for (std::size_t i = 0; i <= grid.cells[0]; ++i)
                field[0][grid.face_index(0, i, j, k)] =
                    std::sin(2.0 * pi * i / grid.cells[0]) * std::cos(2.0 * pi * (j + 0.5) / grid.cells[1]);
    // Use exactly matching duplicated periodic end values.
    for (std::size_t k = 0; k < grid.cells[2]; ++k)
        for (std::size_t j = 0; j < grid.cells[1]; ++j)
            field[0][grid.face_index(0, grid.cells[0], j, k)] = field[0][grid.face_index(0, 0, j, k)];
    ASSERT_GT(maximum_divergence(grid, field), 1.0);
    const auto fraction = sphere_fraction(grid, {0.5, 0.5, 0.5}, 0.28);
    VoFSolver solver(grid, inviscid());
    solver.initialize(fraction, field);
    EXPECT_LT(maximum_divergence(grid, solver.face_velocity()), 1e-8);
    EXPECT_GT(solver.diagnostics().pressure_iterations, 0);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_LT(solver.diagnostics().kinetic_energy, 0.125);
    (void) solver.step(0.2 * solver.stable_timestep());
    EXPECT_LT(maximum_divergence(grid, solver.face_velocity()), 1e-8);
}

TEST_F(VoFSolverTest, PeriodicProjectionAndTransportConserveNonuniformMomentum)
{
    auto grid = cube(8);
    grid.periodic = {true, true, true};
    const auto material = inviscid(10000.0);
    const auto fraction = sphere_fraction(grid, {0.38, 0.48, 0.55}, 0.28);
    std::vector<Vector> initial_velocity(grid.cell_count());
    Vector initial_momentum{};
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto x = grid.cell_center(c);
        initial_velocity[c] = {0.2 + 0.03 * std::sin(2.0 * pi * x[0]) + 0.02 * std::sin(2.0 * pi * x[1]),
            -0.1 + 0.02 * std::cos(2.0 * pi * x[0]), 0.01 * std::sin(2.0 * pi * x[2])};
        const double mass =
            grid.cell_volume() * (material.gas_density * fraction[c] + material.liquid_density * (1.0 - fraction[c]));
        for (std::size_t a = 0; a < 3; ++a)
            initial_momentum[a] += mass * initial_velocity[c][a];
    }
    VoFSolver solver(grid, material);
    solver.initialize(fraction, initial_velocity);
    const auto projected_momentum = momentum(grid, material, solver);
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(projected_momentum[a], initial_momentum[a], 2e-11);
    EXPECT_LT(maximum_divergence(grid, solver.face_velocity()), 1e-8);
    const double volume = gas_volume(grid, fraction);
    for (int step = 0; step < 3; ++step)
        (void) solver.step(0.2 * solver.stable_timestep());
    const auto transported_momentum = momentum(grid, material, solver);
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(transported_momentum[a], initial_momentum[a], 3e-11);
    EXPECT_NEAR(gas_volume(grid, solver.gas_fraction()), volume, 2e-13);
}

TEST_F(VoFSolverTest, EnforcesAdvectiveCapillaryAndViscousTimestepBounds)
{
    auto grid = cube(4);
    grid.periodic = {true, true, true};
    VoFSolverOptions options;
    const std::vector<double> f(grid.cell_count(), 0.0);
    VoFSolver transport(grid, inviscid(), options);
    transport.initialize(f, faces(grid, {2.0, 1.0, 0.0}));
    EXPECT_LE(transport.stable_timestep(), options.max_courant / (3.0 / grid.spacing[0]) * (1.0 + 1e-12));

    const auto material = inviscid(1000.0, 0.5);
    VoFSolver capillary(grid, material, options);
    auto two_phase = f;
    std::fill(two_phase.begin(), two_phase.begin() + two_phase.size() / 2, 1.0);
    capillary.initialize(two_phase);
    const double bound =
        options.capillary_safety * std::sqrt((material.liquid_density + material.gas_density) *
                                             std::pow(grid.spacing[0], 3) / (2.0 * pi * material.surface_tension));
    EXPECT_GT(capillary.stable_timestep(), 0.0);
    EXPECT_LE(capillary.stable_timestep(), bound * (1.0 + 1e-12));

    auto viscous_material = inviscid(1.0);
    viscous_material.liquid_viscosity = 2.0;
    viscous_material.gas_viscosity = 2.0;
    VoFSolver viscous(grid, viscous_material, options);
    viscous.initialize(f);
    const double forward_euler_bound = 1.0 / (2.0 * 2.0 * 3.0 / std::pow(grid.spacing[0], 2));
    EXPECT_GT(viscous.stable_timestep(), 0.0);
    EXPECT_LE(viscous.stable_timestep(), forward_euler_bound * (1.0 + 1e-12));
}

TEST_F(VoFSolverTest, SinglePhaseGravityRemainsHydrostaticAtSlipWalls)
{
    const auto grid = cube(5);
    auto material = inviscid(1.0);
    material.liquid_density = 2.0;
    VoFSolverOptions options;
    options.gravity = {0.0, -0.4, 0.0};
    VoFSolver solver(grid, material, options);
    solver.initialize(std::vector<double>(grid.cell_count(), 0.0));
    for (int step = 0; step < 5; ++step)
        (void) solver.step(0.02);
    const auto reference = grid.cell_center(0);
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const double expected_difference =
            material.liquid_density * options.gravity[1] * (grid.cell_center(c)[1] - reference[1]);
        EXPECT_NEAR(solver.pressure()[c] - solver.pressure()[0], expected_difference, 2e-10);
        for (std::size_t a = 0; a < 3; ++a)
            EXPECT_NEAR(solver.velocity()[c][a], 0.0, 2e-10);
    }
    for (const auto& component : solver.face_velocity())
        for (const double value : component)
            EXPECT_NEAR(value, 0.0, 2e-10);
    EXPECT_LT(maximum_divergence(grid, solver.face_velocity()), 1e-9);
}

TEST_F(VoFSolverTest, ClosedFaceInitializationProjectsAwayTheEntireLongitudinalFlow)
{
    UniformGrid grid;
    grid.cells = {3, 1, 1};
    grid.spacing = {1.0, 1.0, 1.0};
    auto initial_faces = faces(grid);
    initial_faces[0] = {0.0, 1.0, 1.0, 0.0};
    VoFSolver solver(grid, inviscid(1.0));
    solver.initialize(std::vector<double>(grid.cell_count(), 0.0), initial_faces);
    // In a closed one-dimensional column every incompressible face velocity
    // must vanish. The initial cell velocities are the averages [.5,1,.5];
    // the pressure reaction must remove these without inventing wall drift.
    for (const auto& component : solver.face_velocity())
        for (const double value : component)
            EXPECT_NEAR(value, 0.0, 1e-12);
    for (const auto& velocity : solver.velocity())
        for (const double value : velocity)
            EXPECT_NEAR(value, 0.0, 1e-12);
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto lid = static_cast<SimpleFluid::DefaultTpetraTypes::local_ordinal_type>(c);
        const auto value = solver.velocity_field().value(lid);
        EXPECT_NEAR(value.x, 0.0, 1e-12);
        EXPECT_NEAR(value.y, 0.0, 1e-12);
        EXPECT_NEAR(value.z, 0.0, 1e-12);
    }
    // Uniform supplied cell velocities create the same interior face values,
    // but start with [1,1,1] cell momentum and need the cell-initializer wall
    // reaction. The two initialization paths must not share a false predictor.
    VoFSolver cell_solver(grid, inviscid(1.0));
    cell_solver.initialize(
        std::vector<double>(grid.cell_count(), 0.0), std::vector<Vector>(grid.cell_count(), {1.0, 0.0, 0.0}));
    for (const auto& component : cell_solver.face_velocity())
        for (const double value : component)
            EXPECT_NEAR(value, 0.0, 1e-12);
    for (const auto& velocity : cell_solver.velocity())
        for (const double value : velocity)
            EXPECT_NEAR(value, 0.0, 1e-12);
}

TEST_F(VoFSolverTest, CellInitializationPreservesASolenoidalSlipVortex)
{
    auto grid = cube(8);
    grid.cells[2] = 1;
    grid.spacing[2] = 1.0;
    std::vector<Vector> initial_velocity(grid.cell_count());
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto x = grid.cell_center(c);
        initial_velocity[c] = {
            std::sin(pi * x[0]) * std::cos(pi * x[1]), -std::cos(pi * x[0]) * std::sin(pi * x[1]), 0.0};
    }
    VoFSolver solver(grid, inviscid(1.0));
    solver.initialize(std::vector<double>(grid.cell_count(), 0.0), initial_velocity);
    // Interpolated interior faces and zero normal wall faces form an exactly
    // discrete-solenoidal slip vortex. No pressure impulse is needed, including
    // in boundary-adjacent cells whose centered normal velocities are nonzero.
    EXPECT_LT(maximum_divergence(grid, solver.face_velocity()), 1e-12);
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            EXPECT_NEAR(solver.velocity()[c][a], initial_velocity[c][a], 2e-13);
}

TEST_F(VoFSolverTest, PeriodicViscousShearDecaysAtTheContinuumRate)
{
    UniformGrid grid;
    grid.cells = {4, 24, 2};
    grid.spacing = {0.25, 1.0 / 24.0, 0.5};
    grid.periodic = {true, true, true};
    auto material = inviscid(1.0);
    constexpr double viscosity = 0.05;
    constexpr double initial_amplitude = 0.2;
    material.liquid_viscosity = viscosity;
    material.gas_viscosity = viscosity;
    std::vector<Vector> initial_velocity(grid.cell_count());
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
        initial_velocity[c][0] = initial_amplitude * std::sin(2.0 * pi * grid.cell_center(c)[1]);
    VoFSolver solver(grid, material);
    solver.initialize(std::vector<double>(grid.cell_count(), 0.0), initial_velocity);
    constexpr double dt = 0.001;
    for (int step = 0; step < 100; ++step)
        (void) solver.step(dt);
    const double expected_amplitude = initial_amplitude * std::exp(-viscosity * 4.0 * pi * pi * solver.time());
    double measured_amplitude = 0.0;
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const double shape = std::sin(2.0 * pi * grid.cell_center(c)[1]);
        measured_amplitude += 2.0 * solver.velocity()[c][0] * shape / grid.cell_count();
        EXPECT_NEAR(solver.velocity()[c][0], expected_amplitude * shape, 0.002 * initial_amplitude);
        EXPECT_NEAR(solver.velocity()[c][1], 0.0, 1e-12);
        EXPECT_NEAR(solver.velocity()[c][2], 0.0, 1e-12);
    }
    // At 24 cells/wavelength the second-order Laplacian has a 0.57% rate
    // error, giving about 0.1% amplitude error over this physical horizon.
    EXPECT_NEAR(measured_amplitude / expected_amplitude, 1.0, 0.002);
    EXPECT_LT(measured_amplitude, 0.9 * initial_amplitude);
    EXPECT_NEAR(momentum(grid, material, solver)[0], 0.0, 1e-12);
}

TEST_F(VoFSolverTest, RejectsOversizedStepWithoutChangingAcceptedFields)
{
    auto grid = cube(4);
    grid.periodic = {true, true, true};
    VoFSolver solver(grid, inviscid(1000.0, 1.0));
    solver.initialize(std::vector<double>(grid.cell_count(), 0.3), faces(grid, {0.2, 0.0, 0.0}));
    const auto fraction = solver.gas_fraction();
    const auto velocity = solver.velocity();
    const auto face_velocity = solver.face_velocity();
    const auto pressure = solver.pressure();
    EXPECT_THROW((void) solver.step(2.0 * solver.stable_timestep()), std::exception);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_EQ(solver.velocity(), velocity);
    EXPECT_EQ(solver.face_velocity(), face_velocity);
    EXPECT_EQ(solver.pressure(), pressure);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_THROW((void) solver.step(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
}

TEST_F(VoFSolverTest, FailedPressureSolveRollsBackTheWholeTrial)
{
    auto grid = cube(8);
    VoFSolverOptions options;
    options.max_pressure_iterations = 1;
    options.pressure_relative_tolerance = 1e-14;
    VoFSolver solver(grid, inviscid(10000.0, 1.0), options);
    solver.initialize(sphere_fraction(grid, {0.47, 0.51, 0.54}, 0.27));
    const auto fraction = solver.gas_fraction();
    const auto velocity = solver.velocity();
    const auto face_velocity = solver.face_velocity();
    const auto pressure = solver.pressure();
    const auto curvature = solver.curvature();
    EXPECT_THROW((void) solver.step(0.5 * solver.stable_timestep()), std::runtime_error);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_EQ(solver.velocity(), velocity);
    EXPECT_EQ(solver.face_velocity(), face_velocity);
    EXPECT_EQ(solver.pressure(), pressure);
    EXPECT_EQ(solver.curvature(), curvature);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto lid = static_cast<SimpleFluid::DefaultTpetraTypes::local_ordinal_type>(c);
        EXPECT_DOUBLE_EQ(solver.fraction_field().value(lid), fraction[c]);
        EXPECT_DOUBLE_EQ(solver.pressure_field().value(lid), pressure[c]);
    }
}

TEST_F(VoFSolverTest, ComputedSphereCurvatureProducesConvergentLaplacePressure)
{
    // No radius or analytical curvature enters the solver. The pressure jump
    // is measured from pure interior/exterior cells after four accepted steps.
    // This is an early startup check; the driver measures spurious currents at
    // a common physical time because four dt(h) steps are unequal horizons.
    const auto coarse = static_bubble(12);
    const auto fine = static_bubble(24);
    EXPECT_LT(coarse.relative_pressure_error, 0.15);
    EXPECT_LT(fine.relative_pressure_error, 0.08);
    EXPECT_LT(fine.relative_pressure_error, coarse.relative_pressure_error);
    EXPECT_LT(coarse.relative_volume_error, 1e-11);
    EXPECT_LT(fine.relative_volume_error, 1e-11);
    // Capillary speed sqrt(sigma/(rho_l R)) is 2 m/s in this scaled problem.
    EXPECT_LT(coarse.maximum_velocity, 0.2);
    EXPECT_LT(fine.maximum_velocity, 0.2);
}

TEST_F(VoFSolverTest, EvolvingCapillaryWaveHasTheFiniteDepthLinearFrequency)
{
    UniformGrid grid = cube(24);
    grid.cells[2] = 2;
    grid.periodic = {true, false, true};
    constexpr double amplitude = 0.01;
    const auto material = inviscid(10.0, 1.0);
    VoFSolver solver(grid, material);
    solver.initialize(wave_fraction(grid, amplitude));
    const double initial_amplitude = wave_amplitude(grid, solver.gas_fraction());
    const double k = 2.0 * pi;
    // Linear potentials with zero normal velocity at y=0,1 give coth(kh)
    // inertia. No gravity or viscosity is present in this numerical experiment.
    const double omega = std::sqrt(
        material.surface_tension * k * k * k / ((material.liquid_density + material.gas_density) / std::tanh(0.5 * k)));
    const double end_time = pi / omega;
    double previous_time = 0.0;
    double previous_amplitude = initial_amplitude;
    double first_zero = -1.0;
    double sum_squared_error = 0.0;
    std::size_t samples = 0;
    const double volume = gas_volume(grid, solver.gas_fraction());
    while (solver.time() < end_time - 1e-14)
    {
        const double dt = std::min(0.65 * solver.stable_timestep(), end_time - solver.time());
        (void) solver.step(dt);
        const double current = wave_amplitude(grid, solver.gas_fraction());
        if (first_zero < 0.0 && previous_amplitude > 0.0 && current <= 0.0)
            first_zero =
                previous_time + (solver.time() - previous_time) * previous_amplitude / (previous_amplitude - current);
        const double relative_error = current / initial_amplitude - std::cos(omega * solver.time());
        sum_squared_error += relative_error * relative_error;
        ++samples;
        previous_time = solver.time();
        previous_amplitude = current;
        ASSERT_LT(samples, 3000u);
    }
    ASSERT_GT(first_zero, 0.0);
    const double measured_omega = 0.5 * pi / first_zero;
    EXPECT_NEAR(measured_omega / omega, 1.0, 0.15);
    EXPECT_LT(previous_amplitude / initial_amplitude, -0.5);
    EXPECT_LT(std::sqrt(sum_squared_error / samples), 0.25);
    EXPECT_NEAR(gas_volume(grid, solver.gas_fraction()), volume, 2e-12);
}
