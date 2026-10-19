/**
 * @file testGeometricVoF.cc
 * @brief Independent geometry, conservation, curvature, and transport checks.
 */

#include "FVM/GeometricVoF.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <vector>

namespace
{
using namespace SimpleFluid::VoF;

FaceField face_constant(const UniformGrid& grid, const Vector& value)
{
    FaceField result;
    for (std::size_t a = 0; a < 3; ++a)
        result[a].assign(grid.face_count(a), value[a]);
    return result;
}

// Independent quadrature of a disk's vertical chord, with breakpoints at every
// clipping transition. This supplies curvature tests without a known-radius API.
double disk_fraction(double xlo, double xhi, double ylo, double yhi, double radius)
{
    constexpr std::array<double, 8> abscissa{-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    constexpr std::array<double, 8> weights{0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
        0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    const double lo = std::max(xlo, -radius);
    const double hi = std::min(xhi, radius);
    if (hi <= lo)
        return 0.0;
    std::vector<double> breaks{lo, hi};
    for (const auto y : {ylo, yhi})
        if (std::abs(y) < radius)
            for (const auto sign : {-1.0, 1.0})
            {
                const double x = sign * std::sqrt(radius * radius - y * y);
                if (x > lo && x < hi)
                    breaks.push_back(x);
            }
    std::sort(breaks.begin(), breaks.end());
    double area = 0.0;
    for (std::size_t interval = 1; interval < breaks.size(); ++interval)
    {
        const double middle = 0.5 * (breaks[interval - 1] + breaks[interval]);
        const double half = 0.5 * (breaks[interval] - breaks[interval - 1]);
        for (std::size_t point = 0; point < abscissa.size(); ++point)
        {
            const double x = middle + half * abscissa[point];
            const double chord = std::sqrt(std::max(0.0, radius * radius - x * x));
            area += half * weights[point] * std::max(0.0, std::min(yhi, chord) - std::max(ylo, -chord));
        }
    }
    return std::clamp(area / ((xhi - xlo) * (yhi - ylo)), 0.0, 1.0);
}

std::vector<double> disk(const UniformGrid& grid, Vector center, double radius)
{
    std::vector<double> result(grid.cell_count());
    for (std::size_t cell = 0; cell < result.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        result[cell] = disk_fraction(p[0] - center[0] - grid.spacing[0] / 2.0, p[0] - center[0] + grid.spacing[0] / 2.0,
            p[1] - center[1] - grid.spacing[1] / 2.0, p[1] - center[1] + grid.spacing[1] / 2.0, radius);
    }
    return result;
}

// Integrate independently computed circular slices. Pure boxes are classified
// by their closest/farthest corners; interface boxes split at topology changes.
std::vector<double> sphere(const UniformGrid& grid, Vector center, double radius)
{
    constexpr std::array<double, 8> points{-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    constexpr std::array<double, 8> weights{0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
        0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    std::vector<double> result(grid.cell_count());
    for (std::size_t cell = 0; cell < result.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        Vector lo{};
        Vector hi{};
        double near2 = 0.0;
        double far2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            lo[a] = p[a] - center[a] - grid.spacing[a] / 2.0;
            hi[a] = p[a] - center[a] + grid.spacing[a] / 2.0;
            const double near = lo[a] <= 0.0 && hi[a] >= 0.0 ? 0.0 : std::min(std::abs(lo[a]), std::abs(hi[a]));
            const double far = std::max(std::abs(lo[a]), std::abs(hi[a]));
            near2 += near * near;
            far2 += far * far;
        }
        if (near2 >= radius * radius)
            continue;
        if (far2 <= radius * radius)
        {
            result[cell] = 1.0;
            continue;
        }
        const double zlo = std::max(lo[2], -radius);
        const double zhi = std::min(hi[2], radius);
        std::vector<double> breaks{zlo, zhi};
        for (const double x : {lo[0], 0.0, hi[0]})
            for (const double y : {lo[1], 0.0, hi[1]})
                if (x * x + y * y < radius * radius)
                    for (const double sign : {-1.0, 1.0})
                    {
                        const double z = sign * std::sqrt(radius * radius - x * x - y * y);
                        if (z > zlo && z < zhi)
                            breaks.push_back(z);
                    }
        std::sort(breaks.begin(), breaks.end());
        for (std::size_t interval = 1; interval < breaks.size(); ++interval)
        {
            const double middle = 0.5 * (breaks[interval] + breaks[interval - 1]);
            const double half = 0.5 * (breaks[interval] - breaks[interval - 1]);
            for (std::size_t q = 0; q < points.size(); ++q)
            {
                const double z = middle + half * points[q];
                const double slice_radius = std::sqrt(std::max(0.0, radius * radius - z * z));
                result[cell] +=
                    half * weights[q] * disk_fraction(lo[0], hi[0], lo[1], hi[1], slice_radius) / grid.spacing[2];
            }
        }
        result[cell] = std::clamp(result[cell], 0.0, 1.0);
    }
    return result;
}

void check_transport_ledger(const UniformGrid& grid, const std::vector<double>& before, const FaceField& velocity,
    double dt, const AdvectionResult& after)
{
    long double before_sum = 0.0;
    long double after_sum = 0.0;
    for (std::size_t cell = 0; cell < before.size(); ++cell)
    {
        before_sum += before[cell];
        after_sum += after.fraction[cell];
        EXPECT_GE(after.fraction[cell], -2.0e-12);
        EXPECT_LE(after.fraction[cell], 1.0 + 2.0e-12);
        const auto p = grid.cell_coordinates(cell);
        double expected = before[cell];
        for (std::size_t a = 0; a < 3; ++a)
        {
            auto q = p;
            ++q[a];
            const auto lo = grid.face_index(a, p[0], p[1], p[2]);
            const auto hi = grid.face_index(a, q[0], q[1], q[2]);
            expected += (after.gas_volume[a][lo] - after.gas_volume[a][hi]) / grid.cell_volume();
            EXPECT_NEAR(after.gas_volume[a][lo] + after.liquid_volume[a][lo], velocity[a][lo] * dt * grid.face_area(a),
                1.0e-16);
        }
        EXPECT_NEAR(after.fraction[cell], expected, 2.0e-13);
    }
    EXPECT_NEAR(static_cast<double>(before_sum), static_cast<double>(after_sum),
        2.0e-12 * std::max(1.0, static_cast<double>(before_sum)));
}

TEST(GeometricVoF, GridIndexingAndInputGuards)
{
    const UniformGrid grid{{3, 4, 5}, {1.0, 2.0, 3.0}, {0.2, 0.3, 0.4}};
    grid.validate();
    EXPECT_EQ(grid.cell_count(), 60U);
    EXPECT_EQ(grid.face_count(0), 80U);
    EXPECT_EQ(grid.face_count(1), 75U);
    EXPECT_EQ(grid.face_count(2), 72U);
    EXPECT_EQ(grid.cell_index(2, 3, 4), 59U);
    EXPECT_EQ(grid.cell_coordinates(59), (std::array<std::size_t, 3>{2, 3, 4}));
    EXPECT_NEAR(grid.cell_center(59)[0], 1.5, 1.0e-15);
    EXPECT_THROW((void) grid.cell_index(3, 0, 0), std::out_of_range);
    EXPECT_THROW((void) grid.face_index(0, 4, 0, 0), std::out_of_range);
    auto invalid = grid;
    invalid.cells[0] = 0;
    EXPECT_THROW(GeometricVoF{invalid}, std::invalid_argument);
    invalid = grid;
    invalid.spacing[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(GeometricVoF{invalid}, std::invalid_argument);
    invalid = grid;
    invalid.cells = {std::numeric_limits<std::size_t>::max(), 2, 2};
    EXPECT_THROW((void) invalid.cell_index(0, 0, 0), std::invalid_argument);
    EXPECT_THROW((void) invalid.face_index(0, 0, 0, 0), std::invalid_argument);
}

TEST(GeometricVoF, PlaneBoxIntersectionIsExactAndComplementary)
{
    const Vector lower{0.0, 0.0, 0.0};
    const Vector upper{1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(GeometricVoF::box_plane_volume(lower, upper, {1.0, 0.0, 0.0}, 0.3), 0.3);
    EXPECT_NEAR(GeometricVoF::box_plane_volume(lower, upper, {1.0, 1.0, 1.0}, 1.0), 1.0 / 6.0, 1.0e-15);
    EXPECT_NEAR(GeometricVoF::box_plane_volume(lower, upper, {1.0, 1.0, 0.0}, 1.0), 0.5, 1.0e-15);
    for (const auto normal :
        {Vector{1.0, 0.2, 0.7}, Vector{1.0e-12, 1.0, -1.0e-10}, Vector{-0.2, -0.5, 1.0}, Vector{1.0e-200, 1.0, 1.0}})
        for (const double offset : {-0.2, 0.1, 0.7, 1.2})
        {
            const Vector reversed{-normal[0], -normal[1], -normal[2]};
            const double gas = GeometricVoF::box_plane_volume(lower, upper, normal, offset);
            const double liquid = GeometricVoF::box_plane_volume(lower, upper, reversed, -offset);
            EXPECT_NEAR(gas + liquid, 1.0, 2.0e-15);
            EXPECT_GE(gas, 0.0);
            EXPECT_LE(gas, 1.0);
        }
    EXPECT_DOUBLE_EQ(GeometricVoF::box_plane_volume(lower, upper, {}, -1.0), 0.0);
    EXPECT_DOUBLE_EQ(GeometricVoF::box_plane_volume(lower, upper, {}, 1.0), 1.0);
}

TEST(GeometricVoF, ReconstructedPlanesPreserveAnisotropicCellVolume)
{
    const UniformGrid grid{{12, 10, 8}, {}, {0.02, 0.035, 0.045}};
    const Vector direction{0.3, -0.4, 0.7};
    std::vector<double> fraction(grid.cell_count());
    Vector lower{};
    Vector upper{};
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto center = grid.cell_center(cell);
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = center[a] - grid.spacing[a] / 2.0;
            upper[a] = center[a] + grid.spacing[a] / 2.0;
        }
        fraction[cell] =
            std::clamp(GeometricVoF::box_plane_volume(lower, upper, direction, 0.08) / grid.cell_volume(), 0.0, 1.0);
    }
    const auto planes = GeometricVoF(grid).reconstruct(fraction);
    for (std::size_t a = 0; a < 3; ++a)
    {
        lower[a] = -grid.spacing[a] / 2.0;
        upper[a] = grid.spacing[a] / 2.0;
    }
    std::size_t interior_planes = 0;
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        EXPECT_NEAR(
            GeometricVoF::box_plane_volume(lower, upper, planes[cell].normal, planes[cell].offset) / grid.cell_volume(),
            fraction[cell], 1.0e-14);
        const auto p = grid.cell_coordinates(cell);
        if (fraction[cell] > 0.01 && fraction[cell] < 0.99 && p[0] > 0 && p[0] + 1 < grid.cells[0] && p[1] > 0 &&
            p[1] + 1 < grid.cells[1] && p[2] > 0 && p[2] + 1 < grid.cells[2])
        {
            const double alignment =
                std::inner_product(direction.begin(), direction.end(), planes[cell].normal.begin(), 0.0) /
                std::hypot(direction[0], direction[1], direction[2]);
            EXPECT_GT(alignment, 0.98);
            ++interior_planes;
        }
    }
    EXPECT_GT(interior_planes, 20U);
}

TEST(GeometricVoF, PureAndUniformStatesHaveFiniteCurvatureAndZeroVelocityIdentity)
{
    const UniformGrid grid{{6, 5, 1}, {}, {0.1, 0.2, 1.0}};
    const GeometricVoF geometry(grid);
    for (const double constant : {0.0, 0.37, 1.0})
    {
        const std::vector<double> fraction(grid.cell_count(), constant);
        const auto result = geometry.advect(fraction, face_constant(grid, {}), 1.0);
        EXPECT_EQ(result.fraction, fraction);
        const auto curvature = geometry.curvature(fraction);
        for (const auto value : curvature)
            EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(GeometricVoF, PeriodicSlabTranslationIsBoundedConservativeAndExact)
{
    const UniformGrid grid{{20, 3, 2}, {}, {0.05, 0.2, 0.4}, {true, true, true}};
    const GeometricVoF geometry(grid);
    std::vector<double> initial(grid.cell_count());
    for (std::size_t cell = 0; cell < initial.size(); ++cell)
    {
        const auto p = grid.cell_coordinates(cell);
        initial[cell] = p[0] >= 17 || p[0] < 2 ? 1.0 : 0.0;
    }
    auto fraction = initial;
    const auto velocity = face_constant(grid, {0.2, 0.0, 0.0});
    const double dt = 0.0625;
    for (std::size_t step = 0; step < 12; ++step)
    {
        const auto next = geometry.advect(fraction, velocity, dt, step % 3);
        check_transport_ledger(grid, fraction, velocity, dt, next);
        fraction = next.fraction;
    }
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        auto p = grid.cell_coordinates(cell);
        p[0] = (p[0] + grid.cells[0] - 3) % grid.cells[0];
        EXPECT_NEAR(fraction[cell], initial[grid.cell_index(p[0], p[1], p[2])], 5.0e-14);
    }
}

TEST(GeometricVoF, ReversibleVortexRetainsBoundsAndBothPhaseFluxLedgers)
{
    constexpr std::size_t n = 24;
    const UniformGrid grid{{n, n, 1}, {}, {1.0 / n, 1.0 / n, 1.0}};
    auto fraction = disk(grid, {0.5, 0.7, 0.0}, 0.14);
    const auto initial = fraction;
    auto velocity = face_constant(grid, {});
    auto psi = [](std::size_t i, std::size_t j)
    {
        if (i == 0 || j == 0 || i == n || j == n)
            return 0.0;
        const double sx = std::sin(std::numbers::pi * static_cast<double>(i) / n);
        const double sy = std::sin(std::numbers::pi * static_cast<double>(j) / n);
        return sx * sx * sy * sy / (2.0 * std::numbers::pi);
    };
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i <= n; ++i)
            velocity[0][grid.face_index(0, i, j, 0)] = (psi(i, j + 1) - psi(i, j)) / grid.spacing[1];
    for (std::size_t j = 0; j <= n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            velocity[1][grid.face_index(1, i, j, 0)] = -(psi(i + 1, j) - psi(i, j)) / grid.spacing[0];
    const double dt = 0.2 / n;
    const GeometricVoF geometry(grid);
    for (std::size_t step = 0; step < 80; ++step)
    {
        if (step == 40)
            for (auto& direction : velocity)
                for (auto& value : direction)
                    value = -value;
        const auto next = geometry.advect(fraction, velocity, dt, step % 3);
        check_transport_ledger(grid, fraction, velocity, dt, next);
        fraction = next.fraction;
    }
    double shape_error = 0.0;
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        shape_error += std::abs(fraction[cell] - initial[cell]);
    shape_error /= std::accumulate(initial.begin(), initial.end(), 0.0);
    EXPECT_LT(shape_error, 0.08);
    RecordProperty("vortex_relative_shape_error", shape_error);
}

TEST(GeometricVoF, HeightFunctionCylinderCurvatureConvergesAndScalesWithLength)
{
    std::array<double, 2> errors{};
    for (std::size_t level = 0; level < 2; ++level)
    {
        const std::size_t n = 32U << level;
        const UniformGrid grid{{n, n, 1}, {}, {1.0 / n, 1.0 / n, 1.0}};
        const auto fraction = disk(grid, {0.501, 0.497, 0.0}, 0.2);
        const auto curvature = GeometricVoF(grid).curvature(fraction);
        double error = 0.0;
        double weight_sum = 0.0;
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        {
            const double weight = fraction[cell] * (1.0 - fraction[cell]);
            error += weight * std::abs(curvature[cell] - 5.0) / 5.0;
            weight_sum += weight;
        }
        errors[level] = error / weight_sum;
        EXPECT_LT(errors[level], 0.04);
        RecordProperty(level == 0 ? "coarse_curvature_error" : "fine_curvature_error", errors[level]);
        auto scaled = grid;
        for (auto& h : scaled.spacing)
            h *= 0.01;
        const auto scaled_curvature = GeometricVoF(scaled).curvature(fraction);
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
            EXPECT_NEAR(scaled_curvature[cell] * 0.01, curvature[cell], 1.0e-10);
    }
    EXPECT_LT(errors[1], 0.5 * errors[0]);
}

TEST(GeometricVoF, ThreeDimensionalSphereCurvatureAndObliqueTranslation)
{
    constexpr std::size_t n = 32;
    const UniformGrid grid{{n, n, n}, {}, {1.0 / n, 1.0 / n, 1.0 / n}, {true, true, true}};
    const GeometricVoF geometry(grid);
    const auto initial = sphere(grid, {0.501, 0.497, 0.503}, 0.2);
    const auto curvature = geometry.curvature(initial);
    double weighted_error = 0.0;
    double total_weight = 0.0;
    for (std::size_t cell = 0; cell < initial.size(); ++cell)
    {
        const double weight = initial[cell] * (1.0 - initial[cell]);
        weighted_error += weight * std::abs(curvature[cell] - 10.0) / 10.0;
        total_weight += weight;
    }
    EXPECT_LT(weighted_error / total_weight, 0.05);
    RecordProperty("sphere_relative_curvature_error", weighted_error / total_weight);
    auto fraction = initial;
    const auto velocity = face_constant(grid, {0.2, 0.1, -0.1});
    const double dt = 0.5 / n;
    for (std::size_t step = 0; step < 20; ++step)
    {
        const auto next = geometry.advect(fraction, velocity, dt, step % 3);
        check_transport_ledger(grid, fraction, velocity, dt, next);
        fraction = next.fraction;
    }
    double error = 0.0;
    for (std::size_t cell = 0; cell < initial.size(); ++cell)
    {
        auto p = grid.cell_coordinates(cell);
        p[0] = (p[0] + n - 2) % n;
        p[1] = (p[1] + n - 1) % n;
        p[2] = (p[2] + 1) % n;
        error += std::abs(fraction[cell] - initial[grid.cell_index(p[0], p[1], p[2])]);
    }
    const double relative_error = error / std::accumulate(initial.begin(), initial.end(), 0.0);
    EXPECT_LT(relative_error, 0.08);
    RecordProperty("sphere_relative_translation_error", relative_error);
}

TEST(GeometricVoF, AnisotropicHeightFunctionsUsePhysicalDerivatives)
{
    const UniformGrid grid{{64, 32, 1}, {}, {1.0 / 64, 1.0 / 32, 0.03}};
    const auto fraction = disk(grid, {0.501, 0.497, 0.0}, 0.2);
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    double weighted_error = 0.0;
    double weight_sum = 0.0;
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const double weight = fraction[cell] * (1.0 - fraction[cell]);
        weighted_error += weight * std::abs(curvature[cell] - 5.0) / 5.0;
        weight_sum += weight;
    }
    EXPECT_LT(weighted_error / weight_sum, 0.04);
    RecordProperty("anisotropic_relative_curvature_error", weighted_error / weight_sum);
}

TEST(GeometricVoF, TinyGasInventoriesAreTransportedWithoutClipping)
{
    const UniformGrid grid{{8, 1, 1}, {}, {0.01, 0.02, 0.03}, {true, false, false}};
    const GeometricVoF geometry(grid);
    std::vector<double> fraction(grid.cell_count(), 0.0);
    fraction[3] = 1.0e-12;
    const auto velocity = face_constant(grid, {-0.1, 0.0, 0.0});
    const auto transported = geometry.advect(fraction, velocity, 0.04);
    check_transport_ledger(grid, fraction, velocity, 0.04, transported);
    EXPECT_NEAR(std::accumulate(transported.fraction.begin(), transported.fraction.end(), 0.0), 1.0e-12, 1.0e-25);
    EXPECT_GT(transported.fraction[2], 0.0);
}

TEST(GeometricVoF, SurfaceFitRecoversQuadraticGraphCurvatureSignAndPhysicalScale)
{
    // Only three cells in the graph-normal direction deliberately prevent
    // complete height columns. Every cut polygon spans its full x/y footprint,
    // so these fractions integrate the manufactured quadratic exactly.
    const UniformGrid grid{{9, 9, 3}, {-0.045, -0.09, 0.0}, {0.01, 0.02, 0.03}};
    const double a = 0.4;
    const double b = 0.1;
    const double c = 0.2;
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        const double height = 0.045 + a * (p[0] * p[0] + grid.spacing[0] * grid.spacing[0] / 12.0) + b * p[0] * p[1] +
                              c * (p[1] * p[1] + grid.spacing[1] * grid.spacing[1] / 12.0);
        fraction[cell] = std::clamp((height - p[2]) / grid.spacing[2] + 0.5, 0.0, 1.0);
    }
    const auto center = grid.cell_index(4, 4, 1);
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    EXPECT_NEAR(curvature[center], -2.0 * (a + c), 2.0e-9);
    for (auto& value : fraction)
        value = 1.0 - value;
    const auto complemented = GeometricVoF(grid).curvature(fraction);
    EXPECT_NEAR(complemented[center], 2.0 * (a + c), 2.0e-9);
    auto scaled = grid;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        scaled.origin[axis] *= 0.01;
        scaled.spacing[axis] *= 0.01;
    }
    const auto scaled_curvature = GeometricVoF(scaled).curvature(fraction);
    EXPECT_NEAR(scaled_curvature[center] * 0.01, complemented[center], 2.0e-9);
}

TEST(GeometricVoF, TransverseHeightFilterPreservesAnisotropicQuadraticGraphs)
{
    // Complete nine-cell columns exercise height functions, independently of
    // the surface-fit tests that deliberately use only three vertical cells.
    const UniformGrid grid{{9, 9, 9}, {-0.045, -0.09, 0.0}, {0.01, 0.02, 0.03}};
    constexpr double a = 0.4;
    constexpr double b = 0.1;
    constexpr double c = 0.2;
    constexpr double slope_x = 0.03;
    constexpr double slope_y = -0.02;
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        const double height = 0.135 + slope_x * p[0] + slope_y * p[1] +
                              a * (p[0] * p[0] + grid.spacing[0] * grid.spacing[0] / 12.0) + b * p[0] * p[1] +
                              c * (p[1] * p[1] + grid.spacing[1] * grid.spacing[1] / 12.0);
        fraction[cell] = std::clamp((height - p[2]) / grid.spacing[2] + 0.5, 0.0, 1.0);
    }
    const double expected =
        -(2.0 * a * (1.0 + slope_y * slope_y) + 2.0 * c * (1.0 + slope_x * slope_x) - 2.0 * b * slope_x * slope_y) /
        std::pow(1.0 + slope_x * slope_x + slope_y * slope_y, 1.5);
    const auto center = grid.cell_index(4, 4, 4);
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    EXPECT_NEAR(curvature[center], expected, 2.0e-9);
    auto scaled = grid;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        scaled.origin[axis] *= 0.01;
        scaled.spacing[axis] *= 0.01;
    }
    EXPECT_NEAR(GeometricVoF(scaled).curvature(fraction)[center] * 0.01, expected, 2.0e-9);
}

TEST(GeometricVoF, TransverseHeightFilterDampsAlternatingModeAndPreservesTwoDimensionalLimit)
{
    const UniformGrid grid{{7, 7, 9}, {}, {0.01, 0.02, 0.01}};
    constexpr double amplitude = 1.0e-4;
    auto noisy_graph = [&](const UniformGrid& mesh, bool alternating_y)
    {
        std::vector<double> fraction(mesh.cell_count());
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        {
            const auto p = mesh.cell_coordinates(cell);
            const bool positive = (p[0] + (alternating_y ? p[1] : 0)) % 2 == 0;
            const double height = 0.045 + (positive ? amplitude : -amplitude);
            fraction[cell] = std::clamp((height - mesh.cell_center(cell)[2]) / mesh.spacing[2] + 0.5, 0.0, 1.0);
        }
        return fraction;
    };
    const auto curvature = GeometricVoF(grid).curvature(noisy_graph(grid, true));
    const auto center = grid.cell_index(3, 3, 4);
    const double unfiltered =
        4.0 * amplitude * (1.0 / (grid.spacing[0] * grid.spacing[0]) + 1.0 / (grid.spacing[1] * grid.spacing[1]));
    EXPECT_NEAR(curvature[center], (3.0 / 7.0) * unfiltered, 2.0e-10);
    EXPECT_LT(std::abs(curvature[center]), 0.5 * unfiltered);
    RecordProperty("alternating_mode_retention", curvature[center] / unfiltered);

    // A field invariant in y receives no transverse damping in either the
    // extruded 3D grid or the actual one-cell-y verification grid.
    const auto invariant = GeometricVoF(grid).curvature(noisy_graph(grid, false));
    auto two_dimensional = grid;
    two_dimensional.cells[1] = 1;
    const auto reduced = GeometricVoF(two_dimensional).curvature(noisy_graph(two_dimensional, false));
    const double invariant_expected = -4.0 * amplitude / (grid.spacing[0] * grid.spacing[0]);
    EXPECT_NEAR(invariant[center], invariant_expected, 2.0e-10);
    EXPECT_NEAR(reduced[two_dimensional.cell_index(3, 0, 4)], invariant_expected, 2.0e-10);
}

TEST(GeometricVoF, SurfaceFitHandlesTiltedQuadraticAndPlanarGraphs)
{
    const UniformGrid grid{{5, 5, 3}, {-0.025, -0.025, 0.0}, {0.01, 0.01, 0.08}};
    constexpr double slope_x = 0.25;
    constexpr double slope_y = -0.2;
    for (const double quadratic_scale : {0.0, 1.0})
    {
        const double a = 0.4 * quadratic_scale;
        const double b = 0.1 * quadratic_scale;
        const double c = 0.2 * quadratic_scale;
        std::vector<double> fraction(grid.cell_count());
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        {
            const auto p = grid.cell_center(cell);
            const double height = 0.12 + slope_x * p[0] + slope_y * p[1] +
                                  a * (p[0] * p[0] + grid.spacing[0] * grid.spacing[0] / 12.0) + b * p[0] * p[1] +
                                  c * (p[1] * p[1] + grid.spacing[1] * grid.spacing[1] / 12.0);
            fraction[cell] = std::clamp((height - p[2]) / grid.spacing[2] + 0.5, 0.0, 1.0);
        }
        const double expected =
            -(2.0 * a * (1.0 + slope_y * slope_y) + 2.0 * c * (1.0 + slope_x * slope_x) - 2.0 * b * slope_x * slope_y) /
            std::pow(1.0 + slope_x * slope_x + slope_y * slope_y, 1.5);
        const auto curvature = GeometricVoF(grid).curvature(fraction);
        EXPECT_NEAR(curvature[grid.cell_index(2, 2, 1)], expected, 2.0e-4);
    }
}

TEST(GeometricVoF, SurfaceCurveFitUsesTheTwoDimensionalContract)
{
    const UniformGrid grid{{9, 3, 1}, {-0.045, 0.0, 0.0}, {0.01, 0.02, 0.03}};
    std::vector<double> fraction(grid.cell_count());
    constexpr double a = 0.7;
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        const double height = 0.03 + a * (p[0] * p[0] + grid.spacing[0] * grid.spacing[0] / 12.0);
        fraction[cell] = std::clamp((height - p[1]) / grid.spacing[1] + 0.5, 0.0, 1.0);
    }
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    EXPECT_NEAR(curvature[grid.cell_index(4, 1, 0)], -2.0 * a, 2.0e-9);
}

TEST(GeometricVoF, SurfaceFitUsesNearestPeriodicImagesAcrossTheSeam)
{
    const UniformGrid grid{{32, 5, 3}, {0.0, -0.05, 0.0}, {0.02, 0.02, 0.02}, {true, false, false}};
    const double wave_number = 2.0 * std::numbers::pi / (grid.cells[0] * grid.spacing[0]);
    constexpr double amplitude = 0.003;
    const double half_wave_cell = 0.5 * wave_number * grid.spacing[0];
    const double average_factor = std::sin(half_wave_cell) / half_wave_cell;
    std::vector<double> fraction(grid.cell_count());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto p = grid.cell_center(cell);
        const double height = 0.03 + amplitude * average_factor * std::cos(wave_number * p[0]);
        fraction[cell] = std::clamp((height - p[2]) / grid.spacing[2] + 0.5, 0.0, 1.0);
    }
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    for (const std::size_t i : {std::size_t{0}, grid.cells[0] - 1})
    {
        const auto cell = grid.cell_index(i, 2, 1);
        const double phase = wave_number * grid.cell_center(cell)[0];
        const double slope = -amplitude * wave_number * std::sin(phase);
        const double expected =
            amplitude * wave_number * wave_number * std::cos(phase) / std::pow(1.0 + slope * slope, 1.5);
        EXPECT_NEAR(curvature[cell], expected, 0.01 * std::abs(expected));
    }
}

TEST(GeometricVoF, InsufficientSurfaceFitDataRemainsFiniteAndComplementary)
{
    const UniformGrid grid{{3, 3, 3}, {}, {0.01, 0.02, 0.03}};
    std::vector<double> fraction(grid.cell_count(), 0.0);
    fraction[grid.cell_index(1, 1, 1)] = 0.4;
    fraction[grid.cell_index(2, 1, 1)] = 0.2;
    const auto curvature = GeometricVoF(grid).curvature(fraction);
    for (auto& value : fraction)
        value = 1.0 - value;
    const auto complement = GeometricVoF(grid).curvature(fraction);
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        EXPECT_TRUE(std::isfinite(curvature[cell]));
        EXPECT_NEAR(curvature[cell], -complement[cell], 2.0e-12);
    }
}

TEST(GeometricVoF, UnsupportedVelocityAndInvalidStepsRejectBeforeMutation)
{
    const UniformGrid grid{{8, 8, 1}, {}, {0.1, 0.1, 1.0}, {true, true, false}};
    const GeometricVoF geometry(grid);
    const std::vector<double> fraction(grid.cell_count(), 0.4);
    auto velocity = face_constant(grid, {});
    EXPECT_THROW((void) geometry.advect(fraction, velocity, -1.0), std::invalid_argument);
    velocity[0].assign(grid.face_count(0), 1.0);
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1), std::runtime_error);
    velocity[0][grid.face_index(0, 0, 0, 0)] = 0.0;
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.001), std::invalid_argument);
    velocity = face_constant(grid, {});
    velocity[0][grid.face_index(0, 3, 3, 0)] = 0.1;
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.001), std::runtime_error);
    velocity = face_constant(grid, {});
    velocity[2][0] = 0.01;
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.001), std::invalid_argument);
}

TEST(GeometricVoFTest, RequestedStencilResultsMatchFullKernelIncludingBoundaryCells)
{
    UniformGrid grid;
    grid.cells = {8, 8, 8};
    grid.spacing = {0.1, 0.15, 0.2};
    for (const bool periodic : {false, true})
    {
        grid.periodic = {periodic, periodic, periodic};
        GeometricVoF geometry(grid);
        std::vector<double> fraction(grid.cell_count());
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            const auto p = grid.cell_center(c);
            fraction[c] =
                std::clamp(0.5 + 0.7 * std::sin(7.0 * p[0]) * std::cos(6.0 * p[1]) * std::sin(5.0 * p[2]), 0.0, 1.0);
        }
        const std::vector<std::size_t> requested{0, 7, 63, 511, 291, 291, 230};
        const auto all_planes = geometry.reconstruct(fraction), planes = geometry.reconstruct(fraction, requested);
        const auto all_curvature = geometry.curvature(fraction), curvature = geometry.curvature(fraction, requested);
        ASSERT_EQ(planes.size(), requested.size());
        ASSERT_EQ(curvature.size(), requested.size());
        for (std::size_t n = 0; n < requested.size(); ++n)
        {
            EXPECT_EQ(planes[n].normal, all_planes[requested[n]].normal);
            EXPECT_DOUBLE_EQ(planes[n].offset, all_planes[requested[n]].offset);
            EXPECT_DOUBLE_EQ(curvature[n], all_curvature[requested[n]]);
        }
        const std::vector<std::size_t> invalid{fraction.size()};
        EXPECT_THROW((void) geometry.reconstruct(fraction, invalid), std::out_of_range);
        EXPECT_THROW((void) geometry.curvature(fraction, invalid), std::out_of_range);
    }
}
} // namespace
