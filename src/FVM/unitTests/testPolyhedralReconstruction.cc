/** @file testPolyhedralReconstruction.cc
 * @brief Native convex PLIC fitting, minority bounds and invalid-input checks.
 */
#include "FVM/GeometricVoF.hh"
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactSweptGeometry.hh"
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes;
GeometryPoint mapped(GeometryPoint p)
{
    return {p[0] + .25 * p[1] + .125 * p[2], p[1] + .25 * p[2], p[2]};
}
ConvexGeometryPtr skew_cell(GeometryPoint lower, GeometryPoint upper)
{
    std::vector<GeometryPoint> points;
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
                points.push_back(mapped({i ? upper[0] : lower[0], j ? upper[1] : lower[1], k ? upper[2] : lower[2]}));
    return ConvexCellGeometry::make(
        points, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
}
TEST(PolyhedralReconstruction, InfersObliquePlanarNormalFromIndependentSkewNonuniformVolumes)
{
    const std::array<double, 4> edges{-1, -.25, .5, 1.5};
    const GeometryPoint reference_normal{.75, -.5, .625};
    const double offset = .125;
    const auto sample = [&](int i, int j, int k)
    {
        GeometryPoint lower{edges[i], edges[j], edges[k]}, upper{edges[i + 1], edges[j + 1], edges[k + 1]};
        // Analytic box inclusion/exclusion in reference coordinates, independent
        // of the actual skew-polyhedron clipping and nonlinear fit under test.
        const auto gas = GeometricVoF::box_plane_volume(lower, upper, reference_normal, offset);
        const double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        return PolyhedralPhaseSample{skew_cell(lower, upper), gas, volume - gas};
    };
    auto target = sample(1, 1, 1);
    std::vector<PolyhedralPhaseSample> stencil;
    for (auto ijk : std::vector<std::array<int, 3>>{{0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},
             {0, 0, 0}, {2, 2, 2}, {0, 2, 2}, {2, 0, 0}})
        stencil.push_back(sample(ijk[0], ijk[1], ijk[2]));
    auto result = PolyhedralReconstruction::reconstruct(target, stencil);
    GeometryPoint expected{.75, -.5 - .25 * .75, .625 - .125 * .75 - .25 * (-.5 - .25 * .75)};
    const double norm = std::hypot(expected[0], expected[1], expected[2]);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(result.gas.normal[a], expected[a] / norm, 2e-7);
    EXPECT_LT(result.stencil_volume_residual, 2e-9);
    EXPECT_NEAR(target.geometry->clip(result.gas).inside->moments().volume.as_double(), target.gas_volume, 2e-12);
}
TEST(PolyhedralReconstruction, PreparedGeometryRemainsPrivateAfterRejectedReconstruction)
{
    const std::array<double, 4> edges{-1, -.25, .5, 1.5};
    const GeometryPoint reference_normal{.75, -.5, .625};
    const double offset = .125;
    const auto sample = [&](int i, int j, int k)
    {
        GeometryPoint lower{edges[i], edges[j], edges[k]}, upper{edges[i + 1], edges[j + 1], edges[k + 1]};
        // Analytic box inclusion/exclusion in reference coordinates, independent
        // of the actual skew-polyhedron clipping and nonlinear fit under test.
        const auto gas = GeometricVoF::box_plane_volume(lower, upper, reference_normal, offset);
        const double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        return PolyhedralPhaseSample{skew_cell(lower, upper), gas, volume - gas};
    };
    auto target = sample(1, 1, 1);
    std::vector<PolyhedralPhaseSample> stencil;
    for (auto ijk : std::vector<std::array<int, 3>>{{0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},
             {0, 0, 0}, {2, 2, 2}, {0, 2, 2}, {2, 0, 0}})
        stencil.push_back(sample(ijk[0], ijk[1], ijk[2]));
    auto result = PolyhedralReconstruction::reconstruct(target, stencil);
    GeometryPoint expected{.75, -.5 - .25 * .75, .625 - .125 * .75 - .25 * (-.5 - .25 * .75)};
    const double norm = std::hypot(expected[0], expected[1], expected[2]);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(result.gas.normal[a], expected[a] / norm, 2e-7);
    EXPECT_LT(result.stencil_volume_residual, 2e-9);
    EXPECT_NEAR(target.geometry->clip(result.gas).inside->moments().volume.as_double(), target.gas_volume, 2e-12);

    auto rejected = target;
    rejected.gas_volume = -1;
    EXPECT_THROW((void) PolyhedralReconstruction::reconstruct(rejected, stencil), std::invalid_argument);
    const auto repeated = PolyhedralReconstruction::reconstruct(target, stencil);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_EQ(
            std::bit_cast<std::uint64_t>(result.gas.normal[a]), std::bit_cast<std::uint64_t>(repeated.gas.normal[a]));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(result.gas.offset), std::bit_cast<std::uint64_t>(repeated.gas.offset));
}

TEST(PolyhedralReconstruction, IndependentMicroscopicPhasesAreNeverNormalizedOrDiscarded)
{
    for (const double trace : {1e-24, 1e-310})
        for (bool minority_gas : {true, false})
        {
            PolyhedralPhaseSample sample{
                ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1}), minority_gas ? trace : 1., minority_gas ? 1. : trace};
            const auto original = sample;
            auto plane = PolyhedralReconstruction::solve_offset(sample, {1, 0, 0});
            EXPECT_FALSE(plane.pure_gas);
            EXPECT_FALSE(plane.pure_carrier);
            EXPECT_EQ(sample.gas_volume, original.gas_volume);
            EXPECT_EQ(sample.carrier_volume, original.carrier_volume);
            const auto fragment = ExactConvexFragment::from_geometry(*sample.geometry);
            const auto phases = fragment->phases(plane.gas);
            const auto minority = minority_gas ? phases.gas : phases.carrier;
            const auto requested = ExactGeometryMeasure::from_value(trace);
            EXPECT_TRUE(minority.relative_error_at_most(requested, 8e-15));
            if (trace < std::numeric_limits<double>::min())
                EXPECT_EQ(minority.certified().as_double(), trace);
            else
                EXPECT_NEAR(minority.certified().as_double(), trace, 1e-37);
        }
}
TEST(PolyhedralReconstruction, RationalSupportRetainsSignedMinorityOffsets)
{
    const auto box = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    // The exact lower support is 1/10, while its displayed coordinate is
    // slightly larger. The gas plane for a 1e-18 layer must have negative offset.
    const auto cell = box->clip({{-10, 0, 0}, {}, -1}).inside;
    const auto fragment = ExactConvexFragment::from_geometry(*cell);
    for (const double direction : {1., -1.})
        for (const bool minority_gas : {true, false})
        {
            SCOPED_TRACE(::testing::Message() << direction << " " << minority_gas);
            PolyhedralPhaseSample sample{cell, minority_gas ? 1e-18 : .9, minority_gas ? .9 : 1e-18};
            const auto original = sample;
            const auto plane = PolyhedralReconstruction::solve_offset(sample, {direction, 0, 0});
            const auto phases = fragment->phases(plane.gas);
            const auto minority = minority_gas ? phases.gas : phases.carrier;
            EXPECT_TRUE(minority.relative_error_at_most(ExactGeometryMeasure::from_value(1e-18), 8e-15));
            EXPECT_GT(minority.sign(), 0);
            EXPECT_EQ((phases.gas + phases.carrier).compare(fragment->volume()), 0);
            EXPECT_EQ(sample.gas_volume, original.gas_volume);
            EXPECT_EQ(sample.carrier_volume, original.carrier_volume);
            EXPECT_FALSE(plane.pure_gas);
            EXPECT_FALSE(plane.pure_carrier);
            if (direction == 1 && minority_gas)
                EXPECT_NEAR(plane.gas.offset, -4.5511151231257829e-18, 1e-32);
        }
    for (const double direction : {1., -1.})
        for (const bool pure_gas : {true, false})
        {
            const auto plane = PolyhedralReconstruction::solve_offset(
                {cell, pure_gas ? .9 : 0., pure_gas ? 0. : .9}, {direction, 0, 0});
            const auto phases = fragment->phases(plane.gas);
            EXPECT_EQ((pure_gas ? phases.carrier : phases.gas).sign(), 0);
            EXPECT_EQ((pure_gas ? phases.gas : phases.carrier).compare(fragment->volume()), 0);
        }
}
TEST(PolyhedralReconstruction, CurvedVolumeModelSeparatesAngularConvergenceFromPlanarDefect)
{
    constexpr double width = .25, linear_x = .125, linear_y = -.0625, curvature = .03125;
    const auto sample = [&](int i, int j, int k)
    {
        GeometryPoint lower{double(i) - width / 2, double(j) - width / 2, double(k)};
        GeometryPoint upper{lower[0] + width, lower[1] + width, lower[2] + 1};
        const double volume = width * width;
        // Independent exact polynomial integral of z<h(x,y) over each XY
        // rectangle; the graph stays strictly within the middle Z layer.
        const double mean = .5 + linear_x * i + linear_y * j + curvature * (i * i + j * j + width * width / 6);
        const double gas = k < 0 ? volume : k > 0 ? 0 : volume * mean;
        return PolyhedralPhaseSample{ConvexCellGeometry::box(lower, upper), gas, volume - gas};
    };
    const auto target = sample(0, 0, 0);
    std::vector<PolyhedralPhaseSample> stencil;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            if (i || j)
                stencil.push_back(sample(i, j, 0));
    stencil.push_back(sample(0, 0, -1));
    stencil.push_back(sample(0, 0, 1));
    const auto fit = PolyhedralReconstruction::reconstruct(target, stencil);
    const double norm = std::hypot(linear_x, linear_y, 1);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_GT(fit.stencil_volume_residual, 1e-3);
    EXPECT_NEAR(fit.gas.normal[0], -linear_x / norm, 2e-7);
    EXPECT_NEAR(fit.gas.normal[1], -linear_y / norm, 2e-7);
    EXPECT_NEAR(fit.gas.normal[2], 1 / norm, 2e-7);
    PolyhedralReconstructionOptions strict;
    strict.maximum_model_residual = 1e-3;
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil, strict), std::invalid_argument);
    strict = {};
    strict.maximum_fit_iterations = 1;
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil, strict), std::invalid_argument);
}
TEST(PolyhedralReconstruction, RejectsDeficientPhysicalStencilAndInvalidOccupancy)
{
    PolyhedralPhaseSample target{ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1}), .5, .5};
    std::vector<PolyhedralPhaseSample> stencil;
    for (int i : {-2, -1, 1, 2})
        stencil.push_back({ConvexCellGeometry::box({double(i), 0, 0}, {double(i + 1), 1, 1}), .5, .5});
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil), std::invalid_argument);
    PolyhedralReconstructionOptions limited;
    limited.maximum_clip_evaluations = 1;
    target.gas_volume = .3;
    target.carrier_volume = .7;
    EXPECT_THROW(PolyhedralReconstruction::solve_offset(target, {1, 0, 0}, limited), std::invalid_argument);
    limited = {};
    limited.maximum_stencil_cells = 2;
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil, limited), std::invalid_argument);
    target.carrier_volume = .4;
    EXPECT_THROW(PolyhedralReconstruction::solve_offset(target, {1, 0, 0}), std::invalid_argument);
}
} // namespace
