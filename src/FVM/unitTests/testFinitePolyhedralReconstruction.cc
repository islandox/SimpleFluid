/** @file testFinitePolyhedralReconstruction.cc
 * @brief Resolution requests, bounded PLIC fitting and retained finite inventories.
 */
#include "FVM/GeometricVoF.hh"
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactSweptGeometry.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes;

std::vector<std::array<int, 3>> neighbors()
{
    return {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        {-1, -1, -1}, {1, 1, 1}, {-1, 1, 1}, {1, -1, -1}};
}
FinitePolyhedralPhaseSample uniform_sample(std::array<int, 3> index, double scale, double fraction)
{
    GeometryPoint lower{}, upper{};
    for (unsigned a = 0; a < 3; ++a)
    {
        lower[a] = index[a] * scale;
        upper[a] = (index[a] + 1) * scale;
    }
    const double volume = scale * scale * scale;
    return {BoundedConvexGeometry::box(lower, upper), fraction * volume, (1 - fraction) * volume};
}
FinitePolyhedralPhaseSample plane_sample(
    std::array<int, 3> index, GeometryPoint normal, double offset, double scale = 1)
{
    GeometryPoint lower{}, upper{};
    for (unsigned a = 0; a < 3; ++a)
    {
        lower[a] = index[a] * scale;
        upper[a] = (index[a] + 1) * scale;
    }
    const auto gas = GeometricVoF::box_plane_volume(lower, upper, normal, offset * scale);
    const double volume = scale * scale * scale;
    return {BoundedConvexGeometry::box(lower, upper), gas, volume - gas};
}

TEST(FinitePolyhedralReconstruction, DiluteFeatureRequestsRefinementBeforeClippingAndPreservesAmounts)
{
    auto target = uniform_sample({0, 0, 0}, 1, .01);
    std::vector<FinitePolyhedralPhaseSample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(uniform_sample(index, 1, .01));
    const auto before = target;
    reset_bounded_geometry_statistics();
    const auto assessment = PolyhedralReconstruction::assess_resolution(target, stencil);
    EXPECT_EQ(assessment.reason, PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
    EXPECT_EQ(assessment.local_cell_span, 1);
    EXPECT_EQ(assessment.requested_cell_span, .5);
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil), PolyhedralResolutionRequired);
    EXPECT_EQ(target.gas_volume, before.gas_volume);
    EXPECT_EQ(target.carrier_volume, before.carrier_volume);
    EXPECT_EQ(bounded_geometry_statistics().constructions, 0u);
    EXPECT_EQ(bounded_geometry_statistics().rational_measures, 0u);
}
TEST(FinitePolyhedralReconstruction, ResolutionAdmissionIsInvariantUnderUnitScaling)
{
    for (const double scale : {1e-3, 1., 1e3})
    {
        const auto target = uniform_sample({0, 0, 0}, scale, .01);
        std::vector<FinitePolyhedralPhaseSample> stencil;
        for (const auto index : neighbors())
            stencil.push_back(uniform_sample(index, scale, .01));
        const auto assessment = PolyhedralReconstruction::assess_resolution(target, stencil);
        EXPECT_EQ(assessment.reason, PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
        EXPECT_NEAR(assessment.requested_cell_span / scale, .5, 2e-15);
        EXPECT_NEAR(assessment.gas_support_fraction, .01, 2e-17);
    }
}
TEST(FinitePolyhedralReconstruction, ResolvedGrazingPlaneUsesNeighborSupport)
{
    const GeometryPoint normal{1, 1, 1};
    const auto target = plane_sample({0, 0, 0}, normal, .05);
    ASSERT_LT(target.gas_volume, 1e-4);
    std::vector<FinitePolyhedralPhaseSample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(plane_sample(index, normal, .05));
    const auto assessment = PolyhedralReconstruction::assess_resolution(target, stencil);
    EXPECT_TRUE(assessment.admitted());
    EXPECT_GE(assessment.gas_support_fraction, .5);
    EXPECT_GE(assessment.carrier_support_fraction, .5);
    const auto fit = PolyhedralReconstruction::reconstruct(target, stencil);
    EXPECT_TRUE(fit.finite_geometry);
    EXPECT_TRUE(fit.resolution.admitted());
    for (auto n : fit.gas.normal)
        EXPECT_NEAR(n, 1 / std::sqrt(3.), 2e-7);
    EXPECT_TRUE(target.geometry->phases(fit.gas).gas.relative_error_at_most(target.gas_volume, 8e-15));
}
TEST(FinitePolyhedralReconstruction, RefinementRechecksThePhysicalSupportOfAThinLayer)
{
    // Same physical layer .1 <= x <= .2, integrated independently on each
    // mesh. Splitting the old uniform average alone cannot recover its faces.
    const auto layer = [](std::array<int, 3> index, double h, double shift)
    {
        GeometryPoint lower{shift + index[0] * h, index[1] * h, index[2] * h};
        GeometryPoint upper{shift + (index[0] + 1) * h, (index[1] + 1) * h, (index[2] + 1) * h};
        const double width = std::max(0., std::min(upper[0], .2) - std::max(lower[0], .1));
        const double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        const double gas = width * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        return FinitePolyhedralPhaseSample{BoundedConvexGeometry::box(lower, upper), gas, volume - gas};
    };
    const auto coarse = layer({0, 0, 0}, 1, 0);
    std::vector<FinitePolyhedralPhaseSample> coarse_stencil;
    for (const auto index : neighbors())
        coarse_stencil.push_back(layer(index, 1, 0));
    EXPECT_FALSE(PolyhedralReconstruction::assess_resolution(coarse, coarse_stencil).admitted());

    const auto fine = layer({1, 0, 0}, .05, .025);
    std::vector<FinitePolyhedralPhaseSample> fine_stencil;
    for (auto index : neighbors())
    {
        ++index[0];
        fine_stencil.push_back(layer(index, .05, .025));
    }
    EXPECT_TRUE(PolyhedralReconstruction::assess_resolution(fine, fine_stencil).admitted());
    const auto fit = PolyhedralReconstruction::reconstruct(fine, fine_stencil);
    EXPECT_TRUE(fit.resolution.admitted());
    EXPECT_NEAR(fit.gas.normal[0], -1, 2e-7);
}
TEST(FinitePolyhedralReconstruction, DeficientAndUnobservableStencilsRequestMoreSupport)
{
    const auto target = uniform_sample({0, 0, 0}, 1, .5);
    std::vector<FinitePolyhedralPhaseSample> deficient;
    for (int i : {-2, -1, 1, 2})
        deficient.push_back(uniform_sample({i, 0, 0}, 1, i < 0 ? 1 : 0));
    EXPECT_EQ(PolyhedralReconstruction::assess_resolution(target, deficient).reason,
        PolyhedralResolutionAssessment::Reason::InsufficientStencil);
    std::vector<FinitePolyhedralPhaseSample> constant;
    for (const auto index : neighbors())
        constant.push_back(uniform_sample(index, 1, .5));
    EXPECT_EQ(PolyhedralReconstruction::assess_resolution(target, constant).reason,
        PolyhedralResolutionAssessment::Reason::InsufficientStencil);
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, deficient), PolyhedralResolutionRequired);
}
TEST(FinitePolyhedralReconstruction, RationalSupportFitsComplementaryOrdinaryVolumes)
{
    const auto exact = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})->clip({{-10, 0, 0}, {}, -1}).inside;
    const auto numerical = BoundedConvexGeometry::from_geometry(*exact);
    for (double direction : {-1., 1.})
        for (bool minority_gas : {false, true})
        {
            const FinitePolyhedralPhaseSample sample{
                numerical, minority_gas ? .225 : .675, minority_gas ? .675 : .225};
            const auto fit = PolyhedralReconstruction::solve_offset(sample, {direction, 0, 0});
            ASSERT_TRUE(fit.finite_geometry);
            const auto measured = numerical->phases(fit.gas);
            EXPECT_TRUE((minority_gas ? measured.gas : measured.carrier).relative_error_at_most(.225, 8e-15));
            const auto reference = ExactConvexFragment::from_geometry(*exact)->phases(fit.gas);
            EXPECT_TRUE((minority_gas ? reference.gas : reference.carrier)
                            .relative_error_at_most(ExactGeometryMeasure::from_value(.225), 8e-15));
            EXPECT_EQ(sample.gas_volume, minority_gas ? .225 : .675);
            EXPECT_EQ(sample.carrier_volume, minority_gas ? .675 : .225);
        }
}
TEST(FinitePolyhedralReconstruction, PureEndpointsAndRejectedArithmeticDoNotChangePolicyOrAmounts)
{
    const auto geometry = BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1});
    for (double direction : {-1., 1.})
        for (bool gas : {false, true})
        {
            const auto fit = PolyhedralReconstruction::solve_offset(
                {geometry, gas ? 1. : 0., gas ? 0. : 1.}, {direction, 0, 0});
            EXPECT_TRUE(gas ? fit.pure_gas : fit.pure_carrier);
            const auto phases = geometry->phases(fit.gas);
            EXPECT_EQ((gas ? phases.carrier : phases.gas).upper(), 0);
        }
    FinitePolyhedralPhaseSample sample{geometry, .3, .7};
    PolyhedralReconstructionOptions options;
    options.maximum_clip_evaluations = 1;
    EXPECT_THROW(PolyhedralReconstruction::solve_offset(sample, {1, 0, 0}, options), std::invalid_argument);
    EXPECT_EQ(sample.gas_volume, .3);
    EXPECT_EQ(sample.carrier_volume, .7);
    sample.gas_volume = -.1;
    EXPECT_THROW(PolyhedralReconstruction::solve_offset(sample, {1, 0, 0}), std::invalid_argument);
}
} // namespace
