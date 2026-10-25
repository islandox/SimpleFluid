/** @file testPolyhedralReconstruction.cc
 * @brief Resolution requests, bounded PLIC fitting and retained finite inventories.
 */
#include "FVM/GeometricVoF.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/SweptGeometry.hh"
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
FinitePolyhedralPhaseSample uniform_sample(
    std::array<int, 3> index, double scale, double fraction, ArithmeticMode mode = ArithmeticMode::BoundedCertification)
{
    SimpleFluid::PLIC::GeometryPoint lower{}, upper{};
    for (unsigned a = 0; a < 3; ++a)
    {
        lower[a] = index[a] * scale;
        upper[a] = (index[a] + 1) * scale;
    }
    const double volume = scale * scale * scale;
    SimpleFluid::PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    return {SimpleFluid::PLIC::DynamicConvexGeometry::box(lower, upper, limits), fraction * volume,
        (1 - fraction) * volume};
}

TEST(PolyhedralReconstruction, OrdinaryOffsetFitRetainsPhysicalMinorityResidualAndInputAmounts)
{
    for (bool gas : {false, true})
    {
        const auto sample = uniform_sample({0, 0, 0}, 1, gas ? .25 : .75, ArithmeticMode::Ordinary);
        const auto fit = PolyhedralReconstruction::solve_offset(sample, {1, 0, 0});
        const auto phases = sample.geometry->phases(fit.gas);
        const auto measured = gas ? phases.gas : phases.carrier;
        EXPECT_EQ(measured.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(measured.absolute_error, 0);
        EXPECT_EQ(fit.finite_target_volume_difference.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(fit.finite_target_volume_difference.absolute_error, 0);
        EXPECT_EQ(fit.finite_target_volume_difference.value, measured.value - .25L);
        EXPECT_EQ(fit.target_volume_residual, static_cast<double>(measured.value - .25L));
        EXPECT_LE(std::abs(measured.value - .25L), .25L * PolyhedralReconstructionOptions{}.volume_tolerance);
        EXPECT_EQ(sample.gas_volume, gas ? .25 : .75);
        EXPECT_EQ(sample.carrier_volume, gas ? .75 : .25);
        PolyhedralReconstructionOptions capped;
        capped.maximum_clip_evaluations = 1;
        EXPECT_THROW((void) PolyhedralReconstruction::solve_offset(sample, {1, 0, 0}, capped), std::invalid_argument);
        auto invalid = sample;
        invalid.carrier_volume += .01;
        EXPECT_THROW(PolyhedralReconstruction::validate(invalid), std::invalid_argument);
        invalid = sample;
        invalid.gas_volume = -1;
        EXPECT_THROW(PolyhedralReconstruction::validate(invalid), std::invalid_argument);
    }
}

TEST(PolyhedralReconstruction, OrdinarySupportAndModelGatesRemainIndependentOfCertification)
{
    const auto dilute = uniform_sample({0, 0, 0}, 1, .01, ArithmeticMode::Ordinary);
    std::vector<FinitePolyhedralPhaseSample> insufficient;
    for (const auto index : neighbors())
        insufficient.push_back(uniform_sample(index, 1, .01, ArithmeticMode::Ordinary));
    EXPECT_EQ(PolyhedralReconstruction::assess_resolution(dilute, insufficient).reason,
        PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
    EXPECT_THROW((void) PolyhedralReconstruction::reconstruct(dilute, insufficient), PolyhedralResolutionRequired);

    const auto target = uniform_sample({0, 0, 0}, 1, .25, ArithmeticMode::Ordinary);
    std::vector<FinitePolyhedralPhaseSample> stencil;
    for (const auto index :
        std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}})
        stencil.push_back(uniform_sample(index, 1, index[0] < 0 ? 1 : index[0] > 0 ? 0 : .5, ArithmeticMode::Ordinary));
    ASSERT_TRUE(PolyhedralReconstruction::assess_resolution(target, stencil).admitted());
    PolyhedralReconstructionOptions permissive;
    permissive.maximum_model_residual = .5;
    const auto fit = PolyhedralReconstruction::reconstruct(target, stencil, permissive);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_GT(fit.stencil_volume_residual, .1);
    EXPECT_LE(std::abs(fit.target_volume_residual), .25 * permissive.volume_tolerance);
    auto strict = permissive;
    strict.maximum_model_residual = .1;
    try
    {
        (void) PolyhedralReconstruction::reconstruct(target, stencil, strict);
        FAIL() << "Ordinary reconstruction accepted an excessive physical plane-model residual.";
    }
    catch (const PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
    }
    auto mixed = stencil;
    mixed[0] = uniform_sample({-1, 0, 0}, 1, 1, ArithmeticMode::BoundedCertification);
    EXPECT_THROW((void) PolyhedralReconstruction::assess_resolution(target, mixed), std::invalid_argument);
    EXPECT_THROW((void) PolyhedralReconstruction::reconstruct(target, mixed), std::invalid_argument);
    EXPECT_EQ(target.gas_volume, .25);
    EXPECT_EQ(target.carrier_volume, .75);
}

FinitePolyhedralPhaseSample plane_sample(
    std::array<int, 3> index, SimpleFluid::PLIC::GeometryPoint normal, double offset, double scale = 1)
{
    SimpleFluid::PLIC::GeometryPoint lower{}, upper{};
    for (unsigned a = 0; a < 3; ++a)
    {
        lower[a] = index[a] * scale;
        upper[a] = (index[a] + 1) * scale;
    }
    const auto gas = GeometricVoF::box_plane_volume(lower, upper, normal, offset * scale);
    const double volume = scale * scale * scale;
    return {SimpleFluid::PLIC::DynamicConvexGeometry::box(lower, upper), gas, volume - gas};
}

TEST(PolyhedralReconstruction, DiluteFeatureRequestsRefinementBeforeClippingAndPreservesAmounts)
{
    auto target = uniform_sample({0, 0, 0}, 1, .01);
    std::vector<FinitePolyhedralPhaseSample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(uniform_sample(index, 1, .01));
    const auto before = target;
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto assessment = PolyhedralReconstruction::assess_resolution(target, stencil);
    EXPECT_EQ(assessment.reason, PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
    EXPECT_EQ(assessment.local_cell_span, 1);
    EXPECT_EQ(assessment.requested_cell_span, .5);
    EXPECT_THROW(PolyhedralReconstruction::reconstruct(target, stencil), PolyhedralResolutionRequired);
    EXPECT_EQ(target.gas_volume, before.gas_volume);
    EXPECT_EQ(target.carrier_volume, before.carrier_volume);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().constructions, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().rational_measures, 0u);
}
TEST(PolyhedralReconstruction, ResolutionAdmissionIsInvariantUnderUnitScaling)
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
TEST(PolyhedralReconstruction, ResolvedGrazingPlaneUsesNeighborSupport)
{
    const SimpleFluid::PLIC::GeometryPoint normal{1, 1, 1};
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
TEST(PolyhedralReconstruction, RefinementRechecksThePhysicalSupportOfAThinLayer)
{
    // Same physical layer .1 <= x <= .2, integrated independently on each
    // mesh. Splitting the old uniform average alone cannot recover its faces.
    const auto layer = [](std::array<int, 3> index, double h, double shift)
    {
        SimpleFluid::PLIC::GeometryPoint lower{shift + index[0] * h, index[1] * h, index[2] * h};
        SimpleFluid::PLIC::GeometryPoint upper{shift + (index[0] + 1) * h, (index[1] + 1) * h, (index[2] + 1) * h};
        const double width = std::max(0., std::min(upper[0], .2) - std::max(lower[0], .1));
        const double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        const double gas = width * (upper[1] - lower[1]) * (upper[2] - lower[2]);
        return FinitePolyhedralPhaseSample{
            SimpleFluid::PLIC::DynamicConvexGeometry::box(lower, upper), gas, volume - gas};
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
TEST(PolyhedralReconstruction, DeficientAndUnobservableStencilsRequestMoreSupport)
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
TEST(PolyhedralReconstruction, RationalSupportFitsComplementaryOrdinaryVolumes)
{
    const auto exact =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1})
            ->clip({{-10, 0, 0}, {}, -1})
            .inside;
    const auto numerical = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*exact);
    for (double direction : {-1., 1.})
        for (bool minority_gas : {false, true})
        {
            const FinitePolyhedralPhaseSample sample{
                numerical, minority_gas ? .225 : .675, minority_gas ? .675 : .225};
            const auto fit = PolyhedralReconstruction::solve_offset(sample, {direction, 0, 0});
            ASSERT_TRUE(fit.finite_geometry);
            const auto measured = numerical->phases(fit.gas);
            EXPECT_TRUE((minority_gas ? measured.gas : measured.carrier).relative_error_at_most(.225, 8e-15));
            const auto reference =
                SimpleFluid::PLIC::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(*exact)->phases(
                    fit.gas);
            EXPECT_TRUE((minority_gas ? reference.gas : reference.carrier)
                    .relative_error_at_most(
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(.225),
                        8e-15));
            EXPECT_EQ(sample.gas_volume, minority_gas ? .225 : .675);
            EXPECT_EQ(sample.carrier_volume, minority_gas ? .675 : .225);
        }
}
TEST(PolyhedralReconstruction, PureEndpointsAndRejectedArithmeticDoNotChangePolicyOrAmounts)
{
    const auto geometry = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1});
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

TEST(PolyhedralReconstruction, FarMinoritySupportRetainsOffsetPrecisionForBothLabelsAndNormals)
{
    const auto geometry = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1});
    for (double direction : {-1., 1.})
        for (bool gas : {false, true})
        {
            const FinitePolyhedralPhaseSample sample{
                geometry, gas ? 1e-6 : 1 - 1e-6, gas ? 1 - 1e-6 : 1e-6};
            const auto fitted = PolyhedralReconstruction::solve_offset(sample, {direction, 0, 0});
            const auto phases = geometry->phases(fitted.gas);
            EXPECT_TRUE((gas ? phases.gas : phases.carrier).relative_error_at_most(1e-6L, 8e-15));
            EXPECT_LE(std::abs(fitted.finite_target_volume_difference.value) +
                    fitted.finite_target_volume_difference.absolute_error,
                static_cast<long double>(1e-6) * 8e-15);
        }
}
TEST(PolyhedralReconstruction, CoarseFinePassageMinorityFitDoesNotStallBetweenAdjacentOffsets)
{
    // Physical sample captured from the moving coarse/fine case. The old
    // cell-origin offset bracket exhausted adjacent binary64 offsets while its
    // residual exceeded the target; the evaluation radius was already small.
    const long double skew = static_cast<long double>(.6) - .5L;
    const std::array<std::array<long double, 3>, 8> normalized{{
        {0, 0, 0}, {-skew, -.5L, 0}, {.25L - skew, -.5L, 0}, {.25L, 0, 0},
        {0, 0, .5L}, {-skew, -.5L, .5L}, {.25L - skew, -.5L, .5L}, {.25L, 0, .5L}}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 8> local;
    for (std::size_t p = 0; p < local.size(); ++p)
        for (unsigned axis = 0; axis < 3; ++axis)
            local[p].value[axis] = normalized[p][axis] * .5L;
    const auto geometry = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.6, .5, .5}, local, {{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}});
    const FinitePolyhedralPhaseSample sample{geometry, 6.3984438984441292e-8, .0078124360155610159};
    const auto fitted = PolyhedralReconstruction::solve_offset(
        sample, {.96128421995518143, .27109170276597788, .049415960545162486});
    const auto actual = geometry->phases(fitted.gas);
    EXPECT_TRUE(actual.gas.relative_error_at_most(sample.gas_volume, 8e-15));
    EXPECT_LE(std::abs(fitted.finite_target_volume_difference.value) +
            fitted.finite_target_volume_difference.absolute_error,
        static_cast<long double>(sample.gas_volume) * 8e-15);
    EXPECT_EQ(sample.carrier_volume, .0078124360155610159);
}

} // namespace
