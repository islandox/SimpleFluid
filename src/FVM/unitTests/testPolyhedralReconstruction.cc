/** @file testPolyhedralReconstruction.cc
 * @brief Resolution requests, bounded PLIC fitting and retained finite inventories.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/GeometricVoF.hh"
#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include "geometry/PLIC/details/AngularFit.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes;
using BoundedSample = PLIC::PolyhedralPhaseSample<ArithmeticMode::BoundedCertification>;
using OrdinarySample = PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>;
using BoundedGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>;
static_assert(!std::is_convertible_v<BoundedSample, OrdinarySample>);

std::vector<std::array<int, 3>> neighbors()
{
    return {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, -1, -1}, {1, 1, 1}, {-1, 1, 1},
        {1, -1, -1}};
}
template<ArithmeticMode Mode = ArithmeticMode::BoundedCertification>
PLIC::PolyhedralPhaseSample<Mode> uniform_sample(std::array<int, 3> index, double scale, double fraction)
{
    SimpleFluid::PLIC::GeometryPoint lower{}, upper{};
    for (unsigned a = 0; a < 3; ++a)
    {
        lower[a] = index[a] * scale;
        upper[a] = (index[a] + 1) * scale;
    }
    const double volume = scale * scale * scale;
    return {PLIC::ConvexCellGeometry<Mode>::box(lower, upper), fraction * volume, (1 - fraction) * volume};
}

static_assert(std::is_same_v<decltype(PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact>{}.geometry),
    PLIC::ConvexGeometryPtr<ArithmeticMode::Exact>>);
static_assert(std::is_same_v<decltype(PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>{}.geometry),
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>>);
static_assert(!std::is_convertible_v<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>,
    PLIC::ConvexGeometryPtr<ArithmeticMode::Exact>>);

template<ArithmeticMode Mode>
concept HasResolutionPreflight = requires(
    const PLIC::PolyhedralPhaseSample<Mode>& sample, std::span<const PLIC::PolyhedralPhaseSample<Mode>> stencil) {
    PLIC::PolyhedralReconstruction<Mode>::assess_resolution(sample, stencil);
};
static_assert(HasResolutionPreflight<ArithmeticMode::Ordinary>);
static_assert(HasResolutionPreflight<ArithmeticMode::BoundedCertification>);
static_assert(!HasResolutionPreflight<ArithmeticMode::Exact>);

template<ArithmeticMode Mode> void typed_offset_and_reconstruction()
{
    using Sample = PLIC::PolyhedralPhaseSample<Mode>;
    using Geometry = PLIC::ConvexCellGeometry<Mode>;
    const Sample target{Geometry::box({0, 0, 0}, {1, 1, 1}), .25, .75};
    PLIC::reset_geometry_statistics();
    GeneralVoF::reset_geometry_statistics();
    const auto offset = PLIC::PolyhedralReconstruction<Mode>::solve_offset(target, {1, 0, 0});
    const auto phases = target.geometry->phases(offset.gas);
    EXPECT_LE(std::abs(phases.gas.value - .25L), .25L * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
    EXPECT_EQ(offset.finite_target_volume_difference.arithmetic_mode, Mode);
    EXPECT_EQ(target.gas_volume, .25);
    EXPECT_EQ(target.carrier_volume, .75);

    std::vector<Sample> stencil;
    for (const auto index : neighbors())
    {
        const PLIC::GeometryPoint lower{double(index[0]), double(index[1]), double(index[2])};
        const PLIC::GeometryPoint upper{double(index[0] + 1), double(index[1] + 1), double(index[2] + 1)};
        const double gas = index[0] < 0 ? 1 : index[0] > 0 ? 0 : .25;
        stencil.push_back({Geometry::box(lower, upper), gas, 1 - gas});
    }
    const auto fitted = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil);
    EXPECT_TRUE(fitted.fit_converged);
    EXPECT_TRUE(fitted.resolution.admitted());
    EXPECT_LE(std::abs(fitted.target_volume_residual), .25 * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
    EXPECT_LE(fitted.stencil_volume_residual, PLIC::PolyhedralReconstructionOptions{}.fit_tolerance);
    EXPECT_NEAR(fitted.gas.normal[0], 1, 1e-8);
    EXPECT_NEAR(fitted.gas.normal[1], 0, 1e-8);
    EXPECT_NEAR(fitted.gas.normal[2], 0, 1e-8);
    auto invalid = target;
    invalid.carrier_volume = .8;
    EXPECT_THROW(PLIC::PolyhedralReconstruction<Mode>::validate(invalid), std::invalid_argument);
    // Reconstruction is compiled in FVM; its counters may be distinct from
    // the executable's geometry image, so inspect both actual domains.
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
    const auto reconstructed = GeneralVoF::geometry_statistics().bounded;
    EXPECT_GT(reconstructed.volume_evaluations, 0u);
    EXPECT_EQ(reconstructed.runtime_dispatches, 0u);
}

TEST(PolyhedralReconstruction, TypedModeEntriesRetainPlanarFitAndIndependentInventoryGates)
{
    typed_offset_and_reconstruction<ArithmeticMode::Ordinary>();
    typed_offset_and_reconstruction<ArithmeticMode::BoundedCertification>();
}

TEST(PolyhedralReconstruction, OrdinaryOffsetFitRetainsPhysicalMinorityResidualAndInputAmounts)
{
    for (bool gas : {false, true})
    {
        const auto sample = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, gas ? .25 : .75);
        const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(sample, {1, 0, 0});
        const auto phases = sample.geometry->phases(fit.gas);
        const auto measured = gas ? phases.gas : phases.carrier;
        EXPECT_EQ(measured.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(fit.finite_target_volume_difference.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(fit.finite_target_volume_difference.absolute_error, 0);
        EXPECT_EQ(fit.finite_target_volume_difference.value, measured.value - .25L);
        EXPECT_EQ(fit.target_volume_residual, static_cast<double>(measured.value - .25L));
        EXPECT_LE(std::abs(measured.value - .25L), .25L * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
        EXPECT_EQ(sample.gas_volume, gas ? .25 : .75);
        EXPECT_EQ(sample.carrier_volume, gas ? .75 : .25);
        PLIC::PolyhedralReconstructionOptions capped;
        capped.maximum_clip_evaluations = 1;
        EXPECT_THROW(
            (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(sample, {1, 0, 0}, capped),
            std::invalid_argument);
        auto invalid = sample;
        invalid.carrier_volume += .01;
        EXPECT_THROW(
            PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::validate(invalid), std::invalid_argument);
        invalid = sample;
        invalid.gas_volume = -1;
        EXPECT_THROW(
            PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::validate(invalid), std::invalid_argument);
    }
}

TEST(PolyhedralReconstruction, OrdinarySupportAndModelGatesRemainIndependentOfCertification)
{
    const auto dilute = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .01);
    std::vector<OrdinarySample> insufficient;
    for (const auto index : neighbors())
        insufficient.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, .01));
    EXPECT_EQ(PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(dilute, insufficient).reason,
        PLIC::PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
    EXPECT_THROW((void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(dilute, insufficient),
        PLIC::PolyhedralResolutionRequired);

    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .25);
    std::vector<OrdinarySample> stencil;
    for (const auto index :
        std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}})
        stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, index[0] < 0 ? 1 : index[0] > 0 ? 0 : .5));
    ASSERT_TRUE(
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(target, stencil).admitted());
    PLIC::PolyhedralReconstructionOptions permissive;
    permissive.maximum_model_residual = .5;
    const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil, permissive);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_GT(fit.stencil_volume_residual, .1);
    EXPECT_LE(std::abs(fit.target_volume_residual), .25 * permissive.volume_tolerance);
    auto strict = permissive;
    strict.maximum_model_residual = .1;
    try
    {
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil, strict);
        FAIL() << "Ordinary reconstruction accepted an excessive physical plane-model residual.";
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
    }
    // Cross-mode samples cannot enter this typed stencil. The explicit erased
    // storage boundary must reject a foreign-mode shell as well.
    const auto foreign = uniform_sample<ArithmeticMode::BoundedCertification>({-1, 0, 0}, 1, 1);
    EXPECT_THROW(
        (void) PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(foreign.geometry->dynamic_geometry()),
        std::invalid_argument);
    EXPECT_EQ(target.gas_volume, .25);
    EXPECT_EQ(target.carrier_volume, .75);
}

BoundedSample plane_sample(
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
    return {BoundedGeometry::box(lower, upper), gas, volume - gas};
}

TEST(PolyhedralReconstruction, DiluteFeatureRequestsRefinementBeforeClippingAndPreservesAmounts)
{
    auto target = uniform_sample({0, 0, 0}, 1, .01);
    std::vector<BoundedSample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(uniform_sample(index, 1, .01));
    const auto before = target;
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, stencil);
    EXPECT_EQ(assessment.reason, PLIC::PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
    EXPECT_EQ(assessment.local_cell_span, 1);
    EXPECT_EQ(assessment.requested_cell_span, .5);
    EXPECT_THROW(
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::reconstruct(target, stencil),
        PLIC::PolyhedralResolutionRequired);
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
        std::vector<BoundedSample> stencil;
        for (const auto index : neighbors())
            stencil.push_back(uniform_sample(index, scale, .01));
        const auto assessment =
            PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, stencil);
        EXPECT_EQ(assessment.reason, PLIC::PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport);
        EXPECT_NEAR(assessment.requested_cell_span / scale, .5, 2e-15);
        EXPECT_NEAR(assessment.gas_support_fraction, .01, 2e-17);
    }
}
TEST(PolyhedralReconstruction, ResolvedGrazingPlaneUsesNeighborSupport)
{
    const SimpleFluid::PLIC::GeometryPoint normal{1, 1, 1};
    const auto target = plane_sample({0, 0, 0}, normal, .05);
    ASSERT_LT(target.gas_volume, 1e-4);
    std::vector<BoundedSample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(plane_sample(index, normal, .05));
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, stencil);
    EXPECT_TRUE(assessment.admitted());
    EXPECT_GE(assessment.gas_support_fraction, .5);
    EXPECT_GE(assessment.carrier_support_fraction, .5);
    const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::reconstruct(target, stencil);
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
        return BoundedSample{BoundedGeometry::box(lower, upper), gas, volume - gas};
    };
    const auto coarse = layer({0, 0, 0}, 1, 0);
    std::vector<BoundedSample> coarse_stencil;
    for (const auto index : neighbors())
        coarse_stencil.push_back(layer(index, 1, 0));
    EXPECT_FALSE(
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(coarse, coarse_stencil)
            .admitted());

    const auto fine = layer({1, 0, 0}, .05, .025);
    std::vector<BoundedSample> fine_stencil;
    for (auto index : neighbors())
    {
        ++index[0];
        fine_stencil.push_back(layer(index, .05, .025));
    }
    EXPECT_TRUE(
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(fine, fine_stencil)
            .admitted());
    const auto fit =
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::reconstruct(fine, fine_stencil);
    EXPECT_TRUE(fit.resolution.admitted());
    EXPECT_NEAR(fit.gas.normal[0], -1, 2e-7);
}
TEST(PolyhedralReconstruction, DeficientAndUnobservableStencilsRequestMoreSupport)
{
    const auto target = uniform_sample({0, 0, 0}, 1, .5);
    std::vector<BoundedSample> deficient;
    for (int i : {-2, -1, 1, 2})
        deficient.push_back(uniform_sample({i, 0, 0}, 1, i < 0 ? 1 : 0));
    EXPECT_EQ(PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, deficient)
                  .reason,
        PLIC::PolyhedralResolutionAssessment::Reason::InsufficientStencil);
    std::vector<BoundedSample> constant;
    for (const auto index : neighbors())
        constant.push_back(uniform_sample(index, 1, .5));
    EXPECT_EQ(PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, constant)
                  .reason,
        PLIC::PolyhedralResolutionAssessment::Reason::InsufficientStencil);
    EXPECT_THROW(
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::reconstruct(target, deficient),
        PLIC::PolyhedralResolutionRequired);
}
TEST(PolyhedralReconstruction, RationalSupportFitsComplementaryOrdinaryVolumes)
{
    const auto exact =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1})
            ->clip({{-10, 0, 0}, {}, -1})
            .inside;
    const auto numerical = BoundedGeometry::from_geometry(*exact);
    for (double direction : {-1., 1.})
        for (bool minority_gas : {false, true})
        {
            const BoundedSample sample{numerical, minority_gas ? .225 : .675, minority_gas ? .675 : .225};
            const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::solve_offset(
                sample, {direction, 0, 0});
            ASSERT_TRUE(fit.finite_geometry);
            const auto measured = numerical->phases(fit.gas);
            EXPECT_TRUE((minority_gas ? measured.gas : measured.carrier).relative_error_at_most(.225, 8e-15));
            const auto reference =
                SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(*exact)->phases(
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
    const auto geometry = BoundedGeometry::box({0, 0, 0}, {1, 1, 1});
    for (double direction : {-1., 1.})
        for (bool gas : {false, true})
        {
            const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::solve_offset(
                {geometry, gas ? 1. : 0., gas ? 0. : 1.}, {direction, 0, 0});
            EXPECT_TRUE(gas ? fit.pure_gas : fit.pure_carrier);
            const auto phases = geometry->phases(fit.gas);
            EXPECT_EQ((gas ? phases.carrier : phases.gas).upper(), 0);
        }
    BoundedSample sample{geometry, .3, .7};
    PLIC::PolyhedralReconstructionOptions options;
    options.maximum_clip_evaluations = 1;
    EXPECT_THROW((void) PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::solve_offset(
                     sample, {1, 0, 0}, options),
        std::invalid_argument);
    EXPECT_EQ(sample.gas_volume, .3);
    EXPECT_EQ(sample.carrier_volume, .7);
    sample.gas_volume = -.1;
    EXPECT_THROW(
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::solve_offset(sample, {1, 0, 0}),
        std::invalid_argument);
}

TEST(PolyhedralReconstruction, FarMinoritySupportRetainsOffsetPrecisionForBothLabelsAndNormals)
{
    const auto geometry = BoundedGeometry::box({0, 0, 0}, {1, 1, 1});
    for (double direction : {-1., 1.})
        for (bool gas : {false, true})
        {
            const BoundedSample sample{geometry, gas ? 1e-6 : 1 - 1e-6, gas ? 1 - 1e-6 : 1e-6};
            const auto fitted = PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::solve_offset(
                sample, {direction, 0, 0});
            const auto phases = geometry->phases(fitted.gas);
            EXPECT_TRUE((gas ? phases.gas : phases.carrier).relative_error_at_most(1e-6L, 8e-15));
            EXPECT_LE(std::abs(fitted.finite_target_volume_difference.value) +
                          fitted.finite_target_volume_difference.absolute_error,
                static_cast<long double>(1e-6) * 8e-15);
        }
}
template<ArithmeticMode Mode> void coarse_fine_passage_minority_fit()
{
    // Physical sample captured from the moving coarse/fine case. The old
    // cell-origin offset bracket exhausted adjacent binary64 offsets while its
    // residual exceeded the target; the evaluation radius was already small.
    const long double skew = static_cast<long double>(.6) - .5L;
    const std::array<std::array<long double, 3>, 8> normalized{{{0, 0, 0}, {-skew, -.5L, 0}, {.25L - skew, -.5L, 0},
        {.25L, 0, 0}, {0, 0, .5L}, {-skew, -.5L, .5L}, {.25L - skew, -.5L, .5L}, {.25L, 0, .5L}}};
    std::array<PLIC::Point<Mode>, 8> local;
    for (std::size_t p = 0; p < local.size(); ++p)
        for (unsigned axis = 0; axis < 3; ++axis)
            local[p].value[axis] = normalized[p][axis] * .5L;
    const auto geometry = PLIC::ConvexCellGeometry<Mode>::make_local(
        {.6, .5, .5}, local, {{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}});
    const PLIC::PolyhedralPhaseSample<Mode> sample{geometry, 6.3984438984441292e-8, .0078124360155610159};
    const auto fitted = PLIC::PolyhedralReconstruction<Mode>::solve_offset(
        sample, {.96128421995518143, .27109170276597788, .049415960545162486});
    const auto actual = geometry->phases(fitted.gas);
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        EXPECT_TRUE(actual.gas.relative_error_at_most(sample.gas_volume, 8e-15));
    else
        EXPECT_LE(std::abs(actual.gas.value - sample.gas_volume), static_cast<long double>(sample.gas_volume) * 8e-15);
    EXPECT_LE(
        std::abs(fitted.finite_target_volume_difference.value) + fitted.finite_target_volume_difference.absolute_error,
        static_cast<long double>(sample.gas_volume) * 8e-15);
    EXPECT_EQ(sample.carrier_volume, .0078124360155610159);
    EXPECT_EQ(fitted.finite_target_volume_difference.arithmetic_mode, Mode);
}

TEST(PolyhedralReconstruction, CoarseFinePassageMinorityFitDoesNotStallBetweenAdjacentOffsets)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded certification requires its supported FP environment.";
    coarse_fine_passage_minority_fit<ArithmeticMode::BoundedCertification>();
}

TEST(PolyhedralReconstruction, OrdinaryCoarseFinePassageMinorityFitDoesNotStallBetweenAdjacentOffsets)
{
    coarse_fine_passage_minority_fit<ArithmeticMode::Ordinary>();
}

struct AngularEvaluation
{
    std::vector<double> residual;
    double objective = 0;
    long double objective_upper = 0;
    unsigned stamp = 0;
};
using AngularVector = PLIC::angular_detail::Vector;
AngularVector angular_unit(AngularVector value)
{
    const auto length = std::hypot(value[0], value[1], value[2]);
    for (auto& component : value)
        component /= length;
    return value;
}
AngularEvaluation angular_model(AngularVector normal, double target_y, double target_z, unsigned stamp)
{
    normal = angular_unit(normal);
    AngularEvaluation result{{normal[1] - target_y, normal[2] - target_z}, 0, 0, stamp};
    for (const auto value : result.residual)
        result.objective += value * value;
    result.objective_upper = result.objective;
    return result;
}

TEST(PolyhedralReconstruction, AngularOptionalUnresolvedTrialUsesSmallerStepWithoutChangingCurrent)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal{1, 0, 0};
    auto current = angular_model(normal, .2, 0, 0);
    const auto old_normal = normal;
    const auto old_residual = current.residual;
    unsigned evaluations = 0;
    const auto evaluate = [&](AngularVector trial)
    {
        ++evaluations; // The caller owns and charges every attempted evaluation.
        EXPECT_EQ(normal, old_normal);
        EXPECT_EQ(current.residual, old_residual);
        EXPECT_EQ(current.stamp, 0u);
        if (evaluations == 5)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "unresolved trial");
        return angular_model(trial, .2, 0, evaluations);
    };
    angular::Qualification qualification;
    EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
        angular::IterationResult::Advanced);
    EXPECT_EQ(evaluations, 6u);
    EXPECT_EQ(current.stamp, 6u);
    EXPECT_LT(current.objective, .04);
    EXPECT_GT(normal[1], .09);
    EXPECT_LT(normal[1], .11); // The half step, not the failed full step.
}

TEST(PolyhedralReconstruction, AngularFatalTrialsPropagateWithoutChangingCurrentOrTryingSmallerSteps)
{
    namespace angular = PLIC::angular_detail;
    for (const auto code : {PLIC::GeometryErrorCode::ResourceLimit, PLIC::GeometryErrorCode::ProvenanceMismatch,
             PLIC::GeometryErrorCode::InvalidInput, PLIC::GeometryErrorCode::InvalidTopology})
    {
        SCOPED_TRACE(int(code));
        AngularVector normal{1, 0, 0};
        auto current = angular_model(normal, .2, 0, 0);
        const auto old_normal = normal;
        const auto old_residual = current.residual;
        unsigned evaluations = 0;
        const auto evaluate = [&](AngularVector trial)
        {
            if (++evaluations == 5)
                throw PLIC::GeometryError(code, "unresolved-looking text must not select recovery");
            return angular_model(trial, .2, 0, evaluations);
        };
        angular::Qualification qualification;
        try
        {
            (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
            FAIL() << "Fatal angular trial was swallowed.";
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), code);
        }
        EXPECT_EQ(evaluations, 5u);
        EXPECT_EQ(normal, old_normal);
        EXPECT_EQ(current.residual, old_residual);
        EXPECT_EQ(current.stamp, 0u);
    }
    AngularVector normal{1, 0, 0};
    auto current = angular_model(normal, .2, 0, 0);
    unsigned evaluations = 0;
    const auto untyped = [&](AngularVector trial)
    {
        if (++evaluations == 5)
            throw std::invalid_argument("unresolved arithmetic");
        return angular_model(trial, .2, 0, evaluations);
    };
    angular::Qualification qualification;
    EXPECT_THROW((void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, untyped, angular_unit, qualification),
        std::invalid_argument);
    EXPECT_EQ(evaluations, 5u);
    EXPECT_EQ(current.stamp, 0u);
}

TEST(PolyhedralReconstruction, AngularFailedTrialRemainsChargedWhenNextAttemptExhaustsBudget)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal{1, 0, 0};
    auto current = angular_model(normal, .2, 0, 0);
    unsigned evaluations = 0;
    const auto evaluate = [&](AngularVector trial)
    {
        if (evaluations++ >= 5)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "evaluation budget exhausted");
        if (evaluations == 5)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "failed fifth evaluation");
        return angular_model(trial, .2, 0, evaluations);
    };
    angular::Qualification qualification;
    try
    {
        (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
        FAIL() << "Budget exhaustion was swallowed.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
    }
    EXPECT_EQ(evaluations, 6u);
    EXPECT_EQ(normal, (AngularVector{1, 0, 0}));
    EXPECT_EQ(current.stamp, 0u);
}

TEST(PolyhedralReconstruction, AngularQualificationRetriesOnlyArithmeticAndUsesActualProbeStep)
{
    namespace angular = PLIC::angular_detail;
    const AngularVector normal{1, 0, 0}, tangent{0, 1, 0}, second{0, 0, 1};
    const std::array residual{.2, .4};
    unsigned evaluations = 0;
    const auto evaluate = [&](AngularVector trial)
    {
        if (++evaluations == 2)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "second probe unresolved");
        return AngularEvaluation{{2 * trial[1], 4 * trial[2]}};
    };
    const auto qualification = angular::qualify_normal(normal, tangent, second, residual, 1e8, evaluate);
    EXPECT_EQ(evaluations, 6u); // Two first-step attempts, then a complete smaller-step quartet.
    EXPECT_EQ(qualification.probe_step, 5e-5);
    EXPECT_NEAR(qualification.first_step, -.1, 1e-15);
    EXPECT_NEAR(qualification.second_step, -.1, 1e-15);
    EXPECT_NEAR(qualification.condition, 4, 1e-14);

    evaluations = 0;
    const auto fatal = [&](AngularVector trial)
    {
        if (++evaluations == 2)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "probe budget exhausted");
        return AngularEvaluation{{trial[1], trial[2]}};
    };
    EXPECT_THROW((void) angular::qualify_normal(normal, tangent, second, residual, 1e8, fatal), PLIC::GeometryError);
    EXPECT_EQ(evaluations, 2u);
}

TEST(PolyhedralReconstruction, AngularConvergenceStillRequiresObservableCurrentNormal)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal{1, 0, 0};
    auto current = angular_model(normal, 0, 0, 0);
    unsigned evaluations = 0;
    const auto observable = [&](AngularVector trial)
    {
        if (++evaluations > 4)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "unused optional trial");
        return angular_model(trial, 0, 0, evaluations);
    };
    angular::Qualification qualification;
    EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, observable, angular_unit, qualification),
        angular::IterationResult::Converged);
    EXPECT_EQ(evaluations, 4u);
    EXPECT_EQ(current.stamp, 0u);
    evaluations = 0;
    const auto unobservable = [&](AngularVector)
    {
        ++evaluations;
        return AngularEvaluation{{0, 0}};
    };
    EXPECT_THROW((void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, unobservable, angular_unit, qualification),
        std::invalid_argument);
    EXPECT_EQ(evaluations, 4u); // Computed rank failure does not retry another h.
    EXPECT_EQ(current.stamp, 0u);
}

TEST(PolyhedralReconstruction, AngularQualificationLimitsProbeRetriesAndRetainsSignalGate)
{
    namespace angular = PLIC::angular_detail;
    const AngularVector normal{1, 0, 0}, tangent{0, 1, 0}, second{0, 0, 1};
    const std::array residual{0., 0.};
    std::vector<double> attempted;
    const auto unresolved = [&](AngularVector trial) -> AngularEvaluation
    {
        attempted.push_back(std::max(std::abs(trial[1]), std::abs(trial[2])));
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "all probe steps unresolved");
    };
    EXPECT_THROW(
        (void) angular::qualify_normal(normal, tangent, second, residual, 1e8, unresolved), PLIC::GeometryError);
    EXPECT_EQ(attempted, (std::vector<double>{1e-4, 5e-5, 2.5e-5}));
    unsigned evaluations = 0;
    const double signal = .75 * std::sqrt(2.) * 64 * std::numeric_limits<double>::epsilon() / 5e-5;
    const auto weak_signal = [&](AngularVector trial)
    {
        if (++evaluations == 1)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "first probe unresolved");
        return AngularEvaluation{{signal * trial[1], signal * trial[2]}};
    };
    EXPECT_THROW(
        (void) angular::qualify_normal(normal, tangent, second, residual, 1e8, weak_signal), std::invalid_argument);
    EXPECT_EQ(evaluations, 5u); // The signal gate uses h=5e-5 and remains fatal.
}

TEST(PolyhedralReconstruction, OrdinaryOffsetIterationAndClipBudgetsRemainFatal)
{
    const auto sample = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .3);
    for (const bool clips : {false, true})
    {
        PLIC::PolyhedralReconstructionOptions options;
        if (clips)
            options.maximum_clip_evaluations = 1;
        else
            options.maximum_offset_iterations = 1;
        try
        {
            (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(sample, {1, 0, 0}, options);
            FAIL() << "Insufficient offset work budget was accepted.";
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        }
        EXPECT_EQ(sample.gas_volume, .3);
        EXPECT_EQ(sample.carrier_volume, .7);
    }
}

TEST(PolyhedralReconstruction, OrdinaryMeasureOffsetSolvesTinyPhasesWithoutCutCellConstruction)
{
    using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const auto cell = Cell::box({0, 0, 0}, {1, 1, 1});
    const double minority = 1e-27;
    for (const bool gas : {false, true})
    {
        const OrdinarySample sample{cell, gas ? minority : 1., gas ? 1. : minority};
        PLIC::reset_geometry_statistics();
        PLIC::PolyhedralReconstructionOptions options;
        options.measure_only = true;
        const auto fit =
            PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(sample, {1, 1, 1}, options);
        auto plane = fit.gas;
        if (!gas)
        {
            for (auto& n : plane.normal)
                n = -n;
            plane.offset = -plane.offset;
        }
        const auto measured = PLIC::OrdinaryPlaneMeasure(*cell).below(plane);
        EXPECT_LE(std::abs(measured.volume - minority) + measured.numerical_error_estimate,
            minority * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
        EXPECT_GT(fit.target_volume_numerical_error_estimate, 0);
        EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
        EXPECT_EQ(PLIC::geometry_statistics().moment_evaluations, 0u);
        EXPECT_EQ(sample.gas_volume, gas ? minority : 1.);
        EXPECT_EQ(sample.carrier_volume, gas ? 1. : minority);
    }
}

TEST(PolyhedralReconstruction, OrdinaryMeasureRejectsToleranceBelowItsReportedAccuracyFloor)
{
    const auto sample = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .25);
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    options.volume_tolerance = 1e-25;
    try
    {
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(sample, {1, 0, 0}, options);
        FAIL() << "An unattainable numerical target was accepted.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
        EXPECT_NE(std::string(error.what()).find("numerical_error_estimate="), std::string::npos);
    }
    EXPECT_EQ(sample.gas_volume, .25);
    EXPECT_EQ(sample.carrier_volume, .75);
}

} // namespace
