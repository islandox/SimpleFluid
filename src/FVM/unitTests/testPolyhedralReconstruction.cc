/** @file testPolyhedralReconstruction.cc
 * @brief Resolution requests, bounded PLIC fitting and retained finite inventories.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/GeometricVoF.hh"
#include "FVM/details/VoFFailure.hh"
#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include <iomanip>
#include <iostream>
#include "plic_algebraic_planar_corner_fixture.hh"
#include "geometry/PLIC/details/AngularFit.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include "plic_full50_step3_fixture.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
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
        PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal);
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

TEST(PolyhedralReconstruction, WeakPhaseSupportModelRejectionRetainsMeasuredEvidence)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .1);
    std::vector<OrdinarySample> stencil;
    const std::array<std::array<int, 3>, 6> indices{{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}};
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        auto sample = uniform_sample<ArithmeticMode::Ordinary>(indices[i], 1, i == 0 ? .05 : i == 1 ? .2 : .1);
        sample.id = 10 + i;
        stencil.push_back(sample);
    }
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    options.maximum_model_residual = .01;
    const auto assessment = Kernel::assess_resolution(target, stencil, options);
    ASSERT_TRUE(assessment.admitted());
    ASSERT_FALSE(assessment.angular_fit_eligible);
    try
    {
        (void) Kernel::reconstruct(target, stencil, options);
        FAIL() << "Excessive gradient plane-model residual was accepted.";
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
        ASSERT_TRUE(error.has_fit_diagnostics());
        const auto& evidence = error.diagnostics();
        EXPECT_GT(evidence.rms, options.maximum_model_residual);
        EXPECT_EQ(evidence.rms, evidence.best_rms);
        EXPECT_EQ(evidence.normal, evidence.best_normal);
        EXPECT_GT(evidence.condition, 0);
        EXPECT_EQ(evidence.iterations, 0u);
        EXPECT_EQ(evidence.finite_difference_scale, 0);
        EXPECT_TRUE(std::isinf(evidence.angular_step));
        EXPECT_GT(evidence.clip_evaluations, 0u);
        EXPECT_EQ(evidence.stencil_ids, (std::vector<std::uint64_t>{10, 11, 12, 13, 14, 15}));
    }
}

template<ArithmeticMode Mode> void small_phase_gradient_orientation()
{
    using Sample = PLIC::PolyhedralPhaseSample<Mode>;
    const std::array<std::array<int, 3>, 6> indices{
        {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}};
    // Symmetric axis neighbors give the independent weighted-LS normal
    // (1-.4, .9-.5, .8-.6) = .2*(3,2,1), independent of target alpha.
    const std::array<double, 6> fractions{1, .4, .9, .5, .8, .6};
    for (const double scale : {.5, 1., 2.})
        for (const bool gas : {false, true})
        {
            SCOPED_TRACE(scale);
            SCOPED_TRACE(gas);
            auto target = uniform_sample<Mode>({0, 0, 0}, scale, std::ldexp(1., -20));
            std::vector<Sample> stencil;
            for (std::size_t i = 0; i < indices.size(); ++i)
                stencil.push_back(uniform_sample<Mode>(indices[i], scale, fractions[i]));
            if (!gas)
            {
                std::swap(target.gas_volume, target.carrier_volume);
                for (auto& sample : stencil)
                    std::swap(sample.gas_volume, sample.carrier_volume);
            }
            const auto original = target;
            PLIC::PolyhedralReconstructionOptions options;
            options.maximum_model_residual = 1e-12;
            options.maximum_fit_iterations = 1;
            const auto transport =
                PLIC::PolyhedralReconstruction<Mode>::assess_transport_support(target, stencil, options);
            EXPECT_TRUE(transport.admitted());
            EXPECT_EQ(transport.angular_support_cells, 0u);
            const auto fit = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, options);
            EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
            EXPECT_FALSE(fit.fit_converged);
            EXPECT_EQ(fit.fit_iterations, 0u);
            EXPECT_FALSE(fit.stencil_volume_residual_measured);
            EXPECT_EQ(fit.orientation_step, 0);
            EXPECT_FALSE(fit.pure_gas);
            EXPECT_FALSE(fit.pure_carrier);
            EXPECT_TRUE(fit.resolution.admitted());
            EXPECT_NEAR(fit.stencil_condition, 1, 2e-15);
            for (unsigned axis = 0; axis < 3; ++axis)
                EXPECT_NEAR(fit.gas.normal[axis], (gas ? 1. : -1.) * (3. - axis) / std::sqrt(14.), 2e-15);
            // Independent tetrahedron volume: 3*x+2*y+z <= q has V=q^3/36.
            const auto minority = gas ? target.gas_volume : target.carrier_volume;
            EXPECT_NEAR(fit.gas.offset, (gas ? 1 : -1) * std::cbrt(36 * minority) / std::sqrt(14.), 2e-15 * scale);
            EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) +
                          fit.finite_target_volume_difference.absolute_error,
                static_cast<long double>(minority) * options.volume_tolerance);
            EXPECT_EQ(target.gas_volume, original.gas_volume);
            EXPECT_EQ(target.carrier_volume, original.carrier_volume);
            auto capped = options;
            capped.maximum_clip_evaluations = 1;
            EXPECT_THROW((void) PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, capped),
                std::invalid_argument);
            capped = options;
            capped.maximum_offset_iterations = 1;
            EXPECT_THROW((void) PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, capped),
                std::invalid_argument);
            EXPECT_EQ(target.gas_volume, original.gas_volume);
            EXPECT_EQ(target.carrier_volume, original.carrier_volume);
        }
}

TEST(PolyhedralReconstruction, SmallPhaseUsesSupportedGradientWithoutAngularOrModelGate)
{
    small_phase_gradient_orientation<ArithmeticMode::Ordinary>();
    small_phase_gradient_orientation<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void small_phase_policy_boundaries()
{
    using Sample = PLIC::PolyhedralPhaseSample<Mode>;
    for (const double scale : {.5, 1., 2.})
        for (const bool complement : {false, true})
            for (const double fraction : {1. / 4096, 1. / 512, 1. / 64, .0001, .0002})
            {
                SCOPED_TRACE(scale);
                SCOPED_TRACE(complement);
                SCOPED_TRACE(fraction);
                auto target = uniform_sample<Mode>({0, 0, 0}, scale, fraction);
                std::vector<Sample> stencil;
                for (const auto index : std::array<std::array<int, 3>, 6>{
                         {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}})
                    stencil.push_back(uniform_sample<Mode>(index, scale,
                        index[0] < 0   ? 1
                        : index[0] > 0 ? 0
                                       : fraction));
                if (complement)
                {
                    std::swap(target.gas_volume, target.carrier_volume);
                    for (auto& sample : stencil)
                        std::swap(sample.gas_volume, sample.carrier_volume);
                }
                PLIC::PolyhedralReconstructionOptions options;
                options.minimum_optimized_phase_length_ratio = .125;
                const auto fit = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, options);
                const bool gradient = fraction < 1. / 512;
                EXPECT_EQ(fit.orientation_method, gradient ? PLIC::PolyhedralOrientationMethod::Gradient
                                                           : PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
                EXPECT_EQ(fit.fit_converged, !gradient);
                EXPECT_EQ(fit.stencil_volume_residual_measured, !gradient);
                if (fraction == .0001 || fraction == .0002)
                {
                    const auto default_fit = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil);
                    EXPECT_EQ(default_fit.orientation_method,
                        fraction == .0001 ? PLIC::PolyhedralOrientationMethod::Gradient
                                          : PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
                }
                options.minimum_optimized_phase_length_ratio = 0;
                const auto angular = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, options);
                EXPECT_EQ(angular.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
                EXPECT_TRUE(angular.fit_converged);
                EXPECT_TRUE(angular.stencil_volume_residual_measured);
            }
    const auto target = uniform_sample<Mode>({0, 0, 0}, 1, std::ldexp(1., -20));
    for (const double invalid :
        {-1., 1.01, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        PLIC::PolyhedralReconstructionOptions options;
        options.minimum_optimized_phase_length_ratio = invalid;
        EXPECT_THROW(PLIC::PolyhedralReconstruction<Mode>::validate(target, options), std::invalid_argument);
    }
    std::vector<Sample> unsupported;
    for (const auto index : neighbors())
        unsupported.push_back(uniform_sample<Mode>(index, 1, std::ldexp(1., -20)));
    EXPECT_THROW((void) PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, unsupported),
        PLIC::PolyhedralResolutionRequired);
    std::vector<Sample> deficient;
    for (const int x : {-2, -1, 1, 2})
        deficient.push_back(uniform_sample<Mode>({x, 0, 0}, 1, x < 0 ? 1 : 0));
    EXPECT_THROW((void) PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, deficient),
        PLIC::PolyhedralResolutionRequired);
}

TEST(PolyhedralReconstruction, SmallPhaseExclusionRetainsThresholdSupportAndOffsetContracts)
{
    small_phase_policy_boundaries<ArithmeticMode::Ordinary>();
    small_phase_policy_boundaries<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void weak_planar_layer(bool corner, double wall_fraction = .25)
{
    using Sample = PLIC::PolyhedralPhaseSample<Mode>;
    using Kernel = PLIC::PolyhedralReconstruction<Mode>;
    for (const double h : {.5, 1., 2.})
        for (const bool complement : {false, true})
        {
            const double fraction = corner ? .125 : wall_fraction;
            auto target = uniform_sample<Mode>({0, 0, 0}, h, fraction);
            std::vector<Sample> stencil;
            // The wall layer is x<wall_fraction*h. The corner layer x+y<h/2 has
            // triangular cross-section area h^2/8. All volumes are analytic.
            for (const auto index :
                (corner ? std::vector<std::array<int, 3>>{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}
                        : std::vector<std::array<int, 3>>{{1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}))
            {
                const bool inward = index[0] != 0 || (corner && index[1] != 0);
                stencil.push_back(uniform_sample<Mode>(index, h, inward ? 0 : fraction));
            }
            if (complement)
            {
                std::swap(target.gas_volume, target.carrier_volume);
                for (auto& sample : stencil)
                    std::swap(sample.gas_volume, sample.carrier_volume);
            }
            const auto original = target;
            const auto assessment = Kernel::assess_resolution(target, stencil);
            ASSERT_TRUE(assessment.admitted());
            EXPECT_FALSE(assessment.angular_fit_eligible);
            EXPECT_GT(assessment.directional_strength, assessment.directional_uncertainty);
            EXPECT_NEAR(assessment.stencil_condition, 2, 1e-12);
            EXPECT_NEAR(
                complement ? assessment.carrier_support_fraction : assessment.gas_support_fraction, fraction, 1e-15);
            const auto fit = Kernel::reconstruct(target, stencil);
            EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
            EXPECT_FALSE(fit.fit_converged);
            EXPECT_EQ(fit.fit_iterations, 0u);
            EXPECT_TRUE(fit.stencil_volume_residual_measured);
            EXPECT_LE(fit.stencil_volume_residual, PLIC::PolyhedralReconstructionOptions{}.fit_tolerance);
            const auto sign = complement ? -1 : 1;
            const double nx = corner ? 1 / std::sqrt(2.) : 1;
            EXPECT_NEAR(fit.gas.normal[0], sign * nx, 1e-12);
            EXPECT_NEAR(fit.gas.normal[1], corner ? sign * nx : 0, 1e-12);
            EXPECT_NEAR(fit.gas.normal[2], 0, 1e-12);
            const PLIC::GeometryPoint point{corner ? h / 4 : wall_fraction * h, corner ? h / 4 : h / 2, h / 2};
            double distance = -fit.gas.offset;
            for (unsigned axis = 0; axis < 3; ++axis)
                distance += fit.gas.normal[axis] * (point[axis] - fit.gas.anchor[axis]);
            EXPECT_NEAR(distance, 0, 2e-12 * h);
            const auto minority = std::min(target.gas_volume, target.carrier_volume);
            EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) +
                          fit.finite_target_volume_difference.absolute_error +
                          fit.target_volume_numerical_error_estimate,
                static_cast<long double>(minority) * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
            EXPECT_EQ(target.gas_volume, original.gas_volume);
            EXPECT_EQ(target.carrier_volume, original.carrier_volume);
        }
}

TEST(PolyhedralReconstruction, WallAdjacentQuarterLayerAdmitsGradientWithoutMajoritySupport)
{
    weak_planar_layer<ArithmeticMode::Ordinary>(false);
    weak_planar_layer<ArithmeticMode::BoundedCertification>(false);
}

TEST(PolyhedralReconstruction, WallAdjacentTenthLayerAdmitsGradientWithoutMajoritySupport)
{
    weak_planar_layer<ArithmeticMode::Ordinary>(false, .1);
    weak_planar_layer<ArithmeticMode::BoundedCertification>(false, .1);
}

TEST(PolyhedralReconstruction, ObliqueCornerLayerUsesDirectionalInformationInsteadOfAnAxisFallback)
{
    weak_planar_layer<ArithmeticMode::Ordinary>(true);
    weak_planar_layer<ArithmeticMode::BoundedCertification>(true);
}

TEST(PolyhedralReconstruction, RoundoffOnlyDiluteVariationDoesNotEstablishDirection)
{
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .01);
    std::vector<OrdinarySample> stencil;
    for (const auto index : neighbors())
        stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, .01));
    stencil.front().gas_volume = std::nextafter(.01, 1.);
    stencil.front().carrier_volume = 1 - stencil.front().gas_volume;
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(target, stencil);
    EXPECT_EQ(assessment.reason, PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal);
    EXPECT_GT(assessment.stencil_condition, 0);
    EXPECT_LE(assessment.directional_strength, assessment.directional_uncertainty);
    EXPECT_THROW((void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil),
        PLIC::PolyhedralResolutionRequired);
}

TEST(PolyhedralReconstruction, GeometryCoverageAndConditionRemainIndependentAdmissionGates)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .25);
    std::vector<OrdinarySample> constant_geometry;
    for (const auto index : neighbors())
        constant_geometry.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, .25));
    EXPECT_TRUE(Kernel::assess_stencil_geometry(target, constant_geometry).admitted());
    EXPECT_EQ(Kernel::assess_resolution(target, constant_geometry).reason, Reason::NoDirectionalSignal);
    std::vector<OrdinarySample> planar;
    for (const auto index : std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}})
        planar.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, index[0] < 0 ? 1 : 0));
    EXPECT_EQ(Kernel::assess_resolution(target, planar).reason, Reason::RankDeficientStencil);
    EXPECT_THROW((void) Kernel::reconstruct(target, planar), PLIC::PolyhedralResolutionRequired);
    auto incomplete = planar;
    incomplete.resize(3);
    EXPECT_EQ(Kernel::assess_resolution(target, incomplete).reason, Reason::InsufficientStencil);
    planar.front().geometry.reset();
    EXPECT_THROW((void) Kernel::assess_resolution(target, planar), std::invalid_argument);

    constexpr double d = 1e-6;
    std::vector<OrdinarySample> nearly_collinear;
    for (const PLIC::GeometryPoint shift : {PLIC::GeometryPoint{-2, -d, 0}, {-1, d, d}, {1, -d, d}, {2, d, -d}})
    {
        const auto geometry =
            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(shift, {shift[0] + 1, shift[1] + 1, shift[2] + 1});
        const auto volume = geometry->volume().as_double();
        nearly_collinear.push_back({geometry, shift[0] < 0 ? volume : 0, shift[0] < 0 ? 0 : volume});
    }
    const auto assessment = Kernel::assess_resolution(target, nearly_collinear);
    EXPECT_EQ(assessment.reason, Reason::IllConditionedStencil);
    EXPECT_GT(assessment.stencil_condition, PLIC::PolyhedralReconstructionOptions{}.maximum_condition);
    std::vector<OrdinarySample> nearly_coplanar;
    for (const PLIC::GeometryPoint shift :
        {PLIC::GeometryPoint{-1, 0, d}, {1, 0, d}, {0, -1, -d}, {0, 1, -d}})
    {
        const auto geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(
            shift, {shift[0] + 1, shift[1] + 1, shift[2] + 1});
        const auto volume = geometry->volume().as_double();
        nearly_coplanar.push_back({geometry, shift[0] < 0 ? volume : 0, shift[0] < 0 ? 0 : volume});
    }
    EXPECT_EQ(Kernel::assess_resolution(target, nearly_coplanar).reason, Reason::IllConditionedStencil);
    EXPECT_THROW((void) Kernel::reconstruct(target, nearly_coplanar), PLIC::PolyhedralResolutionRequired);
}

TEST(PolyhedralReconstruction, OppositeNeighborPairsPreserveKnownOnePlaneContradiction)
{
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .5);
    std::vector<OrdinarySample> stencil;
    for (const auto index :
        std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}})
        stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, index[0] == 1 ? 0 : 1));
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(target, stencil);
    ASSERT_TRUE(assessment.admitted());
    EXPECT_TRUE(assessment.angular_fit_eligible);
    // A plane bisecting the centrally symmetric target predicts fractions
    // summing to one on each opposite neighbor pair. Actual y/z pairs each
    // sum to two, forcing RMS >= sqrt(1/6), independently of its normal.
    ASSERT_GT(std::sqrt(1. / 6), PLIC::PolyhedralReconstructionOptions{}.maximum_model_residual);
    try
    {
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil);
        FAIL() << "A strict target fit cannot repair this one-plane contradiction.";
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
    }
}

TEST(PolyhedralReconstruction, ModerateWeakGradientRetainsItsConfiguredModelBound)
{
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .25);
    std::vector<OrdinarySample> stencil;
    for (const auto index :
        std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}})
        stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1,
            index[0] < 0   ? 0
            : index[0] > 0 ? .2
                           : .25));
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(target, stencil);
    ASSERT_TRUE(assessment.admitted());
    EXPECT_FALSE(assessment.angular_fit_eligible);
    // The selected gradient normal is -x; its offset predicts gas=1 in the
    // right cell instead of .2. This is a bound on that selected gradient fit,
    // not a proof about every possible orientation.
    try
    {
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil);
        FAIL();
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
    }
}

TEST(PolyhedralReconstruction, CapturedFull50Step3AdmitsSupportedTinyPhaseWithoutClaimingModelAccuracy)
{
    if constexpr (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Lossless native certificates capture the 64-bit-significand source geometry.";
    const auto cells = captured_full50::cells();
    ASSERT_EQ(cells.size(), 16u);
    ASSERT_EQ(cells.front().gid, 21u);
    const auto target = cells.front().sample;
    std::vector<OrdinarySample> stencil;
    for (std::size_t i = 1; i < cells.size(); ++i)
        stencil.push_back(cells[i].sample);
    const auto assessment =
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::assess_resolution(target, stencil);
    ASSERT_TRUE(assessment.admitted());
    EXPECT_FALSE(assessment.angular_fit_eligible);
    EXPECT_NEAR(assessment.gas_support_fraction, .2891142723056233, 1e-15);
    EXPECT_GT(assessment.directional_strength, assessment.directional_uncertainty);
    EXPECT_LT(assessment.stencil_condition, 3);
    const auto fit = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(target, stencil);
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
    EXPECT_FALSE(fit.fit_converged);
    EXPECT_FALSE(fit.stencil_volume_residual_measured);
    EXPECT_FALSE(fit.pure_gas);
    EXPECT_FALSE(fit.pure_carrier);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.gas_volume) * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
    EXPECT_EQ(target.gas_volume, cells.front().sample.gas_volume);
    EXPECT_EQ(target.carrier_volume, cells.front().sample.carrier_volume);
    long double sum = 0;
    for (const auto& sample : stencil)
    {
        const auto residual =
            (sample.geometry->phases(fit.gas).gas.value - sample.gas_volume) / sample.geometry->volume().value;
        sum += residual * residual;
    }
    const auto rms = std::sqrt(sum / stencil.size());
    EXPECT_GT(rms, PLIC::PolyhedralReconstructionOptions{}.maximum_model_residual);
    EXPECT_NEAR(rms, .5689145923321345, 1e-12);
    RecordProperty("stencil_condition", assessment.stencil_condition);
    RecordProperty("directional_strength", assessment.directional_strength);
    RecordProperty("directional_uncertainty", assessment.directional_uncertainty);
    RecordProperty("target_gas_volume", target.gas_volume);
    RecordProperty("strict_offset_residual", double(fit.finite_target_volume_difference.value));
    RecordProperty("frozen_neighbor_rms", double(rms));
    auto without_small_phase_exclusion = PLIC::PolyhedralReconstructionOptions{};
    without_small_phase_exclusion.minimum_optimized_phase_length_ratio = 0;
    try
    {
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::reconstruct(
            target, stencil, without_small_phase_exclusion);
        FAIL() << "Disabling the existing small-phase policy restores the selected-model gate.";
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
    }
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
    EXPECT_EQ(assessment.reason, PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal);
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
        EXPECT_EQ(assessment.reason, PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal);
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
    PLIC::PolyhedralReconstructionOptions options;
    options.minimum_optimized_phase_length_ratio = 0;
    const auto fit =
        PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::reconstruct(target, stencil, options);
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
        PLIC::PolyhedralResolutionAssessment::Reason::RankDeficientStencil);
    std::vector<BoundedSample> constant;
    for (const auto index : neighbors())
        constant.push_back(uniform_sample(index, 1, .5));
    EXPECT_EQ(PLIC::PolyhedralReconstruction<ArithmeticMode::BoundedCertification>::assess_resolution(target, constant)
                  .reason,
        PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal);
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
                SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(
                    *exact)
                    ->phases(fit.gas);
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

TEST(PolyhedralReconstruction, AngularNonlinearNonzeroResidualOptimumUsesBoundedRefinement)
{
    namespace angular = PLIC::angular_detail;
    // The exact minimum has y = 1/80 and z = 0: the derivative of the
    // squared objective with respect to y is zero there, despite nonzero RMS.
    constexpr double optimum_y = 1. / 80.;
    constexpr double initial_z = 1e-5;
    AngularVector normal{std::sqrt(1 - optimum_y * optimum_y - initial_z * initial_z), optimum_y, initial_z};
    unsigned evaluations = 0;
    const auto evaluate = [&](AngularVector candidate)
    {
        if (++evaluations > 32768)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "evaluation budget exhausted");
        candidate = angular_unit(candidate);
        const double y = candidate[1];
        AngularEvaluation result{{y - 9. / 80., 6400 * y * y * y + 1. / 48., candidate[2]}};
        for (const auto value : result.residual)
            result.objective += value * value;
        result.objective_upper = result.objective;
        return result;
    };
    auto current = evaluate(normal);
    angular::Qualification qualification;
    ASSERT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
        angular::IterationResult::Advanced);
    // Removing z should not introduce a substantial displacement from the
    // independently known optimum in y through a biased derivative.
    EXPECT_NEAR(normal[1], optimum_y, 5e-8);
    EXPECT_LT(std::abs(normal[2]), initial_z / 10);
    bool converged = false;
    for (unsigned iteration = 1; iteration < 24; ++iteration)
    {
        const auto status = angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
        ASSERT_NE(status, angular::IterationResult::Stalled);
        if (status == angular::IterationResult::Converged)
        {
            converged = true;
            break;
        }
    }
    EXPECT_TRUE(converged);
    EXPECT_LE(evaluations, 32768u);
    EXPECT_NEAR(normal[1], optimum_y, 5e-8);
    EXPECT_NEAR(normal[2], 0., 2e-8);
    EXPECT_NEAR(std::sqrt(current.objective / 3), std::sqrt((.1 * .1 + (1. / 30) * (1. / 30)) / 3), 1e-11);
}

TEST(PolyhedralReconstruction, AngularOptionalFinerArithmeticFailureRetainsValidCoarseSearch)
{
    namespace angular = PLIC::angular_detail;
    for (const auto scenario : {0, 1, 2, 3})
    {
        SCOPED_TRACE(scenario);
        AngularVector normal{1, 0, 0};
        auto current = angular_model(normal, 5e-5, 0, 0);
        const auto initial_objective = current.objective;
        unsigned evaluations = 0;
        const auto evaluate = [&](AngularVector trial)
        {
            ++evaluations;
            if (scenario == 2 && evaluations == 5)
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "fatal finer probe");
            if (scenario == 3 && evaluations >= 5)
                return AngularEvaluation{{0, 0}}; // Finer qualification is unobservable.
            if (evaluations == 5 || evaluations == 6)
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "unavailable finer probe");
            auto result = angular_model(trial, 5e-5, 0, evaluations);
            if (scenario == 1 && evaluations > 6)
                result.objective = initial_objective + 1;
            return result;
        };
        angular::Qualification qualification;
        if (scenario == 0)
        {
            EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
                angular::IterationResult::Advanced);
            EXPECT_EQ(evaluations, 11u); // Includes qualification of the accepted trial.
            EXPECT_LT(current.objective, initial_objective);
            EXPECT_NEAR(normal[1], 5e-5, 1e-12);
            EXPECT_EQ(qualification.probe_step, 1e-4);
        }
        else
        {
            try
            {
                (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
                FAIL() << "A failed finer qualification must not be reported as convergence.";
            }
            catch (const angular::Unobservable&)
            {
                EXPECT_EQ(scenario, 3);
                EXPECT_EQ(evaluations, 8u);
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_NE(scenario, 3);
                EXPECT_EQ(error.code(), scenario == 1 ? PLIC::GeometryErrorCode::UnresolvedArithmetic
                                                     : PLIC::GeometryErrorCode::ResourceLimit);
                EXPECT_EQ(evaluations, scenario == 1 ? 14u : 5u);
            }
            EXPECT_EQ(normal, (AngularVector{1, 0, 0}));
            EXPECT_EQ(current.objective, initial_objective);
        }
    }
}

AngularEvaluation angular_curvature_model(AngularVector normal)
{
    const double x = normal[1] / normal[0], y = normal[2] / normal[0];
    AngularEvaluation result{{x, y, .3 + .5 * (x + y) * (x + y)}};
    for (const auto value : result.residual)
        result.objective += value * value;
    result.objective_upper = result.objective;
    return result;
}

TEST(PolyhedralReconstruction, AngularResidualCurvatureReachesAnalyticNonzeroMinimumWithinBudget)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal = angular_unit({1, .01, .01});
    auto current = angular_curvature_model(normal);
    unsigned evaluations = 1;
    const auto evaluate = [&](AngularVector candidate)
    {
        if (++evaluations > 32768)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "evaluation budget exhausted");
        return angular_curvature_model(candidate);
    };
    angular::Qualification qualification;
    bool converged = false;
    for (unsigned iteration = 0; iteration < 24; ++iteration)
    {
        const auto status = angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
        ASSERT_NE(status, angular::IterationResult::Stalled);
        if (status == angular::IterationResult::Converged)
        {
            converged = true;
            break;
        }
    }
    EXPECT_TRUE(converged);
    EXPECT_LE(qualification.step, 2e-8);
    EXPECT_LE(evaluations, 32768u);
    EXPECT_NEAR(normal[1], 0., 4e-8);
    EXPECT_NEAR(normal[2], 0., 4e-8);
    EXPECT_NEAR(std::sqrt(current.objective / 3), .3 / std::sqrt(3.), 1e-12);
}

TEST(PolyhedralReconstruction, AngularStrongResidualCurvatureConvergesWithinOriginalIterationCap)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal = angular_unit({1, 1e-4, 1e-4});
    unsigned evaluations = 0;
    const auto evaluate = [&](AngularVector value)
    {
        if (++evaluations > 32768)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "evaluation budget exhausted");
        const double x = value[1] / value[0], y = value[2] / value[0];
        AngularEvaluation result{{x, y, .3 + 50 * (x + y) * (x + y)}};
        for (const double r : result.residual)
            result.objective += r * r;
        result.objective_upper = result.objective;
        return result;
    };
    auto current = evaluate(normal);
    angular::Qualification qualification;
    bool converged = false;
    for (unsigned i = 0; i < 24; ++i)
    {
        const auto status = angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
        ASSERT_NE(status, angular::IterationResult::Stalled);
        if (status == angular::IterationResult::Converged)
        {
            converged = true;
            break;
        }
    }
    EXPECT_TRUE(converged);
    EXPECT_LE(qualification.step, 2e-8);
    EXPECT_LE(evaluations, 32768u);
    EXPECT_NEAR(normal[1], 0., 3e-8);
    EXPECT_NEAR(normal[2], 0., 3e-8);
    EXPECT_NEAR(std::sqrt(current.objective / 3), .3 / std::sqrt(3.), 1e-12);
}

TEST(PolyhedralReconstruction, AngularCurvatureTrialsPreserveFailureAndAccountingContracts)
{
    namespace angular = PLIC::angular_detail;
    // The first twelve calls are the three axial quartets. Calls 13-16
    // sample mixed curvature and call 17 evaluates the Newton candidate.
    for (unsigned failed_call = 13; failed_call <= 17; ++failed_call)
        for (const bool fatal : {false, true})
        {
            SCOPED_TRACE(failed_call);
            SCOPED_TRACE(fatal);
            AngularVector normal = angular_unit({1, 1e-5, 1e-5});
            const auto initial_normal = normal;
            auto current = angular_curvature_model(normal);
            const auto initial_objective = current.objective;
            unsigned evaluations = 0;
            const auto evaluate = [&](AngularVector candidate)
            {
                ++evaluations;
                EXPECT_EQ(normal, initial_normal);
                EXPECT_EQ(current.objective, initial_objective);
                if (evaluations == failed_call)
                    throw PLIC::GeometryError(fatal ? PLIC::GeometryErrorCode::ResourceLimit
                                                   : PLIC::GeometryErrorCode::UnresolvedArithmetic,
                        "optional curvature evaluation");
                return angular_curvature_model(candidate);
            };
            angular::Qualification qualification;
            if (fatal)
            {
                try
                {
                    (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
                    FAIL() << "Fatal optional evaluation was swallowed.";
                }
                catch (const PLIC::GeometryError& error)
                {
                    EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
                }
                EXPECT_EQ(evaluations, failed_call);
                EXPECT_EQ(normal, initial_normal);
                EXPECT_EQ(current.objective, initial_objective);
            }
            else
            {
                EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
                    angular::IterationResult::Advanced);
                EXPECT_EQ(evaluations, failed_call + 5); // GN trial and its four qualification probes.
                EXPECT_LT(current.objective, initial_objective);
            }
        }
}

TEST(PolyhedralReconstruction, AngularCurvatureRejectsMalformedOrNondecreasingOptionalTrials)
{
    namespace angular = PLIC::angular_detail;
    for (const bool malformed : {false, true})
    {
        AngularVector normal = angular_unit({1, 1e-5, 1e-5});
        auto current = angular_curvature_model(normal);
        const auto original = current.objective;
        unsigned calls = 0;
        const auto evaluate = [&](AngularVector candidate)
        {
            ++calls;
            auto result = angular_curvature_model(candidate);
            if (malformed && calls == 13)
                result.residual.pop_back();
            if (!malformed && calls == 17)
                result.objective = original;
            return result;
        };
        angular::Qualification qualification;
        if (malformed)
        {
            try
            {
                (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
                FAIL() << "Malformed optional probe was accepted.";
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::InvalidInput);
            }
            EXPECT_EQ(calls, 13u);
            EXPECT_EQ(current.objective, original);
        }
        else
        {
            EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
                angular::IterationResult::Advanced);
            EXPECT_EQ(calls, 22u);
            EXPECT_LT(current.objective, original);
        }
    }
}

TEST(PolyhedralReconstruction, AngularIndefiniteOptionalCurvatureUsesQualifiedGaussNewtonSearch)
{
    namespace angular = PLIC::angular_detail;
    AngularVector normal = angular_unit({1, 1e-5, 1e-5});
    unsigned calls = 0;
    const auto evaluate = [&](AngularVector candidate)
    {
        ++calls;
        const double x = candidate[1] / candidate[0], y = candidate[2] / candidate[0];
        AngularEvaluation value{{x, y, -1 + .5 * (x + y) * (x + y)}};
        for (const double r : value.residual)
            value.objective += r * r;
        value.objective_upper = value.objective;
        return value;
    };
    auto current = evaluate(normal);
    const auto objective = current.objective;
    calls = 0;
    angular::Qualification qualification;
    EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
        angular::IterationResult::Advanced);
    EXPECT_EQ(calls, 21u); // Twelve axial, four mixed, GN trial and its qualification.
    EXPECT_LT(current.objective, objective);
    EXPECT_GT(normal[1], 1e-5);
    EXPECT_GT(normal[2], 1e-5);
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
    EXPECT_EQ(evaluations, 10u);
    EXPECT_EQ(current.stamp, 6u);
    EXPECT_LT(current.objective, .04);
    EXPECT_GT(normal[1], .09);
    EXPECT_LT(normal[1], .11); // The half step, not the failed full step.
}

TEST(PolyhedralReconstruction, AngularImprovingUnobservableTrialBacktracksAndFatalProbePropagates)
{
    namespace angular = PLIC::angular_detail;
    for (const bool fatal : {false, true})
    {
        AngularVector normal{1, 0, 0};
        auto current = angular_model(normal, .2, 0, 0);
        unsigned evaluations = 0;
        const auto evaluate = [&](AngularVector candidate)
        {
            ++evaluations;
            if (evaluations >= 6 && evaluations <= 9)
            {
                if (fatal)
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "trial qualification budget");
                return AngularEvaluation{{0, 0}};
            }
            return angular_model(candidate, .2, 0, evaluations);
        };
        angular::Qualification qualification;
        if (fatal)
        {
            try
            {
                (void) angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification);
                FAIL() << "Fatal trial qualification was swallowed.";
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
            }
            EXPECT_EQ(evaluations, 6u);
            EXPECT_EQ(normal, (AngularVector{1, 0, 0}));
            EXPECT_EQ(current.stamp, 0u);
        }
        else
        {
            EXPECT_EQ(angular::advance(normal, current, 1e8, 2e-9, 2e-8, evaluate, angular_unit, qualification),
                angular::IterationResult::Advanced);
            EXPECT_EQ(evaluations, 14u);
            EXPECT_GT(normal[1], .09);
            EXPECT_LT(normal[1], .11);
            EXPECT_EQ(current.stamp, 10u);
        }
    }
}

OrdinarySample captured_sheared_sample(PLIC::GeometryPoint lower, PLIC::GeometryPoint upper, double shear,
    double gas, double carrier, std::uint64_t id)
{
    using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    std::array<PLIC::Point<ArithmeticMode::Ordinary>, 8> points;
    for (unsigned i = 0; i < 8; ++i)
    {
        const auto x = i & 1 ? upper[0] : lower[0];
        const auto y = i & 2 ? upper[1] : lower[1];
        points[i].value = {x + shear * y, y, i & 4 ? upper[2] : lower[2]};
    }
    return {Cell::make_points(points,
                {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}}),
        gas, carrier, id};
}

void record_captured_fit(const PLIC::PolyhedralInterface& fit, const PLIC::PolyhedralReconstructionOptions& options,
    const std::string& prefix = "")
{
    const auto record = [&](const std::string& name, long double value)
    {
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
        ::testing::Test::RecordProperty(prefix + name, text.str());
    };
    record("normal_x", fit.gas.normal[0]);
    record("normal_y", fit.gas.normal[1]);
    record("normal_z", fit.gas.normal[2]);
    record("stencil_rms", fit.stencil_volume_residual);
    record("fit_iterations", fit.fit_iterations);
    record("qualified_step", fit.orientation_step);
    record("stencil_condition", fit.stencil_condition);
    record("target_difference", fit.finite_target_volume_difference.value);
    record("target_error_estimate", fit.target_volume_numerical_error_estimate);
    record("clip_evaluation_cap", options.maximum_clip_evaluations);
    // Statistics are local to the calling image; they may not include work
    // performed by a separately linked FVM image.
    const auto statistics = PLIC::geometry_statistics();
    record("calling_image_geometry_volume_evaluations", statistics.volume_evaluations);
    record("calling_image_geometry_constructions", statistics.constructions);
}

TEST(PolyhedralReconstruction, CapturedClosedShearPlaneConvergesWithLargeGaussNewtonStep)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    // Independently integrated plane X + Y/256 <= 257/512 in logical
    // coordinates, with physical x = X + Y/4. All inventories are dyadic.
    const auto target = captured_sheared_sample({.5, 0, 0}, {.625, .25, .25}, .25, 3. / 32768, 253. / 32768, 4);
    const std::vector<OrdinarySample> stencil{
        captured_sheared_sample({.25, 0, 0}, {.5, .5, .5}, .25, 1. / 16, 0, 2),
        captured_sheared_sample({.5, 0, .25}, {.625, .25, .5}, .25, 3. / 32768, 253. / 32768, 5),
        captured_sheared_sample({.625, 0, 0}, {.75, .25, .25}, .25, 0, 1. / 128, 8),
        captured_sheared_sample({.5, .25, 0}, {.625, .5, .25}, .25, 1. / 32768, 255. / 32768, 16)};
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    PLIC::reset_geometry_statistics();
    const auto fit = Kernel::reconstruct(target, stencil, options);
    record_captured_fit(fit, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.fit_iterations, 24u);
    EXPECT_LE(fit.orientation_step, options.orientation_tolerance);
    EXPECT_LT(fit.stencil_volume_residual, 2e-8);
    EXPECT_NEAR(fit.gas.normal[0], .9710284210662322, 3e-8);
    EXPECT_NEAR(fit.gas.normal[1], -.2389640254967681, 3e-8);
    EXPECT_NEAR(fit.gas.normal[2], 0., 3e-8);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.gas_volume) * options.volume_tolerance);
    EXPECT_EQ(target.gas_volume, 3. / 32768);
    EXPECT_EQ(stencil[3].carrier_volume, 255. / 32768);
}

TEST(PolyhedralReconstruction, ClosedShearDyadicPlanesRetainObservableAcceptedNormals)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    for (int k = 1; k <= 7; ++k)
    {
        SCOPED_TRACE(k);
        // Exact plane X + kY/512 = .5 + k/1024, independently integrated.
        const auto target = captured_sheared_sample({.5, 0, 0}, {.625, .25, .25}, .25,
            3. * k / 65536, (512. - 3 * k) / 65536, 4);
        const std::vector<OrdinarySample> stencil{
            captured_sheared_sample({.25, 0, 0}, {.5, .5, .5}, .25, 1. / 16, 0, 2),
            captured_sheared_sample({.5, 0, .25}, {.625, .25, .5}, .25,
                3. * k / 65536, (512. - 3 * k) / 65536, 5),
            captured_sheared_sample({.625, 0, 0}, {.75, .25, .25}, .25, 0, 1. / 128, 8),
            captured_sheared_sample({.5, .25, 0}, {.625, .5, .25}, .25,
                k / 65536., (512. - k) / 65536, 16)};
        PLIC::PolyhedralReconstructionOptions options;
        options.measure_only = true;
        PLIC::reset_geometry_statistics();
        const auto fit = Kernel::reconstruct(target, stencil, options);
        record_captured_fit(fit, options, "k" + std::to_string(k) + "_");
        const auto expected = angular_unit({1, k / 512. - .25, 0});
        EXPECT_TRUE(fit.fit_converged);
        EXPECT_LE(fit.fit_iterations, options.maximum_fit_iterations);
        EXPECT_TRUE(fit.orientation_step <= options.orientation_tolerance ||
                    fit.stencil_volume_residual <= options.fit_tolerance);
        EXPECT_LT(fit.stencil_volume_residual, 2e-8);
        for (unsigned a = 0; a < 3; ++a)
            EXPECT_NEAR(fit.gas.normal[a], expected[a], 3e-8);
        EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
            static_cast<long double>(target.gas_volume) * options.volume_tolerance);
    }
}

TEST(PolyhedralReconstruction, CapturedEarlyQ3CellEightRetainsGaussNewtonForLinearizedFit)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const double shear = (.6 - .5) / .5;
    const auto target = captured_sheared_sample({.625, 0, 0}, {.75, .25, .25}, shear,
        9.007588474508093e-5, .007722424115254919, 8);
    const std::vector<OrdinarySample> stencil{
        captured_sheared_sample({.5, 0, 0}, {.625, .25, .25}, shear, .005742064867932311, .0020704351320676895, 4),
        captured_sheared_sample({.625, 0, .25}, {.75, .25, .5}, shear, 1.616359666623214e-9, .007812498383640333, 9),
        captured_sheared_sample({.625, .25, 0}, {.75, .5, .25}, shear, 0, 1. / 128, 12),
        captured_sheared_sample({.75, 0, 0}, {.875, .25, .5}, shear, 0, 1. / 64, 20)};
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    PLIC::reset_geometry_statistics();
    const auto fit = Kernel::reconstruct(target, stencil, options);
    record_captured_fit(fit, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.fit_iterations, options.maximum_fit_iterations);
    EXPECT_TRUE(fit.orientation_step <= options.orientation_tolerance ||
                fit.stencil_volume_residual <= options.fit_tolerance);
    EXPECT_LT(fit.stencil_volume_residual, 2e-9);
    EXPECT_NEAR(fit.gas.normal[0], .9806444064005432, 3e-6);
    EXPECT_NEAR(fit.gas.normal[1], .1644431456403562, 3e-6);
    EXPECT_NEAR(fit.gas.normal[2], .1062779377257141, 3e-6);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.gas_volume) * options.volume_tolerance);
}

TEST(PolyhedralReconstruction, CapturedQ3CellFourCurvedStencilConvergesWithinOriginalBudgets)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const double shear = (.6 - .5) / .5;
    const auto target = captured_sheared_sample({.5, 0, 0}, {.625, .25, .25}, shear,
        .0077713999221395, 4.110007786049872e-05, 4);
    const std::vector<OrdinarySample> stencil{
        captured_sheared_sample({.25, 0, 0}, {.5, .5, .5}, shear, .061714713926021986, .0007852860739780131, 2),
        captured_sheared_sample({.5, 0, .25}, {.625, .25, .5}, shear, .007319919423482418, .0004925805765175818, 5),
        captured_sheared_sample({.625, 0, 0}, {.75, .25, .25}, shear, .001971454065631996, .005841045934368003, 8),
        captured_sheared_sample({.5, .25, 0}, {.625, .5, .25}, shear, .004123109780909433, .003689390219090566, 16)};
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    PLIC::reset_geometry_statistics();
    const auto fit = Kernel::reconstruct(target, stencil, options);
    record_captured_fit(fit, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.fit_iterations, 24u);
    EXPECT_LE(fit.orientation_step, options.orientation_tolerance);
    EXPECT_NEAR(fit.stencil_volume_residual, .029432749204890543, 2e-9);
    EXPECT_NEAR(fit.gas.normal[0], .9850548959227791, 3e-7);
    EXPECT_NEAR(fit.gas.normal[1], .1608351886840540, 3e-7);
    EXPECT_NEAR(fit.gas.normal[2], .06163516933965175, 3e-7);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.carrier_volume) * options.volume_tolerance);
    EXPECT_EQ(target.carrier_volume, 4.110007786049872e-05);
    EXPECT_EQ(stencil[3].gas_volume, .004123109780909433);
}

TEST(PolyhedralReconstruction, CapturedLaterQ3CellFourStrongCurvatureConvergesWithinOriginalBudgets)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const double shear = (.6 - .5) / .5;
    const auto target = captured_sheared_sample({.5, 0, 0}, {.625, .25, .25}, shear,
        .007809676726198759, 2.823273801241972e-6, 4);
    const std::vector<OrdinarySample> stencil{
        captured_sheared_sample({.25, 0, 0}, {.5, .5, .5}, shear, .061671228173875846, .0008287718261241534, 2),
        captured_sheared_sample({.5, 0, .25}, {.625, .25, .5}, shear, .0075185713757188345, .00029392862428116414, 5),
        captured_sheared_sample({.625, 0, 0}, {.75, .25, .25}, shear, .0023952111507156357, .0054172888492843635, 8),
        captured_sheared_sample({.5, .25, 0}, {.625, .5, .25}, shear, .004602632779987267, .0032098672200127332, 16)};
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    PLIC::reset_geometry_statistics();
    const auto fit = Kernel::reconstruct(target, stencil, options);
    record_captured_fit(fit, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.fit_iterations, 24u);
    EXPECT_LE(fit.orientation_step, options.orientation_tolerance);
    EXPECT_NEAR(fit.stencil_volume_residual, .028726766961985669, 2e-11);
    EXPECT_NEAR(fit.gas.normal[0], .987514804442455656, 2e-7);
    EXPECT_NEAR(fit.gas.normal[1], .157488413593679930, 2e-7);
    EXPECT_NEAR(fit.gas.normal[2], .003451172369582950, 2e-7);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.carrier_volume) * options.volume_tolerance);
    EXPECT_EQ(target.carrier_volume, 2.823273801241972e-6);
    EXPECT_EQ(stencil[3].gas_volume, .004602632779987267);
}

TEST(PolyhedralReconstruction, CapturedQ3CellSeventeenRetainsSubDoubleUlpMeritDecrease)
{
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "This captured sub-double-ulp case requires a wider long double significand.";
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const double shear = (.6 - .5) / .5;
    const auto target = captured_sheared_sample({.5, .25, .25}, {.625, .5, .5}, shear,
        .003822596826635398, .003989903173364601, 17);
    const std::vector<OrdinarySample> stencil{
        captured_sheared_sample({.25, 0, 0}, {.5, .5, .5}, shear, .060042218710596756, .002457781289403253, 2),
        captured_sheared_sample({.5, 0, .25}, {.625, .25, .5}, shear, .007783506533361226, 2.899346663877486e-05, 5),
        captured_sheared_sample({.625, .25, .25}, {.75, .5, .5}, shear, .00018364609720851683, .007628853902791483, 13),
        captured_sheared_sample({.5, .25, 0}, {.625, .5, .25}, shear, .006020009765737339, .0017924902342626602, 16),
        captured_sheared_sample({.5, .25, .5}, {.625, .5, .75}, shear, .0036594846563424662, .004153015343657532, 18),
        captured_sheared_sample({.5, .5, .25}, {.625, .75, .5}, shear, 8.605053223371048e-06, .00780389494677663, 33)};
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    PLIC::reset_geometry_statistics();
    const auto fit = Kernel::reconstruct(target, stencil, options);
    record_captured_fit(fit, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.fit_iterations, options.maximum_fit_iterations);
    EXPECT_TRUE(fit.orientation_step <= options.orientation_tolerance ||
                fit.stencil_volume_residual <= options.fit_tolerance);
    EXPECT_NEAR(fit.stencil_volume_residual, .0766444845050065432, 2e-11);
    EXPECT_NEAR(fit.gas.normal[0], .944569482657690743, 2e-7);
    EXPECT_NEAR(fit.gas.normal[1], .317770032152553390, 2e-7);
    EXPECT_NEAR(fit.gas.normal[2], .082526959822518921, 2e-7);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.gas_volume) * options.volume_tolerance);
    EXPECT_EQ(target.carrier_volume, .003989903173364601);
    EXPECT_EQ(stencil[5].gas_volume, 8.605053223371048e-06);
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

TEST(PolyhedralReconstruction, AngularUnobservableQualificationRetainsProbeEvidence)
{
    namespace angular = PLIC::angular_detail;
    const AngularVector normal{1, 0, 0}, tangent{0, 1, 0}, second{0, 0, 1};
    const std::array residual{.2, .4};
    for (const bool rank_deficient : {true, false})
    {
        unsigned calls = 0;
        const auto evaluate = [&](AngularVector trial)
        {
            ++calls;
            return AngularEvaluation{{trial[1], rank_deficient ? 0. : 1e-3 * trial[2]}};
        };
        angular::Qualification evidence;
        try
        {
            (void) angular::qualify_normal(normal, tangent, second, residual, 100., evaluate, 1e-4, &evidence);
            FAIL() << "An unobservable angular system was accepted.";
        }
        catch (const angular::Unobservable&)
        {
            EXPECT_EQ(calls, 4u);
            EXPECT_EQ(evidence.probe_step, 1e-4);
            EXPECT_TRUE(std::isinf(evidence.step));
            if (rank_deficient)
                EXPECT_TRUE(std::isinf(evidence.condition));
            else
                EXPECT_NEAR(evidence.condition, 1e6, 1e-6);
        }
    }
}

TEST(PolyhedralReconstruction, FitFailureAndModelRejectionRetainDistinctCollectiveEvidence)
{
    namespace failure = SimpleFluid::detail::vof_failure;
    PLIC::PolyhedralFitDiagnostics d;
    d.normal = d.best_normal = {1, 0, 0};
    d.rms = d.best_rms = .3;
    d.angular_step = std::numeric_limits<double>::infinity();
    d.finite_difference_scale = 1e-4;
    d.condition = 1e9;
    d.iterations = 2;
    d.clip_evaluations = 31;
    d.stencil_ids = {8, 17};
    const PLIC::GeometryError cause(PLIC::GeometryErrorCode::InvalidInput, "unobservable");
    const PLIC::PolyhedralFitFailure fit(PLIC::PolyhedralFitFailure::Reason::Unobservable, cause, d);
    EXPECT_EQ(fit.code(), PLIC::GeometryErrorCode::InvalidInput);
    EXPECT_FALSE(fit.recoverable_trial());
    failure::Record record;
    failure::classify(record, std::make_exception_ptr(fit));
    EXPECT_EQ(record.category, failure::Category::Optimizer);
    EXPECT_EQ(record.geometry_code, int(PLIC::GeometryErrorCode::InvalidInput));
    EXPECT_TRUE(record.has(failure::Field::AngularFit));
    EXPECT_EQ(record.fit_normal, d.normal);
    EXPECT_EQ(record.fit_rms, d.rms);
    EXPECT_EQ(record.fit_difference_scale, d.finite_difference_scale);
    EXPECT_EQ(record.fit_condition, d.condition);
    EXPECT_EQ(record.fit_iterations, d.iterations);
    EXPECT_EQ(record.fit_clip_evaluations, d.clip_evaluations);
    EXPECT_EQ(record.fit_stencil_ids[0], 8u);
    const failure::CollectiveFailure<std::invalid_argument> formatted(record, std::make_exception_ptr(fit));
    EXPECT_NE(std::string(formatted.what()).find("stencil_ids=[8,17]"), std::string::npos);
    PLIC::PolyhedralResolutionAssessment assessment;
    assessment.reason = PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel;
    record = {};
    failure::classify(record, std::make_exception_ptr(PLIC::PolyhedralResolutionRequired(assessment, d)));
    EXPECT_EQ(record.category, failure::Category::Resolution);
    EXPECT_TRUE(record.has(failure::Field::Resolution));
    EXPECT_TRUE(record.has(failure::Field::AngularFit));
    EXPECT_EQ(record.fit_rms, d.rms);
    EXPECT_EQ(record.fit_stencil_ids[1], 17u);
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

TEST(PolyhedralReconstruction, CapturedAlgebraicCornerHasDirectionButRetainsPlaneModelRejection)
{
    namespace fixture = captured_algebraic_corner;
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    ASSERT_EQ(fixture::rows.front().gid, 106u);
    std::vector<OrdinarySample> samples;
    for (const auto& row : fixture::rows)
    {
        samples.push_back(fixture::fit_sample(row));
        EXPECT_EQ(samples.back().geometry->volume().value, 0x1p-9L);
        EXPECT_EQ(row.ring, samples.size() == 1 ? 0u : samples.size() <= 6 ? 1u : 2u);
    }
    const auto original_samples = samples;
    const auto physical_inventories = fixture::rows;
    const auto& target = samples.front();
    const std::span<const OrdinarySample> stencil(samples.data() + 1, samples.size() - 1);
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true; // GeometricFace's candidate reconstruction policy.
    ASSERT_EQ(options.maximum_fit_iterations, 24u);
    ASSERT_EQ(options.maximum_clip_evaluations, 32768u);
    ASSERT_EQ(options.maximum_model_residual, .2);
    const auto assessment = Kernel::assess_resolution(target, stencil, options);
    ASSERT_TRUE(assessment.admitted());
    EXPECT_EQ(assessment.stencil_cells, 17u);
    EXPECT_TRUE(assessment.angular_fit_eligible);
    EXPECT_GE(assessment.gas_support_fraction, options.minimum_phase_support_fraction);
    EXPECT_GE(assessment.carrier_support_fraction, options.minimum_phase_support_fraction);
    EXPECT_GT(assessment.directional_strength, assessment.directional_uncertainty);
    EXPECT_GT(assessment.stencil_condition, 1);
    EXPECT_LT(assessment.stencil_condition, 2);
    EXPECT_GE(std::cbrt(static_cast<long double>(std::min(target.gas_volume, target.carrier_volume))) /
                  std::cbrt(target.geometry->volume().value),
        options.minimum_optimized_phase_length_ratio);
    // The capture distinguishes physical inventories from geometric fit labels:
    // the target's independently stored dominant amount differs by one ulp.
    EXPECT_EQ(target.gas_volume, fixture::rows.front().physical_gas);
    EXPECT_NE(target.carrier_volume, fixture::rows.front().physical_liquid);

    const auto offset = Kernel::solve_offset(target, fixture::diagnostic_normal, options);
    EXPECT_EQ(offset.orientation_method, PLIC::PolyhedralOrientationMethod::SuppliedNormal);
    EXPECT_FALSE(offset.fit_converged);
    EXPECT_FALSE(offset.stencil_volume_residual_measured);
    EXPECT_LE(std::abs(offset.finite_target_volume_difference.value) + offset.target_volume_numerical_error_estimate,
        static_cast<long double>(std::min(target.gas_volume, target.carrier_volume)) * options.volume_tolerance);
    // A supported LS direction and a strict target offset do not establish a
    // usable one-plane neighborhood model. This measured witness is deliberately
    // separate from the production optimizer result and proves no global minimum.
    long double squared = 0;
    for (const auto& sample : stencil)
    {
        const auto measured = PLIC::OrdinaryPlaneMeasure(*sample.geometry).below(offset.gas);
        ASSERT_EQ(measured.status, PLIC::PlaneMeasureStatus::Resolved);
        const auto residual = (measured.volume - sample.gas_volume) / sample.geometry->volume().value;
        squared += residual * residual;
    }
    const auto diagnostic_rms = std::sqrt(squared / stencil.size());
    EXPECT_GT(diagnostic_rms, options.maximum_model_residual);
    EXPECT_NEAR(diagnostic_rms, .3695449733143106L, 2e-12L);
    // Accurate optimization of the original broad halo must remain distinct
    // from the selected first-ring result and retain the unchanged model gate.
    try
    {
        (void) Kernel::reconstruct(target, stencil, options);
        FAIL() << "The captured coarse corner must retain reconstruction rejection.";
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        EXPECT_EQ(error.assessment().reason, PLIC::PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel);
        EXPECT_NEAR(error.diagnostics().rms, .354736006697614L, 2e-12L);
        EXPECT_LE(error.diagnostics().angular_step, options.orientation_tolerance);
        EXPECT_EQ(error.diagnostics().stencil_ids.size(), 17u);
        RecordProperty("optimized_original_halo_rms", error.diagnostics().rms);
        RecordProperty("optimized_original_halo_step", error.diagnostics().angular_step);
    }
    catch (const PLIC::PolyhedralFitFailure& error)
    {
        ADD_FAILURE() << "The original halo requires accurate optimization before model rejection: " << error.what();
        EXPECT_STREQ(error.what(), "PLIC angular volume optimizer did not reach stationarity.");
        const auto& d = error.diagnostics();
        EXPECT_EQ(d.stencil_ids.size(), 17u);
        EXPECT_GT(d.rms, options.maximum_model_residual);
        RecordProperty("optimizer_rms", d.rms);
        RecordProperty("optimizer_step", d.angular_step);
        RecordProperty("optimizer_fd", d.finite_difference_scale);
        RecordProperty("optimizer_iterations", d.iterations);
        std::cout << "CAPTURED17 rms=" << std::setprecision(17) << d.rms << " step=" << d.angular_step
                  << " n=" << d.normal[0] << ',' << d.normal[1] << ',' << d.normal[2]
                  << " iterations=" << d.iterations << '\n';
    }
    catch (const PLIC::GeometryError& error)
    {
        ADD_FAILURE() << "Unexpected arithmetic/resource failure in the captured model case: " << error.what();
    }
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        EXPECT_EQ(samples[i].gas_volume, original_samples[i].gas_volume);
        EXPECT_EQ(samples[i].carrier_volume, original_samples[i].carrier_volume);
        EXPECT_EQ(fixture::rows[i].physical_gas, physical_inventories[i].physical_gas);
        EXPECT_EQ(fixture::rows[i].physical_liquid, physical_inventories[i].physical_liquid);
    }
}

TEST(PolyhedralReconstruction, CapturedSinglePlaneTargetUsesCompleteFiveFaceNeighborSupport)
{
    namespace fixture = captured_algebraic_corner;
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const auto target = fixture::fit_sample(fixture::rows.front());
    EXPECT_NEAR(target.gas_volume / target.geometry->volume().value, .4535127712971186L, 1e-16L);
    EXPECT_NEAR(fixture::physical_fraction(fixture::rows.front()), .4535127712971186L, 3.4e-16L);
    std::vector<OrdinarySample> stencil;
    for (const auto& row : fixture::rows)
        if (row.ring == 1)
            stencil.push_back(fixture::fit_sample(row));
    ASSERT_EQ(stencil.size(), 5u);
    for (unsigned i = 0; i < 5; ++i)
        EXPECT_EQ(stencil[i].id, (std::array<std::uint64_t, 5>{74, 104, 107, 108, 138})[i]);
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    const auto geometry = Kernel::assess_stencil_geometry(target, stencil, options);
    ASSERT_TRUE(geometry.admitted());
    EXPECT_NEAR(geometry.stencil_condition, 2., 2e-15);
    const auto transport = Kernel::assess_transport_support(target, stencil, options);
    EXPECT_TRUE(transport.admitted());
    EXPECT_EQ(transport.angular_support_cells, 4u);
    EXPECT_LT(transport.angular_support_condition, 2);
    const auto gradient = Kernel::solve_offset(target, {-.9262719001, -.3768558970, 0}, options);
    const auto physical = Kernel::solve_offset(target, {-std::cos(.37), -std::sin(.37), 0}, options);
    const auto rms = [&](const PLIC::GeometryHalfspace& plane)
    {
        long double sum = 0;
        for (const auto& sample : stencil)
        {
            const auto residual = (PLIC::OrdinaryPlaneMeasure(*sample.geometry).below(plane).volume -
                                     sample.gas_volume) / sample.geometry->volume().value;
            sum += residual * residual;
        }
        return std::sqrt(sum / stencil.size());
    };
    EXPECT_NEAR(rms(gradient.gas), .1953453825474781L, 2e-11L);
    EXPECT_NEAR(rms(physical.gas), .1888730906986648L, 2e-12L);
    const auto fit = Kernel::reconstruct(target, stencil, options);
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_LE(fit.orientation_step, options.orientation_tolerance);
    EXPECT_LE(fit.stencil_volume_residual, options.maximum_model_residual);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        static_cast<long double>(target.gas_volume) * options.volume_tolerance);
    RecordProperty("fitted_rms", fit.stencil_volume_residual);
    RecordProperty("fitted_step", fit.orientation_step);
    auto capped = options;
    capped.maximum_fit_iterations = 1;
    try
    {
        (void) Kernel::reconstruct(target, stencil, capped);
        FAIL() << "A nonstationary first iterate cannot be accepted.";
    }
    catch (const PLIC::PolyhedralFitFailure& error)
    {
        EXPECT_EQ(error.reason(), PLIC::PolyhedralFitFailure::Reason::IterationLimit);
        const auto& d = error.diagnostics();
        EXPECT_EQ(d.iterations, 1u);
        EXPECT_EQ(d.stencil_ids.size(), 5u);
        EXPECT_GT(d.clip_evaluations, 0u);
        EXPECT_LE(d.clip_evaluations, capped.maximum_clip_evaluations);
        EXPECT_GT(d.angular_step, capped.orientation_tolerance);
        EXPECT_GT(d.finite_difference_scale, 0);
        EXPECT_EQ(d.normal, d.best_normal);
        EXPECT_EQ(d.rms, d.best_rms);
    }
    bool saw_optimizer_budget = false;
    for (std::size_t budget = 16; budget <= 8192 && !saw_optimizer_budget; budget *= 2)
    {
        auto limited = options;
        limited.maximum_clip_evaluations = budget;
        try
        {
            (void) Kernel::reconstruct(target, stencil, limited);
        }
        catch (const PLIC::PolyhedralFitFailure& error)
        {
            ASSERT_EQ(error.reason(), PLIC::PolyhedralFitFailure::Reason::Geometry);
            ASSERT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
            EXPECT_FALSE(error.recoverable_trial());
            const auto& d = error.diagnostics();
            EXPECT_GE(d.iterations, 1u);
            EXPECT_EQ(d.clip_evaluations, budget + 1);
            EXPECT_EQ(d.stencil_ids.size(), 5u);
            EXPECT_EQ(d.normal, d.best_normal);
            EXPECT_EQ(d.rms, d.best_rms);
            EXPECT_TRUE(std::isfinite(d.rms));
            SimpleFluid::detail::vof_failure::Record record;
            SimpleFluid::detail::vof_failure::classify(record, std::current_exception());
            EXPECT_EQ(record.category, SimpleFluid::detail::vof_failure::Category::ResourceLimit);
            EXPECT_EQ(record.geometry_code, int(PLIC::GeometryErrorCode::ResourceLimit));
            EXPECT_TRUE(record.has(SimpleFluid::detail::vof_failure::Field::AngularFit));
            EXPECT_EQ(record.fit_clip_evaluations, budget + 1);
            EXPECT_EQ(record.fit_stencil_ids[0], stencil[0].id);
            saw_optimizer_budget = true;
        }
        catch (const PLIC::GeometryError& error)
        {
            ASSERT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        }
    }
    EXPECT_TRUE(saw_optimizer_budget);
}

TEST(PolyhedralReconstruction, TransportAngularSupportExpandsBeforeResidualOnSkewCornerPlane)
{
    using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
    // Independently defined local cells reproduce the Q0 target8 phase amounts
    // and one-mixed-neighbor pattern, with the right neighbor coarse in z.
    // Logical x+.37*y+.11*z<=.025, physical x=X+.2*Y. The target
    // and its left neighbor are mixed, while the other first-ring cells are
    // saturated. An independent positive-part cubic integrates the phases.
    const auto sample = [](PLIC::GeometryPoint lower, PLIC::GeometryPoint upper, std::uint64_t id)
    {
        std::array<PLIC::Point<ArithmeticMode::Ordinary>, 8> points;
        for (unsigned i = 0; i < 8; ++i)
        {
            const auto x = i & 1 ? upper[0] : lower[0];
            const auto y = i & 2 ? upper[1] : lower[1];
            points[i].value = {x + .2 * y, y, i & 4 ? upper[2] : lower[2]};
        }
        const auto geometry = Cell::make_points(points,
            {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}});
        constexpr std::array<long double, 3> normal{1, .37L, .11L};
        long double threshold = .025L, span = 0, volume = 1;
        for (unsigned a = 0; a < 3; ++a)
        {
            threshold -= normal[a] * lower[a];
            span += normal[a] * (upper[a] - lower[a]);
            volume *= upper[a] - lower[a];
        }
        long double gas = 0;
        if (threshold >= span)
            gas = volume;
        else if (threshold > 0)
        {
            for (unsigned corner = 0; corner < 8; ++corner)
            {
                auto distance = threshold;
                unsigned parity = 0;
                for (unsigned a = 0; a < 3; ++a)
                    if (corner & (1u << a))
                    {
                        distance -= normal[a] * (upper[a] - lower[a]);
                        ++parity;
                    }
                if (distance > 0)
                    gas += (parity % 2 ? -1 : 1) * distance * distance * distance;
            }
            gas /= 6 * normal[0] * normal[1] * normal[2];
        }
        return OrdinarySample{geometry, double(gas), double(volume - gas), id};
    };
    const auto target = sample({0, 0, 0}, {.125, .25, .25}, 8);
    std::vector<OrdinarySample> stencil{
        sample({-.125, 0, 0}, {0, .25, .25}, 4), sample({0, 0, .25}, {.125, .25, .5}, 9),
        sample({0, .25, 0}, {.125, .5, .25}, 12), sample({.125, 0, 0}, {.25, .25, .5}, 20)};
    EXPECT_NEAR(target.gas_volume, 6.3984438984439161e-5, 2e-18);
    EXPECT_NEAR(stencil.front().gas_volume, .005561015561015563, 2e-17);
    const auto centroid = Kernel::assess_stencil_geometry(target, stencil);
    ASSERT_TRUE(centroid.admitted());
    EXPECT_NEAR(centroid.stencil_condition, 2.833076923076922, 2e-14);
    PLIC::reset_geometry_statistics();
    const auto first = Kernel::assess_transport_support(target, stencil);
    EXPECT_EQ(first.reason, Reason::RankDeficientAngularSupport);
    EXPECT_TRUE(first.angular_fit_eligible);
    EXPECT_EQ(first.angular_support_cells, 1u);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    EXPECT_EQ(PLIC::geometry_statistics().intersections, 0u);
    // Add every second-ring neighbor of this local four-face configuration.
    for (const auto& [lower, upper] : std::vector<std::pair<PLIC::GeometryPoint, PLIC::GeometryPoint>>{
             {{-.25, 0, 0}, {-.125, .25, .25}}, {{-.125, .25, 0}, {0, .5, .25}},
             {{-.125, 0, .25}, {0, .25, .5}}, {{0, .25, .25}, {.125, .5, .5}},
             {{.125, 0, .5}, {.25, .25, 1}}, {{0, 0, .5}, {.125, .25, .75}},
             {{.125, .25, 0}, {.25, .5, .5}}, {{0, .5, 0}, {.125, .75, .25}},
             {{.25, 0, 0}, {.375, .25, .5}}})
        stencil.push_back(sample(lower, upper, 100 + stencil.size()));
    const auto expanded = Kernel::assess_transport_support(target, stencil);
    ASSERT_TRUE(expanded.admitted());
    EXPECT_EQ(expanded.angular_support_cells, 3u);
    EXPECT_LT(expanded.angular_support_condition, 10);
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    const auto fit = Kernel::reconstruct(target, stencil, options);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    EXPECT_LE(fit.stencil_volume_residual, options.fit_tolerance);
    EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        target.gas_volume * options.volume_tolerance);
}

TEST(PolyhedralReconstruction, TransportAngularSupportRejectsIllConditionedMixedDirectionsBeforeClipping)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .5);
    std::vector<OrdinarySample> stencil;
    for (const auto index : std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, 1}})
        stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>(index, 1, index[0] == 1 ? 0 : 1));
    for (const PLIC::GeometryPoint shift : {PLIC::GeometryPoint{0, 1, 0}, {0, 2, 1e-6}})
        stencil.push_back({Cell::box(shift, {shift[0] + 1, shift[1] + 1, shift[2] + 1}), .5, .5});
    ASSERT_TRUE(Kernel::assess_stencil_geometry(target, stencil).admitted());
    PLIC::reset_geometry_statistics();
    const auto support = Kernel::assess_transport_support(target, stencil);
    EXPECT_EQ(support.reason, PLIC::PolyhedralResolutionAssessment::Reason::IllConditionedAngularSupport);
    EXPECT_EQ(support.angular_support_cells, 2u);
    EXPECT_GT(support.angular_support_condition, PLIC::PolyhedralReconstructionOptions{}.maximum_condition);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    EXPECT_EQ(PLIC::geometry_statistics().intersections, 0u);
}

TEST(PolyhedralReconstruction, TransportAngularSupportDoesNotInventMixedGeometryFromIndependentPureTraces)
{
    using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
    const auto target = uniform_sample<ArithmeticMode::Ordinary>({0, 0, 0}, 1, .5);
    std::vector<OrdinarySample> stencil;
    for (const auto index : std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, 1}})
    {
        auto sample = uniform_sample<ArithmeticMode::Ordinary>(index, 1, index[0] == 1 ? 0 : 1);
        if (index[2] == 1)
        {
            sample.carrier_volume = 1e-25;
            sample.observed_mixed = false; // Validated pure-gas geometry with independent carrier inventory.
        }
        stencil.push_back(sample);
    }
    stencil.push_back(uniform_sample<ArithmeticMode::Ordinary>({0, 1, 0}, 1, .5));
    const auto support = Kernel::assess_transport_support(target, stencil);
    EXPECT_EQ(support.reason, PLIC::PolyhedralResolutionAssessment::Reason::RankDeficientAngularSupport);
    EXPECT_EQ(support.angular_support_cells, 1u);
    EXPECT_EQ(stencil[3].carrier_volume, 1e-25);
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
