/** Captured N16 explicit method selection and unchanged admission gates. */
#include "FVM/details/GeneralVoF/PlaneOrientation.hh"
#include "plic_native_gradient_selection_fixture.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid;
namespace fixture = captured_native_gradient_selection;
using Kernel = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>;
using Reason = PLIC::PolyhedralResolutionAssessment::Reason;

std::vector<fixture::Sample> samples()
{
    std::vector<fixture::Sample> result;
    for (const auto& row : fixture::rows)
        result.push_back(fixture::sample(row));
    return result;
}
PLIC::PolyhedralReconstructionOptions practical()
{
    PLIC::PolyhedralReconstructionOptions options;
    options.measure_only = true;
    options.model_policy = PLIC::PlicModelPolicy::ConservativeApproximation;
    return options;
}
// Inclusion-exclusion integrates a linear cut through an axis-aligned box.
// This uses captured corner bounds, never the production clipping kernel.
long double box_volume(const fixture::Row& row, const PLIC::GeometryHalfspace& plane)
{
    std::array<long double, 3> widths{};
    unsigned dimension = 0;
    long double minimum_projection = 0, total_width = 0, denominator = 1, box_volume = 1;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const auto extent = row.upper[axis] - row.lower[axis];
        box_volume *= extent;
        const auto coefficient = static_cast<long double>(plane.normal[axis]);
        minimum_projection +=
            coefficient * ((coefficient >= 0 ? row.lower[axis] : row.upper[axis]) - plane.anchor[axis]);
        if (coefficient != 0)
        {
            const auto width = std::abs(coefficient) * extent;
            widths[dimension++] = width;
            total_width += width;
            denominator *= width;
        }
    }
    const auto distance = static_cast<long double>(plane.offset) - minimum_projection;
    if (distance <= 0)
        return 0;
    if (distance >= total_width)
        return box_volume;
    for (unsigned factor = 2; factor <= dimension; ++factor)
        denominator *= factor;
    long double integral = 0;
    for (unsigned mask = 0; mask < (1u << dimension); ++mask)
    {
        long double remainder = distance;
        bool negative = false;
        for (unsigned axis = 0; axis < dimension; ++axis)
            if (mask & (1u << axis))
            {
                remainder -= widths[axis];
                negative = !negative;
            }
        if (remainder > 0)
        {
            long double power = 1;
            for (unsigned exponent = 0; exponent < dimension; ++exponent)
                power *= remainder;
            integral += negative ? -power : power;
        }
    }
    return box_volume * integral / denominator;
}

TEST(PolyhedralGradientSelection, CapturedCell138SelectsGradientBeforeOptimizerAndMeasuresModel)
{
    if (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Native capture requires the captured 64-bit long-double significand";
    const auto all = samples();
    const auto& target = all.front();
    const std::span<const fixture::Sample> full(all.data() + 1, all.size() - 1);
    auto options = practical();
    options.maximum_fit_iterations = 1;     // Selection never asks an angular optimizer to converge.
    options.maximum_model_residual = 1e-12; // Quality reporting cannot change the selected method.
    const auto support = Kernel::assess_transport_support(target, full, fixture::fitting_cells, options);
    ASSERT_TRUE(support.admitted());
    EXPECT_FALSE(support.angular_fit_eligible);
    EXPECT_EQ(support.angular_support_cells, 0u);
    EXPECT_TRUE(std::isinf(support.angular_support_condition));
    EXPECT_EQ(support.stencil_cells, 5u);
    EXPECT_NEAR(support.stencil_condition, 2, 1e-14);
    EXPECT_GT(support.directional_strength, support.directional_uncertainty);
    EXPECT_GT(support.gas_support_fraction, .5); // Full halo still supplies phase support.
    const auto expanded = Kernel::assess_transport_support(target, full, full.size(), options);
    ASSERT_TRUE(expanded.admitted());
    EXPECT_FALSE(expanded.angular_fit_eligible);
    EXPECT_EQ(expanded.angular_support_cells, 1u);
    EXPECT_TRUE(std::isinf(expanded.angular_support_condition));

    const auto fit = Kernel::reconstruct(target, full, fixture::fitting_cells, options);
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
    EXPECT_FALSE(fit.fit_converged);
    EXPECT_EQ(fit.fit_iterations, 0u);
    EXPECT_EQ(fit.orientation_step, 0);
    EXPECT_EQ(fit.model_policy, PLIC::PlicModelPolicy::ConservativeApproximation);
    EXPECT_TRUE(fit.stencil_volume_residual_measured);
    EXPECT_EQ(fit.model_quality, PLIC::PlicModelQuality::AboveThreshold);
    EXPECT_FALSE(fit.resolution.angular_fit_eligible);
    EXPECT_EQ(fit.resolution.angular_support_cells, 0u);
    EXPECT_EQ(fit.resolution.stencil_cells, 5u);
    EXPECT_NO_THROW(VoF::plane_wire_detail::model_metadata(fit));

    // The first ring is Cartesian: opposite x/y neighbors and one +z neighbor.
    // Its diagonal inverse-distance weighted LS system has an independent closed form.
    const auto fraction = [](const fixture::Sample& sample)
    {
        return static_cast<long double>(sample.gas_volume) /
               (static_cast<long double>(sample.gas_volume) + sample.carrier_volume);
    };
    std::array<long double, 3> normal{-(fraction(full[4]) - fraction(full[0])) / (2 * .0625L),
        -(fraction(full[3]) - fraction(full[1])) / (2 * .0625L), -(fraction(full[2]) - fraction(target)) / .5L};
    const auto norm = std::hypot(normal[0], normal[1], normal[2]);
    for (unsigned axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(fit.geometry.gas.normal[axis], normal[axis] / norm, 8e-15L);
    const auto requested = static_cast<long double>(target.gas_volume);
    const auto measured_target = box_volume(fixture::rows.front(), fit.geometry.gas);
    EXPECT_LE(std::abs(measured_target - requested), requested * options.volume_tolerance);
    EXPECT_LE(std::abs(fit.target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
        requested * options.volume_tolerance);
    long double square_sum = 0;
    for (std::size_t i = 0; i < fixture::fitting_cells; ++i)
    {
        const auto& row = fixture::rows[i + 1];
        const auto volume =
            (row.upper[0] - row.lower[0]) * (row.upper[1] - row.lower[1]) * (row.upper[2] - row.lower[2]);
        const auto residual = (box_volume(row, fit.geometry.gas) - full[i].gas_volume) / volume;
        square_sum += residual * residual;
    }
    EXPECT_NEAR(fit.stencil_volume_residual, std::sqrt(square_sum / fixture::fitting_cells), 2e-14L);
    // Reconstruction leaves both captured authoritative amounts unchanged.
    for (std::size_t i = 0; i < all.size(); ++i)
    {
        EXPECT_EQ(all[i].gas_volume, fixture::sample(fixture::rows[i]).gas_volume);
        EXPECT_EQ(all[i].carrier_volume, fixture::sample(fixture::rows[i]).carrier_volume);
    }
}

TEST(PolyhedralGradientSelection, CapturedComplementKeepsOppositeQualifiedGradient)
{
    const auto original = samples();
    auto complement = original;
    for (auto& sample : complement)
        std::swap(sample.gas_volume, sample.carrier_volume);
    const auto fit = [&](const auto& all)
    {
        const std::span<const fixture::Sample> full(all.data() + 1, all.size() - 1);
        return Kernel::reconstruct(all.front(), full, fixture::fitting_cells, practical());
    };
    const auto gas = fit(original), carrier = fit(complement);
    EXPECT_EQ(carrier.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
    EXPECT_FALSE(carrier.fit_converged);
    for (unsigned axis = 0; axis < 3; ++axis)
        EXPECT_DOUBLE_EQ(gas.geometry.gas.normal[axis], -carrier.geometry.gas.normal[axis]);
    EXPECT_DOUBLE_EQ(gas.stencil_volume_residual, carrier.stencil_volume_residual);
    auto minority = carrier.geometry.gas;
    for (auto& component : minority.normal)
        component = -component;
    minority.offset = -minority.offset;
    const auto requested = static_cast<long double>(complement.front().carrier_volume);
    EXPECT_LE(
        std::abs(box_volume(fixture::rows.front(), minority) - requested), requested * practical().volume_tolerance);
}

TEST(PolyhedralGradientSelection, CollinearMixedWitnessesKeepRankDefectDiagnostic)
{
    using Geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const fixture::Sample target{Geometry::box({0, 0, 0}, {1, 1, 1}), .25, .75};
    std::vector<fixture::Sample> stencil;
    const std::array<PLIC::GeometryPoint, 6> shifts{
        PLIC::GeometryPoint{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    for (std::size_t i = 0; i < shifts.size(); ++i)
    {
        auto upper = shifts[i];
        for (auto& coordinate : upper)
            coordinate += 1;
        const auto fraction = i == 0 ? 1. : i == 1 ? 0. : .25;
        stencil.push_back({Geometry::box(shifts[i], upper), fraction, 1 - fraction, i, i == 2 || i == 3});
    }
    auto strict = practical();
    strict.model_policy = PLIC::PlicModelPolicy::RequireModelAgreement;
    const auto rejected = Kernel::assess_transport_support(target, stencil, strict);
    EXPECT_EQ(rejected.reason, Reason::RankDeficientAngularSupport);
    EXPECT_EQ(rejected.angular_support_cells, 2u);
    EXPECT_TRUE(std::isinf(rejected.angular_support_condition));
    const auto admitted = Kernel::assess_transport_support(target, stencil, practical());
    EXPECT_TRUE(admitted.admitted());
    EXPECT_FALSE(admitted.angular_fit_eligible);
    EXPECT_EQ(admitted.angular_support_cells, 2u);
    EXPECT_TRUE(std::isinf(admitted.angular_support_condition));
    const auto fit = Kernel::reconstruct(target, stencil, practical());
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
    EXPECT_FALSE(fit.fit_converged);
    EXPECT_TRUE(std::isinf(fit.resolution.angular_support_condition));
    EXPECT_NO_THROW(VoF::plane_wire_detail::model_metadata(fit));
}

TEST(PolyhedralGradientSelection, StrictTransportStillRejectsCapturedDeficientAngularSupport)
{
    const auto all = samples();
    const std::span<const fixture::Sample> full(all.data() + 1, all.size() - 1);
    auto options = practical();
    options.model_policy = PLIC::PlicModelPolicy::RequireModelAgreement;
    for (const auto count : {fixture::fitting_cells, full.size()})
    {
        const auto support = Kernel::assess_transport_support(all.front(), full, count, options);
        EXPECT_FALSE(support.admitted());
        EXPECT_EQ(support.reason, Reason::RankDeficientAngularSupport);
        EXPECT_TRUE(support.angular_fit_eligible);
        EXPECT_EQ(support.angular_support_cells, count == fixture::fitting_cells ? 0u : 1u);
    }
}

TEST(PolyhedralGradientSelection, AvailableAngularSupportRetainsQualifiedAngularFit)
{
    using Geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const fixture::Sample target{Geometry::box({0, 0, 0}, {1, 1, 1}), .25, .75};
    std::vector<fixture::Sample> stencil;
    const std::array<PLIC::GeometryPoint, 6> shifts{
        PLIC::GeometryPoint{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    for (std::size_t i = 0; i < shifts.size(); ++i)
    {
        auto upper = shifts[i];
        for (auto& coordinate : upper)
            coordinate += 1;
        const auto fraction = i == 0 ? 1. : i == 1 ? 0. : .25;
        stencil.push_back({Geometry::box(shifts[i], upper), fraction, 1 - fraction, i, true});
    }
    const auto support = Kernel::assess_transport_support(target, stencil, practical());
    ASSERT_TRUE(support.admitted());
    EXPECT_TRUE(support.angular_fit_eligible);
    EXPECT_EQ(support.angular_support_cells, 4u);
    const auto fit = Kernel::reconstruct(target, stencil, practical());
    EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    EXPECT_TRUE(fit.fit_converged);
    EXPECT_NEAR(fit.geometry.gas.normal[0], 1, 1e-14);
    EXPECT_NEAR(fit.geometry.gas.normal[1], 0, 1e-14);
    EXPECT_NEAR(fit.geometry.gas.normal[2], 0, 1e-14);
    const fixture::Row bounds{0, .25, .75, true, {0, 0, 0}, {1, 1, 1}, {}};
    EXPECT_LE(std::abs(box_volume(bounds, fit.geometry.gas) - .25L), .25L * practical().volume_tolerance);
}

TEST(PolyhedralGradientSelection, IllConditionedAngularSupportRetainsMeasuredCondition)
{
    using Geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const fixture::Sample target{Geometry::box({0, 0, 0}, {1, 1, 1}), .5, .5};
    std::vector<fixture::Sample> stencil;
    const std::array<PLIC::GeometryPoint, 6> shifts{
        PLIC::GeometryPoint{-1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 2, 1e-6}};
    for (std::size_t i = 0; i < shifts.size(); ++i)
    {
        auto upper = shifts[i];
        for (auto& coordinate : upper)
            coordinate += 1;
        const auto fraction = i >= 4 ? .5 : i == 1 ? 0. : 1.;
        stencil.push_back({Geometry::box(shifts[i], upper), fraction, 1 - fraction, i, true});
    }
    auto strict = practical();
    strict.model_policy = PLIC::PlicModelPolicy::RequireModelAgreement;
    const auto rejected = Kernel::assess_transport_support(target, stencil, strict);
    ASSERT_EQ(rejected.reason, Reason::IllConditionedAngularSupport);
    const auto admitted = Kernel::assess_transport_support(target, stencil, practical());
    EXPECT_TRUE(admitted.admitted());
    EXPECT_FALSE(admitted.angular_fit_eligible);
    EXPECT_EQ(admitted.angular_support_cells, 2u);
    EXPECT_GT(admitted.angular_support_condition, practical().maximum_condition);
    EXPECT_EQ(admitted.angular_support_condition, rejected.angular_support_condition);
    EXPECT_LE(admitted.stencil_condition, practical().maximum_condition);
}

TEST(PolyhedralGradientSelection, PracticalPolicyStillRejectsMissingGradientAndGeometry)
{
    auto all = samples();
    for (auto& sample : all)
    {
        const auto volume = sample.geometry->volume().as_double();
        sample.gas_volume = volume * .25;
        sample.carrier_volume = volume * .75;
    }
    const std::span<const fixture::Sample> full(all.data() + 1, all.size() - 1);
    const auto support = Kernel::assess_transport_support(all.front(), full, fixture::fitting_cells, practical());
    EXPECT_EQ(support.reason, Reason::NoDirectionalSignal);
    EXPECT_FALSE(support.admitted());
    EXPECT_THROW((void) Kernel::reconstruct(all.front(), full, fixture::fitting_cells, practical()),
        PLIC::PolyhedralResolutionRequired);
    const auto insufficient = Kernel::assess_transport_support(all.front(), full, 3, practical());
    EXPECT_EQ(insufficient.reason, Reason::InsufficientStencil);
    EXPECT_THROW((void) Kernel::reconstruct(all.front(), full, 3, practical()), PLIC::PolyhedralResolutionRequired);
}

TEST(PolyhedralGradientSelection, PracticalGradientRetainsTargetAccuracyAndWorkFailure)
{
    const auto all = samples();
    const std::span<const fixture::Sample> full(all.data() + 1, all.size() - 1);
    auto options = practical();
    options.maximum_clip_evaluations = 1;
    EXPECT_THROW((void) Kernel::reconstruct(all.front(), full, fixture::fitting_cells, options), PLIC::GeometryError);
    options = practical();
    options.volume_tolerance = 1e-30;
    EXPECT_THROW((void) Kernel::reconstruct(all.front(), full, fixture::fitting_cells, options), PLIC::GeometryError);
}
} // namespace
