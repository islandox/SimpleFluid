/** @file testGeneralVoFCurvature.cc
 * @brief Surface-fit kernel gates using actual clipped PLIC polygons.
 */
#include "solvers/details/GeneralVoFCurvature.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
using Patch = detail::GeneralVoFCurvaturePatch;
} // namespace

namespace
{
template<ArithmeticMode Mode> void typed_physical_patch()
{
    const auto geometry = PLIC::ConvexCellGeometry<Mode>::box({2, 3, 4}, {4, 6, 8});
    PLIC::PolyhedralInterface plane;
    plane.finite_geometry = true;
    plane.gas = {{0, 0, 1}, {2, 3, 4}, 1};
    PLIC::reset_geometry_statistics();
    const auto patch = detail::GeneralVoFCurvature::patch<Mode>(17, *geometry, plane);
    ASSERT_TRUE(patch);
    EXPECT_EQ(patch->cell, 17u);
    ASSERT_TRUE(patch->key);
    EXPECT_EQ(patch->key->kind, VoF::GeneralVoFSurfaceKind::MixedCell);
    EXPECT_EQ(patch->key->identity, 17u);
    EXPECT_NEAR(patch->area, 6, 1e-13);
    EXPECT_NEAR(patch->centroid[0], 3, 1e-13);
    EXPECT_NEAR(patch->centroid[1], 4.5, 1e-13);
    EXPECT_NEAR(patch->centroid[2], 5, 1e-13);
    EXPECT_EQ(patch->normal, (PLIC::GeometryPoint{0, 0, 1}));
    EXPECT_EQ(patch->physical_spans, (PLIC::GeometryPoint{2, 3, 4}));
    EXPECT_NEAR(patch->cell_scale, 12. / 13, 1e-13);
    plane.pure_gas = true;
    EXPECT_FALSE(detail::GeneralVoFCurvature::patch<Mode>(17, *geometry, plane));
    EXPECT_GT(PLIC::geometry_statistics().volume_evaluations, 0u);
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
}

TEST(GeneralVoFCurvatureTest, TypedOrdinaryAndBoundedSectionsRetainPhysicalPatchGeometry)
{
    typed_physical_patch<ArithmeticMode::Ordinary>();
    typed_physical_patch<ArithmeticMode::BoundedCertification>();
}

TEST(GeneralVoFCurvatureTest, TangentialPhysicalSpansAdmitAnisotropicQuadraticAndExcludeFarPatches)
{
    std::vector<Patch> samples;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const double px = .8 * x, py = .4 * y;
            Patch sample;
            sample.cell = samples.size();
            sample.centroid = {px, py, -.5 * (.02 * px * px + .04 * py * py)};
            sample.normal = {.02 * px, .04 * py, 1};
            sample.area = .1;
            sample.cell_scale = .1;
            sample.physical_spans = {1, .5, .1};
            samples.push_back(sample);
        }
    const auto target = samples[4];
    const auto result = detail::GeneralVoFCurvature::fit(target, samples);
    EXPECT_NEAR(result.curvature, .06, 2e-12);
    EXPECT_LT(result.condition, 1e8);
    EXPECT_EQ(result.samples, 9u);
    auto far = samples[0];
    far.cell = 100;
    far.centroid = {3, 0, 20};
    samples.push_back(far);
    auto incoherent = samples[0];
    incoherent.cell = 101;
    incoherent.normal = {0, 0, -1};
    incoherent.centroid[2] = 20;
    samples.push_back(incoherent);
    const auto filtered = detail::GeneralVoFCurvature::fit(target, samples);
    EXPECT_NEAR(filtered.curvature, .06, 2e-12);
    EXPECT_EQ(filtered.samples, 9u);
    auto invalid = target;
    invalid.physical_spans[1] = -1;
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(invalid, samples), std::runtime_error);
}

std::vector<Patch> quadratic_patches(double y_scale = .4, bool rotate = false, double bending = .02)
{
    std::vector<Patch> samples;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const double px = .8 * x, py = y_scale * y;
            Patch sample;
            sample.cell = samples.size();
            // The exact graph has kappa(0) = bending + 2*bending.
            sample.centroid = {px, py, -.5 * bending * (px * px + 2 * py * py) - .5 * bending * px * py};
            sample.normal = {bending * px + .5 * bending * py, 2 * bending * py + .5 * bending * px, 1};
            sample.area = 1 + .1 * samples.size();
            sample.cell_scale = 1e-5;
            sample.physical_spans = {1, 1.25 * y_scale, 1e-5};
            if (rotate)
            {
                const auto p = sample.centroid, n = sample.normal, spans = sample.physical_spans;
                sample.centroid = {p[2], p[0], p[1]};
                sample.normal = {n[2], n[0], n[1]};
                sample.physical_spans = {spans[2], spans[0], spans[1]};
            }
            samples.push_back(sample);
        }
    return samples;
}

TEST(GeneralVoFCurvatureTest, ConditionedAreaWeightedFitRetainsCurvatureAndUnitsInAnisotropicFrames)
{
    for (const auto rotate : {false, true})
        for (const auto y_scale : {.4, .0008})
            for (const auto bending : {0., .02, -.02})
            {
                const auto samples = quadratic_patches(y_scale, rotate, bending);
                const auto result = detail::GeneralVoFCurvature::fit(samples[4], samples);
                EXPECT_NEAR(result.curvature, 3 * bending, 2e-9);
                EXPECT_LT(result.normalized_residual, 1e-9);
                EXPECT_LT(result.condition, 1e8);
                EXPECT_EQ(result.samples, 9u);
            }
}

TEST(GeneralVoFCurvatureTest, CanonicalPatchIdentityAdmitsMixedAndAlignedPatchesSharingCells)
{
    auto samples = quadratic_patches();
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        samples[i].cell = 12;
        samples[i].key = VoF::GeneralVoFSurfaceKey{
            i % 2 ? VoF::GeneralVoFSurfaceKind::AlignedFace : VoF::GeneralVoFSurfaceKind::MixedCell, i / 2};
    }
    const auto result = detail::GeneralVoFCurvature::fit(samples[4], samples);
    EXPECT_NEAR(result.curvature, .06, 2e-12);
    EXPECT_EQ(result.samples, 9u);
    auto duplicate = samples;
    duplicate.push_back(samples[0]);
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(samples[4], duplicate), std::runtime_error);
    duplicate = samples;
    duplicate.push_back(samples[4]);
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(samples[4], duplicate), std::runtime_error);
    auto contradictory_target = samples;
    contradictory_target[4].normal = {0, 0, -1};
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(samples[4], contradictory_target), std::runtime_error);
}

TEST(GeneralVoFCurvatureTest, RejectsRankDeficientAndContradictoryNormalsWithoutZeroFallback)
{
    auto samples = quadratic_patches();
    for (auto& sample : samples)
    {
        sample.centroid[1] = .2 * sample.centroid[0];
        sample.centroid[2] = 0;
        sample.normal = {0, 0, 1};
    }
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(samples[4], samples), std::runtime_error);
    samples = quadratic_patches();
    const auto target = samples[4];
    detail::GeneralVoFCurvatureOptions options;
    options.minimum_normal_cosine = .5;
    // The normals pass target coherence, but oppose this independently
    // prescribed graph enough to violate the fitted-normal cosine gate.
    for (auto& sample : samples)
        if (sample.cell != target.cell)
        {
            sample.centroid[2] = -.5 * .3 * sample.centroid[0] * sample.centroid[0];
            sample.normal = {sample.centroid[0] < 0 ? 1.7 : -1.7, 0, 1};
        }
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(target, samples, options), std::runtime_error);
}

void report_physical_curvature_refinement(bool sphere)
{
    constexpr double radius = 2;
    const double reference = (sphere ? 2 : 1) / radius;
    const detail::GeneralVoFCurvatureOptions controls;
    const auto report = [](const std::string& name, double value)
    {
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        ::testing::Test::RecordProperty(name, text.str());
    };
    report("radius", radius);
    report("reference_curvature", reference);
    std::optional<double> previous_error, previous_spacing;
    unsigned resolution = 0;
    for (const double relative_spacing : {.4, .2, .1})
    {
        const double h = relative_spacing * radius;
        SCOPED_TRACE(::testing::Message() << (sphere ? "sphere" : "cylinder") << " h/R=" << relative_spacing);
        std::vector<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>> geometries;
        std::vector<long double> original_volumes;
        std::vector<Patch> patches;
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
            {
                const double x = i * h, tangent = j * h;
                PLIC::GeometryPoint anchor{}, normal{}, lower{}, upper{};
                if (sphere)
                {
                    const double z = std::sqrt(radius * radius - x * x - tangent * tangent);
                    anchor = {x, tangent, z};
                    normal = {x / radius, tangent / radius, z / radius};
                    lower = {x - .5 * h, tangent - .5 * h, z - h};
                    upper = {x + .5 * h, tangent + .5 * h, z + h};
                }
                else
                {
                    const double y = std::sqrt(radius * radius - x * x);
                    anchor = {x, y, tangent};
                    normal = {x / radius, y / radius, 0};
                    lower = {x - .5 * h, y - h, tangent - .5 * h};
                    upper = {x + .5 * h, y + h, tangent + .5 * h};
                }
                const auto geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(lower, upper);
                const auto source_volume = geometry->volume().value;
                ASSERT_GT(source_volume, 0);
                PLIC::PolyhedralInterface plane;
                plane.finite_geometry = true;
                plane.gas = {normal, anchor, 0};
                // These samples are physical clipped polygons, including their
                // measured centroids/areas, rather than analytic point samples.
                const auto patch =
                    detail::GeneralVoFCurvature::patch<ArithmeticMode::Ordinary>(patches.size(), *geometry, plane);
                ASSERT_TRUE(patch);
                EXPECT_GT(patch->area, 0);
                EXPECT_TRUE(std::isfinite(patch->area));
                EXPECT_EQ(geometry->volume().value, source_volume);
                geometries.push_back(geometry);
                original_volumes.push_back(source_volume);
                patches.push_back(*patch);
            }
        const auto fitted = detail::GeneralVoFCurvature::fit(patches[4], patches, controls);
        EXPECT_TRUE(std::isfinite(fitted.curvature));
        EXPECT_GT(fitted.curvature, 0);
        EXPECT_LE(fitted.condition, controls.maximum_condition);
        EXPECT_LE(fitted.normalized_residual, controls.maximum_normalized_residual);
        EXPECT_EQ(fitted.samples, 9u);
        for (std::size_t cell = 0; cell < geometries.size(); ++cell)
            EXPECT_EQ(geometries[cell]->volume().value, original_volumes[cell]);
        const auto prefix = "resolution_" + std::to_string(resolution++) + "_";
        const double error = std::abs(fitted.curvature - reference);
        report(prefix + "h_over_radius", relative_spacing);
        report(prefix + "curvature", fitted.curvature);
        report(prefix + "absolute_error", error);
        report(prefix + "relative_error", error / reference);
        report(prefix + "condition", fitted.condition);
        report(prefix + "normalized_residual", fitted.normalized_residual);
        // Accuracy and convergence are reported observations. No new physical
        // accuracy/order threshold is introduced by this focused kernel report.
        if (previous_error && *previous_error > 0 && error > 0)
            report(prefix + "observed_order", std::log(*previous_error / error) / std::log(*previous_spacing / h));
        previous_error = error;
        previous_spacing = h;
    }
}

TEST(GeneralVoFCurvatureTest, ReportsActualClippedSphereCurvatureAcrossThreeResolutions)
{
    report_physical_curvature_refinement(true);
}

TEST(GeneralVoFCurvatureTest, ReportsActualClippedCylinderCurvatureAcrossThreeResolutions)
{
    report_physical_curvature_refinement(false);
}
} // namespace
