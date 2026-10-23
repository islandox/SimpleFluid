/** @file testExactGeneralVoFCurvature.cc
 * @brief Surface-fit kernel gates using actual clipped PLIC polygons.
 */
#include "solvers/details/GeneralVoFCurvature.hh"
#include <gtest/gtest.h>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
using Patch = detail::GeneralVoFCurvaturePatch;
} // namespace

namespace
{
std::vector<Patch> paraboloid(double bending, bool permute)
{
    std::vector<Patch> patches;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            VoF::PolyhedralInterface interface;
            interface.gas.normal = {bending * x, bending * y, 1};
            interface.gas.anchor = {double(x), double(y), -.5 * bending * (x * x + y * y)};
            auto cell = Meshes::ConvexCellGeometry::box({x - .5, y - .5, -2}, {x + .5, y + .5, 2});
            auto patch = detail::Exact::GeneralVoFCurvature::patch(patches.size(), *cell, interface);
            EXPECT_TRUE(patch.has_value());
            if (permute)
            {
                // A rigid signed coordinate permutation preserves the physical
                // section while making its target normal nonaxial in the XY fit.
                const auto p = patch->centroid, n = patch->normal;
                patch->centroid = {-p[2] + 4, p[0] - 2, p[1] + 8};
                patch->normal = {-n[2], n[0], n[1]};
                const auto spans = patch->physical_spans;
                patch->physical_spans = {spans[2], spans[0], spans[1]};
            }
            patches.push_back(*patch);
        }
    return patches;
}
TEST(ExactGeneralVoFCurvatureTest, FitsPhysicalQuadraticSectionCentroidsWithCorrectOutwardSign)
{
    for (bool permute : {false, true})
        for (double bending : {0., .05, -.05})
        {
            const auto patches = paraboloid(bending, permute);
            const auto result = detail::GeneralVoFCurvature::fit(patches[4], patches);
            EXPECT_NEAR(result.curvature, 2 * bending, 2e-12);
            EXPECT_LE(result.condition, 1e8);
            EXPECT_LT(result.normalized_residual, 1e-12);
            EXPECT_GE(result.samples, 6);
        }
}

TEST(ExactGeneralVoFCurvatureTest, RejectsDeficientFoldedAndInconsistentSurfaceSupport)
{
    auto patches = paraboloid(.05, false);
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(patches[4], std::span<const Patch>(patches.data(), 3)),
        std::runtime_error);
    auto folded = patches;
    for (auto& patch : folded)
        for (double& value : patch.normal)
            value = -value;
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(patches[4], folded), std::runtime_error);
    auto inconsistent = patches;
    inconsistent[0].centroid[2] += .3;
    detail::GeneralVoFCurvatureOptions options;
    options.maximum_normalized_residual = 1e-5;
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(patches[4], inconsistent, options), std::runtime_error);
    options = {};
    options.maximum_condition = 1;
    EXPECT_THROW((void) detail::GeneralVoFCurvature::fit(patches[4], patches, options), std::runtime_error);
}
} // namespace
