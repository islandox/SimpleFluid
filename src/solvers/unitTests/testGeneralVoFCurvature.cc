/** @file testGeneralVoFCurvature.cc
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
} // namespace
