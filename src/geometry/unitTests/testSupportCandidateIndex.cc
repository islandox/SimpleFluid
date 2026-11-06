/** Conservative represented-rectangle candidates, with exhaustive kernel oracles. */
#include "geometry/mesh/AMR/SupportCandidateIndex.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>
namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto mode = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 8 * 1024 * 1024;
struct Environment
{
    std::fenv_t saved{};
    Environment()
    {
        if (std::fegetenv(&saved) || std::fesetround(FE_TONEAREST))
            throw std::runtime_error("Cannot select test arithmetic environment.");
    }
    ~Environment() { (void) std::fesetenv(&saved); }
};
D::SupportOccurrence rectangle(std::size_t key, double x0, double y0, double x1, double y1, double z = 0)
{
    const std::array<P::GeometryPoint, 4> points{{{x0, y0, z}, {x1, y0, z}, {x1, y1, z}, {x0, y1, z}}};
    D::SupportOccurrence result;
    result.cell = {0, A::ConvexCellAncestry{key, {}}};
    result.domain = {'d'};
    result.rank = 0;
    result.orientation = 1;
    result.face = P::PlanarFace<mode>::make(points);
    return result;
}
std::vector<D::SupportOccurrence> strips(std::size_t count = 32, double offset = 0)
{
    std::vector<D::SupportOccurrence> result;
    for (std::size_t i = 0; i < count; ++i)
        result.push_back(rectangle(i, offset + 2 * i, 0, offset + 2 * i + 1, 1));
    return result;
}
void require_profile(const D::SupportCandidateIndex& index)
{
    const auto controls = A::geometry_construction_controls();
    if (controls.supported && std::numeric_limits<long double>::digits == 64 &&
        (controls.x87_control & 0xf3f) == 0x33f && (controls.simd_control & 0xffc0) == 0x1f80)
        EXPECT_TRUE(index.certified());
}
TEST(SupportCandidateIndex, MixedLevelClosedBoundsPreserveExactKernelOutputs)
{
    Environment environment;
    std::vector<D::SupportOccurrence> entries{
        rectangle(0, 0, 0, 1, 1), rectangle(1, 0, 0, .5, .5), rectangle(2, .5, 0, 1, .5),
        rectangle(3, 0, .5, .5, 1), rectangle(4, .5, .5, 1, 1), rectangle(5, 1, 0, 2, 1),
        rectangle(6, 1, 1, 2, 2)};
    for (std::size_t i = entries.size(); i < 32; ++i)
        entries.push_back(rectangle(i, 2 * i, 0, 2 * i + 1, 1));
    D::Budget budget{ample};
    auto charge = [&](std::size_t n) { budget.add(n); };
    D::SupportCandidateIndex index(entries, ample, charge);
    require_profile(index);
    std::size_t visited = 0;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        std::vector<std::pair<std::size_t, A::AdaptiveBytes>> reference, actual;
        for (std::size_t j = 0; j < entries.size(); ++j)
            if (auto face = entries[i].face->intersection(*entries[j].face))
                reference.emplace_back(j, face->encode_incidence());
        auto visit = [&](std::size_t j)
        {
            ++visited;
            if (auto face = entries[i].face->intersection(*entries[j].face))
                actual.emplace_back(j, face->encode_incidence());
        };
        index.visit(i, entries.size(), visit);
        EXPECT_EQ(actual, reference);
    }
    if (index.certified())
        EXPECT_LT(visited, entries.size() * entries.size() / 2);
    std::vector<std::size_t> touching;
    auto visit = [&](std::size_t j) { touching.push_back(j); };
    index.visit(0, entries.size(), visit);
    for (std::size_t i = 0; i < 7; ++i)
        EXPECT_NE(std::find(touching.begin(), touching.end(), i), touching.end());
    EXPECT_TRUE(std::is_sorted(touching.begin(), touching.end()));
}
TEST(SupportCandidateIndex, UnsupportedShapesScalesAndControlsStayExhaustive)
{
    Environment environment;
    for (unsigned variant = 0; variant < 4; ++variant)
    {
        auto entries = strips();
        if (variant == 0)
        {
            const std::array<P::GeometryPoint, 3> triangle{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
            entries[0].face = P::PlanarFace<mode>::make(triangle);
        }
        else if (variant == 1)
            entries = strips(32, .1);
        else if (variant == 2)
            entries[1] = rectangle(1, 2, 0, 3, 1, .25);
        else
            ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
        D::Budget budget{ample};
        auto charge = [&](std::size_t n) { budget.add(n); };
        D::SupportCandidateIndex index(entries, ample, charge);
        EXPECT_FALSE(index.certified());
        EXPECT_EQ(budget.used, 0u);
        std::vector<std::size_t> actual(entries.size()), expected(entries.size());
        actual.clear();
        std::iota(expected.begin(), expected.end(), 0);
        auto visit = [&](std::size_t j) { actual.push_back(j); };
        index.visit(0, entries.size(), visit);
        EXPECT_EQ(actual, expected);
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    }
}
TEST(SupportCandidateIndex, ChangedVisitorControlsResumeOriginalRemainingOrder)
{
    Environment environment;
    const auto entries = strips();
    D::Budget budget{ample};
    auto charge = [&](std::size_t n) { budget.add(n); };
    D::SupportCandidateIndex index(entries, ample, charge);
    require_profile(index);
    std::vector<std::size_t> actual, expected(entries.size());
    std::iota(expected.begin(), expected.end(), 0);
    auto change = [&](std::size_t j)
    {
        actual.push_back(j);
        if (j == 0)
            EXPECT_EQ(std::fesetround(FE_UPWARD), 0);
    };
    index.visit(0, entries.size(), change);
    EXPECT_EQ(actual, expected);
    actual.clear();
    auto visit = [&](std::size_t j) { actual.push_back(j); };
    index.visit(0, entries.size(), visit);
    EXPECT_EQ(actual, expected);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    actual.clear();
    index.visit(0, entries.size(), visit);
    if (index.certified())
        EXPECT_EQ(actual, std::vector<std::size_t>{0});
}
TEST(SupportCandidateIndex, ProbePreservesEnvironmentAndIndexAdmissionIsExact)
{
    Environment environment;
    const auto entries = strips();
    (void) std::feraiseexcept(FE_INVALID);
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    std::size_t required = 0;
    {
        D::Budget budget{ample};
        auto charge = [&](std::size_t n) { budget.add(n); };
        D::SupportCandidateIndex index(entries, ample, charge);
        require_profile(index);
        required = budget.used;
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    }
    if (!required)
        return; // Unsupported platforms still exercised the exhaustive contract.
    D::Budget exact{required};
    auto exact_charge = [&](std::size_t n) { exact.add(n); };
    EXPECT_NO_THROW(D::SupportCandidateIndex(entries, ample, exact_charge));
    D::Budget short_one{required - 1};
    auto short_charge = [&](std::size_t n) { short_one.add(n); };
    EXPECT_THROW(D::SupportCandidateIndex(entries, ample, short_charge), std::invalid_argument);
    EXPECT_EQ(short_one.used, 0u);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}
} // namespace
