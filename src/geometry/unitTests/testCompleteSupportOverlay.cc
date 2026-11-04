/** Analytic complete-support partitions and stable occurrence-order oracles. */
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 8 * 1024 * 1024;

D::SupportFace rectangle(double x0, double y0, double x1, double y1, double z = 0)
{
    const std::array<P::GeometryPoint, 4> vertices{{{x0, y0, z}, {x1, y0, z}, {x1, y1, z}, {x0, y1, z}}};
    return P::PlanarFace<ordinary>::make(vertices);
}
D::SupportOccurrence occurrence(std::uint64_t root, D::SupportFace face, int orientation)
{
    D::SupportOccurrence result;
    result.cell = {0, AMR::ConvexCellAncestry{root, {}}};
    result.rank = 0;
    result.orientation = orientation;
    result.domain = {'d'};
    result.native_face = root + 100;
    result.face = orientation > 0 ? std::move(face) : face->reversed();
    return result;
}
std::vector<D::SupportOccurrence> crossed()
{
    // Cell-key ordering deliberately differs from occurrence arrival order.
    return {occurrence(40, rectangle(.5, 0, 1, 1), 1), occurrence(10, rectangle(0, .5, 1, 1), -1),
        occurrence(30, rectangle(0, 0, .5, 1), 1), occurrence(20, rectangle(0, 0, 1, .5), -1)};
}
std::vector<AMR::AdaptiveBytes> geometry_snapshot(std::span<const D::SupportOccurrence> entries)
{
    std::vector<AMR::AdaptiveBytes> result;
    for (const auto& entry : entries)
        result.push_back(entry.face->encode_incidence());
    return result;
}
struct Emission
{
    std::size_t first;
    std::optional<std::size_t> second;
    D::SupportFace face;
};
std::vector<Emission> overlay(std::span<const D::SupportOccurrence> entries, std::size_t cap = ample)
{
    D::Budget budget{cap};
    std::vector<Emission> result;
    auto emit = [&](std::size_t first, std::optional<std::size_t> second, D::SupportFace face)
    { result.push_back({first, second, std::move(face)}); };
    auto charge = [&](std::size_t bytes) { budget.add(bytes); };
    auto charge_face = [&](const D::SupportFace& face) { budget.add(face->storage_bytes()); };
    D::overlay_complete_support(entries, cap, emit, charge, charge_face);
    return result;
}

TEST(CompleteSupportOverlay, CrossedQuadrantsPreserveArrivalOrderAndExactPartition)
{
    const auto entries = crossed();
    const auto before = geometry_snapshot(entries);
    const auto result = overlay(entries);
    const std::array<std::pair<std::size_t, std::size_t>, 4> pairs{{{0, 1}, {0, 3}, {2, 1}, {2, 3}}};
    const std::array<std::array<double, 2>, 4> centers{{{.75, .75}, {.75, .25}, {.25, .75}, {.25, .25}}};
    ASSERT_EQ(result.size(), 4u);
    double total_area = 0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(result[i].first, pairs[i].first);
        EXPECT_EQ(result[i].second, std::optional<std::size_t>(pairs[i].second));
        EXPECT_DOUBLE_EQ(result[i].face->area().as_double(), .25);
        const auto center = result[i].face->centroid();
        EXPECT_DOUBLE_EQ(center.x, centers[i][0]);
        EXPECT_DOUBLE_EQ(center.y, centers[i][1]);
        EXPECT_DOUBLE_EQ(center.z, 0);
        const auto normal = result[i].face->area_vector();
        EXPECT_DOUBLE_EQ(normal.x, 0);
        EXPECT_DOUBLE_EQ(normal.y, 0);
        EXPECT_DOUBLE_EQ(normal.z, .25);
        total_area += result[i].face->area().as_double();
    }
    EXPECT_DOUBLE_EQ(total_area, 1);
    EXPECT_EQ(geometry_snapshot(entries), before);
}

TEST(CompleteSupportOverlay, MissingCoverageAndSameSideOverlapRejectWithoutChangingInputs)
{
    for (unsigned defect = 0; defect < 2; ++defect)
    {
        SCOPED_TRACE(defect);
        auto entries = crossed();
        if (!defect)
            entries.pop_back(); // The lower half of both positive members is uncovered.
        else
            entries.push_back(occurrence(99, rectangle(.25, 0, .75, 1), 1));
        const auto before = geometry_snapshot(entries);
        EXPECT_THROW((void) overlay(entries), std::invalid_argument);
        EXPECT_EQ(geometry_snapshot(entries), before);
    }
    EXPECT_EQ(overlay(crossed()).size(), 4u);
}

TEST(CompleteSupportOverlay, CompleteMembershipRequiresUniqueCellsAndConsistentDomainAndBoundary)
{
    for (unsigned defect = 0; defect < 5; ++defect)
    {
        SCOPED_TRACE(defect);
        auto entries = crossed();
        if (defect == 0)
            entries[1].cell = entries[0].cell;
        else if (defect == 1)
            entries[1].domain = {'x'};
        else if (defect == 2)
            entries[1].domain.clear();
        else if (defect == 3)
            entries[1].boundary = "wall";
        else
            entries[1].exterior = true;
        const auto before = geometry_snapshot(entries);
        EXPECT_THROW((void) overlay(entries), std::invalid_argument);
        EXPECT_EQ(geometry_snapshot(entries), before);
    }
    EXPECT_THROW((void) overlay({}), std::invalid_argument);
}

TEST(CompleteSupportOverlay, NativeWholeFacesRejectCrossedMortarsButAdmitNestedRefinement)
{
    auto entries = crossed();
    for (auto& entry : entries)
    {
        entry.native_whole = true;
        entry.whole_geometry = entry.face;
    }
    const auto before = geometry_snapshot(entries);
    EXPECT_THROW((void) overlay(entries), std::invalid_argument);
    EXPECT_EQ(geometry_snapshot(entries), before);

    entries = {occurrence(7, rectangle(0, 0, 1, 1), 1), occurrence(4, rectangle(.5, 0, 1, 1), -1),
        occurrence(9, rectangle(0, 0, .5, 1), -1)};
    for (auto& entry : entries)
    {
        entry.native_whole = true;
        entry.whole_geometry = entry.face;
    }
    const auto result = overlay(entries);
    ASSERT_EQ(result.size(), 2u);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i].first, 0u);
        EXPECT_EQ(result[i].second, std::optional<std::size_t>(i + 1));
        EXPECT_DOUBLE_EQ(result[i].face->area().as_double(), .5);
        EXPECT_DOUBLE_EQ(result[i].face->centroid().x, i == 0 ? .75 : .25);
    }
}

TEST(CompleteSupportOverlay, AuthorityImagePairKeepsPhysicalImagesAndSourceGeometry)
{
    const auto physical_low = rectangle(0, 0, 1, 1);
    const auto physical_high = rectangle(0, 0, 1, 1, 4)->reversed();
    const auto shift = P::displacement<ordinary>({0, 0, 4}, {0, 0, 0});
    auto low = occurrence(3, physical_low, 1);
    auto high = occurrence(8, physical_low, -1);
    high.face = physical_high->translated(shift);
    high.image = shift;
    const auto physical_before = physical_high->encode_incidence();
    const std::array entries{low, high};
    const auto result = overlay(entries);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().first, 0u);
    EXPECT_EQ(result.front().second, std::optional<std::size_t>(1));
    EXPECT_DOUBLE_EQ(result.front().face->area().as_double(), 1);
    EXPECT_DOUBLE_EQ(result.front().face->centroid().z, 0);
    EXPECT_EQ(entries[1].image, shift);
    EXPECT_EQ(physical_high->encode_incidence(), physical_before);
    EXPECT_DOUBLE_EQ(physical_high->centroid().z, 4);
}

TEST(CompleteSupportOverlay, FiniteTranslatedImageRejectsCollapsedAreaWithNonzeroEdges)
{
    constexpr auto digits = std::numeric_limits<long double>::digits;
    if (std::numeric_limits<long double>::radix != 2 || std::numeric_limits<double>::radix != 2 || digits < 5 ||
        digits > std::numeric_limits<long double>::max_exponent || digits > std::numeric_limits<double>::max_exponent)
        GTEST_SKIP() << "Requires binary coordinates and a unit-ULP long-double shift representable by GeometryPoint.";

    std::fenv_t saved{};
    ASSERT_EQ(std::fegetenv(&saved), 0);
    const auto original_rounding = std::fegetround();
    const auto original_flags = std::fetestexcept(FE_ALL_EXCEPT);
    {
        struct Restore
        {
            const std::fenv_t& saved;
            ~Restore() { (void) std::fesetenv(&saved); }
        } restore{saved};
        std::fenv_t held{};
        ASSERT_EQ(std::feholdexcept(&held), 0);
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        const std::array<P::GeometryPoint, 3> vertices{{{0, 0, 0}, {1, 2, 0}, {2, 4.25, 0}}};
        const auto face = P::PlanarFace<ordinary>::make(vertices);
        const auto original_points = face->points();
        const auto original_key = P::canonical_face_key(*face);
        const auto original_incidence = face->encode_incidence();
        EXPECT_DOUBLE_EQ(face->area().as_double(), .125);

        const long double shift = std::ldexp(1.L, digits - 1);
        EXPECT_EQ(std::nextafter(shift, std::numeric_limits<long double>::infinity()) - shift, 1.L);
        // Force the additions at the selected runtime rounding mode. This
        // literal coordinate oracle does not call planar translation/enrichment.
        volatile long double runtime_shift = shift;
        std::array<std::array<long double, 3>, 3> rounded{{{0, 0, 0}, {1, 2, 0}, {2, 4.25L, 0}}};
        for (auto& vertex : rounded)
            vertex[1] += runtime_shift;
        EXPECT_EQ(rounded[0][1], shift);
        EXPECT_EQ(rounded[1][1] - shift, 2.L);
        EXPECT_EQ(rounded[2][1] - shift, 4.L);
        for (std::size_t edge = 0; edge < rounded.size(); ++edge)
            EXPECT_NE(rounded[edge], rounded[(edge + 1) % rounded.size()]);
        const auto twice_area = (rounded[1][0] - rounded[0][0]) * (rounded[2][1] - rounded[0][1]) -
                                (rounded[1][1] - rounded[0][1]) * (rounded[2][0] - rounded[0][0]);
        EXPECT_EQ(twice_area, 0.L);

        const auto image = P::displacement<ordinary>({0, 0, 0}, {0, static_cast<double>(shift), 0});
        try
        {
            (void) face->translated(image);
            ADD_FAILURE() << "A finite image with zero area must reject even when every edge is nonzero.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string_view(error.what()).find("retained native face has zero projected area"),
                std::string_view::npos)
                << error.what();
        }
        EXPECT_EQ(face->points(), original_points);
        EXPECT_EQ(P::canonical_face_key(*face), original_key);
        EXPECT_EQ(face->encode_incidence(), original_incidence);
        EXPECT_DOUBLE_EQ(face->area().as_double(), .125);
    }
    EXPECT_EQ(std::fegetround(), original_rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), original_flags);
}

TEST(CompleteSupportOverlay, ExteriorOrderAndAdmissionFailureLeaveInputsReusable)
{
    std::vector entries{occurrence(9, rectangle(1, 0, 2, 1), 1), occurrence(2, rectangle(0, 0, 1, 1), 1)};
    for (auto& entry : entries)
    {
        entry.exterior = true;
        entry.boundary = "wall";
    }
    const auto before = geometry_snapshot(entries);
    EXPECT_THROW((void) overlay(entries, 1), std::invalid_argument);
    const auto result = overlay(entries);
    ASSERT_EQ(result.size(), 2u);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i].first, i);
        EXPECT_FALSE(result[i].second);
        EXPECT_EQ(result[i].face.get(), entries[i].face.get());
        EXPECT_DOUBLE_EQ(result[i].face->area().as_double(), 1);
    }
    EXPECT_EQ(geometry_snapshot(entries), before);
    entries[1].orientation = -1;
    entries[1].face = entries[1].face->reversed();
    EXPECT_THROW((void) overlay(entries), std::invalid_argument);
}
} // namespace
