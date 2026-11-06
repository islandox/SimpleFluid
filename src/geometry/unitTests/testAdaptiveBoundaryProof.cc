/** Exact physical-face witnesses across distinct periodic support ancestry. */
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto mode = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 8 * 1024 * 1024;
struct Case
{
    P::ConvexGeometryPtr<mode> source = P::ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1});
    std::vector<D::SupportFace> pieces;
    Case()
    {
        for (std::size_t face = 0; face < source->num_faces(); ++face)
            pieces.push_back(source->face_geometry(face));
    }
};
std::vector<A::AdaptiveBytes> encoded(const std::vector<D::SupportFace>& faces)
{
    std::vector<A::AdaptiveBytes> result;
    for (const auto& face : faces)
        result.push_back(face->encode_incidence());
    return result;
}
Case split_image_case()
{
    Case value;
    for (std::size_t face = 0; face < value.pieces.size(); ++face)
    {
        const auto& points = value.pieces[face]->points();
        if (!std::all_of(points.begin(), points.end(), [](const auto& p) { return p[2] == 0; }))
            continue;
        const auto normal = value.pieces[face]->area_vector();
        std::array<D::SupportFace, 2> parts;
        for (unsigned i = 0; i < 2; ++i)
        {
            const std::array<P::GeometryPoint, 4> loop{{{.5 * i, 0, 0}, {.5 * (i + 1), 0, 0},
                {.5 * (i + 1), 1, 0}, {.5 * i, 1, 0}}};
            parts[i] = P::PlanarFace<mode>::make(loop)
                           ->oriented_like({normal.x, normal.y, normal.z})
                           ->translated(P::GeometryPoint{0, 0, 4})
                           ->translated(P::GeometryPoint{0, 0, -4});
        }
        value.pieces[face] = parts[0];
        value.pieces.push_back(parts[1]);
        return value;
    }
    throw std::logic_error("Box lower face missing.");
}
TEST(AdaptiveBoundaryProof, PeriodicImagesRetainIdentityAndExactCoarseFineCoverage)
{
    auto value = split_image_case();
    const auto before = encoded(value.pieces);
    D::AdaptiveBoundaryProofStatistics stats;
    EXPECT_NO_THROW(D::certify_adaptive_physical_boundary(value.source, value.pieces, ample, &stats));
    EXPECT_EQ(stats.source_faces, 6u);
    EXPECT_EQ(stats.pieces, 7u);
    EXPECT_GT(stats.peak_bytes, 0u);
    EXPECT_EQ(encoded(value.pieces), before);
    const auto original = value.source->face_geometry(0);
    const auto roundtrip = original->translated(P::GeometryPoint{0, 0, 4})->translated(P::GeometryPoint{0, 0, -4});
    EXPECT_EQ(original->points(), roundtrip->points());
    EXPECT_NE(P::planar_support<mode>(*original), P::planar_support<mode>(*roundtrip));
}
TEST(AdaptiveBoundaryProof, MissingDuplicateInwardAndShiftedPiecesReject)
{
    for (unsigned defect = 0; defect < 4; ++defect)
    {
        auto value = split_image_case();
        if (defect == 0)
            value.pieces.pop_back();
        else if (defect == 1)
            value.pieces.push_back(value.pieces.back());
        else if (defect == 2)
            value.pieces.back() = value.pieces.back()->reversed();
        else
            value.pieces.back() = value.pieces.back()->translated(P::GeometryPoint{0, 0, .125});
        const auto before = encoded(value.pieces);
        EXPECT_THROW(D::certify_adaptive_physical_boundary(value.source, value.pieces, ample), std::exception);
        EXPECT_EQ(encoded(value.pieces), before);
    }
}
TEST(AdaptiveBoundaryProof, EqualSupportCannotAuthorizeFoldedOffPlanePieces)
{
    const auto source = P::ConvexCellGeometry<mode>::box({0, 0, -1}, {2, 1, 0});
    std::vector<D::SupportFace> pieces, folded;
    D::SupportFace top;
    for (std::size_t face = 0; face < source->num_faces(); ++face)
    {
        auto value = source->face_geometry(face);
        if (std::all_of(value->points().begin(), value->points().end(), [](const auto& point) { return point[2] == 0; }))
            top = value;
        else
            pieces.push_back(std::move(value));
    }
    ASSERT_TRUE(top);
    const auto frame = P::capture_incidence_frame(*top);
    for (unsigned square = 0; square < 2; ++square)
    {
        const long double x = square, height = square ? -1.L / 16 : 1.L / 16;
        using WidePoint = P::PlanarFace<mode>::WidePoint;
        const std::array<WidePoint, 4> corners{
            WidePoint{x, 0, 0}, WidePoint{x + 1, 0, 0}, WidePoint{x + 1, 1, 0}, WidePoint{x, 1, 0}};
        for (unsigned edge = 0; edge < 4; ++edge)
        {
            const std::array<P::PointIdentity<mode>, 3> triangle{
                P::details::GeometryIdentityAccess::point(corners[edge]),
                P::details::GeometryIdentityAccess::point(corners[(edge + 1) % 4]),
                P::details::GeometryIdentityAccess::point({x + .5L, .5L, height})};
            auto piece = P::restore_incidence_loop(frame, triangle);
            EXPECT_EQ(P::planar_support<mode>(*piece), P::planar_support<mode>(*top));
            folded.push_back(piece);
            pieces.push_back(std::move(piece));
        }
    }
    // Projected coverage and cancelling first moments cannot establish the
    // physical plane: the standalone native partition shortcut admits this.
    ASSERT_NO_THROW(top->certify_partition(folded));
    const auto before = encoded(pieces);
    EXPECT_THROW(D::certify_adaptive_physical_boundary(source, pieces, ample), std::exception);
    EXPECT_EQ(encoded(pieces), before);
}

TEST(AdaptiveBoundaryProof, ExactAdditionalProofBudgetRejectsOneByteShortAndRetries)
{
    const auto run = [](std::size_t cap)
    {
        auto value = split_image_case();
        D::AdaptiveBoundaryProofStatistics stats;
        D::certify_adaptive_physical_boundary(value.source, value.pieces, cap, &stats);
        return stats.peak_bytes;
    };
    const auto required = run(ample);
    ASSERT_GT(required, 0u);
    EXPECT_EQ(run(required), required);
    EXPECT_THROW((void) run(required - 1), std::exception);
    EXPECT_EQ(run(required), required);
}
} // namespace
