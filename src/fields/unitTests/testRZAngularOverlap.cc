#include <gtest/gtest.h>

#include "fields/details/RZAngularOverlap.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace
{
namespace Geometry = SimpleFluid::mesh_transfer_detail;
using Point = Geometry::PolygonPoint;
using Polygon = std::vector<Point>;
constexpr double pi = std::numbers::pi;
}

TEST(RZAngularOverlapTest, RectangleMatchesNativeVolumeAndIndependentCylindricalAnnulus)
{
    const Polygon rectangle{{1, 0}, {2, 0}, {2, 3}, {1, 3}};
    const double half = pi / 6;
    const double native = 4.5 * 2 * std::tan(half);
    EXPECT_NEAR(Geometry::rz_angular_volume(rectangle, -half, half), native, 2e-14);
    EXPECT_NEAR(Geometry::rz_angular_sector_volume(rectangle, -half, half,
        0, 10, 0, 2 * pi, 0, 3), native, 2e-14);
    // These circular radii lie strictly between both planar radial faces for
    // the entire angular interval, so the usual cylindrical volume is exact.
    const double annulus = 0.5 * (1.8 * 1.8 - 1.2 * 1.2) * (2 * half) * 3;
    EXPECT_NEAR(Geometry::rz_angular_sector_volume(rectangle, -half, half,
        1.2, 1.8, 0, 2 * pi, 0, 3), annulus, 2e-14);
}

TEST(RZAngularOverlapTest, RectangularRZCellAgreesWithExistingXYCircleIntersection)
{
    const Polygon rectangle{{0.4, -0.3}, {0.8, -0.3}, {0.8, 0.7}, {0.4, 0.7}};
    const double half = 0.23;
    const Polygon xy{{0.4, -0.4 * std::tan(half)}, {0.8, -0.8 * std::tan(half)},
                     {0.8, 0.8 * std::tan(half)}, {0.4, 0.4 * std::tan(half)}};
    for (const auto [inner, outer] : {std::pair{0.2, 0.41}, {0.41, 0.65}, {0.7, 0.81}, {0.81, 0.9}})
    {
        SCOPED_TRACE(inner);
        SCOPED_TRACE(outer);
        const auto independent = Geometry::polygon_sector_area(xy, inner, outer, 0, 2 * pi) * 0.6;
        const auto actual = Geometry::rz_angular_sector_volume(rectangle, -half, half,
            inner, outer, 0, 2 * pi, -0.1, 0.5);
        EXPECT_NEAR(actual, independent, 3e-14);
    }
}

TEST(RZAngularOverlapTest, TriangleZClippingCreatesTheRequiredNewRadialBreakpoints)
{
    const Polygon triangle{{1, 0}, {3, 0}, {1, 2}};
    const double half = 0.19;
    // Integral_z=.5..1.5 Integral_a=1..(3-z) a da dz = 37/24.
    const double expected = 2 * std::tan(half) * 37.0 / 24.0;
    EXPECT_NEAR(Geometry::rz_angular_sector_volume(triangle, -half, half,
        0, 10, 0, 2 * pi, 0.5, 1.5), expected, 3e-14);
    const double whole = Geometry::rz_angular_sector_volume(triangle, -half, half,
        1.1, 2.3, -0.12, 0.17, 0.5, 1.5);
    const double split_z = Geometry::rz_angular_sector_volume(triangle, -half, half,
        1.1, 2.3, -0.12, 0.17, 0.5, 0.9)
        + Geometry::rz_angular_sector_volume(triangle, -half, half,
            1.1, 2.3, -0.12, 0.17, 0.9, 1.5);
    EXPECT_GT(whole, 0);
    EXPECT_NEAR(whole, split_z, 4e-14);
}

TEST(RZAngularOverlapTest, MiteredQuadrilateralsPartitionWithoutVolumeNormalization)
{
    const Polygon rectangle{{1, 0}, {3, 0}, {3, 2}, {1, 2}};
    const Polygon lower{{1, 0}, {3, 0}, {3, 1.5}, {1, 0.5}};
    const Polygon upper{{1, 0.5}, {3, 1.5}, {3, 2}, {1, 2}};
    const auto volume = [](const Polygon& polygon)
    {
        return Geometry::rz_angular_sector_volume(polygon, 2 * pi - 0.21, 2 * pi + 0.21,
            1.1, 2.6, -0.13, 0.17, 0.4, 1.6);
    };
    EXPECT_GT(volume(lower), 0);
    EXPECT_GT(volume(upper), 0);
    EXPECT_NEAR(volume(lower) + volume(upper), volume(rectangle), 5e-14);
}

TEST(RZAngularOverlapTest, AngularWrapAndTargetSplitsAreAdditive)
{
    const Polygon polygon{{0.04, 0.1}, {0.0564140625, 0.1}, {0.051, 0.1164140625}, {0.04, 0.11}};
    constexpr double lo = -0.1, hi = 0.1;
    const auto volume = [&](double lower, double upper)
    {
        return Geometry::rz_angular_sector_volume(polygon, lo, hi, 0.041, 0.055,
            lower, upper, 0.102, 0.115);
    };
    const auto full = volume(0, 2 * pi);
    EXPECT_GT(full, 0);
    EXPECT_NEAR(volume(0, 0.1) + volume(2 * pi - 0.1, 2 * pi), full, 1e-18);
    EXPECT_NEAR(volume(-0.1, 0) + volume(0, 0.1), full, 1e-18);
    EXPECT_EQ(volume(0.2, 0.3), 0);
}

TEST(RZAngularOverlapTest, ConvexityAcceptsObliqueEdgesAndRejectsInvalidWinding)
{
    // On binary64 long-double targets, FMA can make an oblique edge's
    // determinant with itself negative even though the polygon is convex.
    Polygon polygon{{0.04, 0.1}, {0.0564140625, 0.1}, {0.051, 0.1164140625}, {0.04, 0.11}};
    for (size_t start = 0; start < polygon.size(); ++start)
    {
        SCOPED_TRACE(start);
        EXPECT_TRUE(Geometry::convex_polygon(polygon));
        auto reversed = polygon;
        std::reverse(reversed.begin(), reversed.end());
        EXPECT_FALSE(Geometry::convex_polygon(reversed));
        std::rotate(polygon.begin(), polygon.begin() + 1, polygon.end());
    }

    auto repeated = polygon;
    repeated.insert(repeated.end(), polygon.begin(), polygon.end());
    EXPECT_FALSE(Geometry::convex_polygon(repeated));
    const Polygon concave{{1, 0}, {3, 0}, {2, 0.5}, {3, 2}, {1, 2}};
    EXPECT_FALSE(Geometry::convex_polygon(concave));
}

TEST(RZAngularOverlapTest, RejectsUnsupportedGeometryAndReturnsZeroForEmptyCuts)
{
    const Polygon polygon{{1, 0}, {2, 0}, {2, 1}, {1, 1}};
    auto clockwise = polygon;
    std::reverse(clockwise.begin(), clockwise.end());
    EXPECT_THROW(Geometry::rz_angular_volume(clockwise, 0, 0.1), std::invalid_argument);
    EXPECT_THROW(Geometry::rz_angular_volume(polygon, 0, pi), std::invalid_argument);
    EXPECT_EQ(Geometry::rz_angular_sector_volume(polygon, 0, 0.1, 0, 10, 0, 0.1, 1, 2), 0);
    EXPECT_EQ(Geometry::rz_angular_sector_volume(polygon, 0, 0.1, 0, 0.5, 0, 0.1, 0, 1), 0);
    EXPECT_EQ(Geometry::rz_angular_sector_volume(polygon, 0, 0.1, 3, 4, 0, 0.1, 0, 1), 0);
}
