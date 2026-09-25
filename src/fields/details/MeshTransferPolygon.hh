/** @file MeshTransferPolygon.hh
 * @brief Provides polygon clipping and area operations for mesh overlap.
 */
#pragma once

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

namespace SimpleFluid::mesh_transfer_detail
{

using PolygonPoint = std::array<double, 2>;
using WidePoint = std::array<long double, 2>;

inline long double cross(WidePoint a, WidePoint b)
{
    return a[0] * b[1] - a[1] * b[0];
}

inline long double dot(WidePoint a, WidePoint b)
{
    return a[0] * b[0] + a[1] * b[1];
}

inline WidePoint wide(PolygonPoint p) { return {p[0], p[1]}; }

// Translate the shoelace sum to the first vertex to avoid cancellation for
// small cells far from the origin. This is geometry, never a volume correction.
inline long double polygon_area(std::span<const PolygonPoint> polygon)
{
    if (polygon.size() < 3) return 0;
    long double twice = 0;
    for (size_t i = 1; i + 1 < polygon.size(); ++i)
        twice += cross({static_cast<long double>(polygon[i][0]) - polygon[0][0],
                        static_cast<long double>(polygon[i][1]) - polygon[0][1]},
                       {static_cast<long double>(polygon[i + 1][0]) - polygon[0][0],
                        static_cast<long double>(polygon[i + 1][1]) - polygon[0][1]});
    return twice / 2;
}

inline bool convex_polygon(std::span<const PolygonPoint> polygon)
{
    if (polygon.size() < 3 || !(polygon_area(polygon) > 0)) return false;
    // Every other point must lie on the interior side of every directed edge.
    // Checking all points also rejects self-intersecting positive-area loops.
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
            if (polygon[i] == polygon[j]) return false;
        const auto a = wide(polygon[i]), b = wide(polygon[(i + 1) % polygon.size()]);
        const WidePoint edge{b[0] - a[0], b[1] - a[1]};
        if (!(dot(edge, edge) > 0)) return false;
        for (size_t j = 0; j < polygon.size(); ++j)
        {
            // The edge's endpoints are on its supporting line by construction.
            // Fused multiply-subtract can round cross(edge, edge) below zero.
            if (j == i || j == (i + 1) % polygon.size()) continue;
            const auto vertex = polygon[j];
            const WidePoint displacement{static_cast<long double>(vertex[0]) - a[0],
                                         static_cast<long double>(vertex[1]) - a[1]};
            if (cross(edge, displacement) < 0) return false;
        }
    }
    return true;
}

inline std::array<double, 2> polygon_radial_bounds(std::span<const PolygonPoint> polygon)
{
    long double lower2 = std::numeric_limits<long double>::infinity(), upper2 = 0;
    bool contains_origin = true;
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        const auto a = wide(polygon[i]), b = wide(polygon[(i + 1) % polygon.size()]);
        const WidePoint edge{b[0] - a[0], b[1] - a[1]};
        const long double t = std::clamp(-dot(a, edge) / dot(edge, edge), 0.0L, 1.0L);
        const WidePoint closest{a[0] + t * edge[0], a[1] + t * edge[1]};
        lower2 = std::min(lower2, dot(closest, closest));
        upper2 = std::max(upper2, dot(a, a));
        contains_origin = contains_origin && cross(a, b) >= 0;
    }
    // Search bounds must enclose the represented polygon despite narrowing.
    const double lower = contains_origin ? 0 : static_cast<double>(std::sqrt(lower2));
    const double upper = static_cast<double>(std::sqrt(upper2));
    return {std::max(0.0, std::nextafter(lower, -std::numeric_limits<double>::infinity())),
            std::nextafter(upper, std::numeric_limits<double>::infinity())};
}

// Intersect a convex polygon with the half-plane cross(ray,p)*side >= 0.
inline std::vector<WidePoint> clip_ray(std::span<const WidePoint> polygon,
                                     WidePoint ray, long double side)
{
    std::vector<WidePoint> result;
    if (polygon.empty()) return result;
    result.reserve(polygon.size() + 1);
    auto a = polygon.back();
    auto da = side * cross(ray, a);
    for (const auto b : polygon)
    {
        const auto db = side * cross(ray, b);
        if ((da < 0) != (db < 0))
        {
            const auto fraction = da / (da - db);
            result.push_back({a[0] + fraction * (b[0] - a[0]),
                              a[1] + fraction * (b[1] - a[1])});
        }
        if (db >= 0) result.push_back(b);
        a = b;
        da = db;
    }
    return result;
}

// Signed intersection area of the origin-edge triangle and a centered disk.
// Split at exact line/circle roots, then integrate either the straight chord
// (inside the disk) or the circular arc (outside). No polygonal circle proxy.
inline long double disk_edge_area(WidePoint a, WidePoint b, long double radius)
{
    const WidePoint delta{b[0] - a[0], b[1] - a[1]};
    const auto aa = dot(delta, delta);
    if (!(aa > 0)) return 0;
    const auto ab = dot(a, delta), rr = radius * radius;
    const auto discriminant = ab * ab - aa * (dot(a, a) - rr);
    std::array<long double, 4> cuts{0, 1, 0, 0};
    size_t count = 2;
    if (discriminant > 0)
    {
        const auto root = std::sqrt(discriminant);
        // Stable quadratic roots avoid subtracting nearly equal values when
        // an endpoint is close to the circle. Their product is c/a.
        const auto q = -ab - std::copysign(root, ab);
        const std::array<long double, 2> roots{
            q / aa, q == 0 ? -ab / aa : (dot(a, a) - rr) / q};
        for (const auto t : roots)
            if (t > 0 && t < 1) cuts[count++] = t;
    }
    std::sort(cuts.begin(), cuts.begin() + count);
    long double result = 0;
    for (size_t i = 1; i < count; ++i)
    {
        const auto t0 = cuts[i - 1], t1 = cuts[i];
        if (!(t1 > t0)) continue;
        const WidePoint p{a[0] + t0 * delta[0], a[1] + t0 * delta[1]};
        const WidePoint q{a[0] + t1 * delta[0], a[1] + t1 * delta[1]};
        const auto middle = (t0 + t1) / 2;
        const WidePoint m{a[0] + middle * delta[0], a[1] + middle * delta[1]};
        // A tangency at the midpoint has zero discriminant and no interior
        // segment: its contribution follows the arc, not the straight chord.
        result += dot(m, m) < rr ? cross(p, q) / 2
                                 : rr * std::atan2(cross(p, q), dot(p, q)) / 2;
    }
    return result;
}

inline long double disk_polygon_area(std::span<const WidePoint> polygon,
                                     long double radius)
{
    if (!(radius > 0) || polygon.size() < 3) return 0;
    SimpleFluid::detail::CompensatedSum<long double> sum;
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        const auto term = disk_edge_area(polygon[i], polygon[(i + 1) % polygon.size()], radius);
        sum += term;
    }
    return sum.value();
}

inline double polygon_sector_area(std::span<const PolygonPoint> polygon,
                                  double inner, double outer,
                                  double angle, double width)
{
    if (!(outer > inner) || !(width > 0)) return 0;
    std::vector<WidePoint> original;
    original.reserve(polygon.size());
    for (const auto point : polygon) original.push_back(wide(point));
    long double result = 0;
    long double lower = angle, remaining = width;
    // A sector wider than pi is nonconvex. Its <=pi wedges have disjoint
    // interiors and can each be clipped with two oriented half-planes.
    while (remaining > 0)
    {
        const auto span = std::min(remaining, std::numbers::pi_v<long double>);
        const auto upper = lower + span;
        auto clipped = clip_ray(original, {std::cos(lower), std::sin(lower)}, 1);
        clipped = clip_ray(clipped, {std::cos(upper), std::sin(upper)}, -1);
        result += disk_polygon_area(clipped, outer) - disk_polygon_area(clipped, inner);
        remaining -= span;
        lower = upper;
    }
    // An empty intersection can leave a signed cancellation residual in the
    // circular edge integrals. Bound this geometry roundoff explicitly; a
    // larger negative result remains visible and fails collective validation.
    const auto roundoff = 64 * std::numeric_limits<long double>::epsilon()
        * static_cast<long double>(outer) * outer * (polygon.size() + 4);
    if (result < 0 && result >= -roundoff) result = 0;
    return static_cast<double>(result);
}

} // namespace SimpleFluid::mesh_transfer_detail
