#pragma once

#include "fields/details/MeshTransferPolygon.hh"

#include <stdexcept>

namespace SimpleFluid::mesh_transfer_detail
{
namespace rz_overlap_detail
{
// Coordinates are apothem a and physical z. A physical point at angle psi
// from the cell bisector is (a, a*tan(psi), z), with Jacobian a*sec(psi)^2.
inline std::vector<WidePoint> clip_z(std::span<const WidePoint> polygon,
                                    long double bound, bool lower)
{
    std::vector<WidePoint> result;
    if (polygon.empty()) return result;
    const auto append = [&](WidePoint point)
    {
        if (result.empty() || result.back() != point) result.push_back(point);
    };
    auto previous = polygon.back();
    auto distance = [&](WidePoint point) { return lower ? point[1] - bound : bound - point[1]; };
    auto da = distance(previous);
    for (const auto current : polygon)
    {
        const auto db = distance(current);
        if ((da < 0) != (db < 0))
        {
            const auto fraction = da / (da - db);
            append({previous[0] + fraction * (current[0] - previous[0]), bound});
        }
        if (db >= 0) append(current);
        previous = current;
        da = db;
    }
    if (result.size() > 1 && result.front() == result.back()) result.pop_back();
    return result;
}

struct Primitive
{
    // F(t) = integral of a over the portion of the clipped polygon with a<=t.
    long double constant = 0, quadratic = 0, cubic = 0;
};

struct Section
{
    long double lower = 0, upper = 0;
    Primitive primitive;
};

struct Profile
{
    std::vector<long double> vertices;
    std::vector<Section> sections;
    long double moment = 0;
};

inline Profile profile(std::span<const WidePoint> polygon)
{
    Profile result;
    if (polygon.size() < 3) return result;
    for (const auto p : polygon) result.vertices.push_back(p[0]);
    std::sort(result.vertices.begin(), result.vertices.end());
    result.vertices.erase(std::unique(result.vertices.begin(), result.vertices.end()), result.vertices.end());
    long double accumulated = 0, compensation = 0;
    for (size_t interval = 1; interval < result.vertices.size(); ++interval)
    {
        const auto lo = result.vertices[interval - 1], hi = result.vertices[interval];
        if (!(hi > lo)) continue;
        const auto mid = (lo + hi) / 2;
        long double bottom = std::numeric_limits<long double>::infinity();
        long double top = -bottom, bottom_slope = 0, top_slope = 0;
        // Between vertex abscissae a convex polygon has exactly two active
        // nonvertical sides. Subtract a common Z origin before interpolation.
        const auto origin = polygon.front()[1];
        for (size_t i = 0; i < polygon.size(); ++i)
        {
            const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
            if (!(mid > std::min(a[0], b[0]) && mid < std::max(a[0], b[0]))) continue;
            const auto slope = (b[1] - a[1]) / (b[0] - a[0]);
            const auto height = (a[1] - origin) + slope * (mid - a[0]);
            if (height < bottom) { bottom = height; bottom_slope = slope; }
            if (height > top) { top = height; top_slope = slope; }
        }
        if (!std::isfinite(bottom) || !std::isfinite(top) || !(top > bottom)) continue;
        const auto slope = top_slope - bottom_slope;
        const auto height = top - bottom;
        Primitive f;
        f.cubic = slope / 3;
        f.quadratic = (height - slope * mid) / 2;
        f.constant = accumulated + compensation - f.cubic * lo * lo * lo - f.quadratic * lo * lo;
        result.sections.push_back({lo, hi, f});
        const auto width = hi - lo;
        const auto term = width * (mid * height + slope * width * width / 12);
        const auto next = accumulated + term;
        compensation += std::abs(accumulated) >= std::abs(term)
            ? (accumulated - next) + term : (term - next) + accumulated;
        accumulated = next;
    }
    result.moment = accumulated + compensation;
    return result;
}

inline Primitive at(const Profile& profile, long double threshold)
{
    if (profile.sections.empty() || threshold <= profile.sections.front().lower) return {};
    if (threshold >= profile.sections.back().upper) return {profile.moment, 0, 0};
    for (const auto& section : profile.sections)
        if (threshold < section.upper) return section.primitive;
    return {profile.moment, 0, 0};
}

inline long double integrate(const Profile& profile, long double lo, long double hi,
                             long double inner, long double outer)
{
    std::vector<long double> cuts{lo, hi};
    if (lo < 0 && hi > 0) cuts.push_back(0);
    for (const auto radius : {inner, outer})
    {
        if (!(radius > 0)) continue;
        for (const auto vertex : profile.vertices)
        {
            const auto ratio = vertex / radius;
            if (ratio < 0 || ratio > 1) continue;
            const auto angle = std::acos(ratio);
            for (const auto candidate : {-angle, angle})
                if (candidate > lo && candidate < hi) cuts.push_back(candidate);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    long double total = 0, compensation = 0;
    for (size_t i = 1; i < cuts.size(); ++i)
    {
        const auto a = cuts[i - 1], b = cuts[i];
        if (!(b > a)) continue;
        const auto mid = (a + b) / 2, width = b - a;
        const auto first = at(profile, outer * std::cos(mid));
        const auto second = at(profile, inner * std::cos(mid));
        const auto tangent_difference = std::sin(width) / (std::cos(a) * std::cos(b));
        const auto sine_difference = 2 * std::cos(mid) * std::sin(width / 2);
        const auto quadratic = first.quadratic == second.quadratic
            ? first.quadratic * (outer - inner) * (outer + inner)
            : first.quadratic * outer * outer - second.quadratic * inner * inner;
        const auto cubic = first.cubic == second.cubic
            ? first.cubic * (outer - inner) * (outer * outer + outer * inner + inner * inner)
            : first.cubic * outer * outer * outer - second.cubic * inner * inner * inner;
        const auto term = (first.constant - second.constant) * tangent_difference
            + cubic * sine_difference + quadratic * width;
        const auto next = total + term;
        compensation += std::abs(total) >= std::abs(term) ? (total - next) + term : (term - next) + total;
        total = next;
    }
    return total + compensation;
}

inline void validate(std::span<const PolygonPoint> polygon, double theta_lower, double theta_upper)
{
    if (!std::isfinite(theta_lower) || !std::isfinite(theta_upper)
        || !(theta_upper > theta_lower) || theta_upper - theta_lower >= std::numbers::pi_v<double>
        || !convex_polygon(polygon))
        throw std::invalid_argument("RZ angular overlap requires a convex CCW polygon and an angular span in (0,pi).");
    for (const auto p : polygon)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !(p[0] > 0))
            throw std::invalid_argument("RZ angular overlap requires finite coordinates and positive apothems.");
}
} // namespace rz_overlap_detail

/** Exact planar swept-cell volume from its CCW (apothem, physical-Z) polygon. */
inline long double rz_angular_volume(std::span<const PolygonPoint> polygon,
                                     double theta_lower, double theta_upper)
{
    rz_overlap_detail::validate(polygon, theta_lower, theta_upper);
    std::vector<WidePoint> points;
    for (const auto p : polygon) points.push_back(wide(p));
    return 2 * std::tan((static_cast<long double>(theta_upper) - theta_lower) / 2)
        * rz_overlap_detail::profile(points).moment;
}

/**
 * Exact overlap with a coaxial cylindrical sector and Z interval.
 * The cell polygon uses apothem along its angular bisector, not cylindrical
 * radius. Angular and circular boundaries are integrated analytically; no
 * angular quadrature, polygonal circle approximation or volume rescaling is used.
 */
inline double rz_angular_sector_volume(std::span<const PolygonPoint> polygon,
    double theta_lower, double theta_upper, double inner_radius, double outer_radius,
    double sector_lower, double sector_upper, double z_lower, double z_upper)
{
    rz_overlap_detail::validate(polygon, theta_lower, theta_upper);
    for (const auto value : {inner_radius, outer_radius, sector_lower, sector_upper, z_lower, z_upper})
        if (!std::isfinite(value)) throw std::invalid_argument("RZ overlap bounds must be finite.");
    if (inner_radius < 0) throw std::invalid_argument("RZ overlap inner radius must be nonnegative.");
    if (!(outer_radius > inner_radius) || !(sector_upper > sector_lower) || !(z_upper > z_lower)) return 0;
    constexpr long double period = 2 * std::numbers::pi_v<double>;
    const auto sector_width = static_cast<long double>(sector_upper) - sector_lower;
    if (sector_width > period + 64 * std::numeric_limits<double>::epsilon() * period)
        throw std::invalid_argument("RZ overlap cylindrical sector exceeds one revolution.");
    std::vector<WidePoint> points;
    for (const auto p : polygon) points.push_back(wide(p));
    points = rz_overlap_detail::clip_z(points, z_lower, true);
    points = rz_overlap_detail::clip_z(points, z_upper, false);
    const auto profile = rz_overlap_detail::profile(points);
    if (!(profile.moment > 0)) return 0;
    const auto half_width = (static_cast<long double>(theta_upper) - theta_lower) / 2;
    const auto center = static_cast<long double>(theta_lower) + half_width;
    const auto lower = std::remainder(static_cast<long double>(sector_lower) - center, period);
    long double volume = 0;
    for (const int image : {-1, 0, 1})
    {
        const auto lo = std::max(-half_width, lower + image * period);
        const auto hi = std::min(half_width, lower + std::min(sector_width, period) + image * period);
        if (hi > lo) volume += rz_overlap_detail::integrate(profile, lo, hi, inner_radius, outer_radius);
    }
    const auto roundoff = 256 * std::numeric_limits<long double>::epsilon()
        * profile.moment * 2 * std::tan(half_width) * (polygon.size() + 4);
    if (volume < 0 && volume >= -roundoff) volume = 0;
    return static_cast<double>(volume);
}
} // namespace SimpleFluid::mesh_transfer_detail
