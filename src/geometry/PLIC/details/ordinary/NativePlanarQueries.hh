/** @file NativePlanarQueries.hh
 * @brief Shared allocation-free finite planar containment and numerical metric algorithms.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <span>
namespace SimpleFluid::PLIC::details::ordinary_planar_queries
{
namespace on = ordinary_native;
namespace predicates = ordinary_predicates;
inline long double norm(const on::WidePoint& p)
{
    const auto v = std::hypot(p[0], p[1], p[2]);
    on::finite(v);
    return v;
}
inline std::size_t axis_of(const on::WidePoint& n)
{
    std::size_t axis = 0;
    for (std::size_t i = 1; i < 3; ++i)
        if (std::abs(n[i]) > std::abs(n[axis]))
            axis = i;
    return axis;
}
struct DefiningPlane
{
    std::array<std::size_t, 3> vertices;
    unsigned projection;
    int winding;
};
inline DefiningPlane defining_plane(std::span<const on::WidePoint> p)
{
    bool unresolved = false;
    for (std::size_t i = 1; i + 1 < p.size(); ++i)
    {
        unsigned projection = 0;
        predicates::Result best;
        for (unsigned a = 0; a < 3; ++a)
        {
            const auto area = predicates::projected_orientation(p[0], p[i], p[i + 1], a);
            unresolved = unresolved || area.sign == predicates::Sign::Unresolved;
            if ((area.sign == predicates::Sign::Positive || area.sign == predicates::Sign::Negative) &&
                std::abs(area.value) > std::abs(best.value))
            {
                projection = a;
                best = area;
            }
        }
        if (best.sign == predicates::Sign::Positive || best.sign == predicates::Sign::Negative)
            return {{0, i, i + 1}, projection, predicates::require_sign(best)};
    }
    throw GeometryError(unresolved ? GeometryErrorCode::UnresolvedArithmetic : GeometryErrorCode::InvalidTopology,
        "Ordinary native face has no resolved noncollinear defining triangle.");
}
inline bool on_plane(std::span<const on::WidePoint> p, const DefiningPlane& plane, const on::WidePoint& v)
{
    // Incidence of the defining triangle is structural, independent of the
    // rounding in a separately computed normal. No proximity test is used.
    for (const auto index : plane.vertices)
        if (v == p[index])
            return true;
    const auto& t = plane.vertices;
    return predicates::require_sign(predicates::orientation({p[t[0]], p[t[1]], p[t[2]], v})) == 0;
}

inline bool point_inside(std::span<const on::WidePoint> poly, const on::WidePoint& p, bool check_plane = true)
{
    // A represented boundary vertex has structural incidence, including for
    // retained native supports. Keep this query shortcut out of on_plane:
    // polygon validation must still establish planarity of every input vertex.
    if (std::find(poly.begin(), poly.end(), p) != poly.end())
        return true;
    const auto plane = defining_plane(poly);
    if (check_plane && !on_plane(poly, plane, p))
        return false;
    for (std::size_t i = 0; i < poly.size(); ++i)
        if (plane.winding * predicates::require_sign(predicates::projected_orientation(
                                poly[i], poly[(i + 1) % poly.size()], p, plane.projection)) <
            0)
            return false;
    return true;
}

inline on::WidePoint center(std::span<const on::WidePoint> p)
{
    const auto n = on::polygon_vector(p);
    const auto axis = axis_of(n);
    on::WidePoint weighted{};
    long double total = 0;
    for (std::size_t i = 1; i + 1 < p.size(); ++i)
    {
        const auto v = on::cross(on::sub(p[i], p[0]), on::sub(p[i + 1], p[0]));
        total += v[axis];
        weighted = on::add(weighted, on::mul(on::add(on::add(p[0], p[i]), p[i + 1]), v[axis] / 3));
    }
    if (total == 0)
        on::reject("native polygon centroid has zero area.");
    return on::mul(weighted, 1 / total);
}

/** Same retained-loop numerical checks without owning a polygon or uniqueness set. */
inline void validate_retained_loop(std::span<const on::WidePoint> vertices, on::WidePoint normal)
{
    if (vertices.size() < 3 || vertices.size() > 4096 || !(norm(normal) > 0))
        on::reject("invalid retained native face support.");
    for (std::size_t index = 0; index < vertices.size(); ++index)
    {
        const auto& p = vertices[index];
        for (auto value : p)
            on::finite(value);
        for (std::size_t previous = 0; previous < index; ++previous)
            if (p == vertices[previous])
                on::reject("repeated retained native face vertex.");
    }
    const auto projected = on::polygon_vector(vertices);
    const auto axis = axis_of(normal);
    if (projected[axis] == 0)
        on::reject("retained native face has zero projected area.");
    const auto winding = projected[axis] < 0 ? -1.L : 1.L;
    for (std::size_t edge = 0; edge < vertices.size(); ++edge)
        for (const auto& p : vertices)
            if (winding * on::cross(on::sub(vertices[(edge + 1) % vertices.size()], vertices[edge]),
                              on::sub(p, vertices[edge]))[axis] <
                0)
                on::reject("nonconvex retained native face loop.");
}
inline GeometryValue<ArithmeticMode::Ordinary> area(std::span<const on::WidePoint> polygon)
{
    return {norm(on::polygon_vector(polygon))};
}
inline GeometryPoint centroid(std::span<const on::WidePoint> polygon)
{
    return on::display(center(polygon));
}
inline GeometryPoint area_vector(std::span<const on::WidePoint> polygon)
{
    return on::display(on::polygon_vector(polygon));
}
inline int orientation(std::span<const on::WidePoint> polygon)
{
    const auto normal = on::polygon_vector(polygon);
    return normal[axis_of(normal)] < 0 ? -1 : 1;
}
inline bool contains(std::span<const on::WidePoint> first, std::span<const on::WidePoint> second, bool same_support)
{
    return std::all_of(
        second.begin(), second.end(), [&](const auto& point) { return point_inside(first, point, !same_support); });
}
} // namespace SimpleFluid::PLIC::details::ordinary_planar_queries
