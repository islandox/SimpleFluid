#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace SimpleFluid::PLIC
{
namespace
{
using Vector = std::array<long double, 3>;
constexpr auto epsilon = std::numeric_limits<long double>::epsilon();
Vector subtract(Vector a, Vector b)
{
    for (unsigned i = 0; i < 3; ++i)
        a[i] -= b[i];
    return a;
}
long double dot(Vector a, Vector b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(Vector a, Vector b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
long double length(Vector a)
{
    return std::hypot(a[0], a[1], a[2]);
}
Vector normal(const GeometryHalfspace& plane)
{
    Vector n{};
    for (unsigned i = 0; i < 3; ++i)
    {
        if (!std::isfinite(plane.normal[i]) || !std::isfinite(plane.anchor[i]))
            throw std::invalid_argument("Ordinary plane measure has nonfinite plane coordinates.");
        n[i] = plane.normal[i];
    }
    const auto norm = length(n);
    if (!(norm > 0) || !std::isfinite(plane.offset))
        throw std::invalid_argument("Ordinary plane measure requires a finite nonzero normal.");
    for (auto& x : n)
        x /= norm;
    return n;
}
// Interpolate from the retained endpoint. This preserves a small intersection
// near that endpoint instead of subtracting it from the rejected endpoint.
Vector intersection(Vector inside, Vector outside, long double di, long double dout)
{
    const auto fraction = di / (di - dout);
    for (unsigned i = 0; i < 3; ++i)
        inside[i] += fraction * (outside[i] - inside[i]);
    return inside;
}
std::vector<Vector> clipped(std::span<const Vector> polygon, Vector n, long double offset)
{
    std::vector<Vector> result;
    result.reserve(polygon.size() + 2);
    if (polygon.empty())
        return result;
    auto previous = polygon.back();
    auto dp = dot(n, previous) - offset;
    for (const auto& current : polygon)
    {
        const auto dc = dot(n, current) - offset;
        if ((dp < 0 && dc > 0) || (dp > 0 && dc < 0))
            result.push_back(
                dp < 0 ? intersection(previous, current, dp, dc) : intersection(current, previous, dc, dp));
        if (dc <= 0)
            result.push_back(current);
        previous = current;
        dp = dc;
    }
    return result;
}
Vector polygon_vector_area(std::span<const Vector> polygon)
{
    Vector area{};
    if (polygon.size() < 3)
        return area;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto term = cross(subtract(polygon[i], polygon[0]), subtract(polygon[i + 1], polygon[0]));
        for (unsigned axis = 0; axis < 3; ++axis)
            area[axis] += term[axis] / 2;
    }
    return area;
}
void check_result(long double measure, long double error)
{
    if (!std::isfinite(measure) || !std::isfinite(error))
        throw GeometryError(
            GeometryErrorCode::UnresolvedArithmetic, "Ordinary plane measure exceeds arithmetic range.");
    if (measure < 0)
        throw GeometryError(
            GeometryErrorCode::UnresolvedArithmetic, "Ordinary plane measure has negative signed volume.");
}
} // namespace
WholeFacePlaneSide OrdinaryPlaneMeasure::strict_face_side(std::span<const Point<ArithmeticMode::Ordinary>> vertices,
    const GeometryHalfspace& plane, std::array<long double, 2> offsets)
{
    if (vertices.size() < 3 || !std::isfinite(offsets[0]) || !std::isfinite(offsets[1]) || offsets[0] > offsets[1])
        throw std::invalid_argument("Ordinary strict face side requires a face and finite ordered offsets.");
    (void) normal(plane);
    for (const auto& point : vertices)
        for (const auto value : point.value)
            if (!std::isfinite(value))
                throw std::invalid_argument("Ordinary strict face side has nonfinite coordinates.");
    namespace predicates = details::ordinary_predicates;
    predicates::Point n{}, anchor{};
    for (unsigned a = 0; a < 3; ++a)
    {
        n[a] = plane.normal[a];
        anchor[a] = plane.anchor[a];
    }
    bool negative = true, positive = true;
    for (const auto& point : vertices)
        for (const auto offset : offsets)
        {
            if (!negative && !positive)
                break;
            const auto sign = predicates::plane_distance(point.value, n, anchor, offset).sign;
            negative = negative && sign == predicates::Sign::Negative;
            positive = positive && sign == predicates::Sign::Positive;
        }
    return negative ? WholeFacePlaneSide::StrictNegative
        : positive ? WholeFacePlaneSide::StrictPositive : WholeFacePlaneSide::UnknownOrContact;
}

OrdinaryPlaneMeasure::OrdinaryPlaneMeasure(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell)
    : origin_(cell.origin()), faces_(cell.faces())
{
    const auto points = cell.local_points();
    for (const auto& point : points)
        for (const auto value : point.value)
            scale_ = std::max(scale_, std::abs(value));
    if (!(scale_ > 0) || !std::isfinite(scale_))
        throw std::invalid_argument("Ordinary plane measure requires a nondegenerate finite cell.");
    vertices_.reserve(points.size());
    for (const auto& point : points)
    {
        auto vertex = point.value;
        for (auto& value : vertex)
            value /= scale_;
        vertices_.push_back(vertex);
    }
}
PlaneVolumeMeasure OrdinaryPlaneMeasure::below(const GeometryHalfspace& plane, bool section_area) const
{
    const auto n = normal(plane);
    const auto norm = std::hypot(static_cast<long double>(plane.normal[0]), static_cast<long double>(plane.normal[1]),
        static_cast<long double>(plane.normal[2]));
    const auto offset = static_cast<long double>(plane.offset) / norm / scale_;
    Vector anchor{}, reference{};
    for (unsigned i = 0; i < 3; ++i)
    {
        anchor[i] = (static_cast<long double>(plane.anchor[i]) - origin_[i]) / scale_;
        reference[i] = n[i] * offset;
    }
    PlaneVolumeMeasure result;
    long double absolute_terms = 0, boundary_area = 0, retained_depth = 0;
    Vector area_vector{};
    bool any_inside = false, any_outside = false;
    for (const auto& vertex : vertices_)
    {
        const auto d = dot(n, subtract(vertex, anchor)) - offset;
        any_inside = any_inside || d < 0;
        any_outside = any_outside || d > 0;
    }
    if (!any_inside)
        return result;
    std::vector<Vector> polygon;
    for (const auto& face : faces_)
    {
        polygon.clear();
        polygon.reserve(face.size());
        for (auto vertex : face)
            polygon.push_back(subtract(vertices_[vertex], anchor));
        const auto covered = clipped(polygon, n, offset);
        if (covered.size() < 3)
            continue;
        const auto area = polygon_vector_area(covered);
        boundary_area += length(area);
        if (section_area)
            for (unsigned axis = 0; axis < 3; ++axis)
                area_vector[axis] += area[axis];
        for (const auto& vertex : covered)
            retained_depth = std::max(retained_depth, std::abs(dot(n, vertex) - offset));
        const auto a = subtract(covered[0], reference);
        for (std::size_t i = 1; i + 1 < covered.size(); ++i)
        {
            const auto b = subtract(covered[i], reference), c = subtract(covered[i + 1], reference);
            const auto bc = cross(b, c);
            result.volume += dot(a, bc) / 6;
            for (unsigned axis = 0; axis < 3; ++axis)
                absolute_terms += std::abs(a[axis]) *
                                  (std::abs(b[(axis + 1) % 3] * c[(axis + 2) % 3]) +
                                      std::abs(b[(axis + 2) % 3] * c[(axis + 1) % 3])) /
                                  6;
        }
    }
    // Local roundoff indicator, deliberately not published as an interval.
    // Includes cancellation in integration and the retained boundary scale.
    const auto volume_scale = scale_ * scale_ * scale_;
    result.numerical_error_estimate = 128 * epsilon * (absolute_terms + boundary_area * retained_depth) * volume_scale;
    result.volume *= volume_scale;
    if (section_area && any_outside)
        result.section_area = std::abs(dot(area_vector, n)) * scale_ * scale_;
    check_result(result.volume, result.numerical_error_estimate);
    if (result.volume <= result.numerical_error_estimate)
        result.status = PlaneMeasureStatus::RoundoffLimited;
    return result;
}
PlaneAreaMeasure ordinary_plane_face_area(
    std::span<const Vector> vertices, const GeometryHalfspace& plane, GeometryPoint origin)
{
    const auto n = normal(plane);
    const auto norm = std::hypot(static_cast<long double>(plane.normal[0]), static_cast<long double>(plane.normal[1]),
        static_cast<long double>(plane.normal[2]));
    const auto offset = static_cast<long double>(plane.offset) / norm;
    std::vector<Vector> local;
    local.reserve(vertices.size());
    for (const auto& vertex : vertices)
    {
        Vector p{};
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(vertex[axis]))
                throw std::invalid_argument("Ordinary face area has nonfinite coordinates.");
            p[axis] = vertex[axis] + (static_cast<long double>(origin[axis]) - plane.anchor[axis]);
        }
        local.push_back(p);
    }
    const auto covered = clipped(local, n, offset);
    const auto area = length(polygon_vector_area(covered));
    long double absolute_products = 0;
    for (std::size_t i = 1; i + 1 < covered.size(); ++i)
    {
        const auto a = subtract(covered[i], covered[0]), b = subtract(covered[i + 1], covered[0]);
        for (unsigned axis = 0; axis < 3; ++axis)
            absolute_products +=
                std::abs(a[(axis + 1) % 3] * b[(axis + 2) % 3]) + std::abs(a[(axis + 2) % 3] * b[(axis + 1) % 3]);
    }
    const auto error = 128 * epsilon * (area + absolute_products);
    check_result(area, error);
    return {
        area, error, error > 0 && area <= error ? PlaneMeasureStatus::RoundoffLimited : PlaneMeasureStatus::Resolved};
}
} // namespace SimpleFluid::PLIC
