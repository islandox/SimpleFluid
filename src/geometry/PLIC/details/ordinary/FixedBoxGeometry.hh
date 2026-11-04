/** @file FixedBoxGeometry.hh
 * @brief Shared fixed-box corner, frame and represented query algorithms.
 */
#pragma once
#include "geometry/PLIC/details/ordinary/OrdinaryIntegration.hh"
#include <sstream>
namespace SimpleFluid::PLIC::details::ordinary_kernel
{
inline constexpr std::array<std::array<std::size_t, 4>, 6> fixed_box_faces{
    {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}};
inline std::array<GeometryPoint, 8> fixed_box_corners(GeometryPoint lower, GeometryPoint upper)
{
    std::array<GeometryPoint, 8> vertices;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!(upper[axis] > lower[axis]))
            reject("box requires positive represented spans.");
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t axis = 0; axis < 3; ++axis)
            vertices[i][axis] = (i >> axis) & 1 ? upper[axis] : lower[axis];
    return vertices;
}
inline std::vector<Face> fixed_box_face_rows()
{
    std::vector<Face> rows;
    rows.reserve(fixed_box_faces.size());
    for (const auto& face : fixed_box_faces)
        rows.emplace_back(face.begin(), face.end());
    return rows;
}
inline DynamicPoint represented_source_relative(GeometryPoint vertex, GeometryPoint origin)
{
    return bounded<kernel_mode>(point<kernel_mode>(vertex) - point<kernel_mode>(origin));
}
inline Frame represented_source_frame(GeometryPoint origin, std::span<const Point> input)
{
    Frame frame;
    frame.origin = origin;
    long double extent = 0;
    for (const auto& vertex : input)
        for (const auto x : vertex)
            extent = std::max(extent, std::abs(x.value));
    if (!(extent > 0))
        reject("degenerate shell extent.");
    std::frexp(extent, &frame.exponent);
    return frame;
}
template<class Geometry>
DynamicSupport represented_support(const Geometry& poly, GeometryPoint normal, GeometryPoint anchor)
{
    if (!nonzero<kernel_mode>(point<kernel_mode>(normal)))
        reject("support requires a nonzero normal.");
    const auto offset = point<kernel_mode>(poly.frame.origin) - point<kernel_mode>(anchor);
    std::optional<Scalar> minimum, maximum;
    for (const auto& vertex : poly.vertices)
    {
        const auto projection = dot<kernel_mode>(
            point<kernel_mode>(normal), scaled<kernel_mode>(vertex.coordinates, poly.frame.exponent) + offset);
        minimum = minimum ? Scalar(std::min(minimum->value, projection.value)) : projection;
        maximum = maximum ? Scalar(std::max(maximum->value, projection.value)) : projection;
    }
    return {bounded<kernel_mode>(*minimum), bounded<kernel_mode>(*maximum)};
}
inline std::array<long double, 3> represented_world_corner(const Frame& frame, const Point& coordinates)
{
    return bounded<kernel_mode>(scaled<kernel_mode>(coordinates, frame.exponent) + point<kernel_mode>(frame.origin))
        .value;
}
inline DynamicGeometryMoments represented_moments(Integrals result, const Frame& frame, GeometryPoint origin)
{
    const auto shift = point<kernel_mode>(frame.origin) - point<kernel_mode>(origin);
    result.first = result.first + shift * result.volume;
    DynamicGeometryMoments output;
    output.origin = origin;
    output.volume = bounded<kernel_mode>(result.volume);
    for (std::size_t axis = 0; axis < 3; ++axis)
        output.first_moment[axis] = bounded<kernel_mode>(result.first[axis]);
    return output;
}
inline GeometryPoint represented_centroid(
    const DynamicGeometryMoments& moments, StorageScalar volume, GeometryPoint reference_origin)
{
    GeometryPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto coordinate =
            scalar_value<kernel_mode>(moments.first_moment[axis]) / volume + Scalar(reference_origin[axis]);
        result[axis] = static_cast<double>(bounded<kernel_mode>(coordinate).value);
        if (!std::isfinite(result[axis]))
            reject("centroid outside publication range.");
    }
    return result;
}
template<class Geometry> std::array<DynamicGeometryValue, 3> represented_axis_spans(const Geometry& poly)
{
    std::array<DynamicGeometryValue, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Scalar minimum = poly.vertices.front().coordinates[axis], maximum = minimum;
        for (const auto& vertex : poly.vertices)
        {
            const auto x = vertex.coordinates[axis];
            minimum = Scalar(std::min(minimum.value, x.value));
            maximum = Scalar(std::max(maximum.value, x.value));
        }
        result[axis] = bounded<kernel_mode>(scaled<kernel_mode>(maximum - minimum, poly.frame.exponent));
    }
    return result;
}
template<class CoordinateView>
std::array<std::array<long double, 2>, 3> represented_closed_bounds(const CoordinateView& coordinates)
{
    if (coordinates.size() == 0)
        throw GeometryError(GeometryErrorCode::InvalidTopology, "Closed bounds need a nonempty actual cell.");
    std::array<std::array<long double, 2>, 3> bounds{};
    const auto first = coordinates.at(0);
    for (std::size_t axis = 0; axis < 3; ++axis)
        bounds[axis] = {first[axis], first[axis]};
    for (std::size_t vertex = 1; vertex < coordinates.size(); ++vertex)
    {
        const auto point = coordinates.at(vertex);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            bounds[axis][0] = std::min(bounds[axis][0], point[axis]);
            bounds[axis][1] = std::max(bounds[axis][1], point[axis]);
        }
    }
    return bounds;
}
struct RepresentedSourceFaceNormal
{
    Point normal;
    std::array<std::size_t, 3> defining;
};
template<class Geometry, class FaceLoop>
RepresentedSourceFaceNormal represented_source_face_normal(const Geometry& poly, const FaceLoop& face)
{
    const auto anchor = poly.vertices[face[0]].coordinates;
    std::size_t second = 1, third = 2;
    Point normal;
    for (; third < face.size(); ++third)
    {
        normal = cross<kernel_mode>(
            poly.vertices[face[second]].coordinates - anchor, poly.vertices[face[third]].coordinates - anchor);
        if (nonzero<kernel_mode>(normal))
            break;
    }
    if (third == face.size())
        reject("face normal unresolved or degenerate.");
    return {normal, {face[0], face[second], face[third]}};
}
template<class Geometry, class FaceLoop>
void represented_source_face_winding(const Geometry& poly, const FaceLoop& face, const Point& normal)
{
    for (std::size_t i = 0; i < face.size(); ++i)
    {
        const auto& a = poly.vertices[face[i]].coordinates;
        const auto& b = poly.vertices[face[(i + 1) % face.size()]].coordinates;
        for (const auto index : face)
        {
            if (index == face[i] || index == face[(i + 1) % face.size()])
                continue;
            const auto orientation =
                dot<kernel_mode>(cross<kernel_mode>(b - a, poly.vertices[index].coordinates - a), normal);
            if (orientation.value < 0)
                reject("nonconvex face loop.");
        }
    }
}
inline void represented_positive_volume(StorageScalar volume)
{
    if (!(volume.value > 0))
    {
        std::ostringstream reason;
        reason.precision(std::numeric_limits<long double>::max_digits10);
        reason << "computed signed volume is not positive (volume=" << volume.value << ").";
        reject(reason.str());
    }
}
/** Invocation-owned fixed geometry; it has no topology/provenance/plane owners. */
struct FixedBoxEvaluation
{
    struct FixedVertex
    {
        Point coordinates;
    };
    Frame frame;
    std::array<FixedVertex, 8> vertices;
    decltype(fixed_box_faces) faces = fixed_box_faces;
    StorageScalar volume;
    FixedBoxEvaluation(GeometryPoint lower, GeometryPoint upper)
    {
        const auto corners = fixed_box_corners(lower, upper);
        std::array<Point, 8> input;
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = point<kernel_mode>(represented_source_relative(corners[i], corners.front()));
        frame = represented_source_frame(corners.front(), input);
        for (std::size_t i = 0; i < input.size(); ++i)
            vertices[i].coordinates = scaled<kernel_mode>(input[i], -frame.exponent);
    }
    std::array<long double, 3> world_corner(std::size_t i) const
    {
        return represented_world_corner(frame, vertices.at(i).coordinates);
    }
};
} // namespace SimpleFluid::PLIC::details::ordinary_kernel
