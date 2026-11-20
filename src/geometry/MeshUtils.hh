/**
 * @file MeshUtils.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Mesh utility functions for geometry computations (volume, area, centroid).
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "dataclass/vec3.hh"
#include "utils/debug_check.hh"

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Return the invalid sentinel for an integral identifier type.
 * @tparam ID Signed or unsigned integral identifier type.
 * @return Minus one for signed IDs, otherwise the maximum value.
 */
template <typename ID>
constexpr ID invalid_id() noexcept
{
    if constexpr (std::is_signed_v<ID>)
    {
        return static_cast<ID>(-1);
    }
    else
    {
        return std::numeric_limits<ID>::max();
    }
}


namespace detail
{

/**
 * @brief Safely convert a size_t to an ordinal type with overflow checking.
 *
 * @tparam Ordinal Target ordinal type.
 * @param value Size value to convert.
 * @param label Descriptive label used in error messages.
 * @return The value cast to the ordinal type.
 * @throws std::overflow_error if the value exceeds the ordinal type's maximum.
 */
template <class Ordinal>
inline Ordinal checked_size_to_ordinal(size_t value, std::string_view label)
{
    CHECK(value < static_cast<size_t>(std::numeric_limits<Ordinal>::max()),
          "Value for " + std::string(label)
          + " exceeds maximum representable by ordinal type.",
          std::overflow_error);

    return static_cast<Ordinal>(value);
}

} // namespace detail


namespace MeshUtils
{
/**
 * @brief Enumeration of supported cell element types.
 */
enum class CellType : uint8_t
{
    INVALID = 0,
    TETRAHEDRON = 1,
    HEXAHEDRON = 2,
    TRIPRISM = 3,
    POLYHEDRON = 4
};

/**
 * @brief Enumeration of supported face element types.
 */
enum class FaceType : uint8_t
{
    INVALID = 0,
    TRIANGLE = 1,
    QUAD = 2,
    POLYGON = 3
};

using Vec3 = vec3<real_t>;

/**
 * @brief Map a mesh cell type to its VTU cell type identifier.
 *
 * VTU convention: 12 = hexahedron, 13 = wedge/triangular prism.
 *
 * @param type Mesh cell type.
 * @return VTU cell type code.
 * @throws std::runtime_error if the cell type cannot be exported.
 */
inline int vtu_cell_type_code(CellType type)
{
    switch (type)
    {
        case CellType::HEXAHEDRON:
            return 12;
        case CellType::TRIPRISM:
            return 13;
        case CellType::POLYHEDRON:
            return 42;
        default:
            break;
    }

    throw std::runtime_error("VTU export encountered an unsupported cell type.");
}

/**
 * @brief Compute the arithmetic average of a list of points.
 *
 * @tparam Vec3 Vector type supporting addition and scalar division.
 * @param points Points to average.
 * @return Centroid of the input points.
 */
template <class Vec3>
inline Vec3 average(const std::vector<Vec3>& points)
{
    Vec3 result;
    if (points.empty())
    {
        return result;
    }

    for (const auto& point : points)
    {
        result = result + point;
    }

    return result / static_cast<typename Vec3::scalar_t>(points.size());
}

/**
 * @brief Compute the volume of a tetrahedron.
 *
 * @tparam Vec3 Vector type supporting dot and cross products.
 * @param a First tetrahedron vertex.
 * @param b Second tetrahedron vertex.
 * @param c Third tetrahedron vertex.
 * @param d Fourth tetrahedron vertex.
 * @return Signed volume magnitude of the tetrahedron.
 */
template <class Vec3>
inline real_t tetra_volume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
    return std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
}

/**
 * @brief Compute the centroid of a tetrahedron.
 *
 * @tparam Vec3 Vector type supporting addition, subtraction and scalar division.
 * @param a First tetrahedron vertex.
 * @param b Second tetrahedron vertex.
 * @param c Third tetrahedron vertex.
 * @param d Fourth tetrahedron vertex.
 * @return Exact tetrahedron volume centroid.
 */
template <class Vec3>
inline Vec3 tetra_centroid(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
    return a + ((b - a) + (c - a) + (d - a))
             / static_cast<typename Vec3::scalar_t>(4);
}

/**
 * @brief Estimate hexahedral cell volume using tetrahedral decomposition.
 *
 * Vertex coordinates are first shifted to the hexahedron's first vertex and
 * the tetrahedral fan is accumulated in long double, so meshes placed far
 * from the coordinate origin do not lose cancellation-limited low bits.
 *
 * @param Vec3 Vector type used for point coordinates.
 * @param x Hexahedral vertex coordinates.
 * @return Total hexahedral volume.
 */
template <class Vec3>
inline real_t hex_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 8);
    using long_vec3 = vec3<long double>;
    auto local = [&](size_t i)
    {
        return long_vec3{static_cast<long double>(x[i].x), static_cast<long double>(x[i].y),
            static_cast<long double>(x[i].z)}
             - long_vec3{static_cast<long double>(x[0].x), static_cast<long double>(x[0].y),
                 static_cast<long double>(x[0].z)};
    };
    constexpr std::array<std::array<size_t, 4>, 5> tetrahedra{{
        {{0, 1, 3, 4}},
        {{1, 2, 3, 6}},
        {{1, 4, 5, 6}},
        {{3, 4, 6, 7}},
        {{1, 3, 4, 6}},
    }};
    long double volume = 0.0L;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto a = local(tetrahedron[0]);
        const auto b = local(tetrahedron[1]);
        const auto c = local(tetrahedron[2]);
        const auto d = local(tetrahedron[3]);
        volume += std::abs((b - a).dot((c - a).cross(d - a))) / 6.0L;
    }
    return static_cast<real_t>(volume);
}

/**
 * @brief Compute a hexahedral volume centroid using the volume decomposition.
 *
 * The arithmetic mean of HEX_8 vertices is not the cell centroid for a
 * graded annular sector or another skewed hexahedron.  Weighting the
 * component tetrahedron centroids by their volumes gives the exact centroid
 * of a convex, planar-faced hexahedron represented by this decomposition.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Hexahedral vertex coordinates in HEX_8 order.
 * @return Volume-weighted hexahedron centroid.
 * @throws std::runtime_error If the hexahedron has zero volume.
 */
template <class Vec3>
inline Vec3 hex_centroid(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 8);
    constexpr std::array<std::array<size_t, 4>, 5> tetrahedra{{
        {{0, 1, 3, 4}},
        {{1, 2, 3, 6}},
        {{1, 4, 5, 6}},
        {{3, 4, 6, 7}},
        {{1, 3, 4, 6}},
    }};

    const auto& origin = x[0];
    Vec3 weighted_offset{};
    real_t total_volume = 0.0;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto volume =
            tetra_volume(x[tetrahedron[0]], x[tetrahedron[1]],
                         x[tetrahedron[2]], x[tetrahedron[3]]);
        const auto centroid_offset =
            ((x[tetrahedron[0]] - origin) + (x[tetrahedron[1]] - origin)
           + (x[tetrahedron[2]] - origin) + (x[tetrahedron[3]] - origin))
          / static_cast<typename Vec3::scalar_t>(4);
        weighted_offset = weighted_offset + centroid_offset * volume;
        total_volume += volume;
    }
    if (total_volume <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-volume hexahedron.");
    }
    return origin + weighted_offset / total_volume;
}

/**
 * @brief Estimate wedge cell volume using tetrahedral decomposition.
 *
 * Vertex coordinates are first shifted to the wedge's first vertex and the
 * fan is accumulated in long double, mirroring @ref hex_volume.
 *
 * @param Vec3 Vector type used for point coordinates.
 * @param x Wedge vertex coordinates.
 * @return Total wedge volume.
 */
template <class Vec3>
inline real_t wedge_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 6);
    using long_vec3 = vec3<long double>;
    auto local = [&](size_t i)
    {
        return long_vec3{static_cast<long double>(x[i].x), static_cast<long double>(x[i].y),
            static_cast<long double>(x[i].z)}
             - long_vec3{static_cast<long double>(x[0].x), static_cast<long double>(x[0].y),
                 static_cast<long double>(x[0].z)};
    };
    constexpr std::array<std::array<size_t, 4>, 3> tetrahedra{{
        {{0, 1, 2, 3}},
        {{1, 2, 4, 3}},
        {{2, 4, 5, 3}},
    }};
    long double volume = 0.0L;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto a = local(tetrahedron[0]);
        const auto b = local(tetrahedron[1]);
        const auto c = local(tetrahedron[2]);
        const auto d = local(tetrahedron[3]);
        volume += std::abs((b - a).dot((c - a).cross(d - a))) / 6.0L;
    }
    return static_cast<real_t>(volume);
}

/**
 * @brief Compute a wedge volume centroid using the volume decomposition.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Wedge vertex coordinates in WEDGE_6 order.
 * @return Volume-weighted wedge centroid.
 * @throws std::runtime_error If the wedge has zero volume.
 */
template <class Vec3>
inline Vec3 wedge_centroid(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 6);
    constexpr std::array<std::array<size_t, 4>, 3> tetrahedra{{
        {{0, 1, 2, 3}},
        {{1, 2, 4, 3}},
        {{2, 4, 5, 3}},
    }};

    const auto& origin = x[0];
    Vec3 weighted_offset{};
    real_t total_volume = 0.0;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto volume =
            tetra_volume(x[tetrahedron[0]], x[tetrahedron[1]],
                         x[tetrahedron[2]], x[tetrahedron[3]]);
        const auto centroid_offset =
            ((x[tetrahedron[0]] - origin) + (x[tetrahedron[1]] - origin)
           + (x[tetrahedron[2]] - origin) + (x[tetrahedron[3]] - origin))
          / static_cast<typename Vec3::scalar_t>(4);
        weighted_offset = weighted_offset + centroid_offset * volume;
        total_volume += volume;
    }
    if (total_volume <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-volume wedge.");
    }
    return origin + weighted_offset / total_volume;
}

/**
 * @brief Compute an area-weighted centroid for a triangular or quadrilateral
 *        face.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Face vertex coordinates in traversal order.
 * @return Exact centroid for a planar triangle or quadrilateral.
 * @throws std::runtime_error If the face has zero area.
 */
template <class Vec3>
inline Vec3 face_centroid(const std::vector<Vec3>& x)
{
    CHECK(x.size() == 3 || x.size() == 4);
    if (x.size() == 3)
    {
        return x[0] + ((x[1] - x[0]) + (x[2] - x[0]))
                    / static_cast<typename Vec3::scalar_t>(3);
    }

    const auto first_offset = x[1] - x[0];
    const auto second_offset = x[2] - x[0];
    const auto third_offset = x[3] - x[0];
    const auto first_area = first_offset.cross(second_offset).norm() * 0.5;
    const auto second_area = second_offset.cross(third_offset).norm() * 0.5;
    const auto total_area = first_area + second_area;
    if (total_area <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-area face.");
    }

    const auto first_centroid_offset =
        (first_offset + second_offset)
      / static_cast<typename Vec3::scalar_t>(3);
    const auto second_centroid_offset =
        (second_offset + third_offset)
      / static_cast<typename Vec3::scalar_t>(3);
    return x[0] + (first_centroid_offset * first_area
                 + second_centroid_offset * second_area)
                / total_area;
}

/**
 * @brief Compute the oriented area vector for a face.
 *
 * The fan is anchored at the first vertex (a local origin) and accumulated
 * in long double before one final cast, keeping far-from-origin meshes free
 * of cancellation-limited roundoff in the stored face geometry.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Face vertex coordinates in order.
 * @return Area vector of the face.
 */
template <class Vec3>
inline Vec3 face_area_vector(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 3);
    vec3<long double> area{};
    for (size_t i = 1; i + 1 < x.size(); ++i)
    {
        const vec3<long double> a{x[i].x - x[0].x, x[i].y - x[0].y, x[i].z - x[0].z};
        const vec3<long double> b{x[i + 1].x - x[0].x, x[i + 1].y - x[0].y, x[i + 1].z - x[0].z};
        area = area + a.cross(b) * (0.5L);
    }
    return {static_cast<real_t>(area.x), static_cast<real_t>(area.y), static_cast<real_t>(area.z)};
}

/**
 * @brief Signed-fan centroid of a simple planar polygon, including concavity.
 * The caller validates planarity and simplicity. Signed triangle weights are
 * essential: a fan from a concave polygon vertex can extend outside the face.
 * Moments are accumulated in long double around the first-vertex local origin.
 */
template <class Vec3>
inline Vec3 polygon_centroid(const std::vector<Vec3>& x)
{
    if (x.size() < 3)
        throw std::invalid_argument("A polygon requires at least three nodes.");
    const auto area_vector = face_area_vector(x);
    const auto area = area_vector.norm();
    if (!(area > 0.0) || !std::isfinite(area))
        throw std::invalid_argument("Cannot compute a degenerate polygon centroid.");
    const auto normal = area_vector / area;
    vec3<long double> moment{};
    for (size_t i = 1; i + 1 < x.size(); ++i)
    {
        const vec3<long double> a{x[i].x - x[0].x, x[i].y - x[0].y, x[i].z - x[0].z};
        const vec3<long double> b{x[i + 1].x - x[0].x, x[i + 1].y - x[0].y, x[i + 1].z - x[0].z};
        const auto weight = a.cross(b).dot(vec3<long double>{normal.x, normal.y, normal.z}) * (0.5L);
        moment = moment + (a + b) * (weight / 3.0L);
    }
    return x[0] + Vec3{static_cast<real_t>(moment.x / area), static_cast<real_t>(moment.y / area),
                       static_cast<real_t>(moment.z / area)};
}

/**
 * @brief Compute consecutive differences in an ordered value array.
 * @tparam T Arithmetic value type.
 * @param arr Input values.
 * @return Array containing `arr[i + 1] - arr[i]`, or empty for fewer than
 *         two values.
 */
template <class T>
inline Arr<T> consec_diff(const Arr<T>& arr)
{
    if (arr.size() < 2)
    {
        return {};
    }

    Arr<T> result(arr.size() - 1);
    for (size_t i = 0; i + 1 < arr.size(); ++i)
    {
        result[i] = arr[i + 1] - arr[i];
    }
    return result;
}

/**
 * @brief Compute midpoints of consecutive values in an ordered array.
 *
 * Uses std::midpoint: the difference is taken before halving, so a large
 * coordinate offset shared by both edges cannot contaminate the stored
 * centroid, and the operation cannot overflow.
 * @tparam T Arithmetic value type.
 * @param arr Input values.
 * @return Array containing consecutive arithmetic midpoints, or empty for
 *         fewer than two values.
 */
template <class T>
inline Arr<T> consec_mid(const Arr<T>& arr)
{
    if (arr.size() < 2)
    {
        return {};
    }

    Arr<T> result(arr.size() - 1);
    for (size_t i = 0; i + 1 < arr.size(); ++i)
    {
        result[i] = std::midpoint(arr[i], arr[i + 1]);
    }
    return result;
}

} // namespace MeshUtils

} // namespace SimpleFluid
