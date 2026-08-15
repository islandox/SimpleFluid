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
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{

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
    TRIPRISM = 3
};

/**
 * @brief Enumeration of supported face element types.
 */
enum class FaceType : uint8_t
{
    INVALID = 0,
    TRIANGLE = 1,
    QUAD = 2
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
 * @tparam Vec3 Vector type supporting addition and scalar division.
 * @param a First tetrahedron vertex.
 * @param b Second tetrahedron vertex.
 * @param c Third tetrahedron vertex.
 * @param d Fourth tetrahedron vertex.
 * @return Exact tetrahedron volume centroid.
 */
template <class Vec3>
inline Vec3 tetra_centroid(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
    return (a + b + c + d) / static_cast<typename Vec3::scalar_t>(4);
}

/**
 * @brief Estimate hexahedral cell volume using tetrahedral decomposition.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Hexahedral vertex coordinates.
 * @return Total hexahedral volume.
 */
template <class Vec3>
inline real_t hex_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 8);
    return tetra_volume(x[0], x[1], x[3], x[4])
         + tetra_volume(x[1], x[2], x[3], x[6])
         + tetra_volume(x[1], x[4], x[5], x[6])
         + tetra_volume(x[3], x[4], x[6], x[7])
         + tetra_volume(x[1], x[3], x[4], x[6]);
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

    Vec3 weighted_centroid{};
    real_t total_volume = 0.0;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto volume =
            tetra_volume(x[tetrahedron[0]], x[tetrahedron[1]],
                         x[tetrahedron[2]], x[tetrahedron[3]]);
        weighted_centroid =
            weighted_centroid
          + tetra_centroid(x[tetrahedron[0]], x[tetrahedron[1]],
                           x[tetrahedron[2]], x[tetrahedron[3]])
          * volume;
        total_volume += volume;
    }
    if (total_volume <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-volume hexahedron.");
    }
    return weighted_centroid / total_volume;
}

/**
 * @brief Estimate wedge cell volume using tetrahedral decomposition.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Wedge vertex coordinates.
 * @return Total wedge volume.
 */
template <class Vec3>
inline real_t wedge_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 6);
    return tetra_volume(x[0], x[1], x[2], x[3])
         + tetra_volume(x[1], x[2], x[4], x[3])
         + tetra_volume(x[2], x[4], x[5], x[3]);
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

    Vec3 weighted_centroid{};
    real_t total_volume = 0.0;
    for (const auto& tetrahedron : tetrahedra)
    {
        const auto volume =
            tetra_volume(x[tetrahedron[0]], x[tetrahedron[1]],
                         x[tetrahedron[2]], x[tetrahedron[3]]);
        weighted_centroid =
            weighted_centroid
          + tetra_centroid(x[tetrahedron[0]], x[tetrahedron[1]],
                           x[tetrahedron[2]], x[tetrahedron[3]])
          * volume;
        total_volume += volume;
    }
    if (total_volume <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-volume wedge.");
    }
    return weighted_centroid / total_volume;
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
        return (x[0] + x[1] + x[2])
             / static_cast<typename Vec3::scalar_t>(3);
    }

    const auto first_area =
        (x[1] - x[0]).cross(x[2] - x[0]).norm() * 0.5;
    const auto second_area =
        (x[2] - x[0]).cross(x[3] - x[0]).norm() * 0.5;
    const auto total_area = first_area + second_area;
    if (total_area <= 0.0)
    {
        throw std::runtime_error(
            "Cannot compute the centroid of a zero-area face.");
    }

    const auto first_centroid =
        (x[0] + x[1] + x[2])
      / static_cast<typename Vec3::scalar_t>(3);
    const auto second_centroid =
        (x[0] + x[2] + x[3])
      / static_cast<typename Vec3::scalar_t>(3);
    return (first_centroid * first_area
          + second_centroid * second_area)
         / total_area;
}

/**
 * @brief Compute the oriented area vector for a face.
 *
 * @tparam Vec3 Vector type used for point coordinates.
 * @param x Face vertex coordinates in order.
 * @return Area vector of the face.
 */
template <class Vec3>
inline Vec3 face_area_vector(const std::vector<Vec3>& x)
{
    CHECK(x.size() == 3 || x.size() == 4);

    if (x.size() == 3)
    {
        return (x[1] - x[0]).cross(x[2] - x[0]) * 0.5;
    }

    return ((x[1] - x[0]).cross(x[2] - x[0])
          + (x[2] - x[0]).cross(x[3] - x[0])) * 0.5;
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
        result[i] = (arr[i] + arr[i + 1]) / static_cast<T>(2);
    }
    return result;
}

} // namespace MeshUtils

} // namespace SimpleFluid
