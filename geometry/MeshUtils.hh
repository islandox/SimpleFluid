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


} // namespace MeshUtils

} // namespace SimpleFluid
