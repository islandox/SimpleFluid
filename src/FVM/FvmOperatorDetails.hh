/**
 * @file FvmOperatorDetails.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Internal finite-volume helper utilities.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/MeshUtils.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace SimpleFluid::FvmOperators::detail
{

/**
 * @brief Return a mutable reference to a component of a 3-D vector.
 *
 * @param vector The vector to index.
 * @param index Component index (0, 1, or 2).
 * @return Mutable reference to the requested component.
 */
inline real_t& component(MeshUtils::Vec3& vector, std::size_t index)
{
    return vector.component(index);
}

/**
 * @brief Solve a 3×3 linear system Ax = b using Gaussian elimination with
 *        partial pivoting.
 *
 * @param[in,out] a The 3×3 coefficient matrix, modified in place.
 * @param[in,out] b The right-hand-side vector; on output contains the
 *        solution x.
 * @return The solution vector (same as @p b on return, or zero if
 *         singular).
 */
inline MeshUtils::Vec3 solve_3x3(std::array<std::array<real_t, 3>, 3>& a,
                                 MeshUtils::Vec3& b)
{
    for (std::size_t pivot = 0; pivot < 3; ++pivot)
    {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < 3; ++row)
        {
            if (std::abs(a[row][pivot]) > std::abs(a[best][pivot]))
            {
                best = row;
            }
        }

        if (std::abs(a[best][pivot]) < 1.0e-14)
        {
            b = {};
            return {};
        }

        if (best != pivot)
        {
            std::swap(a[best], a[pivot]);
            std::swap(component(b, best), component(b, pivot));
        }

        const auto inv = 1.0 / a[pivot][pivot];
        for (std::size_t col = pivot; col < 3; ++col)
        {
            a[pivot][col] *= inv;
        }
        component(b, pivot) *= inv;

        for (std::size_t row = 0; row < 3; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const auto factor = a[row][pivot];
            for (std::size_t col = pivot; col < 3; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
            }
            component(b, row) -= factor * component(b, pivot);
        }
    }

    return b;
}

/**
 * @brief Return a const value of a component of a 3-D vector.
 *
 * @param vector The vector to index.
 * @param index Component index (0, 1, or 2).
 * @return Value of the requested component.
 */
inline real_t component_value(const MeshUtils::Vec3& vector, std::size_t index)
{
    return vector.component(index);
}

/**
 * @brief Decompose the face area vector into the component orthogonal to
 *        the cell-center connecting vector.
 *
 * @param area_vector The face area vector.
 * @param cell_center_vector Vector between adjacent cell centers.
 * @return The orthogonal (implicit) component of the area vector.
 * @throws std::runtime_error if the cell centers are coincident.
 */
inline MeshUtils::Vec3 orthogonal_area_vector(
    const MeshUtils::Vec3& area_vector,
    const MeshUtils::Vec3& cell_center_vector)
{
    const auto d2 = cell_center_vector.dot(cell_center_vector);
    if (d2 <= 0.0)
    {
        throw std::runtime_error(
            "Cannot decompose face area vector across coincident points.");
    }

    return cell_center_vector
         * (area_vector.dot(cell_center_vector) / d2);
}

/**
 * @brief Compute the non-orthogonal (tangential) component of the face
 *        area vector.
 *
 * @param area_vector The face area vector.
 * @param cell_center_vector Vector between adjacent cell centers.
 * @return The tangential (explicit) component of the area vector.
 */
inline MeshUtils::Vec3 non_orthogonal_area_vector(
    const MeshUtils::Vec3& area_vector,
    const MeshUtils::Vec3& cell_center_vector)
{
    return area_vector
         - orthogonal_area_vector(area_vector, cell_center_vector);
}

/**
 * @brief Compute the diffusion coefficient at an interior face for a
 *        two-point flux approximation.
 *
 * @tparam MeshType The mesh type.
 * @param mesh The computational mesh.
 * @param face_lid Local ID of the interior face.
 * @param cell_lid Local ID of the cell on one side.
 * @param other_lid Local ID of the cell on the opposite side.
 * @param diffusivity Constant scalar diffusivity.
 * @return The diffusion flux coefficient for the face.
 * @throws std::runtime_error if the two cell centers are coincident.
 */
template<class MeshType>
inline auto interior_diffusion_coefficient(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::local_ordinal_type other_lid,
    typename MeshType::scalar_type diffusivity)
    -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;

    const auto d = mesh.cell_centroid(other_lid) - mesh.cell_centroid(cell_lid);
    const auto d2 = d.dot(d);
    if (d2 <= scalar_type{0})
    {
        throw std::runtime_error(
            "Cannot assemble diffusion across coincident cells.");
    }

    const auto& normal = mesh.face_normal_outward(face_lid, cell_lid);
    const auto projected = mesh.face_area(face_lid) * normal.dot(d) / d2;
    if (projected > scalar_type{0})
    {
        return diffusivity * projected;
    }

    const auto distance = mesh.face_cell_center_distance(face_lid);
    if (distance <= scalar_type{0})
    {
        throw std::runtime_error(
            "Cannot assemble diffusion across coincident cells.");
    }
    return diffusivity * mesh.face_area(face_lid) / distance;
}

/**
 * @brief Compute the diffusion coefficient at a boundary face for a
 *        two-point flux approximation.
 *
 * @tparam MeshType The mesh type.
 * @param mesh The computational mesh.
 * @param face_lid Local ID of the boundary face.
 * @param cell_lid Local ID of the adjacent owned cell.
 * @param diffusivity Constant scalar diffusivity.
 * @return The diffusion flux coefficient, or zero if the cell and face
 *         centroids coincide.
 */
template<class MeshType>
inline auto boundary_diffusion_coefficient(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::scalar_type diffusivity)
    -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;

    const auto d = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
    const auto d2 = d.dot(d);
    if (d2 <= scalar_type{0})
    {
        return scalar_type{};
    }

    const auto& normal = mesh.face_normal_outward(face_lid, cell_lid);
    const auto projected = mesh.face_area(face_lid) * normal.dot(d) / d2;
    if (projected > scalar_type{0})
    {
        return diffusivity * projected;
    }

    const auto distance = mesh.cell_to_face_distance(face_lid, cell_lid);
    return distance > scalar_type{0}
         ? diffusivity * mesh.face_area(face_lid) / distance
         : scalar_type{};
}

} // namespace SimpleFluid::FvmOperators::detail
