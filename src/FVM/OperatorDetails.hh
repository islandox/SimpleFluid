/**
 * @file FVM/OperatorDetails.hh
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
#include <unordered_map>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/**
 * @brief Return a mutable reference to a component of a 3-D vector.
 *
 * @param vector The vector to index.
 * @param index Component index (0, 1, or 2).
 * @return Mutable reference to the requested component.
 */
inline real_t& component(MeshUtils::Vec3& vector, size_t index)
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
    for (size_t pivot = 0; pivot < 3; ++pivot)
    {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < 3; ++row)
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
        for (size_t col = pivot; col < 3; ++col)
        {
            a[pivot][col] *= inv;
        }
        component(b, pivot) *= inv;

        for (size_t row = 0; row < 3; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const auto factor = a[row][pivot];
            for (size_t col = pivot; col < 3; ++col)
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
inline real_t component_value(const MeshUtils::Vec3& vector, size_t index)
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

/**
 * @brief Entry in a least-squares gradient reconstruction stencil.
 *
 * @tparam MeshType The mesh type.
 */
template<class MeshType>
struct LeastSquaresGradientStencilEntry
{
    typename MeshType::local_ordinal_type cell_lid{};
    typename MeshType::Vec3 coefficient{};
};

/**
 * @brief Stencil type for least-squares gradient reconstruction.
 *
 * @tparam MeshType The mesh type.
 */
template<class MeshType>
using LeastSquaresGradientStencil =
    std::vector<LeastSquaresGradientStencilEntry<MeshType>>;

/**
 * @brief Accumulate a weighted direction contribution into a
 *        gradient-coefficient map.
 *
 * @tparam MeshType The mesh type.
 * @param[in,out] coefficients Map from cell LID to coefficient vector.
 * @param cell_lid Target cell LID.
 * @param coefficient Contribution to add.
 */
template<class MeshType>
void add_gradient_coefficient(
    std::unordered_map<typename MeshType::local_ordinal_type,
                       typename MeshType::Vec3>& coefficients,
    typename MeshType::local_ordinal_type cell_lid,
    const typename MeshType::Vec3& coefficient)
{
    auto& value = coefficients[cell_lid];
    value = {value.x + coefficient.x,
             value.y + coefficient.y,
             value.z + coefficient.z};
}

/**
 * @brief Compute least-squares gradient reconstruction stencils for all
 *        owned cells.
 *
 * @tparam MeshType The mesh type.
 * @param mesh The computational mesh.
 * @return Per-cell list of stencil entries (neighbor LID + coefficient).
 */
template<class MeshType>
std::vector<LeastSquaresGradientStencil<MeshType>>
least_squares_gradient_stencils(const MeshType& mesh)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;

    std::vector<LeastSquaresGradientStencil<MeshType>> stencils(
        mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        std::array<std::array<real_t, 3>, 3> normal{};
        std::vector<typename MeshType::Vec3> directions;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto d = mesh.cell_center_vector(face_lid, cell_lid);
            directions.push_back(d);

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];

        std::unordered_map<local_ordinal_type, typename MeshType::Vec3>
            coefficients;
        coefficients.reserve(directions.size() + 1);

        size_t direction_id = 0;
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            auto rhs = directions[direction_id++];
            auto local_normal = normal;
            const auto basis = solve_3x3(local_normal, rhs);
            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);

            add_gradient_coefficient<MeshType>(
                coefficients, other, basis);
            add_gradient_coefficient<MeshType>(
                coefficients, cell_lid,
                {-basis.x, -basis.y, -basis.z});
        }

        stencils[owned].reserve(coefficients.size());
        for (const auto& [entry_lid, coefficient] : coefficients)
        {
            stencils[owned].push_back({entry_lid, coefficient});
        }
    }

    return stencils;
}

/**
 * @brief Location descriptor for a boundary face.
 *
 * @tparam MeshType The mesh type.
 */
template<class MeshType>
struct BoundaryFaceLocation
{
    bool active = false;
    int patch_id = MeshType::invalid_boundary_id;
    size_t in_patch_id = 0;
};

template<class MeshType, class FaceID>
size_t packed_face_local_id(const MeshType& mesh, FaceID face_id)
{
    if constexpr (requires { mesh.face_local_id(face_id); })
    {
        return mesh.face_local_id(face_id);
    }
    else
    {
        return static_cast<size_t>(face_id);
    }
}

/**
 * @brief Compute a per-face lookup of boundary-face locations.
 *
 * Supports both the legacy Mesh API (boundary_patches() map) and the
 * new view-based MeshBase API (boundary_patch_ids() + boundary_face_patch()).
 *
 * @tparam MeshType The mesh type.
 * @param mesh The computational mesh.
 * @return Per-face boundary location (index by face LID).
 */
template<class MeshType>
std::vector<BoundaryFaceLocation<MeshType>>
boundary_face_locations(const MeshType& mesh)
{
    std::vector<BoundaryFaceLocation<MeshType>> locations(mesh.num_faces());

    if constexpr (std::ranges::range<
                      decltype(mesh.boundary_face_patch(0))>)
    {
        // New view-based API: boundary_face_patch() is directly iterable
        for (int patch_id : mesh.boundary_patch_ids())
        {
            size_t in_patch_id = 0;
            for (auto face_id : mesh.boundary_face_patch(patch_id))
            {
                locations[packed_face_local_id(mesh, face_id)] =
                    {true, patch_id, in_patch_id};
                ++in_patch_id;
            }
        }
    }
    else
    {
        // Legacy materialized-map API
        for (const auto& [patch_id, boundary_patch] :
             mesh.boundary_patches())
        {
            for (size_t in_patch_id = 0;
                 in_patch_id < boundary_patch.face_lids.size();
                 ++in_patch_id)
            {
                const auto face_lid =
                    boundary_patch.face_lids[in_patch_id];
                locations[packed_face_local_id(mesh, face_lid)] =
                    {true, patch_id, in_patch_id};
            }
        }
    }

    return locations;
}

/**
 * @brief Accumulate a matrix entry into a sparse row map.
 *
 * @tparam LocalOrdinal Local ordinal type.
 * @tparam Scalar Scalar type.
 * @param[in,out] row_values Map from column LID to accumulated value.
 * @param column Column LID.
 * @param value Value to add.
 */
template<class LocalOrdinal, class Scalar>
void add_matrix_entry(std::unordered_map<LocalOrdinal, Scalar>& row_values,
                      LocalOrdinal column,
                      Scalar value)
{
    row_values[column] += value;
}

} // namespace SimpleFluid::FVM::detail
