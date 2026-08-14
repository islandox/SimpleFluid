/**
 * @file FVM/details/OperatorDetails.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Internal finite-volume helper utilities.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "FVM/FaceCoefficientInterpolation.hh"
#include "geometry/MeshUtils.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
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
 * @brief Solve a symmetric positive-semidefinite 3x3 system with a
 *        Moore-Penrose pseudoinverse.
 *
 * Least-squares gradient normal matrices become rank deficient on
 * one-cell-thick or one-dimensional meshes. Jacobi diagonalization retains
 * the resolvable in-plane components and sets null-space components to zero.
 */
inline MeshUtils::Vec3 solve_symmetric_3x3_pseudoinverse(
    const std::array<std::array<real_t, 3>, 3>& input,
    const MeshUtils::Vec3& rhs)
{
    auto matrix = input;
    std::array<std::array<real_t, 3>, 3> eigenvectors{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}}};

    for (size_t sweep = 0; sweep < 24; ++sweep)
    {
        size_t p = 0;
        size_t q = 1;
        auto largest = std::abs(matrix[p][q]);
        for (const auto [row, column] :
             std::array<std::array<size_t, 2>, 3>{{
                 {{0, 2}}, {{1, 2}}, {{0, 1}}}})
        {
            const auto magnitude = std::abs(matrix[row][column]);
            if (magnitude > largest)
            {
                largest = magnitude;
                p = row;
                q = column;
            }
        }

        const auto scale = std::max({
            std::abs(matrix[0][0]),
            std::abs(matrix[1][1]),
            std::abs(matrix[2][2])});
        if (scale == real_t{}
            || largest <= std::numeric_limits<real_t>::epsilon()
                              * real_t{64} * scale)
        {
            break;
        }

        const auto app = matrix[p][p];
        const auto aqq = matrix[q][q];
        const auto apq = matrix[p][q];
        const auto tau = (aqq - app) / (real_t{2} * apq);
        const auto tangent =
            tau >= real_t{}
          ? real_t{1} / (tau + std::sqrt(real_t{1} + tau * tau))
          : -real_t{1} / (-tau + std::sqrt(real_t{1} + tau * tau));
        const auto cosine =
            real_t{1} / std::sqrt(real_t{1} + tangent * tangent);
        const auto sine = tangent * cosine;

        for (size_t row = 0; row < 3; ++row)
        {
            if (row == p || row == q)
            {
                continue;
            }
            const auto arp = matrix[row][p];
            const auto arq = matrix[row][q];
            matrix[row][p] = matrix[p][row] =
                cosine * arp - sine * arq;
            matrix[row][q] = matrix[q][row] =
                sine * arp + cosine * arq;
        }
        matrix[p][p] = app - tangent * apq;
        matrix[q][q] = aqq + tangent * apq;
        matrix[p][q] = matrix[q][p] = real_t{};

        for (size_t row = 0; row < 3; ++row)
        {
            const auto vrp = eigenvectors[row][p];
            const auto vrq = eigenvectors[row][q];
            eigenvectors[row][p] =
                cosine * vrp - sine * vrq;
            eigenvectors[row][q] =
                sine * vrp + cosine * vrq;
        }
    }

    const auto largest_eigenvalue = std::max({
        std::abs(matrix[0][0]),
        std::abs(matrix[1][1]),
        std::abs(matrix[2][2])});
    if (largest_eigenvalue == real_t{})
    {
        return {};
    }
    const auto threshold =
        largest_eigenvalue * real_t{1.0e-12};

    MeshUtils::Vec3 solution{};
    for (size_t eigenvector = 0; eigenvector < 3; ++eigenvector)
    {
        const auto eigenvalue = matrix[eigenvector][eigenvector];
        if (eigenvalue <= threshold)
        {
            continue;
        }
        const auto projection =
            eigenvectors[0][eigenvector] * rhs.x
          + eigenvectors[1][eigenvector] * rhs.y
          + eigenvectors[2][eigenvector] * rhs.z;
        const auto weight = projection / eigenvalue;
        solution.x += eigenvectors[0][eigenvector] * weight;
        solution.y += eigenvectors[1][eigenvector] * weight;
        solution.z += eigenvectors[2][eigenvector] * weight;
    }
    return solution;
}

/**
 * @brief Solve a 3×3 linear system Ax = b using Gaussian elimination with
 *        partial pivoting.
 *
 * @param[in,out] a The 3×3 coefficient matrix, modified in place.
 * @param[in,out] b The right-hand-side vector; on output contains the
 *        solution x.
 * @return The solution vector. Rank-deficient symmetric normal systems use
 *         a Moore-Penrose pseudoinverse.
 */
inline MeshUtils::Vec3 solve_3x3(std::array<std::array<real_t, 3>, 3>& a,
                                 MeshUtils::Vec3& b)
{
    const auto original_matrix = a;
    const auto original_rhs = b;

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
            b = solve_symmetric_3x3_pseudoinverse(
                original_matrix, original_rhs);
            return b;
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
 * @brief Convert a packed cell LID to the mesh's native cell identifier.
 * @tparam MeshType Mesh interface type.
 * @param mesh Mesh that owns the identifier mapping.
 * @param cell_lid Packed local cell identifier.
 * @return Native cell identifier when available, otherwise @p cell_lid.
 */
template<class MeshType>
auto query_cell_id(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type cell_lid)
{
    if constexpr (requires { mesh.cell_id(cell_lid); })
    {
        return mesh.cell_id(cell_lid);
    }
    else
    {
        return cell_lid;
    }
}

/**
 * @brief Convert a packed face LID to the mesh's native face identifier.
 * @tparam MeshType Mesh interface type.
 * @param mesh Mesh that owns the identifier mapping.
 * @param face_lid Packed local face identifier.
 * @return Native face identifier when available, otherwise @p face_lid.
 */
template<class MeshType>
auto query_face_id(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid)
{
    if constexpr (requires { mesh.face_id(face_lid); })
    {
        return mesh.face_id(face_lid);
    }
    else
    {
        return face_lid;
    }
}

/**
 * @brief Convert a native cell identifier to its packed local identifier.
 * @tparam MeshType Mesh interface type.
 * @tparam CellID Native cell identifier type.
 * @param mesh Mesh that owns the identifier mapping.
 * @param cell_id Native or already-packed cell identifier.
 * @return Packed local cell identifier.
 */
template<class MeshType, class CellID>
auto packed_cell_local_id(const MeshType& mesh, CellID cell_id)
    -> typename MeshType::local_ordinal_type
{
    if constexpr (requires { mesh.cell_local_id(cell_id); })
    {
        return mesh.cell_local_id(cell_id);
    }
    else
    {
        return static_cast<typename MeshType::local_ordinal_type>(cell_id);
    }
}

/**
 * @brief Distance-weighted harmonic interpolation of a positive cell field
 *        to an interior face.
 *
 * A zero value on either side produces a zero face coefficient. Negative
 * values are invalid because they would hide a material-property bug by
 * silently removing diffusion at the face. If geometric cell-to-face distances
 * are unavailable, the ordinary harmonic mean is used.
 * @throws std::invalid_argument if either cell coefficient is negative.
 */
template<class MeshType>
inline auto harmonic_face_value(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::local_ordinal_type other_lid,
    typename MeshType::scalar_type cell_value,
    typename MeshType::scalar_type other_value)
    -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;

    if (cell_value < scalar_type{} || other_value < scalar_type{})
    {
        throw std::invalid_argument(
            "harmonic_face_value requires non-negative cell values.");
    }
    if (cell_value == scalar_type{} || other_value == scalar_type{})
    {
        return scalar_type{};
    }

    const auto face_id = query_face_id(mesh, face_lid);
    const auto cell_distance =
        mesh.cell_to_face_distance(face_id, query_cell_id(mesh, cell_lid));
    const auto other_distance =
        mesh.cell_to_face_distance(face_id, query_cell_id(mesh, other_lid));
    const auto total_distance = cell_distance + other_distance;
    if (cell_distance > scalar_type{}
        && other_distance > scalar_type{}
        && total_distance > scalar_type{})
    {
        return total_distance
             / (cell_distance / cell_value
              + other_distance / other_value);
    }

    return scalar_type{2} * cell_value * other_value
         / (cell_value + other_value);
}

/**
 * @brief Field-based convenience overload for harmonic interpolation.
 *
 * Hot loops should prefer the scalar-value overload after acquiring one local
 * field view for the whole operator.
 */
template<class MeshType, class FieldType>
inline auto harmonic_face_value(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::local_ordinal_type other_lid,
    const FieldType& field) -> typename MeshType::scalar_type
{
    return harmonic_face_value(
        mesh, face_lid, cell_lid, other_lid,
        field.local_value(cell_lid), field.local_value(other_lid));
}

/**
 * @brief Distance-weighted linear interpolation of a non-negative cell field
 *        to an interior face.
 */
template<class MeshType>
inline auto linear_face_value(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::local_ordinal_type other_lid,
    typename MeshType::scalar_type cell_value,
    typename MeshType::scalar_type other_value)
    -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;
    if (cell_value < scalar_type{} || other_value < scalar_type{})
    {
        throw std::invalid_argument(
            "linear_face_value requires non-negative cell values.");
    }

    const auto face_id = query_face_id(mesh, face_lid);
    const auto cell_distance = static_cast<scalar_type>(
        mesh.cell_to_face_distance(
            face_id, query_cell_id(mesh, cell_lid)));
    const auto other_distance = static_cast<scalar_type>(
        mesh.cell_to_face_distance(
            face_id, query_cell_id(mesh, other_lid)));
    const auto total_distance = cell_distance + other_distance;
    if (cell_distance >= scalar_type{}
        && other_distance >= scalar_type{}
        && total_distance > scalar_type{})
    {
        return (
            other_distance * cell_value
            + cell_distance * other_value) / total_distance;
    }
    return scalar_type{0.5} * (cell_value + other_value);
}

/**
 * @brief Interpolate a non-negative transport coefficient to an interior face.
 * @throws std::invalid_argument for an unknown interpolation selection or a
 *         negative coefficient.
 */
template<class MeshType>
inline auto face_coefficient_value(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid,
    typename MeshType::local_ordinal_type other_lid,
    typename MeshType::scalar_type cell_value,
    typename MeshType::scalar_type other_value,
    FaceCoefficientInterpolation interpolation)
    -> typename MeshType::scalar_type
{
    switch (interpolation)
    {
        case FaceCoefficientInterpolation::Harmonic:
            return harmonic_face_value(
                mesh, face_lid, cell_lid, other_lid,
                cell_value, other_value);
        case FaceCoefficientInterpolation::Linear:
            return linear_face_value(
                mesh, face_lid, cell_lid, other_lid,
                cell_value, other_value);
    }
    throw std::invalid_argument(
        "Unknown face-coefficient interpolation.");
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

    const auto face_id = query_face_id(mesh, face_lid);
    const auto cell_id = query_cell_id(mesh, cell_lid);
    const auto other_id = query_cell_id(mesh, other_lid);
    const auto d = mesh.cell_centroid(other_id) - mesh.cell_centroid(cell_id);
    const auto d2 = d.dot(d);
    if (d2 <= scalar_type{0})
    {
        throw std::runtime_error(
            "Cannot assemble diffusion across coincident cells "
            + std::to_string(cell_lid)
            + " and "
            + std::to_string(other_lid)
            + ".");
    }

    const auto& normal = mesh.face_normal_outward(face_id, cell_id);
    const auto projected = mesh.face_area(face_id) * normal.dot(d) / d2;
    if (projected > scalar_type{0})
    {
        return diffusivity * projected;
    }

    const auto stored_distance =
        mesh.face_cell_center_distance(face_id);
    const auto distance =
        stored_distance > scalar_type{0}
      ? stored_distance
      : std::sqrt(d2);
    return diffusivity * mesh.face_area(face_id) / distance;
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

    const auto face_id = query_face_id(mesh, face_lid);
    const auto cell_id = query_cell_id(mesh, cell_lid);
    const auto d = mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id);
    const auto d2 = d.dot(d);
    if (d2 <= scalar_type{0})
    {
        return scalar_type{};
    }

    const auto& normal = mesh.face_normal_outward(face_id, cell_id);
    const auto projected = mesh.face_area(face_id) * normal.dot(d) / d2;
    if (projected > scalar_type{0})
    {
        return diffusivity * projected;
    }

    const auto distance = mesh.cell_to_face_distance(face_id, cell_id);
    return distance > scalar_type{0}
         ? diffusivity * mesh.face_area(face_id) / distance
         : scalar_type{};
}

/**
 * @brief Signed distance from a cell centroid to a boundary-face plane.
 *
 * The projection uses the unit normal pointing outward from the query cell.
 * Unlike the Euclidean cell-to-face-centroid distance, this is the distance
 * appropriate for converting a prescribed outward normal derivative into a
 * cell-to-boundary value increment on a skew face.
 */
template<class MeshType>
inline auto boundary_normal_distance(
    const MeshType& mesh,
    typename MeshType::local_ordinal_type face_lid,
    typename MeshType::local_ordinal_type cell_lid)
    -> typename MeshType::scalar_type
{
    const auto face_id = query_face_id(mesh, face_lid);
    const auto cell_id = query_cell_id(mesh, cell_lid);
    const auto direction =
        mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id);
    return direction.dot(mesh.face_normal_outward(face_id, cell_id));
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
 * @brief Cell coefficients plus the boundary-data constant of a gradient.
 * @tparam MeshType Mesh interface type.
 */
template<class MeshType>
struct AffineLeastSquaresGradientStencil
{
    LeastSquaresGradientStencil<MeshType> entries;
    typename MeshType::Vec3 constant{};
};

/**
 * @brief Component-wise boundary constants for a vector gradient.
 * @tparam MeshType Mesh interface type.
 */
template<class MeshType>
struct VectorAffineLeastSquaresGradientStencil
{
    LeastSquaresGradientStencil<MeshType> entries;
    std::array<typename MeshType::Vec3, 3> constants{};
};

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
        const auto cell_id = query_cell_id(mesh, cell_lid);
        std::array<std::array<real_t, 3>, 3> normal{};
        std::vector<typename MeshType::Vec3> directions;

        for (const auto face_id : mesh.faces(cell_id))
        {
            if (!mesh.is_interior_face(face_id))
            {
                continue;
            }

            const auto d = mesh.cell_center_vector(face_id, cell_id);
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
        for (const auto face_id : mesh.faces(cell_id))
        {
            if (!mesh.is_interior_face(face_id))
            {
                continue;
            }

            auto rhs = directions[direction_id++];
            auto local_normal = normal;
            const auto basis = solve_3x3(local_normal, rhs);
            const auto other_id =
                mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
            const auto other_lid = packed_cell_local_id(mesh, other_id);

            add_gradient_coefficient<MeshType>(
                coefficients, other_lid, basis);
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
    int batch_id = MeshType::invalid_boundary_id;
    size_t in_batch_id = 0;
};

/** @brief Boundary policy that enables diffusion on every queried face. */
struct AlwaysDiffuseBoundary
{
    template<class... Args>
    constexpr bool operator()(Args&&...) const noexcept
    {
        return true;
    }
};

/**
 * @brief Convert a native face identifier to its packed local identifier.
 * @tparam MeshType Mesh interface type.
 * @tparam FaceID Native face identifier type.
 * @param mesh Mesh that owns the identifier mapping.
 * @param face_id Native or already-packed face identifier.
 * @return Packed local face identifier.
 */
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
 * Supports both the legacy Mesh API (boundary_batches() map) and the
 * new view-based MeshBase API (boundary_batch_ids() + boundary_face_batch()).
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
                      decltype(mesh.boundary_face_batch(0))>)
    {
        // New view-based API: boundary_face_batch() is directly iterable
        for (int batch_id : mesh.boundary_batch_ids())
        {
            size_t in_batch_id = 0;
            for (auto face_id : mesh.boundary_face_batch(batch_id))
            {
                locations[packed_face_local_id(mesh, face_id)] =
                    {true, batch_id, in_batch_id};
                ++in_batch_id;
            }
        }
    }
    else
    {
        // Legacy materialized-map API
        for (const auto& [batch_id, boundary_batch] :
             mesh.boundary_batches())
        {
            for (size_t in_batch_id = 0;
                 in_batch_id < boundary_batch.face_lids.size();
                 ++in_batch_id)
            {
                const auto face_lid =
                    boundary_batch.face_lids[in_batch_id];
                locations[packed_face_local_id(mesh, face_lid)] =
                    {true, batch_id, in_batch_id};
            }
        }
    }

    return locations;
}

/** @brief Static least-squares geometry for one boundary sample. */
template<class MeshType>
struct BoundaryGradientSampleGeometry
{
    typename MeshType::local_ordinal_type face_lid{};
    BoundaryFaceLocation<MeshType> location{};
    typename MeshType::Vec3 basis{};
    real_t normal_distance{};
};

/** @brief Mesh-only part of one boundary-aware gradient reconstruction. */
template<class MeshType>
struct BoundaryAwareGradientCellGeometry
{
    LeastSquaresGradientStencil<MeshType> interior_entries;
    std::vector<BoundaryGradientSampleGeometry<MeshType>> boundary_samples;
};

/**
 * @brief Build reusable boundary-aware least-squares basis geometry.
 *
 * Boundary values and condition types are deliberately excluded. Both
 * Dirichlet and Neumann samples use the same normal matrix and basis; their
 * different affine contributions are materialized at assembly time.
 */
template<class MeshType>
std::vector<BoundaryAwareGradientCellGeometry<MeshType>>
boundary_aware_gradient_geometry(
    const MeshType& mesh,
    const std::vector<BoundaryFaceLocation<MeshType>>& boundary_locations)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    std::vector<BoundaryAwareGradientCellGeometry<MeshType>> geometry(
        mesh.num_owned_cells());
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        std::array<std::array<real_t, 3>, 3> normal{};
        auto add_direction = [&](const vec_type& direction)
        {
            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        };

        for (const auto face_id : mesh.faces(cell_id))
        {
            if (mesh.is_interior_face(face_id))
            {
                add_direction(mesh.cell_center_vector(face_id, cell_id));
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(
                packed_face_local_id(mesh, face_id));
            if (location.active)
            {
                add_direction(
                    mesh.face_centroid(face_id)
                    - mesh.cell_centroid(cell_id));
            }
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        std::unordered_map<local_ordinal_type, vec_type> coefficients;
        auto& cell_geometry = geometry[owned];

        for (const auto face_id : mesh.faces(cell_id))
        {
            if (mesh.is_interior_face(face_id))
            {
                auto direction = mesh.cell_center_vector(face_id, cell_id);
                auto local_normal = normal;
                const auto basis = solve_3x3(local_normal, direction);
                const auto other_id =
                    mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid =
                    packed_cell_local_id(mesh, other_id);
                add_gradient_coefficient<MeshType>(
                    coefficients, other_lid, basis);
                add_gradient_coefficient<MeshType>(
                    coefficients, cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto packed_face_lid = static_cast<local_ordinal_type>(
                packed_face_local_id(mesh, face_id));
            const auto location = boundary_locations.at(
                static_cast<size_t>(packed_face_lid));
            if (!location.active)
            {
                continue;
            }
            auto direction =
                mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id);
            auto local_normal = normal;
            cell_geometry.boundary_samples.push_back({
                packed_face_lid,
                location,
                solve_3x3(local_normal, direction),
                static_cast<real_t>(
                    boundary_normal_distance(mesh, face_id, cell_id))});
        }

        cell_geometry.interior_entries.reserve(coefficients.size());
        for (const auto& [entry_lid, coefficient] : coefficients)
        {
            cell_geometry.interior_entries.push_back(
                {entry_lid, coefficient});
        }
    }
    return geometry;
}

/** @brief Add or accumulate an entry in a materialized gradient stencil. */
template<class MeshType>
void add_materialized_gradient_coefficient(
    LeastSquaresGradientStencil<MeshType>& entries,
    typename MeshType::local_ordinal_type cell_lid,
    const typename MeshType::Vec3& coefficient)
{
    const auto iter = std::find_if(
        entries.begin(), entries.end(),
        [cell_lid](const auto& entry)
        {
            return entry.cell_lid == cell_lid;
        });
    if (iter == entries.end())
    {
        entries.push_back({cell_lid, coefficient});
        return;
    }
    iter->coefficient = {
        iter->coefficient.x + coefficient.x,
        iter->coefficient.y + coefficient.y,
        iter->coefficient.z + coefficient.z};
}

/** @brief Materialize scalar affine stencils from cached mesh geometry. */
template<class MeshType,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
std::vector<AffineLeastSquaresGradientStencil<MeshType>>
materialize_scalar_affine_gradient_stencils(
    const std::vector<BoundaryAwareGradientCellGeometry<MeshType>>& geometry,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    std::vector<AffineLeastSquaresGradientStencil<MeshType>> stencils(
        geometry.size());
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto& cell_geometry = geometry[owned];
        auto& stencil = stencils[owned];
        stencil.entries = cell_geometry.interior_entries;
        vec_type cell_coefficient{};
        bool has_dirichlet_sample = false;

        for (const auto& sample : cell_geometry.boundary_samples)
        {
            const auto condition = boundary_condition(
                sample.location.batch_id,
                sample.location.in_batch_id);
            real_t boundary_increment{};
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                has_dirichlet_sample = true;
                cell_coefficient.x -= sample.basis.x;
                cell_coefficient.y -= sample.basis.y;
                cell_coefficient.z -= sample.basis.z;
                boundary_increment = static_cast<real_t>(boundary_value(
                    sample.location.batch_id,
                    sample.location.in_batch_id));
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                boundary_increment =
                    static_cast<real_t>(condition.value)
                    * sample.normal_distance;
            }
            else
            {
                throw std::invalid_argument(
                    "Affine scalar gradients support only Dirichlet and "
                    "Neumann boundary conditions.");
            }
            stencil.constant.x += sample.basis.x * boundary_increment;
            stencil.constant.y += sample.basis.y * boundary_increment;
            stencil.constant.z += sample.basis.z * boundary_increment;
        }

        if (has_dirichlet_sample)
        {
            add_materialized_gradient_coefficient<MeshType>(
                stencil.entries, cell_lid, cell_coefficient);
        }
    }
    return stencils;
}

/** @brief Materialize vector affine stencils from cached mesh geometry. */
template<class MeshType, class BoundaryValueProvider>
std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>>
materialize_vector_affine_gradient_stencils(
    const std::vector<BoundaryAwareGradientCellGeometry<MeshType>>& geometry,
    BoundaryValueProvider boundary_value)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>> stencils(
        geometry.size());
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto& cell_geometry = geometry[owned];
        auto& stencil = stencils[owned];
        stencil.entries = cell_geometry.interior_entries;
        vec_type cell_coefficient{};
        for (const auto& sample : cell_geometry.boundary_samples)
        {
            cell_coefficient.x -= sample.basis.x;
            cell_coefficient.y -= sample.basis.y;
            cell_coefficient.z -= sample.basis.z;
            const auto value = static_cast<vec_type>(boundary_value(
                sample.location.batch_id,
                sample.location.in_batch_id));
            for (size_t component_id = 0; component_id < 3; ++component_id)
            {
                const auto value_component = value.component(component_id);
                stencil.constants[component_id].x +=
                    sample.basis.x * value_component;
                stencil.constants[component_id].y +=
                    sample.basis.y * value_component;
                stencil.constants[component_id].z +=
                    sample.basis.z * value_component;
            }
        }
        if (!cell_geometry.boundary_samples.empty())
        {
            add_materialized_gradient_coefficient<MeshType>(
                stencil.entries, cell_lid, cell_coefficient);
        }
    }
    return stencils;
}

/**
 * @brief Build scalar least-squares gradient stencils including boundary data.
 *
 * The returned affine representation exactly matches the boundary-aware
 * `cell_gradient`: cell-dependent terms are stored in `entries`, while
 * prescribed Dirichlet values and Neumann increments are stored in
 * `constant` for transfer to an implicit transport RHS.
 */
template<class MeshType,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
std::vector<AffineLeastSquaresGradientStencil<MeshType>>
scalar_affine_gradient_stencils(
    const MeshType& mesh,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    const auto boundary_locations = boundary_face_locations(mesh);
    std::vector<AffineLeastSquaresGradientStencil<MeshType>> stencils(
        mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        std::array<std::array<real_t, 3>, 3> normal{};

        auto add_direction = [&](const vec_type& direction)
        {
            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        };

        for (const auto face_id : mesh.faces(cell_id))
        {
            if (mesh.is_interior_face(face_id))
            {
                add_direction(mesh.cell_center_vector(face_id, cell_id));
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(
                packed_face_local_id(mesh, face_id));
            if (!location.active)
            {
                continue;
            }
            const auto condition = boundary_condition(
                location.batch_id, location.in_batch_id);
            if (condition.type != BoundaryConditionType::Dirichlet
                && condition.type != BoundaryConditionType::Neumann)
            {
                throw std::invalid_argument(
                    "Affine scalar gradients support only Dirichlet and "
                    "Neumann boundary conditions.");
            }
            add_direction(
                mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id));
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        std::unordered_map<local_ordinal_type, vec_type> coefficients;
        auto& stencil = stencils[owned];

        for (const auto face_id : mesh.faces(cell_id))
        {
            vec_type direction{};
            if (mesh.is_interior_face(face_id))
            {
                direction = mesh.cell_center_vector(face_id, cell_id);
                auto local_normal = normal;
                const auto basis = solve_3x3(local_normal, direction);
                const auto other_id =
                    mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid = packed_cell_local_id(mesh, other_id);
                add_gradient_coefficient<MeshType>(
                    coefficients, other_lid, basis);
                add_gradient_coefficient<MeshType>(
                    coefficients, cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(
                packed_face_local_id(mesh, face_id));
            if (!location.active)
            {
                continue;
            }
            direction =
                mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id);
            auto local_normal = normal;
            const auto basis = solve_3x3(local_normal, direction);
            const auto condition = boundary_condition(
                location.batch_id, location.in_batch_id);
            real_t boundary_increment{};
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                add_gradient_coefficient<MeshType>(
                    coefficients, cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                boundary_increment = static_cast<real_t>(boundary_value(
                    location.batch_id, location.in_batch_id));
            }
            else
            {
                boundary_increment = condition.value
                    * static_cast<real_t>(boundary_normal_distance(
                        mesh, face_id, cell_id));
            }
            stencil.constant.x += basis.x * boundary_increment;
            stencil.constant.y += basis.y * boundary_increment;
            stencil.constant.z += basis.z * boundary_increment;
        }

        stencil.entries.reserve(coefficients.size());
        for (const auto& [entry_lid, coefficient] : coefficients)
        {
            stencil.entries.push_back({entry_lid, coefficient});
        }
    }
    return stencils;
}

/** @brief Build component-wise affine vector gradient stencils. */
template<class MeshType, class BoundaryValueProvider>
std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>>
vector_affine_gradient_stencils(
    const MeshType& mesh,
    BoundaryValueProvider boundary_value)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    const auto boundary_locations = boundary_face_locations(mesh);
    std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>> stencils(
        mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        std::array<std::array<real_t, 3>, 3> normal{};
        auto add_direction = [&](const vec_type& direction)
        {
            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        };

        for (const auto face_id : mesh.faces(cell_id))
        {
            if (mesh.is_interior_face(face_id))
            {
                add_direction(mesh.cell_center_vector(face_id, cell_id));
            }
            else if (mesh.is_boundary_face(face_id))
            {
                const auto location = boundary_locations.at(
                    packed_face_local_id(mesh, face_id));
                if (location.active)
                {
                    add_direction(mesh.face_centroid(face_id)
                                  - mesh.cell_centroid(cell_id));
                }
            }
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        std::unordered_map<local_ordinal_type, vec_type> coefficients;
        auto& stencil = stencils[owned];

        for (const auto face_id : mesh.faces(cell_id))
        {
            vec_type direction{};
            if (mesh.is_interior_face(face_id))
            {
                direction = mesh.cell_center_vector(face_id, cell_id);
                auto local_normal = normal;
                const auto basis = solve_3x3(local_normal, direction);
                const auto other_id =
                    mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid = packed_cell_local_id(mesh, other_id);
                add_gradient_coefficient<MeshType>(
                    coefficients, other_lid, basis);
                add_gradient_coefficient<MeshType>(
                    coefficients, cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(
                packed_face_local_id(mesh, face_id));
            if (!location.active)
            {
                continue;
            }
            direction =
                mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id);
            auto local_normal = normal;
            const auto basis = solve_3x3(local_normal, direction);
            add_gradient_coefficient<MeshType>(
                coefficients, cell_lid,
                {-basis.x, -basis.y, -basis.z});
            const auto value = static_cast<vec_type>(boundary_value(
                location.batch_id, location.in_batch_id));
            for (size_t component_id = 0; component_id < 3; ++component_id)
            {
                const auto component_value = value.component(component_id);
                stencil.constants[component_id].x +=
                    basis.x * component_value;
                stencil.constants[component_id].y +=
                    basis.y * component_value;
                stencil.constants[component_id].z +=
                    basis.z * component_value;
            }
        }

        stencil.entries.reserve(coefficients.size());
        for (const auto& [entry_lid, coefficient] : coefficients)
        {
            stencil.entries.push_back({entry_lid, coefficient});
        }
    }
    return stencils;
}

/**
 * @brief Reusable direct-slot scratch storage for one sparse matrix row.
 *
 * A single overlap-sized column-to-slot table makes both first insertion and
 * repeated accumulation constant time.  `clear()` invalidates only the
 * columns used by the preceding row, while the compact column and value
 * buffers retain their capacity.  After growing to the widest encountered
 * row, assembly performs no per-row allocations.  The buffers can be passed
 * directly to Tpetra, avoiding hash nodes, sorting, and map-to-array copies.
 *
 * @tparam LocalOrdinal Local ordinal type.
 * @tparam Scalar Matrix scalar type.
 */
template<class LocalOrdinal, class Scalar>
class FlatMatrixRow
{
public:
    /**
     * @param num_local_columns Number of columns in the overlap map.
     * @param capacity Expected maximum row width.
     */
    explicit FlatMatrixRow(size_t num_local_columns,
                           size_t capacity = 0)
        : d_column_slots(
              num_local_columns,
              std::numeric_limits<size_t>::max())
    {
        d_columns.reserve(capacity);
        d_values.reserve(capacity);
    }

    /** @brief Remove row entries while retaining all allocated storage. */
    void clear() noexcept
    {
        for (const auto column : d_columns)
        {
            d_column_slots[static_cast<size_t>(column)] =
                std::numeric_limits<size_t>::max();
        }
        d_columns.clear();
        d_values.clear();
    }

    /** @brief Ensure a column is present with an initial zero value. */
    void ensure(LocalOrdinal column)
    {
        (void)find_or_insert(column);
    }

    /** @brief Accumulate a coefficient through its direct row slot. */
    void add(LocalOrdinal column, Scalar value)
    {
        d_values[find_or_insert(column)] += value;
    }

    /** @brief Assign a coefficient through its direct row slot. */
    void set(LocalOrdinal column, Scalar value)
    {
        d_values[find_or_insert(column)] = value;
    }

    /** @brief Assign the same value to every stored coefficient. */
    void fill(Scalar value)
    {
        std::fill(d_values.begin(), d_values.end(), value);
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return d_columns.size();
    }

    [[nodiscard]] const LocalOrdinal* column_data() const noexcept
    {
        return d_columns.data();
    }

    [[nodiscard]] const Scalar* value_data() const noexcept
    {
        return d_values.data();
    }

private:
    [[nodiscard]] size_t column_index(LocalOrdinal column) const
    {
        if constexpr (std::numeric_limits<LocalOrdinal>::is_signed)
        {
            if (column < LocalOrdinal{})
            {
                throw std::out_of_range(
                    "transport matrix column is negative.");
            }
        }
        const auto index = static_cast<size_t>(column);
        if (index >= d_column_slots.size())
        {
            throw std::out_of_range(
                "transport matrix column is outside the overlap map.");
        }
        return index;
    }

    [[nodiscard]] size_t find_or_insert(LocalOrdinal column)
    {
        const auto column_id = column_index(column);
        auto& position = d_column_slots[column_id];
        if (position == std::numeric_limits<size_t>::max())
        {
            const auto new_position = d_columns.size();
            d_columns.push_back(column);
            try
            {
                d_values.push_back(Scalar{});
            }
            catch (...)
            {
                d_columns.pop_back();
                throw;
            }
            position = new_position;
        }
        return position;
    }

    std::vector<LocalOrdinal> d_columns;
    std::vector<Scalar> d_values;
    std::vector<size_t> d_column_slots;
};

/** @brief Accumulate a matrix entry through a direct row slot. */
template<class LocalOrdinal, class Scalar>
void add_matrix_entry(FlatMatrixRow<LocalOrdinal, Scalar>& row_values,
                      LocalOrdinal column,
                      Scalar value)
{
    row_values.add(column, value);
}

/**
 * @brief Accumulate a matrix entry into a legacy sparse row map.
 *
 * Non-transport assembly paths still use map staging. This overload keeps
 * their existing behavior while transport rows use FlatMatrixRow.
 */
template<class LocalOrdinal, class Scalar>
void add_matrix_entry(std::unordered_map<LocalOrdinal, Scalar>& row_values,
                      LocalOrdinal column,
                      Scalar value)
{
    row_values[column] += value;
}

} // namespace SimpleFluid::FVM::detail
