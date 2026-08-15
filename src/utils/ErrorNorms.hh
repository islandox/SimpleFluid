/**
 * @file ErrorNorms.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Error-norm utilities for comparing numerical solutions against
 *        analytical/exact solutions.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"

#include <cmath>
#include <cstddef>
#include <functional>

namespace SimpleFluid
{

/**
 * @brief Compute the L2 error norm between a numerical scalar field and
 *        an analytical solution.
 *
 * @f[
 *   \|u_h - u_{\text{exact}}\|_{L^2} =
 *   \sqrt{\frac{\sum_i (u_h^i - u_{\text{exact}}(\mathbf{x}_i))^2 \cdot V_i}
 *               {\sum_i V_i}}
 * @f]
 *
 * @tparam Pack Tpetra type pack.
 * @param field Numerical cell-centered scalar field.
 * @param exact A callable `scalar_type(vec3<scalar_type>)` returning the
 *        exact value at a given position.
 * @return The volume-weighted L2 error norm.
 */
template<TpetraTypePack Pack, class ExactFunc>
    requires std::invocable<ExactFunc, vec3<typename Pack::scalar_type>>
typename Pack::scalar_type l2_error(
    const CellField<Pack>& field,
    ExactFunc&& exact)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = field.mesh();
    scalar_type squared_sum = 0.0;
    scalar_type total_volume = 0.0;

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto numerical = field.value(cell_lid);
        const auto analytic = exact(mesh.cell_centroid(cell_lid));
        const auto diff = numerical - analytic;
        const auto volume = mesh.cell_volume(cell_lid);

        squared_sum += diff * diff * volume;
        total_volume += volume;
    }

    if (total_volume <= 0.0)
    {
        return scalar_type{};
    }

    return std::sqrt(squared_sum / total_volume);
}

/**
 * @brief Compute the L∞ (maximum) error norm between a numerical scalar
 *        field and an analytical solution.
 *
 * @f[
 *   \|u_h - u_{\text{exact}}\|_{L^\infty} =
 *   \max_i |u_h^i - u_{\text{exact}}(\mathbf{x}_i)|
 * @f]
 *
 * @tparam Pack Tpetra type pack.
 * @param field Numerical cell-centered scalar field.
 * @param exact A callable `scalar_type(vec3<scalar_type>)` returning the
 *        exact value at a given position.
 * @return The pointwise maximum absolute error.
 */
template<TpetraTypePack Pack, class ExactFunc>
    requires std::invocable<ExactFunc, vec3<typename Pack::scalar_type>>
typename Pack::scalar_type linf_error(
    const CellField<Pack>& field,
    ExactFunc&& exact)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = field.mesh();
    scalar_type max_error = 0.0;

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto numerical = field.value(cell_lid);
        const auto analytic = exact(mesh.cell_centroid(cell_lid));
        const auto diff = std::abs(numerical - analytic);

        if (diff > max_error)
        {
            max_error = diff;
        }
    }

    return max_error;
}

/**
 * @brief Compute the L2 error norm between a numerical vector field and
 *        an analytical vector-valued solution.
 *
 * @tparam Pack Tpetra type pack.
 * @param field Numerical cell-centered vector field.
 * @param exact A callable `vec3<scalar_type>(vec3<scalar_type>)` returning
 *        the exact vector at a given position.
 * @return The component-averaged L2 error norm.
 */
template<TpetraTypePack Pack, class ExactFunc>
    requires std::invocable<ExactFunc, vec3<typename Pack::scalar_type>>
typename Pack::scalar_type l2_error(
    const VectorCellField<Pack>& field,
    ExactFunc&& exact)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = field.mesh();
    scalar_type squared_sum = 0.0;
    scalar_type total_volume = 0.0;

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto numerical = field.value(cell_lid);
        const auto analytic = exact(mesh.cell_centroid(cell_lid));
        const auto diff = numerical - analytic;
        const auto volume = mesh.cell_volume(cell_lid);

        squared_sum += diff.dot(diff) * volume;
        total_volume += volume;
    }

    if (total_volume <= 0.0)
    {
        return scalar_type{};
    }

    return std::sqrt(squared_sum / total_volume);
}

/**
 * @brief Compute the L∞ error norm between a numerical vector field and
 *        an analytical vector-valued solution.
 *
 * @tparam Pack Tpetra type pack.
 * @param field Numerical cell-centered vector field.
 * @param exact A callable `vec3<scalar_type>(vec3<scalar_type>)` returning
 *        the exact vector at a given position.
 * @return The pointwise maximum Euclidean distance.
 */
template<TpetraTypePack Pack, class ExactFunc>
    requires std::invocable<ExactFunc, vec3<typename Pack::scalar_type>>
typename Pack::scalar_type linf_error(
    const VectorCellField<Pack>& field,
    ExactFunc&& exact)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = field.mesh();
    scalar_type max_error = 0.0;

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto numerical = field.value(cell_lid);
        const auto analytic = exact(mesh.cell_centroid(cell_lid));
        const auto diff = (numerical - analytic).norm();

        if (diff > max_error)
        {
            max_error = diff;
        }
    }

    return max_error;
}

} // namespace SimpleFluid
