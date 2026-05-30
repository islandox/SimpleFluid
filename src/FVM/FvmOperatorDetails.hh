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

} // namespace SimpleFluid::FvmOperators::detail
