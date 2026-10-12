/** @file CachedGradientRows.hh
 * @brief Ordered numerical rows without expanding compatibility geometry.
 */
#pragma once

#include <cstddef>
#include <utility>

namespace SimpleFluid::FVM::detail
{
template<bool IncludeBoundary, class Cache, class Visitor>
void visit_cached_gradient_rows(const Cache& cache, Visitor&& visitor)
{
    if constexpr (requires { cache.structured_geometry(); })
        if (const auto* structured = cache.structured_geometry())
        {
            structured->template visit_rows<IncludeBoundary>(visitor);
            return;
        }
    const auto& interior = cache.interior_geometry();
    if constexpr (IncludeBoundary)
    {
        const auto& boundary = cache.boundary_geometry();
        for (size_t cell = 0; cell < boundary.size(); ++cell)
            visitor(cell, boundary[cell].boundary_samples.empty() ? interior[cell] : boundary[cell]);
    }
    else
        for (size_t cell = 0; cell < interior.size(); ++cell) visitor(cell, interior[cell]);
}
} // namespace SimpleFluid::FVM::detail
