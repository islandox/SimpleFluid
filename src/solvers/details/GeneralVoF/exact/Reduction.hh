/** @file Reduction.hh
 * @brief Collective rational inventory reduction within an explicit packet budget.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include <span>

namespace SimpleFluid::detail::general_vof::exact
{
using Measure = PLIC::GeometryMeasure<ArithmeticMode::Exact>;
/** Collective, including empty owners. Authenticates one contribution per rank
 * and broadcasts the exact sum through the existing budgeted codec. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure global_sum(
    const MeshHandle<>& mesh, std::span<const Measure> values, std::size_t byte_budget);
} // namespace SimpleFluid::detail::general_vof::exact
