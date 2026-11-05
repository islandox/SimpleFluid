/** @file CellGeometryEvaluation.hh
 * @brief Private evaluation state over immutable mesh-input representations.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"

namespace SimpleFluid::PLIC::details
{
/** Observe cache completion when the immutable source binding is published. */
template<ArithmeticMode Mode>
    requires(Mode != ArithmeticMode::Exact)
[[nodiscard]] SIMPLEFLUID_LOCAL bool cell_evaluation_cache_ready(const ConvexGeometryPtr<Mode>&);
extern template bool cell_evaluation_cache_ready(const ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
extern template bool cell_evaluation_cache_ready(const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&);
/** Share validated numerical geometry and certificates with a distinct moment
 * cache. The immutable caller policy selects completed factory facts or a
 * fresh lazy cache, independently of later template queries. Explicit caller-owned inputs may deliberately retain
 * their cache instead and should bypass this factory. */
template<ArithmeticMode Mode>
    requires(Mode != ArithmeticMode::Exact)
[[nodiscard]] SIMPLEFLUID_LOCAL ConvexGeometryPtr<Mode> clone_cell_evaluation(
    const ConvexGeometryPtr<Mode>&, bool copy_established_integrals, GeometryLimits = {});
extern template ConvexGeometryPtr<ArithmeticMode::Ordinary> clone_cell_evaluation(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, bool, GeometryLimits);
extern template ConvexGeometryPtr<ArithmeticMode::BoundedCertification> clone_cell_evaluation(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, bool, GeometryLimits);
} // namespace SimpleFluid::PLIC::details
