/** @file GeneralVoFExactPressure.hh
 * @brief Bounded exact CG on caller-certified orthogonal positive face graphs.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
namespace SimpleFluid::detail
{
struct GeneralVoFExactPressureFace
{
    DefaultTpetraTypes::global_ordinal_type face = 0, first = 0, second = 0;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>
        mobility; ///< Exact k=dt*beta*A/d, strictly positive.
};
struct GeneralVoFExactPressureResult
{
    std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> owned_pressure,
        overlap_pressure;
    std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>
        face_rate_increment; ///< Input face order; k*(pFirst-pSecond).
    int iterations = 0;
    std::size_t operator_applications = 0;
    std::size_t admitted_storage_bytes = 0, storage_bytes = 0;
};
/** Collective. Caller proves exact orthogonality; this helper does NOT qualify
 * a nonorthogonal gradient operator. Native canonical internal faces must be
 * covered once by actual face owners. rhs is -netq in owned-cell map order.
 * Solve A p=rhs in the compatible semidefinite graph space (no artificial gauge
 * row); pressure has the zero-start CG nullspace convention. Every nonzero
 * residual requires positive pAp. Exact recomputation A p-rhs==0 is mandatory.
 * CG steps consume the caller's remaining shared pressure iteration budget.
 * Operator applications (including final verification) are reported separately. Resource,
 * incompatible-component and algebraic breakdown failures reject collectively.
 * Storage bounds cover admitted application capacities, including a separate
 * 16*K scalar-operation lease, not MPI/backend/allocator process RSS.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFExactPressureResult solve_general_vof_exact_pressure(
    const SP<const MeshHandle<>>&, std::span<const GeneralVoFExactPressureFace>,
    std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> owned_rhs,
    int remaining_iterations, std::size_t byte_budget);
} // namespace SimpleFluid::detail
