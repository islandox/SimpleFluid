/** @file ConservativeVoFExactArithmetic.hh
 * @brief Bounded exact normalization and original physical residual reductions.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/MeshHandle.hh"
namespace SimpleFluid::detail::vof_exact
{
using Measure = Meshes::ExactGeometryMeasure;
/** Addition/scaling admit bounded retained exact scalar growth, never a
 * floating substitute. Division/squaring use the core bit-growth preflight.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure add(const Measure&, const Measure&, std::size_t budget);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure scale(const Measure&, double, std::size_t budget);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure divide(const Measure&, const Measure&, std::size_t budget);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure square(const Measure&, std::size_t budget);
/** Caller reserves a separate16*K operation workspace in addition to ALL
 * persistent and pending temporary scalar slots. Each input/result is capped
 * by K; one synchronous operation may use16*K for its conservative
 *8*(input_a+input_b) growth lease. Legacy routines retain their single-budget
 * contract. No pending intermediate/corner slots are borrowed for this lease. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure leased_add(const Measure&, const Measure&, std::size_t result_cap);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure leased_scale(const Measure&, double, std::size_t result_cap);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure leased_divide(const Measure&, const Measure&, std::size_t result_cap);
/** Explicit binary64 inner view plus outward conversion enclosure. A rounded
 * zero keeps exact_nonzero; the exact physical operator remains authoritative.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL double view(const Measure&, Meshes::CertifiedGeometryValue& enclosure);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure maximum_absolute(
    const std::vector<Measure>&, const DefaultTpetraTypes::map_type&, std::size_t budget);
/** Leased pressure reduction: caller reserves16*K operation storage and a
 * separate communication arena. Inputs/result remain capped by K; rank-local
 * communication admission includes exports, engine buffers and broadcasts.
 * Legacy maximum_absolute keeps its original single-budget contract. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Measure leased_maximum_absolute(const std::vector<Measure>&,
    const DefaultTpetraTypes::map_type&, std::size_t result_cap, std::size_t communication_arena);
struct PhysicalNorm
{
    bool accepted = false;
    double relative_residual = 0;
};
/** Same original L2 norm as the scalar backend, including absolute reference1
 * when the RHS is identically zero. Only aggregate scalar certificates move.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL PhysicalNorm physical_norm(const std::vector<Measure>& residual,
    const std::vector<Measure>& rhs, double tolerance, const DefaultTpetraTypes::map_type&, std::size_t budget);
/** Separate total scratch lease for exact-input momentum. Caller retains inputs;
 * eight pending scalar slots,16 operation slots and a disjoint packet arena fit
 * this total. Legacy physical_norm retains its original budget contract. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL PhysicalNorm leased_physical_norm(const std::vector<Measure>& residual,
    const std::vector<Measure>& rhs, double tolerance, const DefaultTpetraTypes::map_type&, std::size_t total_budget);
} // namespace SimpleFluid::detail::vof_exact
