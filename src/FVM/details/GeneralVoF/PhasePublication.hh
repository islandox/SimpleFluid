/** @file PhasePublication.hh
 * @brief Internal finite phase publication and post-reduction conservation checks.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include <array>
namespace SimpleFluid::VoF::general_vof_detail
{
struct Cell;
/** Reduced phase pairs: change [0,1], publication uncertainty [2,3],
 * magnitude [4,5], and old inventory [6,7]. */
using PhaseConservationSummary = std::array<double, 8>;
/** Ordinary admits the physical relative allowance; Bounded uses arithmetic uncertainty only. */
enum class PhaseConservationPolicy
{
    Ordinary,
    Bounded
};
/** Outcome of the independent gas and carrier post-reduction checks. */
enum class PhaseConservationResult
{
    Accepted,
    ReductionBoundExhausted,
    Nonconserved
};
/** Check already reduced finite phase summaries without communication or exceptions.
 * Ordinary and Bounded policies are compiled in PhasePublication.cc. */
template<PhaseConservationPolicy Policy>
SIMPLEFLUID_FVM_LOCAL PhaseConservationResult check_phase_conservation(
    const PhaseConservationSummary&, int ranks, double relative_tolerance) noexcept;
/** Clear the six owned publication arrays while retaining their capacity. */
SIMPLEFLUID_FVM_LOCAL void reset_finite_phase_publication(GeneralVoF::Data&) noexcept;
/** Append already computed cell values; the caller coordinates allocation failures collectively. */
SIMPLEFLUID_FVM_LOCAL void append_finite_phase_publication(GeneralVoF::Data&, const Cell&);
} // namespace SimpleFluid::VoF::general_vof_detail
