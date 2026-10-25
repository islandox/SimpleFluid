/** @file AdaptedFaceTransfer.hh
 * @brief Collective canonical face transfer without solver-state ownership.
 */
#pragma once
#include "solvers/details/GeneralVoF/FaceTransfer.hh"
namespace SimpleFluid::detail::general_vof
{
/** Collective on source and target communicators, including empty owners.
 * Rejection leaves both publications unchanged. */
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL FaceTransferResult transfer_adapted_faces(
    const FaceTransferSource& source_input, const FaceTransferTarget& target, FaceTransferControls controls,
    const VoF::GeneralVoF::FiniteRemapTrial& trial);
} // namespace SimpleFluid::detail::general_vof
