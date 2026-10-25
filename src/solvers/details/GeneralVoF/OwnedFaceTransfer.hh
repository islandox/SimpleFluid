/** @file OwnedFaceTransfer.hh
 * @brief Collective canonical face transfer without solver-state ownership.
 */
#pragma once
#include "solvers/details/GeneralVoF/FaceTransfer.hh"
namespace SimpleFluid::detail::general_vof
{
/** Collective on source and target communicators, including empty owners.
 * Rejection leaves both publications unchanged. */
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL FaceTransferResult transfer_owned_faces(
    const FaceTransferSource& source_input, const FaceTransferTarget& target, FaceTransferControls controls);
} // namespace SimpleFluid::detail::general_vof
