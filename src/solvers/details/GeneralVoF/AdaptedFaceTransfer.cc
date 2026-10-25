/** @file AdaptedFaceTransfer.cc
 * @brief Supported Ordinary/Bounded canonical face transfer instantiations.
 */
#include "solvers/details/GeneralVoF/AdaptedFaceTransfer.tcc"
namespace SimpleFluid::detail::general_vof
{
template FaceTransferResult transfer_adapted_faces<ArithmeticMode::Ordinary>(const FaceTransferSource&,
    const FaceTransferTarget&, FaceTransferControls, const VoF::GeneralVoF::FiniteRemapTrial&);
template FaceTransferResult transfer_adapted_faces<ArithmeticMode::BoundedCertification>(const FaceTransferSource&,
    const FaceTransferTarget&, FaceTransferControls, const VoF::GeneralVoF::FiniteRemapTrial&);
} // namespace SimpleFluid::detail::general_vof
