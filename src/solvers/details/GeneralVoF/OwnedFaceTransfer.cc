/** @file OwnedFaceTransfer.cc
 * @brief Supported Ordinary/Bounded canonical face transfer instantiations.
 */
#include "solvers/details/GeneralVoF/OwnedFaceTransfer.tcc"
namespace SimpleFluid::detail::general_vof
{
template FaceTransferResult transfer_owned_faces<ArithmeticMode::Ordinary>(
    const FaceTransferSource&, const FaceTransferTarget&, FaceTransferControls);
template FaceTransferResult transfer_owned_faces<ArithmeticMode::BoundedCertification>(
    const FaceTransferSource&, const FaceTransferTarget&, FaceTransferControls);
} // namespace SimpleFluid::detail::general_vof
