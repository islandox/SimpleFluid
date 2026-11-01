/** @file Reconstruction.hh
 * @brief Private observed Ordinary owner reconstruction used by sparse plane imports.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
/** Record carried metadata only; the caller already admitted this plane under
 * its immutable source options. No geometric measurement or fit is invoked. */
inline void record_transport_plane_quality(GeneralVoFReconstructionQualityDiagnostics& summary, std::uint64_t gid,
    const PLIC::PolyhedralInterface& plane)
{
    ++summary.plane_count;
    if (plane.model_policy == PLIC::PlicModelPolicy::ConservativeApproximation)
        ++summary.approximation_plane_count;
    else
        ++summary.strict_plane_count;
    if (!plane.stencil_volume_residual_measured)
        return;
    ++summary.measured_plane_count;
    if (plane.model_quality == PLIC::PlicModelQuality::AboveThreshold)
        ++summary.above_model_threshold_count;
    if (summary.measured_plane_count == 1 || plane.stencil_volume_residual > summary.maximum_model_residual ||
        (plane.stencil_volume_residual == summary.maximum_model_residual && gid < summary.worst_cell_gid))
    {
        summary.maximum_model_residual = plane.stencil_volume_residual;
        summary.worst_cell_gid = gid;
        summary.worst_model_policy = plane.model_policy;
        summary.worst_orientation_method = plane.orientation_method;
        summary.worst_fit_converged = plane.fit_converged;
    }
}
SIMPLEFLUID_FVM_LOCAL PLIC::PolyhedralInterface reconstruct_observed_ordinary_cell(const GeneralVoF::Data& data,
    const Key& key, GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics, GeneralVoFWorkCell* row);
} // namespace SimpleFluid::VoF::general_vof_detail
