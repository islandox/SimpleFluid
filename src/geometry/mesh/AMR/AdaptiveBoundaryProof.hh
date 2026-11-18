/** @file AdaptiveBoundaryProof.hh
 * @brief Exact physical boundary validation across periodic support images.
 */
#pragma once
#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct AdaptiveBoundaryProofStatistics
{
    std::size_t source_faces = 0, pieces = 0, peak_bytes = 0, pair_tests = 0;
};
/** A validated convex source plus unique outward containment and a complete
 * exact partition of each original face proves the same closed boundary.
 * Keep the existing 4096 face/vertex and 262144 aggregate proof-work limits,
 * without re-importing a label-subdivided shell into another convex geometry.
 * Coordinates and construction identities are unchanged. Admission covers
 * retained inputs and the additional physical boundary proof workspace.
 * When supplied, whole-face proofs align with the pieces and authenticate the
 * actual complete source face for each non-null entry. */
void certify_adaptive_physical_boundary(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&,
    std::span<const SupportFace>, std::size_t byte_budget, AdaptiveBoundaryProofStatistics* = nullptr,
    std::span<const SupportFace> whole_faces = {});
/** Private qualified seam: source/native equivalence has already authenticated
 * every outward native face. Recheck a changed incidence partition without
 * constructing another source-cell shell. */
void certify_adaptive_physical_boundary_faces(std::span<const SupportFace> source, std::span<const SupportFace> pieces,
    std::size_t byte_budget, AdaptiveBoundaryProofStatistics* = nullptr);
void certify_adaptive_physical_boundary_native(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&,
    std::span<const SupportFace>, std::size_t byte_budget, const geometry_detail::NativeConstructionAdmission&,
    std::span<const SupportFace> whole_faces = {});
void certify_adaptive_physical_boundary_faces_native(std::span<const SupportFace>, std::span<const SupportFace>,
    std::size_t byte_budget, const geometry_detail::NativeConstructionAdmission&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
