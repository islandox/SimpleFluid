/** @file GeneralVoFCurvature.hh
 * @brief Conditioned physical surface fits from actual reconstructed PLIC sections.
 */
#pragma once
#include "FVM/PolyhedralReconstruction.hh"
#include "SimpleFluidExport.hh"
#include <optional>
#include <span>
namespace SimpleFluid::detail
{
/** Numerical surface sample built from an actual cell/PLIC intersection. */
struct GeneralVoFCurvaturePatch
{
    std::uint64_t cell = 0;
    Meshes::GeometryPoint centroid{}, normal{};
    double area = 0, cell_scale = 0;
    Meshes::GeometryPoint physical_spans{}; ///< Actual AABB spans; zero only for manufactured compatibility samples.
};
struct GeneralVoFCurvatureOptions
{
    std::size_t minimum_samples = 6;
    double support_radius = 6, minimum_normal_cosine = .5;
    double maximum_condition = 1e8, maximum_normalized_residual = .15, maximum_slope = 1;
};
struct GeneralVoFCurvatureResult
{
    double curvature = 0, condition = 0, normalized_residual = 0;
    std::size_t samples = 0;
};
/** Compiled numerical fit; source patch geometry/planes must be authenticated by
 * the transport authority before MPI support is supplied. Missing or deficient
 * local support rejects; no pure-cell zero-normal fallback is introduced.
 */
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFCurvature
{
public:
    /** Pure cells return no patch. A mixed cell must have a positive physical
     * section; centroid integrates its convex polygon in physical coordinates.
     */
    [[nodiscard]] static std::optional<GeneralVoFCurvaturePatch> patch(
        std::uint64_t cell, const Meshes::BoundedConvexGeometry&, const VoF::PolyhedralInterface&);
    /** Fit a quadratic Monge surface in the target normal's tangential frame.
     * Curvature is div(n) [1/m], positive for convex gas with outward normals.
     */
    [[nodiscard]] static GeneralVoFCurvatureResult fit(const GeneralVoFCurvaturePatch& target,
        std::span<const GeneralVoFCurvaturePatch> support, GeneralVoFCurvatureOptions = {});
};
namespace Exact
{
struct SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFCurvature
{
    [[nodiscard]] static std::optional<GeneralVoFCurvaturePatch> patch(
        std::uint64_t cell, const Meshes::ConvexCellGeometry&, const VoF::PolyhedralInterface&);
};
} // namespace Exact
} // namespace SimpleFluid::detail
