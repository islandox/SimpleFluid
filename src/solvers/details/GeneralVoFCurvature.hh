/** @file GeneralVoFCurvature.hh
 * @brief Conditioned physical surface fits from actual reconstructed PLIC sections.
 */
#pragma once
#include "FVM/GeneralVoFSurface.hh"
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include <optional>
#include <span>
namespace SimpleFluid::detail
{
/** Numerical surface sample built from an actual cell/PLIC intersection. */
struct GeneralVoFCurvaturePatch
{
    std::uint64_t cell = 0;
    SimpleFluid::PLIC::GeometryPoint centroid{}, normal{};
    double area = 0, cell_scale = 0;
    SimpleFluid::PLIC::GeometryPoint
        physical_spans{}; ///< Actual AABB spans; zero only for manufactured compatibility samples.
    std::optional<VoF::GeneralVoFSurfaceKey> key; ///< Unkeyed compatibility samples use their mixed-cell identity.
};
struct GeneralVoFCurvatureOptions
{
    std::size_t minimum_samples = 6;
    double support_radius = 6, minimum_normal_cosine = .5;
    double maximum_condition = 1e8, maximum_normalized_residual = .15, maximum_slope = 1;
};
struct GeneralVoFCurvatureResult
{
    double curvature = 0;
    double condition = 0; ///< Squared infinity-norm condition estimate of the coordinate-scaled QR factor.
    double normalized_residual = 0;
    std::size_t samples = 0;
};
/** Compiled numerical fit; source patch geometry/planes must be authenticated by
 * the surface authority before MPI support is supplied. The caller supplies a
 * connected, topology-authenticated neighborhood.
 * Missing or deficient local support rejects; no pure-cell zero-normal fallback is introduced.
 */
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFCurvature
{
public:
    /** Pure cells return no patch. A mixed cell must have a positive physical
     * section; centroid integrates its convex polygon in physical coordinates.
     */
    template<ArithmeticMode Mode>
        requires(Mode == ArithmeticMode::Exact)
    [[nodiscard]] static std::optional<GeneralVoFCurvaturePatch> patch(
        std::uint64_t cell, const PLIC::ConvexCellGeometry<Mode>&, const PLIC::PolyhedralInterface&);
    /** Finite physical coordinates are published within the caller's cumulative admission scope. */
    template<ArithmeticMode Mode>
        requires(Mode != ArithmeticMode::Exact)
    [[nodiscard]] static std::optional<GeneralVoFCurvaturePatch> patch(std::uint64_t cell,
        const PLIC::ConvexCellGeometry<Mode>&, const PLIC::PolyhedralInterface&,
        PLIC::GeometryAllocationLedger&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Fit a quadratic Monge surface in the target normal's tangential frame.
     * Curvature is div(n) [1/m], positive for convex gas with outward normals.
     */
    [[nodiscard]] static GeneralVoFCurvatureResult fit(const GeneralVoFCurvaturePatch& target,
        std::span<const GeneralVoFCurvaturePatch> support, GeneralVoFCurvatureOptions = {});
};
template<>
[[nodiscard]] std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch<ArithmeticMode::Exact>(
    std::uint64_t cell, const PLIC::ConvexCellGeometry<ArithmeticMode::Exact>&, const PLIC::PolyhedralInterface&);
} // namespace SimpleFluid::detail
