/** @file PolyhedralReconstruction.hh
 * @brief Physical-cell volume-fit PLIC reconstruction on certified convex cells.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/ConvexCellGeometry.hh"
#include <span>

namespace SimpleFluid::VoF
{
/** Both represented extensive phases are authoritative, including minority tails.
 * Geometry is the actual physical cell, never a bounding-box replacement. */
struct PolyhedralPhaseSample
{
    Meshes::ConvexGeometryPtr geometry;
    double gas_volume = 0, carrier_volume = 0;
};
struct PolyhedralReconstructionOptions
{
    unsigned maximum_offset_iterations = 192, maximum_fit_iterations = 24;
    std::size_t maximum_stencil_cells = 128, maximum_clip_evaluations = 32768;
    double volume_tolerance = 8e-15, fit_tolerance = 2e-9;
    double maximum_condition = 1e8;
    double orientation_tolerance = 2e-8, maximum_model_residual = .2;
};
struct PolyhedralInterface
{
    Meshes::GeometryHalfspace gas;
    double target_volume_residual = 0, stencil_volume_residual = 0;
    Meshes::CertifiedGeometryValue target_volume_difference; ///< Exact signed minority residual and conversion bound.
    double stencil_condition = 0, orientation_step = 0;
    unsigned fit_iterations = 0;
    bool fit_converged = false;
    bool pure_gas = false, pure_carrier = false;
};
/** Stateless compiled kernel. The LS gradient supplies only an initial normal;
 * the accepted normal minimizes actual clipped-volume residuals with the target
 * offset solved anew at every angular evaluation. Curved stencils may have a
 * nonzero reported model residual; angular optimizer convergence and the explicit
 * model-defect bound are distinct from the strict planar-volume oracle.
 * Insufficient or ill-conditioned
 * three-dimensional stencils reject rather than silently selecting an axis. */
class SIMPLEFLUID_FVM_EXPORT PolyhedralReconstruction
{
public:
    static void validate(const PolyhedralPhaseSample&, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface solve_offset(
        const PolyhedralPhaseSample&, Meshes::GeometryPoint gas_normal, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const PolyhedralPhaseSample&,
        std::span<const PolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions = {});
};
} // namespace SimpleFluid::VoF
