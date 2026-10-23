/** @file PolyhedralReconstruction.hh
 * @brief Physical-cell volume-fit PLIC reconstruction on certified convex cells.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/BoundedConvexGeometry.hh"
#include <span>
#include <stdexcept>

namespace SimpleFluid::VoF
{
/** Both represented extensive phases are authoritative, including minority tails.
 * Geometry is the actual physical cell, never a bounding-box replacement. */
struct PolyhedralPhaseSample
{
    Meshes::ConvexGeometryPtr geometry;
    double gas_volume = 0, carrier_volume = 0;
};
/** Independent finite inventories on geometry carrying explicit uncertainty. */
struct FinitePolyhedralPhaseSample
{
    Meshes::BoundedConvexGeometryPtr geometry;
    double gas_volume = 0, carrier_volume = 0;
};
/** A necessary interface-support screen, separate from arithmetic acceptance and
 * spatial convergence qualification. Small cuts use their whole neighborhood.
 * A failed assessment requests one refinement level and a fresh assessment. */
struct PolyhedralResolutionAssessment
{
    enum class Reason
    {
        Admitted,
        InsufficientStencil,
        InsufficientPhaseSupport,
        UnresolvedPlaneModel
    };
    Reason reason = Reason::Admitted;
    double local_cell_span = 0, requested_cell_span = 0;
    double gas_support_fraction = 0, carrier_support_fraction = 0;
    std::size_t stencil_cells = 0;
    [[nodiscard]] bool admitted() const noexcept { return reason == Reason::Admitted; }
};
/** The caller may refine/reinitialize the interface neighborhood and retry.
 * Without an adaptation controller the operation rejects without mutation. */
class SIMPLEFLUID_FVM_EXPORT PolyhedralResolutionRequired : public std::invalid_argument
{
public:
    explicit PolyhedralResolutionRequired(PolyhedralResolutionAssessment);
    [[nodiscard]] const PolyhedralResolutionAssessment& assessment() const noexcept { return assessment_; }

private:
    PolyhedralResolutionAssessment assessment_;
};
struct PolyhedralReconstructionOptions
{
    unsigned maximum_offset_iterations = 192, maximum_fit_iterations = 24;
    std::size_t maximum_stencil_cells = 128, maximum_clip_evaluations = 32768;
    double volume_tolerance = 8e-15, fit_tolerance = 2e-9;
    double maximum_condition = 1e8;
    double orientation_tolerance = 2e-8, maximum_model_residual = .2;
    /** Each phase must occupy at least this fraction of some target/stencil
     * cell before a mixed finite reconstruction. This requests more spatial
     * support; it never erases or clips a smaller target-cell phase amount.
     * The default requires majority support on both sides of an interface. */
    double minimum_phase_support_fraction = .5;
};
struct PolyhedralInterface
{
    Meshes::GeometryHalfspace gas;
    double target_volume_residual = 0, stencil_volume_residual = 0;
    Meshes::CertifiedGeometryValue target_volume_difference; ///< Exact-mode signed minority residual.
    Meshes::BoundedGeometryValue finite_target_volume_difference; ///< Finite construction/arithmetic enclosure.
    bool finite_geometry = false; ///< Selects numerical residual evidence; no exact authority is implied.
    PolyhedralResolutionAssessment resolution;
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
    static void validate(const FinitePolyhedralPhaseSample&, PolyhedralReconstructionOptions = {});
    /** Read-only preflight. Missing/dilute support requests refinement, never
     * increased exact precision. The angular model gate remains independent. */
    [[nodiscard]] static PolyhedralResolutionAssessment assess_resolution(
        const FinitePolyhedralPhaseSample&, std::span<const FinitePolyhedralPhaseSample> stencil,
        PolyhedralReconstructionOptions = {});
    /** Numerical offset utility for a supplied normal. Physical interface
     * admission belongs to reconstruct()/assess_resolution(). */
    [[nodiscard]] static PolyhedralInterface solve_offset(
        const FinitePolyhedralPhaseSample&, Meshes::GeometryPoint gas_normal, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const FinitePolyhedralPhaseSample&,
        std::span<const FinitePolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions = {});
};
namespace Exact
{
class SIMPLEFLUID_FVM_EXPORT PolyhedralReconstruction
{
public:
    static void validate(const PolyhedralPhaseSample&, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface solve_offset(
        const PolyhedralPhaseSample&, Meshes::GeometryPoint gas_normal, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const PolyhedralPhaseSample&,
        std::span<const PolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions = {});
};
} // namespace Exact
} // namespace SimpleFluid::VoF
