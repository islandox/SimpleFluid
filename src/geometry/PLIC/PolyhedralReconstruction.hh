/** @file PolyhedralReconstruction.hh
 * @brief Physical-cell volume-fit PLIC reconstruction on certified convex cells.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <span>
#include <stdexcept>

namespace SimpleFluid::VoF
{
/** Both represented extensive phases are authoritative, including minority tails.
 * Geometry is the actual physical cell, never a bounding-box replacement. */
struct PolyhedralPhaseSample
{
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
    double gas_volume = 0, carrier_volume = 0;
};
/** Fixed-mode reconstruction input. Select the mode before entering a cell or
 * stencil loop; all geometry operations retain that mode through the fit. */
template<ArithmeticMode Mode> struct PolyhedralPhaseSampleFor
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    PLIC::ConvexGeometryPtr<Mode> geometry;
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
    SimpleFluid::PLIC::GeometryHalfspace gas;
    double target_volume_residual = 0, stencil_volume_residual = 0;
    SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>
        target_volume_difference;                               ///< Exact-mode signed minority residual.
    PLIC::DynamicGeometryValue finite_target_volume_difference; ///< Finite construction/arithmetic enclosure.
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
    /** Compiled fixed-mode entries for callers that already selected their
     * arithmetic policy. No runtime adapter is entered during the fit. */
    template<ArithmeticMode Mode>
    static void validate_finite(const PolyhedralPhaseSampleFor<Mode>&, PolyhedralReconstructionOptions = {});
    /** Read-only preflight. Missing/dilute support requests refinement; the
     * angular model gate remains independent of arithmetic admission. */
    template<ArithmeticMode Mode>
    [[nodiscard]] static PolyhedralResolutionAssessment assess_resolution_finite(const PolyhedralPhaseSampleFor<Mode>&,
        std::span<const PolyhedralPhaseSampleFor<Mode>>, PolyhedralReconstructionOptions = {});
    /** Offset utility for a supplied normal. Physical interface admission
     * belongs to reconstruct_finite()/assess_resolution_finite(). */
    template<ArithmeticMode Mode>
    [[nodiscard]] static PolyhedralInterface solve_offset_finite(
        const PolyhedralPhaseSampleFor<Mode>&, SimpleFluid::PLIC::GeometryPoint, PolyhedralReconstructionOptions = {});
    template<ArithmeticMode Mode>
    [[nodiscard]] static PolyhedralInterface reconstruct_finite(const PolyhedralPhaseSampleFor<Mode>&,
        std::span<const PolyhedralPhaseSampleFor<Mode>>, PolyhedralReconstructionOptions = {});
};
namespace Exact
{
class SIMPLEFLUID_FVM_EXPORT PolyhedralReconstruction
{
public:
    static void validate(const PolyhedralPhaseSample&, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface solve_offset(const PolyhedralPhaseSample&,
        SimpleFluid::PLIC::GeometryPoint gas_normal, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const PolyhedralPhaseSample&,
        std::span<const PolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions = {});
};
} // namespace Exact
} // namespace SimpleFluid::VoF
