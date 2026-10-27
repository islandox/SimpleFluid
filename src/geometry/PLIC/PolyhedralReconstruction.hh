/** @file PolyhedralReconstruction.hh
 * @brief Physical-cell volume-fit PLIC reconstruction on certified convex cells.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <span>
#include <stdexcept>

namespace SimpleFluid::PLIC
{
/** Both represented extensive phases are authoritative, including minority tails.
 * Geometry retains the selected arithmetic mode through every fit operation.
 * Exact callers explicitly lift represented native geometry at their selected
 * backend entry; that lift does not recover earlier ordinary construction history. */
template<ArithmeticMode Mode> struct PolyhedralPhaseSample
{
    ConvexGeometryPtr<Mode> geometry;
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
    /** Opt-in Ordinary boundary-integration path. BoundedCertification and
     * Exact retain their existing arithmetic/construction paths. */
    bool measure_only = false;
    unsigned maximum_offset_iterations = 192, maximum_fit_iterations = 24;
    std::size_t maximum_stencil_cells = 128, maximum_clip_evaluations = 32768;
    /** Relative to the independently supplied smaller phase. Ordinary fits
     * include their numerical roundoff indicator and reject below that floor;
     * bounded fits retain the construction/arithmetic enclosure gate. */
    double volume_tolerance = 8e-15, fit_tolerance = 2e-9;
    double maximum_condition = 1e8;
    double orientation_tolerance = 2e-8, maximum_model_residual = .2;
    /** Each phase must occupy at least this fraction of some target/stencil
     * cell before a mixed finite reconstruction. This requests more spatial
     * support; it never erases or clips a smaller target-cell phase amount.
     * The default requires majority support on both sides of an interface. */
    double minimum_phase_support_fraction = .5;
    /** Retain the supported weighted-gradient normal when the
     * volume-equivalent minority/cell length ratio cbrt(min(G,L))/cbrt(Vcell)
     * is strictly below this value. Only the target offset is then fitted;
     * angular/model-residual gates do not run. Zero disables this exclusion.
     * This provisional scale criterion never changes either phase inventory. */
    double minimum_optimized_phase_length_ratio = .05;
};
enum class PolyhedralOrientationMethod
{
    SuppliedNormal,
    Gradient,
    AngularVolumeFit
};
struct PolyhedralInterface
{
    GeometryHalfspace gas;
    double target_volume_residual = 0;
    /** Kernel angular-model residual. Gradient results leave it zero and
     * unmeasured; callers admitting supplied planes may publish their own RMS. */
    double stencil_volume_residual = 0;
    GeometryValue<ArithmeticMode::Exact> target_volume_difference; ///< Exact-mode signed minority residual.
    DynamicGeometryValue finite_target_volume_difference;          ///< Finite construction/arithmetic enclosure.
    /** Ordinary-only roundoff indicator; this is not a certified enclosure. */
    long double target_volume_numerical_error_estimate = 0;
    bool finite_geometry = false; ///< Selects numerical residual evidence; no exact authority is implied.
    PolyhedralResolutionAssessment resolution;
    double stencil_condition = 0, orientation_step = 0;
    unsigned fit_iterations = 0;
    PolyhedralOrientationMethod orientation_method = PolyhedralOrientationMethod::SuppliedNormal;
    /** Angular optimizer convergence only. Gradient/supplied-normal results
     * leave this false; their target offset still passes its strict gate. */
    bool fit_converged = false;
    bool pure_gas = false, pure_carrier = false;
};
/** Stateless compiled kernel. The supported LS gradient supplies the normal for
 * subthreshold phases; otherwise it initializes the angular volume fit,
 * with the target offset solved anew at every angular evaluation. Curved
 * stencils may have a nonzero reported model residual; angular optimizer convergence and the explicit
 * model-defect bound are distinct from the strict planar-volume oracle.
 * Insufficient or ill-conditioned
 * three-dimensional stencils reject rather than silently selecting an axis. */
template<ArithmeticMode Mode> class SIMPLEFLUID_FVM_EXPORT PolyhedralReconstruction
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);

public:
    static void validate(const PolyhedralPhaseSample<Mode>&, PolyhedralReconstructionOptions = {});
    /** Read-only preflight. Missing/dilute support requests refinement; the
     * angular model gate remains independent of arithmetic admission. */
    [[nodiscard]] static PolyhedralResolutionAssessment assess_resolution(const PolyhedralPhaseSample<Mode>&,
        std::span<const PolyhedralPhaseSample<Mode>>, PolyhedralReconstructionOptions = {});
    /** Offset utility for a supplied normal. Physical interface admission
     * belongs to reconstruct()/assess_resolution(). */
    [[nodiscard]] static PolyhedralInterface solve_offset(
        const PolyhedralPhaseSample<Mode>&, GeometryPoint, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const PolyhedralPhaseSample<Mode>&,
        std::span<const PolyhedralPhaseSample<Mode>>, PolyhedralReconstructionOptions = {});
};
/** Explicit optional exact backend. The same small-phase orientation policy
 * retains its strict exact offset arithmetic. The finite spatial-resolution
 * preflight is not an exact-backend admission gate. */
template<> class SIMPLEFLUID_FVM_EXPORT PolyhedralReconstruction<ArithmeticMode::Exact>
{
public:
    static void validate(const PolyhedralPhaseSample<ArithmeticMode::Exact>&, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface solve_offset(const PolyhedralPhaseSample<ArithmeticMode::Exact>&,
        GeometryPoint gas_normal, PolyhedralReconstructionOptions = {});
    [[nodiscard]] static PolyhedralInterface reconstruct(const PolyhedralPhaseSample<ArithmeticMode::Exact>&,
        std::span<const PolyhedralPhaseSample<ArithmeticMode::Exact>> stencil, PolyhedralReconstructionOptions = {});
};
extern template class PolyhedralReconstruction<ArithmeticMode::Ordinary>;
extern template class PolyhedralReconstruction<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::PLIC
