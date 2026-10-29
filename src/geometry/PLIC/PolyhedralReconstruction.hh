/** @file PolyhedralReconstruction.hh
 * @brief Physical-cell volume-fit PLIC reconstruction on certified convex cells.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include <span>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <vector>

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
    std::uint64_t id = std::numeric_limits<std::uint64_t>::max(); ///< Optional caller's persistent cell ID.
    /** Transport support evidence only. Independent inventory on validated pure
     * geometry is not an observed mixed-interface angular constraint. */
    bool observed_mixed = true;
};
/** Evidence from the final retained (and best accepted) angular iterate. */
struct PolyhedralFitDiagnostics
{
    GeometryPoint normal{}, best_normal{};
    double rms = 0, best_rms = 0, angular_step = 0, finite_difference_scale = 0, condition = 0;
    unsigned iterations = 0;
    std::size_t clip_evaluations = 0;
    std::vector<std::uint64_t> stencil_ids;
};
/** Numerical nonstationarity is distinct from an incompatible plane model. */
class SIMPLEFLUID_FVM_EXPORT PolyhedralFitFailure : public GeometryError
{
public:
    enum class Reason { Nonstationary, IterationLimit, Unobservable, Geometry };
    PolyhedralFitFailure(Reason, PolyhedralFitDiagnostics);
    PolyhedralFitFailure(Reason, const GeometryError&, PolyhedralFitDiagnostics);
    [[nodiscard]] Reason reason() const noexcept { return reason_; }
    [[nodiscard]] const PolyhedralFitDiagnostics& diagnostics() const noexcept { return diagnostics_; }
private:
    Reason reason_;
    PolyhedralFitDiagnostics diagnostics_;
};
/** Directional admission, separate from arithmetic acceptance and spatial
 * convergence qualification. A failed assessment requests fresh support. */
struct PolyhedralResolutionAssessment
{
    enum class Reason
    {
        Admitted,
        InsufficientStencil,
        InsufficientPhaseSupport,
        UnresolvedPlaneModel,
        RankDeficientStencil,
        IllConditionedStencil,
        NoDirectionalSignal,
        RankDeficientAngularSupport,
        IllConditionedAngularSupport
    };
    Reason reason = Reason::Admitted;
    double local_cell_span = 0, requested_cell_span = 0;
    double gas_support_fraction = 0, carrier_support_fraction = 0;
    /** Dimensionless LS conditioning and direction strength. Uncertainty is a
     * propagated roundoff estimate, not a certified enclosure. */
    double stencil_condition = 0, directional_strength = 0, directional_uncertainty = 0;
    /** Transport-only observed mixed-neighbor support in the seed tangent
     * plane. Not a clipped-volume optimizer convergence certificate. */
    double angular_support_condition = 0;
    std::size_t angular_support_cells = 0;
    /** True only after directional admission and sufficient phase support. */
    bool angular_fit_eligible = false;
    std::size_t stencil_cells = 0;
    [[nodiscard]] bool admitted() const noexcept { return reason == Reason::Admitted; }
};
/** The caller may refine/reinitialize the interface neighborhood and retry.
 * Without an adaptation controller the operation rejects without mutation. */
class SIMPLEFLUID_FVM_EXPORT PolyhedralResolutionRequired : public std::invalid_argument
{
public:
    explicit PolyhedralResolutionRequired(PolyhedralResolutionAssessment);
    PolyhedralResolutionRequired(PolyhedralResolutionAssessment, PolyhedralFitDiagnostics);
    [[nodiscard]] const PolyhedralResolutionAssessment& assessment() const noexcept { return assessment_; }
    [[nodiscard]] const PolyhedralFitDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    [[nodiscard]] bool has_fit_diagnostics() const noexcept { return has_fit_diagnostics_; }

private:
    PolyhedralResolutionAssessment assessment_;
    PolyhedralFitDiagnostics diagnostics_;
    bool has_fit_diagnostics_ = false;
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
     * cell for angular fitting. A usable gradient remains admissible below
     * this support threshold. The plane-model bound applies unless the
     * existing small-phase exclusion below suppresses neighbor fitting. */
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
    /** Neighbor-volume RMS when measured. The supported small-phase gradient
     * exclusion leaves it zero and unmeasured. */
    double stencil_volume_residual = 0;
    bool stencil_volume_residual_measured = false;
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
 * weak angular support and supported subthreshold phases; otherwise it initializes the angular volume fit,
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
    /** Read-only preflight of rank, conditioning and meaningful direction.
     * Majority-phase support selects angular eligibility rather than admission. */
    [[nodiscard]] static PolyhedralResolutionAssessment assess_resolution(const PolyhedralPhaseSample<Mode>&,
        std::span<const PolyhedralPhaseSample<Mode>>, PolyhedralReconstructionOptions = {});
    /** Geometry-only complete-ring selection preflight. No fractions, signal,
     * phase support or neighbor volume-model residual determine selection. */
    [[nodiscard]] static PolyhedralResolutionAssessment assess_stencil_geometry(const PolyhedralPhaseSample<Mode>&,
        std::span<const PolyhedralPhaseSample<Mode>>, PolyhedralReconstructionOptions = {});
    /** Complete-ring transport admission additionally requires two resolved
     * observed mixed-neighbor directions for the angular branch. This contains
     * no clipping or model-residual evaluation. Explicit-stencil reconstruct()
     * retains its separate optimizer/model contract. */
    [[nodiscard]] static PolyhedralResolutionAssessment assess_transport_support(const PolyhedralPhaseSample<Mode>&,
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
