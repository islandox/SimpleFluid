/** @file ConservativeVoFTransport.hh
 * @brief Implicit conservative tracer transport with prescribed extensive storage and flux.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include <memory>

namespace SimpleFluid::detail
{
/** Private native-grid transport boundary shared by VoF momentum and species.
 *
 * Solves M_new*c_new + div(F*c_new_upwind) = Q_old + S. F is an
 * integrated carrier flux; M_new is the exact final carrier inventory, including
 * its compression/source treatment. It is never inferred from a donor velocity.
 * With M_new + div(F) = M_old > 0 and S=0 this is a conservative M-matrix
 * update preserving constant tracers and their bounds. No phase division is
 * needed for momentum: its carrier is the strictly positive mixture mass.
 * All calls are collective, including ranks with no owned cells. Workspaces
 * belong to the context, never to an accepted snapshot or retained trial.
 */
class SIMPLEFLUID_SOLVERS_LOCAL ConservativeVoFTransport
{
public:
    struct Result
    {
        std::vector<double> amount, flux, trace;
        int physical_refinements = 0, iterations = 0;
        double physical_relative_residual = 0;
        /** Exact-input calls retain the implicit owned trace directly. Their
         * legacy flux/trace arrays are empty; exact_flux and flux_enclosure use
         * OWNED canonical face order and must not be confused with scalar views.
         */
        bool exact_input = false;
        std::vector<Meshes::CertifiedGeometryValue> inner_conversion_enclosures;
        std::vector<double> cell_trace;
        std::vector<Meshes::ExactGeometryMeasure> exact_flux;
        std::vector<Meshes::CertifiedGeometryValue> flux_enclosure;
    };
    explicit ConservativeVoFTransport(SP<const MeshHandle<>> mesh);
    /** Extensive Q_old/S/M_new; overlap-face F in canonical owner orientation.
     * Optional row equilibration preserves microscopic carrier rows without a
     * reciprocal diagonal overflow; the original physical residual still must
     * meet tolerance. Attainable scaled solves supply candidates and bounded
     * physical-defect corrections within one iteration budget. Scaling changes
     * conditioning, not the transport equation or its physical acceptance.
     */
    [[nodiscard]] Result solve(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& flux, bool nonnegative, int maximum_iterations,
        double tolerance = 1e-12, bool equilibrate_rows = false);

    /** Independent nonnegative owner-to-neighbor and neighbor-to-owner amounts.
     * Counterflow survives assembly, physical residuals and flux publication.
     * Each direction uses the incident donor's implicit trace.
     */
    [[nodiscard]] Result solve_streams(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& forward, const std::vector<double>& reverse,
        bool nonnegative, int maximum_iterations, double tolerance = 1e-12, bool equilibrate_rows = false);

    /** Positive nonself donor-to-recipient amount, stored on recipient owner.
     * Every (recipient, donor) pair must be aggregated exactly once by the
     * certified ledger; duplicate routes reject before numerical assembly.
     */
    struct Transfer
    {
        DefaultTpetraTypes::global_ordinal_type donor = 0, recipient = 0;
        double amount = 0;
    };
    /** Raw canonical owned-face contribution and its original geometric donor. */
    struct FaceContribution
    {
        DefaultTpetraTypes::local_ordinal_type face = 0;
        DefaultTpetraTypes::global_ordinal_type donor = 0;
        double amount = 0; ///< Signed integrated carrier amount.
        /** Optional unrounded signed mass group. Sparse physical fluxes apply
         * the implicit donor trace before exact face aggregation/narrowing.
         * The double amount remains its coefficient-view conversion for A.
         */
        std::shared_ptr<const Meshes::ExactGeometryMeasure> exact_amount;
    };
    /** Certified closed-cell pullback assembly. The caller supplies independently
     * certified donor outgoing amounts and positive recipient-owned transfers;
     * self-retained transfers are excluded. Every original signed face stream
     * remains in the physical residual, including nonincident geometric donors.
     * Sparse routes precondition that same equation; they do not replace its
     * acceptance or relax its physical tolerance.
     */
    [[nodiscard]] Result solve_transfers(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& outgoing, const std::vector<Transfer>& incoming,
        const std::vector<FaceContribution>& contributions, bool nonnegative, int maximum_iterations,
        double tolerance = 1e-12, bool equilibrate_rows = true, std::size_t byte_budget = 64 * 1024 * 1024);

    struct ExactTransfer
    {
        DefaultTpetraTypes::global_ordinal_type donor = 0, recipient = 0;
        Meshes::ExactGeometryMeasure amount;
    };
    struct ExactFaceContribution
    {
        DefaultTpetraTypes::local_ordinal_type face = 0;
        DefaultTpetraTypes::global_ordinal_type donor = 0;
        Meshes::ExactGeometryMeasure amount; ///< Signed canonical carrier measure.
    };
    /** Form row diagonals, coefficients and RHS ratios exactly before narrowing.
     * The original physical quantity/face equation and shared correction budget
     * remain authoritative. Only final owned amounts require strict binary64
     * publication; internal exact face fluxes have typed diagnostic enclosures.
     */
    [[nodiscard]] Result solve_exact(const std::vector<Meshes::ExactGeometryMeasure>& old,
        const std::vector<Meshes::ExactGeometryMeasure>& source,
        const std::vector<Meshes::ExactGeometryMeasure>& capacity,
        const std::vector<Meshes::ExactGeometryMeasure>& outgoing, const std::vector<ExactTransfer>& incoming,
        const std::vector<ExactFaceContribution>& contributions, bool nonnegative, int maximum_iterations,
        double tolerance = 1e-12, std::size_t byte_budget = 64 * 1024 * 1024);

private:
    using Pack = DefaultTpetraTypes;
    struct SparseAssembly;
    [[nodiscard]] Result solve_impl(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& forward, const std::vector<double>& reverse,
        bool nonnegative, int maximum_iterations, double tolerance, bool equilibrate_rows,
        const SparseAssembly* sparse);
    SP<const MeshHandle<>> mesh_;
    Pack::import_type cell_import_, face_import_;
    BelosLinearSolver<> solver_;
    Teuchos::RCP<Pack::matrix_type> matrix_;
    Pack::vector_type rhs_, solution_;
    mutable Pack::vector_type cell_owned_, cell_overlap_, face_owned_, face_overlap_;
    std::vector<double> capacity_, flux_, reverse_flux_;
    bool numeric_valid_ = false, equilibrated_ = false;
    std::vector<double> diagonal_;
    std::size_t row_capacity_ = 1;
    [[nodiscard]] std::vector<double> cells(const std::vector<double>& owned) const;
    [[nodiscard]] std::vector<double> faces(const std::vector<double>& overlap) const;
};
} // namespace SimpleFluid::detail
