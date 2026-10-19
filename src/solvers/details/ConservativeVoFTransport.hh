/** @file ConservativeVoFTransport.hh
 * @brief Implicit conservative tracer transport with prescribed extensive storage and flux.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"

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
    };
    explicit ConservativeVoFTransport(SP<const MeshHandle<>> mesh);
    /** Extensive Q_old/S/M_new; overlap-face F in canonical owner orientation.
     * Optional row equilibration preserves microscopic carrier rows without a
     * reciprocal diagonal overflow; the original physical residual still must
     * meet tolerance. It changes conditioning, not the transport equation.
     */
    [[nodiscard]] Result solve(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& flux, bool nonnegative,
        int maximum_iterations, double tolerance = 1e-12, bool equilibrate_rows = false);

private:
    using Pack = DefaultTpetraTypes;
    SP<const MeshHandle<>> mesh_;
    Pack::import_type cell_import_, face_import_;
    BelosLinearSolver<> solver_;
    Teuchos::RCP<Pack::matrix_type> matrix_;
    Pack::vector_type rhs_, solution_;
    mutable Pack::vector_type cell_owned_, cell_overlap_, face_owned_, face_overlap_;
    std::vector<double> capacity_, flux_;
    bool numeric_valid_ = false, equilibrated_ = false;
    std::vector<double> diagonal_;
    std::size_t row_capacity_ = 1;
    [[nodiscard]] std::vector<double> cells(const std::vector<double>& owned) const;
    [[nodiscard]] std::vector<double> faces(const std::vector<double>& overlap) const;
};
} // namespace SimpleFluid::detail
