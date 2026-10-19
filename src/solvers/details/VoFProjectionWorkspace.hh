/** @file VoFProjectionWorkspace.hh
 * @brief Private native pressure graph, numeric assembly and linear work storage.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"

namespace SimpleFluid::detail
{
/** Integrated symmetric two-point pressure preconditioner on immutable geometry.
 * Numeric refresh always invalidates factors; the graph and vectors survive.
 * The caller owns nonorthogonal corrections and physical divergence acceptance.
 */
class SIMPLEFLUID_SOLVERS_LOCAL VoFProjectionWorkspace
{
public:
    using Pack = DefaultTpetraTypes;
    explicit VoFProjectionWorkspace(SP<const MeshHandle<>> mesh);
    /** Refresh A + C/dt^2; null susceptibility selects the fixed pressure gauge. */
    void refresh(const std::vector<double>& mobility, const std::vector<double>* susceptibility,
        double dt, Pack::global_ordinal_type gauge);
    Teuchos::RCP<Pack::matrix_type> matrix;
    Pack::vector_type rhs, solution;
    BelosLinearSolver<> solver;

private:
    SP<const MeshHandle<>> mesh_;
    std::size_t row_capacity_ = 1;
};
} // namespace SimpleFluid::detail
