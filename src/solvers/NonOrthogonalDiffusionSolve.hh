/**
 * @file solvers/NonOrthogonalDiffusionSolve.hh
 * @brief  Solver-owned steady non-orthogonal diffusion entry points.
 *
 * These wrappers iterate linear solves over FVM assembly. They are compiled
 * into `SimpleFluidSolvers`, not `SimpleFluidFVM`: the finite-volume
 * discretization library owns reconstruction, fluxes, and discrete operators,
 * while the solver component owns every Belos/Ifpack2/MueLu solve. Keeping the
 * split here is what lets `libSimpleFluidFVM` link without any symbol that only
 * `libSimpleFluidSolvers` defines (Mach-O `ld64` rejects such a DSO outright).
 *
 * The `SimpleFluid::FVM` namespace is retained for source compatibility with
 * existing callers; only the binary ownership moved.
 */
#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/NonOrthogonalConvergenceOptions.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "SimpleFluidExport.hh"
#include "fields/CellField.hh"
#include "fields/FieldStored.hh"
#include "geometry/Mesh.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/SteadyDiffusionConvergence.hh"

#include <utility>

namespace SimpleFluid::FVM
{

/**
 * @brief Solve a scalar diffusion equation with explicit non-orthogonal
 *        correction sweeps.
 *
 * A value of zero performs only the orthogonal solve. Positive values run
 * that solve followed by @p nNonOrthogonalCorrectors explicit correction
 * solves using the latest solution gradient.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @param[in,out] solution Solution field; zeroed before each solve and
 *        updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 * @throws std::invalid_argument if @p solution is not on the target mesh
 *         or @p nNonOrthogonalCorrectors is negative.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, CellField<Pack>& solution, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {});

/**
 * @brief Solve scalar diffusion using the selected non-orthogonal treatment.
 *
 * For `explicit`, @p nNonOrthogonalCorrectors has the same meaning as in
 * solve_explicit_non_orthogonal_diffusion(). For `implicit`, lagged remote
 * gradients are iterated up to correction_options.max_iterations. Success
 * requires the global relative update and a freshly rebuilt full-equation
 * residual to satisfy correction_options; a problem without nonzero partition
 * corrections retains a single solve and only needs the residual check.
 * For `hybrid`, an initial half-implicit solve
 * is followed by @p nNonOrthogonalCorrectors correction solves that put the
 * remaining half of the tangential term on the RHS.
 *
 * @return For Implicit, `true` only when the corrected equation converges.
 *         Returns `false` on linear failure or correction-limit exhaustion.
 *         Explicit/Hybrid retain their fixed-sweep linear-convergence status.
 *         On failure, solution retains the last iterate with synchronized ghosts.
 * @throws std::invalid_argument collectively if a mesh/option input is invalid
 *         or ranks disagree on solver controls, treatment, or diffusivity.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, CellField<Pack>& solution, NonOrthogonalTreatment treatment,
    int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {});

/** @brief Steady solve with explicit implicit-correction convergence controls. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, CellField<Pack>& solution, NonOrthogonalTreatment treatment,
    int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options,
    const NonOrthogonalConvergenceOptions& correction_options);

/**
 * @brief Solve a scalar diffusion equation with zero source and explicit
 *        non-orthogonal correction sweeps.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param[in,out] solution Solution field; updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    CellField<Pack>& solution, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {});

/**
 * @brief Solve a zero-source scalar diffusion equation using the selected
 *        non-orthogonal treatment.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    CellField<Pack>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {});

/** @brief Zero-source steady solve with explicit implicit-correction convergence controls. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_EXPORT bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    CellField<Pack>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options, const NonOrthogonalConvergenceOptions& correction_options);

/**
 * @brief Solve steady non-orthogonal diffusion on a mapped FieldStored mesh.
 *
 * Implicit converges lagged partition gradients with correction_options and
 * checks the freshly rebuilt full-equation residual before success. Explicit
 * and Hybrid retain nNonOrthogonalCorrectors fixed sweeps and report linear
 * convergence only. A false return retains the last synchronized iterate.
 * Invalid or rank-divergent controls are rejected collectively before assembly.
 */
template<TpetraTypePack Pack, class MeshType>
bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {}, const NonOrthogonalConvergenceOptions& correction_options = {})
{
    return detail::solve_stored_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        std::move(right_hand_source), solution, treatment, nNonOrthogonalCorrectors, linear_options,
        correction_options);
}

/** @brief Solve zero-source mapped steady non-orthogonal diffusion. */
template<TpetraTypePack Pack, class MeshType>
bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellFieldStored<Pack, MeshType>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {},
    const NonOrthogonalConvergenceOptions& correction_options = {})
{
    auto zero_source = [](typename Pack::local_ordinal_type) -> typename Pack::scalar_type
    { return typename Pack::scalar_type{}; };
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        ScalarCellValueProvider<Pack>{zero_source}, solution, treatment, nNonOrthogonalCorrectors, linear_options,
        correction_options);
}

} // namespace SimpleFluid::FVM
