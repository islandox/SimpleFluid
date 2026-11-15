/** @file NonOrthogonalDiffusionSolve.hh
 * @brief Solver-owned steady non-orthogonal diffusion on native stored fields.
 */
#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/NonOrthogonalConvergenceOptions.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshConcepts.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/SteadyDiffusionConvergence.hh"

namespace SimpleFluid::FVM
{

/**
 * @brief Solve steady diffusion with the selected non-orthogonal treatment.
 *
 * Implicit correction requires both update and freshly rebuilt full-equation
 * residual gates. Explicit and Hybrid retain fixed correction sweeps. A false
 * return retains the last synchronized iterate. Invalid or rank-divergent
 * controls are rejected collectively before assembly.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {}, const NonOrthogonalConvergenceOptions& correction_options = {});

/** @brief Solve zero-source steady diffusion on a native stored field. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellFieldStored<Pack, MeshType>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {},
    const NonOrthogonalConvergenceOptions& correction_options = {});

/** @brief Solve native diffusion with fixed explicit correction sweeps. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_explicit_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {});

/** @brief Solve zero-source native diffusion with explicit correction sweeps. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_explicit_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellFieldStored<Pack, MeshType>& solution,
    int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {});

} // namespace SimpleFluid::FVM

#include "solvers/NonOrthogonalDiffusionSolve.ipp"
