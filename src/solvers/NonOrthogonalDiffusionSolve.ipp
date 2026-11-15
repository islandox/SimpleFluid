/** @file NonOrthogonalDiffusionSolve.ipp
 * @brief Inline forwarding to the native steady diffusion implementation.
 */
#pragma once

#include "solvers/NonOrthogonalDiffusionSolve.hh"

#include <utility>

namespace SimpleFluid::FVM
{

/**
 * @brief Solve steady non-orthogonal diffusion on a mapped FieldStored mesh.
 *
 * Implicit converges lagged partition gradients with correction_options and
 * checks the freshly rebuilt full-equation residual before success. Explicit
 * and Hybrid retain nNonOrthogonalCorrectors fixed sweeps and report linear
 * convergence only. A false return retains the last synchronized iterate.
 * Invalid or rank-divergent controls are rejected collectively before assembly.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options, const NonOrthogonalConvergenceOptions& correction_options)
{
    return detail::solve_stored_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        std::move(right_hand_source), solution, treatment, nNonOrthogonalCorrectors, linear_options,
        correction_options);
}

/** @brief Solve zero-source mapped steady non-orthogonal diffusion. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellFieldStored<Pack, MeshType>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options,
    const NonOrthogonalConvergenceOptions& correction_options)
{
    auto zero_source = [](typename Pack::local_ordinal_type) -> typename Pack::scalar_type
    { return typename Pack::scalar_type{}; };
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        ScalarCellValueProvider<Pack>{zero_source}, solution, treatment, nNonOrthogonalCorrectors, linear_options,
        correction_options);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_explicit_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options)
{
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        std::move(right_hand_source), solution, NonOrthogonalTreatment::Explicit, nNonOrthogonalCorrectors,
        linear_options);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline bool solve_explicit_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellFieldStored<Pack, MeshType>& solution,
    int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options)
{
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition), solution,
        NonOrthogonalTreatment::Explicit, nNonOrthogonalCorrectors, linear_options);
}

} // namespace SimpleFluid::FVM
