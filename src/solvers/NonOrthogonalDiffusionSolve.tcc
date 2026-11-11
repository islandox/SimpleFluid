/**
 * @file NonOrthogonalDiffusionSolve.tcc
 * @brief Out-of-line template implementations for solver-owned steady diffusion solves.
 */
#pragma once

#include "solvers/NonOrthogonalDiffusionSolve.hh"

#include <utility>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack>
bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    CellField<Pack>& solution, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options)
{
    if (&solution.mesh() != &mesh)
    {
        throw std::invalid_argument("solve_explicit_non_orthogonal_diffusion requires solution on the target mesh.");
    }
    if (nNonOrthogonalCorrectors < 0)
    {
        throw std::invalid_argument("nNonOrthogonalCorrectors cannot be negative.");
    }

    bool converged = true;
    for (int corrector = 0; corrector <= nNonOrthogonalCorrectors; ++corrector)
    {
        auto system = corrector == 0 ? diffusion_system<Pack>(mesh, diffusivity, boundary_condition, right_hand_source)
                                     : explicit_non_orthogonal_diffusion_system<Pack>(
                                           mesh, diffusivity, boundary_condition, right_hand_source, solution);

        solution.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        converged = solve_linear_system<Pack>(matrix, *system.rhs, solution.owned_data(), linear_options);
        mesh.sync_periodic_boundaries(solution);
        if (!converged)
        {
            return false;
        }
    }

    return converged;
}

template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    CellField<Pack>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options)
{
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition),
        std::move(right_hand_source), solution, treatment, nNonOrthogonalCorrectors, linear_options,
        NonOrthogonalConvergenceOptions{});
}

template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    CellField<Pack>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options, const NonOrthogonalConvergenceOptions& correction_options)
{
    using scalar_type = typename Pack::scalar_type;

    detail::validate_steady_diffusion_controls<Pack>(mesh, diffusivity, &solution.mesh() == &mesh,
        treatment, nNonOrthogonalCorrectors, linear_options, correction_options);

    if (treatment == NonOrthogonalTreatment::Explicit)
    {
        return solve_explicit_non_orthogonal_diffusion<Pack>(mesh, diffusivity, boundary_condition, right_hand_source,
            solution, nNonOrthogonalCorrectors, linear_options);
    }

    if (treatment == NonOrthogonalTreatment::Implicit)
    {
        return detail::solve_implicit_diffusion_corrections<Pack>(mesh, diffusivity, solution,
            [&](const auto& correction) { return fully_implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source, &correction); },
            [&] { mesh.sync_periodic_boundaries(solution); }, linear_options, correction_options);
    }

    if (treatment != NonOrthogonalTreatment::Hybrid)
    {
        throw std::invalid_argument("Unknown NonOrthogonalTreatment value.");
    }

    bool converged = true;
    for (int corrector = 0; corrector <= nNonOrthogonalCorrectors; ++corrector)
    {
        auto system = corrector == 0 ? implicit_non_orthogonal_diffusion_system<Pack>(mesh, diffusivity,
                                           boundary_condition, right_hand_source, scalar_type{0.5}, &solution)
                                     : non_orthogonal_diffusion_system<Pack>(mesh, diffusivity, boundary_condition,
                                           right_hand_source, NonOrthogonalTreatment::Hybrid, &solution);

        solution.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        converged = solve_linear_system<Pack>(matrix, *system.rhs, solution.owned_data(), linear_options);
        mesh.sync_periodic_boundaries(solution);
        if (!converged)
        {
            return false;
        }
    }

    return converged;
}

template<TpetraTypePack Pack>
bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, CellField<Pack>& solution, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    return solve_explicit_non_orthogonal_diffusion<Pack>(
        mesh, diffusivity, boundary_condition, zero_source, solution, nNonOrthogonalCorrectors, linear_options);
}

template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, CellField<Pack>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options)
{
    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, std::move(boundary_condition), solution,
        treatment, nNonOrthogonalCorrectors, linear_options, NonOrthogonalConvergenceOptions{});
}

template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, CellField<Pack>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options,
    const NonOrthogonalConvergenceOptions& correction_options)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    return solve_non_orthogonal_diffusion<Pack>(mesh, diffusivity, boundary_condition, zero_source, solution, treatment,
        nNonOrthogonalCorrectors, linear_options, correction_options);
}

} // namespace SimpleFluid::FVM
