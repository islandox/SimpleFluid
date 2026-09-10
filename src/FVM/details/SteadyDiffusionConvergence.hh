/** @file SteadyDiffusionConvergence.hh @brief Shared native/mapped steady diffusion convergence. */
#pragma once

#include "FVM/NonOrthogonalConvergenceOptions.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/OperatorDetails.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

/** Validate every rank before field synchronization, assembly, or a linear solve. */
template<TpetraTypePack Pack, class MeshType>
void validate_steady_diffusion_controls(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    bool solution_matches_mesh, NonOrthogonalTreatment treatment, int correctors,
    const LinearSolverOptions& linear_options, const NonOrthogonalConvergenceOptions& correction_options)
{
    int invalid = !solution_matches_mesh || !std::isfinite(diffusivity) || diffusivity < 0 || correctors < 0 ||
                  correction_options.max_iterations <= 0 || !std::isfinite(correction_options.update_tolerance) ||
                  correction_options.update_tolerance <= 0 || !std::isfinite(correction_options.residual_tolerance) ||
                  correction_options.residual_tolerance <= 0;
    try
    {
        static_cast<void>(to_string(treatment));
        BelosLinearSolver<Pack>::validate_options(linear_options);
    }
    catch (const std::invalid_argument&)
    {
        invalid = 1;
    }
    const auto comm = mesh.owned_cell_map()->getComm();
    int globally_invalid = invalid;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &invalid, &globally_invalid);
    if (globally_invalid)
    {
        throw std::invalid_argument("solve_non_orthogonal_diffusion requires a solution on the target mesh, finite "
                                    "non-negative diffusivity, valid treatment and linear controls, non-negative "
                                    "correctors, and positive finite correction controls on every rank.");
    }

    const std::array<int, 8> integral{static_cast<int>(treatment), correctors, correction_options.max_iterations,
        linear_options.max_iterations, linear_options.verbosity, static_cast<int>(linear_options.backend),
        static_cast<int>(linear_options.preconditioner), linear_options.reuse_preconditioner ? 1 : 0};
    auto integral_min = integral, integral_max = integral;
    const std::array<real_t, 4> real{diffusivity, linear_options.tolerance, correction_options.update_tolerance,
        correction_options.residual_tolerance};
    auto real_min = real, real_max = real;
    if (comm->getSize() > 1)
    {
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, int(integral.size()), integral.data(), integral_min.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(integral.size()), integral.data(), integral_max.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, int(real.size()), real.data(), real_min.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(real.size()), real.data(), real_max.data());
    }
    if (integral_min != integral_max || real_min != real_max)
    {
        throw std::invalid_argument("solve_non_orthogonal_diffusion requires identical solver controls and "
                                    "diffusivity on every rank.");
    }
}

/**
 * Converge the existing matrix/RHS split without changing either assembly path.
 * A serial or orthogonal problem retains its single implicit solve. A problem
 * with lagged nonzero partition gradients additionally requires a global update
 * check. Every successful return checks a freshly rebuilt full equation.
 */
template<TpetraTypePack Pack, class MeshType, class Field, class Assemble, class Synchronize>
bool solve_implicit_diffusion_corrections(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    Field& solution, Assemble assemble, Synchronize synchronize, const LinearSolverOptions& linear_options,
    const NonOrthogonalConvergenceOptions& correction_options)
{
    using scalar_type = typename Pack::scalar_type;
    const auto execution = acquire_mesh_execution(mesh);
    int local_partition_correction = 0;
    if (diffusivity > scalar_type{})
    {
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<typename Pack::local_ordinal_type>(owned);
            for (const auto face : mesh.faces(cell))
            {
                if (!mesh.is_interior_face(face)) continue;
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face, cell);
                if (!mesh.is_owned_cell(other) &&
                    non_orthogonal_area_vector(mesh.face_area_vector_outward(face, cell),
                        mesh.cell_center_vector(face, cell)).norm() > scalar_type{})
                {
                    local_partition_correction = 1;
                }
            }
        }
    }
    int partition_correction = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
        &local_partition_correction, &partition_correction);
    real_t physical_rhs_norm = 0;
    if (partition_correction)
    {
        // Remove every field-dependent correction from the residual scale.
        // In a homogeneous problem rhs(phi) shrinks with phi and is unsuitable
        // as a denominator, even after the absolute defect has converged.
        // Reuse assembly with a zero field to retain exactly the physical
        // source and boundary load, including its original numerical ordering.
        Field zero_correction(solution.mesh_ptr(), "steady_diffusion_physical_rhs");
        zero_correction.sync_ghosts();
        physical_rhs_norm = assemble(zero_correction).rhs->norm2();
    }
    typename Pack::vector_type update(mesh.owned_cell_map(), true), residual(mesh.owned_cell_map(), true);
    synchronize();
    for (int iteration = 0; iteration < correction_options.max_iterations; ++iteration)
    {
        update.assign(solution.owned_data());
        auto system = assemble(solution);
        if (!partition_correction && iteration == 0) physical_rhs_norm = system.rhs->norm2();
        solution.owned_data().putScalar(scalar_type{});
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto converged = solve_linear_system<Pack>(matrix, *system.rhs, solution.owned_data(), linear_options);
        synchronize();
        if (!converged) return false;
        update.update(scalar_type{-1}, solution.owned_data(), scalar_type{1});
        const auto relative_update = update.normInf() /
            std::max(real_t{1}, static_cast<real_t>(solution.owned_data().normInf()));
        if (partition_correction &&
            (!std::isfinite(relative_update) || relative_update > correction_options.update_tolerance)) continue;

        // Assembly must re-evaluate remote gradients from the synchronized new
        // solution: the residual of the last lagged linear system is insufficient.
        const auto corrected = assemble(solution);
        corrected.matrix->apply(solution.owned_data(), residual);
        residual.update(scalar_type{-1}, *corrected.rhs, scalar_type{1});
        const auto relative_residual = residual.norm2() /
            (physical_rhs_norm > 0 ? physical_rhs_norm : real_t{1});
        if (std::isfinite(relative_residual) && relative_residual <= correction_options.residual_tolerance)
            return true;
    }
    return false;
}

} // namespace SimpleFluid::FVM::detail
