/**
 * @file WallDistanceEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementation of distributed Poisson wall distance.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "WallDistanceEquation.hh"

#include "FVM/CellOperators.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "equations/EquationValidation.hh"
#include "equations/turbulence/TurbulenceCollectiveValidation.hh"
#include "fields/VectorCellField.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{

namespace wall_distance_detail
{

/**
 * @brief Require the sorted wall-name sequence to be identical on all ranks.
 *
 * Length and character reductions avoid relying on implementation-specific
 * string hashes and ensure every rank executes the same boundary branches.
 */
template <TpetraTypePack Pack>
void require_uniform_wall_names(const Mesh<Pack>& mesh,
                                const ArrString& names)
{
    turbulence_detail::require_uniform_integral(
        mesh, static_cast<int>(names.size()),
        "Poisson wall-distance boundary count");
    const auto communicator = mesh.owned_cell_map()->getComm();
    for (size_t name_id = 0; name_id < names.size(); ++name_id)
    {
        const auto& name = names[name_id];
        const auto local_length = static_cast<int>(name.size());
        turbulence_detail::require_uniform_integral(
            mesh, local_length,
            "Poisson wall-distance boundary-name length");
        for (const char character : name)
        {
            const int local_character =
                static_cast<unsigned char>(character);
            int minimum_character = 0;
            int maximum_character = 0;
            Teuchos::reduceAll(
                *communicator, Teuchos::REDUCE_MIN, 1,
                &local_character, &minimum_character);
            Teuchos::reduceAll(
                *communicator, Teuchos::REDUCE_MAX, 1,
                &local_character, &maximum_character);
            if (minimum_character != maximum_character)
            {
                throw std::invalid_argument(
                    "Poisson wall-distance boundary names must agree "
                    "on every rank.");
            }
        }
    }
}

/**
 * @brief Publish a fully synchronized candidate without a throwing commit step.
 *
 * The caller validates that both fields use the mesh's exact owned and overlap
 * maps before entering the wall-distance solve. Consequently these updates
 * cannot encounter a map mismatch and require no communication or allocation.
 * Treating an unexpected backend failure as fatal preserves the solve()
 * contract: no exception can expose a partially replaced output field.
 */
template <TpetraTypePack Pack>
void publish_synced_candidate(
    CellField<Pack>& output, const CellField<Pack>& candidate) noexcept
{
    using scalar_type = typename Pack::scalar_type;
    output.owned_data().update(
        scalar_type{1}, candidate.owned_data(), scalar_type{0});
    output.overlap_data().update(
        scalar_type{1}, candidate.overlap_data(), scalar_type{0});
}

} // namespace wall_distance_detail

/**
 * @brief Construct a Poisson wall-distance equation.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param mesh Computational mesh.
 */
template <TpetraTypePack Pack>
PoissonWallDistanceEquation<Pack>::PoissonWallDistanceEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "PoissonWallDistanceEquation"))
{
}

/**
 * @brief Validate, solve, reconstruct, and atomically publish wall distance.
 * @tparam Pack Tpetra type pack used by the equation.
 */
template <TpetraTypePack Pack>
void PoissonWallDistanceEquation<Pack>::solve(
    const ArrString& wall_boundary_names,
    field_type& wall_distance,
    const WallDistanceEquationOptions& options) const
{
    constexpr const char* class_name = "PoissonWallDistanceEquation";
    ArrString selected_names = wall_boundary_names;

    turbulence_detail::collective_local_validation(
        *d_mesh, "Poisson wall-distance input validation",
        [&]
        {
            EquationValidation::require_mesh_match(
                *d_mesh, wall_distance, class_name);
            if (wall_distance.owned_data().getMap().get()
                    != d_mesh->owned_cell_map().get()
                || wall_distance.overlap_data().getMap().get()
                    != d_mesh->overlap_cell_map().get())
            {
                throw std::invalid_argument(
                    "PoissonWallDistanceEquation output storage maps do "
                    "not match its mesh.");
            }
            if (selected_names.empty())
            {
                throw std::invalid_argument(
                    "PoissonWallDistanceEquation requires at least one "
                    "wall boundary name.");
            }
            if (selected_names.size()
                > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "Poisson wall-distance boundary list is too large.");
            }
            for (const auto& name : selected_names)
            {
                if (name.empty())
                {
                    throw std::invalid_argument(
                        "Poisson wall-distance boundary names cannot be "
                        "empty.");
                }
                if (name.size()
                    > static_cast<size_t>(
                        std::numeric_limits<int>::max()))
                {
                    throw std::invalid_argument(
                        "Poisson wall-distance boundary name is too long.");
                }
            }
            std::sort(selected_names.begin(), selected_names.end());
            if (std::adjacent_find(
                    selected_names.begin(), selected_names.end())
                != selected_names.end())
            {
                throw std::invalid_argument(
                    "Poisson wall-distance boundary names must be unique.");
            }
            validate_wall_distance_equation_options(options);
        });

    wall_distance_detail::require_uniform_wall_names(
        *d_mesh, selected_names);
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        static_cast<int>(options.non_orthogonal_treatment),
        "Poisson wall-distance non-orthogonal treatment");
    turbulence_detail::require_uniform_integral(
        *d_mesh, options.non_orthogonal_correctors,
        "Poisson wall-distance non-orthogonal correctors");
    turbulence_detail::require_uniform_integral(
        *d_mesh, options.linear_solver.max_iterations,
        "Poisson wall-distance maximum iterations");
    turbulence_detail::require_uniform_real(
        *d_mesh, options.linear_solver.tolerance,
        "Poisson wall-distance linear tolerance");
    turbulence_detail::require_uniform_integral(
        *d_mesh, options.linear_solver.verbosity,
        "Poisson wall-distance linear verbosity");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(options.linear_solver.preconditioner),
        "Poisson wall-distance preconditioner");
    turbulence_detail::require_uniform_integral(
        *d_mesh, options.linear_solver.reuse_preconditioner ? 1 : 0,
        "Poisson wall-distance preconditioner reuse");

    const auto is_selected =
        [&](const std::string& name)
    {
        return std::binary_search(
            selected_names.begin(), selected_names.end(), name);
    };

    const auto communicator = d_mesh->owned_cell_map()->getComm();
    for (const auto& selected_name : selected_names)
    {
        int local_found = 0;
        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            if (d_mesh->boundary_batch_name(batch_id) != selected_name)
            {
                continue;
            }
            for (const auto face_lid : batch.face_lids)
            {
                if (d_mesh->is_owned_face(face_lid)
                    && d_mesh->is_boundary_face(face_lid))
                {
                    local_found = 1;
                    break;
                }
            }
        }

        int globally_found = 0;
        Teuchos::reduceAll(
            *communicator, Teuchos::REDUCE_MAX, 1,
            &local_found, &globally_found);
        if (globally_found == 0)
        {
            throw std::invalid_argument(
                "Poisson wall-distance boundary '" + selected_name
                + "' has no physical faces on the distributed mesh.");
        }
    }

    const auto boundary_condition =
        [&](int batch_id, size_t) -> BoundaryCondition
    {
        return is_selected(d_mesh->boundary_batch_name(batch_id))
             ? BoundaryCondition{
                   BoundaryConditionType::Dirichlet, scalar_type{}}
             : BoundaryCondition{
                   BoundaryConditionType::Neumann, scalar_type{}};
    };
    const auto source =
        [](typename Pack::local_ordinal_type) -> scalar_type
    {
        return scalar_type{1};
    };

    field_type potential(d_mesh, "wall_distance_potential");
    const auto converged =
        FVM::solve_non_orthogonal_diffusion<Pack>(
            *d_mesh, scalar_type{1}, boundary_condition, source,
            potential, options.non_orthogonal_treatment,
            options.non_orthogonal_correctors, options.linear_solver);

    turbulence_detail::collective_local_validation(
        *d_mesh, "Poisson wall-distance potential validation",
        [&]
        {
            if (!converged)
            {
                throw std::runtime_error(
                    "Poisson wall-distance solve did not converge.");
            }
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<typename Pack::local_ordinal_type>(owned);
                const auto value = potential.value(cell_lid);
                if (!std::isfinite(value) || value <= scalar_type{})
                {
                    throw std::runtime_error(
                        "Poisson wall-distance solve produced a non-finite "
                        "or non-positive potential.");
                }
            }
        });

    VectorCellField<Pack> potential_gradient(
        d_mesh, "wall_distance_potential_gradient");
    const auto boundary_value =
        [&](int batch_id, size_t in_batch_id) -> scalar_type
    {
        return boundary_condition(batch_id, in_batch_id).value;
    };
    turbulence_detail::collective_local_validation(
        *d_mesh, "Poisson wall-distance gradient reconstruction",
        [&]
        {
            FVM::cell_gradient(
                potential, boundary_condition, boundary_value,
                potential_gradient);
        });

    field_type candidate(d_mesh, "wall_distance_candidate");
    turbulence_detail::collective_local_validation(
        *d_mesh, "Poisson wall-distance reconstruction validation",
        [&]
        {
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<typename Pack::local_ordinal_type>(owned);
                const auto psi = potential.value(cell_lid);
                const auto gradient_magnitude =
                    potential_gradient.value(cell_lid).norm();
                const auto radicand =
                    gradient_magnitude * gradient_magnitude
                    + scalar_type{2} * psi;
                const auto denominator =
                    std::sqrt(radicand) + gradient_magnitude;
                const auto distance =
                    scalar_type{2} * psi / denominator;
                if (!std::isfinite(gradient_magnitude)
                    || !std::isfinite(radicand)
                    || radicand <= scalar_type{}
                    || !std::isfinite(distance)
                    || distance <= scalar_type{})
                {
                    throw std::runtime_error(
                        "Poisson wall-distance reconstruction produced a "
                        "non-finite or non-positive distance.");
                }
                candidate.set_owned_value(cell_lid, distance);
            }
        });

    candidate.sync_ghosts();
    turbulence_detail::collective_local_validation(
        *d_mesh, "Synchronized Poisson wall-distance validation",
        [&]
        {
            for (size_t local = 0;
                 local < d_mesh->num_local_cells(); ++local)
            {
                const auto cell_lid =
                    static_cast<typename Pack::local_ordinal_type>(local);
                const auto distance = candidate.local_value(cell_lid);
                if (!std::isfinite(distance)
                    || distance <= scalar_type{})
                {
                    throw std::runtime_error(
                        "Synchronized Poisson wall-distance field contains "
                        "a non-finite or non-positive distance.");
                }
            }
        });

    wall_distance_detail::publish_synced_candidate(
        wall_distance, candidate);
}

} // namespace SimpleFluid
