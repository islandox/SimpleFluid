/**
 * @file TemperatureDiffusionEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume temperature diffusion and convection equation.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "FVM/FvmOperators.hh"
#include "FVM/BoundaryCache.hh"
#include "solvers/BelosLinearSolver.hh"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Finite-volume heat equation for cell-centered temperature.
 *
 * The class owns the boundary-condition lookup needed by the equation while
 * the caller owns field storage and time integration order.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class TemperatureDiffusionEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using face_velocity_field_type = VectorFaceField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    TemperatureDiffusionEquation(SP<const mesh_type> mesh,
                                 const BoundaryConditionSet& boundary_conditions);

    void refresh_boundary_cache();

    void advance_explicit(const std::vector<scalar_type>& old_temperature,
                          scalar_type time_step,
                          scalar_type thermal_diffusivity,
                          field_type& temperature) const;

    void advance_semi_implicit(
        const std::vector<scalar_type>& old_temperature,
        const std::vector<scalar_type>& face_fluxes,
        scalar_type time_step,
        scalar_type thermal_diffusivity,
        field_type& temperature,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_semi_implicit(
        const std::vector<scalar_type>& old_temperature,
        const face_velocity_field_type& face_velocity,
        scalar_type time_step,
        scalar_type thermal_diffusivity,
        field_type& temperature,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
    BoundaryCache<Pack> d_face_boundary_temperature;
    SP<BoundaryConditionMap>  d_boundary_condition;
};

/**
 * @brief Construct a temperature diffusion equation with mesh and boundary conditions.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the assembled mesh.
 * @param boundary_conditions Boundary condition set for temperature field.
 * @throws std::invalid_argument if the mesh is null.
 */
template<TpetraTypePack Pack>
TemperatureDiffusionEquation<Pack>::TemperatureDiffusionEquation(
    SP<const mesh_type> mesh,
    const BoundaryConditionSet& boundary_conditions)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "TemperatureDiffusionEquation")),
      d_boundary_condition(std::make_shared<BoundaryConditionMap>(boundary_conditions.temperature))
{
    refresh_boundary_cache();
}

/**
 * @brief Rebuild the cached lookup of Dirichlet temperature boundary values.
 *
 * @tparam Pack Tpetra type pack.
 * @param boundary_conditions Current boundary-condition set.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::refresh_boundary_cache()
{
    for (const auto& [patch_id, boundary_patch] : d_mesh->boundary_patches())
    {
        const auto iter =
            d_boundary_condition->find(
                d_mesh->boundary_patch_name(patch_id));
        if (iter == d_boundary_condition->end()
            || iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        for (auto face_lid : boundary_patch.face_lids)
        {
            if (!d_mesh->is_owned_face(face_lid))
            {
                continue;
            }

            // Cache only Dirichlet temperature values for now; Neumann and
            // NoSlip conditions are handled implicitly in the diffusion solve.
            d_face_boundary_temperature.value[patch_id] = Arr<
                typename Pack::scalar_type>(boundary_patch.face_lids.size(), iter->second.value);
        }
    }
}

/**
 * @brief Advance temperature with explicit thermal diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Local-cell temperature values from the previous time level.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Constant thermal diffusivity.
 * @param temperature Output temperature field over owned cells.
 * @throws std::invalid_argument if the field belongs to a different mesh or
 *         coefficients are non-physical.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_explicit(
    const std::vector<scalar_type>& old_temperature,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step",
                                             "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity",
                                             "TemperatureDiffusionEquation");
    EquationValidation::assert_sufficient_cache_size(old_temperature.size(),
                                                     d_mesh->num_local_cells());

    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto temp_p = old_temperature[cell];
        scalar_type laplacian = 0.0;

        const auto& faces = d_mesh->faces(cell_lid);
        const auto& face_distances = d_mesh->face_distances(cell_lid);

        bool is_interior_only = true;
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
        {
            const auto face_lid = faces[face_index];

            if (d_mesh->is_interior_face(face_lid))
            {
                const auto other = d_mesh->opposite_cell(face_lid, cell_lid);
                const auto distance = d_mesh->face_cell_center_distance(face_lid);
                if (distance > 0.0)
                {
                    laplacian +=
                        (old_temperature[static_cast<std::size_t>(other)] - temp_p)
                      * d_mesh->face_area(face_lid)
                      / distance;
                }
            }
            else
            {
                is_interior_only = false;
            }
        }
        if (!is_interior_only) continue;

        laplacian /= d_mesh->cell_volume(cell_lid);
        temperature.set_owned_value(cell_lid,
                                    temp_p
                                  + time_step * thermal_diffusivity * laplacian);
    }

    // Apply boundary conditions.
    for (const auto& [patch_id, boundary_patch] : d_mesh->boundary_patches())
    {
        if (d_boundary_condition->find(d_mesh->boundary_patch_name(patch_id))
            == d_boundary_condition->end()) continue;

        if (d_boundary_condition->at(d_mesh->boundary_patch_name(patch_id)).type
            == BoundaryConditionType::Dirichlet)
        {
            for (size_t in_patch_id = 0; in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
            {
                const auto boundary_face_lid = boundary_patch.face_lids[in_patch_id];
                if (d_mesh->is_owned_face(boundary_face_lid))
                {
                    const auto boundary_temperature = d_face_boundary_temperature.value.at(patch_id)[in_patch_id];
                    const auto owner = d_mesh->owner_cell(boundary_face_lid);

                    const auto temp_p = old_temperature[owner];
                    scalar_type laplacian = 0.0;
                    const auto& faces = d_mesh->faces(owner);
                    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
                    {
                        const auto face_lid = faces[face_index];
                        if (face_lid == boundary_face_lid)
                        {
                            const auto distance_to_face = d_mesh->face_cell_center_distance(face_lid);
                            laplacian += (boundary_temperature - temp_p)
                                       * d_mesh->face_area(face_lid)
                                       / distance_to_face;
                        }
                        else
                        {
                            const auto other = d_mesh->opposite_cell(face_lid, owner);
                            const auto distance = d_mesh->face_cell_center_distance(face_lid);
                            if (distance > 0.0)
                            {
                                laplacian +=
                                    (old_temperature[static_cast<std::size_t>(other)] - temp_p)
                                * d_mesh->face_area(face_lid)
                                / distance;
                            }

                        }
                    }   
                    temperature.set_owned_value(owner, temp_p
                                              + time_step * thermal_diffusivity * laplacian / d_mesh->cell_volume(owner));
                }
            }
        }
        else
        {

        }
    }
}

/**
 * @brief Advance temperature with semi-implicit upwind convection and diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Local-cell temperature values from the previous time level.
 * @param face_fluxes Owner-oriented integrated mass fluxes.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Constant thermal diffusivity.
 * @param temperature Output temperature field over owned cells.
 * @param linear_options Belos solver options for the transport solve.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_semi_implicit(
    const std::vector<scalar_type>& old_temperature,
    const std::vector<scalar_type>& face_fluxes,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step",
                                             "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity",
                                             "TemperatureDiffusionEquation");
    EquationValidation::assert_sufficient_cache_size(old_temperature.size(),
                                                     d_mesh->num_local_cells());

    auto boundary_value =
        [&](int patch_id, size_t in_patch_id)
    {
        const auto& boundary =
            d_face_boundary_temperature.value.at(patch_id)[in_patch_id];
        return boundary;
    };

    auto system = FvmOperators::transport_system<Pack>(
        *d_mesh, old_temperature, face_fluxes, time_step,
        thermal_diffusivity, boundary_value);

    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto converged =
        solve_linear_system<Pack>(matrix, system.rhs,
                                  temperature.owned_data(), linear_options);
    if (!converged)
    {
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            if (!std::isfinite(temperature.value(cell_lid)))
            {
                throw std::runtime_error(
                    "TemperatureDiffusionEquation transport solve produced a non-finite value.");
            }
        }
    }
    temperature.sync_ghosts();
}

template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_semi_implicit(
    const std::vector<scalar_type>& old_temperature,
    const face_velocity_field_type& face_velocity,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature,
    const LinearSolverOptions& linear_options) const
{
    const auto face_fluxes =
        FvmOperators::normal_face_fluxes<Pack>(*d_mesh, face_velocity);
    advance_semi_implicit(old_temperature, face_fluxes, time_step,
                          thermal_diffusivity, temperature, linear_options);
}

} // namespace SimpleFluid
