/**
 * @file TemperatureDiffusionEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for TemperatureDiffusionEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "TemperatureDiffusionEquation.hh"

namespace SimpleFluid
{

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

        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
        {
            const auto face_lid = faces[face_index];

            if (d_mesh->is_interior_face(face_lid))
            {
                const auto other =
                    d_mesh->opposite_or_periodic_neighbor_cell(face_lid,
                                                               cell_lid);
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

        laplacian /= d_mesh->cell_volume(cell_lid);
        temperature.set_owned_value(cell_lid,
                                    temp_p
                                  + time_step * thermal_diffusivity * laplacian);
    }

    // Apply Dirichlet boundary contributions on top of the interior-face
    // laplacian already computed above.
    for (const auto& [patch_id, boundary_patch] : d_mesh->boundary_patches())
    {
        auto boundary_name = d_mesh->boundary_patch_name(patch_id);
        if (!d_boundary_condition->contains(boundary_name)) continue;

        auto BC = d_boundary_condition->at(d_mesh->boundary_patch_name(patch_id));
        if (BC.type == BoundaryConditionType::Dirichlet)
        {
            for (size_t in_patch_id = 0; in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
            {
                const auto boundary_face_lid = boundary_patch.face_lids[in_patch_id];
                if (d_mesh->is_owned_face(boundary_face_lid))
                {
                    const auto boundary_temperature = d_face_boundary_temperature.value.at(patch_id)[in_patch_id];
                    const auto owner = d_mesh->owner_cell(boundary_face_lid);
                    const auto temp_p = old_temperature[owner];
                    const auto distance_to_face = d_mesh->cell_to_face_distance(boundary_face_lid, owner);

                    if (distance_to_face > 0.0)
                    {
                        const auto boundary_contrib =
                            (boundary_temperature - temp_p)
                          * d_mesh->face_area(boundary_face_lid)
                          / distance_to_face;

                        temperature.sum_into_value(owner,
                            time_step * thermal_diffusivity
                                    * boundary_contrib / d_mesh->cell_volume(owner));
                    }
                }
            }
        }
        else if (BC.type == BoundaryConditionType::Neumann)
        {
            if (BC.value == scalar_type{0.0}) continue; // No contribution for zero Neumann flux.
            // Nonzero Neumann Condition is to be implemented in the future; currently treated as zero flux.
        }
        else if (BC.type == BoundaryConditionType::NoSlip)
        {
            continue; // No contribution for no-slip velocity boundary.
        }
        else if (BC.type == BoundaryConditionType::Robin)
        {
            throw std::runtime_error("Robin boundary conditions are not yet implemented in TemperatureDiffusionEquation.");
        }
    }

    temperature.sync_ghosts();
}

template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_semi_implicit(
    const field_type& old_temperature,
    const FaceField<Pack>& face_fluxes,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, old_temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step",
                                             "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity",
                                             "TemperatureDiffusionEquation");

    auto boundary_value =
        [&](int patch_id, size_t in_patch_id) -> typename Pack::scalar_type
    {
        const auto cache_it = d_face_boundary_temperature.value.find(patch_id);
        if (cache_it == d_face_boundary_temperature.value.end())
        {
            return typename Pack::scalar_type{};
        }
        return cache_it->second[in_patch_id];
    };

    auto system = FvmOperators::transport_system<Pack>(
        old_temperature, face_fluxes, time_step,
        thermal_diffusivity, boundary_value, d_cached_transport_matrix);

    if (d_cached_transport_matrix.is_null())
    {
        d_cached_transport_matrix = system.matrix;
    }

    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto converged =
        solve_linear_system<Pack>(matrix, *system.rhs,
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

} // namespace SimpleFluid
