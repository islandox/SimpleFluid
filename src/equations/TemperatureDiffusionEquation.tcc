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
#include "equations/CollectiveValidation.hh"

#include <Teuchos_CommHelpers.hpp>

namespace SimpleFluid
{

/**
 * @brief Construct a TemperatureDiffusionEquation on the given mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param boundary_conditions Boundary-condition set for temperature.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack, class MeshType>
TemperatureDiffusionEquation<Pack, MeshType>::TemperatureDiffusionEquation(
    SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions)
    : d_mesh(EquationValidation::require_non_null_mesh(std::move(mesh), "TemperatureDiffusionEquation")),
      d_transport_geometry_cache(*d_mesh), d_candidate_temperature(d_mesh, "temperature_candidate", false),
      d_face_boundary_temperature{{}, d_mesh},
      d_boundary_condition(std::make_shared<BoundaryConditionMap>(boundary_conditions.temperature))
{
    refresh_boundary_cache();
}

/**
 * @brief Refresh the cached Dirichlet boundary temperature values.
 *
 * Scans all boundary batches and stores the prescribed Dirichlet
 * temperature for each owned boundary face. Neumann and NoSlip conditions
 * are applied directly by the advance routines and are not cached here.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack, class MeshType>
void TemperatureDiffusionEquation<Pack, MeshType>::refresh_boundary_cache()
{
    d_face_boundary_temperature.value.clear();
    for (const auto& [batch_id, boundary_batch] : d_mesh->boundary_batches())
    {
        const auto iter = d_boundary_condition->find(d_mesh->boundary_batch_name(batch_id));
        if (iter == d_boundary_condition->end() || iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        for (auto face_lid : boundary_batch.face_lids)
        {
            if (!d_mesh->is_owned_face(face_lid))
            {
                continue;
            }

            d_face_boundary_temperature.value[batch_id] =
                Arr<typename Pack::scalar_type>(boundary_batch.face_lids.size(), iter->second.value);
        }
    }
}

/**
 * @brief Advance the temperature field explicitly with a zero source term.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Temperature values from the previous time step.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Thermal diffusivity coefficient.
 * @param[out] temperature Updated temperature field on return.
 */
template<TpetraTypePack Pack, class MeshType>
void TemperatureDiffusionEquation<Pack, MeshType>::advance_explicit(const std::vector<scalar_type>& old_temperature,
    scalar_type time_step, scalar_type thermal_diffusivity, field_type& temperature) const
{
    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    advance_explicit(old_temperature, time_step, thermal_diffusivity, temperature, zero_source);
}

/**
 * @brief Advance the temperature field explicitly with a right-hand source
 *        term.
 *
 * Uses a forward-Euler update with a two-point flux approximation for
 * diffusion and handles Dirichlet, Neumann, and NoSlip boundary
 * conditions. Periodic boundaries are synchronised after the update.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Temperature values from the previous time step.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Thermal diffusivity coefficient.
 * @param[out] temperature Updated temperature field on return.
 * @param right_hand_source Per-cell scalar source provider.
 * @throws std::invalid_argument on mesh mismatch, negative time step, or
 *         negative diffusivity.
 * @throws std::runtime_error if a Robin boundary condition is encountered.
 */
template<TpetraTypePack Pack, class MeshType>
void TemperatureDiffusionEquation<Pack, MeshType>::advance_explicit(const std::vector<scalar_type>& old_temperature,
    scalar_type time_step, scalar_type thermal_diffusivity, field_type& temperature,
    const source_type& right_hand_source) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature, "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step", "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity", "TemperatureDiffusionEquation");
    EquationValidation::assert_sufficient_cache_size(old_temperature.size(), d_mesh->num_local_cells());

    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto temp_p = old_temperature[cell];
        scalar_type laplacian = 0.0;

        const auto& faces = d_mesh->faces(cell_lid);

        for (size_t face_index = 0; face_index < faces.size(); ++face_index)
        {
            const auto face_lid = faces[face_index];

            if (d_mesh->is_interior_face(face_lid))
            {
                const auto other = d_mesh->opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto distance = d_mesh->face_cell_center_distance(face_lid);
                if (distance > 0.0)
                {
                    laplacian +=
                        (old_temperature[static_cast<size_t>(other)] - temp_p) * d_mesh->face_area(face_lid) / distance;
                }
            }
        }

        laplacian /= d_mesh->cell_volume(cell_lid);
        temperature.set_owned_value(
            cell_lid, temp_p + time_step * (thermal_diffusivity * laplacian + right_hand_source(cell_lid)));
    }

    // Apply Dirichlet boundary contributions on top of the interior-face
    // laplacian already computed above.
    for (const auto& [batch_id, boundary_batch] : d_mesh->boundary_batches())
    {
        auto boundary_name = d_mesh->boundary_batch_name(batch_id);
        if (!d_boundary_condition->contains(boundary_name))
            continue;

        auto BC = d_boundary_condition->at(d_mesh->boundary_batch_name(batch_id));
        if (BC.type == BoundaryConditionType::Dirichlet)
        {
            for (size_t in_batch_id = 0; in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
            {
                const auto boundary_face_lid = boundary_batch.face_lids[in_batch_id];
                if (d_mesh->is_owned_face(boundary_face_lid))
                {
                    const auto boundary_temperature = d_face_boundary_temperature.value.at(batch_id)[in_batch_id];
                    const auto owner = d_mesh->owner_cell(boundary_face_lid);
                    const auto temp_p = old_temperature[owner];
                    const auto distance_to_face = d_mesh->cell_to_face_distance(boundary_face_lid, owner);

                    if (distance_to_face > 0.0)
                    {
                        const auto boundary_contrib =
                            (boundary_temperature - temp_p) * d_mesh->face_area(boundary_face_lid) / distance_to_face;

                        temperature.sum_into_value(
                            owner, time_step * thermal_diffusivity * boundary_contrib / d_mesh->cell_volume(owner));
                    }
                }
            }
        }
        else if (BC.type == BoundaryConditionType::Neumann)
        {
            for (const auto boundary_face_lid : boundary_batch.face_lids)
            {
                if (!d_mesh->is_owned_face(boundary_face_lid))
                {
                    continue;
                }

                const auto owner = d_mesh->owner_cell(boundary_face_lid);
                const auto boundary_contribution =
                    static_cast<scalar_type>(BC.value) * d_mesh->face_area(boundary_face_lid);
                temperature.sum_into_value(
                    owner, time_step * thermal_diffusivity * boundary_contribution / d_mesh->cell_volume(owner));
            }
        }
        else if (BC.type == BoundaryConditionType::NoSlip)
        {
            continue; // No contribution for no-slip velocity boundary.
        }
        else if (BC.type == BoundaryConditionType::Robin)
        {
            throw std::runtime_error(
                "Robin boundary conditions are not yet implemented in TemperatureDiffusionEquation.");
        }
    }

    d_mesh->sync_periodic_boundaries(temperature);
}

/**
 * @brief Advance the temperature field semi-implicitly with a zero source
 *        term.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Temperature field from the previous time step.
 * @param face_fluxes Pre-computed volumetric face fluxes.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Thermal diffusivity coefficient.
 * @param[out] temperature Updated temperature field on return.
 * @param linear_options Linear solver configuration.
 */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_semi_implicit(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
    field_type& temperature, const LinearSolverOptions& linear_options) const -> LinearSolveStatistics
{
    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    return advance_semi_implicit(
        old_temperature, face_fluxes, time_step, thermal_diffusivity, temperature, zero_source, linear_options);
}

/**
 * @brief Advance the temperature field semi-implicitly with a right-hand
 *        source term.
 *
 * Assembles and solves an advection-diffusion transport system for
 * temperature using the pre-computed face fluxes. The converged candidate is
 * published only after the solve succeeds, so @p old_temperature and
 * @p temperature may safely refer to the same accepted field.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Temperature field from the previous time step.
 * @param face_fluxes Pre-computed volumetric face fluxes.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Thermal diffusivity coefficient.
 * @param[out] temperature Updated temperature field on return.
 * @param right_hand_source Per-cell scalar source provider.
 * @param linear_options Linear solver configuration.
 * @throws std::invalid_argument on mesh mismatch, negative time step, or
 *         negative diffusivity.
 * @throws std::runtime_error if the linear transport solve does not converge.
 */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_semi_implicit(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
    field_type& temperature, const source_type& right_hand_source, const LinearSolverOptions& linear_options) const
    -> LinearSolveStatistics
{
    return advance_semi_implicit_impl(old_temperature, face_fluxes, time_step, thermal_diffusivity, temperature,
        right_hand_source, std::nullopt, linear_options);
}

/** @brief Advance zero-source transport with selected non-orthogonal diffusion. */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_semi_implicit(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
    field_type& temperature, FVM::NonOrthogonalTreatment treatment, const LinearSolverOptions& linear_options) const
    -> LinearSolveStatistics
{
    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };
    return advance_semi_implicit(old_temperature, face_fluxes, time_step, thermal_diffusivity, temperature, zero_source,
        treatment, linear_options);
}

/** @brief Advance sourced transport with selected non-orthogonal diffusion. */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_semi_implicit(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
    field_type& temperature, const source_type& right_hand_source, FVM::NonOrthogonalTreatment treatment,
    const LinearSolverOptions& linear_options) const -> LinearSolveStatistics
{
    return advance_semi_implicit_impl(old_temperature, face_fluxes, time_step, thermal_diffusivity, temperature,
        right_hand_source, treatment, linear_options);
}

/** @brief Shared orthogonal and non-orthogonal transport implementation. */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_semi_implicit_impl(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
    field_type& temperature, const source_type& right_hand_source, std::optional<FVM::NonOrthogonalTreatment> treatment,
    const LinearSolverOptions& linear_options) const -> LinearSolveStatistics
{
    EquationValidation::require_mesh_match(*d_mesh, old_temperature, "TemperatureDiffusionEquation");
    EquationValidation::require_mesh_match(*d_mesh, temperature, "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step", "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity", "TemperatureDiffusionEquation");

    const auto old_temperature_values = old_temperature.local_read_view();
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto name = d_mesh->boundary_batch_name(batch_id);
        const auto iter = d_boundary_condition->find(name);
        return iter == d_boundary_condition->end() ? BoundaryCondition{} : iter->second;
    };

    auto boundary_value = [&](int batch_id, size_t in_batch_id) -> typename Pack::scalar_type
    {
        const auto cache_it = d_face_boundary_temperature.value.find(batch_id);
        if (cache_it != d_face_boundary_temperature.value.end())
        {
            return cache_it->second[in_batch_id];
        }

        const auto face_lid = d_mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        const auto owner = d_mesh->owner_cell(face_lid);
        return old_temperature_values(owner, 0);
    };

    const auto requires_non_orthogonal_graph = treatment && *treatment != FVM::NonOrthogonalTreatment::Explicit;
    if (requires_non_orthogonal_graph && !d_cached_transport_graph_supports_non_orthogonal_correction)
    {
        d_cached_transport_matrix = Teuchos::null;
    }
    auto system = [&]()
    {
        try
        {
            if (treatment)
            {
                const auto* correction_field =
                    *treatment == FVM::NonOrthogonalTreatment::Implicit ? nullptr : &old_temperature;
                return FVM::non_orthogonal_transport_system<Pack>(old_temperature, face_fluxes, time_step,
                    thermal_diffusivity, boundary_condition, boundary_value, right_hand_source, *treatment,
                    correction_field, d_cached_transport_matrix, &d_transport_geometry_cache);
            }
            return FVM::transport_system<Pack>(old_temperature, face_fluxes, time_step, thermal_diffusivity,
                boundary_condition, boundary_value, right_hand_source, d_cached_transport_matrix);
        }
        catch (...)
        {
            // Failed reuse leaves the matrix in resume-fill mode.
            d_cached_transport_matrix = Teuchos::null;
            d_cached_transport_graph_supports_non_orthogonal_correction = false;
            throw;
        }
    }();
    d_cached_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph && thermal_diffusivity > scalar_type{})
    {
        d_cached_transport_graph_supports_non_orthogonal_correction = true;
    }

    if (system.rhs->norm2() == scalar_type{})
    {
        temperature.owned_data().putScalar(0.0);
        d_mesh->sync_periodic_boundaries(temperature);
        return {true, 0, 0.0};
    }

    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    auto& candidate_temperature = d_candidate_temperature;
    candidate_temperature.owned_data().update(scalar_type{1}, old_temperature.owned_data(), scalar_type{0});
    const auto statistics =
        d_linear_solver.solve_with_statistics(matrix, *system.rhs, candidate_temperature.owned_data(), linear_options);
    if (!statistics.converged)
    {
        throw std::runtime_error("TemperatureDiffusionEquation transport solve did not converge.");
    }
    temperature.owned_data().update(scalar_type{1}, candidate_temperature.owned_data(), scalar_type{0});
    d_mesh->sync_periodic_boundaries(temperature);
    return statistics;
}

/**
 * @brief Advance conservative temperature transport with physical properties.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_temperature Accepted temperature from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param time_step Positive time-step size.
 * @param material Density, heat-capacity, and conductivity fields.
 * @param[out] temperature Updated temperature field.
 * @param power_density Per-cell volumetric heat-source provider.
 * @param treatment Non-orthogonal diffusion treatment.
 * @param linear_options Linear-solver configuration.
 * @param thermal_conductivity_override Optional effective conductivity field.
 * @param boundary_thermal_conductivity Optional boundary conductivity cache.
 * @return Linear-solve statistics for the accepted update.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if the linear solve fails or is non-finite.
 */
template<TpetraTypePack Pack, class MeshType>
auto TemperatureDiffusionEquation<Pack, MeshType>::advance_physical(const field_type& old_temperature,
    const face_flux_field_type& face_fluxes, scalar_type time_step, const material_type& material,
    field_type& temperature, const source_type& power_density, FVM::NonOrthogonalTreatment treatment,
    const LinearSolverOptions& linear_options, const field_type* thermal_conductivity_override,
    const boundary_cache_type* boundary_thermal_conductivity,
    FVM::FaceCoefficientInterpolation coefficient_interpolation) const -> LinearSolveStatistics
{
    collective_detail::collective_local_validation(*d_mesh, "TemperatureDiffusionEquation physical input validation",
        [&]
        {
            EquationValidation::require_mesh_match(*d_mesh, old_temperature, "TemperatureDiffusionEquation");
            EquationValidation::require_mesh_match(*d_mesh, face_fluxes, "TemperatureDiffusionEquation");
            EquationValidation::require_mesh_match(*d_mesh, temperature, "TemperatureDiffusionEquation");
            EquationValidation::require_mesh_match(*d_mesh, material.density, "TemperatureDiffusionEquation");
            EquationValidation::require_mesh_match(
                *d_mesh, material.specific_heat_capacity, "TemperatureDiffusionEquation");
            EquationValidation::require_mesh_match(
                *d_mesh, material.thermal_conductivity, "TemperatureDiffusionEquation");
            if (!std::isfinite(time_step) || time_step <= scalar_type{})
            {
                throw std::invalid_argument("TemperatureDiffusionEquation requires a finite positive time step.");
            }
            if (!power_density)
            {
                throw std::invalid_argument("TemperatureDiffusionEquation requires a power-density provider.");
            }
            if (thermal_conductivity_override != nullptr)
            {
                EquationValidation::require_mesh_match(
                    *d_mesh, *thermal_conductivity_override, "TemperatureDiffusionEquation");
                for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto value = thermal_conductivity_override->value(static_cast<local_ordinal_type>(owned));
                    if (!std::isfinite(value) || value < scalar_type{})
                    {
                        throw std::invalid_argument("TemperatureDiffusionEquation thermal-conductivity "
                                                    "override must contain finite non-negative values.");
                    }
                }
            }
            FVM::validate_boundary_coefficient_cache<Pack>(
                *d_mesh, boundary_thermal_conductivity, "TemperatureDiffusionEquation");
        });

    collective_detail::require_uniform_value(*d_mesh, thermal_conductivity_override == nullptr ? 0 : 1,
        "TemperatureDiffusionEquation conductivity-override selection");
    collective_detail::require_uniform_value(*d_mesh, boundary_thermal_conductivity == nullptr ? 0 : 1,
        "TemperatureDiffusionEquation boundary-conductivity-cache selection");

    const auto& thermal_conductivity =
        thermal_conductivity_override == nullptr ? material.thermal_conductivity : *thermal_conductivity_override;
    const auto conductivity_values = thermal_conductivity.local_read_view();

    const auto old_temperature_values = old_temperature.local_read_view();
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto name = d_mesh->boundary_batch_name(batch_id);
        const auto iter = d_boundary_condition->find(name);
        return iter == d_boundary_condition->end() ? BoundaryCondition{} : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id) -> scalar_type
    {
        const auto iter = d_face_boundary_temperature.value.find(batch_id);
        if (iter != d_face_boundary_temperature.value.end())
        {
            return iter->second[in_batch_id];
        }

        const auto face_lid = d_mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        const auto owner = d_mesh->owner_cell(face_lid);
        return old_temperature_values(owner, 0);
    };

    int local_all_conductivities_positive = 1;
    for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(local);
        local_all_conductivities_positive =
            local_all_conductivities_positive && conductivity_values(cell_lid, 0) > scalar_type{};
    }
    int all_conductivities_positive = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_all_conductivities_positive,
        &all_conductivities_positive);
    const auto* correction_field = treatment == FVM::NonOrthogonalTreatment::Implicit ? nullptr : &old_temperature;
    const auto requires_non_orthogonal_graph = treatment != FVM::NonOrthogonalTreatment::Explicit;
    if (requires_non_orthogonal_graph && !d_cached_physical_graph_supports_non_orthogonal_correction)
    {
        d_cached_physical_transport_matrix = Teuchos::null;
    }
    auto system = [&]()
    {
        try
        {
            return FVM::physical_temperature_transport_system<Pack>(old_temperature, face_fluxes, time_step,
                material.density, material.specific_heat_capacity, thermal_conductivity, boundary_condition,
                boundary_value, power_density, treatment, correction_field, d_cached_physical_transport_matrix,
                boundary_thermal_conductivity, &d_transport_geometry_cache, coefficient_interpolation);
        }
        catch (...)
        {
            d_cached_physical_transport_matrix = Teuchos::null;
            d_cached_physical_graph_supports_non_orthogonal_correction = false;
            throw;
        }
    }();
    d_cached_physical_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph && all_conductivities_positive != 0)
    {
        d_cached_physical_graph_supports_non_orthogonal_correction = true;
    }

    auto& candidate_temperature = d_candidate_temperature;
    candidate_temperature.owned_data().update(scalar_type{1}, old_temperature.owned_data(), scalar_type{0});
    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto statistics =
        d_linear_solver.solve_with_statistics(matrix, *system.rhs, candidate_temperature.owned_data(), linear_options);
    if (!statistics.converged)
    {
        throw std::runtime_error("TemperatureDiffusionEquation physical transport solve did not converge.");
    }
    temperature.owned_data().update(scalar_type{1}, candidate_temperature.owned_data(), scalar_type{0});
    d_mesh->sync_periodic_boundaries(temperature);
    return statistics;
}

} // namespace SimpleFluid
