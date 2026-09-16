/** @file DelayedNeutronPrecursorModel.tcc @brief Compiled template implementations. */
#pragma once

#include "equations/DelayedNeutronPrecursorModel.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::configure(const DelayedNeutronPrecursorOptions& options)
{
    validate_collective_configuration(options);
    validate_delayed_neutron_precursor_options(options);
    d_options = options;
    d_fields.clear();
    d_inventories.clear();
    d_sources.clear();
    d_last_inventory_diagnostics.clear();
    d_output_fields.clear();
    d_inventory_initialized = false;
    const auto count = static_cast<size_t>(d_options.group_count);
    d_decay_constants = complete_vector(
        d_options.decay_constants, count, 0.0);
    d_source_terms = complete_vector(
        d_options.source_terms, count, 0.0);
    d_power_yields = complete_vector(
        d_options.power_yields, count, 0.0);
    d_initial_concentrations = complete_vector(
        d_options.initial_concentrations, count, 0.0);

    for (size_t group = 0; group < count; ++group)
    {
        const auto suffix = std::to_string(group + 1);
        auto field = std::make_unique<field_type>(
            d_mesh, d_initial_concentrations[group], "C_" + suffix);
        auto inventory = std::make_unique<field_type>(
            d_mesh, 0.0, "alpha_l_C_" + suffix);
        auto source = std::make_unique<field_type>(
            d_mesh, 0.0, "S_C_" + suffix);
        d_output_fields.emplace(field->name(), field.get());
        d_output_fields.emplace(source->name(), source.get());
        d_fields.push_back(std::move(field));
        d_inventories.push_back(std::move(inventory));
        d_sources.push_back(std::move(source));
        d_last_inventory_diagnostics.emplace_back();
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::initialize_inventory(const field_type& alpha_l)
{
    validate_liquid_fraction_field(alpha_l);
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        auto& field = *d_fields[group];
        auto& inventory = *d_inventories[group];
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto liquid_fraction =
                checked_liquid_fraction(alpha_l.value(cell_lid));
            inventory.set_owned_value(
                cell_lid,
                liquid_fraction * field.value(cell_lid));
        }
        inventory.sync_ghosts();
    }
    d_inventory_initialized = true;
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::advance(
    scalar_type time_step,
    const field_type& alpha_l,
    const field_type* fission_power_density,
    const face_flux_field_type* liquid_face_flux)
{
    validate_collective_advance_selection(
        time_step, fission_power_density, liquid_face_flux);
    if (!enabled())
    {
        return;
    }
    int local_invalid_input =
        !std::isfinite(time_step) || time_step <= scalar_type{}
        || &alpha_l.mesh() != d_mesh.get()
        || (fission_power_density != nullptr
            && &fission_power_density->mesh() != d_mesh.get())
        || (liquid_face_flux != nullptr
            && &liquid_face_flux->mesh() != d_mesh.get());
    const auto invalid_input = global_max(local_invalid_input);
    if (invalid_input != 0)
    {
        throw std::invalid_argument(
            "Precursor update requires a positive finite time step and "
            "fields on the model mesh.");
    }
    validate_liquid_fraction_field(alpha_l);
    int local_invalid_source = 0;
    if (fission_power_density != nullptr)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto value = fission_power_density->value(
                static_cast<local_ordinal_type>(owned));
            if (!std::isfinite(value) || value < scalar_type{})
            {
                local_invalid_source = 1;
            }
        }
    }
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        const auto& inventory = *d_inventories[group];
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto source =
                d_source_terms[group]
              + (fission_power_density
                     ? d_power_yields[group]
                     * fission_power_density->value(cell_lid)
                     : scalar_type{});
            const auto old_inventory = inventory.value(cell_lid);
            if (!std::isfinite(source) || source < scalar_type{}
                || !std::isfinite(old_inventory)
                || old_inventory < scalar_type{})
            {
                local_invalid_source = 1;
            }
        }
    }
    if (liquid_face_flux != nullptr)
    {
        for (const auto face_lid : liquid_face_flux->owned_face_ids())
        {
            if (!std::isfinite(liquid_face_flux->value(face_lid)))
            {
                local_invalid_source = 1;
            }
        }
    }
    if (global_max(local_invalid_source) != 0)
    {
        throw std::invalid_argument(
            "Precursor inventories, sources, fission power, and liquid "
            "fluxes must be finite and non-negative where applicable.");
    }
    if (!d_inventory_initialized)
    {
        initialize_inventory(alpha_l);
    }

    std::vector<scalar_type> reaction_inventory_reference(
        d_fields.size());
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        auto& inventory = *d_inventories[group];
        auto& source_field = *d_sources[group];
        auto& diagnostics = d_last_inventory_diagnostics[group];
        diagnostics = {};
        const auto lambda = d_decay_constants[group];
        scalar_type local_inventory_before{};
        scalar_type local_source_added{};
        scalar_type local_decay_removed{};
        scalar_type local_reaction_inventory_reference{};
        scalar_type local_positivity_adjustment{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto source =
                d_source_terms[group]
              + (fission_power_density
                     ? d_power_yields[group]
                     * fission_power_density->value(cell_lid)
                     : scalar_type{});
            const auto old_inventory = inventory.value(cell_lid);
            const auto volume = d_mesh->cell_volume(cell_lid);
            scalar_type new_inventory{};
            scalar_type source_added{};
            scalar_type decay_removed{};
            if (lambda > scalar_type{})
            {
                const auto decay_exponent = lambda * time_step;
                const auto decay = std::exp(-decay_exponent);
                const auto reacted_fraction = -std::expm1(-decay_exponent);
                // Exact source integral dt * phi_1(-decay_exponent).
                auto source_response = time_step;
                if (decay_exponent > scalar_type{})
                {
                    source_response = std::isfinite(decay_exponent)
                                          ? time_step * (reacted_fraction / decay_exponent)
                                          : reacted_fraction / lambda;
                }
                const auto surviving_source = source * source_response;
                source_added = time_step * source;
                decay_removed =
                    old_inventory * reacted_fraction
                  + (source_added - surviving_source);
                new_inventory = std::fma(old_inventory, decay, surviving_source);
            }
            else
            {
                source_added = time_step * source;
                new_inventory = old_inventory + source_added;
            }
            const auto unclipped_inventory = new_inventory;
            new_inventory = std::max(new_inventory, scalar_type{});
            local_inventory_before += old_inventory * volume;
            local_source_added += source_added * volume;
            local_decay_removed += decay_removed * volume;
            local_reaction_inventory_reference +=
                unclipped_inventory * volume;
            local_positivity_adjustment +=
                (new_inventory - unclipped_inventory) * volume;
            inventory.set_owned_value(cell_lid, new_inventory);
            source_field.set_owned_value(cell_lid, source);
        }
        inventory.sync_ghosts();
        source_field.sync_ghosts();
        diagnostics.inventory_before = global_sum(local_inventory_before);
        diagnostics.source_added = global_sum(local_source_added);
        diagnostics.decay_removed = global_sum(local_decay_removed);
        reaction_inventory_reference[group] =
            global_sum(local_reaction_inventory_reference);
        diagnostics.positivity_adjustment =
            global_sum(local_positivity_adjustment);
    }

    if (liquid_face_flux != nullptr
        || d_options.effective_diffusivity > scalar_type{})
    {
        transport(time_step, alpha_l, liquid_face_flux);
    }
    else
    {
        reconstruct_concentrations(alpha_l);
    }
    update_inventory_diagnostics(
        time_step,
        alpha_l,
        liquid_face_flux,
        reaction_inventory_reference);
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::validate_collective_configuration(
    const DelayedNeutronPrecursorOptions& options) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    int local_options_are_valid = 1;
    std::string local_error;
    try
    {
        validate_delayed_neutron_precursor_options(options);
    }
    catch (const std::invalid_argument& error)
    {
        local_options_are_valid = 0;
        local_error = error.what();
    }

    int all_options_are_valid = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_options_are_valid,
        &all_options_are_valid);
    if (all_options_are_valid == 0)
    {
        if (communicator->getSize() == 1 && !local_error.empty())
        {
            throw std::invalid_argument(local_error);
        }
        throw std::invalid_argument(
            "Precursor options must be valid on every rank.");
    }

    const auto local_group_count = options.group_count;
    int minimum_group_count = 0;
    int maximum_group_count = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_group_count,
        &minimum_group_count);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_group_count,
        &maximum_group_count);
    if (minimum_group_count != maximum_group_count)
    {
        throw std::invalid_argument(
            "Precursor group count must match on every rank.");
    }

    const auto count = static_cast<size_t>(minimum_group_count);
    const auto decay_constants = complete_vector(
        options.decay_constants, count, 0.0);
    const auto initial_concentrations = complete_vector(
        options.initial_concentrations, count, 0.0);
    const auto source_terms = complete_vector(
        options.source_terms, count, 0.0);
    const auto power_yields = complete_vector(
        options.power_yields, count, 0.0);
    std::vector<real_t> local_configuration;
    local_configuration.reserve(1 + 4 * count);
    local_configuration.push_back(options.effective_diffusivity);
    for (size_t group = 0; group < count; ++group)
    {
        local_configuration.push_back(decay_constants[group]);
        local_configuration.push_back(initial_concentrations[group]);
        local_configuration.push_back(source_terms[group]);
        local_configuration.push_back(power_yields[group]);
    }

    std::vector<real_t> minimum_configuration(
        local_configuration.size());
    std::vector<real_t> maximum_configuration(
        local_configuration.size());
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        static_cast<int>(local_configuration.size()),
        local_configuration.data(),
        minimum_configuration.data());
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        static_cast<int>(local_configuration.size()),
        local_configuration.data(),
        maximum_configuration.data());
    if (minimum_configuration != maximum_configuration)
    {
        throw std::invalid_argument(
            "Precursor decay, initial concentration, source, yield, and "
            "diffusivity options must match on every rank.");
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::validate_collective_advance_selection(
    scalar_type time_step,
    const field_type* fission_power_density,
    const face_flux_field_type* liquid_face_flux) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const std::array<int, 7> local_state{{
        enabled() ? 1 : 0,
        static_cast<int>(d_fields.size()),
        fission_power_density != nullptr ? 1 : 0,
        liquid_face_flux != nullptr ? 1 : 0,
        d_inventory_initialized ? 1 : 0,
        d_options.effective_diffusivity > scalar_type{} ? 1 : 0,
        std::isfinite(time_step) && time_step > scalar_type{} ? 1 : 0}};
    std::array<int, 7> minimum_state{};
    std::array<int, 7> maximum_state{};
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        static_cast<int>(local_state.size()),
        local_state.data(),
        minimum_state.data());
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        static_cast<int>(local_state.size()),
        local_state.data(),
        maximum_state.data());
    if (minimum_state != maximum_state)
    {
        throw std::invalid_argument(
            "Precursor enabled state, group count, optional fields, and "
            "valid timestep selection must agree on every rank.");
    }

    std::vector<scalar_type> local_configuration;
    local_configuration.reserve(2 + 4 * d_fields.size());
    local_configuration.push_back(d_options.effective_diffusivity);
    if (minimum_state.back() != 0)
    {
        local_configuration.push_back(time_step);
    }
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        local_configuration.push_back(d_decay_constants[group]);
        local_configuration.push_back(
            d_initial_concentrations[group]);
        local_configuration.push_back(d_source_terms[group]);
        local_configuration.push_back(d_power_yields[group]);
    }
    std::vector<scalar_type> minimum_configuration(
        local_configuration.size());
    std::vector<scalar_type> maximum_configuration(
        local_configuration.size());
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        static_cast<int>(local_configuration.size()),
        local_configuration.data(),
        minimum_configuration.data());
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        static_cast<int>(local_configuration.size()),
        local_configuration.data(),
        maximum_configuration.data());
    if (minimum_configuration != maximum_configuration)
    {
        throw std::invalid_argument(
            "Precursor timestep, decay, initial concentration, source, "
            "yield, and diffusivity options must agree on every rank.");
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::validate_liquid_fraction_field(const field_type& alpha_l) const
{
    int local_invalid = &alpha_l.mesh() != d_mesh.get();
    if (local_invalid == 0)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto value = alpha_l.value(
                static_cast<local_ordinal_type>(owned));
            if (!std::isfinite(value) || value <= scalar_type{}
                || value > scalar_type{1})
            {
                local_invalid = 1;
            }
        }
    }
    if (global_max(local_invalid) != 0)
    {
        throw std::invalid_argument(
            "Precursor liquid fractions must be on the model mesh, "
            "finite, and in (0, 1].");
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::reconstruct_concentrations(const field_type& alpha_l)
{
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        auto& field = *d_fields[group];
        const auto& inventory = *d_inventories[group];
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto liquid_fraction =
                checked_liquid_fraction(alpha_l.value(cell_lid));
            field.set_owned_value(
                cell_lid,
                inventory.value(cell_lid) / liquid_fraction);
        }
        field.sync_ghosts();
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::transport(
    scalar_type time_step,
    const field_type& alpha_l,
    const face_flux_field_type* liquid_face_flux)
{
    face_flux_field_type zero_flux(
        d_mesh, 0.0, "precursor_zero_face_flux");
    const auto& face_flux =
        liquid_face_flux == nullptr ? zero_flux : *liquid_face_flux;
    field_type storage_weight(
        d_mesh, 1.0, "precursor_storage_weight");
    field_type advection_weight(
        d_mesh, 1.0, "precursor_advection_weight");
    field_type diffusion_weight(
        d_mesh, 0.0, "precursor_diffusion_weight");
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto liquid_fraction =
            checked_liquid_fraction(alpha_l.value(cell_lid));
        storage_weight.set_owned_value(cell_lid, liquid_fraction);
        advection_weight.set_owned_value(cell_lid, liquid_fraction);
        diffusion_weight.set_owned_value(
            cell_lid,
            liquid_fraction * d_options.effective_diffusivity);
    }
    storage_weight.sync_ghosts();
    advection_weight.sync_ghosts();
    diffusion_weight.sync_ghosts();

    auto boundary_condition =
        [](int, size_t)
    {
        return BoundaryCondition{
            BoundaryConditionType::Neumann, 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> scalar_type
    {
        return scalar_type{};
    };
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        auto& field = *d_fields[group];
        auto& inventory = *d_inventories[group];
        field_type old_concentration(
            d_mesh, "precursor_old_concentration");
        scalar_type concentration_scale{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            old_concentration.set_owned_value(
                cell_lid,
                inventory.value(cell_lid)
              / storage_weight.value(cell_lid));
            concentration_scale = std::max(
                concentration_scale,
                std::abs(old_concentration.value(cell_lid)));
        }
        old_concentration.sync_ghosts();
        concentration_scale = global_maximum(concentration_scale);

        auto system = FVM::weighted_scalar_transport_system<Pack>(
            FVM::MeshWeightedScalarTransportRequest<Pack, mesh_type>{
                .old_values = old_concentration,
                .face_fluxes = face_flux,
                .time_step = time_step,
                .storage_weight = storage_weight,
                .advection_weight = advection_weight,
                .diffusivity = diffusion_weight,
                .boundary_condition = boundary_condition,
                .boundary_value = boundary_value,
                .source = zero_source,
                .treatment = FVM::NonOrthogonalTreatment::Explicit,
                .correction_field = &old_concentration,
                .geometry_cache = &*d_transport_geometry_cache});
        field_type solution(d_mesh, "precursor_diffusion_solution");
        const auto statistics =
            d_transport_solver.solve_with_statistics(
                system.matrix,
                *system.rhs,
                solution.owned_data(),
                LinearSolverOptions{});
        if (global_max(statistics.converged ? 0 : 1) != 0)
        {
            throw std::runtime_error(
                "Precursor transport solve did not converge.");
        }

        const auto negative_tolerance =
            scalar_type{100}
          * std::numeric_limits<scalar_type>::epsilon()
          * std::max(concentration_scale, scalar_type{1});
        int local_nonfinite = 0;
        int local_too_negative = 0;
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            auto concentration = solution.value(cell_lid);
            if (!std::isfinite(concentration))
            {
                local_nonfinite = 1;
            }
            if (concentration < -negative_tolerance)
            {
                local_too_negative = 1;
            }
        }
        if (global_max(local_nonfinite) != 0)
        {
            throw std::runtime_error(
                "Precursor transport produced a non-finite value.");
        }
        if (global_max(local_too_negative) != 0)
        {
            throw std::runtime_error(
                "Precursor transport produced a negative value.");
        }

        scalar_type local_positivity_adjustment{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            auto concentration = solution.value(cell_lid);
            if (concentration < scalar_type{})
            {
                local_positivity_adjustment +=
                    -concentration * storage_weight.value(cell_lid)
                  * d_mesh->cell_volume(cell_lid);
                concentration = scalar_type{};
            }
            field.set_owned_value(cell_lid, concentration);
            inventory.set_owned_value(
                cell_lid,
                storage_weight.value(cell_lid) * concentration);
        }
        d_last_inventory_diagnostics[group].positivity_adjustment +=
            global_sum(local_positivity_adjustment);
        field.sync_ghosts();
        inventory.sync_ghosts();
    }
}

template<TpetraTypePack Pack, class MeshType>
void DelayedNeutronPrecursorModel<Pack, MeshType>::update_inventory_diagnostics(
    scalar_type time_step,
    const field_type& alpha_l,
    const face_flux_field_type* liquid_face_flux,
    const std::vector<scalar_type>& reaction_inventory_reference)
{
    for (size_t group = 0; group < d_fields.size(); ++group)
    {
        scalar_type local_inventory_after{};
        scalar_type local_boundary_outflow_rate{};
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            local_inventory_after += d_inventories[group]->value(cell_lid)
                                     * d_mesh->cell_volume(cell_lid);
        }

        if (liquid_face_flux != nullptr)
        {
            for (const auto& [batch_id, boundary_batch] :
                 d_mesh->boundary_batches())
            {
                static_cast<void>(batch_id);
                for (const auto face_lid : boundary_batch.face_lids)
                {
                    if (!d_mesh->is_boundary_face(face_lid)
                        || !liquid_face_flux->is_owned_face(face_lid))
                    {
                        continue;
                    }
                    const auto out_flux = liquid_face_flux->value(face_lid);
                    if (out_flux <= scalar_type{})
                    {
                        continue;
                    }
                    const auto owner = d_mesh->owner_cell(face_lid);
                    local_boundary_outflow_rate +=
                        out_flux * alpha_l.value(owner)
                        * d_fields[group]->value(owner);
                }
            }
        }

        auto& diagnostics = d_last_inventory_diagnostics[group];
        diagnostics.inventory_after = global_sum(local_inventory_after);
        diagnostics.boundary_outflow =
            time_step * global_sum(local_boundary_outflow_rate);
        // Gross production and decay can be nearly equal and cannot
        // reliably reconstruct a small surviving inventory. Use the
        // directly integrated reaction state for balance closure.
        const auto expected_inventory =
            reaction_inventory_reference[group]
            + diagnostics.positivity_adjustment
            - diagnostics.boundary_outflow;
        diagnostics.balance_error =
            diagnostics.inventory_after - expected_inventory;
    }
}

} // namespace SimpleFluid
