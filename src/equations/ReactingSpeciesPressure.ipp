/** @file ReactingSpeciesPressure.ipp
 * @brief Pure current-inventory candidate EOS checks and liquid-volume query.
 */
#pragma once
namespace SimpleFluid
{
template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::validate_species_pressure_metadata(scalar_type pressure_offset) const
{
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate pressure metadata",
        [&]
        {
            reacting_species_detail::require(species_enabled() && d_initial_state_initialized &&
                                                 (d_options.pressure_mode == ReactingPressureMode::Constant ||
                                                     d_options.pressure_mode == ReactingPressureMode::Reconstructed) &&
                                                 std::isfinite(pressure_offset),
                "Candidate pressure requires initialized species, supported pressure mode and finite offset.");
        });
    collective_detail::require_uniform_value(*d_mesh, pressure_offset, "Reacting candidate pressure offset");
    const auto lower = minimum_valid_absolute_pressure_offset();
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate pressure floor",
        [&]
        {
            reacting_species_detail::require(
                pressure_offset >= lower, "Candidate pressure is below its collective floor.");
        });
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::evaluate_liquid_volume(scalar_type pressure_offset) const -> scalar_type
{
    using reacting_species_detail::require;
    validate_species_pressure_metadata(pressure_offset);
    detail::CompensatedSum<long double> local;
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate liquid volume",
        [&]
        {
            const auto& catalog = d_options.reaction_mechanism->catalog();
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto state = cell_species_state(cell);
                std::vector<double> liquid(catalog.species().size());
                for (std::size_t s = 0; s < liquid.size(); ++s)
                {
                    detail::CompensatedSum<long double> amount;
                    amount += state.amount(catalog, catalog.species()[s].id, Reacting::SpeciesPhase::Liquid);
                    amount += state.amount(catalog, catalog.species()[s].id, Reacting::SpeciesPhase::Dissolved);
                    liquid[s] = reacting_species_detail::narrow(amount.value());
                }
                // An exactly empty phase has no fictitious EOS state.
                if (reacting_species_detail::total(liquid) == 0)
                    continue;
                const auto pressure =
                    reacting_species_detail::narrow(static_cast<long double>(d_absolute_pressure.value(cell)) +
                                                    pressure_offset - d_absolute_pressure_offset);
                const auto point = d_options.liquid_thermodynamics->evaluate(
                    d_previous_temperature.value(cell), pressure, reacting_species_detail::composition(liquid));
                const auto volume = reacting_species_detail::mass(catalog, liquid) / point.density;
                require(std::isfinite(volume) && volume > 0, "Candidate liquid EOS returned invalid volume.");
                local += volume;
            }
            require(std::isfinite(local.value()), "Candidate local liquid-volume sum overflowed.");
        });
    scalar_type value{}, global{};
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate liquid-volume reduction input",
        [&] { value = reacting_species_detail::narrow(local.value()); });
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &value, &global);
    scalar_type result{};
    collective_detail::collective_local_validation(
        *d_mesh, "Reacting candidate global liquid volume", [&] { result = reacting_species_detail::narrow(global); });
    return result;
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::validate_species_candidate_pressure(scalar_type pressure_offset) const
{
    using reacting_species_detail::require;
    // This validates metadata and every occupied liquid mixture before any accepted pressure writes.
    (void) evaluate_liquid_volume(pressure_offset);
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate gas occupancy",
        [&]
        {
            const auto& catalog = d_options.reaction_mechanism->catalog();
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto volume = d_mesh->cell_volume(cell);
                const auto state = cell_species_state(cell, volume);
                auto environment = species_environment(cell, 0, d_previous_temperature.value(cell), volume);
                environment.absolutePressure =
                    reacting_species_detail::narrow(static_cast<long double>(environment.absolutePressure) +
                                                    pressure_offset - d_absolute_pressure_offset);
                const auto numbers = cell_population_numbers(cell, volume);
                for (std::size_t p = 0; p < numbers.size(); ++p)
                {
                    const auto amount = reacting_species_detail::total(state.dispersedPopulations[p].amounts);
                    require(std::isfinite(numbers[p].number) && numbers[p].number >= 0 &&
                                ((amount == 0 && numbers[p].number == 0) || (amount > 0 && numbers[p].number > 0)),
                        "Candidate bubble amount/count pairing is invalid.");
                }
                const auto conditions = population_conditions(state, numbers, environment);
                detail::CompensatedSum<long double> gas;
                std::vector<double> resolved(catalog.species().size());
                for (std::size_t s = 0; s < resolved.size(); ++s)
                {
                    resolved[s] = state.amount(catalog, catalog.species()[s].id, Reacting::SpeciesPhase::Resolved);
                    require(state.amount(catalog, catalog.species()[s].id, Reacting::SpeciesPhase::Dispersed) == 0,
                        "Candidate dispersed inventory must name a population.");
                }
                if (reacting_species_detail::total(resolved) > 0)
                    gas += reacting_species_detail::mass(catalog, resolved) /
                           d_options.gas_thermodynamics
                               ->evaluate(environment.temperature, environment.absolutePressure,
                                   reacting_species_detail::composition(resolved))
                               .density;
                for (std::size_t p = 0; p < conditions.size(); ++p)
                    if (reacting_species_detail::total(state.dispersedPopulations[p].amounts) > 0)
                        gas += reacting_species_detail::mass(catalog, state.dispersedPopulations[p].amounts) /
                               conditions[p].properties.density;
                const auto fraction = gas.value() / volume;
                require(std::isfinite(fraction) && fraction >= 0 && fraction <= d_options.alpha_max,
                    "Candidate gas occupancy exceeds configured phase capacity.");
            }
        });
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::set_species_absolute_pressure_offset(scalar_type pressure_offset)
{
    // Pure all-phase validation precedes any writes. An unexpected publication
    // failure also restores fields, pressure and compensated accepted histories.
    validate_species_candidate_pressure(pressure_offset);
    StateSnapshot accepted;
    collective_detail::collective_local_validation(
        *d_mesh, "Reacting candidate pressure snapshot", [&] { accepted = snapshot(); });
    try
    {
        collective_detail::collective_local_validation(*d_mesh, "Reacting candidate pressure publication",
            [&]
            {
                for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell = static_cast<local_ordinal_type>(owned);
                    d_absolute_pressure.owned_write_view()(owned, 0) =
                        reacting_species_detail::narrow(static_cast<long double>(d_absolute_pressure.value(cell)) +
                                                        pressure_offset - d_absolute_pressure_offset);
                }
                d_absolute_pressure_offset = pressure_offset;
            });
        reconstruct_species_derived(d_previous_temperature);
        collective_detail::collective_local_validation(*d_mesh, "Reacting pressure derived-state publication",
            [&]
            {
                for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell = static_cast<local_ordinal_type>(owned);
                    d_previous_alpha_g.owned_write_view()(owned, 0) = d_alpha_g.value(cell);
                }
            });
        d_last_statistics.void_volume = global_integral(d_alpha_g);
        sync_all_fields();
    }
    catch (...)
    {
        restore(accepted);
        throw;
    }
}
} // namespace SimpleFluid
