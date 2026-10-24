/** @file ReactingSpeciesHistory.ipp
 * @brief Compiled accepted event accounting; never advances chemistry or transport.
 */
#pragma once
namespace SimpleFluid
{
template<TpetraTypePack Pack, class MeshType>
std::size_t ReactingGasModel<Pack, MeshType>::species_compartment_index(
    Reacting::SpeciesId species, Reacting::SpeciesPhase phase, std::string_view population) const
{
    const auto& catalog = d_options.reaction_mechanism->catalog();
    const auto s = catalog.index(species), ns = catalog.species().size();
    reacting_species_detail::require(static_cast<std::size_t>(phase) < 4, "Invalid history phase.");
    if (population.empty())
        return static_cast<std::size_t>(phase) * ns + s;
    reacting_species_detail::require(
        phase == Reacting::SpeciesPhase::Dispersed, "Only dispersed events name populations.");
    const auto found = std::ranges::find(d_options.population_ids, population);
    reacting_species_detail::require(found != d_options.population_ids.end(), "Unknown history population.");
    return (4 + static_cast<std::size_t>(found - d_options.population_ids.begin())) * ns + s;
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::publish_species_history(std::span<const Reacting::CellSpeciesState> states,
    species_step_sums step, bool initialize, std::span<const Reacting::ReactionLocalReceipt> chemical_receipts)
{
    namespace H = Reacting::history_detail;
    using Sum = H::Sum;
    using reacting_species_detail::narrow;
    using reacting_species_detail::require;
    const auto& mechanism = *d_options.reaction_mechanism;
    const auto& catalog = mechanism.catalog();
    const auto ns = catalog.species().size(), compartments = 4 + d_options.population_ids.size();
    H::State candidate;
    std::vector<double> local, global;
    collective_detail::collective_local_validation(*d_mesh, "Reacting accepted history events",
        [&]
        {
            require(states.size() == d_mesh->num_owned_cells(), "History states must follow owned order.");
            require(initialize || d_species_history.initialized, "Initialize species before recording histories.");
            candidate = initialize ? H::State{} : d_species_history;
            if (initialize)
            {
                candidate.initial_local.resize(ns);
                candidate.net_local.resize(ns);
                candidate.cumulative_local.resize(ns);
                candidate.initialized = true;
                step.resize(ns * compartments);
            }
            require(step.size() == ns * compartments && candidate.initial_local.size() == ns &&
                        candidate.cumulative_local.size() == ns && candidate.net_local.size() == ns,
                "Species history schema/topology differs.");
            std::vector<Sum> current(ns);
            for (const auto& state : states)
            {
                state.validate(catalog);
                for (std::size_t s = 0; s < ns; ++s)
                    for (std::size_t phase = 0; phase < 4; ++phase)
                        current[s] += state.phase_amount(
                            catalog, catalog.species()[s].id, static_cast<Reacting::SpeciesPhase>(phase));
            }
            if (initialize)
                candidate.initial_local = current;
            const auto add = [&](std::size_t slot, H::EventMetric metric, long double amount)
            {
                require(std::isfinite(amount) && amount >= 0, "History event amount is invalid.");
                step[slot].amount[metric] += amount;
            };
            const auto total = [&](std::size_t s, H::CumulativeMetric metric, long double amount)
            {
                require(std::isfinite(amount) && amount >= 0, "Cumulative history amount is invalid.");
                candidate.cumulative_local[s][metric] += amount;
                // Gross reporting totals never seed the independently accumulated signed net.
            };
            if (!initialize)
            {
                require(d_species_source_ledgers.size() == states.size() &&
                            d_species_escape_ledgers.size() == states.size() &&
                            chemical_receipts.size() == states.size(),
                    "Accepted event ledgers/raw chemistry receipts have wrong owned size.");
                std::vector<std::vector<std::pair<std::size_t, long double>>> net(mechanism.reactions().size());
                for (std::size_t r = 0; r < mechanism.reactions().size(); ++r)
                {
                    std::vector<Sum> coefficients(ns);
                    for (const auto& term : mechanism.reactions()[r].changes)
                        coefficients[catalog.index(term.species)] += term.coefficient;
                    for (std::size_t s = 0; s < ns; ++s)
                        if (coefficients[s].value() != 0)
                            net[r].push_back({s, coefficients[s].value()});
                }
                for (std::size_t cell = 0; cell < states.size(); ++cell)
                {
                    const auto& source = d_species_source_ledgers[cell];
                    const auto& raw = chemical_receipts[cell];
                    require(raw.schemaFingerprint == catalog.fingerprint(),
                        "Accepted raw chemistry history belongs to another ordered schema.");
                    for (const auto& change : raw.chemicalChanges)
                    {
                        require(std::isfinite(change.moles), "Raw accepted chemical change is nonfinite.");
                        const auto slot = species_compartment_index(change.species, change.phase, change.population);
                        const auto s = catalog.index(change.species);
                        step[slot].source_change += change.moles;
                        candidate.net_local[s] += change.moles;
                        if (change.phase != Reacting::SpeciesPhase::Liquid)
                            step[s].gas_subset_source += change.moles;
                        if (change.phase == Reacting::SpeciesPhase::Liquid ||
                            change.phase == Reacting::SpeciesPhase::Dissolved)
                            step[s].solution_subset_source += change.moles;
                    }
                    require(source.forwardReactionExtents.size() == mechanism.reactions().size() &&
                                source.reverseReactionExtents.size() == mechanism.reactions().size(),
                        "Directional chemistry provenance is absent or has the wrong mechanism.");
                    for (std::size_t r = 0; r < mechanism.reactions().size(); ++r)
                    {
                        // Rounded directional extents are gross provenance only. Signed channels above
                        // consume the kernel's accepted keyed local receipt, never reconstructed extents.
                        require(std::isfinite(source.forwardReactionExtents[r]) &&
                                    source.forwardReactionExtents[r] >= 0 &&
                                    std::isfinite(source.reverseReactionExtents[r]) &&
                                    source.reverseReactionExtents[r] >= 0,
                            "Directional extents must be finite and nonnegative.");
                        for (long double extent : {static_cast<long double>(source.forwardReactionExtents[r]),
                                 -static_cast<long double>(source.reverseReactionExtents[r])})
                        {
                            for (const auto& term : mechanism.reactions()[r].changes)
                            {
                                const long double delta = term.coefficient * extent;
                                add(species_compartment_index(term.species, term.phase, term.population),
                                    delta >= 0 ? H::ChemicalProduction : H::ChemicalConsumption, std::abs(delta));
                            }
                            // Net species coefficients avoid treating phase relocation as molecular creation.
                            for (const auto& [s, coefficient] : net[r])
                            {
                                const long double delta = coefficient * extent;
                                total(s, delta >= 0 ? H::Produced : H::Consumed, std::abs(delta));
                            }
                        }
                    }
                    for (const auto& transfer : source.externalTransfers)
                    {
                        const auto amount = std::abs(static_cast<long double>(transfer.moles));
                        const auto slot =
                            species_compartment_index(transfer.species, transfer.phase, transfer.population);
                        const auto s = catalog.index(transfer.species);
                        candidate.net_local[s] += transfer.moles;
                        step[slot].source_change += transfer.moles;
                        if (transfer.phase != Reacting::SpeciesPhase::Liquid)
                            step[s].gas_subset_source += transfer.moles;
                        if (transfer.phase == Reacting::SpeciesPhase::Liquid ||
                            transfer.phase == Reacting::SpeciesPhase::Dissolved)
                            step[s].solution_subset_source += transfer.moles;
                        add(slot, transfer.moles >= 0 ? H::ExternalSupply : H::ExternalWithdrawal, amount);
                        total(
                            catalog.index(transfer.species), transfer.moles >= 0 ? H::Supplied : H::Withdrawn, amount);
                    }
                    // Physical transport escape has distinct typed provenance. No reservoir-name inference.
                    for (const auto& event : d_species_escape_ledgers[cell].externalTransfers)
                    {
                        require(std::isfinite(event.moles) && event.moles <= 0 && std::isfinite(event.mass) &&
                                    event.mass <= 0 && std::isfinite(event.energy),
                            "Invalid physical escape trace.");
                        const auto slot = species_compartment_index(event.species, event.phase, event.population);
                        add(slot, H::PhysicalOutflow, -static_cast<long double>(event.moles));
                        total(catalog.index(event.species), H::Escaped, -static_cast<long double>(event.moles));
                        candidate.net_local[catalog.index(event.species)] += event.moles;
                        step[slot].escaped_mass += -static_cast<long double>(event.mass);
                        step[slot].escaped_enthalpy += -static_cast<long double>(event.energy);
                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            require(std::isfinite(event.momentum[axis]), "Nonfinite escape momentum trace.");
                            step[slot].escaped_momentum[axis] += -static_cast<long double>(event.momentum[axis]);
                        }
                    }
                }
            }
            const auto append = [&](long double value)
            {
                require(std::isfinite(value), "History reduction input overflowed.");
                local.push_back(narrow(value));
            };
            for (std::size_t s = 0; s < ns; ++s)
            {
                append(candidate.initial_local[s].value());
                append(current[s].value());
                append(candidate.net_local[s].value());
                for (const auto& amount : candidate.cumulative_local[s])
                    append(amount.value());
            }
            for (std::size_t s = 0; s < ns; ++s)
            {
                Sum gas, solution;
                for (std::size_t compartment = 0; compartment < compartments; ++compartment)
                {
                    gas += step[compartment * ns + s].gas_subset_source.value();
                    solution += step[compartment * ns + s].solution_subset_source.value();
                }
                append(gas.value());
                append(solution.value());
            }
            for (const auto& event : step)
            {
                append(event.source_change.value());
                for (const auto& amount : event.amount)
                    append(amount.value());
                append(event.escaped_mass.value());
                append(event.escaped_enthalpy.value());
                for (const auto& momentum : event.escaped_momentum)
                    append(momentum.value());
            }
            require(local.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "History reduction exceeds collective ordinal range.");
            global.resize(local.size());
        });
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
        local.data(), global.data());
    collective_detail::collective_local_validation(*d_mesh, "Reacting global history publication",
        [&]
        {
            std::size_t position = 0;
            candidate.inventory.resize(ns);
            for (std::size_t s = 0; s < ns; ++s)
            {
                auto& report = candidate.inventory[s];
                report.species = catalog.species()[s].id;
                const auto initial = global[position++], current = global[position++];
                const auto net_change = global[position++];
                std::array<long double, H::CumulativeMetricCount> amounts;
                for (auto& amount : amounts)
                    amount = global[position++];
                report.initial_moles = narrow(initial);
                report.current_moles = narrow(current);
                report.chemical_production = narrow(amounts[H::Produced]);
                report.chemical_consumption = narrow(amounts[H::Consumed]);
                report.external_supply = narrow(amounts[H::Supplied]);
                report.external_withdrawal = narrow(amounts[H::Withdrawn]);
                report.physical_outflow = narrow(amounts[H::Escaped]);
                // Gross diagnostic channels may round away a small difference.
                // The independently accumulated net event remains authoritative.
                Sum residual;
                residual += initial;
                residual += net_change;
                residual += -current;
                report.inventory_error = narrow(residual.value());
                const auto scale = std::max<long double>({std::abs(initial), std::abs(current), std::abs(net_change)});
                require(std::abs(residual.value()) <= 4096 * std::numeric_limits<double>::epsilon() * scale,
                    "Cumulative species history fails closure on its actual inventory scale.");
            }
            candidate.subset_step.resize(ns);
            for (std::size_t s = 0; s < ns; ++s)
                candidate.subset_step[s] = {
                    catalog.species()[s].id, narrow(global[position++]), narrow(global[position++])};
            candidate.step.clear();
            candidate.escape.clear();
            for (std::size_t compartment = 0; compartment < compartments; ++compartment)
                for (std::size_t s = 0; s < ns; ++s)
                {
                    Reacting::SpeciesCompartmentEvents report;
                    report.species = catalog.species()[s].id;
                    report.phase = compartment < 4 ? static_cast<Reacting::SpeciesPhase>(compartment)
                                                   : Reacting::SpeciesPhase::Dispersed;
                    report.population = compartment < 4 ? std::string{} : d_options.population_ids[compartment - 4];
                    report.signed_source_change = narrow(global[position++]);
                    std::array<long double, H::EventMetricCount> amount;
                    for (auto& value : amount)
                        value = global[position++];
                    report.chemical_production = narrow(amount[H::ChemicalProduction]);
                    report.chemical_consumption = narrow(amount[H::ChemicalConsumption]);
                    report.external_supply = narrow(amount[H::ExternalSupply]);
                    report.external_withdrawal = narrow(amount[H::ExternalWithdrawal]);
                    report.interphase_inflow = narrow(amount[H::InterphaseInflow]);
                    report.interphase_outflow = narrow(amount[H::InterphaseOutflow]);
                    report.physical_outflow = narrow(amount[H::PhysicalOutflow]);
                    const auto mass = global[position++], enthalpy = global[position++];
                    std::array<double, 3> momentum;
                    for (auto& value : momentum)
                        value = narrow(global[position++]);
                    if (amount[H::PhysicalOutflow] > 0)
                        candidate.escape.push_back(
                            {report.species, report.phase, report.population, report.physical_outflow, narrow(mass),
                                narrow(enthalpy), momentum, mechanism.energy_convention(),
                                Reacting::PhysicalEscapeEnergyContract::SuppliedPartialMolarEnthalpyTrace, false});
                    candidate.step.push_back(std::move(report));
                }
            require(position == global.size(), "History reduction topology mismatch.");
        });
    d_species_history = std::move(candidate);
}
} // namespace SimpleFluid
