/** @file ReactingGasSpecies.ipp
 * @brief Schema-indexed conservative field and population execution.
 */
#pragma once
#include <bit>
#include <utility>

namespace SimpleFluid
{
namespace reacting_species_detail
{
inline double narrow(long double value)
{
    const auto result = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(result) || (value != 0.0L && result == 0.0))
        throw std::invalid_argument("Species field conversion overflows or underflows its physical amount.");
    return result;
}
inline void require(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}
inline std::vector<double> composition(std::span<const double> amounts)
{
    detail::CompensatedSum<long double> total;
    for (const auto amount : amounts)
        total += amount;
    std::vector<double> result(amounts.size());
    if (total.value() > 0.0L)
        for (std::size_t i = 0; i < amounts.size(); ++i)
            result[i] = narrow(static_cast<long double>(amounts[i]) / total.value());
    return result;
}
inline double total(std::span<const double> amounts)
{
    detail::CompensatedSum<long double> sum;
    for (const auto value : amounts)
        sum += value;
    return narrow(sum.value());
}
inline double mass(const Reacting::SpeciesCatalog& catalog, std::span<const double> amounts)
{
    detail::CompensatedSum<long double> sum;
    for (std::size_t i = 0; i < amounts.size(); ++i)
        sum += static_cast<long double>(amounts[i]) * catalog.species()[i].molarMass;
    return narrow(sum.value());
}
inline std::uint64_t configuration_signature(const ReactingGasOptions& options)
{
    std::uint64_t signature = 14695981039346656037ULL;
    const auto integer = [&signature](std::uint64_t value)
    {
        for (unsigned int byte = 0; byte < 8; ++byte)
        {
            signature ^= (value >> (8 * byte)) & 0xff;
            signature *= 1099511628211ULL;
        }
    };
    const auto number = [&integer](double value) { integer(std::bit_cast<std::uint64_t>(value)); };
    const auto string = [&integer](std::string_view value)
    {
        integer(value.size());
        for (const unsigned char c : value)
            integer(c);
    };
    integer(options.reaction_mechanism->catalog().fingerprint());
    integer(static_cast<std::uint64_t>(options.reaction_mechanism->energy_convention()));
    for (const auto& reaction : options.reaction_mechanism->reactions())
    {
        integer(reaction.id);
        string(reaction.name);
        number(reaction.heatPerExtent);
        integer(reaction.changes.size());
        for (const auto& term : reaction.changes)
        {
            integer(term.species);
            integer(static_cast<std::uint64_t>(term.phase));
            number(term.coefficient);
            string(term.population);
        }
    }
    integer(static_cast<std::uint64_t>(options.pressure_mode));
    integer(static_cast<std::uint64_t>(options.dissolved_transport));
    integer(static_cast<std::uint64_t>(options.bulk_diffusion));
    integer(static_cast<std::uint64_t>(options.bubble_transport));
    integer(static_cast<std::uint64_t>(options.rise_velocity_mode));
    integer(static_cast<std::uint64_t>(options.mass_transfer_mode));
    integer(static_cast<std::uint64_t>(options.surface_tension_mode));
    integer(static_cast<bool>(options.species_source));
    integer(static_cast<bool>(options.population_process));
    integer(static_cast<bool>(options.initial_species_state));
    integer(static_cast<bool>(options.initial_population_numbers));
    integer(static_cast<bool>(options.surface_tension_correlation));
    for (const double value :
        {options.reference_pressure, options.minimum_absolute_pressure, options.alpha_min, options.alpha_max,
            options.min_radius, options.max_radius, options.surface_tension, options.constant_slip_velocity,
            options.bubble_gravity, options.rise_velocity_tolerance, options.transport_solver_tolerance,
            options.micro_dissolution_frequency, options.large_dissolution_frequency,
            options.micro_conversion_frequency, options.reaction_advance.maximumSubstep})
        number(value);
    integer(options.max_subcycles);
    integer(options.max_rise_velocity_iterations);
    integer(options.reaction_advance.maximumSubsteps);
    integer(options.reaction_advance.maximumRetries);
    integer(options.population_ids.size());
    for (const auto& id : options.population_ids)
        string(id);
    integer(options.free_surface_patches.size());
    for (const auto& id : options.free_surface_patches)
        string(id);
    integer(options.pressure_history_times.size());
    for (const auto value : options.pressure_history_times)
        number(value);
    integer(options.pressure_history_values.size());
    for (const auto value : options.pressure_history_values)
        number(value);
    integer(options.species_properties.size());
    for (const auto& property : options.species_properties)
    {
        integer(property.species);
        number(property.diffusivity);
        number(property.henry_coefficient);
        integer(static_cast<bool>(property.diffusivity_provider));
        integer(static_cast<bool>(property.equilibrium_concentration));
        integer(static_cast<bool>(property.carried_molar_enthalpy));
    }
    for (const auto& eos : {options.liquid_thermodynamics, options.gas_thermodynamics})
    {
        integer(eos->catalog().fingerprint());
        integer(static_cast<std::uint64_t>(eos->kind()));
        integer(static_cast<std::uint64_t>(eos->energy_convention()));
        const auto& bounds = eos->bounds();
        number(bounds.minimum_temperature);
        number(bounds.maximum_temperature);
        number(bounds.minimum_pressure);
        number(bounds.maximum_pressure);
    }
    return signature;
}
inline Reacting::CellSpeciesState zero_like(const Reacting::CellSpeciesState& state)
{
    auto result = state;
    std::fill(result.amounts.begin(), result.amounts.end(), 0.0);
    for (auto& population : result.dispersedPopulations)
        std::fill(population.amounts.begin(), population.amounts.end(), 0.0);
    return result;
}
} // namespace reacting_species_detail

template<TpetraTypePack Pack, class MeshType> void ReactingGasModel<Pack, MeshType>::configure_species_fields()
{
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<int>(species_enabled()), "Reacting typed schema presence");
    collective_detail::require_uniform_value(*d_mesh, static_cast<int>(d_options.mode), "Reacting model selection");
    if (species_enabled())
        collective_detail::require_uniform_value(*d_mesh,
            static_cast<unsigned long long>(reacting_species_detail::configuration_signature(d_options)),
            "Reacting species mechanism/property/source control-flow metadata");
    ++d_species_generation;
    d_species_fields.clear();
    d_additional_population_numbers.clear();
    d_population_volume_fractions.clear();
    d_population_slip_fields.clear();
    d_species_source_ledgers.clear();
    d_species_population_ledgers.clear();
    d_species_escape_ledgers.clear();
    d_species_history = {};
    d_species_thermal_source.put_scalar(0.0);
    d_species_material_source.put_scalar(0.0);
    d_species_liquid_mass.put_scalar(0.0);
    d_species_liquid_density.put_scalar(0.0);
    d_state_sync_owned = Teuchos::null;
    d_state_sync_overlap = Teuchos::null;
    d_state_sync_import = Teuchos::null;
    if (!species_enabled())
        return;
    const auto& catalog = d_options.reaction_mechanism->catalog();
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<unsigned long long>(catalog.fingerprint()), "Reacting ordered species schema");
    std::uint64_t populations = 14695981039346656037ULL;
    for (const auto& name : d_options.population_ids)
    {
        for (const unsigned char c : name)
        {
            populations ^= c;
            populations *= 1099511628211ULL;
        }
        populations ^= 0xff;
        populations *= 1099511628211ULL;
    }
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<unsigned long long>(populations), "Reacting ordered population schema");
    register_output_fields();
    for (const auto& species : catalog.species())
    {
        auto& bundle = d_species_fields.emplace_back();
        bundle.id = species.id;
        for (std::size_t phase = 0; phase < 4; ++phase)
        {
            const auto name = "species_" + std::to_string(species.id) + "_phase_" + std::to_string(phase);
            bundle.phases[phase] = std::make_unique<field_type>(d_mesh, 0.0, name);
            d_output_fields.emplace(name, bundle.phases[phase].get());
        }
        for (const auto& population : d_options.population_ids)
        {
            const auto name = "species_" + std::to_string(species.id) + "_population_" + population;
            bundle.populations.push_back(std::make_unique<field_type>(d_mesh, 0.0, name));
            d_output_fields.emplace(name, bundle.populations.back().get());
        }
        bundle.escape = std::make_unique<field_type>(d_mesh, 0.0, "species_escape_" + std::to_string(species.id));
    }
    for (const auto& population : d_options.population_ids)
        if (population != "micro" && population != "large")
            d_additional_population_numbers.push_back(
                std::make_unique<field_type>(d_mesh, 0.0, "population_number_" + population));
    for (const auto& population : d_options.population_ids)
    {
        d_population_volume_fractions.push_back(
            std::make_unique<field_type>(d_mesh, 0.0, "population_alpha_" + population));
        d_population_slip_fields.push_back(std::make_unique<field_type>(d_mesh, 0.0, "population_slip_" + population));
    }
    d_output_fields.emplace("species_thermal_source", &d_species_thermal_source);
    d_output_fields.emplace("species_material_volume_source", &d_species_material_source);
    d_initial_state_initialized = false;
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::mutable_species_field(
    Reacting::SpeciesId id, Reacting::SpeciesPhase phase, std::string_view population) -> field_type&
{
    using reacting_species_detail::require;
    require(species_enabled(), "Species fields require an explicit reaction mechanism.");
    const auto index = d_options.reaction_mechanism->catalog().index(id);
    require(static_cast<std::size_t>(phase) < 4, "Invalid species phase.");
    if (population.empty())
        return *d_species_fields.at(index).phases.at(static_cast<std::size_t>(phase));
    require(phase == Reacting::SpeciesPhase::Dispersed, "Named populations require Dispersed phase.");
    const auto found = std::ranges::find(d_options.population_ids, population);
    require(found != d_options.population_ids.end(), "Unknown species population.");
    return *d_species_fields.at(index).populations.at(
        static_cast<std::size_t>(found - d_options.population_ids.begin()));
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::species_field(
    Reacting::SpeciesId id, Reacting::SpeciesPhase phase, std::string_view population) const -> const field_type&
{
    return const_cast<ReactingGasModel*>(this)->mutable_species_field(id, phase, population);
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::mutable_population_number(std::string_view population) -> field_type&
{
    if (population == "micro")
        return d_micro_number;
    if (population == "large")
        return d_large_number;
    std::size_t extra = 0;
    for (const auto& id : d_options.population_ids)
        if (id != "micro" && id != "large")
        {
            if (id == population)
                return *d_additional_population_numbers.at(extra);
            ++extra;
        }
    throw std::invalid_argument("Unknown population number field.");
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::population_number_field(std::string_view population) const -> const field_type&
{
    reacting_species_detail::require(
        std::ranges::find(d_options.population_ids, population) != d_options.population_ids.end(),
        "Unknown population number field.");
    return const_cast<ReactingGasModel*>(this)->mutable_population_number(population);
}

template<TpetraTypePack Pack, class MeshType>
Reacting::CellSpeciesState ReactingGasModel<Pack, MeshType>::cell_species_state(
    local_ordinal_type cell, double volume) const
{
    using reacting_species_detail::narrow;
    using reacting_species_detail::require;
    require(species_enabled() && cell >= 0 && static_cast<std::size_t>(cell) < d_mesh->num_owned_cells() &&
                std::isfinite(volume) && volume > 0.0,
        "Species state conversion requires an owned cell and positive bulk volume.");
    const auto& catalog = d_options.reaction_mechanism->catalog();
    Reacting::CellSpeciesState result(catalog);
    for (const auto& id : d_options.population_ids)
        result.dispersedPopulations.push_back({id, std::vector<double>(catalog.species().size())});
    for (const auto& species : catalog.species())
    {
        for (std::size_t phase = 0; phase < 4; ++phase)
            result.set_amount(catalog, species.id, static_cast<Reacting::SpeciesPhase>(phase),
                narrow(static_cast<long double>(
                           species_field(species.id, static_cast<Reacting::SpeciesPhase>(phase)).value(cell)) *
                       volume));
        for (const auto& id : d_options.population_ids)
            result.set_amount(catalog, species.id, Reacting::SpeciesPhase::Dispersed,
                narrow(static_cast<long double>(
                           species_field(species.id, Reacting::SpeciesPhase::Dispersed, id).value(cell)) *
                       volume),
                id);
    }
    result.validate(catalog);
    return result;
}

template<TpetraTypePack Pack, class MeshType>
Reacting::CellSpeciesState ReactingGasModel<Pack, MeshType>::cell_species_state(local_ordinal_type cell) const
{
    return cell_species_state(cell, d_mesh->cell_volume(cell));
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::publish_species_states(
    std::span<const Reacting::CellSpeciesState> states, std::span<const double> volumes)
{
    const auto& catalog = d_options.reaction_mechanism->catalog();
    for (std::size_t owned = 0; owned < states.size(); ++owned)
        for (const auto& species : catalog.species())
        {
            for (std::size_t phase = 0; phase < 4; ++phase)
                mutable_species_field(species.id, static_cast<Reacting::SpeciesPhase>(phase))
                    .owned_write_view()(owned, 0) = reacting_species_detail::narrow(
                    static_cast<long double>(
                        states[owned].amount(catalog, species.id, static_cast<Reacting::SpeciesPhase>(phase))) /
                    volumes[owned]);
            for (const auto& id : d_options.population_ids)
                mutable_species_field(species.id, Reacting::SpeciesPhase::Dispersed, id).owned_write_view()(owned, 0) =
                    reacting_species_detail::narrow(static_cast<long double>(states[owned].amount(
                                                        catalog, species.id, Reacting::SpeciesPhase::Dispersed, id)) /
                                                    volumes[owned]);
        }
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::initialize_species(
    std::span<const Reacting::CellSpeciesState> states, std::span<const std::vector<double>> numbers)
{
    using reacting_species_detail::require;
    StateSnapshot prior;
    collective_detail::collective_local_validation(*d_mesh, "Reacting species initialization snapshot", [&]
    {
        prior = snapshot();
    });
    try
    {
        std::vector<double> volumes(d_mesh->num_owned_cells());
        collective_detail::collective_local_validation(*d_mesh, "Reacting species initialization",
            [&]
            {
                require(species_enabled() && states.size() == d_mesh->num_owned_cells() &&
                            (numbers.empty() || numbers.size() == states.size()),
                    "Species initialization must match owned cells.");
                const auto& catalog = d_options.reaction_mechanism->catalog();
                for (std::size_t owned = 0; owned < states.size(); ++owned)
                {
                    states[owned].validate(catalog);
                    volumes[owned] = d_mesh->cell_volume(static_cast<local_ordinal_type>(owned));
                    require(std::isfinite(volumes[owned]) && volumes[owned] > 0.0 &&
                                states[owned].dispersedPopulations.size() == d_options.population_ids.size(),
                        "Species initialization has invalid volume or population topology.");
                    require(numbers.empty() || numbers[owned].size() == d_options.population_ids.size(),
                        "Population number initialization must match the population schema.");
                    for (std::size_t p = 0; p < d_options.population_ids.size(); ++p)
                    {
                        const auto& population = states[owned].dispersedPopulations[p];
                        const double count = numbers.empty() ? 0.0 : numbers[owned][p];
                        const auto amount = reacting_species_detail::total(population.amounts);
                        require(population.id == d_options.population_ids[p] && std::isfinite(count) && count >= 0.0 &&
                                    ((amount == 0.0 && count == 0.0) || (amount > 0.0 && count > 0.0)),
                            "Initialized bubble number and species inventory must be paired.");
                    }
                }
            });
        collective_detail::collective_local_validation(
            *d_mesh, "Reacting species field publication", [&] { publish_species_states(states, volumes); });
        for (std::size_t owned = 0; owned < states.size(); ++owned)
            for (std::size_t p = 0; p < d_options.population_ids.size(); ++p)
                mutable_population_number(d_options.population_ids[p]).owned_write_view()(owned, 0) =
                    numbers.empty() ? 0.0 : numbers[owned][p] / volumes[owned];
        d_species_source_ledgers.clear();
        d_species_population_ledgers.clear();
        d_species_escape_ledgers.clear();
        d_initial_state_initialized = true;
        d_history_initialized = false;
        publish_species_history(states, {}, true);
        sync_all_fields();
        collective_detail::collective_local_validation(*d_mesh, "Reacting initialized receipt generation",
            [&]
            {
                require(d_species_generation != std::numeric_limits<std::uint64_t>::max(),
                    "Reacting initialization generation exhausted.");
            });
        ++d_species_generation;
        d_species_event_receipt = {
            false, 0, d_species_generation, d_options.reaction_mechanism->catalog().fingerprint(), 0, 0};
    }
    catch (...)
    {
        restore(prior);
        throw;
    }
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::append_species_fields(std::vector<field_type*>& fields)
{
    fields.push_back(&d_species_thermal_source);
    fields.push_back(&d_species_material_source);
    fields.push_back(&d_species_liquid_mass);
    fields.push_back(&d_species_liquid_density);
    for (auto& species : d_species_fields)
    {
        for (auto& phase : species.phases)
            fields.push_back(phase.get());
        for (auto& population : species.populations)
            fields.push_back(population.get());
        fields.push_back(species.escape.get());
    }
    for (auto& number : d_additional_population_numbers)
        fields.push_back(number.get());
    for (auto& alpha : d_population_volume_fractions)
        fields.push_back(alpha.get());
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::append_species_fields(std::vector<const field_type*>& fields) const
{
    fields.push_back(&d_species_thermal_source);
    fields.push_back(&d_species_material_source);
    fields.push_back(&d_species_liquid_mass);
    fields.push_back(&d_species_liquid_density);
    for (const auto& species : d_species_fields)
    {
        for (const auto& phase : species.phases)
            fields.push_back(phase.get());
        for (const auto& population : species.populations)
            fields.push_back(population.get());
        fields.push_back(species.escape.get());
    }
    for (const auto& number : d_additional_population_numbers)
        fields.push_back(number.get());
    for (const auto& alpha : d_population_volume_fractions)
        fields.push_back(alpha.get());
}

template<TpetraTypePack Pack, class MeshType>
std::vector<BubbleGasPhysics::PopulationNumberState> ReactingGasModel<Pack, MeshType>::cell_population_numbers(
    local_ordinal_type cell, double volume) const
{
    std::vector<BubbleGasPhysics::PopulationNumberState> result;
    for (const auto& population : d_options.population_ids)
        result.push_back(
            {population, reacting_species_detail::narrow(
                             static_cast<long double>(population_number_field(population).value(cell)) * volume)});
    return result;
}

template<TpetraTypePack Pack, class MeshType>
Reacting::ReactionEnvironment ReactingGasModel<Pack, MeshType>::species_environment(
    local_ordinal_type cell, double time, double temperature, double volume) const
{
    // This field adapter reports physical EOS solution volume on a prescribed carrier mesh.
    // Local material occupancy is closed only by the separate native low-Mach flow owner.
    using reacting_species_detail::require;
    require(cell >= 0 && static_cast<std::size_t>(cell) < d_mesh->num_owned_cells(),
        "A source location requires an actual owned cell.");
    Reacting::ReactionEnvironment result{
        time, temperature, d_absolute_pressure.value(cell), d_alpha_l.value(cell) * volume, volume,
        Reacting::ReactionVolumeClosure::ReferenceControlVolume};
    const auto geometryCell = [&]
    {
        if constexpr (requires { d_mesh->cell_geometry_global_id(cell); })
            return d_mesh->cell_geometry_global_id(cell);
        else
            return d_mesh->cell_global_id(cell); // Legacy Mesh retains its original physical global identities.
    }();
    require(std::in_range<std::uint64_t>(geometryCell), "Canonical source cell identity is not representable.");
    Reacting::ReactionLocation location;
    location.geometryCellId = static_cast<std::uint64_t>(geometryCell);
    if constexpr (requires { d_mesh->geometry_epoch(); })
        location.geometryEpoch = d_mesh->geometry_epoch();
    const auto center = d_mesh->cell_centroid(cell);
    location.centroid = {center.x, center.y, center.z};
    location.ownedCell = static_cast<std::size_t>(cell);
    result.location = location;
    return result;
}

template<TpetraTypePack Pack, class MeshType>
std::vector<BubbleGasPhysics::PopulationCondition> ReactingGasModel<Pack, MeshType>::population_conditions(
    const Reacting::CellSpeciesState& state, const std::vector<BubbleGasPhysics::PopulationNumberState>& numbers,
    const Reacting::ReactionEnvironment& environment) const
{
    using reacting_species_detail::require;
    std::vector<BubbleGasPhysics::PopulationCondition> result;
    const auto& catalog = d_options.reaction_mechanism->catalog();
    const double sigma = d_options.surface_tension_mode == SurfaceTensionMode::External
                             ? d_options.surface_tension_correlation(environment.temperature)
                             : d_options.surface_tension;
    require(std::isfinite(sigma) && sigma >= 0.0, "Supplied surface tension must be finite and nonnegative.");
    for (std::size_t p = 0; p < numbers.size(); ++p)
    {
        const auto& amounts = state.dispersedPopulations.at(p).amounts;
        BubbleGasPhysics::PopulationCondition condition{numbers[p].population, numbers[p].number};
        if (reacting_species_detail::total(amounts) > 0.0)
        {
            if (condition.number > 0.0)
            {
                const auto bubble = Reacting::solve_bubble_radius(*d_options.gas_thermodynamics, amounts,
                    environment.temperature, environment.absolutePressure, sigma, condition.number,
                    d_options.min_radius, d_options.max_radius);
                condition.radius = bubble.radius;
                condition.properties = bubble.properties;
            }
            else
                condition.properties = d_options.gas_thermodynamics->evaluate(environment.temperature,
                    environment.absolutePressure, reacting_species_detail::composition(amounts));
        }
        result.push_back(std::move(condition));
    }
    return result;
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::reconstruct_species_derived(
    const field_type& temperature, std::span<const real_t> volumes)
{
    using reacting_species_detail::require;
    collective_detail::collective_local_validation(*d_mesh, "Reacting full mixture reconstruction",
        [&]
        {
            const auto& catalog = d_options.reaction_mechanism->catalog();
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const double volume = volumes.empty() ? d_mesh->cell_volume(cell) : volumes[owned];
                const auto state = cell_species_state(cell, volume);
                const auto numbers = cell_population_numbers(cell, volume);
                const auto environment = species_environment(cell, 0.0, temperature.value(cell), volume);
                const auto populations = population_conditions(state, numbers, environment);
                std::vector<double> liquid(catalog.species().size()), dissolved(catalog.species().size()),
                    resolved(catalog.species().size());
                for (std::size_t i = 0; i < catalog.species().size(); ++i)
                {
                    const auto id = catalog.species()[i].id;
                    dissolved[i] = state.amount(catalog, id, Reacting::SpeciesPhase::Dissolved);
                    liquid[i] = state.amount(catalog, id, Reacting::SpeciesPhase::Liquid) + dissolved[i];
                    resolved[i] = state.amount(catalog, id, Reacting::SpeciesPhase::Resolved);
                    require(state.amount(catalog, id, Reacting::SpeciesPhase::Dispersed) == 0.0,
                        "Bubble fields require named population ownership for every dispersed species.");
                }
                double liquidVolume = 0.0, resolvedVolume = 0.0, gasVolume = 0.0, microMoles = 0.0, largeMoles = 0.0;
                if (reacting_species_detail::total(liquid) > 0.0)
                    liquidVolume = reacting_species_detail::mass(catalog, liquid) /
                                   d_options.liquid_thermodynamics
                                       ->evaluate(environment.temperature, environment.absolutePressure,
                                           reacting_species_detail::composition(liquid))
                                       .density;
                if (reacting_species_detail::total(resolved) > 0.0)
                    resolvedVolume = reacting_species_detail::mass(catalog, resolved) /
                                     d_options.gas_thermodynamics
                                         ->evaluate(environment.temperature, environment.absolutePressure,
                                             reacting_species_detail::composition(resolved))
                                         .density;
                d_alpha_g_micro.owned_write_view()(owned, 0) = 0.0;
                d_alpha_g_large.owned_write_view()(owned, 0) = 0.0;
                for (std::size_t p = 0; p < populations.size(); ++p)
                {
                    const auto amount = reacting_species_detail::total(state.dispersedPopulations[p].amounts);
                    require((amount == 0.0 && populations[p].number == 0.0) ||
                                (amount > 0.0 && populations[p].number > 0.0),
                        "Bubble species amounts and number are inconsistent.");
                    const auto occupiedVolume =
                        amount > 0.0 ? reacting_species_detail::mass(catalog, state.dispersedPopulations[p].amounts) /
                                           populations[p].properties.density
                                     : 0.0;
                    d_population_volume_fractions[p]->owned_write_view()(owned, 0) = occupiedVolume / volume;
                    gasVolume += occupiedVolume;
                    if (populations[p].id == "micro")
                    {
                        microMoles = amount;
                        d_micro_radius.owned_write_view()(owned, 0) = populations[p].radius;
                        d_alpha_g_micro.owned_write_view()(owned, 0) = occupiedVolume / volume;
                    }
                    if (populations[p].id == "large")
                    {
                        largeMoles = amount;
                        d_large_radius.owned_write_view()(owned, 0) = populations[p].radius;
                        d_alpha_g_large.owned_write_view()(owned, 0) = occupiedVolume / volume;
                    }
                }
                const double gasFraction = (gasVolume + resolvedVolume) / volume;
                require(std::isfinite(gasFraction) && gasFraction >= 0.0 && gasFraction <= d_options.alpha_max,
                    "Mixture gas occupancy exceeds the declared phase capacity; reduce/recompute the physical trial.");
                require(std::isfinite(liquidVolume) && liquidVolume >= 0.0, "Mixture liquid volume is invalid.");
                const double liquidMass = reacting_species_detail::mass(catalog, liquid);
                d_species_liquid_mass.owned_write_view()(owned, 0) = liquidMass / volume;
                d_species_liquid_density.owned_write_view()(owned, 0) =
                    liquidVolume > 0.0 ? liquidMass / liquidVolume : 0.0;
                d_alpha_l.owned_write_view()(owned, 0) = liquidVolume / volume;
                d_alpha_g.owned_write_view()(owned, 0) = gasFraction;
                d_alpha_g_raw.owned_write_view()(owned, 0) = gasVolume / volume;
                d_alpha_g_excess.owned_write_view()(owned, 0) = 0.0;
                d_dissolved_hydrogen_inventory.owned_write_view()(owned, 0) =
                    reacting_species_detail::total(dissolved) / volume;
                d_dissolved_hydrogen.owned_write_view()(owned, 0) =
                    liquidVolume > 0.0 ? reacting_species_detail::total(dissolved) / liquidVolume : 0.0;
                d_micro_moles.owned_write_view()(owned, 0) = microMoles / volume;
                d_large_moles.owned_write_view()(owned, 0) = largeMoles / volume;
                d_previous_temperature.owned_write_view()(owned, 0) = environment.temperature;
                d_previous_alpha_g.owned_write_view()(owned, 0) = gasFraction;
            }
        });
    d_alpha_l.sync_ghosts();
    d_alpha_g.sync_ghosts();
    d_micro_radius.sync_ghosts();
    d_large_radius.sync_ghosts();
    for (auto& alpha : d_population_volume_fractions)
        alpha->sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_species_moles(Reacting::SpeciesId id) const -> scalar_type
{
    scalar_type total = 0;
    for (std::size_t p = 0; p < 4; ++p)
        total += global_integral(species_field(id, static_cast<Reacting::SpeciesPhase>(p)));
    for (const auto& population : d_options.population_ids)
        total += global_integral(species_field(id, Reacting::SpeciesPhase::Dispersed, population));
    return total;
}

template<TpetraTypePack Pack, class MeshType>
BubbleGasPhysics::MultispeciesPopulationPacket ReactingGasModel<Pack, MeshType>::population_packet(
    const Reacting::ReactionContext& chemistry, const std::vector<BubbleGasPhysics::PopulationCondition>& populations,
    double duration) const
{
    using P = Reacting::SpeciesPhase;
    using reacting_species_detail::require;
    if (d_options.population_process)
        return d_options.population_process({chemistry, populations}, duration);
    BubbleGasPhysics::MultispeciesPopulationPacket packet;
    for (std::size_t p = 0; p < populations.size(); ++p)
    {
        const auto& population = populations[p];
        const auto& amounts = chemistry.state.dispersedPopulations[p].amounts;
        const double frequency = population.id == "micro"   ? d_options.micro_dissolution_frequency
                                 : population.id == "large" ? d_options.large_dissolution_frequency
                                                            : 0.0;
        const double fraction = -std::expm1(-frequency * duration);
        const double conversion =
            population.id == "micro"
                ? std::exp(-frequency * duration) * -std::expm1(-d_options.micro_conversion_frequency * duration)
                : 0.0;
        if (fraction > 0.0 || conversion > 0.0)
        {
            double convertedNumber = population.number * conversion;
            packet.numberEvents.push_back({population.id, 0.0, population.number * fraction + convertedNumber});
            if (conversion > 0.0)
                packet.numberEvents.push_back({"large", convertedNumber, 0.0});
            for (std::size_t s = 0; s < amounts.size(); ++s)
            {
                if (amounts[s] * fraction > 0.0)
                    packet.transfers.push_back({chemistry.catalog.species()[s].id, P::Dispersed, P::Dissolved,
                        population.id, {}, amounts[s] * fraction});
                if (amounts[s] * conversion > 0.0)
                    packet.transfers.push_back({chemistry.catalog.species()[s].id, P::Dispersed, P::Dispersed,
                        population.id, "large", amounts[s] * conversion});
            }
        }
        if (population.number == 0.0 || population.radius == 0.0 || chemistry.environment.liquidVolume == 0.0)
            continue;
        std::vector<double> liquid(chemistry.catalog.species().size());
        for (std::size_t s = 0; s < liquid.size(); ++s)
            liquid[s] = chemistry.state.amount(chemistry.catalog, chemistry.catalog.species()[s].id, P::Liquid) +
                        chemistry.state.amount(chemistry.catalog, chemistry.catalog.species()[s].id, P::Dissolved);
        require(reacting_species_detail::total(liquid) > 0.0,
            "Interfacial transfer requires a populated physical liquid phase.");
        const auto liquidPoint = d_options.liquid_thermodynamics->evaluate(chemistry.environment.temperature,
            chemistry.environment.absolutePressure, reacting_species_detail::composition(liquid));
        for (const auto& property : d_options.species_properties)
        {
            if (property.henry_coefficient == 0.0 && !property.equilibrium_concentration)
                continue;
            const double diffusivity =
                property.diffusivity_provider ? property.diffusivity_provider(chemistry) : property.diffusivity;
            require(std::isfinite(diffusivity) && diffusivity >= 0.0, "Species transfer diffusivity is invalid.");
            if (diffusivity == 0.0)
                continue;
            const auto s = chemistry.catalog.index(property.species);
            require(property.equilibrium_concentration || population.properties.fugacities.size() == liquid.size(),
                "Henry-law transfer requires full mixture fugacities or a supplied equilibrium closure.");
            const double equilibrium = property.equilibrium_concentration
                                           ? property.equilibrium_concentration(chemistry, population.properties)
                                           : property.henry_coefficient * population.properties.fugacities[s];
            require(std::isfinite(equilibrium) && equilibrium >= 0.0, "Supplied equilibrium concentration is invalid.");
            const auto law = BubbleGasPhysics::bubble_mass_transfer(d_options.mass_transfer_mode, diffusivity,
                population.radius, liquidPoint.density, liquidPoint.dynamic_viscosity,
                d_options.rise_velocity_mode == BubbleRiseVelocityMode::ConstantSlip ? d_options.constant_slip_velocity
                                                                                     : 0.0);
            const double concentration = chemistry.state.amount(chemistry.catalog, property.species, P::Dissolved) /
                                         chemistry.environment.liquidVolume;
            const double amount = duration * law.coefficient * 4 * std::numbers::pi * population.radius *
                                  population.radius * population.number * (concentration - equilibrium);
            require(std::isfinite(amount), "Species interface transfer amount overflowed.");
            if (amount > 0.0)
                packet.transfers.push_back({property.species, P::Dissolved, P::Dispersed, {}, population.id, amount});
            else if (amount < 0.0)
                packet.transfers.push_back({property.species, P::Dispersed, P::Dissolved, population.id, {}, -amount});
        }
    }
    return packet;
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::advance_species_fields(scalar_type time, scalar_type dt,
    const field_type& temperature, const field_type& gauge_pressure, const velocity_field_type& velocity,
    const face_flux_field_type& liquid_flux, const material_type& material, const FVM::ALEControlVolumeState* ale,
    Dimension axis)
{
    using reacting_species_detail::narrow;
    using reacting_species_detail::require;
    using P = Reacting::SpeciesPhase;
    const auto backup = snapshot();
    try
    {
        using Execution = decltype(acquire_mesh_execution(*d_mesh));
        std::unique_ptr<Execution> execution;
        collective_detail::collective_local_validation(*d_mesh, "Reacting species geometry execution",
            [&] { execution.reset(new Execution(acquire_mesh_execution(*d_mesh))); });
        collective_detail::collective_local_validation(*d_mesh, "Reacting species advance inputs",
            [&]
            {
                require(d_initial_state_initialized && std::isfinite(time) && std::isfinite(dt) && dt > 0.0,
                    "General reacting fields need initialized species and a finite positive interval.");
                require(&temperature.mesh() == d_mesh.get() && &gauge_pressure.mesh() == d_mesh.get() &&
                            &velocity.mesh() == d_mesh.get() && &liquid_flux.mesh() == d_mesh.get(),
                    "Reacting input field mesh mismatch.");
                require(d_options.bubble_transport == BubbleTransportMode::General,
                    "General multispecies transport requires the owner-oriented general carrier/slip flux.");
                if (ale)
                    ale->validate(*d_mesh, dt);
            });
        d_last_statistics = {};
        d_species_thermal_source.put_scalar(0.0);
        d_species_material_source.put_scalar(0.0);
        d_species_liquid_mass.put_scalar(0.0);
        d_species_liquid_density.put_scalar(0.0);
        reconstruct_absolute_pressure(time, gauge_pressure);
        collective_detail::collective_local_validation(*d_mesh, "Reacting absolute mixture pressure",
            [&]
            {
                require(d_last_statistics.pressure_floor_cells == 0,
                    "General mixture pressure cannot be clipped to a floor; recompute the physical trial.");
                for (std::size_t i = 0; i < d_mesh->num_owned_cells(); ++i)
                    require(std::isfinite(d_absolute_pressure.value(static_cast<local_ordinal_type>(i))) &&
                                d_absolute_pressure.value(static_cast<local_ordinal_type>(i)) >=
                                    d_options.minimum_absolute_pressure,
                        "General mixture absolute pressure is invalid.");
            });
        const auto oldVolumes = ale ? ale->old_cell_volumes() : std::span<const real_t>{};
        reconstruct_species_derived(temperature, oldVolumes);
        const auto& catalog = d_options.reaction_mechanism->catalog();
        std::vector<double> volumes(d_mesh->num_owned_cells());
        std::vector<Reacting::CellSpeciesState> before;
        for (std::size_t owned = 0; owned < volumes.size(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            volumes[owned] = ale ? ale->new_cell_volumes()[owned] : d_mesh->cell_volume(cell);
            before.push_back(cell_species_state(cell, ale ? oldVolumes[owned] : volumes[owned]));
        }
        if (!d_transport_workspace)
            d_transport_workspace = std::make_unique<TransportWorkspace>(d_mesh);
        auto& workspace = *d_transport_workspace;
        workspace.operator_ready.fill(false);
        d_species_escape_ledgers.resize(volumes.size());
        for (std::size_t owned = 0; owned < volumes.size(); ++owned)
        {
            d_species_escape_ledgers[owned] = {};
            d_species_escape_ledgers[owned].externalChange = reacting_species_detail::zero_like(before[owned]);
            d_species_escape_ledgers[owned].chemicalChange = reacting_species_detail::zero_like(before[owned]);
        }
        std::vector<detail::CompensatedSum<long double>> escapedEnergy(volumes.size()), escapedMass(volumes.size());
        const auto property_for = [&](Reacting::SpeciesId id) -> const ReactingSpeciesProperties*
        {
            for (const auto& property : d_options.species_properties)
                if (property.species == id)
                    return &property;
            return nullptr;
        };
        const auto record_escape =
            [&](Reacting::SpeciesId id, P phase, std::string_view population, const field_type& rates)
        {
            collective_detail::collective_local_validation(*d_mesh, "Reacting species boundary budgets",
                [&]
                {
                    const auto* property = property_for(id);
                    for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                    {
                        const auto cell = static_cast<local_ordinal_type>(owned);
                        const double amount = rates.value(cell) * volumes[owned] * dt;
                        if (amount == 0.0)
                            continue;
                        require(property && property->carried_molar_enthalpy,
                            "An escaping chemical species requires its supplied carried molar enthalpy closure.");
                        require(std::isfinite(amount) && amount > 0.0, "Escaping species amount is invalid.");
                        auto& ledger = d_species_escape_ledgers[owned];
                        const double mass = amount * catalog.species()[catalog.index(id)].molarMass;
                        Reacting::ExternalSpeciesTransfer event{"physical boundary", id, phase, -amount, -mass, 0.0,
                            {-mass * velocity.owned_read_view()(owned, 0), -mass * velocity.owned_read_view()(owned, 1),
                                -mass * velocity.owned_read_view()(owned, 2)},
                            std::string(population)};
                        ledger.externalTransfers.push_back(event);
                        escapedMass[owned] += event.mass;
                        // A given species/compartment is transported once, so no endpoint subtraction is needed.
                        if (population.empty())
                            ledger.externalChange.amounts[static_cast<std::size_t>(phase) * catalog.species().size() +
                                                          catalog.index(id)] = -amount;
                        else
                        {
                            const auto found = std::ranges::find(d_options.population_ids, population);
                            ledger.externalChange.dispersedPopulations[found - d_options.population_ids.begin()]
                                .amounts[catalog.index(id)] = -amount;
                        }
                    }
                });
        };
        std::vector<std::vector<detail::CompensatedSum<long double>>> liquid_molar_packets;
        std::unique_ptr<face_flux_field_type> liquid_packet_scratch;
        collective_detail::collective_local_validation(*d_mesh, "Liquid canonical packet workspace",
            [&]
            {
                liquid_molar_packets.resize(catalog.species().size());
                for (auto& species : liquid_molar_packets)
                    species.resize(liquid_flux.owned_face_ids().size());
                liquid_packet_scratch = std::make_unique<face_flux_field_type>(d_mesh, "liquid_species_molar_packet");
            });
        face_flux_field_type* active_liquid_packet = nullptr;
        const auto transport_one = [&](field_type& field, field_type& escape, const face_flux_field_type& flux,
                                       field_type* slip, bool liquidWeighted, bool diffuse, std::size_t slot,
                                       bool reuse)
        {
            escape.put_scalar(0.0);
            if (active_liquid_packet)
                active_liquid_packet->put_scalar(0.0);
            if (global_integral(field, oldVolumes) == scalar_type{})
            {
                field.put_scalar(0.0);
                return;
            }
            if (!reuse)
            {
                workspace.systems.at(slot) = {};
                workspace.symbolic_plans.at(slot) = {};
                workspace.operator_ready.at(slot) = false;
            }
            transport_scalar(field, dt, flux, slip, 0.0, diffuse, liquidWeighted, escape, ale, axis, slot, reuse,
                nullptr, nullptr, diffuse ? &d_species_diffusivity : nullptr, active_liquid_packet);
        };
        for (auto& species : d_species_fields)
        {
            const auto* property = property_for(species.id);
            collective_detail::collective_local_validation(*d_mesh, "Reacting species diffusivity",
                [&]
                {
                    for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                    {
                        const auto cell = static_cast<local_ordinal_type>(owned);
                        const auto state = before[owned];
                        const auto environment = species_environment(
                            cell, time - dt, temperature.value(cell), ale ? oldVolumes[owned] : volumes[owned]);
                        const Reacting::ReactionContext context{catalog, state, environment};
                        const double value =
                            property ? (property->diffusivity_provider ? property->diffusivity_provider(context)
                                                                       : property->diffusivity)
                                     : 0.0;
                        require(std::isfinite(value) && value >= 0.0, "Supplied species diffusivity is invalid.");
                        d_species_diffusivity.owned_write_view()(owned, 0) = value;
                    }
                });
            active_liquid_packet = liquid_packet_scratch.get();
            transport_one(*species.phases[0], *species.escape, liquid_flux, nullptr, true, false, 0, false);
            record_escape(species.id, P::Liquid, {}, *species.escape);
            const auto species_index = catalog.index(species.id);
            for (const auto face : liquid_packet_scratch->owned_face_ids())
                liquid_molar_packets[species_index][liquid_packet_scratch->owned_row(face)] +=
                    liquid_packet_scratch->value(face);
            const auto& dissolvedFlux =
                d_options.dissolved_transport == ReactingTransportMode::Advective ? liquid_flux : workspace.zero_flux;
            transport_one(*species.phases[1], *species.escape, dissolvedFlux, nullptr, true, false, 0,
                false); // Film D never silently enables an unclosed bulk mass/energy diffusion model.
            record_escape(species.id, P::Dissolved, {}, *species.escape);
            for (const auto face : liquid_packet_scratch->owned_face_ids())
                liquid_molar_packets[species_index][liquid_packet_scratch->owned_row(face)] +=
                    liquid_packet_scratch->value(face);
            active_liquid_packet = nullptr;
            // This dispersed-bubble backend declares a stationary resolved compartment.
            // Its zero carrier still needs conservative accepted-old/trial-new ALE storage.
            // Advecting resolved interfaces are owned by the VoF phase-flux backend.
            transport_one(*species.phases[3], *species.escape, workspace.zero_flux, nullptr, false, false, 3, false);
        }
        for (std::size_t p = 0; p < d_options.population_ids.size(); ++p)
        {
            const auto& id = d_options.population_ids[p];
            auto& slip = *d_population_slip_fields[p];
            collective_detail::collective_local_validation(*d_mesh, "Reacting mixture population slip",
                [&]
                {
                    for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                    {
                        const auto cell = static_cast<local_ordinal_type>(owned);
                        const auto environment = species_environment(
                            cell, time - dt, temperature.value(cell), ale ? oldVolumes[owned] : volumes[owned]);
                        const auto conditions = population_conditions(before[owned],
                            cell_population_numbers(cell, ale ? oldVolumes[owned] : volumes[owned]), environment);
                        const auto& condition = conditions[p];
                        double speed = 0.0;
                        if (condition.number > 0.0 &&
                            d_options.rise_velocity_mode == BubbleRiseVelocityMode::ConstantSlip)
                            speed = d_options.constant_slip_velocity;
                        if (condition.number > 0.0 &&
                            d_options.rise_velocity_mode == BubbleRiseVelocityMode::Celata2007)
                        {
                            const double sigma = d_options.surface_tension_mode == SurfaceTensionMode::External
                                                     ? d_options.surface_tension_correlation(environment.temperature)
                                                     : d_options.surface_tension;
                            const auto result = BubbleGasPhysics::celata2007_bubble_rise_velocity(condition.radius,
                                material.density.value(cell), condition.properties.density,
                                material.dynamic_viscosity.value(cell), sigma, d_options.bubble_gravity,
                                d_options.max_rise_velocity_iterations, d_options.rise_velocity_tolerance);
                            require(result.converged, "Mixture bubble terminal velocity did not converge.");
                            speed = result.velocity;
                        }
                        slip.owned_write_view()(owned, 0) = speed;
                    }
                });
            slip.sync_ghosts();
            workspace.operator_ready[1] = false;
            transport_one(
                mutable_population_number(id), d_escape_number_rate, liquid_flux, &slip, false, false, 1, false);
            for (auto& species : d_species_fields)
            {
                transport_one(*species.populations[p], *species.escape, liquid_flux, &slip, false, false, 1, true);
                record_escape(species.id, P::Dispersed, id, *species.escape);
            }
        }
        reconstruct_species_derived(temperature);
        collective_detail::collective_local_validation(*d_mesh, "Reacting transported mixture boundary energy",
            [&]
            {
                for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                {
                    const auto cell = static_cast<local_ordinal_type>(owned);
                    const auto state = cell_species_state(cell, volumes[owned]);
                    const auto environment =
                        species_environment(cell, time - dt, temperature.value(cell), volumes[owned]);
                    const Reacting::ReactionContext context{catalog, state, environment};
                    std::array<detail::CompensatedSum<long double>, 3> momentum;
                    for (auto& event : d_species_escape_ledgers[owned].externalTransfers)
                    {
                        const auto* property = property_for(event.species);
                        const double enthalpy =
                            property->carried_molar_enthalpy(context, event.phase, event.population);
                        require(std::isfinite(enthalpy), "Supplied boundary species enthalpy is invalid.");
                        event.energy = narrow(static_cast<long double>(event.moles) * enthalpy);
                        escapedEnergy[owned] += event.energy;
                        for (std::size_t component = 0; component < 3; ++component)
                            momentum[component] += event.momentum[component];
                    }
                    for (std::size_t component = 0; component < 3; ++component)
                        d_species_escape_ledgers[owned].externalMomentum[component] =
                            narrow(momentum[component].value());
                }
            });
        for (std::size_t owned = 0; owned < volumes.size(); ++owned)
            d_transport_alpha_g_raw.owned_write_view()(owned, 0) =
                d_alpha_g_raw.value(static_cast<local_ordinal_type>(owned));
        // Own the thermal trace used by canonical face EOS work. Do not require
        // writable/synchronized ghost storage from a borrowed temperature input.
        d_previous_temperature.sync_ghosts();
        d_transport_liquid_material_volume_flux.put_scalar(0.0);
        int local_incomplete_carrier = 0, incomplete_carrier = 0;
        collective_detail::collective_local_validation(*d_mesh, "Reacting canonical liquid material-volume flux",
            [&]
            {
                for (const auto face : d_transport_liquid_material_volume_flux.owned_face_ids())
                {
                    const auto owner = d_mesh->owner_cell(face);
                    const bool boundary = d_mesh->is_boundary_face(face);
                    const auto neighbor = boundary ? owner : d_mesh->opposite_or_periodic_neighbor_cell(face, owner);
                    const bool open = boundary && std::ranges::find(d_options.free_surface_patches,
                                                      d_mesh->boundary_batch_name(d_mesh->boundary_id(face))) !=
                                                      d_options.free_surface_patches.end();
                    double carrier = liquid_flux.value(face);
                    if (boundary)
                        carrier = open ? std::max(carrier, 0.0) : 0.0;
                    require(std::isfinite(carrier), "Liquid material carrier is nonfinite.");
                    if (carrier == 0)
                        continue;
                    const auto donor = carrier >= 0 ? owner : neighbor;
                    std::vector<double> concentration(catalog.species().size());
                    detail::CompensatedSum<long double> amount, dissolved, mass_rate;
                    const auto face_row = d_transport_liquid_material_volume_flux.owned_row(face);
                    for (std::size_t i = 0; i < concentration.size(); ++i)
                    {
                        const auto id = catalog.species()[i].id;
                        const auto signed_rate = liquid_molar_packets[i][face_row].value();
                        require(std::isfinite(signed_rate) && (signed_rate == 0 || (signed_rate > 0) == (carrier > 0)),
                            "Canonical liquid species packet has an inconsistent donor/orientation.");
                        concentration[i] = narrow(std::abs(signed_rate));
                        amount += std::abs(signed_rate);
                        mass_rate += signed_rate * catalog.species()[i].molarMass;
                        dissolved += species_field(id, P::Dissolved).local_value(donor);
                    }
                    if (d_options.dissolved_transport != ReactingTransportMode::Advective && dissolved.value() > 0)
                    {
                        // Selectively moving part of a nonideal solution requires
                        // supplied partial molar volumes. Existing ALE gates require
                        // Advective, so the full liquid carrier always closes there.
                        local_incomplete_carrier = 1;
                        continue;
                    }
                    if (amount.value() == 0)
                        continue;
                    const auto point = d_options.liquid_thermodynamics->evaluate(
                        d_previous_temperature.local_value(donor), d_absolute_pressure.local_value(donor),
                        reacting_species_detail::composition(concentration));
                    const double packet = narrow(mass_rate.value() / point.density);
                    require((carrier > 0 && packet >= 0) || (carrier < 0 && packet <= 0),
                        "Liquid material-volume packet changed canonical orientation.");
                    d_transport_liquid_material_volume_flux.owned_write_view()(
                        d_transport_liquid_material_volume_flux.owned_row(face), 0) = packet;
                }
            });
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_incomplete_carrier,
            &incomplete_carrier);
        d_liquid_transport_volume_receipt = {{}, mesh_geometry_epoch(*d_mesh), incomplete_carrier == 0};
        if constexpr (requires { d_transport_liquid_material_volume_flux.sync_ghosts(); })
            d_transport_liquid_material_volume_flux.sync_ghosts();

        // Exact owner-oriented carrier/slip volume ledger uses each population's
        // post-transport occupancy and the same combined-flux upwind selector.
        d_transport_bubble_slip_volume_flux.put_scalar(0.0);
        d_transport_bubble_carrier_volume_flux.put_scalar(0.0);
        const int component = static_cast<int>(axis);
        collective_detail::collective_local_validation(*d_mesh, "Reacting canonical mixture volume flux",
            [&]
            {
                for (const auto face : d_transport_bubble_slip_volume_flux.owned_face_ids())
                {
                    const auto owner = d_mesh->owner_cell(face);
                    const bool boundary = d_mesh->is_boundary_face(face);
                    const auto neighbor = boundary ? owner : d_mesh->opposite_or_periodic_neighbor_cell(face, owner);
                    const bool open = boundary && std::ranges::find(d_options.free_surface_patches,
                                                      d_mesh->boundary_batch_name(d_mesh->boundary_id(face))) !=
                                                      d_options.free_surface_patches.end();
                    if (boundary && !open)
                        continue;
                    double carrierSum = 0.0, slipSum = 0.0;
                    for (std::size_t p = 0; p < d_population_slip_fields.size(); ++p)
                    {
                        const double carrier = liquid_flux.value(face);
                        const double slipSpeed = boundary
                                                     ? d_population_slip_fields[p]->local_value(owner)
                                                     : 0.5 * (d_population_slip_fields[p]->local_value(owner) +
                                                                 d_population_slip_fields[p]->local_value(neighbor));
                        const double slipFlux =
                            slipSpeed * d_mesh->face_area_vector(face).component(static_cast<std::size_t>(component));
                        const double combined = carrier + slipFlux;
                        if (boundary && combined <= 0.0)
                            continue;
                        const auto upwind = combined >= 0.0 ? owner : neighbor;
                        const double alpha = d_population_volume_fractions[p]->local_value(upwind);
                        carrierSum += carrier * alpha;
                        slipSum += slipFlux * alpha;
                    }
                    d_transport_bubble_slip_volume_flux.owned_write_view()(
                        d_transport_bubble_slip_volume_flux.owned_row(face), 0) = slipSum;
                    d_transport_bubble_carrier_volume_flux.owned_write_view()(
                        d_transport_bubble_carrier_volume_flux.owned_row(face), 0) = carrierSum;
                }
            });
        species_step_sums interphase_step;
        std::vector<Reacting::CellSpeciesState> candidates;
        std::vector<Reacting::ReactionLocalReceipt> chemical_receipts;
        std::vector<std::vector<BubbleGasPhysics::PopulationNumberState>> candidateNumbers;
        std::vector<double> transportedGas(volumes.size()), transportedMaterial(volumes.size());
        d_species_source_ledgers.resize(volumes.size());
        d_species_population_ledgers.resize(volumes.size());
        collective_detail::collective_local_validation(*d_mesh, "Reacting history scratch", [&]
        {
            interphase_step.resize(catalog.species().size() * (4 + d_options.population_ids.size()));
            chemical_receipts.resize(volumes.size());
        });
        collective_detail::collective_local_validation(*d_mesh, "Reacting joint species and population preview",
            [&]
            {
                for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                {
                    const auto cell = static_cast<local_ordinal_type>(owned);
                    auto state = cell_species_state(cell, volumes[owned]);
                    auto numbers = cell_population_numbers(cell, volumes[owned]);
                    auto environment = species_environment(cell, time - dt, temperature.value(cell), volumes[owned]);
                    transportedGas[owned] = d_alpha_g.value(cell);
                    transportedMaterial[owned] = d_alpha_l.value(cell) + d_alpha_g.value(cell);
                    Reacting::ReactionContext context{catalog, state, environment};
                    const auto source = d_options.species_source ? d_options.species_source(context, dt)
                                                                 : Reacting::ReactionSourceInput{};
                    const auto chemistry = Reacting::advance_reactions(
                        *d_options.reaction_mechanism, state, environment, dt, source, d_options.reaction_advance);
                    state = chemistry.state;
                    d_species_source_ledgers[owned] = chemistry.ledger;
                    // Reporting ledgers may round away a channel before a later physical cancellation.
                    // Accumulate the accepted local receipt and each original reservoir packet before one division.
                    const auto& raw = chemistry.localReceipt;
                    chemical_receipts[owned] = raw; // Transient exact keyed chemistry input to accepted history.
                    require(raw.schemaFingerprint == catalog.fingerprint() &&
                                raw.reactionHeatContributions.size() ==
                                    d_options.reaction_mechanism->reactions().size(),
                        "Field thermal receipt does not match its mechanism/schema.");
                    detail::CompensatedSum<long double> thermalEnergy;
                    thermalEnergy += raw.externalHeat;
                    for (const auto heat : raw.reactionHeatContributions)
                        thermalEnergy += heat;
                    for (const auto& transfer : chemistry.ledger.externalTransfers)
                        thermalEnergy += transfer.energy;
                    const auto measure = static_cast<long double>(dt) * volumes[owned];
                    require(std::isfinite(measure) && measure > 0,
                        "Field thermal source needs a finite positive time-volume measure.");
                    const auto powerDensity = thermalEnergy.value() / measure;
                    require(thermalEnergy.value() == 0 || powerDensity != 0,
                        "Field thermal source underflows its physical rate.");
                    d_species_thermal_source.owned_write_view()(owned, 0) = narrow(powerDensity);
                    auto populationLedger = Reacting::SpeciesSourceLedger{};
                    populationLedger.chemicalChange = reacting_species_detail::zero_like(state);
                    populationLedger.externalChange = reacting_species_detail::zero_like(state);
                    const auto populationInitial = state;
                    const auto initialNumbers = numbers;
                    std::vector<detail::CompensatedSum<long double>> countEndpoints(numbers.size());
                    for (std::size_t p = 0; p < numbers.size(); ++p)
                        countEndpoints[p] += numbers[p].number;
                    std::vector<detail::CompensatedSum<long double>> changes(
                        state.amounts.size() + state.dispersedPopulations.size() * catalog.species().size());
                    auto materialEndpoints = changes;
                    require(raw.postSourceAmounts.size() == materialEndpoints.size(),
                        "Raw source endpoint has the wrong compartment schema.");
                    std::size_t initialSlot = 0;
                    const auto seedMaterial = [&](double represented)
                    {
                        const auto& parts = raw.postSourceAmounts[initialSlot];
                        require(std::isfinite(parts[0]) && std::isfinite(parts[1]),
                            "Raw source endpoint contains a nonfinite part.");
                        auto& stock = materialEndpoints[initialSlot++];
                        stock += parts[0];
                        stock += parts[1];
                        const auto amount = stock.value();
                        require(std::isfinite(amount) && amount >= 0 && narrow(amount) == represented,
                            "Raw source endpoint does not project to the actual chemical state.");
                    };
                    for (const auto value : state.amounts)
                        seedMaterial(value);
                    for (const auto& population : state.dispersedPopulations)
                        for (const auto value : population.amounts)
                            seedMaterial(value);
                    double elapsed = 0.0, next = dt;
                    std::size_t substeps = 0;
                    while (elapsed < dt)
                    {
                        require(substeps < static_cast<std::size_t>(d_options.max_subcycles),
                            "Population local subcycle limit exceeded.");
                        double step = std::min(next, dt - elapsed);
                        std::size_t retries = 0;
                        for (;;)
                        {
                            require(step > 0.0 && elapsed + step > elapsed,
                                "Population step cannot advance representable time.");
                            try
                            {
                                auto stageEnvironment = environment;
                                stageEnvironment.time = environment.time + elapsed;
                                require(stageEnvironment.time + step > stageEnvironment.time,
                                    "Population callback interval cannot advance representable absolute time.");
                                std::vector<double> stageLiquid(catalog.species().size());
                                for (std::size_t s = 0; s < stageLiquid.size(); ++s)
                                    stageLiquid[s] = state.amount(catalog, catalog.species()[s].id, P::Liquid) +
                                                     state.amount(catalog, catalog.species()[s].id, P::Dissolved);
                                stageEnvironment.liquidVolume =
                                    reacting_species_detail::total(stageLiquid) > 0.0
                                        ? reacting_species_detail::mass(catalog, stageLiquid) /
                                              d_options.liquid_thermodynamics
                                                  ->evaluate(stageEnvironment.temperature,
                                                      stageEnvironment.absolutePressure,
                                                      reacting_species_detail::composition(stageLiquid))
                                                  .density
                                        : 0.0;
                                const auto conditions = population_conditions(state, numbers, stageEnvironment);
                                const Reacting::ReactionContext stageContext{catalog, state, stageEnvironment};
                                const auto packet = population_packet(stageContext, conditions, step);
                                const auto result = BubbleGasPhysics::advance_multispecies_population_exchange(
                                    catalog, state, numbers, stageEnvironment, packet);
                                auto acceptedChanges = changes;
                                auto acceptedMaterial = materialEndpoints;
                                auto donorMargins = materialEndpoints;
                                auto acceptedCounts = countEndpoints;
                                // The pure kernel creates one reaction per original transfer slot,
                                // with no admission/scaling. Reporting net chemicalChange is NOT a receipt.
                                require(result.ledger.reactionExtents.size() == packet.transfers.size(),
                                    "Population material receipt changed its original transfer dimensions.");
                                for (std::size_t event = 0; event < packet.transfers.size(); ++event)
                                {
                                    const auto& transfer = packet.transfers[event];
                                    require(std::isfinite(transfer.moles) && transfer.moles >= 0 &&
                                                result.ledger.reactionExtents[event] == transfer.moles,
                                        "Population material receipt changed an accepted original transfer.");
                                    const auto donor = species_compartment_index(
                                        transfer.species, transfer.donor, transfer.donorPopulation);
                                    const auto receiver = species_compartment_index(
                                        transfer.species, transfer.receiver, transfer.receiverPopulation);
                                    const auto amount = static_cast<long double>(transfer.moles);
                                    donorMargins[donor] += -amount; // Never credit incoming material to a gross donor.
                                    acceptedMaterial[donor] += -amount;
                                    acceptedMaterial[receiver] += amount;
                                    acceptedChanges[donor] += -amount;
                                    acceptedChanges[receiver] += amount;
                                }
                                for (const auto& margin : donorMargins)
                                    if (!std::isfinite(margin.value()) || margin.value() < 0)
                                        throw std::runtime_error(
                                            "Raw population transfers overdraw an original material donor.");
                                auto acceptedState = populationInitial;
                                require(result.state.amounts.size() == acceptedState.amounts.size() &&
                                            result.state.dispersedPopulations.size() ==
                                                acceptedState.dispersedPopulations.size(),
                                    "Population sequential material state changed its compartment schema.");
                                std::size_t i = 0;
                                const auto endpoint = [&](std::size_t index, double sequential)
                                {
                                    const auto wide = acceptedMaterial[index].value();
                                    if (!std::isfinite(wide) || wide < 0)
                                        throw std::runtime_error(
                                            "Raw population material endpoint is negative or nonfinite.");
                                    const auto value = narrow(wide);
                                    const auto density = static_cast<double>(wide / volumes[owned]);
                                    if (value < 0 ||
                                        (wide > 0 && (value == 0 || density == 0 || !std::isfinite(density))))
                                        throw std::runtime_error("Raw population material endpoint cannot be "
                                                                 "represented by its stored field.");
                                    require(std::isfinite(sequential) && sequential >= 0 &&
                                                ((value == 0) == (sequential == 0)),
                                        "Raw population material replay changed an empty/nonempty sequential state.");
                                    const long double scale =
                                        std::max(std::abs(wide), std::abs(static_cast<long double>(sequential)));
                                    require(std::abs(wide - sequential) <=
                                                64 * std::numeric_limits<double>::epsilon() * scale,
                                        "Raw population material replay differs from its actual sequential state.");
                                    return value;
                                };
                                for (auto& value : acceptedState.amounts)
                                {
                                    value = endpoint(i, result.state.amounts[i]);
                                    ++i;
                                }
                                for (std::size_t population = 0; population < acceptedState.dispersedPopulations.size();
                                    ++population)
                                {
                                    auto& output = acceptedState.dispersedPopulations[population];
                                    const auto& sequential = result.state.dispersedPopulations[population];
                                    require(output.id == sequential.id &&
                                                output.amounts.size() == sequential.amounts.size(),
                                        "Population sequential material state changed a configured bin.");
                                    for (std::size_t species = 0; species < output.amounts.size(); ++species)
                                        output.amounts[species] = endpoint(i++, sequential.amounts[species]);
                                }
                                auto acceptedNumbers = initialNumbers;
                                require(result.localNumberChanges.size() == acceptedNumbers.size() &&
                                            result.numbers.size() == acceptedNumbers.size(),
                                    "Population raw number receipt has the wrong dimensions.");
                                for (std::size_t p = 0; p < acceptedNumbers.size(); ++p)
                                {
                                    const auto& raw = result.localNumberChanges[p];
                                    require(std::isfinite(raw.high) && std::isfinite(raw.correction),
                                        "Population raw number receipt is nonfinite.");
                                    acceptedCounts[p] += raw.high;
                                    acceptedCounts[p] += raw.correction;
                                    const auto wide = acceptedCounts[p].value();
                                    acceptedNumbers[p].number = narrow(wide);
                                    if (acceptedNumbers[p].number < 0 || (wide > 0 && acceptedNumbers[p].number == 0))
                                        throw std::runtime_error(
                                            "Accumulated population count endpoint is invalid or unrepresentable.");
                                    const double sequential = result.numbers[p].number;
                                    require(result.numbers[p].population == acceptedNumbers[p].population,
                                        "Population raw number receipt changed its configured identity.");
                                    require(std::isfinite(sequential) && sequential >= 0 &&
                                                ((acceptedNumbers[p].number == 0) == (sequential == 0)),
                                        "Population raw count replay changed an empty/nonempty sequential state.");
                                    const long double scale =
                                        std::max(std::abs(wide), std::abs(static_cast<long double>(sequential)));
                                    require(std::abs(wide - sequential) <=
                                                64 * std::numeric_limits<double>::epsilon() * scale,
                                        "Population raw count replay differs from its actual sequential state.");
                                }
                                changes = std::move(acceptedChanges);
                                materialEndpoints = std::move(acceptedMaterial);
                                countEndpoints = std::move(acceptedCounts);
                                state = std::move(acceptedState);
                                numbers = std::move(acceptedNumbers);
                                // Only accepted packets enter gross interphase histories. Rejected retries
                                // cannot debit donors or increment diagnostic events.
                                for (const auto& transfer : packet.transfers)
                                {
                                    const auto donor = species_compartment_index(
                                        transfer.species, transfer.donor, transfer.donorPopulation);
                                    const auto receiver = species_compartment_index(
                                        transfer.species, transfer.receiver, transfer.receiverPopulation);
                                    interphase_step[donor].source_change += -static_cast<long double>(transfer.moles);
                                    interphase_step[receiver].source_change += transfer.moles;
                                    const auto s = catalog.index(transfer.species);
                                    const bool donor_gas = transfer.donor != P::Liquid;
                                    const bool receiver_gas = transfer.receiver != P::Liquid;
                                    const bool donor_solution =
                                        transfer.donor == P::Liquid || transfer.donor == P::Dissolved;
                                    const bool receiver_solution =
                                        transfer.receiver == P::Liquid || transfer.receiver == P::Dissolved;
                                    if (donor_gas != receiver_gas)
                                        interphase_step[s].gas_subset_source +=
                                            receiver_gas ? static_cast<long double>(transfer.moles)
                                                         : -static_cast<long double>(transfer.moles);
                                    if (donor_solution != receiver_solution)
                                        interphase_step[s].solution_subset_source +=
                                            receiver_solution ? static_cast<long double>(transfer.moles)
                                                              : -static_cast<long double>(transfer.moles);
                                    interphase_step[donor].amount[Reacting::history_detail::InterphaseOutflow] +=
                                        transfer.moles;
                                    interphase_step[receiver].amount[Reacting::history_detail::InterphaseInflow] +=
                                        transfer.moles;
                                }
                                ++substeps;
                                elapsed = step == dt - elapsed ? dt : elapsed + step;
                                next = std::min(dt, 2 * step);
                                break;
                            }
                            catch (const std::runtime_error&)
                            {
                                if (retries++ >= d_options.reaction_advance.maximumRetries)
                                    throw;
                                ++populationLedger.retries;
                                step *= 0.5;
                            }
                        }
                    }
                    std::size_t i = 0;
                    for (auto& value : populationLedger.chemicalChange.amounts)
                        value = narrow(changes[i++].value());
                    for (auto& population : populationLedger.chemicalChange.dispersedPopulations)
                        for (auto& value : population.amounts)
                            value = narrow(changes[i++].value());
                    populationLedger.substeps = substeps;
                    d_species_population_ledgers[owned] = std::move(populationLedger);
                    candidates.push_back(std::move(state));
                    candidateNumbers.push_back(std::move(numbers));
                    d_species_escape_ledgers[owned].externalMass = narrow(escapedMass[owned].value());
                    d_species_escape_ledgers[owned].transferEnergy = narrow(escapedEnergy[owned].value());
                }
            });
        collective_detail::collective_local_validation(
            *d_mesh, "Reacting species field publication", [&] { publish_species_states(candidates, volumes); });
        for (std::size_t owned = 0; owned < volumes.size(); ++owned)
            for (std::size_t p = 0; p < d_options.population_ids.size(); ++p)
                mutable_population_number(d_options.population_ids[p]).owned_write_view()(owned, 0) =
                    candidateNumbers[owned][p].number / volumes[owned];
        reconstruct_species_derived(temperature);
        for (std::size_t owned = 0; owned < volumes.size(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            d_source_alpha_reaction.owned_write_view()(owned, 0) = (d_alpha_g.value(cell) - transportedGas[owned]) / dt;
            d_species_material_source.owned_write_view()(owned, 0) =
                (d_alpha_l.value(cell) + d_alpha_g.value(cell) - transportedMaterial[owned]) / dt;
        }
        // Independent species closure includes internal stoichiometry and physical boundary withdrawals.
        for (const auto& species : catalog.species())
        {
            std::array<double, 3> local{}, global{};
            collective_detail::collective_local_validation(*d_mesh, "Reacting species closure input", [&]
            {
                std::array<detail::CompensatedSum<long double>, 3> sums;
                for (std::size_t owned = 0; owned < volumes.size(); ++owned)
                {
                    const auto& after = candidates[owned];
                    for (std::size_t p = 0; p < 4; ++p)
                    {
                        const auto phase = static_cast<P>(p);
                        sums[0] += before[owned].phase_amount(catalog, species.id, phase);
                        sums[1] += after.phase_amount(catalog, species.id, phase);
                        sums[2] += d_species_source_ledgers[owned].chemicalChange.phase_amount(
                            catalog, species.id, phase);
                        sums[2] += d_species_source_ledgers[owned].externalChange.phase_amount(
                            catalog, species.id, phase);
                        sums[2] += d_species_escape_ledgers[owned].externalChange.phase_amount(
                            catalog, species.id, phase);
                    }
                }
                for (std::size_t i = 0; i < local.size(); ++i)
                    local[i] = narrow(sums[i].value());
            });
            Teuchos::reduceAll(
                *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 3, local.data(), global.data());
            const double scale = std::max({std::abs(global[0]), std::abs(global[1]), std::abs(global[2])});
            require(
                std::abs((global[1] - global[0]) - global[2]) <= 4096 * std::numeric_limits<double>::epsilon() * scale,
                "Accepted multispecies source/transport/escape ledger does not close.");
        }
        publish_species_history(candidates, std::move(interphase_step), false, chemical_receipts);
        d_history_initialized = true;
        sync_all_fields();
        if constexpr (requires { d_transport_bubble_slip_volume_flux.sync_ghosts(); })
        {
            d_transport_bubble_slip_volume_flux.sync_ghosts();
            d_transport_bubble_carrier_volume_flux.sync_ghosts();
            d_transport_liquid_material_volume_flux.sync_ghosts();
        }
        collective_detail::collective_local_validation(*d_mesh, "Reacting accepted interval receipt",
            [&]
            {
                require(d_species_event_receipt.schema_fingerprint == catalog.fingerprint() &&
                            d_species_event_receipt.initialization_generation == d_species_generation &&
                            d_species_event_receipt.revision != std::numeric_limits<std::uint64_t>::max() &&
                            d_species_publication_nonce != std::numeric_limits<std::uint64_t>::max() &&
                            std::isfinite(time - dt) && time > time - dt,
                    "Reacting accepted interval receipt is invalid or exhausted.");
            });
        collective_detail::require_uniform_value(
            *d_mesh, d_species_event_receipt.revision, "Reacting accepted event revision");
        collective_detail::require_uniform_value(
            *d_mesh, d_species_publication_nonce, "Reacting publication nonce allocator");
        collective_detail::require_uniform_value(*d_mesh, time, "Reacting accepted interval end");
        collective_detail::require_uniform_value(*d_mesh, dt, "Reacting accepted interval duration");
        d_species_event_receipt = {true, d_species_event_receipt.revision + 1, d_species_generation,
            catalog.fingerprint(), time, dt, ++d_species_publication_nonce};
        d_liquid_transport_volume_receipt.event = d_species_event_receipt;
    }
    catch (...)
    {
        restore(backup);
        throw;
    }
}
template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::evaluate_species_bubble_volume(
    scalar_type candidate_absolute_pressure_offset) const -> scalar_type
{
    validate_species_pressure_metadata(candidate_absolute_pressure_offset);
    scalar_type local = 0.0;
    collective_detail::collective_local_validation(*d_mesh, "Reacting candidate mixture bubble volume",
        [&]
        {
            if (!d_initial_state_initialized || !std::isfinite(candidate_absolute_pressure_offset))
                throw std::invalid_argument(
                    "Mixture volume evaluation requires initialized state and finite pressure offset.");
            const auto& catalog = d_options.reaction_mechanism->catalog();
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const double volume = d_mesh->cell_volume(cell);
                const auto state = cell_species_state(cell, volume);
                auto environment = species_environment(cell, 0.0, d_previous_temperature.value(cell), volume);
                environment.absolutePressure += candidate_absolute_pressure_offset - d_absolute_pressure_offset;
                const auto conditions =
                    population_conditions(state, cell_population_numbers(cell, volume), environment);
                for (std::size_t p = 0; p < conditions.size(); ++p)
                    if (reacting_species_detail::total(state.dispersedPopulations[p].amounts) > 0.0)
                        local += reacting_species_detail::mass(catalog, state.dispersedPopulations[p].amounts) /
                                 conditions[p].properties.density;
            }
        });
    return global_sum(local);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::refresh_species_thermodynamics(scalar_type time, const field_type& temperature,
    const field_type& gauge_pressure, const velocity_field_type& velocity, const material_type& material)
{
    if (!species_enabled())
        throw std::logic_error("Species EOS refresh requires a general species schema.");
    initialize_state(time, temperature, gauge_pressure, velocity, material, false);
}
template<TpetraTypePack Pack, class MeshType>
ReactingGasOptions ReactingGasModel<Pack, MeshType>::require_options(
    const SP<const mesh_type>& mesh, ReactingGasOptions options)
{
    collective_detail::collective_local_validation(
        *mesh, "Reacting model options", [&] { validate_reacting_gas_options(options); });
    collective_detail::require_uniform_value(
        *mesh, static_cast<int>(static_cast<bool>(options.reaction_mechanism)), "Reacting typed schema presence");
    collective_detail::require_uniform_value(*mesh, static_cast<int>(options.mode), "Reacting model selection");
    if (options.reaction_mechanism)
        collective_detail::require_uniform_value(*mesh,
            static_cast<unsigned long long>(reacting_species_detail::configuration_signature(options)),
            "Reacting ordered mechanism/property/source control-flow metadata");
    return options;
}
} // namespace SimpleFluid

#include "equations/ReactingSpeciesHistory.ipp"

#include "equations/ReactingSpeciesPressure.ipp"
