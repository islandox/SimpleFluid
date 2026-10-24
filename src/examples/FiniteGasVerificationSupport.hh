/** @file
 * @brief Manufactured finite-liquid/two-gas ALE verification laws.
 * These laws are not a calibrated phase internal-energy model.
 */
#pragma once
#include "ReactingMixtureExampleSupport.hh"
#include "equations/ReactingGasModel.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numbers>
#include <span>
#include <string>

namespace SimpleFluid::Verification::FiniteGas
{
namespace R = Reacting;
namespace M = Examples::ReactingMixture;
using P = R::SpeciesPhase;
inline constexpr std::array gas_ids{M::gas_one, M::gas_two};
inline constexpr double one_fraction = .25, molar_feed_mass = one_fraction * .002 + (1 - one_fraction) * .004;
struct Parameters
{
    double temperature, pressure, density, cp, expansion, viscosity, conductivity, sigma;
    double dissolved_moles_per_bulk_volume, micro_moles_per_bulk_volume, micro_numbers_per_bulk_volume;
    double large_moles_per_bulk_volume, large_numbers_per_bulk_volume, micro_lifetime, large_lifetime, slip;
    double large_film_diffusivity, large_film_henry;
};
using Rate = std::function<double(const R::ReactionContext&)>;                // External mol/(m3 actual bulk s).
using IncomingTemperature = std::function<double(const R::ReactionContext&)>; // Explicit material-feed K law.
inline double density(const Parameters& p, double temperature)
{
    const double value = p.density * (1 - p.expansion * (temperature - p.temperature));
    M::require(std::isfinite(value) && value > 0, "Manufactured weak-Bouss density is invalid.");
    return value;
}
inline M::Mixture mixture(Parameters p)
{
    M::Mixture value(false, false, R::EnergyConvention::SensibleEnergy, 0);
    constexpr R::ThermodynamicBounds liquid_bounds{250, 600, 1, 1e8};
    // A manufactured Gibbs closure supplies a stable off-reference extension.
    // Its caloric slope is NOT the physical Boussinesq cp*T storage coefficient.
    constexpr double eos_compressibility = 1e-9, eos_heat_capacity = 1e8;
    M::require(std::isfinite(p.temperature) && p.temperature >= liquid_bounds.minimum_temperature &&
                   p.temperature <= liquid_bounds.maximum_temperature && std::isfinite(p.pressure) &&
                   p.pressure >= liquid_bounds.minimum_pressure && p.pressure <= liquid_bounds.maximum_pressure &&
                   std::isfinite(p.density) && p.density > 0 && std::isfinite(p.cp) && p.cp > 0 &&
                   std::isfinite(p.expansion),
        "Manufactured reference liquid properties are outside their declared domain.");
    const double f_low = 1 - p.expansion * (liquid_bounds.minimum_temperature - p.temperature);
    const double f_high = 1 - p.expansion * (liquid_bounds.maximum_temperature - p.temperature);
    const double f_min = std::min(f_low, f_high), f_max = std::max(f_low, f_high);
    M::require(std::isfinite(f_min) && std::isfinite(f_max) && f_min > 0,
        "Manufactured density is not positive over its declared temperature range.");
    const double volume_pressure = eos_compressibility / p.density;
    const double dp_max = std::max(p.pressure - liquid_bounds.minimum_pressure,
        liquid_bounds.maximum_pressure - p.pressure);
    const double phi_t_max = std::abs(p.expansion) / (p.density * f_min * f_min);
    const double phi_tt_max = 2 * p.expansion * p.expansion / (p.density * f_min * f_min * f_min);
    const double cp_min = eos_heat_capacity - liquid_bounds.maximum_temperature * phi_tt_max * dp_max;
    M::require(1 / (p.density * f_max) - volume_pressure * dp_max > 0 &&
                   eos_heat_capacity - p.pressure * phi_t_max -
                           (phi_t_max + liquid_bounds.maximum_temperature * phi_tt_max) * dp_max >
                       0 &&
                   cp_min > 0 &&
                   cp_min - liquid_bounds.maximum_temperature * phi_t_max * phi_t_max / volume_pressure > 0,
        "Manufactured Gibbs liquid is not stable over its full declared T/P range.");
    value.liquid = std::make_shared<const R::MixtureThermodynamics>(
        value.catalog, liquid_bounds, R::EOSKind::Liquid,
        [p, volume_pressure, eos_heat_capacity](double temperature, double pressure, std::span<const double>)
        {
            const double rho_reference = density(p, temperature), phi = 1 / rho_reference;
            const double phi_t = p.density * p.expansion / (rho_reference * rho_reference);
            const double phi_tt = 2 * p.density * p.density * p.expansion * p.expansion /
                                  (rho_reference * rho_reference * rho_reference);
            const double dp = pressure - p.pressure, volume = phi - volume_pressure * dp;
            // At constant p_ref this retains the application's exact rho(T),
            // including its independent surface/temperature/escape oracles.
            const double rho = rho_reference / (1 - volume_pressure * dp * rho_reference);
            const double h = p.cp * p.temperature + p.pressure / p.density +
                             eos_heat_capacity * (temperature - p.temperature) +
                             (phi - temperature * phi_t) * dp - .5 * volume_pressure * dp * dp;
            const double e_t = eos_heat_capacity - p.pressure * phi_t - (phi_t + temperature * phi_tt) * dp;
            return R::MixtureThermodynamicPoint{rho, h - pressure * volume, h, e_t,
                -temperature * phi_t + pressure * volume_pressure, -phi_t * rho * rho,
                volume_pressure * rho * rho, {}, {}, p.viscosity, p.conductivity};
        },
        R::EnergyConvention::SensibleEnergy);
    std::vector<R::IdealGasCaloricSpecies> gas_caloric;
    for (const auto& species : value.catalog->species())
        gas_caloric.push_back({M::molar_heat_capacity(species.id), p.temperature, 0});
    value.gas = R::make_ideal_gas_mixture(
        value.catalog, std::move(gas_caloric), {250, 600, 1, 1e12}, R::EnergyConvention::SensibleEnergy);
    std::vector<R::ReactionDefinition> phases;
    for (std::size_t pop = 0; pop < 2; ++pop)
        for (std::size_t s = 0; s < gas_ids.size(); ++s)
            phases.push_back({static_cast<R::ReactionId>(310 + 2 * pop + s),
                "same-species " + std::string(pop == 0 ? "micro" : "large") + " dissolution " + std::to_string(gas_ids[s]),
                {{gas_ids[s], P::Dispersed, -1, pop == 0 ? "micro" : "large"}, {gas_ids[s], P::Dissolved, 1}}, 0});
    value.mechanism = std::make_shared<const R::ReactionMechanism>(
        value.catalog, std::move(phases), R::EnergyConvention::SensibleEnergy);
    return value;
}
inline M::InitialCell initial(const M::Mixture& mixture, const Parameters& p, const R::ReactionEnvironment& e)
{
    const auto& catalog = *mixture.catalog;
    M::InitialCell result{R::CellSpeciesState(catalog), {}, e.bulkVolume};
    for (std::size_t pop = 0; pop < 2; ++pop)
    {
        const std::string name = pop == 0 ? "micro" : "large";
        const double moles = (pop == 0 ? p.micro_moles_per_bulk_volume : p.large_moles_per_bulk_volume) * e.bulkVolume;
        const double number =
            (pop == 0 ? p.micro_numbers_per_bulk_volume : p.large_numbers_per_bulk_volume) * e.bulkVolume;
        M::require((moles == 0) == (number == 0), "Explicit initial gas and counts must be paired.");
        result.species.dispersedPopulations.push_back({name, std::vector<double>(catalog.species().size())});
        result.population_numbers.push_back(number);
        result.species.set_amount(catalog, gas_ids[0], P::Dispersed, one_fraction * moles, name);
        result.species.set_amount(catalog, gas_ids[1], P::Dispersed, (1 - one_fraction) * moles, name);
        if (number > 0)
            result.liquid_volume -=
                R::solve_bubble_radius(*mixture.gas, result.species.dispersedPopulations.back().amounts, e.temperature,
                    e.absolutePressure, p.sigma, number, 1e-12, .1)
                    .volume;
    }
    M::require(std::isfinite(result.liquid_volume) && result.liquid_volume > 0,
        "Initial finite liquid must have positive physical room beside the actual capillary bubbles.");
    double solute_mass = 0;
    for (std::size_t s = 0; s < gas_ids.size(); ++s)
    {
        const double amount =
            (s == 0 ? one_fraction : 1 - one_fraction) * p.dissolved_moles_per_bulk_volume * e.bulkVolume;
        result.species.set_amount(catalog, gas_ids[s], P::Dissolved, amount);
        solute_mass += catalog.species()[catalog.index(gas_ids[s])].molarMass * amount;
    }
    const double solvent_mass = density(p, e.temperature) * result.liquid_volume - solute_mass;
    M::require(std::isfinite(solvent_mass) && solvent_mass >= 0, "Finite solutes exceed initial solution mass.");
    result.species.set_amount(catalog, M::solvent, P::Liquid, solvent_mass / .018);
    return result;
}
struct Residence
{
    double old_fraction, new_fraction;
};
inline Residence residence(double dt, double lifetime)
{
    M::require(std::isfinite(dt) && dt > 0 && std::isfinite(lifetime) && lifetime > 0,
        "Explicit residence law needs positive finite time scales.");
    const double x = dt / lifetime, retention = std::exp(-x);
    const double source_fraction =
        x < 1e-3 ? 1 + x * (-.5 + x * (1. / 6 + x * (-1. / 24 + x / 120))) : -std::expm1(-x) / x;
    M::require(std::isfinite(source_fraction) && source_fraction >= 0 && source_fraction <= 1,
        "Continuous-feed survival fraction is invalid.");
    return {1 - retention, source_fraction};
}
inline ReactingGasOptions options(
    const M::Mixture& mixture, Parameters p, Rate rate, IncomingTemperature incoming_temperature)
{
    ReactingGasOptions value;
    value.mode = ReactingGasMode::Species;
    value.reference_pressure = p.pressure;
    value.solution_transport = ReactingTransportMode::Advective;
    value.rise_velocity_mode = p.slip > 0 ? BubbleRiseVelocityMode::ConstantSlip : BubbleRiseVelocityMode::ZeroSlip;
    value.constant_slip_velocity = p.slip;
    value.surface_tension = p.sigma;
    value.min_radius = 1e-12;
    value.max_radius = .1;
    value.reaction_mechanism = mixture.mechanism;
    value.liquid_thermodynamics = mixture.liquid;
    value.gas_thermodynamics = mixture.gas;
    for (const auto& species : mixture.catalog->species())
    {
        ReactingSpeciesProperties property;
        property.species = species.id;
        property.carried_molar_enthalpy = [p, id = species.id, mass = species.molarMass](
                                              const R::ReactionContext& c, P phase, std::string_view)
        {
            const double temperature = c.environment.temperature;
            if (phase == P::Liquid || phase == P::Dissolved)
                return mass * (p.cp * temperature + c.environment.absolutePressure / density(p, temperature));
            return M::molar_heat_capacity(id) * (temperature - p.temperature) + M::gas_constant * temperature;
        };
        value.species_properties.push_back(std::move(property));
    }
    value.initial_species_state = [mixture, p](const R::ReactionEnvironment& e, std::int64_t)
    { return initial(mixture, p, e).species; };
    value.initial_population_numbers = [mixture, p](const R::ReactionContext& c)
    { return initial(mixture, p, c.environment).population_numbers; };
    value.species_source = [mixture, p, rate, incoming_temperature](const R::ReactionContext& c, double dt)
    {
        const auto& catalog = c.catalog;
        const double source = rate(c);
        M::require(std::isfinite(source) && source >= 0, "Explicit external gas feed rate is invalid.");
        M::require(bool(incoming_temperature), "The external dissolved feed requires its supplied temperature law.");
        const double feed_temperature = incoming_temperature(c);
        M::require(
            std::isfinite(feed_temperature) && feed_temperature > 0, "Incoming reference temperature is invalid.");
        const auto micro = residence(dt, p.micro_lifetime), large = residence(dt, p.large_lifetime);
        R::ReactionSourceInput input;
        for (std::size_t s = 0; s < gas_ids.size(); ++s)
        {
            const auto id = gas_ids[s];
            const double incoming = source * c.environment.bulkVolume * dt * (s == 0 ? one_fraction : 1 - one_fraction);
            const double retained = incoming * micro.new_fraction, dissolved = incoming - retained;
            for (std::size_t pop = 0; pop < 2; ++pop)
            {
                const double old_amount = c.state.amount(catalog, id, P::Dispersed, pop == 0 ? "micro" : "large");
                input.integratedExtents.push_back({static_cast<R::ReactionId>(310 + 2 * pop + s),
                    old_amount * (pop == 0 ? micro.old_fraction : large.old_fraction)});
            }
            for (const auto& [phase, amount, population] :
                std::array{std::tuple{P::Dispersed, retained, std::string{"micro"}},
                    std::tuple{P::Dissolved, dissolved, std::string{}}})
                if (amount > 0)
                    input.externalTransfers.push_back({"explicit verification gas feed", id, phase, amount,
                        catalog.species()[catalog.index(id)].molarMass * amount,
                        phase == P::Dissolved
                            ? catalog.species()[catalog.index(id)].molarMass * amount * p.cp * feed_temperature
                            : 0,
                        {}, population});
        }
        return input;
    };
    value.population_process = [p, rate](const BubbleGasPhysics::MultispeciesPopulationContext& c, double dt)
    {
        const auto& e = c.chemistry.environment;
        const auto micro = residence(dt, p.micro_lifetime), large = residence(dt, p.large_lifetime);
        const double radius = 6.6359547089482127e-8;
        const double birth_moles = 4 * std::numbers::pi * radius * radius * radius / 3 *
                                   (e.absolutePressure + 2 * p.sigma / radius) / (M::gas_constant * e.temperature);
        const double incoming = rate(c.chemistry) * e.bulkVolume * dt * micro.new_fraction;
        BubbleGasPhysics::MultispeciesPopulationPacket packet;
        for (const auto& population : c.populations)
            packet.numberEvents.push_back({population.id, population.id == "micro" ? incoming / birth_moles : 0,
                population.number * (population.id == "micro" ? micro.old_fraction : large.old_fraction)});
        // Preserve the original large-population film law, including its explicit
        // negligible-Henry complete-escape fixture. Micro residence is owned above.
        for (const auto& population : c.populations)
        {
            if (population.id != "large" || population.number == 0 || p.large_film_diffusivity == 0)
                continue;
            M::require(e.liquidVolume > 0, "A supplied film law requires actual finite liquid.");
            const double coefficient = BubbleGasPhysics::hughmark_mass_transfer_coefficient(
                p.large_film_diffusivity, population.radius, density(p, e.temperature), p.viscosity, p.slip);
            for (const auto id : gas_ids)
            {
                const auto s = c.chemistry.catalog.index(id);
                M::require(population.properties.fugacities.size() == c.chemistry.catalog.species().size(),
                    "Mixed-gas film transfer requires full supplied fugacities.");
                const double concentration =
                    c.chemistry.state.amount(c.chemistry.catalog, id, P::Dissolved) / e.liquidVolume;
                const double equilibrium = p.large_film_henry * population.properties.fugacities[s];
                const double amount = dt * coefficient * 4 * std::numbers::pi * population.radius * population.radius *
                                      population.number * (concentration - equilibrium);
                M::require(std::isfinite(amount), "The supplied film packet must be finite.");
                if (amount > 0)
                    packet.transfers.push_back({id, P::Dissolved, P::Dispersed, {}, "large", amount});
                else if (amount < 0)
                    packet.transfers.push_back({id, P::Dispersed, P::Dissolved, "large", {}, -amount});
            }
        }
        return packet;
    };
    value.free_surface_patches = {"zmax"};
    return value;
}
template<class Model> double total(const Model& model)
{
    return model.global_species_moles(gas_ids[0]) + model.global_species_moles(gas_ids[1]);
}
template<class Model> double produced(const Model& model)
{
    double result = 0;
    for (const auto& h : model.species_histories())
        if (std::ranges::find(gas_ids, h.species) != gas_ids.end())
            result += h.chemical_production - h.chemical_consumption + h.external_supply - h.external_withdrawal;
    return result;
}
template<class Model> double escaped(const Model& model)
{
    double result = 0;
    for (const auto& h : model.species_histories())
        if (std::ranges::find(gas_ids, h.species) != gas_ids.end())
            result += h.physical_outflow;
    return result;
}
template<class Model> double phase(const Model& model, P kind, std::string_view population = {})
{
    const auto& grid = model.alpha_g().mesh();
    double local = 0, global = 0;
    for (const auto id : gas_ids)
        for (std::size_t c = 0; c < grid.num_owned_cells(); ++c)
        {
            const auto cell = static_cast<typename Model::local_ordinal_type>(c);
            local += model.species_field(id, kind, population).value(cell) * grid.cell_volume(cell);
        }
    Teuchos::reduceAll(*grid.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
template<class Model> double step_escape(const Model& model, std::string_view population)
{
    double result = 0;
    for (const auto& event : model.physical_escape_events())
        if (event.phase == P::Dispersed && event.population == population)
            result += event.moles;
    return result;
}
template<class Map> double vented(const Map& entries)
{
    double result = 0;
    for (const auto* name : {"gas_one", "gas_two"})
        if (const auto found = entries.find(name); found != entries.end())
            result += found->second;
    return result;
}

template<class Model> double inventory_error(const Model& model)
{
    double result = 0;
    for (const auto& h : model.species_histories())
        result = std::max(result, std::abs(h.inventory_error));
    return result;
}
template<class Model> double gas_mass(const Model& model)
{
    const auto& grid = model.alpha_g().mesh();
    const auto& catalog = model.options().reaction_mechanism->catalog();
    double local = 0, global = 0;
    for (const auto id : gas_ids)
        for (std::size_t c = 0; c < grid.num_owned_cells(); ++c)
        {
            const auto cell = static_cast<typename Model::local_ordinal_type>(c);
            const auto state = model.cell_species_state(cell);
            double amount = state.amount(catalog, id, P::Resolved);
            for (const auto& population : model.options().population_ids)
                amount += state.amount(catalog, id, P::Dispersed, population);
            local += amount * catalog.species()[catalog.index(id)].molarMass;
        }
    Teuchos::reduceAll(*grid.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
template<class Model> double escaped_mass(const Model& model)
{
    const auto& catalog = model.options().reaction_mechanism->catalog();
    double result = 0;
    for (const auto& h : model.species_histories())
        result += h.physical_outflow * catalog.species()[catalog.index(h.species)].molarMass;
    return result;
}

template<class Model> auto fields(const Model& model)
{
    std::map<std::string, const typename Model::field_type*> result;
    for (const auto& species : model.options().reaction_mechanism->catalog().species())
    {
        for (const auto p : {P::Liquid, P::Dissolved, P::Resolved})
            result.emplace("species_" + std::to_string(species.id) + "_phase_" + std::to_string(static_cast<int>(p)),
                &model.species_field(species.id, p));
        for (const auto& population : model.options().population_ids)
            result.emplace("species_" + std::to_string(species.id) + "_population_" + population,
                &model.species_field(species.id, P::Dispersed, population));
    }
    for (const auto& population : model.options().population_ids)
        result.emplace("population_" + population + "_number_per_m3", &model.population_number_field(population));
    result.emplace("alpha_g", &model.alpha_g());
    result.emplace("liquid_mass_kg_m3", &model.liquid_mass_density());
    result.emplace("source_scalar_sensible_W_m3", &model.thermal_source());
    result.emplace("absolute_pressure_Pa", &model.absolute_pressure());
    return result;
}

template<class Model> double step_escape_total(const Model& model)
{
    double result = 0;
    for (const auto& event : model.physical_escape_events())
        if ((event.phase == P::Dispersed || event.phase == P::Resolved) &&
            std::ranges::find(gas_ids, event.species) != gas_ids.end())
            result += event.moles;
    return result;
}

} // namespace SimpleFluid::Verification::FiniteGas
