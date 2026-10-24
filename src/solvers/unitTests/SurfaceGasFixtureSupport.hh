/** @file SurfaceGasFixtureSupport.hh
 * @brief Explicit synthetic sources and initial inventories for surface regression oracles.
 * The dose correlation is a NAMED EXTERNAL FEED approximation, not solution chemistry.
 * Source energy is explicitly zero; the existing Boussinesq cp*T heat owner applies
 * the prescribed joules. A paired supplied nucleation packet owns every new count.
 */
#pragma once
#include "ReactingMixtureTestSupport.hh"
#include "equations/BubblePopulationKinetics.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <numeric>

namespace SimpleFluid::Testing::SurfaceGas
{
namespace M = ReactingMixture;
namespace R = Reacting;
using Phase = R::SpeciesPhase;
inline constexpr auto gas_id = M::gas_one;
inline constexpr const char* gas_name = "gas_one";
inline constexpr const char* micro = "micro";
inline constexpr const char* large = "large";
inline constexpr double supplied_mol_per_joule = 2e-7;
struct Initial
{
    double liquid_fraction = 1, liquid_density = 10, expansion = 1e-3, compressibility = 1e-9;
    double reference_temperature = 300;
    double dissolved_concentration = 1;
    double second_gas_fraction = 0;
    double micro_number_density = 0, micro_molar_density = 0;
    double large_number_density = 0, large_molar_density = 0;
    double minimum_pressure = 10000;
};
using Power = std::function<double(const R::ReactionContext&)>;
inline ReactingGasOptions options(
    const Initial& initial = {}, Power power = {}, double feed_coefficient = supplied_mol_per_joule)
{
    M::Mixture mixture(false, false, R::EnergyConvention::SensibleEnergy, 0);
    auto bounds = M::bounds;
    bounds.minimum_pressure = initial.minimum_pressure;
    // The original supplied nucleation radius has p+2*sigma/r about2.21MPa.
    // Declare its physical constitutive domain explicitly; no solver bound is relaxed.
    auto gas_bounds = bounds;
    gas_bounds.maximum_pressure = 3e6;
    std::vector<R::IdealGasCaloricSpecies> caloric;
    for (const auto& species : mixture.catalog->species())
        caloric.push_back({M::molar_heat_capacity(species.id), 300, 100000 * species.molarMass});
    mixture.gas = R::make_ideal_gas_mixture(
        mixture.catalog, std::move(caloric), gas_bounds, R::EnergyConvention::SensibleEnergy, M::gas_constant);
    mixture.liquid = R::make_linear_compressible_mixture_liquid(
        mixture.catalog, bounds,
        [initial](std::span<const double>)
        {
            // cv>0 over the full supplied EOS range. This is NOT the Boussinesq cp*T owner's cp.
            return R::LinearCompressibleMixtureLiquid{initial.reference_temperature, 101325, initial.liquid_density,
                1e8, initial.expansion, initial.compressibility, 0, .001, 0};
        },
        R::EnergyConvention::SensibleEnergy);
    M::InitialMixture empty;
    empty.reactant_concentration = 0;
    empty.product_concentration = 0;
    empty.dissolved_one_concentration = 0;
    empty.dissolved_two_concentration = 0;
    auto result = M::field_options(mixture, empty);
    result.reference_pressure = 101325;
    result.species_properties = {mixture.transport(gas_id, 0, 1e-5), mixture.transport(M::gas_two, 0, 2e-5)};
    result.population_ids = {micro, large};
    result.initial_species_state = [mixture, initial](const R::ReactionEnvironment& environment, std::int64_t)
    {
        R::CellSpeciesState state(*mixture.catalog);
        state.dispersedPopulations = {{micro, std::vector<double>(mixture.catalog->species().size())},
            {large, std::vector<double>(mixture.catalog->species().size())}};
        const double liquid_volume = initial.liquid_fraction * environment.bulkVolume;
        const double dissolved = initial.dissolved_concentration * liquid_volume;
        const double other_dissolved = initial.second_gas_fraction * dissolved;
        const double liquid_mass = initial.liquid_density * liquid_volume;
        M::require(liquid_mass >= .002 * dissolved + .004 * other_dissolved,
            "Finite dissolved gas exceeds declared solution mass.");
        state.set_amount(*mixture.catalog, M::solvent, Phase::Liquid,
            (liquid_mass - .002 * dissolved - .004 * other_dissolved) / .018);
        state.set_amount(*mixture.catalog, gas_id, Phase::Dissolved, dissolved);
        state.set_amount(*mixture.catalog, M::gas_two, Phase::Dissolved, other_dissolved);
        state.set_amount(
            *mixture.catalog, gas_id, Phase::Dispersed, initial.micro_molar_density * environment.bulkVolume, micro);
        state.set_amount(
            *mixture.catalog, gas_id, Phase::Dispersed, initial.large_molar_density * environment.bulkVolume, large);
        state.set_amount(*mixture.catalog, M::gas_two, Phase::Dispersed,
            initial.second_gas_fraction * initial.micro_molar_density * environment.bulkVolume, micro);
        state.validate(*mixture.catalog);
        return state;
    };
    result.initial_population_numbers = [initial](const R::ReactionContext& context)
    {
        return std::vector<double>{initial.micro_number_density * context.environment.bulkVolume,
            initial.large_number_density * context.environment.bulkVolume};
    };
    const auto supply = [power, feed_coefficient](const R::ReactionContext& context, double dt)
    {
        const double q = power ? power(context) : 0;
        M::require(std::isfinite(q) && std::isfinite(dt) && dt > 0, "Invalid fixture heat/time measure.");
        // Positive dose drives this explicitly supplied feed law; cooling supplies no material.
        return q > 0 ? feed_coefficient * q * dt * context.environment.bulkVolume : 0;
    };
    result.species_source = [supply](const R::ReactionContext& context, double dt)
    {
        R::ReactionSourceInput input;
        const double amount = supply(context, dt);
        if (amount > 0)
            input.externalTransfers.push_back(
                {"test.external_dose_correlated_feed", gas_id, Phase::Dissolved, amount, .002 * amount, 0, {}, {}});
        return input;
    };
    result.population_process = [supply](const BubbleGasPhysics::MultispeciesPopulationContext& context, double dt)
    {
        BubbleGasPhysics::MultispeciesPopulationPacket packet;
        const double amount = supply(context.chemistry, dt);
        if (amount > 0)
        {
            // Explicit synthetic nucleation law with fixed supplied radius; it is not inferred from endpoint moles.
            constexpr double radius = 6.6359547089482127e-8, sigma = .07;
            const auto& environment = context.chemistry.environment;
            const double mol_per_bubble = 4 * std::numbers::pi * radius * radius * radius / 3 *
                                          (environment.absolutePressure + 2 * sigma / radius) /
                                          (M::gas_constant * environment.temperature);
            packet.transfers.push_back({gas_id, Phase::Dissolved, Phase::Dispersed, {}, micro, amount});
            packet.numberEvents.push_back({micro, amount / mol_per_bubble, 0});
        }
        return packet;
    };
    result.surface_tension = .07;
    result.min_radius = 1e-12;
    result.max_radius = .1;
    return result;
}

/** Manufactured positive two-species film oracle. Its equilibrium concentrations
 * make each exact supplied D/r film stage return a known finite dissolution rate.
 * Radius is independently inverted from the explicit ideal-mixture/Laplace law;
 * no production bubble-radius or film helper is reused by this oracle. */
inline ReactingGasOptions film_options(Initial initial)
{
    initial.dissolved_concentration = 0;
    initial.second_gas_fraction = 1;
    auto result = options(initial);
    result.species_source = {};
    result.population_process = {};
    result.mass_transfer_mode = BubbleMassTransferMode::SphericalDiffusion;
    for (std::size_t component = 0; component < result.species_properties.size(); ++component)
    {
        auto& property = result.species_properties[component];
        const double diffusivity = component == 0 ? 1e-8 : 2e-8;
        const double prescribed_rate = component == 0 ? 1e-7 : 2e-7; // mol/(m3 bulk s), out of gas.
        const auto species = property.species;
        property.diffusivity = diffusivity;
        property.equilibrium_concentration = [initial, species, diffusivity, prescribed_rate](
                                                 const R::ReactionContext& context, const R::MixtureThermodynamicPoint&)
        {
            const auto& e = context.environment;
            const double count = initial.micro_number_density * e.bulkVolume;
            double mol = 0;
            for (const auto& definition : context.catalog.species())
                mol += context.state.amount(context.catalog, definition.id, Phase::Dispersed, micro);
            M::require(count > 0 && mol > 0 && e.liquidVolume > 0, "Film oracle requires occupied donors.");
            const double rhs = 3 * mol * M::gas_constant * e.temperature / (4 * std::numbers::pi * count);
            double lo = 0, hi = .1;
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                const double r = std::midpoint(lo, hi);
                if (e.absolutePressure * r * r * r + 2 * .07 * r * r < rhs)
                    lo = r;
                else
                    hi = r;
            }
            const double radius = std::midpoint(lo, hi);
            const double coefficient = diffusivity / radius;
            const double conductance = coefficient * 4 * std::numbers::pi * radius * radius * count;
            const double concentration =
                context.state.amount(context.catalog, species, Phase::Dissolved) / e.liquidVolume;
            return concentration + prescribed_rate * e.bulkVolume / conductance;
        };
    }
    return result;
}

/** Read-only view of an explicitly configured external heat driver. The current
 * owned row is used only within one immutable source invocation frame. */
template<class Field> Power prescribed_power(const Field* field)
{
    return [field](const R::ReactionContext& context)
    {
        if (!field || !context.environment.location || !context.environment.location->ownedCell)
            throw std::logic_error("Dose-correlated fixture requires its actual source location/field.");
        return field->value(static_cast<typename Field::local_ordinal_type>(*context.environment.location->ownedCell));
    };
}

template<class Model> const R::SpeciesInventoryHistory& history(const Model& model)
{
    const auto& all = model.species_histories();
    const auto found = std::find_if(all.begin(), all.end(), [](const auto& h) { return h.species == gas_id; });
    if (found == all.end())
        throw std::logic_error("Fixture gas history is absent.");
    return *found;
}
template<class Model> double supplied(const Model& model)
{
    return history(model).external_supply;
}
template<class Model> double escaped(const Model& model)
{
    return history(model).physical_outflow;
}
template<class Model> double inventory_error(const Model& model)
{
    return history(model).inventory_error;
}
template<class Model> double step_escaped(const Model& model)
{
    double amount = 0;
    for (const auto& event : model.physical_escape_events())
        if (event.species == gas_id && (event.phase == Phase::Dispersed || event.phase == Phase::Resolved))
            amount += event.moles;
    return amount;
}
template<class Model> double compartment(const Model& model, Phase phase, std::string_view population = {})
{
    const auto& field = model.species_field(gas_id, phase, population);
    double local = 0, global = 0;
    for (std::size_t i = 0; i < field.mesh().num_owned_cells(); ++i)
        local += field.value(static_cast<typename Model::local_ordinal_type>(i)) *
                 field.mesh().cell_volume(static_cast<typename Model::local_ordinal_type>(i));
    Teuchos::reduceAll(*field.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
} // namespace SimpleFluid::Testing::SurfaceGas
