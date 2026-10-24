/** @file ReactingMixtureExampleSupport.hh
 * @brief Explicit synthetic reacting-mixture setup for examples; not a material correlation.
 */
#pragma once

#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasProperties.hh"
#include "equations/ReactingThermodynamics.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace SimpleFluid::Examples::ReactingMixture
{
namespace R = Reacting;
using Phase = R::SpeciesPhase;

inline constexpr R::SpeciesId solvent = 44, reactant = 15, product = 19, gas_one = 101, gas_two = 709;
inline constexpr R::ReactionId conversion = 83;
inline constexpr double gas_constant = 8.31446261815324, reference_temperature = 300, reference_pressure = 100000;
inline constexpr R::ThermodynamicBounds bounds{250, 500, 10000, 2000000};

inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

inline std::shared_ptr<const R::SpeciesCatalog> make_catalog(bool reordered = false)
{
    std::vector<R::SpeciesDefinition> definitions{
        {gas_two, "gas_two", .004, {{"X", 2}}, {Phase::Dissolved, Phase::Dispersed, Phase::Resolved}},
        {solvent, "solvent", .018, {{"S", 1}}, {Phase::Liquid, Phase::Dispersed, Phase::Resolved}},
        {reactant, "finite_reactant", .010, {{"X", 5}}, {Phase::Liquid}},
        {gas_one, "gas_one", .002, {{"X", 1}}, {Phase::Dissolved, Phase::Dispersed, Phase::Resolved}},
        {product, "finite_product", .004, {{"X", 2}}, {Phase::Liquid}}};
    if (reordered)
        std::rotate(definitions.begin(), definitions.begin() + 1, definitions.end());
    return std::make_shared<const R::SpeciesCatalog>(std::move(definitions), 7);
}

inline double covolume(R::SpeciesId species, bool nonideal)
{
    return nonideal ? (species == gas_two ? .003 : .001) : 0;
}
inline double molar_heat_capacity(R::SpeciesId species)
{
    return species == gas_two ? 40 : 20;
}

struct Mixture
{
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    bool nonideal = true;

    explicit Mixture(bool reordered = false, bool use_nonideal_gas = true,
        R::EnergyConvention convention = R::EnergyConvention::SensibleEnergy, double chemical_heat = 100)
        : catalog(make_catalog(reordered)), nonideal(use_nonideal_gas)
    {
        require(std::isfinite(chemical_heat), "Manufactured chemical heat must be finite.");
        require(convention == R::EnergyConvention::SensibleEnergy || chemical_heat == 0,
            "Formation-energy storage cannot also add chemical heat.");
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{{conversion, "finite_reactant_to_product_and_two_gases",
                {{reactant, Phase::Liquid, -1}, {product, Phase::Liquid, 1}, {gas_one, Phase::Dissolved, 1},
                    {gas_two, Phase::Dissolved, 1}},
                chemical_heat}},
            convention);
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, bounds,
            [](std::span<const double>)
            {
                return R::LinearCompressibleMixtureLiquid{
                    reference_temperature, reference_pressure, 1000, 4000, 2e-4, 5e-9, 100000, .001, .6};
            },
            convention);
        // Explicit excluded-volume mixture: v_m=R*T/p+sum(x_i*b_i).
        // The matching fugacity is f_i=x_i*p*exp(b_i*p/(R*T)).
        gas = std::make_shared<const R::MixtureThermodynamics>(
            catalog, bounds, R::EOSKind::Gas,
            [schema = catalog, nonideal = nonideal](double temperature, double pressure, std::span<const double> x)
            {
                double mass = 0, volume = gas_constant * temperature / pressure, energy = 0, cv = 0;
                R::MixtureThermodynamicPoint point;
                point.fugacities.resize(x.size());
                for (std::size_t i = 0; i < x.size(); ++i)
                {
                    const auto& species = schema->species()[i];
                    const double b = covolume(species.id, nonideal);
                    mass += x[i] * species.molarMass;
                    volume += x[i] * b;
                    cv += x[i] * molar_heat_capacity(species.id);
                    energy += x[i] * (100000 * species.molarMass +
                                         molar_heat_capacity(species.id) * (temperature - reference_temperature));
                    point.fugacities[i] = x[i] * pressure * std::exp(b * pressure / (gas_constant * temperature));
                }
                point.density = mass / volume;
                point.internal_energy = energy / mass;
                point.enthalpy = point.internal_energy + pressure / point.density;
                point.internal_energy_temperature = cv / mass;
                point.internal_energy_pressure = 0;
                point.density_temperature = -mass * gas_constant / (pressure * volume * volume);
                point.density_pressure = mass * gas_constant * temperature / (pressure * pressure * volume * volume);
                point.dynamic_viscosity = 1e-5;
                point.thermal_conductivity = .02;
                return point;
            },
            convention);
    }

    [[nodiscard]] double molar_enthalpy(const R::ReactionContext& context, R::SpeciesId species, Phase phase) const
    {
        require(context.catalog.fingerprint() == catalog->fingerprint(), "Foreign manufactured enthalpy schema.");
        const double temperature = context.environment.temperature, pressure = context.environment.absolutePressure;
        require(std::isfinite(temperature) && temperature >= bounds.minimum_temperature &&
                    temperature <= bounds.maximum_temperature && std::isfinite(pressure) &&
                    pressure >= bounds.minimum_pressure && pressure <= bounds.maximum_pressure,
            "Manufactured carried enthalpy is outside its supplied T/P domain.");
        if (phase == Phase::Liquid || phase == Phase::Dissolved)
        {
            std::vector<double> pure(catalog->species().size());
            pure[catalog->index(species)] = 1;
            return catalog->species()[catalog->index(species)].molarMass *
                   liquid->evaluate(temperature, pressure, pure).enthalpy;
        }
        return 100000 * catalog->species()[catalog->index(species)].molarMass +
               molar_heat_capacity(species) * (temperature - reference_temperature) + gas_constant * temperature +
               covolume(species, nonideal) * pressure;
    }

    /** A prescribed molar reaction rate; no heat or dose is used to derive it. */
    [[nodiscard]] R::ReactionRateProvider reaction_rate(
        double value, R::SourceVolumeBasis basis = R::SourceVolumeBasis::LiquidVolume) const
    {
        require(std::isfinite(value), "Manufactured reaction rate must be finite.");
        R::ReactionRateProvider provider;
        provider.identity = "manufactured.constant_molar_reaction";
        provider.revision = 1;
        provider.schemaRevision = catalog->revision();
        provider.schemaFingerprint = catalog->fingerprint();
        provider.validityDescription = "Synthetic finite-donor mixture within the declared T/P domain.";
        provider.parameterUnits = "mol reaction/(m3 of declared basis s)";
        provider.validate = [schema = catalog](const R::ReactionContext& context)
        {
            require(context.catalog.fingerprint() == schema->fingerprint(), "Foreign manufactured source schema.");
            require(context.environment.temperature >= bounds.minimum_temperature &&
                        context.environment.temperature <= bounds.maximum_temperature &&
                        context.environment.absolutePressure >= bounds.minimum_pressure &&
                        context.environment.absolutePressure <= bounds.maximum_pressure,
                "Manufactured source is outside its supplied T/P domain.");
        };
        provider.evaluate = [value, basis](const R::ReactionContext&)
        { return std::vector<R::ReactionRate>{{conversion, value, basis}}; };
        return provider;
    }

    [[nodiscard]] ReactingSpeciesProperties transport(R::SpeciesId species, double diffusivity, double henry) const
    {
        require(catalog->allows(species, Phase::Dissolved) && std::isfinite(diffusivity) && diffusivity >= 0 &&
                    std::isfinite(henry) && henry >= 0,
            "Invalid manufactured species transport.");
        ReactingSpeciesProperties properties;
        properties.species = species;
        properties.diffusivity = diffusivity;
        properties.henry_coefficient = henry;
        properties.carried_molar_enthalpy = [mixture = *this, species](
                                                const R::ReactionContext& context, Phase phase, std::string_view)
        { return mixture.molar_enthalpy(context, species, phase); };
        return properties;
    }
};

/** Independent external heat; deliberately unrelated to reaction_rate(). */
inline R::ThermalInput external_heat(
    double power_density, R::SourceVolumeBasis basis = R::SourceVolumeBasis::BulkVolume)
{
    require(std::isfinite(power_density), "Manufactured heat rate must be finite.");
    R::ThermalInput heat;
    heat.rate = R::VolumetricHeatRate{power_density, basis};
    return heat;
}

struct PopulationSeed
{
    std::string name;
    double number_density = 0;   ///< Number per bulk m3.
    double moles_per_bubble = 0; ///< Actual total mol per bubble.
    double gas_one_fraction = .25;
};
struct InitialMixture
{
    double reactant_concentration = .1, product_concentration = 0;
    double dissolved_one_concentration = .2, dissolved_two_concentration = .1; ///< mol/liquid m3.
    double resolved_volume_fraction = 0, resolved_gas_one_fraction = .3;
    std::vector<PopulationSeed> populations{{"micro"}, {"large"}};
};
struct InitialCell
{
    R::CellSpeciesState species;
    std::vector<double> population_numbers;
    double liquid_volume = 0;
};

/** Actual species/number amounts at supplied T/P; liquid solvent fills only
 * the physical volume left by the independently specified gas populations.
 * No amount is renormalized after construction, and no trace phase is clipped.
 */
inline InitialCell initial_cell(
    const Mixture& mixture, const R::ReactionEnvironment& environment, const InitialMixture& input = {})
{
    require(std::isfinite(environment.bulkVolume) && environment.bulkVolume > 0 &&
                input.resolved_volume_fraction >= 0 && input.resolved_volume_fraction <= 1 &&
                input.resolved_gas_one_fraction >= 0 && input.resolved_gas_one_fraction <= 1,
        "Invalid manufactured initial physical volume/composition.");
    const auto& catalog = *mixture.catalog;
    InitialCell result{R::CellSpeciesState(catalog), {}, environment.bulkVolume};
    std::vector<double> composition(catalog.species().size());
    for (const auto& seed : input.populations)
    {
        require(!seed.name.empty() && std::isfinite(seed.number_density) && seed.number_density >= 0 &&
                    std::isfinite(seed.moles_per_bubble) && seed.moles_per_bubble >= 0 && seed.gas_one_fraction >= 0 &&
                    seed.gas_one_fraction <= 1 && ((seed.number_density == 0) == (seed.moles_per_bubble == 0)),
            "Initial gas count and actual species must be paired.");
        result.species.dispersedPopulations.push_back({seed.name, std::vector<double>(catalog.species().size())});
        const double number = seed.number_density * environment.bulkVolume, moles = number * seed.moles_per_bubble;
        require(std::isfinite(number) && std::isfinite(moles) && ((number == 0) == (moles == 0)) &&
                    (seed.number_density == 0 || number > 0),
            "Initial gas count/species overflowed or cannot be represented without underflow.");
        result.population_numbers.push_back(number);
        result.species.set_amount(catalog, gas_one, Phase::Dispersed, seed.gas_one_fraction * moles, seed.name);
        result.species.set_amount(catalog, gas_two, Phase::Dispersed, (1 - seed.gas_one_fraction) * moles, seed.name);
        if (moles > 0)
        {
            std::fill(composition.begin(), composition.end(), 0);
            composition[catalog.index(gas_one)] = seed.gas_one_fraction;
            composition[catalog.index(gas_two)] = 1 - seed.gas_one_fraction;
            const auto point =
                mixture.gas->evaluate(environment.temperature, environment.absolutePressure, composition);
            const double mass = moles * (seed.gas_one_fraction * .002 + (1 - seed.gas_one_fraction) * .004);
            result.liquid_volume -= mass / point.density;
        }
    }
    const double resolved_volume = input.resolved_volume_fraction * environment.bulkVolume;
    if (resolved_volume > 0)
    {
        std::fill(composition.begin(), composition.end(), 0);
        composition[catalog.index(gas_one)] = input.resolved_gas_one_fraction;
        composition[catalog.index(gas_two)] = 1 - input.resolved_gas_one_fraction;
        const auto point = mixture.gas->evaluate(environment.temperature, environment.absolutePressure, composition);
        const double W = input.resolved_gas_one_fraction * .002 + (1 - input.resolved_gas_one_fraction) * .004;
        const double moles = resolved_volume * point.density / W;
        result.species.set_amount(catalog, gas_one, Phase::Resolved, input.resolved_gas_one_fraction * moles);
        result.species.set_amount(catalog, gas_two, Phase::Resolved, (1 - input.resolved_gas_one_fraction) * moles);
    }
    result.liquid_volume -= resolved_volume;
    require(result.liquid_volume >= 0, "Specified gas inventories exceed the physical cell volume.");
    double non_solvent_mass = 0;
    for (const auto& [species, phase, concentration] :
        std::vector<std::tuple<R::SpeciesId, Phase, double>>{{reactant, Phase::Liquid, input.reactant_concentration},
            {product, Phase::Liquid, input.product_concentration},
            {gas_one, Phase::Dissolved, input.dissolved_one_concentration},
            {gas_two, Phase::Dissolved, input.dissolved_two_concentration}})
    {
        require(std::isfinite(concentration) && concentration >= 0, "Invalid finite liquid concentration.");
        const double moles = concentration * result.liquid_volume;
        result.species.set_amount(catalog, species, phase, moles);
        non_solvent_mass += moles * catalog.species()[catalog.index(species)].molarMass;
    }
    if (result.liquid_volume > 0)
    {
        std::fill(composition.begin(), composition.end(), 0);
        composition[catalog.index(solvent)] = 1;
        const double mass =
            result.liquid_volume *
            mixture.liquid->evaluate(environment.temperature, environment.absolutePressure, composition).density;
        require(mass >= non_solvent_mass, "Declared finite solutes exceed actual liquid mass.");
        result.species.set_amount(catalog, solvent, Phase::Liquid, (mass - non_solvent_mass) / .018);
    }
    result.species.validate(catalog);
    return result;
}

/** The only defaults are explicitly declared synthetic laws, never a chemistry
 * fallback. A zero second-gas diffusivity disables only that film exchange.
 */
inline ReactingGasOptions field_options(const Mixture& mixture, const InitialMixture& seed = {},
    double molar_reaction_rate = 0, double independent_heat_rate = 0)
{
    ReactingGasOptions options;
    options.mode = ReactingGasMode::TwoPopulation;
    options.reference_pressure = reference_pressure;
    options.reaction_mechanism = mixture.mechanism;
    options.liquid_thermodynamics = mixture.liquid;
    options.gas_thermodynamics = mixture.gas;
    options.species_properties = {mixture.transport(gas_one, 1e-8, 1e-5), mixture.transport(gas_two, 0, 2e-5)};
    options.population_ids.clear();
    for (const auto& population : seed.populations)
        options.population_ids.push_back(population.name);
    options.initial_species_state = [mixture, seed](const R::ReactionEnvironment& environment, std::int64_t)
    { return initial_cell(mixture, environment, seed).species; };
    options.initial_population_numbers = [mixture, seed](const R::ReactionContext& context)
    { return initial_cell(mixture, context.environment, seed).population_numbers; };
    options.species_source = [provider = mixture.reaction_rate(molar_reaction_rate),
                                 heat = external_heat(independent_heat_rate)](const R::ReactionContext&, double)
    {
        R::ReactionSourceInput input;
        input.rates = provider;
        input.heat = heat;
        return input;
    };
    options.dissolved_transport = ReactingTransportMode::Advective;
    options.bubble_transport = BubbleTransportMode::General;
    options.mass_transfer_mode = BubbleMassTransferMode::SphericalDiffusion;
    options.surface_tension = 0;
    options.min_radius = 1e-10;
    options.max_radius = .1;
    return options;
}
} // namespace SimpleFluid::Examples::ReactingMixture
