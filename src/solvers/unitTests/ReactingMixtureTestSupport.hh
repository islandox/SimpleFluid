/** @file ReactingMixtureTestSupport.hh
 * @brief Test-only manufactured finite-reactant/two-gas mixture; no physical material calibration.
 */
#pragma once

#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasProperties.hh"
#include "equations/ReactingThermodynamics.hh"
#include "equations/SpeciesInventoryHistory.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace SimpleFluid::Testing::ReactingMixture
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
    options.mode = ReactingGasMode::Species;
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
    options.solution_transport = ReactingTransportMode::Advective;
    options.bubble_transport = BubbleTransportMode::General;
    options.mass_transfer_mode = BubbleMassTransferMode::SphericalDiffusion;
    options.surface_tension = 0;
    options.min_radius = 1e-10;
    options.max_radius = .1;
    return options;
}

/** Compare every canonical compartment, including named mixed populations. */
inline void expect_same_species_state(const R::CellSpeciesState& actual, const R::CellSpeciesState& expected)
{
    EXPECT_EQ(actual.schemaRevision, expected.schemaRevision);
    EXPECT_EQ(actual.schemaFingerprint, expected.schemaFingerprint);
    EXPECT_EQ(actual.amounts, expected.amounts);
    EXPECT_EQ(actual.dispersedPopulations.size(), expected.dispersedPopulations.size());
    const auto count = std::min(actual.dispersedPopulations.size(), expected.dispersedPopulations.size());
    for (std::size_t p = 0; p < count; ++p)
    {
        SCOPED_TRACE(expected.dispersedPopulations[p].id);
        EXPECT_EQ(actual.dispersedPopulations[p].id, expected.dispersedPopulations[p].id);
        EXPECT_EQ(actual.dispersedPopulations[p].amounts, expected.dispersedPopulations[p].amounts);
    }
}

/** The accepted ledger includes individual reservoirs, not only net species/heat. */
inline void expect_same_ledger(const R::SpeciesSourceLedger& actual, const R::SpeciesSourceLedger& expected)
{
    expect_same_species_state(actual.chemicalChange, expected.chemicalChange);
    expect_same_species_state(actual.externalChange, expected.externalChange);
    EXPECT_EQ(actual.reactionExtents, expected.reactionExtents);
    EXPECT_EQ(actual.forwardReactionExtents, expected.forwardReactionExtents);
    EXPECT_EQ(actual.reverseReactionExtents, expected.reverseReactionExtents);
    EXPECT_EQ(actual.externalTransfers.size(), expected.externalTransfers.size());
    const auto count = std::min(actual.externalTransfers.size(), expected.externalTransfers.size());
    for (std::size_t event = 0; event < count; ++event)
    {
        SCOPED_TRACE(event);
        const auto& a = actual.externalTransfers[event];
        const auto& b = expected.externalTransfers[event];
        EXPECT_EQ(a.reservoir, b.reservoir);
        EXPECT_EQ(a.species, b.species);
        EXPECT_EQ(a.phase, b.phase);
        EXPECT_EQ(a.population, b.population);
        EXPECT_DOUBLE_EQ(a.moles, b.moles);
        EXPECT_DOUBLE_EQ(a.mass, b.mass);
        EXPECT_DOUBLE_EQ(a.energy, b.energy);
        EXPECT_EQ(a.momentum, b.momentum);
    }
    EXPECT_DOUBLE_EQ(actual.externalHeat, expected.externalHeat);
    EXPECT_DOUBLE_EQ(actual.reactionHeat, expected.reactionHeat);
    EXPECT_DOUBLE_EQ(actual.transferEnergy, expected.transferEnergy);
    EXPECT_DOUBLE_EQ(actual.externalMass, expected.externalMass);
    EXPECT_EQ(actual.externalMomentum, expected.externalMomentum);
    EXPECT_EQ(actual.substeps, expected.substeps);
    EXPECT_EQ(actual.retries, expected.retries);
    EXPECT_DOUBLE_EQ(actual.thermal_energy(), expected.thermal_energy());
}

inline void expect_same_ledgers(
    const std::vector<R::SpeciesSourceLedger>& actual, const std::vector<R::SpeciesSourceLedger>& expected)
{
    EXPECT_EQ(actual.size(), expected.size());
    for (std::size_t cell = 0; cell < std::min(actual.size(), expected.size()); ++cell)
    {
        SCOPED_TRACE(cell);
        expect_same_ledger(actual[cell], expected[cell]);
    }
}

/** A local-only accepted-state witness. Empty owners retain empty arrays and
 * perform no differently ordered collective calls. Private model generations
 * remain outside this public-API witness; public histories/events and the accepted pressure are included.
 * A surface-dependent caller additionally requires its event-receipt API explicitly.
 */
template<class Model> class SavedReactingState
{
public:
    explicit SavedReactingState(const Model& model)
        : fingerprint_(model.options().reaction_mechanism->catalog().fingerprint()),
          populations_(model.options().population_ids), initialized_(model.initial_state_initialized()),
          sources_(model.source_ledgers()), population_events_(model.population_ledgers()),
          escapes_(model.escape_ledgers()), histories_(model.species_histories()),
          step_events_(model.species_step_events()), physical_escapes_(model.physical_escape_events()),
          pressure_offset_(model.absolute_pressure_offset())
    {
        const auto& mesh = model.alpha_g().mesh();
        for (std::size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
            species_.push_back(model.cell_species_state(static_cast<typename Model::local_ordinal_type>(cell)));
        for (const auto& name : populations_)
        {
            auto& counts = numbers_.emplace_back();
            const auto& field = model.population_number_field(name);
            for (std::size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
                counts.push_back(field.value(static_cast<typename Model::local_ordinal_type>(cell)) *
                                 mesh.cell_volume(static_cast<typename Model::local_ordinal_type>(cell)));
        }
        for (const auto& [name, field] : model.output_fields())
            output_.emplace(name, field_values(*field));
        if constexpr (requires { model.species_event_receipt(); })
        {
            const auto receipt = model.species_event_receipt();
            receipt_check_ = [receipt](const Model& current)
            { EXPECT_EQ(current.species_event_receipt(), receipt); };
        }
        thermal_ = field_values(model.thermal_source());
        volume_source_ = field_values(model.material_volume_source());
        liquid_mass_ = field_values(model.liquid_mass_density());
        liquid_density_ = field_values(model.pure_liquid_density());
    }

    void expect_restored(const Model& model) const
    {
        EXPECT_EQ(model.options().reaction_mechanism->catalog().fingerprint(), fingerprint_);
        EXPECT_EQ(model.options().population_ids, populations_);
        EXPECT_EQ(model.initial_state_initialized(), initialized_);
        const auto& mesh = model.alpha_g().mesh();
        EXPECT_EQ(mesh.num_owned_cells(), species_.size());
        for (std::size_t cell = 0; cell < std::min(mesh.num_owned_cells(), species_.size()); ++cell)
            expect_same_species_state(
                model.cell_species_state(static_cast<typename Model::local_ordinal_type>(cell)), species_[cell]);
        if (model.options().population_ids == populations_)
            for (std::size_t p = 0; p < populations_.size(); ++p)
            {
                const auto& field = model.population_number_field(populations_[p]);
                for (std::size_t cell = 0; cell < std::min(mesh.num_owned_cells(), numbers_[p].size()); ++cell)
                    EXPECT_DOUBLE_EQ(field.value(static_cast<typename Model::local_ordinal_type>(cell)) *
                                         mesh.cell_volume(static_cast<typename Model::local_ordinal_type>(cell)),
                        numbers_[p][cell]);
            }
        EXPECT_EQ(model.output_fields().size(), output_.size());
        for (const auto& [name, expected] : output_)
        {
            const auto found = model.output_fields().find(name);
            EXPECT_NE(found, model.output_fields().end());
            if (found != model.output_fields().end())
                EXPECT_EQ(field_values(*found->second), expected) << name;
        }
        EXPECT_EQ(field_values(model.thermal_source()), thermal_);
        EXPECT_EQ(field_values(model.material_volume_source()), volume_source_);
        EXPECT_EQ(field_values(model.liquid_mass_density()), liquid_mass_);
        EXPECT_EQ(field_values(model.pure_liquid_density()), liquid_density_);
        expect_same_ledgers(model.source_ledgers(), sources_);
        expect_same_ledgers(model.population_ledgers(), population_events_);
        expect_same_ledgers(model.escape_ledgers(), escapes_);
        EXPECT_EQ(model.species_histories(), histories_);
        EXPECT_EQ(model.species_step_events(), step_events_);
        EXPECT_EQ(model.physical_escape_events(), physical_escapes_);
        EXPECT_DOUBLE_EQ(model.absolute_pressure_offset(), pressure_offset_);
        if (receipt_check_)
            receipt_check_(model);
    }

private:
    template<class Field> static std::vector<double> field_values(const Field& field)
    {
        std::vector<double> result;
        result.reserve(field.num_owned_cells());
        for (std::size_t cell = 0; cell < field.num_owned_cells(); ++cell)
            result.push_back(field.value(static_cast<typename Model::local_ordinal_type>(cell)));
        return result;
    }

    std::uint64_t fingerprint_;
    std::vector<std::string> populations_;
    bool initialized_;
    std::vector<R::CellSpeciesState> species_;
    std::vector<std::vector<double>> numbers_;
    std::map<std::string, std::vector<double>> output_;
    std::vector<double> thermal_, volume_source_, liquid_mass_, liquid_density_;
    std::vector<R::SpeciesSourceLedger> sources_, population_events_, escapes_;
    std::vector<R::SpeciesInventoryHistory> histories_;
    std::vector<R::SpeciesCompartmentEvents> step_events_;
    std::vector<R::SpeciesPhysicalEscapeEvent> physical_escapes_;
    double pressure_offset_;
    std::function<void(const Model&)> receipt_check_;
};

} // namespace SimpleFluid::Testing::ReactingMixture
