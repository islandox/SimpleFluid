/** @file GeneralGasModelTestSupport.hh
 * @brief Explicit physical inputs for general native/legacy field-model tests.
 * No enabled null-mechanism path, preferred species slot or implicit chemical yield.
 */
#pragma once
#include "ReactingMixtureTestSupport.hh"
#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid::Testing::GeneralGasModel
{
namespace R = Reacting;
namespace M = ReactingMixture;
namespace B = BubbleGasPhysics;
using Phase = R::SpeciesPhase;
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
inline constexpr auto one = M::gas_one, two = M::gas_two, solvent = M::solvent;
inline constexpr double gas_constant = 8.31446261815324;
struct Parameters
{
    double pressure = 100000, temperature = 300, density = 1000, viscosity = .001, sigma = .07;
    double molar_mass_one = .002, molar_mass_two = .004;
    double dissolved_one = 0, dissolved_two = 0, reactant_concentration = 0, product_concentration = 0;
    std::array<double, 2> number_density{}, molar_one{}, molar_two{};
    std::array<double, 2> diffusivity{}, henry{1e-5, 2e-5};
    std::array<double, 2> reservoir_mol_per_joule{};
    double independent_heat_rate = 0;
    std::optional<double> micro_decay_time;
    double micro_conversion_frequency = 0;
    bool nucleate_supplied_gas = false;
    std::string nucleation_population = "micro";
    double nucleation_radius = 6.6359547089482127e-8;
    ReactingGasOptions controls;
    std::function<R::ReactionSourceInput(const R::ReactionContext&, double)> source;
    std::function<B::MultispeciesPopulationPacket(const B::MultispeciesPopulationContext&, double)> population;
    std::function<double(const R::ReactionContext&)> diffusivity_one, diffusivity_two, nucleation_radius_law;
    std::function<double(double)> viscosity_temperature;
};
inline Parameters populated()
{
    Parameters result;
    result.dissolved_one = .2;
    result.dissolved_two = .1;
    result.number_density = {1e10, 2e8};
    result.molar_one = {2e-6, 6e-7};
    result.molar_two = {1e-6, 3e-7};
    result.controls.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    result.controls.constant_slip_velocity = .4;
    result.controls.free_surface_patches = {"zmax"};
    return result;
}
/** Independent ideal-mixture/Laplace radius oracle. Source amount n and count N
 * are extensive. This does not call production bubble/EOS inversion helpers. */
inline double radius(double n, double number, double temperature, double pressure, double sigma)
{
    M::require(std::isfinite(n) && std::isfinite(number) && n >= 0 && number >= 0 && ((n == 0) == (number == 0)),
        "Oracle gas/count pair is invalid.");
    if (n == 0)
        return 0;
    const long double rhs =
        3 * static_cast<long double>(n) * gas_constant * temperature / (4 * std::numbers::pi_v<long double> * number);
    long double lo = 0, hi = .1;
    M::require(pressure * hi * hi * hi + 2 * sigma * hi * hi >= rhs, "Oracle radius exceeds declared domain.");
    for (int i = 0; i < 160; ++i)
    {
        const auto r = std::midpoint(lo, hi);
        if (pressure * r * r * r + 2 * sigma * r * r < rhs)
            lo = r;
        else
            hi = r;
    }
    return static_cast<double>(std::midpoint(lo, hi));
}
inline double volume(double n, double number, double temperature, double pressure, double sigma)
{
    const double r = radius(n, number, temperature, pressure, sigma);
    return number * 4 * std::numbers::pi * r * r * r / 3;
}
inline double nucleation_moles(const Parameters& p, const R::ReactionContext& context)
{
    const auto& environment = context.environment;
    const double r = p.nucleation_radius_law ? p.nucleation_radius_law(context) : p.nucleation_radius;
    M::require(std::isfinite(r) && r > 0, "Supplied nucleation radius is invalid.");
    return 4 * std::numbers::pi / 3 * (environment.absolutePressure * r * r * r + 2 * p.sigma * r * r) /
           (gas_constant * environment.temperature);
}
template<class Power> ReactingGasOptions options(const Parameters& p, const Power* power = nullptr)
{
    M::require(!p.micro_decay_time || (std::isfinite(*p.micro_decay_time) && *p.micro_decay_time > 0),
        "Supplied micro decay time must be finite and positive.");
    M::require(std::isfinite(p.micro_conversion_frequency) && p.micro_conversion_frequency >= 0,
        "Supplied conversion frequency must be finite and nonnegative.");
    M::Mixture mixture(false, false, R::EnergyConvention::SensibleEnergy, 0);
    if (p.molar_mass_one != .002 || p.molar_mass_two != .004)
    {
        auto definitions = mixture.catalog->species();
        for (auto& species : definitions)
            if (species.id == one || species.id == two)
            {
                species.molarMass = species.id == one ? p.molar_mass_one : p.molar_mass_two;
                species.name = species.id == one ? "supplied ordinary gas one" : "supplied ordinary gas two";
                species.elements = {{species.id == one ? "G1" : "G2", 1}};
            }
        mixture.catalog = std::make_shared<const R::SpeciesCatalog>(std::move(definitions), 8);
        mixture.mechanism = std::make_shared<const R::ReactionMechanism>(
            mixture.catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
    }
    const R::ThermodynamicBounds bounds{250, 600, 1, 3e7};
    std::vector<R::IdealGasCaloricSpecies> caloric;
    for (const auto& species : mixture.catalog->species())
        caloric.push_back({M::molar_heat_capacity(species.id), 300, 100000 * species.molarMass});
    mixture.gas = R::make_ideal_gas_mixture(
        mixture.catalog, std::move(caloric), bounds, R::EnergyConvention::SensibleEnergy, gas_constant);
    mixture.liquid = R::make_linear_compressible_mixture_liquid(
        mixture.catalog, bounds, [p](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, p.pressure, p.density, 4200, 0, 0, 0, p.viscosity, 0}; },
        R::EnergyConvention::SensibleEnergy);
    if (p.viscosity_temperature)
    {
        const auto base = mixture.liquid;
        mixture.liquid = std::make_shared<const R::MixtureThermodynamics>(
            mixture.catalog, bounds, R::EOSKind::Liquid,
            [base, law = p.viscosity_temperature](double T, double pressure, std::span<const double> x)
            {
                auto point = base->evaluate(T, pressure, x);
                point.dynamic_viscosity = law(T);
                return point;
            },
            R::EnergyConvention::SensibleEnergy);
    }
    auto result = p.controls;
    result.mode = ReactingGasMode::Species;
    result.reference_pressure = p.pressure;
    result.reaction_mechanism = mixture.mechanism;
    result.gas_thermodynamics = mixture.gas;
    result.liquid_thermodynamics = mixture.liquid;
    result.population_ids = {"micro", "large"};
    result.species_properties = {
        mixture.transport(one, p.diffusivity[0], p.henry[0]), mixture.transport(two, p.diffusivity[1], p.henry[1])};
    if (p.diffusivity_one)
        result.species_properties[0].diffusivity_provider = p.diffusivity_one;
    if (p.diffusivity_two)
        result.species_properties[1].diffusivity_provider = p.diffusivity_two;
    for (const auto& species : mixture.catalog->species())
        if (species.id != one && species.id != two)
        {
            ReactingSpeciesProperties property;
            property.species = species.id;
            property.carried_molar_enthalpy = [mixture, id = species.id](const R::ReactionContext& c, Phase phase,
                                                  std::string_view) { return mixture.molar_enthalpy(c, id, phase); };
            result.species_properties.push_back(std::move(property));
        }
    result.surface_tension = p.sigma;
    result.min_radius = 1e-12;
    result.max_radius = 1e-3;
    result.solution_transport = ReactingTransportMode::Advective;
    result.bubble_transport = BubbleTransportMode::General;
    result.initial_species_state = [p, catalog = mixture.catalog](const R::ReactionEnvironment& e, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(catalog->species().size())},
            {"large", std::vector<double>(catalog->species().size())}};
        double liquid = e.bulkVolume;
        for (std::size_t bin = 0; bin < 2; ++bin)
        {
            const std::string name = bin == 0 ? "micro" : "large";
            const double n1 = p.molar_one[bin] * e.bulkVolume, n2 = p.molar_two[bin] * e.bulkVolume;
            liquid -= volume(n1 + n2, p.number_density[bin] * e.bulkVolume, e.temperature, e.absolutePressure, p.sigma);
            state.set_amount(*catalog, one, Phase::Dispersed, n1, name);
            state.set_amount(*catalog, two, Phase::Dispersed, n2, name);
        }
        M::require(liquid >= 0, "Declared finite gas volume exceeds its control volume.");
        const double n1 = p.dissolved_one * liquid, n2 = p.dissolved_two * liquid;
        const double reactant = p.reactant_concentration * liquid, product = p.product_concentration * liquid;
        const double remaining =
            p.density * liquid - p.molar_mass_one * n1 - p.molar_mass_two * n2 - .010 * reactant - .004 * product;
        M::require(remaining >= 0, "Declared dissolved stocks exceed actual finite solution mass.");
        state.set_amount(*catalog, solvent, Phase::Liquid, remaining / .018);
        state.set_amount(*catalog, M::reactant, Phase::Liquid, reactant);
        state.set_amount(*catalog, M::product, Phase::Liquid, product);
        state.set_amount(*catalog, one, Phase::Dissolved, n1);
        state.set_amount(*catalog, two, Phase::Dissolved, n2);
        state.validate(*catalog);
        return state;
    };
    result.initial_population_numbers = [p](const R::ReactionContext& c)
    {
        return std::vector<double>{
            p.number_density[0] * c.environment.bulkVolume, p.number_density[1] * c.environment.bulkVolume};
    };
    const auto supplied = [p, power](const R::ReactionContext& c, double dt, std::size_t component)
    {
        if (!power)
            return 0.;
        M::require(c.environment.location && c.environment.location->ownedCell,
            "A field-supplied reservoir law requires its exact owned source frame.");
        const double q = power->value(static_cast<LO>(*c.environment.location->ownedCell));
        M::require(std::isfinite(q), "External fixture power is nonfinite.");
        return q > 0 ? p.reservoir_mol_per_joule[component] * q * dt * c.environment.bulkVolume : 0.;
    };
    result.species_source = p.source ? p.source : [p, supplied](const R::ReactionContext& c, double dt)
    {
        R::ReactionSourceInput input;
        for (std::size_t component = 0; component < 2; ++component)
        {
            const double n = supplied(c, dt, component);
            const bool small_loss = p.micro_decay_time && static_cast<long double>(dt) / *p.micro_decay_time < .5L;
            if (n > 0)
                input.externalTransfers.push_back({"test.declared_external_gas_feed", component == 0 ? one : two,
                    small_loss && p.nucleate_supplied_gas ? Phase::Dispersed : Phase::Dissolved, n,
                    (component == 0 ? p.molar_mass_one : p.molar_mass_two) * n, 0, {},
                    small_loss && p.nucleate_supplied_gas ? p.nucleation_population : ""});
        }
        input.heat.rate = R::VolumetricHeatRate{p.independent_heat_rate, R::SourceVolumeBasis::BulkVolume};
        return input;
    };
    result.population_process = p.population;
    if (!result.population_process &&
        (p.nucleate_supplied_gas || p.micro_decay_time || p.micro_conversion_frequency > 0))
        result.population_process = [p, supplied](const B::MultispeciesPopulationContext& c, double dt)
        {
            B::MultispeciesPopulationPacket packet;
            const auto found = std::find_if(c.populations.begin(), c.populations.end(),
                [&](const auto& population) { return population.id == p.nucleation_population; });
            M::require(found != c.populations.end(), "Configured supplied population is absent.");
            const long double x = p.micro_decay_time ? static_cast<long double>(dt) / *p.micro_decay_time : 0;
            const long double retention = std::exp(-x), loss = -std::expm1(-x);
            const long double fraction = x == 0 ? 1 : -std::expm1(-x) / x;
            const double conversion =
                std::exp(-static_cast<double>(x)) * -std::expm1(-p.micro_conversion_frequency * dt);
            const double old_count = found->number;
            long double births = 0;
            const double mol_per_birth = nucleation_moles(p, c.chemistry);
            for (std::size_t component = 0; component < 2; ++component)
            {
                const auto species = component == 0 ? one : two;
                const double source = p.nucleate_supplied_gas ? supplied(c.chemistry, dt, component) : 0;
                const bool small_loss = p.micro_decay_time && x < .5L;
                const long double original =
                    c.chemistry.state.amount(c.chemistry.catalog, species, Phase::Dispersed, p.nucleation_population) -
                    (small_loss ? static_cast<long double>(source) : 0.L);
                M::require(original >= 0, "External feed partition exceeded its actual supplied stage stock.");
                if (loss > 0 && original > 0)
                    packet.transfers.push_back({species, Phase::Dispersed, Phase::Dissolved, p.nucleation_population,
                        {}, static_cast<double>(original * loss)});
                if (conversion > 0 && original > 0)
                    packet.transfers.push_back({species, Phase::Dispersed, Phase::Dispersed, "micro", "large",
                        static_cast<double>(original * conversion)});
                const long double added = static_cast<long double>(source) * fraction;
                if (small_loss)
                {
                    // Retain the independently small positive continuous-source decay channel.
                    // Avoid subtracting a rounded near-total retained birth from full production.
                    const long double loss_fraction =
                        x < 1e-3L ? x / 2 - x * x / 6 + x * x * x / 24 - x * x * x * x / 120 : 1 - fraction;
                    const long double decayed = static_cast<long double>(source) * loss_fraction;
                    const double narrowed = static_cast<double>(decayed);
                    M::require(decayed == 0 || narrowed > 0, "Supplied source decay is not representable.");
                    if (narrowed > 0)
                        packet.transfers.push_back(
                            {species, Phase::Dispersed, Phase::Dissolved, "micro", {}, narrowed});
                }
                else
                {
                    const double narrowed = static_cast<double>(added);
                    M::require(added == 0 || narrowed > 0,
                        "Supplied gas birth cannot be represented without losing material.");
                    if (narrowed > 0)
                        packet.transfers.push_back(
                            {species, Phase::Dissolved, Phase::Dispersed, {}, p.nucleation_population, narrowed});
                }
                births += added / mol_per_birth;
            }
            packet.numberEvents.push_back(
                {p.nucleation_population, static_cast<double>(births), static_cast<double>(old_count * loss)});
            if (conversion > 0)
            {
                packet.numberEvents.push_back({"micro", 0, old_count * conversion});
                packet.numberEvents.push_back({"large", old_count * conversion, 0});
            }
            return packet;
        };
    return result;
}
inline ReactingGasOptions options(const Parameters& p = {})
{
    using Field = ScalarCellFieldStored<Pack>;
    return options<Field>(p, nullptr);
}
template<class Grid> struct Fixture
{
    using Model = ReactingGasModel<Pack, Grid>;
    using Field = typename Model::field_type;
    using Velocity = typename Model::velocity_field_type;
    using Flux = typename Model::face_flux_field_type;
    SP<Grid> grid;
    Parameters parameters;
    Field temperature, pressure, power;
    Velocity velocity;
    Flux flux;
    typename Model::material_type material;
    Model model;
    Fixture(SP<Grid> mesh, Parameters config = {})
        : grid(std::move(mesh)), parameters(std::move(config)),
          temperature(grid, parameters.temperature, "temperature"), pressure(grid, 0., "pressure"),
          power(grid, 0., "independent_power"), velocity(grid, typename Grid::Vec3{}, "velocity"),
          flux(grid, 0., "flux"), material(
                                      grid,
                                      [&]
                                      {
                                          BoussinesqModelOptions p;
                                          p.reference_density = p.density = parameters.density;
                                          p.specific_heat_capacity = 4200;
                                          p.dynamic_viscosity = parameters.viscosity;
                                          p.thermal_conductivity = 0;
                                          return p;
                                      }(),
                                      TimeStepperOptions{}),
          model(grid, options(parameters, &power))
    {
    }
    void initialize() { model.initialize_state(0, temperature, pressure, velocity, material); }
    void advance(double time, double dt, const FVM::ALEControlVolumeState* ale = nullptr)
    {
        model.advance(time, dt, temperature, pressure, velocity, flux, material, ale);
    }
};
/** Clear only geometry/numeric transport state. Full configure() changes physical
 * initialization authority and must never be used as a cache-reset shortcut.
 * This exact witness verifies that numeric invalidation publishes no new state. */
template<class Model> void force_numeric_reassembly(Model& model)
{
    const M::SavedReactingState<Model> state(model);
    const auto histories = model.species_histories();
    const auto step_events = model.species_step_events();
    const auto subset_events = model.species_subset_source_changes();
    const auto escapes = model.physical_escape_events();
    const auto species_event = model.species_event_receipt();
    const auto number_event = model.population_escape_receipt();
    const auto liquid_event = model.liquid_transport_volume_receipt();
    const auto pressure_offset = model.absolute_pressure_offset();
    const auto local_outputs = [&model]
    {
        std::map<std::string, std::vector<double>> result;
        for (const auto& [name, field] : model.output_fields())
            for (std::size_t row = 0; row < model.alpha_g().mesh().num_local_cells(); ++row)
                result[name].push_back(field->local_value(static_cast<typename Model::local_ordinal_type>(row)));
        return result;
    };
    const auto face_values = [](const auto& field)
    {
        std::vector<std::pair<typename Model::local_ordinal_type, double>> values;
        for (const auto face : field.owned_face_ids())
            values.emplace_back(face, field.value(face));
        return values;
    };
    const auto outputs = local_outputs();
    const auto liquid_packet = face_values(model.transported_liquid_material_volume_flux());
    const auto carrier_packet = face_values(model.transported_bubble_carrier_volume_flux());
    const auto slip_packet = face_values(model.transported_bubble_slip_volume_flux());
    model.refresh_geometry();
    state.expect_restored(model);
    EXPECT_EQ(model.species_histories(), histories);
    EXPECT_EQ(model.species_step_events(), step_events);
    EXPECT_EQ(model.species_subset_source_changes(), subset_events);
    EXPECT_EQ(model.physical_escape_events(), escapes);
    EXPECT_EQ(model.species_event_receipt(), species_event);
    EXPECT_EQ(model.population_escape_receipt(), number_event);
    EXPECT_EQ(model.liquid_transport_volume_receipt(), liquid_event);
    EXPECT_DOUBLE_EQ(model.absolute_pressure_offset(), pressure_offset);
    EXPECT_EQ(local_outputs(), outputs);
    EXPECT_EQ(face_values(model.transported_liquid_material_volume_flux()), liquid_packet);
    EXPECT_EQ(face_values(model.transported_bubble_carrier_volume_flux()), carrier_packet);
    EXPECT_EQ(face_values(model.transported_bubble_slip_volume_flux()), slip_packet);
}

template<class Model> const R::SpeciesInventoryHistory& history(const Model& model, R::SpeciesId species = one)
{
    const auto& values = model.species_histories();
    const auto found =
        std::find_if(values.begin(), values.end(), [&](const auto& value) { return value.species == species; });
    if (found == values.end())
        throw std::logic_error("Expected catalogued accepted history is absent.");
    return *found;
}
template<class Field> double integral(const Field& field)
{
    detail::CompensatedSum<long double> local;
    for (std::size_t i = 0; i < field.mesh().num_owned_cells(); ++i)
        local += static_cast<long double>(field.value(static_cast<LO>(i))) *
                 field.mesh().cell_volume(static_cast<LO>(i));
    const long double local_value = local.value();
    // Two double channels carry the high part and local narrowing residual.
    const double high = static_cast<double>(local_value);
    const std::array<double, 2> wire{high, static_cast<double>(local_value - static_cast<long double>(high))};
    std::array<double, 2> global{};
    Teuchos::reduceAll(*field.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, wire.data(), global.data());
    detail::CompensatedSum<long double> result;
    result += global[0];
    result += global[1];
    return static_cast<double>(result.value());
}
template<class Model> double compartment(const Model& model, R::SpeciesId id, Phase phase, std::string_view pop = {})
{
    return integral(model.species_field(id, phase, pop));
}
template<class Model> double step_supply(const Model& model, R::SpeciesId id = one)
{
    double total = 0;
    for (const auto& event : model.species_step_events())
        if (event.species == id)
            total += event.external_supply;
    return total;
}
template<class Model> double step_escape(const Model& model, R::SpeciesId id = one, bool bubble_only = false)
{
    double total = 0;
    for (const auto& event : model.physical_escape_events())
        if (event.species == id && (!bubble_only || event.phase == Phase::Dispersed || event.phase == Phase::Resolved))
            total += event.moles;
    return total;
}

struct FilmObservation
{
    double radius = 0, coefficient = 0, transfer = 0, gas_before = 0, gas_after = 0, count_before = 0, count_after = 0;
};
template<class Grid>
std::vector<FilmObservation> advance_film(
    Fixture<Grid>& f, double end, double dt, std::string_view bin = "large", R::SpeciesId species = one)
{
    std::vector<FilmObservation> result(f.grid->num_owned_cells());
    std::vector<double> denominator(result.size());
    const auto& catalog = f.model.options().reaction_mechanism->catalog();
    for (std::size_t row = 0; row < result.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const double cv = f.grid->cell_volume(cell);
        const auto state = f.model.cell_species_state(cell, cv);
        auto& value = result[row];
        value.gas_before = state.amount(catalog, species, Phase::Dispersed, bin);
        value.count_before = f.model.population_number_field(bin).value(cell) * cv;
        double total = 0;
        for (const auto& definition : catalog.species())
            total += state.amount(catalog, definition.id, Phase::Dispersed, bin);
        value.radius = radius(total, value.count_before, f.temperature.value(cell),
            f.model.absolute_pressure().value(cell), f.parameters.sigma);
        const double liquid = f.model.liquid_mass_density().value(cell) * cv / f.parameters.density;
        const double concentration = state.amount(catalog, species, Phase::Dissolved) / liquid;
        const double gas_pressure = f.model.absolute_pressure().value(cell) + 2 * f.parameters.sigma / value.radius;
        const double henry = f.parameters.henry[species == one ? 0 : 1];
        const double equilibrium = henry * gas_pressure * value.gas_before / total;
        denominator[row] = dt * 4 * std::numbers::pi * value.radius * value.radius * value.count_before *
                           (concentration - equilibrium);
        M::require(std::isfinite(denominator[row]) && denominator[row] != 0,
            "Film oracle needs finite nonzero driving force.");
    }
    f.advance(end, dt);
    for (std::size_t row = 0; row < result.size(); ++row)
    {
        auto& value = result[row];
        const auto cell = static_cast<LO>(row);
        const double cv = f.grid->cell_volume(cell);
        value.transfer =
            f.model.population_ledgers()[row].chemicalChange.amount(catalog, species, Phase::Dispersed, bin);
        value.coefficient = value.transfer / denominator[row];
        value.gas_after = f.model.species_field(species, Phase::Dispersed, bin).value(cell) * cv;
        value.count_after = f.model.population_number_field(bin).value(cell) * cv;
    }
    return result;
}
template<class Model> double number_escape(const Model& model, std::string_view bin)
{
    const auto& receipt = model.population_escape_receipt();
    M::require(
        receipt.event == model.species_event_receipt(), "Number escape must belong to this accepted species event.");
    const auto event = std::find_if(
        receipt.events.begin(), receipt.events.end(), [&](const auto& value) { return value.population == bin; });
    M::require(event != receipt.events.end(), "Number escape omitted a configured bin.");
    return event->number;
}
} // namespace SimpleFluid::Testing::GeneralGasModel
