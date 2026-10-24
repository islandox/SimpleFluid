/** @file
 * @brief Explicit hydrogen moment-reference adapter.
 * This reference has no finite-solvent, phase-energy, or inlet contract.
 */
#pragma once
#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasProperties.hh"
#include "equations/SpeciesInventoryHistory.hh"
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace SimpleFluid::Verification::HydrogenReference
{
namespace R = Reacting;
inline constexpr R::SpeciesId hydrogen = 211;
inline constexpr double universal_gas_constant = 8.31446261815324;
using Rate = std::function<double(const R::ReactionContext&)>; // Explicit mol/(m3 bulk s).
using Radius = std::function<double(double temperature, double pressure)>;

struct Parameters
{
    double temperature, pressure, molar_mass, gas_constant, surface_tension;
    double carrier_density, carrier_cp, carrier_viscosity, carrier_conductivity;
    double slip_velocity, initial_micro_moles_per_bulk_volume;
};
inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
inline double bubble_moles(const Parameters& p, double temperature, double pressure, double radius)
{
    require(std::isfinite(radius) && radius > 0, "Reference nucleation radius must be finite and positive.");
    const double value = 4 * std::numbers::pi * radius * radius * radius / 3 *
                         (pressure + 2 * p.surface_tension / radius) / (p.gas_constant * temperature);
    require(std::isfinite(value) && value > 0, "Reference bubble amount must be finite and positive.");
    return value;
}
inline ReactingGasOptions field_options(Parameters p, Rate production_rate, Radius nucleation_radius)
{
    using P = R::SpeciesPhase;
    require(p.gas_constant == universal_gas_constant && p.molar_mass > 0 && p.temperature > 0 && p.pressure > 0 &&
                p.surface_tension >= 0 && p.initial_micro_moles_per_bulk_volume >= 0 && production_rate &&
                nucleation_radius,
        "Explicit H2 reference parameters/laws are invalid.");
    auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{hydrogen, "H2", p.molar_mass, {{"H", 2}}, {P::Dispersed, P::Resolved}}});
    ReactingGasOptions result;
    result.mode = ReactingGasMode::Species;
    result.source_schedule = ReactingSourceSchedule::PrescribedCarrierAfterTemperature;
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
    const R::ThermodynamicBounds bounds{250, 600, 1, 1e8};
    // Only a supplied ideal-H2 trace law; the carrier owns cp*T and no gas-U debit is asserted.
    result.gas_thermodynamics =
        R::make_ideal_gas_mixture(catalog, {{20.4, p.temperature, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
    // Required typed provider, deliberately unused: this reference has no model-owned liquid amount.
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds,
        [p](std::span<const double>)
        {
            return R::LinearCompressibleMixtureLiquid{p.temperature, p.pressure, p.carrier_density, p.carrier_cp, 0,
                1e-9, 0, p.carrier_viscosity, p.carrier_conductivity};
        },
        R::EnergyConvention::SensibleEnergy);
    result.reference_pressure = p.pressure;
    result.surface_tension = p.surface_tension;
    result.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    result.constant_slip_velocity = p.slip_velocity;
    result.min_radius = 1e-12;
    result.max_radius = 1e-3;
    result.max_subcycles = 1;
    result.initial_species_state = [catalog, p](const R::ReactionEnvironment& context, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(1)}, {"large", std::vector<double>(1)}};
        state.set_amount(
            *catalog, hydrogen, P::Dispersed, p.initial_micro_moles_per_bulk_volume * context.bulkVolume, "micro");
        return state;
    };
    result.initial_population_numbers = [p, nucleation_radius](const R::ReactionContext& c)
    {
        const auto& e = c.environment;
        const double n_b =
            bubble_moles(p, e.temperature, e.absolutePressure, nucleation_radius(e.temperature, e.absolutePressure));
        return std::vector<double>{p.initial_micro_moles_per_bulk_volume * e.bulkVolume / n_b, 0};
    };
    result.species_source = [p, production_rate](const R::ReactionContext& c, double dt)
    {
        const double rate = production_rate(c);
        require(std::isfinite(rate) && rate >= 0, "Explicit reference H2 injection rate is invalid.");
        const double amount = rate * c.environment.bulkVolume * dt;
        R::ReactionSourceInput source;
        if (amount > 0)
            source.externalTransfers.push_back({"prescribed reference H2 feed", hydrogen, P::Dispersed, amount,
                p.molar_mass * amount, 0, {}, "micro"});
        // Mass is supplied by this named reservoir. Independent carrier heating is NOT copied here.
        return source;
    };
    result.population_process = [p, production_rate, nucleation_radius](
                                    const BubbleGasPhysics::MultispeciesPopulationContext& context, double dt)
    {
        const auto& c = context.chemistry;
        const auto& e = c.environment;
        const double amount = production_rate(c) * e.bulkVolume * dt;
        require(std::isfinite(amount) && amount >= 0, "Explicit reference bubble birth amount is invalid.");
        BubbleGasPhysics::MultispeciesPopulationPacket packet;
        if (amount > 0)
        {
            const double n_b = bubble_moles(
                p, e.temperature, e.absolutePressure, nucleation_radius(e.temperature, e.absolutePressure));
            packet.numberEvents.push_back({"micro", amount / n_b, 0});
        }
        return packet;
    };
    ReactingSpeciesProperties transport;
    transport.species = hydrogen;
    transport.carried_molar_enthalpy = [p](const R::ReactionContext& c, P phase, std::string_view)
    {
        require(phase == P::Dispersed || phase == P::Resolved, "H2 reference has no finite liquid or dissolved phase.");
        return 20.4 * (c.environment.temperature - p.temperature) + p.gas_constant * c.environment.temperature;
    };
    result.species_properties = {transport}; // No film, implicit dissolution or conversion law.
    result.free_surface_patches = {"zmax"};  // Escape only; no free-surface geometry/model owner.
    return result;
}
template<class Model> const R::SpeciesInventoryHistory& history(const Model& model)
{
    const auto& histories = model.species_histories();
    const auto found = std::ranges::find(histories, hydrogen, &R::SpeciesInventoryHistory::species);
    require(found != histories.end(), "The explicit H2 reference history is absent.");
    return *found;
}
template<class Model> double step_escape(const Model& model, std::string_view population)
{
    double result = 0;
    for (const auto& event : model.physical_escape_events())
        if (event.species == hydrogen && event.phase == R::SpeciesPhase::Dispersed && event.population == population)
            result += event.moles;
    return result; // Already globally reduced reporting receipt; never reduce it a second time.
}
} // namespace SimpleFluid::Verification::HydrogenReference
