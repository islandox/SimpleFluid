/** @file SpeciesSurfaceAccounting.hh
 * @brief receipt-bound gas-subset event accounting; no independent transport.
 */
#pragma once
#include "equations/SpeciesInventoryHistory.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace SimpleFluid::detail::surface_species
{
using Sum = ::SimpleFluid::detail::CompensatedSum<long double>;
SIMPLEFLUID_SOLVERS_LOCAL inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
SIMPLEFLUID_SOLVERS_LOCAL inline double narrow(long double value)
{
    const double stored = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(stored) && (value == 0 || stored != 0),
        "Species surface amount is not representable.");
    return stored;
}
struct SIMPLEFLUID_SOLVERS_LOCAL Accounting
{
    using StateSnapshot = Accounting;
    bool initialized = false;
    Reacting::SpeciesEventReceipt receipt;
    double accepted_time = 0;
    GasMolesBySpecies initial;
    GasMoleSums production, consumption, net_source, dissolved_outflow;
    [[nodiscard]] StateSnapshot snapshot() const { return *this; }
    void restore(const StateSnapshot& saved) { *this = saved; }

    void initialize(const Reacting::SpeciesCatalog& catalog, const Reacting::SpeciesEventReceipt& accepted,
        GasMolesBySpecies current_gas, double initial_time = 0)
    {
        require(accepted.schema_fingerprint == catalog.fingerprint(), "Species surface receipt has a foreign catalog.");
        *this = {};
        initialized = true;
        receipt = accepted;
        require(std::isfinite(initial_time), "Species surface initial clock is nonfinite.");
        accepted_time = initial_time;
        initial = std::move(current_gas);
        for (const auto& species : catalog.species())
        {
            require(initial.contains(species.name) && std::isfinite(initial.at(species.name)) &&
                        initial.at(species.name) >= 0,
                "Incomplete initial gas-subset inventory.");
            production[species.name] += 0;
            consumption[species.name] += 0;
            net_source[species.name] += 0;
            dissolved_outflow[species.name] += 0;
        }
    }
    void require_before_advance(const Reacting::SpeciesEventReceipt& accepted, double clock) const
    {
        require(initialized && accepted == receipt && clock == accepted_time,
            "Reacting model was advanced/reinitialized outside its free-surface controller.");
    }
    static GasMolesBySpecies report(const GasMoleSums& sums)
    {
        GasMolesBySpecies result;
        for (const auto& [name, sum] : sums)
            result[name] = narrow(sum.value());
        return result;
    }

    /** Caller fences this pure local preparation before the next mesh collective. */
    Accounting preview(const Reacting::SpeciesCatalog& catalog, const Reacting::SpeciesEventReceipt& accepted,
        double interval_end, double dt, std::span<const Reacting::SpeciesCompartmentEvents> events,
        std::span<const Reacting::SpeciesSubsetSourceChange> subsets,
        std::span<const Reacting::SpeciesPhysicalEscapeEvent> escape, GasMolesBySpecies& gas_escape,
        double& liquid_source_mass, double& liquid_outflow_mass) const
    {
        require(
            initialized && receipt.revision != std::numeric_limits<std::uint64_t>::max() && accepted.accepted &&
                accepted.revision == receipt.revision + 1 && accepted.publication_nonce > receipt.publication_nonce &&
                accepted.initialization_generation == receipt.initialization_generation &&
                accepted.schema_fingerprint == catalog.fingerprint() &&
                accepted.schema_fingerprint == receipt.schema_fingerprint && accepted.interval_end == interval_end &&
                accepted.time_step == dt && std::isfinite(dt) && dt > 0 && interval_end == accepted_time + dt,
            "Free surface must consume exactly the next accepted model event interval.");
        Accounting result = *this;
        Sum mass_change, mass_outflow;
        for (const auto& species : catalog.species())
            gas_escape[species.name] = 0;
        for (const auto& event : events)
        {
            const auto& definition = catalog.species().at(catalog.index(event.species));
            if (event.phase == Reacting::SpeciesPhase::Liquid)
                continue;
            result.production[definition.name] += event.chemical_production;
            result.production[definition.name] += event.external_supply;
            result.production[definition.name] += event.interphase_inflow;
            result.consumption[definition.name] += event.chemical_consumption;
            result.consumption[definition.name] += event.external_withdrawal;
            result.consumption[definition.name] += event.interphase_outflow;
        }
        require(subsets.size() == catalog.species().size(), "Missing signed accepted subset source budget.");
        for (std::size_t s = 0; s < subsets.size(); ++s)
        {
            const auto& definition = catalog.species()[s];
            const auto& source = subsets[s];
            require(source.species == definition.id && std::isfinite(source.gas_source_moles) &&
                        std::isfinite(source.solution_source_moles),
                "Foreign signed source subset schema.");
            result.net_source[definition.name] += source.gas_source_moles;
            mass_change += static_cast<long double>(source.solution_source_moles) * definition.molarMass;
        }
        GasMoleSums escaped;
        for (const auto& event : escape)
        {
            const auto& definition = catalog.species().at(catalog.index(event.species));
            require(std::isfinite(event.moles) && event.moles >= 0 && std::isfinite(event.mass) && event.mass >= 0,
                "Invalid accepted physical escape.");
            if (event.phase == Reacting::SpeciesPhase::Dissolved || event.phase == Reacting::SpeciesPhase::Liquid)
                mass_outflow += event.mass;
            if (event.phase == Reacting::SpeciesPhase::Dissolved)
                result.dissolved_outflow[definition.name] += event.moles;
            else if (event.phase == Reacting::SpeciesPhase::Resolved ||
                     event.phase == Reacting::SpeciesPhase::Dispersed)
                escaped[definition.name] += event.moles;
            // No heat/momentum debit is invented from Boussinesq carried traces.
        }
        for (const auto& [name, amount] : escaped)
            gas_escape[name] = narrow(amount.value());
        liquid_source_mass = narrow(mass_change.value());
        liquid_outflow_mass = narrow(mass_outflow.value());
        result.receipt = accepted;
        result.accepted_time = interval_end;
        return result;
    }
};
} // namespace SimpleFluid::detail::surface_species
