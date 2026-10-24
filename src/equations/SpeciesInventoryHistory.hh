/** @file SpeciesInventoryHistory.hh
 * @brief Catalog-keyed accepted source histories and distinct physical escape traces.
 */
#pragma once
#include "equations/ReactionMechanism.hh"
#include "utils/CompensatedSum.hh"
#include <array>
#include <string>
#include <vector>

namespace SimpleFluid::Reacting
{
/** Global molar inventory report. Chemical production/consumption use each reaction's
 * net coefficient for this species across compartments; phase relocation alone
 * cannot create or destroy species. Public doubles are rounded views of retained
 * compensated local cumulative counters.
 */
struct SpeciesInventoryHistory
{
    SpeciesId species = 0;
    double initial_moles = 0, current_moles = 0;
    double chemical_production = 0, chemical_consumption = 0;
    double external_supply = 0, external_withdrawal = 0;
    double physical_outflow = 0, inventory_error = 0;
    bool operator==(const SpeciesInventoryHistory&) const = default;
};

/** Global accepted-step gross molar events, never differences of cumulative totals.
 * Chemistry, declared external reservoirs, internal interphase transfer and actual
 * transport escape retain separate provenance. Keys include every configured
 * compartment; no species or dispersed-population identity is reserved.
 */
struct SpeciesCompartmentEvents
{
    SpeciesId species = 0;
    SpeciesPhase phase = SpeciesPhase::Liquid;
    std::string population;
    double chemical_production = 0, chemical_consumption = 0;
    double external_supply = 0, external_withdrawal = 0;
    double interphase_inflow = 0, interphase_outflow = 0;
    double physical_outflow = 0;
    bool operator==(const SpeciesCompartmentEvents&) const = default;
};

enum class PhysicalEscapeEnergyContract
{
    SuppliedPartialMolarEnthalpyTrace
};

/** Actual transport outflow, positive mol/mass outward; enthalpy and momentum
 * retain signed components in the declared caloric reference. Selective-species
 * energy requires the supplied partial-molar enthalpy closure. It is never
 * inferred by multiplying a bulk mixture enthalpy by one species molar mass.
 *
 * The Boussinesq field adapter does not own extensive phase energy. Its supplied
 * carried-enthalpy packet is therefore a trace, NOT a claim that this energy was
 * conservatively removed from an owned phase-U field. A controller requiring
 * that debit must own and apply it exactly once, or reject that coupling.
 * Carrier momentum uses the borrowed continuous velocity; it excludes bubble slip
 * and is not a debit of independently owned dispersed-phase momentum.
 * External reservoir withdrawal is not a physical escape event.
 */
struct SpeciesPhysicalEscapeEvent
{
    SpeciesId species = 0;
    SpeciesPhase phase = SpeciesPhase::Liquid;
    std::string population;
    double moles = 0, mass = 0, carried_enthalpy_trace = 0;
    std::array<double, 3> carrier_momentum_trace{};
    EnergyConvention energy_convention = EnergyConvention::SensibleEnergy;
    PhysicalEscapeEnergyContract energy_contract = PhysicalEscapeEnergyContract::SuppliedPartialMolarEnthalpyTrace;
    bool phase_energy_debited = false, phase_momentum_debited = false;
    bool operator==(const SpeciesPhysicalEscapeEvent&) const = default;
};

namespace history_detail
{
using Sum = ::SimpleFluid::detail::CompensatedSum<long double>;
enum EventMetric : std::size_t
{
    ChemicalProduction,
    ChemicalConsumption,
    ExternalSupply,
    ExternalWithdrawal,
    InterphaseInflow,
    InterphaseOutflow,
    PhysicalOutflow,
    EventMetricCount
};
enum CumulativeMetric : std::size_t
{
    Produced,
    Consumed,
    Supplied,
    Withdrawn,
    Escaped,
    CumulativeMetricCount
};
/** Fixed schema-sized stage scratch; no array is proportional to mesh cell count. */
struct CompartmentSums
{
    std::array<Sum, EventMetricCount> amount;
    Sum escaped_mass, escaped_enthalpy;
    std::array<Sum, 3> escaped_momentum;
};
/** Owned by one model and copied in full by its transaction snapshot. The
 * compensated local counters, not rounded MPI diagnostics, seed later steps.
 */
struct State
{
    bool initialized = false;
    std::vector<Sum> initial_local, net_local; ///< Net events retain cancellation independently of gross reports.
    std::vector<std::array<Sum, CumulativeMetricCount>> cumulative_local;
    std::vector<SpeciesInventoryHistory> inventory;
    std::vector<SpeciesCompartmentEvents> step;
    std::vector<SpeciesPhysicalEscapeEvent> escape;
};
} // namespace history_detail
} // namespace SimpleFluid::Reacting
