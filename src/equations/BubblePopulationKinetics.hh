/** @file BubblePopulationKinetics.hh
 * @brief Pure conservative microbubble and fixed-pivot population reactions.
 */
#pragma once

#include "equations/ReactingGasProperties.hh"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace SimpleFluid::BubbleGasPhysics
{

/** All inventories and number densities are per total control-volume volume. */
struct MicrobubbleState
{
    real_t dissolved_inventory = 0.0;
    real_t number_density = 0.0;
    real_t molar_inventory = 0.0;
};

/** A physical birth or an explicitly returned unresolved population. */
struct BubblePopulationBirth
{
    real_t number_density = 0.0;
    real_t molar_inventory = 0.0;
};

/** Frozen coefficients preserve the legacy source/decay/conversion ordering. */
struct MicrobubbleSubstep
{
    real_t duration = 0.0;
    real_t bulk_production_rate = 0.0;   ///< mol/(m3 bulk s), already scaled by its source provider.
    real_t liquid_fraction = 1.0;        ///< Actual aqueous volume fraction, held fixed during this substep.
    real_t nucleation_moles = 0.0;       ///< mol per newly nucleated microbubble.
    real_t critical_concentration = 0.0; ///< mol/m3 aqueous liquid.
    real_t micro_decay_fraction = 0.0;
    real_t micro_retention = 1.0;
    real_t conversion_pressure_factor = 0.0; ///< Original conversion coefficient times liquid pressure.
    ReactingHeavisideMode heaviside_mode = ReactingHeavisideMode::Exact;
    real_t smooth_heaviside_width = 0.0;
};

struct MicrobubbleSubstepResult
{
    MicrobubbleState state;
    BubblePopulationBirth converted;
    real_t number_change = 0.0; ///< Event sum, retained even when the stored population rounds unchanged.
    real_t molar_change = 0.0;  ///< Production minus physical decay and conversion, without endpoint subtraction.
    real_t produced_moles = 0.0;
    real_t dissolved_moles = 0.0;
    real_t decayed_number = 0.0;
    real_t inventory_error = 0.0;
    real_t number_error = 0.0;
};

/**
 * Preserve the existing source, analytic micro decay, then supersaturation
 * conversion arithmetic. This function supplies no material correlation,
 * upper-population growth law, clipping, transport, or MPI side effect.
 * Callers explicitly own the chosen source law and subcycle schedule.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT MicrobubbleSubstepResult advance_microbubble_substep(
    const MicrobubbleState& initial, const MicrobubbleSubstep& step);

/** Strictly increasing positive gas species moles moles per representative bubble. */
struct MolarPivotGrid
{
    std::vector<real_t> mol_per_bubble;
};

/**
 * Fixed pivots have one canonical number field per bin. Bin moles are
 * mol_per_bubble[i]*number_density[i], avoiding inconsistent independent
 * number/molar fields. Both inventories are per total control-volume volume,
 * including cells containing resolved gas. Both moments are checked on return.
 */
struct UpperBubblePopulation
{
    real_t dissolved_inventory = 0.0;
    std::vector<real_t> number_density;
};

/**
 * Caller-supplied thermophysical state; solution-specific properties are external.
 * Zero values may mark unused material inputs. An enabled kernel's validator
 * must check every physical input required by its declared model.
 */
struct BubblePopulationEnvironment
{
    real_t time = 0.0;
    real_t temperature = 0.0;
    real_t liquid_pressure = 0.0;
    real_t liquid_fraction = 1.0;  ///< True aqueous fraction alpha_l.
    real_t carrier_fraction = 1.0; ///< Nonresolved host alpha_l+alpha_d; mathematically 1-f.
    real_t liquid_density = 0.0;
    real_t dynamic_viscosity = 0.0;
    real_t surface_tension = 0.0;
    real_t diffusivity = 0.0;
    real_t gas_constant = 8.31446261815324;
};

struct BubblePopulationContext
{
    const MolarPivotGrid& grid;
    const UpperBubblePopulation& state;
    const BubblePopulationEnvironment& environment;
    real_t dissolved_concentration = 0.0; ///< state inventory / physical aqueous fraction.
};

/** Expected daughter count at a positive molar size, per breakup event. */
struct BubbleBreakupDaughter
{
    real_t mol_per_bubble = 0.0;
    real_t multiplicity = 0.0;
};

/**
 * No empirical rate is built in. Any enabled rate requires a nonempty model
 * name, a declared validity description and a validator which must reject
 * states outside the caller's physical/calibrated domain. Callbacks are pure
 * and reentrant; they may capture immutable external material parameters.
 * Empty rates explicitly disable their processes.
 */
struct BubblePopulationKernels
{
    std::string model_name;
    std::string validity_description;
    std::function<void(const BubblePopulationContext&)> validate;
    /** Signed mol/(bubble s); positive growth consumes dissolved gas species moles. */
    std::function<real_t(std::size_t, const BubblePopulationContext&)> growth_rate;
    /** Physical symmetric collision kernel [m3/s], evaluated for i<=j.
     * Centers have host concentration N_i/carrier_fraction, so bulk event
     * density is K*N_i*N_j/carrier_fraction (halved for identical pairs).
     * The kernel must not include another carrier-volume normalization.
     */
    std::function<real_t(std::size_t, std::size_t, const BubblePopulationContext&)> coalescence_kernel;
    /** Nonnegative breakup frequency, 1/s. */
    std::function<real_t(std::size_t, const BubblePopulationContext&)> breakup_frequency;
    /** Daughters must conserve the parent's moles; expected count is at least one. */
    std::function<std::vector<BubbleBreakupDaughter>(std::size_t, const BubblePopulationContext&)> breakup_daughters;
};

enum class BubbleUnderflowPolicy
{
    Reject,
    ReturnToMicro ///< Return positive sub-grid molar sizes and their counts explicitly.
};

struct BubblePopulationStepOptions
{
    BubbleUnderflowPolicy underflow = BubbleUnderflowPolicy::Reject;
};

struct BubblePopulationMoments
{
    real_t number_density = 0.0;
    real_t molar_inventory = 0.0;
};

struct BubblePopulationStepResult
{
    UpperBubblePopulation state;
    std::vector<real_t> number_change; ///< Compensated per-bin birth/consumption event sums, per bulk volume.
    std::vector<real_t> molar_change;  ///< mol_per_bubble[i]*number_change[i]; never subtracts stored endpoints.
    BubblePopulationBirth returned_to_micro;
    real_t dissolved_to_gas_moles = 0.0;
    real_t gas_to_dissolved_moles = 0.0;
    real_t coalescence_events = 0.0;
    real_t breakup_events = 0.0;
    real_t breakup_created_number = 0.0; ///< Daughter count minus parent count.
    real_t dissolved_bubble_number = 0.0;
    real_t moles_before = 0.0;
    real_t moles_added_by_births = 0.0;
    real_t moles_after = 0.0; ///< Includes explicitly returned microbubble moles.
    real_t molar_error = 0.0;
    real_t number_error = 0.0;
};

[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT BubblePopulationMoments population_moments(
    const MolarPivotGrid& grid, const UpperBubblePopulation& state);

[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT std::vector<real_t> population_bin_moles(
    const MolarPivotGrid& grid, const UpperBubblePopulation& state);

/**
 * One explicit reaction substep: deposit supplied births, grow/dissolve,
 * coalesce, then break up. Rates use the state at the start of their stage;
 * simultaneous donors are budgeted before any stage is published.
 * A populated bin or positive birth requires a positive carrier fraction.
 * Fraction ordering allows floating-point roundoff only; values are never
 * normalized or clipped. Growth and breakup remain rates per bubble.
 *
 * Births and daughters are split between adjacent molar pivots preserving
 * their number and moles. Coalescence changes number by minus one per event;
 * breakup uses its declared daughter count. Exact zero-size dissolution
 * removes a bubble and returns its gas to dissolved inventory.
 *
 * Insufficient dissolved gas species moles, excessive donor-number consumption,
 * negative post-growth size, above-grid particles and disallowed underflow
 * throw without modifying the input. The caller must reduce the substep or
 * extend its physical size grid; no moles or number are silently clipped.
 * This kernel supplies no legacy large-bubble lifetime loss.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT BubblePopulationStepResult advance_upper_bubble_population(
    const MolarPivotGrid& grid, const UpperBubblePopulation& initial, const BubblePopulationEnvironment& environment,
    real_t duration, const BubblePopulationKernels& kernels = {}, const std::vector<BubblePopulationBirth>& births = {},
    BubblePopulationStepOptions options = {});

/** A conservative transfer of one chemical species between exact physical compartments. */
struct SpeciesPhaseTransfer
{
    Reacting::SpeciesId species = 0;
    Reacting::SpeciesPhase donor = Reacting::SpeciesPhase::Dissolved;
    Reacting::SpeciesPhase receiver = Reacting::SpeciesPhase::Dispersed;
    std::string donorPopulation, receiverPopulation;
    double moles = 0.0; ///< Nonnegative integrated mol per cell; reverse the endpoints for reverse transfer.
};

struct PopulationNumberState
{
    std::string population;
    double number = 0.0; ///< Extensive count per cell, independent of species amounts.
};

struct PopulationNumberEvent
{
    std::string population;
    double births = 0.0, deaths = 0.0; ///< Nonnegative integrated physical counts per cell.
};

struct PopulationCondition
{
    std::string id;
    double number = 0.0, radius = 0.0;
    Reacting::MixtureThermodynamicPoint properties; ///< Full mixture fugacities, never total pressure per species.
};

struct MultispeciesPopulationContext
{
    const Reacting::ReactionContext& chemistry;
    std::vector<PopulationCondition> populations;
};

/** Supplied nucleation, transfer, conversion, coalescence or breakup event packet.
 * Material enters only through paired same-species transfers; chemistry/external
 * reservoirs are owned by ReactionSourceInput. Count births require paired gas.
 */
struct MultispeciesPopulationPacket
{
    std::vector<SpeciesPhaseTransfer> transfers;
    std::vector<PopulationNumberEvent> numberEvents;
};

/** Local original-event sum, before reporting narrows into numberChanges.
 * Replay MUST append both components to a stock-seeded accumulator. The pair
 * is transient local data, not an MPI serialization or persistent state format.
 */
struct PopulationNumberChangeReceipt
{
    long double high = 0, correction = 0;
};

struct MultispeciesPopulationResult
{
    Reacting::CellSpeciesState state;
    std::vector<PopulationNumberState> numbers;
    Reacting::SpeciesSourceLedger ledger;
    std::vector<double> numberChanges; ///< Rounded reporting only; never replay into a stock.
    std::vector<PopulationNumberChangeReceipt> localNumberChanges; ///< Exact population order, finite components.
};

/** Atomic joint donor reservation, including competing populations and all gases.
 * Exact complete dissolution requires its declared operator's explicit number-death
 * event; arbitrary material disappearance never removes counts. A nonempty population must have positive count; empty
 * populations have zero count. No mass is created and no number/species inventory is clipped.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT MultispeciesPopulationResult advance_multispecies_population_exchange(
    const Reacting::SpeciesCatalog& catalog, const Reacting::CellSpeciesState& initial,
    const std::vector<PopulationNumberState>& numbers, const Reacting::ReactionEnvironment& environment,
    const MultispeciesPopulationPacket& packet);
} // namespace SimpleFluid::BubbleGasPhysics
