/** @file SpeciesBubbleProcesses.hh
 * @brief Composable supplied-law processes for typed, finite bubble populations.
 */
#pragma once

#include "equations/BubblePopulationKinetics.hh"

namespace SimpleFluid::BubbleGasPhysics
{

using PopulationProcess =
    std::function<MultispeciesPopulationPacket(const MultispeciesPopulationContext&, double)>;
/** A supplied nonnegative local rate, evaluated against the frozen source-stage state. */
using PopulationRate = std::function<double(const MultispeciesPopulationContext&)>;

struct SpeciesMolarRate
{
    Reacting::SpeciesId species = 0;
    PopulationRate rate; ///< mol/(m3 bulk s).
};

struct SpeciesFilmLaw
{
    Reacting::SpeciesId species = 0;
    std::function<double(const MultispeciesPopulationContext&, const PopulationCondition&)> coefficient; ///< m/s.
    std::function<double(const MultispeciesPopulationContext&, const PopulationCondition&)>
        equilibrium_concentration; ///< mol/m3 actual liquid, using the supplied mixture fugacities.
};

struct PopulationFilmLaws
{
    std::string population;
    std::vector<SpeciesFilmLaw> species;
};

/** All processes see the SAME state; concatenate their original events without
 * sequential incoming credits. The field adapter or advance_multispecies_population_exchange
 * performs the single atomic joint donor reservation. Laws must be pure/reentrant,
 * MPI-free and own their validity checks. No constitutive law or chemistry is inferred.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess compose_population_processes(
    std::vector<PopulationProcess> processes);

/** Dissolved->named Dispersed transfers and their physical number births.
 * Rates use bulk volume exactly once. Positive births and gas amounts are paired;
 * number_rate is count/(m3 bulk s), independently supplied by the physical birth law.
 * Chemistry/external generation must already have supplied the finite dissolved donor.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_nucleation_process(
    std::string population, std::vector<SpeciesMolarRate> species, PopulationRate number_rate);

/** Explicit count events for an independently owned material source or a declared
 * complete dissolution. Rates are count/(m3 bulk s). This helper creates no gas;
 * the material/count consistency and gross original-number budget remain mandatory.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_population_number_process(
    std::string population, PopulationRate birth_rate = {}, PopulationRate death_rate = {});

/** Supplied first-order dissolution: fraction=-expm1(-frequency*dt).
 * Every species returns to its own Dissolved compartment; proportional number
 * deaths belong to this declared dissolution operator. Frequency is 1/s.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_population_dissolution_process(
    std::string population, PopulationRate frequency);

/** Supplied first-order population conversion, preserving every species and count.
 * The receiver keeps the converted physical count; no preferred species or size
 * pivot is introduced. Frequency is 1/s. Donor and receiver must be distinct.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_population_conversion_process(
    std::string donor, std::string receiver, PopulationRate frequency);

/** Signed film request dt*kL*4*pi*R^2*N*(n_dissolved/V_liquid-Ceq).
 * Radius/count and full-mixture properties come from the supplied context.
 * Positive requests consume Dissolved; negative requests consume named Dispersed.
 * Film alone never changes count, including exact depletion: a separate explicit
 * number-death owner is then required. No clipping, implicit donor rescaling,
 * enthalpy or chemical heat is introduced. The caller must declare its thermal
 * ownership or approximation; these helpers establish no phase-energy owner.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_film_transfer_process(
    std::string population, std::vector<SpeciesFilmLaw> species);

/** Backward Euler film exchange, sharing one end-of-stage dissolved inventory
 * per species across all supplied populations. Liquid volume, area and supplied
 * laws are evaluated at the frozen stage state. Each signed film request becomes
 * a paired transfer of the same species. The executor still reserves gross
 * requests against the original donors of the joint packet. Film emits no count
 * or energy events. Exact population depletion needs an explicit count owner.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT PopulationProcess make_implicit_film_transfer_process(
    std::vector<PopulationFilmLaws> populations);

} // namespace SimpleFluid::BubbleGasPhysics
