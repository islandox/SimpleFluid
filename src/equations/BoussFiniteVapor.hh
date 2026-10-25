/** @file BoussFiniteVapor.hh
 * @brief Finite material admission for an explicitly prescribed rho_ref*cp*T thermal owner.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "equations/BubblePopulationKinetics.hh"

#include <functional>
#include <memory>
#include <span>

namespace SimpleFluid::Reacting
{
/** This is a scalar-volume approximation, not a mixture EOS or phase-U owner.
 * Additive gas molar volumes are supplied in catalog order. Forbidden gas species
 * have zero entries. The liquid reference density and thermal rho_ref/cp are fixed.
 * Carried sensible enthalpy remains a trace; only declared heat below debits cp*T.
 */
struct BoussReferenceVolume
{
    std::string identity;
    std::uint64_t revision = 0, schemaFingerprint = 0;
    double liquidDensity = 0, thermalDensity = 0, heatCapacity = 0;
    std::vector<double> gasMolarVolumes; ///< m3/mol, independent of temperature and pressure.
};

/** Requests, unlike actual ReactionSourceInput extents, explicitly authorize
 * ordered capacity admission in the returned order, with joint donor reservation
 * across all requests and vapor (no admitted incoming credit). Only forward,
 * gas-volume-producing mechanism reactions are eligible. No external reservoirs
 * or hidden heat source are admitted through this request channel.
 */
struct BoussGasAdmissionLaw
{
    std::string identity;
    std::uint64_t revision = 0, schemaFingerprint = 0;
    std::function<std::vector<IntegratedReactionExtent>(const ReactionContext&, double)> request;
};

enum class BoussVaporAdmissionPolicy
{
    ProportionalRequests ///< One declared scale for competing bulk and wall proposals; actual chemistry is unscaled.
};
struct BoussWallHeatBinding
{
    std::string patch;
    double inwardHeatFlux = 0; ///< W/m2; must match the actual temperature owner's immutable trial flux snapshot.
    double evaporationFraction = 0;
};
struct BoussWallHeatPacket
{
    std::size_t wall = 0;
    std::int64_t geometryFace = 0;
    long double inwardEnergy = 0; ///< Full BC heat J, never added again as a volumetric source.
};
/** Host-certified per-cell wall input. Pure laws perform no mesh operations.
 * The field/controller validate actual producer identity, epoch, interval and
 * immutable source configuration before executing this budget.
 */
struct BoussWallHeatBudget
{
    const void* producer = nullptr;
    std::uint64_t geometryEpoch = 0, configurationSignature = 0;
    double intervalEnd = 0, duration = 0;
    std::vector<BoussWallHeatPacket> packets;
};
struct BoussEvaporationAdmission
{
    bool wall = false;
    std::string patch;
    std::int64_t geometryFace = 0;
    double requestedMoles = 0, admittedMoles = 0;
    long double suppliedWallEnergy = 0, allocatedWallEnergy = 0;
};
struct BoussVaporBinding
{
    SpeciesId solvent = 0; ///< One real catalog species; liquid and vapor are the same material.
    SpeciesPhase vaporPhase = SpeciesPhase::Resolved;
    std::string population; ///< Required exactly for named Dispersed vapor.
    double saturationTemperature = 0, latentHeat = 0, timeScale = 0;
    /** Supplied signed J/kg of evaporation, reversed for condensation.
     * This is an explicit scalar-storage correction, not an automatic material-h debit.
     * The resulting net evaporation cooling latentHeat-sensibleCorrection must be positive.
     */
    double sensibleCorrection = 0;
    bool bulkEnabled = true;
    BoussVaporAdmissionPolicy admissionPolicy = BoussVaporAdmissionPolicy::ProportionalRequests;
    std::vector<BoussWallHeatBinding> walls;
};

struct BoussAdmissionReceipt
{
    ReactionId reaction = 0;
    double requestedMoles = 0, admittedMoles = 0;
};
struct BoussVaporReceipt
{
    double requestedEvaporation = 0, requestedCondensation = 0; ///< mol/cell.
    double evaporation = 0, condensation = 0;                   ///< Admitted mol/cell.
    long double latentEnergy = 0, sensibleEnergy = 0;           ///< Signed J into scalar thermal owner.
    std::vector<long double> thermalContributions; ///< Raw accepted event heats, before reporting aggregation.
    std::vector<BoussEvaporationAdmission> evaporationAdmissions; ///< Raw origins; totals above are reporting views.
};

struct BoussAcceptedMaterialContext
{
    const ReactionContext& transported;
    const CellSpeciesState& acceptedMaterial;
    std::span<const BubbleGasPhysics::PopulationNumberState> transportedNumbers;
    std::span<const BoussAdmissionReceipt> gas;
    const BoussVaporReceipt& vapor;
};

/** Reference-volume process context deliberately contains no EOS/fugacity/caloric point. */
struct BoussPopulationCondition
{
    std::string population;
    double number = 0, radius = 0, volume = 0, density = 0;
};
struct BoussPopulationContext
{
    const ReactionContext& chemistry;
    std::span<const BoussPopulationCondition> populations;
};
/** Immutable shared source configuration. Replacing it does not reset inventories.
 * Optional laws may be enabled in either order. Every dispersed birth/death is
 * supplied explicitly after accepted material admission; no automatic count inference.
 */
struct BoussCoordinatedSources
{
    std::string identity;
    std::uint64_t revision = 0, schemaFingerprint = 0;
    std::optional<BoussGasAdmissionLaw> gas;
    std::optional<BoussVaporBinding> vapor;
    std::function<BubbleGasPhysics::MultispeciesPopulationPacket(const BoussPopulationContext&, double)>
        populationProcess; ///< Paired two-/multi-population process; no EOS/heat channel. Bound solvent phase change is
                           ///< excluded.

    std::function<std::vector<BubbleGasPhysics::PopulationNumberEvent>(const BoussAcceptedMaterialContext&)>
        populationEvents;
};

struct BoussAcceptedReceipt
{
    std::vector<BoussAdmissionReceipt> gas;
    BoussVaporReceipt vapor;
    std::vector<BubbleGasPhysics::PopulationNumberEvent> numberEvents;
    std::uint64_t configurationSignature = 0;
    std::vector<long double>
        thermalContributions; ///< Original accepted stage channels, never gross-report reconstruction.
};
struct BoussSourceAdvance
{
    ReactionAdvance chemistry; ///< Actual chemistry plus explicitly admitted gas-request chemistry.
    CellSpeciesState state;    ///< After the additional paired solvent/vapor transfer.
    std::vector<BubbleGasPhysics::PopulationNumberState> numbers;
    std::vector<BubbleGasPhysics::SpeciesPhaseTransfer> transfers;
    std::vector<BubbleGasPhysics::PopulationNumberEvent> numberEvents;
    std::vector<BoussAdmissionReceipt> admissions;
    BoussVaporReceipt vapor;
    std::vector<long double> thermalContributions;
    long double thermalEnergy = 0; ///< Reporting sum of original accepted applied-energy channels.
};

/** Validation and algebra are MPI-free. A distributed owner fences failures and
 * verifies identical immutable metadata before any callback.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT std::uint64_t bouss_configuration_signature(
    const BoussReferenceVolume&, const BoussCoordinatedSources&);
SIMPLEFLUID_EQUATIONS_EXPORT void validate_bouss_reference(const ReactionMechanism&, const BoussReferenceVolume&);
SIMPLEFLUID_EQUATIONS_EXPORT void validate_bouss_sources(
    const ReactionMechanism&, const BoussReferenceVolume&, const BoussCoordinatedSources&);
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT double bouss_gas_volume(
    const SpeciesCatalog&, const BoussReferenceVolume&, const CellSpeciesState&);
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT double bouss_liquid_volume(
    const SpeciesCatalog&, const BoussReferenceVolume&, const CellSpeciesState&);
/** Actual source executes first and rejects overcapacity without rescaling. Gas
 * requests then reserve remaining finite stocks jointly/capacity in request order; vapor uses
 * the remainder and the available sensible superheat/subcooling. Accepted source
 * heating reduces condensation's available subcooling without changing its request.
 * Each stage is atomic. Admission rounds toward zero to a representable admissible extent;
 * resulting endpoint/receipt representability is still checked, never clipped.
 */
/** Same explicit finite-count/capacity rules for subsequent paired micro/large
 * process packets. Emptying a bin requires a declared death event, never auto-collapse.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT BubbleGasPhysics::MultispeciesPopulationResult
advance_bouss_population_exchange(const SpeciesCatalog&, const BoussReferenceVolume&, const CellSpeciesState&,
    std::span<const BubbleGasPhysics::PopulationNumberState>, const ReactionEnvironment&, double alphaMaximum,
    const BubbleGasPhysics::MultispeciesPopulationPacket&);
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT BoussSourceAdvance advance_bouss_sources(const ReactionMechanism&,
    const BoussReferenceVolume&, const BoussCoordinatedSources&, const CellSpeciesState&,
    std::span<const BubbleGasPhysics::PopulationNumberState>, const ReactionEnvironment&, double duration,
    double alphaMaximum, const ReactionSourceInput&, const ReactionAdvanceOptions& = {},
    const BoussWallHeatBudget* wallHeat = nullptr);
} // namespace SimpleFluid::Reacting
