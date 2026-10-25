/** @file ReactionMechanism.hh
 * @brief Species-resolved, conservative local chemistry without mesh or MPI ownership.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid::Reacting
{
using SpeciesId = std::uint32_t;
using ReactionId = std::uint32_t;

enum class SpeciesPhase : std::size_t
{
    Liquid,
    Dissolved,
    Dispersed,
    Resolved,
    Count
};

struct SpeciesDefinition
{
    SpeciesId id = 0;
    std::string name;
    double molarMass = 0.0;                 ///< kg/mol.
    std::map<std::string, double> elements; ///< Element atom counts per molecule.
    std::vector<SpeciesPhase> allowedPhases;
};

/** Immutable after construction; IDs and names are unique, independent of vector ordering. */
class SIMPLEFLUID_EQUATIONS_EXPORT SpeciesCatalog
{
public:
    explicit SpeciesCatalog(std::vector<SpeciesDefinition> species, std::uint64_t schemaRevision = 1);
    [[nodiscard]] const std::vector<SpeciesDefinition>& species() const noexcept;
    [[nodiscard]] std::size_t index(SpeciesId id) const;
    [[nodiscard]] std::uint64_t revision() const noexcept;
    /** Deterministic ordered schema signature, including all species metadata and revision. */
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] bool allows(SpeciesId id, SpeciesPhase phase) const;

private:
    std::vector<SpeciesDefinition> d_species;
    std::uint64_t d_revision;
    std::uint64_t d_fingerprint;
};

/** One distinct dispersed bin/representation; not a duplicate of aggregate dispersed amounts. */
struct SpeciesPopulationState
{
    std::string id;
    std::vector<double> amounts; ///< mol per cell, catalog order.
};

/** Extensive mol per cell. Liquid and Dissolved share the same physical liquid volume.
 * amounts is phase-major, then catalog order. Its Dispersed entries are the unassigned
 * dispersed compartment; named populations are separate inventories. phase_amount()
 * derives an aggregate. Reaction donors always select an exact compartment.
 */
struct SIMPLEFLUID_EQUATIONS_EXPORT CellSpeciesState
{
    std::uint64_t schemaRevision = 0;
    std::uint64_t schemaFingerprint = 0;
    std::vector<double> amounts;
    std::vector<SpeciesPopulationState> dispersedPopulations;

    CellSpeciesState() = default;
    explicit CellSpeciesState(const SpeciesCatalog& catalog);
    [[nodiscard]] double amount(
        const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase, std::string_view population = {}) const;
    [[nodiscard]] double phase_amount(const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase) const;
    void set_amount(const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase, double value,
        std::string_view population = {});
    void validate(const SpeciesCatalog& catalog) const;
};

struct StoichiometricChange
{
    SpeciesId species = 0;
    SpeciesPhase phase = SpeciesPhase::Liquid;
    double coefficient = 0.0; ///< Signed mol species/mol reaction; negative consumes.
    std::string population;   ///< Named dispersed compartment, otherwise the base phase.
};

struct ReactionDefinition
{
    ReactionId id = 0;
    std::string name;
    std::vector<StoichiometricChange> changes;
    double heatPerExtent = 0.0; ///< J/mol reaction, positive adds sensible heat; reverse extent reverses heat.
};

enum class EnergyConvention
{
    SensibleEnergy,         ///< heatPerExtent contains only chemical energy absent from species sensible storage.
    IncludesFormationEnergy ///< Species thermodynamics already contains chemical energy: heatPerExtent must be zero.
};

/** Every internal reaction conserves supplied molar mass and each declared element.
 * No empirical chemistry, molar mass, formation energy, or preferred species is built in.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT ReactionMechanism
{
public:
    ReactionMechanism(std::shared_ptr<const SpeciesCatalog> catalog, std::vector<ReactionDefinition> reactions,
        EnergyConvention convention = EnergyConvention::SensibleEnergy);
    [[nodiscard]] const SpeciesCatalog& catalog() const noexcept;
    [[nodiscard]] const std::vector<ReactionDefinition>& reactions() const noexcept;
    [[nodiscard]] std::size_t index(ReactionId id) const;
    [[nodiscard]] EnergyConvention energy_convention() const noexcept;

private:
    std::shared_ptr<const SpeciesCatalog> d_catalog;
    std::vector<ReactionDefinition> d_reactions;
    EnergyConvention d_convention;
};

enum class SourceVolumeBasis
{
    LiquidVolume,
    BulkVolume
};

struct ReactionRate
{
    ReactionId reaction = 0;
    double value = 0.0; ///< Signed mol reaction/(m3 of declared basis s).
    SourceVolumeBasis basis = SourceVolumeBasis::LiquidVolume;
};

struct IntegratedReactionExtent
{
    ReactionId reaction = 0;
    double moles = 0.0; ///< Signed mol reaction per cell over the entire interval; no further dt/volume scaling.
};

/** Accepted occupancy is the default. A thermodynamic predictor carries the
 * actual provisional EOS liquid volume during a flow corrector; its owner must
 * close and validate material occupancy before accepting the physical state.
 * ReferenceControlVolume identifies the weak Boussinesq field adapter: physical
 * EOS liquid volume is a rate basis distinct from its prescribed carrier CV,
 * and no local phase-occupancy closure is claimed by that approximation.
 */
enum class ReactionVolumeClosure
{
    Accepted,
    ThermodynamicPredictor,
    ReferenceControlVolume
};

/** Immutable position of one local source-donor frame.
 * Canonical geometry identity is distinct from algebraic map GIDs and owned rows.
 * The centroid/epoch belong to the actual donor stage (trial-new geometry on ALE).
 * An owned row is supplied only by a mesh executor; it is an ephemeral callback index,
 * not a persistent identity or permission to retain a view across mesh/owner changes.
 */
struct ReactionLocation
{
    std::uint64_t geometryCellId = 0;
    std::uint64_t geometryEpoch = 0;
    std::array<double, 3> centroid{}; ///< m, in the actual source frame.
    std::optional<std::size_t> ownedCell;
    bool operator==(const ReactionLocation&) const = default;
};


struct ReactionEnvironment
{
    double time = 0.0;
    double temperature = 0.0;      ///< K.
    double absolutePressure = 0.0; ///< Pa.
    double liquidVolume = 0.0;     ///< m3, shared by Liquid and Dissolved.
    double bulkVolume = 0.0;       ///< Actual geometric m3, including during a predictor.
    ReactionVolumeClosure volumeClosure = ReactionVolumeClosure::Accepted;
    std::optional<ReactionLocation> location; ///< Optional for pure kernels; spatial laws explicitly require it.
};

struct ReactionContext
{
    const SpeciesCatalog& catalog;
    const CellSpeciesState& state;
    const ReactionEnvironment& environment;
};

/** Pure deterministic, rank-local and reentrant callbacks; captures must be immutable.
 * Identity, revision and schema are validated outside the numerical stage evaluations.
 * validate() checks the supplied physical validity domain at every stage.
 */
struct ReactionRateProvider
{
    std::string identity;
    std::uint64_t revision = 1;
    std::uint64_t schemaRevision = 1;
    std::uint64_t schemaFingerprint = 0; ///< Must match catalog.fingerprint(), including ordered species metadata.
    std::string validityDescription;
    std::string parameterUnits;
    std::function<void(const ReactionContext&)> validate;
    std::function<std::vector<ReactionRate>(const ReactionContext&)> evaluate;
};

struct ExternalSpeciesTransfer
{
    std::string reservoir;
    SpeciesId species = 0;
    SpeciesPhase phase = SpeciesPhase::Liquid;
    double moles = 0.0;               ///< Signed mol per cell, positive enters this cell.
    double mass = 0.0;                ///< Signed kg; must equal molarMass*moles.
    double energy = 0.0;              ///< Signed J, independent of external heat and reaction allocation.
    std::array<double, 3> momentum{}; ///< Signed kg m/s.
    std::string population;
};

struct VolumetricHeatRate
{
    double value = 0.0; ///< Signed W/m3 of declared basis.
    SourceVolumeBasis basis = SourceVolumeBasis::BulkVolume;
};

/** Independent external heating, excluding heatPerExtent and transfer energy.
 * Caller budgets must exclude energy already accounted for by heatPerExtent or
 * stored species formation energies; the thermal solver uses the same convention.
 */
struct ThermalInput
{
    std::optional<VolumetricHeatRate> rate;
    std::optional<double> integratedEnergy; ///< Signed J/cell, alternative to rate.
};

struct ReactionSourceInput
{
    std::optional<ReactionRateProvider> rates;
    std::vector<IntegratedReactionExtent> integratedExtents;
    std::vector<ExternalSpeciesTransfer> externalTransfers;
    ThermalInput heat;
};

struct ReactionAdvanceOptions
{
    double maximumSubstep = std::numeric_limits<double>::infinity();
    std::size_t maximumSubsteps = 10000;
    std::size_t maximumRetries = 40;
};

/** Signed event deltas use the same phase/population topology as state, never endpoint subtraction.
 * Reservoir events retain their individual identity. All sums use compensated accumulation.
 */
struct SpeciesSourceLedger
{
    CellSpeciesState chemicalChange;
    CellSpeciesState externalChange;
    std::vector<double> reactionExtents; ///< Signed net extents, mechanism order.
    /** Nonnegative accepted directional extents; retain gross chemistry when a
     * reversible rate changes sign within an accepted SSPRK2 interval.
     */
    std::vector<double> forwardReactionExtents, reverseReactionExtents;
    std::vector<ExternalSpeciesTransfer> externalTransfers;
    double externalHeat = 0.0;
    double reactionHeat = 0.0;
    double transferEnergy = 0.0;
    double externalMass = 0.0;
    std::array<double, 3> externalMomentum{};
    std::size_t substeps = 0;
    std::size_t retries = 0;
    [[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT double thermal_energy() const;
};

/** One accepted net chemical change before narrowing into the reporting ledger.
 * Identity is chemical species + exact phase/population, never a preferred vector slot.
 */
struct ReactionCompartmentChange
{
    SpeciesId species = 0;
    SpeciesPhase phase = SpeciesPhase::Liquid;
    std::string population;
    long double moles = 0;
};

/** Local composition receipt from the kernel's actual accepted sums.
 * These channels precede double reporting: do not reconstruct them from reactionExtents.
 * chemicalChanges contains only nonzero entries in state phase/catalog/population order.
 * Heat channels retain high and correction separately, including when long double equals double.
 * reactionHeatContributions is in mechanism order and contains accepted extent * heatPerExtent.
 * External material/energy/momentum packets retain their individual ledger.externalTransfers records.
 * This is O(compartments + reactions), with no per-substep diary or MPI serialization contract.
 * It is transient local composition data, not an alternative authoritative stored species state.
 */
struct ReactionLocalReceipt
{
    std::uint64_t schemaFingerprint = 0;
    std::vector<ReactionCompartmentChange> chemicalChanges;
    std::vector<std::array<long double, 2>> reactionHeatContributions;
    std::array<long double, 2> externalHeat{};
    /** Actual post-source endpoint, including chemistry and external material.
     * Dense state phase/catalog/named-population order; each pair is high and correction.
     * Later material stages must retain both terms for finite donor reservation.
     * This is local composition authority, not an MPI/reporting or persistent-state format.
     */
    std::vector<std::array<long double, 2>> postSourceAmounts;
};

struct ReactionAdvance
{
    CellSpeciesState state;
    SpeciesSourceLedger ledger;        ///< Double reporting views of the accepted events.
    ReactionLocalReceipt localReceipt; ///< Use raw local channels when composing later physical source stages.
};

/** Atomic pure preview. External transfers and supplied integrated extents are one
 * simultaneous event; all donors are checked jointly before publication. Rate and
 * integrated forms are alternatives for an interval. Rates use conservative SSPRK2
 * with bounded subcycling/retries; maximumSubstep controls temporal accuracy.
 * Each Euler stage reserves gross donor consumption jointly (including reverse
 * reactions), without borrowing simultaneous production. Negative/nonfinite amounts
 * reject; no clipping or tolerance relaxation. Initially occupied Liquid/Dissolved
 * compartments require liquidVolume>0, and every occupied state needs bulkVolume>0.
 * Positive reaction/transfer products may create a previously absent liquid phase;
 * the caller must recover its new volume through the EOS before acceptance.
 * Constant rates after donor exhaustion
 * reject when bounded retries cannot produce a feasible physical step.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT ReactionAdvance advance_reactions(const ReactionMechanism& mechanism,
    const CellSpeciesState& initial, const ReactionEnvironment& environment, double duration,
    const ReactionSourceInput& input = {}, const ReactionAdvanceOptions& options = {});

} // namespace SimpleFluid::Reacting
