/** @file SpeciesDiffusion.hh
 * @brief Pure conservative face packets for finite mixture counterdiffusion.
 */
#pragma once

#include "equations/ReactionMechanism.hh"

#include <array>
#include <optional>
#include <span>

namespace SimpleFluid::Reacting
{

struct DiffusionCompartment
{
    SpeciesPhase phase = SpeciesPhase::Dissolved;
    std::string population;
};

/** Supplied zero-net-mixture-mass convention with an actual finite counter-species.
 * Providers are pure and rank-local; geometry/map/epoch identity and collective
 * acceptance belong to the caller. No solvent, gas or species ID is reserved.
 */
struct SpeciesDiffusionBinding
{
    DiffusionCompartment transported{SpeciesPhase::Dissolved, {}};
    DiffusionCompartment counter{SpeciesPhase::Liquid, {}};
    std::optional<SpeciesId> counterSpecies; ///< Required explicit selection; species ID zero is an ordinary valid ID.
    EnergyConvention energyConvention = EnergyConvention::IncludesFormationEnergy;
    std::string identity;
    std::uint64_t revision = 1;
    std::uint64_t schemaFingerprint = 0;
};

/** One canonical interior face, positive owner->neighbor. Amounts are already
 * integrated mol, not molar rates or concentrations. No dt, area or volume is
 * applied by the packet kernel. An absent row is a remote cell whose physical
 * donor budget is validated on its owner before the caller accepts collectively.
 * At least one row is locally owned; halo-sampled caloric/velocity data are
 * supplied on BOTH sides in the same canonical orientation on every rank.
 * The counter's primary entry must be zero if both compartments are identical.
 * Empty inactive property entries may use zero without evaluating an absent EOS.
 */
struct CanonicalDiffusionFace
{
    std::uint64_t identity = 0;
    std::optional<std::size_t> owner, neighbor;
    std::vector<double> moles;                                     ///< Catalog-ordered integrated primary molar fluxes.
    std::vector<double> ownerMolarEnthalpy, neighborMolarEnthalpy; ///< J/mol, same reference as stored phase energies.
    double ownerCounterMolarEnthalpy = 0.0, neighborCounterMolarEnthalpy = 0.0;
    std::array<double, 3> ownerVelocity{}, neighborVelocity{}; ///< m/s; donor carrier momentum convention.
};

struct SpeciesDiffusionFaceLedger
{
    std::uint64_t identity = 0;
    std::optional<std::size_t> owner, neighbor;
    std::vector<double> moles;
    double counterMoles = 0.0;        ///< Exactly zero when the primary mass sum is exactly zero.
    double enthalpy = 0.0;            ///< Rounded canonical J report; cell sums retain individual raw donor terms.
    std::array<double, 3> momentum{}; ///< Canonical integrated kg m/s owner->neighbor.
    double massResidual = 0.0, absoluteMassFlux = 0.0; ///< kg; only representational roundoff may remain.
};

struct SpeciesDiffusionCellLedger
{
    CellSpeciesState change;          ///< Signed molar event deltas, never stored endpoint subtraction.
    double enthalpy = 0.0;            ///< J entering this owned cell.
    std::array<double, 3> momentum{}; ///< kg m/s entering this owned cell.
};

struct SpeciesDiffusionPreview
{
    std::string identity;
    std::uint64_t revision = 0, schemaFingerprint = 0;
    std::vector<CellSpeciesState> state;
    std::vector<SpeciesDiffusionCellLedger> cells;
    /** Sorted local-incident events. Global face diagnostics count only the canonical event owner
     * (owner.has_value() under the declared row binding); both ranks apply their owned side.
     */
    std::vector<SpeciesDiffusionFaceLedger> faces;
};

/** Atomic preview without mesh, MPI, material or external-provider state.
 * All outgoing primary and counter donors across ALL incident faces are reserved
 * jointly before incoming transfers are credited. Exhaustion rejects without
 * modifying any input. Accepted stocks retain initial amounts and individual
 * signed events in one compensated sum; a rounded net ledger is a reporting view.
 * A caller applying one narrowed net source must reject unrepresentable endpoint
 * replay rather than silently erase a finite residual stock. The owned states are
 * the actual donor snapshot for this
 * sequential stage (e.g. after advection and chemistry). Independently reserved
 * chemistry/diffusion previews from one shared base cannot be merged without
 * a joint reservation ledger. There is no clipping, donor borrowing, rate capping or
 * inventory renormalization. Source/energy sums are compensated.
 *
 * One counter flux per face cancels sum(W_i*Delta n_i). A 64-epsilon relative
 * arithmetic gate checks the mass residual after storing that flux; it never
 * relaxes the strictly nonnegative finite-donor gate. The donor-side molar
 * enthalpy is selected independently for every signed species flux, including
 * the counter. The same raw donor-enthalpy terms are applied with opposite signs on the
 * two sides; each cell energy is narrowed once, preserving small terms when
 * large face transfers cancel. Rounded face reports cannot reconstruct that sum.
 *
 * The caller computes fluxes from actual phase concentrations and geometry,
 * owns boundary reservoirs separately, validates retained trial identities,
 * and supplies these enthalpy/momentum deltas once to its chosen energy storage.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT SpeciesDiffusionPreview preview_species_diffusion(
    const SpeciesCatalog& catalog, std::span<const CellSpeciesState> owned, const SpeciesDiffusionBinding& binding,
    std::span<const CanonicalDiffusionFace> faces);

} // namespace SimpleFluid::Reacting
