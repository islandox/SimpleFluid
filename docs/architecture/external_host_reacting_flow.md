# External-host integration with catalogued reacting flow

This contract defines how an external application supplies chemistry, material
laws and feedback to SimpleFluid's conservative species, transport, population
and flow transactions. It does not establish build compatibility, executed test
coverage or physical validation. Every gas species, finite donor and process law
must be declared; no application-specific species default or deficit field is
part of the integration contract.

The target is the supplied-EOS species adapter with the existing Boussinesq
sensible-temperature solver and constrained planar ALE. Read the
[species contract](reacting_species_contract.md),
[surface contract](reacting_species_surface.md) and
[coupler ownership contract](multiphysics_solver_coupler.md) together.
Native low-Mach phase energy is a separate backend and is not an interchangeable
replacement for a moving-pool configuration with an external thermal participant.

## Host integration interfaces

| Integration responsibility | Required contract |
| --- | --- |
| Build and capability checks | Check the generic species/source/surface APIs and use matching headers and producer libraries. Retain ABI and shared Trilinos backend checks. |
| Model setup | Construct `ReactingGasOptions` with `ReactingGasMode::Species`, a supplied catalog/mechanism/EOS, finite initial inventories and explicit process laws. |
| Thermal and surface coupling | Retain the thermal participant and planar-ALE transaction on the supplied-EOS species route. Bind liquid mass as a follower of species rather than a second transported inventory. |
| Feedback and material projection | Derive the conservative coupling vector and solution composition from authoritative catalog fields and accepted receipts. |
| Optional bubble closures | Select `BubbleGasPhysics` closures only within their declared material and validity ranges. Each reported diagnostic needs an actual producer. |
| Interval advancement | Preserve checkpoint replay and thermal interval energy. Consume accepted process and volume diagnostics after each substep. |
| Output and validation | Use accepted catalog histories/events and derived output quantities, with finite species and accepted production, consumption and escape balances. |
| Bulk dissolved transport | Require a supported conservative bulk-transport owner; film diffusivity alone does not supply it. |

Set `ReactingPressureMode` explicitly and select dissolved advection through
`ReactingTransportMode::Advective` and `solution_transport`. Preserve the
configured pressure reference and distinguish absolute pressure from gauge
output. `BubbleMassTransferMode::Hughmark`, Feng and spherical-diffusion film
selections retain their declared validity limits. Slip selection and
transport-solver settings remain explicit.

Source yields, release efficiencies, nucleation laws and kinetic integration
policies belong to declared host providers. Reporting deficits are not finite
material inventories. Unsupported configuration selections must reject
collectively; an input must not be accepted when its process is not applied.
Keep host configuration, resolved input provenance, parser tests and user-facing
documentation consistent with these choices.

## Material, source and process ownership

The host must supply the following immutable, rank-consistent configuration:

1. A `Reacting::SpeciesCatalog` with stable IDs, SI molar masses, elemental
   composition and allowed phases; a conservative `ReactionMechanism`; and an
   `initial_species_state` returning extensive mol/cell in Liquid, Dissolved,
   Resolved and each declared Dispersed population. All finite solvent, donor and
   coproduct inventories must be explicit. No catalog position is reserved.
2. `liquid_thermodynamics` and `gas_thermodynamics` using the same
   `EnergyConvention::SensibleEnergy` as the mechanism. Providers must supply
   composition-dependent properties, declared pressure/temperature bounds and
   gas fugacities or an explicit equilibrium closure. Temperature remains owned
   by the Boussinesq solver; supplied EOS data does not introduce phase-energy
   storage into this adapter.
3. Per-species `ReactingSpeciesProperties`: film diffusivity or
   `diffusivity_provider`, Henry coefficient or `equilibrium_concentration`, and
   required carried enthalpy for actual open-boundary species. External material
   correlations require explicit units, local state inputs and validity ranges.
   No diffusion or nucleation law is inferred from a species name.
4. Named `population_ids`, `initial_population_numbers` and an explicit
   `population_process` returning paired `MultispeciesPopulationPacket`
   transfers and `PopulationNumberEvent` births/deaths. Preserve every species
   during micro/large conversion. New gas does not imply an inferred birth;
   emptied gas does not authorize an inferred count reset. A host nucleation law
   must declare both its material transfer and its number event.
5. A pure, reentrant `species_source` returning `ReactionSourceInput` rates or
   integrated extents, independent external heat and typed reservoir transfers.
   Rate volume basis, duration and molar versus mass units must be explicit.

Reaction stoichiometry is application-specific. A gas yield alone does not
define a conservative reaction: the host must supply finite donor consumption,
all coproducts and elemental/mass balance, or declare actual external reservoirs.
A reporting deficit is not material stock. Each chosen mechanism requires its
own physical justification; an aggregate gas field cannot determine that mechanism.

Any mapped energy used by a host rate law must carry the same cell/interval
provenance as the associated thermal input. Thermal deposition is applied once
through its existing owner. Reaction heat, if any, is separately declared; no
heat-to-gas conversion is inferred by SimpleFluid. The host must provide a tested
immutable per-cell/subinterval energy access contract for its source callback.
Every corrector replays the same committed interval and recomputes the trial
without accumulating production or heat.

Read state through `find_reacting_gas_model()`, `species_field(id,phase,population)`,
`cell_species_state()`, `population_number_field()` and derived
`liquid_mass_density()` / `pure_liquid_density()`. Stored species fields are
mol/m3 of current bulk volume; multiply by the actual cell volume exactly once.
The species-bound `LiquidMassInventory` follows already transported material.
Do not retain an independent liquid transport/debit or project a reporting
deficit as finite material. Derive external material and elemental projections
from actual species composition and the intended host material model.

Use `SpeciesEventReceipt`, `source_ledgers()`, `species_step_events()`,
`species_histories()` and `physical_escape_events()` for accepted output and
conservative feedback. Consume each event once; a thermodynamic refresh publishes
no source interval. Checkpoints must restore fields, geometry, counts, ledgers,
receipt state, host clocks and accepted output cursors together.

## Prerequisites and support gates

**SimpleFluid prerequisites.** The supplied-EOS species path must retain complete
solution advection, exact accepted material/escape and liquid-volume transport
receipts, coherent population slip/film evaluation, resolved-gas volume accounting,
collective rejection and atomic rollback. Qualify the exact mesh, pressure,
thermal-participant and planar-ALE configuration used by the host. General moving
interfaces, VoF, adiabatic headspace energy and a two-fluid model remain outside
this integration's claim.

Bulk dissolved diffusion remains a gap. `ReactingSpeciesProperties::diffusivity`
is film data; `ReactingBulkDiffusionMode::ConservativeMixture` is currently
rejected. A host two-cell diffusion case requires actual bulk transport. Either
implement and qualify a conservative finite-counterflux/canonical-energy
diffusion owner in SimpleFluid, or explicitly reject the corresponding host
configuration pending that work. A `diffusivity_provider` alone is insufficient.

**Host prerequisites.** Implement the catalog/material/source/process providers,
finite coupling-vector and composition projection, receipt-based output and
validators, capability probes and coherent configuration. Select matching
SimpleFluid source/build pairs and rebuild affected producers; importing a
library does not rebuild its dependencies. Preserve compiler, standard-library,
MPI and shared Trilinos backend compatibility.

The fixed `BoussReferenceVolume` / `BoussCoordinatedSources` approximation is not
a shortcut for moving-pool coupling: SimpleFluid rejects it with free surface/ALE,
material updaters or a thermal participant. Do not relax that rejection to make
configuration pass. The supplied-EOS route retains the Boussinesq sensible-energy
approximation; a new thermal owner needs a separate design and support contract.

## Representative acceptance work

Begin with cell-law oracles for zero deposition, finite deposition, donor
exhaustion, a second gas species and reordered catalogs. Check finite stock,
elemental/mass balance, heat ownership, explicit count events and actual accepted
source receipts. Add a local-temperature/backward-Euler diffusion oracle only
when its bulk-transport prerequisite is supported.

Use a small coupled fixture with zero prescribed slip and a short endpoint for
initial integration checks. Run source-off, transport-only, source-only,
thermal-only and combined cases with explicitly populated gas where needed.
Early inactive startup alone does not qualify nucleation, slip, transfer or donor
depletion. Retain input/build provenance and fresh artifact locations for each run.

Run focused serial cases first, then corresponding two- and four-rank MPI cases
with `OMP_NUM_THREADS=1`; include a partition with empty owners. Verify actual
rank counts and executed cases. Cover supported annular, thermal-participant and
XY/Z configurations, with active-process variants. Exercise rank-local provider
failure and rejected-step rollback/retry against a control.

Gate accepted intervals on deposited/thermal energy, liquid and all species
masses, elemental inventories including escape, population counts, void/volume/GCL,
mesh-top/level consistency and exactly-once receipt publication. Preserve physical
tolerances independently of the linear solver. Compare matched time-step/substep
refinements, record input/executable/library provenance, and distinguish numerical
verification from experimental material validation. Long transients and broad
performance or rank campaigns are separate qualification work.
