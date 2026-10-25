# external host migration to catalogued reacting flow

This is the companion migration contract and implementation plan for external host's
legacy reacting integration. It does not establish build compatibility,
executed test coverage or physical validation. external host owns supplied reaction sources chemistry,
material laws and external-source feedback; SimpleFluid owns generic conservative species,
transport, population and flow transactions. No hydrogen-specific default or
legacy deficit field is to be restored in SimpleFluid.

The target is the supplied-EOS species adapter with the existing Boussinesq
sensible-temperature solver and constrained planar ALE. Read the
[species contract](reacting_species_contract.md),
[surface contract](reacting_species_surface.md) and
[coupler ownership contract](multiphysics_solver_coupler.md) together.
Native low-Mach phase energy is a separate backend and is not an interchangeable
replacement for external host's existing moving-pool/thermal-participant configuration.

## Required host changes

The following paths are in external host. Line anchors identify the legacy integration
points; implement by symbol if nearby source has moved.

| Host integration point | Required migration |
| --- | --- |
| `cmake/HydraTFDependencies.cmake:35-41` | Replace the `ReactingGasProperties.hh` / `ReactingKineticsMode` probe with checks for the generic species/source/surface APIs and matching rebuilt producer libraries. Retain ABI and shared Trilinos backend checks. |
| `src/CoupledSimulationSetup.cc:243-299` | Replace `ReactingGasOptions`, `Sheng2024TwoPopulation`, kinetics mode and `configure_reacting_gas()` with `ReactingGasOptions`, `ReactingGasMode::Species`, supplied catalog/mechanism/EOS, finite initial inventories and explicit process laws. |
| `src/CoupledSimulationSetup.cc:302-305,325-359` | Retain the thermal participant and planar-ALE transaction on the supplied-EOS species route. Bind liquid mass as a follower of species rather than a second transported material inventory. |
| `src/CoupledSimulationTransfer.cc:15-30` | Replace `find_reacting_gas_model()` and hydrogen-specific inventory queries in the conservative coupling vector; derive amounts and solution mass from the authoritative catalog fields. |
| `src/CoupledSimulationImpl.hh:53-84` | Replace `BubbleGasPhysics` with generic `BubbleGasPhysics` where the selected closure still supplies equivalent transfer diagnostics. Review each diagnostic's producer rather than preserving empty statistics. |
| `src/CoupledSimulation.cc:544-572,670,688` | Preserve interval checkpoint replay and thermal interval energy; consume generic process/void diagnostics after each substep. |
| `src/CoupledSimulationOutput.cc:80,323-355` | Replace legacy model getters, production/escape counters and donor deficits with accepted catalog histories/events and derived output quantities. |
| `tests/test_winter_diffusion.cc:2-90` | Port to supported generic transport only after resolving the bulk-diffusion prerequisite below; the old test exercises real bulk diffusion, not film transfer. |
| `verification/_coupled_history.py:516-537` | Replace deficit-based assumptions with finite species, accepted production/consumption and escape balances; retain independent physical acceptance gates. |

`ReactingPressureMode` maps to `ReactingPressureMode`; dissolved advection maps
from `ReactingTransportMode::Advective` to
`ReactingTransportMode::Advective` through `solution_transport`. Preserve the
configured pressure reference and distinguish absolute pressure from gauge
output. `LegacyHughmark` maps to `BubbleMassTransferMode::Hughmark` with its
validity limits; Feng and spherical-diffusion film selections retain their actual
semantics. Slip selection and transport-solver settings remain explicit.

`hydrogen_yield_mol_per_j`, `gas_release_efficiency`, legacy nucleation controls,
kinetics-integrator selections, donor-deficit tracking and skip-zero auxiliary
transport are not generic species configuration aliases. Move their physical
meaning into host laws, remove obsolete options, or reject unsupported selections
collectively. Do not silently accept an input whose process is no longer applied.
Update `CaseOptions`, XML cases, resolved input provenance, parser tests and
user-facing configuration documentation with these choices.

## Material, source and process ownership

external host must supply the following immutable, rank-consistent configuration:

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
   required carried enthalpy for actual open-boundary species. Retain external
   Winter/surface-tension material correlations with explicit units and local
   state inputs. There is no automatic hydrogen diffusion/nucleation law.
4. Named `population_ids`, `initial_population_numbers` and an explicit
   `population_process` returning paired `MultispeciesPopulationPacket`
   transfers and `PopulationNumberEvent` births/deaths. Preserve every species
   during micro/large conversion. New gas does not imply an inferred birth;
   emptied gas does not authorize an inferred count reset. A host nucleation law
   must declare both its material transfer and its number event.
5. A pure, reentrant `species_source` returning `ReactionSourceInput` rates or
   integrated extents, independent external heat and typed reservoir transfers.
   Rate volume basis, duration and molar versus mass units must be explicit.

supplied reaction sources stoichiometry is application-specific. A hydrogen yield alone does
not define a conservative reaction: external host must supply finite donor consumption,
all coproducts and elemental/mass balance, or declare actual external reservoirs.
The former donor-hydrogen deficit is a reporting quantity, not material stock.
A chosen water/radical/oxygen mechanism requires its own physical justification;
this migration must not invent one from the old hydrogen-only fields.

external host's mapped external-source deposited energy must drive the declared supplied reaction sources law
with the same cell/interval provenance as thermal input. Thermal deposition is
applied once through its existing owner. Reaction heat, if any, is separately
declared; no heat-to-gas conversion is inferred by SimpleFluid. The host must
provide a tested immutable per-cell/subinterval energy access contract for its
source callback; a symbol rename does not establish that contract. Every
corrector replays the same committed interval and recomputes the trial without
accumulating production or heat.

Read state through `find_reacting_gas_model()`, `species_field(id,phase,population)`,
`cell_species_state()`, `population_number_field()` and derived
`liquid_mass_density()` / `pure_liquid_density()`. Stored species fields are
mol/m3 of current bulk volume; multiply by the actual cell volume exactly once.
The species-bound `LiquidMassInventory` follows already transported material.
Do not retain an independent liquid transport/debit or project a deficit as a
finite external-source isotope. Rework the external-source atom projection from actual species
composition and the intended host material model.

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
thermal-participant and planar-ALE configuration used by external host. General moving
interfaces, VoF, adiabatic headspace energy and a two-fluid model remain outside
this migration's claim.

A genuine gap is bulk dissolved diffusion. `ReactingSpeciesProperties::diffusivity`
is film data; `ReactingBulkDiffusionMode::ConservativeMixture` is currently
rejected. external host's Winter two-cell test requires bulk transport. Either implement
and qualify a conservative finite-counterflux/canonical-energy diffusion owner in
SimpleFluid, or explicitly reject the corresponding external host configuration pending
that work. Mapping Winter to `diffusivity_provider` alone is insufficient.

**external host prerequisites.** Implement the catalog/material/source/process providers,
finite coupling-vector and isotope projection, receipt-based output/validators,
new capability probes and coherent configuration conversion. Select explicit
matching SimpleFluid source/build pairs and rebuild affected producers; external host
imports libraries rather than rebuilding dependencies automatically. Preserve
compiler, standard-library, MPI and shared Trilinos backend compatibility.

The fixed `BoussReferenceVolume` / `BoussCoordinatedSources` approximation is not
a shortcut for this application: SimpleFluid rejects it with free surface/ALE,
material updaters or a thermal participant. external host currently enables planar ALE
and can install vessel thermal participation. Do not relax that rejection to make
configuration pass. The supplied-EOS route retains the Boussinesq sensible-energy
approximation; a new thermal owner needs a separate design and support contract.

## Representative acceptance work

Begin with cell-law oracles for zero deposition, finite deposition, donor
exhaustion, a second gas species and reordered catalogs. Check finite stock,
atom/mass balance, heat ownership, explicit count events and actual accepted
source receipts. Restore the Winter local-temperature/backward-Euler diffusion
oracle only when its transport prerequisite is supported.

Use `cases/annular/case.xml` as the first coupled fixture: its small 2x4x4 mesh,
zero prescribed slip and short endpoint are suitable integration inputs. Run
source-off, transport-only, source-only, thermal-only and combined cases with
explicitly populated gas where needed. Early inactive startup alone does not
qualify nucleation, slip, transfer or donor depletion. Use the existing
`run_coupled_test.py` provenance/validation workflow and fresh artifact locations.

Run focused serial cases first, then corresponding two- and four-rank MPI cases
with `OMP_NUM_THREADS=1`; include a partition with empty owners. Verify the actual
rank counts and executed cases. Existing `coupled_annular_mpi`,
`coupled_annular_thermal_mpi` and `coupled_annular_xy_z_mpi` are starting points;
add explicit serial/four-rank and active-process variants where absent. Exercise
rank-local provider failure and rejected-step rollback/retry against a control.

Gate accepted intervals on deposited/thermal energy, liquid and all species
masses, elemental hydrogen including escape, population counts, void/volume/GCL,
mesh-top/level consistency and exactly-once receipt publication. Preserve physical
tolerances independently of the linear solver. Compare matched time-step/substep
refinements, record input/executable/library provenance, and distinguish numerical
verification from experimental material validation. R100/R196 transients and
broad performance or rank campaigns are separate qualification work.
