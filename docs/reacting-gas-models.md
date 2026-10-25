# Reacting gas in solution: API and transport contract

The active reacting API uses an explicit species catalog, conservative reactions,
supplied properties and real material inventories. No species name, gas constant,
reaction yield or source of heat selects chemistry implicitly. See the
[source contract](modeling/reacting_gas_sources.md), the
[native flow architecture](architecture/low_mach_reacting_flow.md), and the
[boiling/thermal ownership guide](modeling/reacting_bubble_boiling.md).

## Choose the physical owner

| Path | Owner and scope |
| --- | --- |
| `ReactingGasModel` with supplied liquid/gas EOS | Schema-indexed fields and named bubble populations coupled to the weak Boussinesq carrier; no independent phase-energy or gas-momentum equation |
| `ReactingGasModel` with `bouss_reference` and `bouss_sources` | Explicit fixed-density, scalar-temperature approximation with finite chemical and solvent/vapor inventories, coordinated bulk/wall admission |
| `LowMachMixtureSolver` with `ReactingMixtureCoupler` | Native conservative species, phase internal energy, common pressure, carrier momentum and explicit phase-number ownership |
| Standalone `ScalarVoidFractionModel` and `BoilingSourceModel` | Low-order scalar void/boiling controls; these are not a general chemical inventory |
| Explicit H2/IF97 verification applications | Declared physical reference laws and reservoirs, with the approximation documented by each application |

These paths have different energy and volume contracts. Enabling one does not
authorize combining every other model. VoF remains unavailable in this refactor;
no experimental-enable path supplies those absent sources. Bounded
floating-point and exact-rational VoF/general-AMR geometry paths remain deferred.

## Catalog, reactions and units

`Reacting::SpeciesCatalog` is immutable. Each entry supplies a unique stable ID
and name, positive molar mass, elemental composition and allowed compartments.
A fingerprint binds the ordered schema; matching size or revision alone is not
enough. Zero is an ordinary species ID, not a solvent or inert-gas sentinel.

`CellSpeciesState` stores extensive mol per cell. `Liquid` and `Dissolved` are
chemical compartments in one physical liquid phase and share its volume and
temperature. Every named `Dispersed` population has its own composition;
`Resolved` is distinct. Reactions select exact donor compartments, including the
population name. Aggregates are derived views and cannot supply an unnamed donor.

| Quantity | Unit |
| --- | --- |
| Cell species amount / integrated reaction extent | mol/cell / mol reaction per cell |
| Model species field | mol/m3 of geometric control volume |
| Liquid-basis concentration | mol/m3 of actual physical liquid |
| Rate | mol reaction/(m3 of declared basis s) |
| Absolute pressure / temperature | Pa / K |
| Integrated energy / external momentum | J/cell / kg m/s per cell |
| Population number / number field | count/cell / count per m3 control volume |
| Diffusivity / partial-molar enthalpy | m2/s / J/mol |
| Surface tension / bubble radius | N/m / m |

`ReactionMechanism` checks molar-mass and declared-element conservation.
`ReactionSourceInput` accepts signed rates with `LiquidVolume` or `BulkVolume`
basis, or integrated extents. Rate volume and time are applied once; integrated
extents receive no additional scaling. All finite internal donors are reserved
jointly. The bounded SSPRK2 source kernel retries smaller explicit local steps
when needed; supplied integrated extents provide an alternative for externally
integrated chemistry. It does not clip inventory or relax a physical gate.

A named `ExternalSpeciesTransfer` declares external species, amount, mass,
energy and momentum explicitly. Supply and withdrawal are distinct from an
internal reaction, phase relocation or physical transport escape. Arbitrary
positive source terms cannot silently manufacture mass.

Independent external heat excludes chemical heat. With `SensibleEnergy`,
`heatPerExtent` supplies chemical energy absent from sensible storage; reverse
extent reverses its sign. With `IncludesFormationEnergy`, species caloric
storage already owns that energy and per-reaction heat must be zero. Do not add
latent or pressure work a second time.

## Field model setup and advancement

The C++ modes are `ReactingGasMode::Disabled` and `Species`. An enabled model
requires an explicit mechanism plus matching supplied liquid/gas EOS, or the
separate finite Boussinesq reference binding. Set actual initial species and
population numbers through the initialization callbacks or collective
`initialize_species(...)`. Batch initialization includes empty owners; it is
not a sequence of rank-local collective setters.

Use `species_field(id, phase, population)`, `population_number_field(name)` and
`cell_species_state(...)` for authoritative state. There is no reserved H2 slot
or single-species inventory facade. Liquid follows the carrier;
`solution_transport` selects whether Dissolved follows it too. Complete solution
advection is required by the species-bound ALE path. Gas and number transport
use the same named population carrier-plus-slip flux. The field adapter's
Resolved compartment is stationary; it does not advect resolved interfaces.
Actual physical escape comes from accepted upwind transport, not an endpoint
loss estimate.
Before the first population solve, each transport interval freezes paired
extensive species amounts and counts. ALE forms both from accepted-old control
volumes and publishes transported concentrations on trial-new volumes. Every
bin's slip and radius evaluation uses that frozen pair; a later bin cannot mix
old material with a prior bin's already transported count.

The field source environment uses `ReferenceControlVolume`: actual EOS liquid
volume sets liquid-basis rates, while geometric CV sets bulk-basis rates and
heat. EOS liquid volume may differ from CV under this weak Boussinesq
approximation. It is not clipped, and this path does not claim local total
phase occupancy. The strict pure-kernel `Accepted` context and native flow's
`ThermodynamicPredictor` retain their separate occupancy owners.

`MultiphysicsSolverCoupler` retains shared ownership of the solver and owns
optional models and advancement. Construct it around the solver, configure
models, then use its `step()`/`run()` interface. Normal typed sources advance
before scalar temperature; their accepted thermal input enters that equation
once, followed by a nonadvancing thermodynamic refresh. The Boussinesq scalar
temperature owner requires sensible-energy accounting.

The explicitly selected `PrescribedCarrierAfterTemperature` schedule exists
for gas-only reference applications. It preserves a documented flow-temperature-
gas split and independently heated external carrier. It rejects finite
Liquid/Dissolved stock, film/implicit interphase processes, boiling, both finite
Boussinesq bindings, free-surface/ALE and independent phase-energy ownership.
Every individual applied chemical, external and transfer heat channel must be
zero, including canceling channels. Physical escape enthalpy/momentum remain
carried traces, not scalar-temperature heat inputs.

## Properties and population closures

Supply composition-aware liquid and gas thermodynamics with matching catalogs,
energy conventions, validity bounds and derivatives. Gas radius and volume use
actual bin composition and number; exact empty bins have zero number, amount
and volume. Finite gas capacity or invalid radius brackets reject rather than
discarding material.

`ReactingSpeciesProperties` binds per-species diffusivity/equilibrium and
carried partial-molar enthalpy laws. Film exchange uses supplied mixture
fugacities and finite phase donors. A zero species diffusivity explicitly
disables that film exchange. No implicit H2 solubility or ideal-mixture
partial enthalpy is substituted. Field bulk diffusion is rejected until that
backend has a matching counterflux/face-energy owner; film diffusivity does
not enable a hidden independent Fick solve.

Supplied population packets describe finite same-species transfers and explicit
number births/deaths. Optional first-order micro/large dissolution and conversion
frequencies retain neutral moment mathematics when explicitly selected.
Representative populations are not a resolved size distribution. Birth, death
and material amounts must remain paired; no automatic count floor or collapse
is inferred.

Zero/prescribed slip and selected correlation helpers are closures, not gas
momentum equations. `Hughmark`, `HughmarkSchmidtDomain`, spherical diffusion and
the named Feng-Michaelides laws retain their documented applicability checks;
see [liquid-side mass transfer](bubble-mass-transfer.md). The Celata helper
retains its published-envelope diagnostics; using it outside that envelope
needs application qualification. None of these reference correlations selects
a chemical species.

Field pressure modes are constant, prescribed absolute-pressure history and
reconstructed reference-plus-zero-mean mechanical pressure. The old weak
inertial pressure mode is retired. Invalid pressure/EOS states reject.
`evaluate_liquid_volume(candidate_offset)` and
`evaluate_submerged_bubble_volume(candidate_offset)` are pure collective
queries; pressure publication validates liquid and gas together before writes.

## Native conservative flow

`LowMachMixtureSolver` owns phase species, internal energy, carrier momentum and
common thermodynamic pressure. `ReactingMixtureCoupler` maps chemical
compartments to physical phases and composes actual transported state,
chemistry, ordered same-carrier face stages, configured local interphase
stages and explicit population-number events. Each stage uses its actual
predecessor; independently reserving all stages against the same accepted stock
is forbidden.

Explicit face diffusion names its counter-species, contact/conductance and true
partial-molar enthalpy law. Joint outgoing face donors are reserved before
credits; the same opposite-signed heat packet reaches the two incident cells.
The declared same-carrier zero-net-mass convention contributes exactly zero
diffusive momentum. Fourier contact likewise needs supplied conductance; a gas
bin does not imply contact with another bin.

Configured local interphase amounts are frozen before caloric pressure recovery.
Supplied partial-molar enthalpies and their derivatives are reevaluated at each
pressure-conditioned anchor; kinetics are not reintegrated inside pressure
roots. The phase-energy owner applies `-p*(Vfinal-Vadv)` once. See
[pressure-conditioned thermal transfer](architecture/reacting_interphase_thermal.md).

This native backend is closed and orthogonal, with common phase pressure and
zero surface tension. It supports common or separate phase temperatures and
explicit dispersed roles with extensive counts. Prescribed planar ALE uses live
motion authority, accepted-old/new CV storage and relative phase transport.
Preview restores accepted geometry; commit and restore do not call source or
EOS laws. The moving path retains its inviscid/slip-wall, topology and
nonperiodic restrictions. Upstream complete depletion without owned pre-transfer
state and work accounting, nonlocal pressure-conditioned transfer, evolving slip
momentum, capillary energy and general moving interfaces are outside this support boundary.

## Configuration, output and accepted receipts

Flat database parsing cannot construct a catalog, EOS or pure source provider.
The parser therefore rejects `reacting_species_model=species` or
`enable_reacting_gas=true`; construct enabled options explicitly in C++.
Disabled flat configuration remains available.
Old `idealGasSource`, `twoPopulation`, fixed-species/yield/property/source-decay
keys and aliases are retired. `solution_transport_mode` replaces the specialized
dissolved-transport key. Application reference property files may still contain
explicitly named physical yield/correlation inputs; those are application laws,
not solver defaults.

`include_reacting_gas_fields` exports dynamic
`species_<id>_phase_<phase-index>` and
`species_<id>_population_<name>` inventories with aggregate pressure/void/source
diagnostics. `include_sources` and `include_material_properties` retain their
thermal/material roles. Aggregate micro/large summaries, when present, are
derived total-population views, not a chemical identity or an independent stock.

`species_histories()` reports separate chemical production/consumption,
external supply/withdrawal, interphase relocation and physical outflow.
`species_step_events()` and `physical_escape_events()` retain exact accepted
interval provenance. `population_escape_receipt()` carries actual per-bin
outward number events with the same accepted `SpeciesEventReceipt`; it is not
inferred from moles or count endpoint differences. Snapshots include inventories,
raw histories, receipts, geometry-related state and accepted ledgers. Public
MPI totals are rounded reports and never seed private compensated accumulation.

Public source metadata, schemas, options and failures are checked collectively,
including actual empty owners. User property/source laws are deterministic,
MPI-free and reentrant with immutable captures. Optional `ReactionLocation`
names the canonical geometry cell, trial centroid/epoch and ephemeral owned row;
algebraic GIDs and retained mesh views are not spatial identities.

## Verification and reference provenance

The source tree contains independent finite-donor, multiple-gas, energy,
transport, pressure, replay and empty-owner regression fixtures. Their existence
does not mean every combination or rank count has run. Use current focused
serial/MPI logs, required ranks, skips and ABI results for execution evidence;
manufactured tests do not physically validate supplied laws.

The dispersed and bottom-heated OpenFOAM references explicitly select H2 and
IF97 water. Their prescribed external carrier, named gas/count reservoirs,
temperature schedule and correlation inputs remain documented approximations.
They do not claim finite-water inlet or separate gas-energy/momentum closure.
The planar ALE verification uses finite two-gas inventories. The opt-in
coupling benchmark's `fixture_schema=2` is a different finite mixture workload;
old fallback timings are not a matched comparison.
