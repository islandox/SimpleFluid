# Reacting gas sources in solution

## Scope and implementation status

The general model describes gas produced or consumed by reactions in a liquid
solution, followed by dissolution, bubble growth, transport, and escape. The
application supplies the reaction mechanism, material properties, and closure
validity ranges. No particular energy-deposition mechanism, gas species,
solution composition, or vessel application defines the model.

`Reacting::SpeciesCatalog`, `ReactionMechanism` and `CellSpeciesState` define
supplied chemistry and finite inventories. Every enabled `ReactingGasModel`
requires an explicit mechanism, material providers and compatible population
state. There is no library-default gas species, energy yield or implicit donor.
`PrescribedHeatSource` remains an independent external thermal source. A named
application adapter may explicitly correlate material injection with prescribed
heat, provided it supplies the external mass, species and energy budgets.

There are two distinct flow integrations:

- The field adapter and `MultiphysicsSolverCoupler` retain the documented weak
  Boussinesq, prescribed-carrier and sensible-temperature approximations. They
  transport species and named population counts and publish accepted species
  receipts. Their EOS material volume need not equal a fixed reference cell's
  geometric volume.
- `LowMachMixtureSolver` and its external `ReactingMixtureCoupler` own conservative
  species, internal energy and momentum with an EOS material-volume closure.
  Pressure, phase transport, source iteration and prescribed closed planar motion
  follow the [native low-Mach contract](../architecture/low_mach_reacting_flow.md).

A resolved-gas inventory compartment does not imply interface reconstruction.
VoF and reacting-VoF integration are disabled; bounded floating-point and
exact-rational VoF/general-AMR paths are deferred from this refactor. Shared
lower-level geometry and finite-volume utilities remain available.

This migration deliberately breaks old source/configuration interfaces. Update
includes, types, setup code, field readers and executable paths, then rebuild.
There are no compatibility aliases or enabled single-species fallbacks.

| Retired interface or assumption | Replacement |
| --- | --- |
| `PrescribedHeatSource`, its options/profile, and `prescribed_heat_*` setup | Independent `PrescribedHeatSource`, `PrescribedHeatSourceOptions`, `PrescribedHeatProfile` and `prescribed_heat_*` setup |
| `ReactingGasModel`, `ReactingGasOptions`, `reacting_*` setup | `ReactingGasModel` and explicit typed `ReactingGasOptions` |
| Specialized `idealGasSource` / `twoPopulation` selectors and energy-yield source | `ReactingGasMode::Species`, a supplied schema/mechanism and declared material/source laws |
| H2 aggregate fields, reduced donor deficit and cumulative-counter escape inference | `species_field(id, phase, population)`, finite reactants, typed histories and accepted event receipts |
| `reaction_driver_energy` as general chemistry input | Supplied reaction extents/rates or named external material packets, with independent heat ownership |
| Resolved/dispersed interface coupling | Deferred; no supported implementation is included in this snapshot |
| `reacting_solution_tank_demo` / `heated_solution_tank_sst` | `reacting_solution_tank_demo` / `heated_solution_tank_sst` |

Explicit retired keys reject even when the model is disabled. A flat database
cannot manufacture a species catalog or infer material physics; enabled setup
requires the typed C++ provider seam. The delayed-external-source model and its coupler,
output and checkpoint hooks are removed. `ScalarSourceDecayTransport` retains
standalone analytic source/decay and conservative scalar transport, with no
external-source or heat-yield coupling.

Hydrogen and IF97 data remain explicit application choices in the matched
reference cases. Their provenance and H2 result labels describe those chosen
materials, not mandatory library species or slots. Numerical verification,
external comparison and physical validation remain separate claims. See the
[species contract](../architecture/reacting_species_contract.md),
[transport guide](../reacting-gas-models.md),
[bubble/boiling guide](reacting_bubble_boiling.md) and [roadmap](../../TODO.md).

## Source ownership and inputs

An external physics controller owns the reaction provider and its accepted
ledgers. The flow solver owns the mesh, primary fields, clock, and conservative
transport. The controller retains shared ownership of the solver and prepares
sources without calling a second full transport advance. This preserves the
[existing coupling ownership contract](../architecture/multiphysics_solver_coupler.md).

The provider takes a declared species schema, actual transported or ordered-stage
predecessor inventories, trial temperature and absolute pressure, liquid
composition, cell geometry, and a time interval. It returns either rates with an
explicit integration rule or integrated reaction extents for that interval. These are alternative input
forms; supplying both for the same reaction must reject. Mesh source contexts carry `ReactionLocation`: canonical geometry cell ID,
actual centroid and geometry epoch, plus an ephemeral owned row. These describe
the actual donor stage, including trial-new ALE geometry. Algebraic map GIDs
are not physical source identities. Pure local kernels may omit location; a
spatial law must explicitly require it. Source receipts separately bind schema,
provider/revision, accepted generation, time interval and transport state.

| Quantity | Units and convention |
| --- | --- |
| Species amount `n_i` | mol per cell, with its liquid, dissolved, dispersed, or resolved carrier identified |
| Molar mass `W_i` | kg/mol; supplied for every species |
| Stoichiometric coefficient `nu_ir` | mol of species `i` per mol of reaction `r`; negative for reactants, positive for products |
| Reaction rate `r_r` | mol of reaction/(m3 liquid s), unless a different reference basis is explicitly selected |
| Integrated extent `Delta xi_r` | mol of reaction per owned cell over the stated interval |
| External species transfer | Signed mol per cell per interval, with reservoir identity and associated mass/energy/momentum |
| Dissolved concentration `C_i` | mol/m3 of actual liquid, not bulk cell volume |
| External heat deposition `q_ext` | W/m3 on an explicitly declared volume basis, independent of reaction rate |
| Integrated thermal input `E_net` | J per cell per interval under the declared energy convention |

Rates may depend on local temperature, pressure, and composition or be supplied
by an external chemistry calculation. A rate law supplies its name, parameter
units and valid domain through an immutable, deterministic, MPI-free and reentrant
callback. A stateful external chemistry engine requires a separately owned
transaction protocol; this callback seam does not checkpoint or restore external
mutable provider state.
An energy-dependent yield can be an application adapter, but an energy field
is not a required input to the general gas-source model. A prescribed gas
injection must identify an external material reservoir; it is not an internal
reaction that creates mass without consuming a donor.

## Reaction and species accounting

For liquid-based rates, integrate each reaction over the actual liquid volume:

```text
Delta xi_r = integral_interval r_r(T, p_abs, composition) V_l dt,
Delta n_i,chem = sum_r nu_ir Delta xi_r.
```

For a rate per bulk volume use the bulk volume instead, and for an already
integrated extent apply neither another volume factor nor another timestep.
The reference basis is metadata, not an inference from a field name. A liquid
reaction has no capacity in a cell with no liquid unless the application
explicitly supplies another reactive carrier.

For every closed internal reaction, the supplied mechanism must satisfy
`sum_i W_i nu_ir = 0` and conserve each declared elemental inventory. Charged
species additionally require charge balance and a separately qualified
transport/electrical closure. Gas moles themselves need not be conserved by a
reaction. Per-species ledgers reconcile accepted amounts with stoichiometric
production/consumption, boundary transfers, and external reservoirs; total
mass and elemental ledgers test the invariants across reacting species.

All reactants and products, including nonvolatile liquid products, participate
in the inventory. A single gas-equivalent donor can be a documented reduced
model, but it cannot establish full solution-composition conservation. Solvent,
solute, and dissolved gas must have consistent membership in total liquid
mass so that the same mass is not counted twice. Interphase transfer removes
`W_i Delta n_i` from its donor phase and adds it to its receiver phase.

Competing reactions share the same available donors. Integrate them together
or reserve donors through one event ledger; independently capping each
reaction against the original stock is invalid. All accepted species and
population moments remain nonnegative. Donor exhaustion calls for a bounded
integration method, a smaller recomputed physical step, or atomic rejection.
Do not clip an independently computed source or renormalize inventories after
the fact. Reverse reactions and gas consumption debit the reservoir of every species
with a negative signed change, according to the reaction extent.

## Dissolved gas, bubbles, and phase volume

Production first enters its declared phase. A reaction that produces dissolved
gas does not also create a bubble source. Nucleation or direct bubble birth
requires a supplied closure and an explicit transfer of gas from the source
reservoir into paired bubble-number and species inventories. Number and moles
must remain compatible with the selected population representation.

For dilute interfacial transfer, a possible supplied closure is

```text
C_i,eq = H_i(T, composition) p_i,
p_i = y_i p_g,
dot n_i,l_to_g = k_L,i A (C_i - C_i,eq).
```

Here `H_i` uses mol/(m3 liquid Pa), `k_L,i` is in m/s, area `A` is in m2,
and positive transfer leaves the liquid. Other Henry-law conventions require
explicit conversion at the provider boundary. The same signed amount is
removed and added across the interface, so transfer cancels in the total
species budget. Gas dissolution reverses that exchange. For an ideal mixture,
each species uses its own partial pressure rather than total pressure. Nonideal exchange uses the supplied mixture fugacity or explicit
equilibrium law; ideal partial pressure cannot silently replace that law.
Film thickness, nucleation, coalescence, breakup, and slip laws require declared
applicability ranges.

An optional dilute, prescribed-pressure approximation can convert the net
molar rate entering the gas phase to a bulk volume-fraction source:

```text
S_alpha = dot n_g,bulk R T / p_g.
```

`dot n_g,bulk` is in mol/(m3 bulk s), so `S_alpha` is in 1/s. This relation
assumes an ideal gas at fixed temperature and pressure for that source step.
It does not turn all chemical production into free gas, supply a gas-mass
ledger, or capture compression and thermal expansion. The inventory-based
path instead obtains gas volume from species amounts and the chosen EOS,
with the appropriate absolute phase pressure and capillary jump applied once.

Liquid volume follows physical liquid mass, composition, temperature, and the
supplied liquid EOS. Reaction and phase transfer may change both liquid and
gas volumes; continuity uses their combined material-volume change. Keep true
liquid material volume, dispersed gas occupancy, and resolved gas occupancy
distinct; each backend separately declares its transport carrier. A scalar
void model, a dispersed population model, and a resolved interface must not
each independently own the same gas amount or volume source.

## Heat, momentum, and boiling

External heat deposition is independent of gas production. Zero external heat
can coexist with a composition-driven reaction, and heating alone need not
produce gas. The provider declares one energy convention:

- With sensible-energy storage, supply only the chemical-energy contribution
  excluded from the stored species energies. Distinguish an enthalpy or
  temperature formulation from conservative internal-energy storage; use
  consistent formation enthalpies or internal energies and account for
  pressure work without duplicating composition-dependent sensible energy.
  Include phase-change/transfer heat exactly once under that same convention.
- With formation energy included in thermodynamic species energies, let the
  conservative composition change account for it; do not add the same heat
  of reaction again.

For native phase-U storage and recovery, the thermal equation and EOS must use
the same caloric reference, storage potential, pressure-work convention, and
source units. A caller-supplied net
thermal budget already includes its declared reaction-energy allocation.
Driving energy used to evaluate a rate or yield is not automatically another
heat source. Report external heating, reaction allocation, latent exchange,
boundary enthalpy, and unresolved energy terms separately.

Gas phase changes and external injection carry their physical mass, momentum,
and enthalpy. Internal transfers have equal and opposite contributions.
Prescribed slip, subgrid breakup/coalescence, and unresolved surface energy
need explicit energy closures before total-energy conservation can be claimed.
Boiling remains a separate solvent phase-change mechanism: distinguish vapor
from reaction products and avoid counting latent heat or occupied gas volume
twice. A combined steam/reactive-gas EOS and condensation law require their
own qualification; existing boiling coverage does not establish that support.

## Transport and accepted-state transactions

Each conservative inventory uses its declared phase/carrier flux exactly once.
In the field adapter, Liquid follows the carrier and solution_transport selects
whether Dissolved follows it too; named dispersed populations use their common
species/count carrier-plus-slip flux. The field Resolved bucket is stationary:
zero carrier with conservative ALE storage, without resolved-interface advection.
Native low-Mach phases instead use their explicitly owned physical-phase fluxes.
Species diffusion requires the backend's declared mixture-mass and energy owner;
field bulk diffusion remains rejected. Geometric interface capture and
VoF/dispersed representation conversion remain deferred. Explicit local
species-compartment transfers do not reconstruct an interface. Any future
representation conversion must avoid duplicate chemical sources or false
physical escape.
Only physical boundary discharge or an explicitly budgeted external withdrawal
(including a vent reservoir) removes material from the domain.

Every corrector starts from the same accepted state and recomputes sources for
the current trial. The controller validates schema and maps collectively,
assembles actual transport, then previews reactions and ordered transfer from that transported
predecessor and closes material-volume/thermal effects. Acceptance requires all
physical gates. Commit publishes geometry, fields, time, source budgets and
histories once; it does not invoke provider laws or publish external mutable
provider state. Rejection restores them together; retained stale
trials cannot be committed after reinitialization or a schema change.

For ALE, use accepted-old/trial-new volumes and mesh-relative transport fluxes.
Reevaluate liquid-based rates on the trial liquid volume, or preserve already
integrated extensive source budgets through mapping. Do not rescale an
integrated amount as a fresh volumetric rate on every corrector. Geometry
changes invalidate geometry-dependent closure and transport caches. Any future reacting-flow regridding integration must
remap species, mass, momentum, energy and population moments conservatively
at accepted boundaries before obtaining the next source context. The current
fixed-topology ALE path does not qualify adaptive reacting remap.

MPI ownership, canonical face orientation, empty-rank participation, shared
interface transfers, and collective failure order follow the existing solver
contracts. These requirements do not extend the current ALE or general-mesh
support matrix; each backend must qualify the new source interface separately.

## Numerical and physical acceptance gates

1. Verify the immutable species/reaction schema, alternative rate/integrated
   input forms, volume basis, donor accounting and independent energy contract
   with multiple ordinary species and nontrivial catalog IDs.
2. Verify the field, population and controller adapters against that contract.
   Missing required schemas, retired keys and unsupported combinations reject
   collectively. General examples supply their own finite manufactured material
   laws; matched reference applications retain their explicit chosen physics.
3. Verify a closed-cell analytic reaction such as `A_l -> B_l + G_d` with
   `W_A = W_B + W_G`. For a first-order law check
   `n_A(t)=n_A(0) exp(-k t)`, product stoichiometry, and total mass/element balance.
   Test competing donors, reverse reactions, zero source, and exhausted donors.
4. Test liquid-volume versus bulk-volume scaling, integrated-source units,
   zero-liquid cells, independent heat-only and reaction-only cases, and no
   duplicate reaction heat. Verify gas EOS volume and the coupled continuity
   target with independently calculated phase changes.
5. Check dissolution equilibrium and reversal using partial pressure, finite
   donor transfer, nucleation and population moments, transport/escape ledgers,
   boiling interaction, and conservative resolved/dispersed conversion.
6. Require serial and actual multi-rank inventory closure, empty owners,
   partition-independent sources, forced rejection, checkpoint replay, stale
   trial rejection, and supported ALE/remap conservation. Repeated correctors
   and retries must not consume a budget multiple times.
7. Qualify material/rate data and physical cases separately from numerical
   tests, including spatial/time convergence and the intended pressure,
   temperature, concentration, bubble-size, and turbulence ranges.

No historical test log or roadmap checkmark substitutes for these gates.
Architecture documents define each implemented support/rejection matrix.
Numerical tests use supplied manufactured closures; experimental calibration,
external comparison and physical validation remain separate requirements.
