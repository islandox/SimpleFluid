# TODO: SimpleFluid Multiphysics Roadmap

**Workstream scope:** staged VoF source development on the generic
reacting-flow foundation. The roadmap records which interface and AMR
components are present in this snapshot. Retained source does not establish
supported generic reacting-VoF execution or physical qualification.

**Experimental VoF:** `SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF=ON` builds the retained
VoF solvers for source-free development and focused qualification; the default
remains `OFF`. The unmigrated `ReactingVoFCoupler` and its dependent fixtures stay
excluded. This option does not enable multispecies reacting-VoF integration or
establish full-flow physical qualification. Shared geometry and FVM utilities
remain available independently.

**Documentation cleanup date:** October 29, 2026

Completed work is summarized here; detailed contracts and recorded evidence live
in `docs/modeling/`, `docs/architecture/`, `docs/verification/` and local
`docs/qualification/` reports. This cleanup changes no implementation or
qualification status and includes no new runtime or full-suite results.

**Near-term goal:** qualify supplied reacting-flow materials and closures,
external-source protocols and applications against the implemented generic
species/donor/energy contracts. Preserve the supported weak Boussinesq and
conservative native low-Mach ownership boundaries. The current model is not a
full Euler–Euler two-fluid solver.

## Roadmap dashboard

The numbered phases record dependency and implementation history; they are not
the current execution order. Work from the priorities below, and use each
phase's unchecked tasks and acceptance criteria as its completion contract.

### Current priorities

The retained Phase 22/23 VoF sources are deferred, including bounded floating-point
and exact-rational execution paths. Their historical component checkmarks below
do not enable a supported reacting-VoF application. Independent hierarchy,
geometry/remap and mesh-partition infrastructure remains available within its
own documented scope; production scaling and broader physical qualification
remain open.

1. Qualify supplied reaction/property laws and physical cases against the
   implemented Phases 12/14.1 species, donor, phase and thermal contracts.
2. Qualify the authenticated pitzDaily acceptance manifest with converged,
   checked-in OpenFOAM and SimpleFluid profiles and evidence-based tolerances.
3. Decide whether Phase 14's optional scalar-void transport/slip path is still
   needed now that Phase 14.1 owns bubble transport; implement it or explicitly
   defer it.
4. Add Phase 16 demo validation checks for ParaView-readable output and
   disabled-reaction/boiling baseline behavior for the manufactured general demo.
5. Complete the Phase 18 user-facing configuration reference.
6. Add quantitative Phase 14.1 physical validation and a checked-in combined
   RANS-plus-bubble user example.
7. Physically validate and harden the constrained Phase 20.1 planar-ALE path,
   then qualify wider vapor transport/escape, prescribed inlet composition
   and phase-aware conservative coupling before widening its support
   matrix. Reusable cell-field overlap transfer already exists; geometric
   occupancy and coupled inventory acceptance remain open.
8. Close the remaining foundational validation gaps: Ghia profile checks and
   bundled OpenFOAM centerline tolerances.

### Status map

| Workstream | State | Principal open gate |
| --- | --- | --- |
| Foundation, Phases -1 to 8 | Implemented with verification gaps | Ghia and bundled OpenFOAM profile validation |
| Two-equation RANS | Implemented and focused-tested | Quantitative external validation and phase-aware bubbly-flow scope |
| Phase 9, performance | Substantially complete | Backend-portable device assembly API |
| Phases 10–12, sources/materials/reactions | Explicit generic schemas, rates/extents, donors and thermal ownership implemented | Material/rate calibration, stiff chemistry and broader physical cases |
| Phase 13, boiling | Standalone and finite Boussinesq bulk/wall vapor paths implemented | Calibrated wall partitioning and wider phase/thermal models |
| Phase 14, scalar void | Partial | Advection and disposition of the low-order slip path |
| Phase 14.1, bubble populations | Named multispecies populations and accepted event receipts implemented | Physical closure validation, live reacting remap and combined example |
| Phases 15–16, feedback/demo | Partial | Property extensions and demo acceptance checks |
| Phase 17, focused tests | General reaction/species/donor/energy and transaction cases implemented | Physical convergence, wider solver/material combinations and hosted coverage |
| Phase 18, documentation | Partial | Complete key/default/unit/validity reference |
| Phase 19, solution species | Conservative generic species and finite reaction/transfer inventories implemented | Prescribed inlet composition, reacting remap and broader diffusion support |
| Phase 20, external reaction/source maps | Typed in-memory source/receipt callbacks and reusable conservative cell transfer implemented | External-provider protocol and phase-aware mapping |
| Phase 20.1, planar free surface | Fixed-grid path plus constrained solver-integrated planar ALE | Physical validation, steam/composition transport, wider model support, geometric occupancy, and coupled inventory mapping remain open |
| Phase 21, Euler–Euler | Deferred | Requires validated lower-order models first |
| Phase 22, resolved/dispersed bubbles | Retained source: uniform VoF, Cartesian adaptive VoF, resolved/dispersed coupling, representation conversion | Generic reacting integration and physical qualification remain open |
| Phase 23, general native AMR | Retained source: common hierarchy, conservative remap, extruded XY/Z, polyhedral, local products, mixed providers, distributed native migration, general flow | Source presence does not establish full physical or scaling qualification |

### Checklist conventions

- `[x]` means the stated implementation and its cited scope exist; it does not
  imply that broader physical validation is complete.
- `[ ]` means required work or acceptance evidence is still open.
- A phase status statement defines the boundary between implemented behavior
  and remaining acceptance work. Keep that statement synchronized with its
  checkboxes.
- Verify implementation, focused tests, and relevant serial/MPI runtime modes
  before changing a checkbox.

## Foundation and cross-cutting programs

### Implemented foundation — squashed Phases -1 to 8

The foundation is implemented with the verification gaps below. Capability and
support details live in [README](README.md), [mesh decomposition](docs/architecture/orthogonal_decomposition.md)
and the [parallel semi-structured guide](docs/semi_structured_parallel.md).

- [x] Mesh/partition infrastructure, native `MeshHandle`/`FieldStored`, distributed
      fields, VTU/Exodus I/O and configuration utilities.
- [x] Reconstruction, diffusion, scalar/vector transport, momentum and pressure
      coupling; opt-in constant-step BDF2 and bounded linear upwind retain the
      Backward Euler and first-order-upwind defaults.
- [x] Incompressible/Boussinesq flow, temperature and solid conduction, native
      solver execution and legacy object-identity compatibility.
- [x] Analytic/manufactured and flow smoke regressions, OpenFOAM comparison
      tooling, and launchers aligned with current build presets.

#### Known foundational gaps

- [ ] Add quantitative lid-driven-cavity profile checks against Ghia et al.
- [ ] Bundle validated OpenFOAM centerline profiles and enforce tolerances in
      an automated test.

### Compact region composition

- [x] Compact serial/MPI composition, canonical coarse/fine seams, translated
      periodic geometry, native diffusion/flow execution, common axial affine
      ALE, annular/swept providers and explicit region partitioning.

Contracts and recorded scope are maintained in [region meshes](docs/region_meshes.md),
[region execution](docs/region_execution.md), [annular meshes](docs/annular_meshes.md)
and [composite partitioning](docs/composite_partitioning.md). The
[scaling guide](docs/region_scaling.md) defines measurement contracts; retained
local reports under `docs/qualification/` do not establish physical pool-flow
validation or production scaling. Explicit global constituents remain rejected
by compact composition; explicit-region source union/graph construction remains
centralized on the source rank.

- [ ] General mortar/AMI intersections, curved/nonconvex face subdivision,
      wider polygonal volume/output types and rotated periodic field transforms.
- [ ] Nonaffine composite motion and moving periodic lengths: require consistent
      shared interface geometry, exact GCL and full transactional solver tests.

### Two-equation RANS turbulence

This program was added after the numbered multiphysics roadmap. It is tracked
separately so its implemented scope and remaining validation work are not
mistaken for foundation or Phase 9 completion.

#### Implemented scope

- [x] Six transported RANS closures (standard/RNG/realizable k-epsilon and
      standard/BSL/SST k-omega), laminar selection, momentum/thermal coupling,
      wall treatments, distributed wall distance and diagnostics.
- [x] Focused formula/transport/solver/MPI checks, isothermal `pitz_daily`, and
      heated-solution-tank comparison tooling.

Current transport is single-continuum and full-cell; phase-weighted turbulence
remains Phase 21 work.

#### SST-1994 plus SAS source extension

- [x] Opt-in SST-1994 SAS source, supported mesh/boundary/gradient paths,
      rollback-safe optional physics and NOX integration, with focused checks
      and an activation example.

The [SAS model and qualification scope](docs/modeling/sst_sas.md#qualification-status)
record supported combinations and exclusions. Parent SST and pitzDaily
acceptance gates remain unchanged.

- [ ] Establish spatial/temporal resolution and sustained three-dimensional
      scale-resolution evidence beyond source activation.
- [ ] Perform qualified whole-flow external comparisons with explicit parent-SST
      and discretization differences, then quantitative physical validation.

#### Remaining work and acceptance

- [ ] Establish a converged pitzDaily reference configuration and checked-in
      velocity-profile tolerances. The comparator and launcher now fail closed
      unless a qualified physical manifest authenticates retained inputs,
      records exact run settings and rank count, and supplies finite-data,
      sample-count, profile-span, station-offset, and error limits. The
      checked-in physical manifest remains pending because no converged real
      profile pair supports those tolerances yet.
- [ ] Add quantitative validation for every closure before describing the
      turbulence program as validated rather than implemented and tested.
- [ ] Define and validate the applicability range of single-continuum RANS in
      void-bearing cases. Liquid-volume-fraction weighting, interphase
      turbulence transfer, and bubble-induced turbulence belong to Phase 21.

**Status:** substantial implementation with focused serial/MPI tests;
quantitative external-reference validation criteria and a phase-aware
turbulent bubbly-flow formulation remain open.

## Core infrastructure and source phases — Phases 9 to 12

### Phase 9 — Performance and solver-regression benchmarks

**Status:** the benchmark harness and major setup/allocation optimizations are
implemented. Cross-machine qualification and general device assembly remain open.

- [x] Repeatable CTest/manual benchmarks with machine-readable provenance,
      iteration/time/memory/residual measurements, MPI scaling and stored baselines.
- [x] Debug smoke-regression ceilings and compatible graph, geometry, solver,
      preconditioner and VTU reuse; distributed ownership lookup and batched imports.
- [x] Coupled block-composite operators and streamed products with supported
      solver/native/legacy/ALE regressions. See
      [coupled backend contracts](docs/architecture/coupled_operator_backends.md).

Timing ceilings are smoke guards; Release scaling/profiling remains a manual
measurement workflow. Local results do not qualify accelerator execution.

- [ ] Migrate the remaining host-side finite-volume row assembly loops to
      backend-portable Kokkos kernels after defining a device assembly API.
- [ ] Revisit scoped geometry execution in
      `PlanarALEMeshMotion::compute_trial_state` only with stronger application
      evidence. The reverted C2 trial-read lease passed parity but failed the
      declared wall-time gate (at least 1% lower median and nonoverlapping ranges).
      Preserve arithmetic, traversal, collective failures and physical gates;
      inclusive phase timing alone is insufficient. Patch and decision evidence:
      external-host local `build/next-optimization-20261010/` (`C2.patch`,
      `C2-comparison.json`, `C2-decision.json`, `report.json`).
- [ ] Add native scalar face-based orthogonal diffusion and frozen upwind/BE
      transport, with independent operator/diagonal/preconditioner inputs.
- [ ] Qualify coupled memory/performance at larger converged sizes and on
      accelerator builds; separate preconditioner/Krylov allocation accounting.

### Phase 10 — Source-term and material-property infrastructure

- [x] Generic scalar/thermal sources, updateable material properties,
      validated configuration, initialization and VTU output, with analytic
      heat-source, invalid-input and disabled-model equivalence checks.

### Phase 11 — Prescribed and reaction heat-source coupling

- [x] Independent `PrescribedHeatSource` profiles/imports, total-power
      normalization, time multipliers and output; retired source keys reject.
- [x] Accepted-extent reaction heat and separate external, latent and carried
      thermal channels, with focused donor/energy/rollback checks.

See [source and heat ownership](docs/modeling/reacting_gas_sources.md) and the
[species energy contract](docs/architecture/reacting_species_contract.md).
Physical material and multiphysics qualification remains open in Phase 12.

### Phase 12 — General reacting gas source in solution

- [x] Immutable species/reaction schemas, supplied rates/extents, finite donors,
      accepted material/thermal ledgers and weak/native low-Mach adapters.
- [x] Componentwise conservative transport and bounded refinement for absent
      gas/tiny arrivals, with focused serial and two-/four-rank rollback checks.

The [source design](docs/modeling/reacting_gas_sources.md),
[species contract](docs/architecture/reacting_species_contract.md) and
[native transport contract](docs/architecture/low_mach_reacting_flow.md#componentwise-transport-consistency)
define units, ownership, support and rejection gates. Manufactured-law checks
do not calibrate physical chemistry or qualify untested combinations.

- [ ] Improve or explicitly bound large-volume/small-timestep EOS-minus-advected-volume
      cancellation while retaining independent physical continuity and occupancy gates.
- [ ] Add a separately qualified stiff-chemistry/Jacobian integration backend.
- [ ] Calibrate supplied reaction and property data and verify physical
      pressure/temperature/composition ranges with spatial/time convergence.
- [ ] Qualify additional mesh, inlet, source and multiphysics combinations before
      relaxing the documented support/rejection matrix.

## Active multiphysics phases — Phases 13 to 20

### Phase 13 — Bulk and wall boiling source model

**Status:** explicit bulk/wall boiling and latent-energy coupling are implemented;
RPI-style heat-flux partitioning remains open.

- [x] Bounded boiling sources, configuration/output and focused threshold,
      latent-energy, conservative wall-distribution and disabled-path checks.

Equations, finite-vapor ownership and supported combinations are maintained in
[reacting gas and boiling](docs/modeling/reacting_bubble_boiling.md) and the
[Boussinesq finite-vapor contract](docs/architecture/bouss_finite_vapor.md).

- [ ] Leave a clean interface for future wall heat-flux partitioning:
  - [ ] convective heat transfer
  - [ ] quenching heat transfer
  - [ ] evaporative heat transfer

### Phase 14 — Low-order void-fraction field update and optional transport

- [x] Standalone bounded scalar source/collapse update, conservative diffusion,
      synchronized gas fields and state-derived `S_alpha_total` output, with
      analytic, boundedness and zero-flux conservation checks.

The [standalone scalar contract](docs/modeling/reacting_bubble_boiling.md#standalone-scalar-controls)
defines this low-order approximation. It has no advective population balance
and rejects nonzero reserved slip; operational bubble slip belongs to Phase 14.1.
Generic species own their material and accepted void publication. Scalar
clipping cannot establish species/energy conservation; collapse needs an
explicit dissolved, headspace or external sink ledger.

- [ ] Connect accepted general reaction sources and conservative collapse/sink
      ledgers without using post-update clipping to satisfy inventory gates.
- [ ] Add low-order scalar advection with liquid face fluxes.
- [ ] Add configurable upward slip velocity aligned opposite gravity.
- [ ] Advective transport conserves total gas inventory up to boundary flux and sources.

### Phase 14.1 — Dissolved-gas transport and bubble populations

- [x] Schema-indexed species and named populations, declared carrier/slip
      transport, EOS/radius and absolute-pressure handling, finite joint donors,
      accepted events and fixed/ALE rollback transactions.
- [x] Native pressure-conditioned interphase thermal stages and weak RANS/SAS
      species coupling within their separate support matrices.

See [reacting gas sources](docs/modeling/reacting_gas_sources.md),
[native low-Mach flow](docs/architecture/low_mach_reacting_flow.md) and
[interphase thermal ownership](docs/architecture/reacting_interphase_thermal.md).
The field adapter retains weak sensible-temperature storage; the native solver
owns phase internal energy and pressure work. Prescribed slip supplies no gas
momentum equation, and numerical checks do not establish physical validation.

- [ ] Qualify material/rate, nucleation, film, slip and boiling closures against
      physical gas-release/bubble-column data and mesh/timestep convergence.
- [ ] Add a combined RANS/reacting-gas user example and phase-aware turbulence.
- [ ] Add implicit stiff kinetics and nonlocal pressure-dependent interphase
      transfer; qualify simultaneous implicit interface laws separately.
- [ ] Extend allocated local updates toward allocation-free device execution.
- [ ] Qualify live reacting repartition/remap and general interface representation
      changes. VoF and bounded-FP/exact-rational paths remain deferred.

### Phase 15 — Temperature- and void-dependent feedback properties

- [x] Constant, Boussinesq and mixture-density feedback; constant viscosity,
      positive floors, momentum coupling and VTU output, with focused limiting
      behavior and disabled-model checks.

See [material feedback](docs/modeling/reacting_bubble_boiling.md#boussinesq-support-and-rejection-limits).
Temperature/void viscosity extensions and natural-convection acceptance remain open.

- [ ] Add temperature-dependent viscosity if existing support is available.
- [ ] Add an optional void correction to viscosity.
- [ ] Natural-convection benchmark remains stable with temperature-dependent density enabled.

### Phase 16 — Reacting-solution tank demonstration

- [x] Manufactured finite-reactant/two-gas tank and cylinder demos with explicit
      material/source/energy ownership, independent heating and accepted output.

`reacting_solution_tank_demo` and `constant_power_cylinder_vessel` retain their
documented manufactured inputs; H2/IF97 reference applications retain explicit
reference materials and external-reservoir laws. See
[source inputs and examples](docs/modeling/reacting_gas_sources.md#current-interfaces).

- [ ] Verify output opens in ParaView.
- [ ] Verify disabled reactions/boiling recover the independently heated
      natural-convection baseline over a complete physical run.
- [ ] Add matched serial/MPI and mesh/timestep convergence before presenting
      any example as physical validation.

### Phase 17 — Unit, regression, and conservation tests for reacting gas and boiling

- [x] Focused species/donor/count/energy, boundary/empty-owner, fixed/ALE,
      rejection/replay and API/export checks within documented configurations.
- [x] Recorded 1008-cell, scale-1 bottom-convection NOX startup through two serial
      steps with GMRES/BiCGStab and pressure-Schur AMG/SGS under unchanged gates;
      this does not qualify developed flow, longer runs or MPI application behavior.

See [reacting acceptance gates](docs/modeling/reacting_gas_sources.md#numerical-and-physical-acceptance-gates)
and [NOX validation scope](docs/architecture/coupled_nonlinear_solver.md#validation-and-remaining-qualification).
Skipped or omitted tests are not passes; focused local checks are not full-suite
or hosted CI results.

- [ ] Qualify broader material, toolchain, solver and mesh combinations and hosted
      coverage without treating local focused tests as a full-suite/CI result.
- [ ] Complete independent physical validation and convergence campaigns.

### Phase 18 — Documentation and configuration examples

- [x] Governing quantities, storage/energy ownership, supplied providers,
      migration/rejection rules and model limits in the
      [source design](docs/modeling/reacting_gas_sources.md) and architecture guides.

The complete key/default/unit/validity reference and runnable user coverage remain open.

- [ ] Complete the [external host companion migration](docs/architecture/external_host_reacting_flow.md),
      including finite species/isotope feedback and a supported bulk Winter-diffusion
      owner before claiming parity with the removed reacting APIs.
- [ ] Document every supported configuration key with default, SI unit,
      validity range and migration mapping; retired source keys must reject.
- [ ] Provide runnable zero-reaction, closed-batch and convecting-solution
      examples with auditable source and property inputs.
- [ ] Ensure a developer can configure each supported model from the user
      reference without consulting implementation headers.

### Phase 19 — Solution-species transport and reaction inventories

- [x] Conservative solution-species and gas transport, shared accepted source
      events and fixed/planar-ALE transactions; separate scalar source/decay utility.

Liquid/Dissolved species share one physical solution phase while retaining
individual amounts. The [species contract](docs/architecture/reacting_species_contract.md)
and [native flow guide](docs/architecture/low_mach_reacting_flow.md) define
carrier, boundary and thermal limits; the standalone scalar utility supplies no chemistry.

- [ ] Qualify prescribed inlet compositions, wider boundary exchange and
      transport/reaction convergence for physical material laws.
- [ ] Qualify generalized field-adapter bulk-mixture diffusion and broader
      nonlocal thermal transport. Pure/native face kernels do not establish
      support in every adapter.
- [ ] Add reacting conservative remap/repartition and broader ALE only with
      species, mass, energy, momentum and count closure.

### Phase 20 — External reaction/source coupling and feedback maps

- [x] In-memory heat/source callbacks, feedback snapshots and subcycle scaffold;
      typed schema/unit/location/interval/accepted-receipt contracts.
- [x] Scalar/vector/tensor cell transfer, MPI donor exchange, supported conservative
      overlaps and explicit coverage/geometry invalidation diagnostics.

See [cell transfer](docs/mesh_to_mesh_transfer.md) and
[external-host contracts](docs/architecture/external_host_reacting_flow.md).
Spatially mapped chemistry and a production provider protocol remain open.

- [ ] Exchange pure-liquid composition/occupancy and absolute pressure without
      substituting void-reduced mixture density or gauge pressure.
- [ ] Conserve species and heat over spatial maps and subcycles, including
      partial coverage, provider rollback and exact-once accepted source use.
- [ ] Validate zero-source behavior, rate-basis conversions, stale maps,
      species-order changes and rejection/replay in serial/MPI.
- [ ] Select and validate a production transport/protocol only when a concrete
      external reaction/source provider is selected.

### Phase 20.1 — Planar free-surface volume budget and constrained ALE

**Status:** fixed-grid volume bookkeeping and a constrained solver-integrated
planar-ALE core are implemented with conservation-focused tests. Physical
validation, wider vapor support and conservative occupancy/mapping remain open.

The [model, configuration and support matrix](docs/modeling/planar_free_surface_volume_budget.md),
[species/surface contract](docs/architecture/reacting_species_surface.md) and
[coupler ownership](docs/architecture/multiphysics_solver_coupler.md) own the
detailed equations, transactions, diagnostics and verification scope.

#### Milestone A — implemented fixed-grid components

- [x] Vessel maps, reference/cellwise/species-bound liquid inventories, supplied
      gas/headspace closure and exact accepted escape/phase-change accounting.
- [x] Atomic acceptance/rollback, diagnostics, history/CSV, opt-in VTU/feedback
      and analytic verification drivers, with focused serial/MPI coverage.

Cell-centre occupancy remains approximate. Reference `cellMassInventory`
requires error depletion and zero physical-boundary liquid flux; typed solvent/
solute following does not qualify every evaporation or inlet law.

- [ ] Add conservative steam transport and an exact steam escape transfer
      before sending boiling vapor to the headspace/vent.
- [ ] Couple boiling saturation to absolute headspace pressure before allowing
      active boiling with a closed headspace; retain the explicit setup
      rejection while saturation temperature is fixed.

#### Milestone B — constrained solver-integrated planar ALE core

- [x] Transactional geometry/GCL, old/new-volume and relative-flux transport,
      geometry-epoch cache refresh, generalized continuity, moving-top/escape
      ownership and whole-step rollback for the supported matrix.

The integrated scope is native mutable structured/extruded geometry,
constant-area vessels, Backward Euler, laminar temperature, species-bound
liquid mass, vented headspace and supported population transport/escape.
Unsupported model combinations reject; geometry-only support does not widen
this solver contract.

- [ ] Complete the whole-solver fault-injection matrix for an actual
      mesh-quality rejection, deliberately corrupted GCL, volume-source and
      liquid-mass gates, and rollback with nonzero pending escape. Current
      coverage combines component-level quality/GCL/source failures with
      whole-step motion-limit, momentum-solve, pressure-solve, outer-corrector,
      and gas-closure rollback tests.
- [ ] Add quantitative physical validation for thermally driven level motion
      and moving-top gas escape; component and conservation regressions do not
      satisfy this gate.

#### Milestone C — feedback and conservative mapping

The reusable [cell-field transfer](docs/mesh_to_mesh_transfer.md) supplies
supported geometric overlaps and intensive/extensive inventory reports.
The remaining tasks below concern phase occupancy, composition, and integrated
physical closure; generic cell transfer alone does not complete them.

- [ ] Replace or augment the diagnostic cell-centre occupancy with geometric
      cut fractions or a demonstrably conservative mapper; do not call the
      current Heaviside field conservative.
- [ ] Export pure-liquid composition and occupancy without using void-reduced
      mixture density to derive material number density.
- [ ] Preserve solvent/solute mass, dissolved and gas species, and energy
      under same-topology motion and cross-mesh mapping, with global residuals.
- [ ] Validate pool-volume and extensive-inventory closure across supported MPI
      partition counts.

Full VoF interface capturing and macroscopic surface breakup belong to the
separate Phase 22 program below; Euler–Euler gas momentum remains Phase 21.
Completing this planar phase must not imply those capabilities.

## Deferred architecture — Phase 21

### Phase 21 — Full Euler–Euler two-fluid extension

This phase is deferred until the generalized scalar-source and population
paths are implemented, verified and physically qualified.

**Status:** deferred by design; none of its implementation or validation
criteria are claimed complete.

The ability to configure RANS and supplied multispecies bubble populations in the same
`BoussinesqSolver` does not complete this phase. The current bubble populations
follow liquid flux plus prescribed/correlated slip, void feeds momentum only
through loose material-density/buoyancy feedback, and the RANS equations are
not liquid-volume-fraction weighted.

- [ ] Add separate gas velocity field `U_g`.
- [ ] Add gas-phase continuity equation.
- [ ] Add liquid/gas momentum coupling.
- [ ] Add interphase drag model interface.
- [ ] Add candidate drag laws:
  - [ ] Schiller–Naumann
  - [ ] Tomiyama correlated
  - [ ] Ishii–Zuber
- [ ] Add lift, virtual mass, turbulent dispersion, and wall-lubrication hooks.
- [ ] Add gas/liquid heat-transfer closure.
- [ ] Add optional bubble diameter field.
- [ ] Add optional full size-distribution population-balance model.
- [ ] Formulate liquid-phase turbulence transport with consistent phase-volume
      and density weighting and recover the existing single-phase limit.
- [ ] Add bubble-induced turbulence and interphase turbulence-transfer model
      interfaces with documented validity ranges.
- [ ] Validate against a bubbly-plume or bubble-column benchmark over the intended reacting-solution operating range.

#### Phase 21 acceptance criteria

- [ ] Scalar `alpha_g` model remains available as a lower-order option.
- [ ] Two-fluid equations conserve phase volume/mass under controlled tests.
- [ ] Drag-law unit tests reproduce expected limiting behavior.
- [ ] Phase-aware turbulence equations recover the current RANS solutions as
      `alpha_g` approaches zero.
- [ ] Bubble-induced-turbulence and interphase-transfer closures reproduce
      selected reference limits.
- [ ] Full model is validated against at least one independent bubbly-flow benchmark.

## Phase 22 — Resolved and dispersed reacting-gas bubbles

**Status:** Stage 1's serial uniform backend and Stage 2's distributed Cartesian
AMR backend are implemented and focused-tested. Stage 3 adds numerical
low-Mach single-gas source and resolved/dispersed interface coupling. Stage 4 adds
opt-in conservative representation trials with a persistent birth registry.
Adaptive levels share one timestep. The target is centimetric resolved gas in
reacting solutions. Existing implementation/checkmarks below refer to the
legacy single-gas substrate. General species/rate integration, physical
calibration, application resolution studies and the listed extensions remain
open; numerical tests alone do not qualify reacting-solution transients.

### Physical and ownership contracts

- [x] Separate resolved/dispersed/liquid inventories and accepted material,
      momentum/thermal and representation histories with external solver ownership.

The retained [single-gas contract](docs/modeling/reacting_vof.md) and
[transport/state contract](docs/architecture/vof_transport.md) define these
historical components. `ReactingVoFCoupler` and dependent fixtures remain
excluded even when experimental source-free VoF is enabled.

- [ ] Generalize the legacy single-gas substrate to the species/rate and shared
      donor/heat contracts in [Phase 12](#phase-12--general-reacting-gas-source-in-solution)
      and [Phase 14.1](#phase-141--dissolved-gas-transport-and-bubble-populations).
      Verify each species through dissolved/dispersed/resolved/headspace storage,
      source acceptance, representation changes, adaptation and MPI exchange.

### Stage 1 — Uniform-mesh geometric VoF

- [x] Serial uniform Cartesian geometric PLIC/transport, variable-density flow,
      capillarity, timestep control and atomic failure handling with focused checks.

See [the solver guide](docs/modeling/vof.md) and
[verification contracts](docs/verification/vof_solver_contracts.md). High-density
and capillary checks do not qualify long-time parasitic currents, viscous rising
bubbles or reacting solutions. The two-rank check verifies rejection, not MPI flow.

### Stage 2 — Conservative three-dimensional AMR

- [x] Distributed Cartesian 2:1 AMR, conservative PLIC/extensive remap,
      refinement policies, shared-timestep transport and atomic regrid/replay.

The [adaptive guide](docs/modeling/adaptive_vof.md) and
[numerical verification contracts](docs/verification/vof_solver_contracts.md#numerical-assertions)
maintain the numerical scope, including lower-contrast curved-seam passage
and finite-duration static bubbles with a uniformly fine interface band.

- [ ] Qualify active curved interfaces at high density contrast directly on a
      coarse/fine seam. The retained strict qualification probe remains disabled;
      short fine-band static and lower-contrast moving cases do not close it.
- [ ] Qualify prism subdivision and vessel-boundary geometry through the
      separate [Phase 23](#phase-23--amr-for-native-extruded-polyhedral-and-mixed-region-meshes)
      geometry and solver gates.
- [ ] Perform finer centimetric-bubble resolution studies. Interface resolution
      alone is not wake DNS or mass-transfer resolution; the startup pressure
      comparison does not qualify rise speed, shape dynamics, or long-time
      parasitic currents.
- [ ] Add optional level subcycling only with conservative flux-register
      corrections. Synchronous levels currently use the same canonical
      subface flux directly and require no time-subcycling flux register.

### Stage 3 — Reaction-source and interface coupling

- [x] Retained single-gas source/population, prescribed-film, low-Mach
      mass/momentum/energy, capture and vent ledgers with focused transaction checks.

See [the retained coupling contract](docs/modeling/reacting_vof.md) and
[numerical verification contracts](docs/verification/vof_solver_contracts.md#numerical-assertions).
These records do not enable generic reacting-VoF integration or establish
physical calibration.

- [ ] Supply and qualify aqueous growth, breakup, coalescence, capture and
      film-thickness calibrations over the intended size/flow/temperature
      range. A shared slip selector does not automatically qualify both
      microbubbles and upper populations; per-population physical closure
      selection remains application work.
- [ ] Qualify high-Peclet centimetric-bubble transfer, source/time/mesh
      convergence and grid-scale coalescence sensitivity. The prescribed
      film is not the full reconstructed boundary-layer SGS algorithm.
- [ ] Qualify application caloric/chemical-energy references and any required
      surface-energy, drag or viscous-heating closures. Variable surface
      tension/Marangoni stresses and a planar-ALE extension remain separate
      implementation and qualification gates.

### Stage 4 — Bidirectional representation changes

- [x] Retained birth registry, size/resolution hysteresis, complete-component
      selection, conservative zero-time conversion and subsequent physical
      relaxation, with distributed ownership and replay checks.

The [representation guide](docs/modeling/bubble_representation.md) and
[numerical verification contracts](docs/verification/vof_solver_contracts.md#numerical-assertions)
define the supported numerical scope and exclusions.

- [ ] Qualify birth localization, demotion host-volume assumptions and physical
      threshold sensitivity for the intended aqueous bubble population.
- [ ] Extend wrapped conversion stencils, dispersed inert species and
      incompatible carried-temperature mixtures beyond the current defer/
      reject policies. General advected material-lineage tracking remains
      separate from the persistent birth registry.
- [ ] Use Hysing-style algorithm checks followed by three-dimensional
      centimetric-bubble convergence of rise speed, shape, wake, and gas
      transfer. Coupled reacting-solution qualification is an additional application gate.
      The original Hysing no-slip horizontal walls are outside the present
      adaptive slip-wall contract.

## Phase 23 — AMR for native extruded, polyhedral, and mixed-region meshes

**Status: Stages 1–6 implemented and verified within the stated scope; Stage 7
implemented and under qualification; Stages 8–9 remain proposed.** See the
[general AMR design](docs/architecture/general_amr.md) for the algorithms,
ownership/API contracts, admission rules, and focused acceptance matrix.
Geometry refinement and VoF/physical-coupling qualification are separate gates.
Preserve the corrected Cartesian [transport/state contracts](docs/architecture/vof_transport.md)
and [numerical verification gates](docs/verification/vof_solver_contracts.md)
throughout; each non-Cartesian algorithm also needs its own focused and physical gates.

### Stages 1–6 — Hierarchy, geometry and distributed migration

- [x] Common hierarchy/identity contracts, immutable trial state and Cartesian adapter.
- [x] Certified actual-cell geometry and ancestry-based conservative remap.
- [x] Whole-product XY/Z and convex polyhedral refinement/coarsening.
- [x] Mixed-region closure and compact local XY/Z products.
- [x] Detached distributed planning/migration, sparse native packets, measured
      work balancing and focused empty-owner/collective-failure checks.

Stages 1–6 are consolidated in the
[general AMR scope](docs/architecture/general_amr.md#qualification-status) and
[stage acceptance matrix](docs/architecture/general_amr.md#implementation-stages-and-focused-acceptance).
Bounded bootstrap/catalog and interface-tile costs remain; focused geometry,
remap and migration coverage does not establish scalable planner storage/RSS,
production performance, sparse ALE or general physical-flow qualification.

### Stage 7 — General VoF and flow qualification

The transport roadmap has two explicit paths, independent of the arithmetic
selector. `SweptPartition` remains the default; this is not a migration plan
that replaces geometric sweeps with the opt-in face backend. Both paths share
source-pinned phase/momentum preparation and collective rejection. See the
[shared transport contract](docs/architecture/general_vof_transport.md).

#### Shared foundation — repair Stage 1

- [x] Source-pinned immutable phase/momentum ledgers, bounded collective
      failure records and reusable compatible momentum preparation; each
      component retains its independent physical residual/iteration gate.

See [prepared transport and solver consumption](docs/architecture/general_vof_transport.md#one-detached-finite-ledger).

#### Path A — algebraic face-volume transport

- [x] Opt-in `DonorCell`/`GeometricFace` transport, measure-only PLIC candidates
      and explicit face-aligned surfaces within the
      [fixed-mesh face transport contract](docs/architecture/conservative_face_vof.md).

- [ ] Qualify this path against the shared motion/accuracy gates below, including
      high-density-ratio drop translation and rotation/deformation.
- [ ] Qualify open boundary interfaces and multiple physical streams per face.

#### Path B — geometric swept-tetrahedra transport

- [x] Ordinary homogeneous specialization (repair Stage 2), interface-local
      signed face ledgers (Stage 3) and resolved aggregate geometry (Stage 4),
      retaining independent phase budgets and unchanged momentum/physical gates.

The [transport design](docs/architecture/general_vof_transport.md) owns
trajectory/support authority, aggregate accounting and focused checks.
Unresolved approximations require the explicit ledger-debit work below;
BoundedCertification and Exact retain separate paths and qualification.

#### Shared surfaces, interfaces and qualification — repair Stages 5–6

- [x] Unified capillary surfaces (repair Stage 5), checked prepared interfaces
      and provider/build boundaries (Stage 6): see the
      [capillary model](docs/modeling/general_vof_capillarity.md) and
      [implementation boundaries](docs/architecture/general_vof_implementation.md).
- [x] Ordinary support/contact predicates, construction replay, indexed face
      candidates and face-local storage under the
      [transport contract](docs/architecture/general_vof_transport.md).
- [x] Directional PLIC admission and the provisional 5% volume-equivalent
      cell-length threshold for angular optimization, retaining strict offsets
      and gradient normals: see [PLIC](docs/modeling/plic.md).
- [x] Default budgeted cleanup for Ordinary `SweptPartition`/`GeometricFace`,
      with cutoff-zero opt-out, separate mass/momentum corrections and unchanged
      strict PLIC/raw physical gates. See [phase cleanup](docs/modeling/phase_cleanup.md);
      coupled remap and complete solver restart remain outside its policy.
- [x] Recorded 50-cell ten-step passage with cleanup enabled in serial/MPI2/MPI4,
      using a `1e-14` cell-relative cutoff and cumulative caps of `1e-13` times
      each initial global phase inventory, under the original acceptance gates.
- [x] Finite typed PLIC, independent phase ledgers, compatible flow operators,
      support/refinement requests, cache reuse and atomic transaction coverage.
      See [FP implementation and scope](docs/architecture/general_amr.md#floating-point-geometry-and-arithmetic).

**Qualification boundary:** `ArithmeticMode::Ordinary` remains the primary/default
arithmetic; BoundedCertification and Exact are explicit choices independent of
transport backend. The source-free scope is closed, nonperiodic convex-native
geometry. Periodic images, material/species sources, low-Mach thermodynamics,
population drift and representation coupling remain outside this backend.
Full physical qualification remains open. The optional rational-route history
and backlog are maintained in [ExactVoF](docs/architecture/exact_vof.md#recorded-implementation-and-remaining-work),
[ExactTransport](docs/architecture/exact_transport.md) and
[ExactGeometry](docs/architecture/exact_geometry.md).

The following gates retain their original physical tolerances, resource limits
and [complete flow horizons](docs/verification/vof_solver_contracts.md#general-native-amr-and-flow-checks).
Per-rank memory guards are engineering admission limits, not physical constraints.

- [ ] Qualify the original 50-cell skew/coarse-fine passage case with cleanup
      disabled over its ten-step
      serial/MPI horizon under the uncertainty/conservation gates and documented
      storage-lifetime/work policies. Support-aware reach and defining-support
      predicates resolve the captured contacts; face-local storage admits complete
      transport steps, and directional PLIC admission completes the serial
      horizon. Strict offset representability for microscopic minorities still
      blocks that MPI configuration; retain the original volume tolerance.
      Budgeted cleanup is a distinct explicit policy; wider plane representation
      remains optional robustness work for uncleaned inputs. The conservative
      frontier supporting-halfspace screen remains an admission limit for other
      constructions; finite-polygon containment is separate outstanding work.
- [ ] Establish a current native-cell-complex coverage certificate before replacing
      repeated donor-complement subtraction. Bind complete cavity/halo frontiers,
      source containment and complete donor queries to the same geometry/partition
      observations; retain structural coverage for arbitrary public donor sets.
- [ ] Complete the opt-in eight-step analytic closed-motion accuracy probe.
      Reassess it under the current directional-admission policy, retaining the
      complete horizon, independent phase-distribution oracle and original gates
      rather than truncating the case or increasing numerical budgets.
- [ ] Exercise complete homogeneous/hydrostatic, interface-passage,
      translation/rotation/deformation and curved-capillary horizons in serial
      and MPI, including empty owners and mixed/face transitions. Keep existing
      conservation, boundedness, residual and rollback gates. Report accuracy,
      convergence and local work counts separately from those passes.
- [ ] Agree and meet quantitative interface and capillary accuracy/convergence
      thresholds before claiming full physical qualification. Conservation-only
      checks do not establish those thresholds or authorize a default switch.
- [ ] Qualify physical resolution and convergence, including caller/controller
      refinement or reinitialization, reevaluation and exhausted/unavailable
      refinement. Use neither an absolute phase-volume cutoff nor arbitrary
      precision as a substitute for spatial resolution. Mandatory arbitrary
      subcell-phase fit success is retired; preserve accepted mass throughout.
- [ ] Complete full-flow qualification of the implemented default
      [finite arithmetic policy](docs/architecture/general_amr.md#primary-floating-point-policy)
      through the shared motion and accuracy gates above; this is the arithmetic
      scope of that qualification, not a separate campaign.
      Ordinary FP uses represented values and physical residual tolerances;
      interval certification remains an explicit mode. Mode tags and source
      provenance are separate from imported native authority. Numerical finite
      geometry and ledgers use no rational construction/measure or fallback.
      Count native authority imports separately. Keep explicit Exact work
      separate, and retain independent inventories, physical tolerances and caps.
- [ ] Complete matched candidate measurements against the recorded full-horizon
      Release baseline: fragment/pair work, certificate reuse,
      exact operations by predicate/construction/measure/ledger, operand sizes,
      bounded retries/rejections, per-rank compute/wait and peak memory. The
      serial/MPI2/MPI4 baseline passed; candidate full-horizon qualification is
      still open. Reference/oracle work must be separately attributable;
      sampling alone is insufficient.
- [ ] Qualify canonical FP intersections/support, volume evaluation, only required
      first moments, and immutable geometry/measure reuse with epoch invalidation.
      Propagate construction/arithmetic error and allow only bounded FP retries
      under the FP arithmetic policy; unresolved arithmetic rejects.
      Preserve ordinary-volume signed support, complementary fits and phase-aware
      accuracy. No exact result for each positive intermediate piece is required;
      bounded aggregate accounting must retain all contributions without mass loss.
- [ ] Extend aggregate geometry to unresolved contact-like contributions only
      with conservative bounds and explicit shared face/cell/phase ledger debits.
      Resolved Ordinary aggregate witnesses already avoid per-child metric
      admission; unresolved approximations still reject. Keep signed streams,
      independent inventories, momentum and replay consistent, and retain separate
      moment/section requirements for surface-sensitive consumers.
- [ ] Remove certification-only communication and closure preparation from
      Ordinary projection; index pure-cell curvature support and qualify its work
      with surface tension enabled.
- [ ] Qualify the implemented structural partition evidence, bounded closure
      and lazy moment evaluation across full physical horizons. Finite face
      rates, inventory calculations and residuals use scaled/compensated FP with
      propagated bounds. Preserve canonical shared transfers, nonnegative amounts
      and unchanged physical acceptance; no clipping, inventory renormalization
      or policy switching. Keep rational geometry out of finite caches/MPI payloads.
- [ ] Complete the [FP geometry acceptance gates](docs/verification/vof_solver_contracts.md#floating-point-acceptance-gates):
      independent exact/analytic oracles, resolution/refinement and unit invariance,
      MPI/shared-face and coarse/fine agreement, conservative aggregate accounting,
      boundedness, atomic rejection and full physical horizons. Assert zero
      rational construction/measure/ledger operations in finite numerical steps,
      with native authority imports and selected reference work counted
      separately. Compare matched Release serial and 2/4-rank runs with empty
      owners, operation counts, operand sizes and memory.
- [ ] Complete a coherent physical campaign on the integrated source, including
      reflected-cylinder capillarity and finite-duration viscous decay. Later
      optimization campaigns did not repeat those two long methods.
- [ ] Qualify high-contrast curved interfaces directly on refinement seams,
      nonzero-time regrid momentum across the intended mesh families, and
      long-horizon spatial/time convergence before claiming general physical use.
- [ ] Resolve the recorded Release failure in
      `DistributedAdaptiveBalance.MeasuredOperationsDrivePinnedHierarchyMigration`;
      the later selected balance matrix excluded it explicitly.
- [ ] Qualify production runtime and distributed scaling with matched source,
      runtime and complete physical horizons. Small-case speedups and operation
      counts do not establish whole-process memory bounds or production scaling.

### Stage 8 — Reacting-gas and representation integration

- [ ] Extend Phase 22 source/thermodynamic ledgers, film/aperture integration,
      birth supports, donor selection, component inspection, and coupled maps
      to actual native cell geometry. Retain external controller ownership.
- [ ] Rebind required population names collectively at source preparation,
      including same-mesh reinitialization; verify reorder/removal and stale
      trial rejection without misrouting sources or altering accepted state.
- [ ] Preserve liquid-carrier ConstantSlip and populated-side radius-closure
      drift at empty fronts. Use paired accepted-plus-signed-source moments
      for support, requiring finite nonnegative paired results after sources,
      including complete depletion; transport and source-drift timestep
      estimation must share the canonical face rule.
- [ ] Preserve birth IDs/events and accepted inventories while invalidating
      geometry-dependent reservations and labels. Continue blocking topology
      change during active checkpoints or pending representation relaxation;
      exact history-preserving repartition and mesh-inclusive replay are
      separate later gates.
- [ ] Verify source-divergence/occupancy closure, all phase mass/species/energy
      and physical momentum budgets, tiny retained carriers, front speed and
      timestep parity, remote deposition, donor competition, retained-trial
      isolation after rejection, and checkpoint replay. Application calibration
      and centimetric-bubble convergence remain required.

### Stage 9 — Certified nonconvex cells

- [ ] Admit star-shaped cells through certified kernel points and canonical
      constrained face triangulations; then add boundary-conforming constrained
      tetrahedral/convex decomposition for non-star-shaped cells.
- [ ] Preserve the original shell, shared boundary subdivisions, disconnected
      clipped components, and multiple phase patches. Do not infer a kernel
      from a centroid or replace a nonconvex parent with its convex hull.
- [ ] Certify shell/union/first-moment recovery and disjoint positive pieces,
      reject self-intersections, and repeat geometry, VoF, and physical-coupling
      gates for every admitted class before extending support claims.

Every stage requires focused affected checks; serial/MPI correctness evidence
does not establish scaling or physical application accuracy. Assert actual
ownership in empty-rank fixtures; choose rank counts from global cell counts or
deliberate sparse partitions, not root-block counts. Keep dated commands/results
in local validation artifacts and maintain support status and remaining gates in
the design and modeling documents. Level subcycling, simultaneous
ALE/regridding, curved-cell refinement, and rotated periodic transformations
remain outside this design.

## Global acceptance criteria

**Overall status: not met.** Historical configure/build/test completion does
not close the open model, verification and documentation gates above or establish
a current-tree full-suite/CI result.

- [x] Recorded local GCC Debug configure/build/serial/MPI checks; CI requires
      representative GCC Debug plus GCC/LLVM Release coverage.
- [x] Disabled-physics regressions, bounded fields, source units, VTU/configuration
      support and documented model limitations within the implemented scope.

Continue to report launcher restrictions separately from code failures and
identify actual ranks, skips and run provenance.

- [ ] Complete the Phase 18 key/default/unit reference.
- [ ] Satisfy every acceptance criterion in the non-deferred roadmap phases.
- [ ] Close the remaining quantitative RANS validation criteria.
