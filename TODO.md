# TODO: SimpleFluid Multiphysics Roadmap

**Current work (`feature/vof`):** qualify experimental source-free VoF on the
generic reacting-flow foundation. `SweptPartition` and Ordinary arithmetic
remain defaults; algebraic face transport is opt-in. Generic reacting-VoF
integration and physical application qualification remain open.

`SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF=ON` enables source-free solver development
(default `OFF`). The unmigrated `ReactingVoFCoupler` and dependent fixtures remain
excluded; shared geometry/FVM utilities are available independently.

Completed entries summarize scope. Detailed contracts live in `docs/modeling/`
and `docs/architecture/`, numerical gates in `docs/verification/`, and local
qualification records in build artifacts or `docs/qualification/`. Checkmarks
and historical results do not establish current-tree runtime qualification.

## Roadmap dashboard

Numbered phases retain dependency history. Their unchecked tasks and linked
acceptance contracts remain open regardless of the current work order.

### Current priorities

Work through the [Stage 7 checklist](#stage-7--general-vof-and-flow-qualification)
in this dependency order:

1. Preserve the implemented Ordinary compatible projection and its independent
   pressure-work/energy checks while completing coupled-flow qualification.
2. Use the faithful native replays to distinguish optimizer defects from
   remaining model rejection; retain strict target-volume and work gates.
3. Resolve Q2 storage/accounting blockers and complete the required refinement
   sequence, separating capacity results from numerical convergence.
4. Complete required GeometricFace Q3 coupled motion; recheck DonorCell where
   shared operators change without erasing its displacement failure.
5. Complete the existing GeometricFace capillary horizon after shared-flow
   prerequisites pass; defer DonorCell curvature-support expansion.

Use one decisive reproducer, affected regressions and the relevant qualification
case per repair: short serial development prefixes, then the full serial horizon,
then required MPI2/MPI4. SweptPartition stays default; its default-policy tasks
remain separate, not prerequisites for this algebraic repair sequence.

Other foundation, reacting-flow, RANS, demo, documentation and planar-ALE work
remains in Phases -1–20.1. Reassess whether Phase 14's optional scalar advection/
slip path is needed before implementing its open transport tasks. Euler–Euler
remains deferred in Phase 21.

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
- [ ] Revisit `PlanarALEMeshMotion::compute_trial_state` execution optimization
      only with stronger application evidence: at least 1% lower median wall time
      and nonoverlapping ranges. Preserve arithmetic, traversal, collective
      failures and physical gates; the reverted candidate passed parity but
      inclusive phase timing did not establish the required application benefit.
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

**Status:** uniform serial VoF and distributed Cartesian AMR have focused numerical
coverage; adaptive levels share one timestep. Single-gas reacting/representation
components remain legacy work; the reacting coupler and dependent fixtures are
excluded. Species migration, closure calibration and centimetric-bubble
application qualification remain open.

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

- [x] Serial Cartesian PLIC/transport, variable-density flow, capillarity and
      atomic timestep/failure handling; see [the solver guide](docs/modeling/vof.md)
      and [numerical contracts](docs/verification/vof_solver_contracts.md).

Two-rank coverage verifies rejection, not MPI flow. Long-time parasitic currents,
viscous rising bubbles and reacting solutions remain unqualified.

### Stage 2 — Conservative three-dimensional AMR

- [x] Distributed Cartesian 2:1 AMR, conservative PLIC/extensive remap,
      refinement policies, shared-timestep transport and atomic regrid/replay.

See [adaptive VoF](docs/modeling/adaptive_vof.md) and its
[numerical scope](docs/verification/vof_solver_contracts.md#numerical-assertions):
lower-contrast curved-seam passage and finite-duration static bubbles in a uniformly fine band.

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

The [retained coupling contract](docs/modeling/reacting_vof.md) and
[numerical assertions](docs/verification/vof_solver_contracts.md#numerical-assertions)
do not establish generic reacting integration or physical calibration.

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

**Status:** Stages 1–6 have focused coverage, Stage 7 is under qualification,
and Stages 8–9 remain proposed. The [general AMR design](docs/architecture/general_amr.md)
owns support, identity, admission and transaction contracts; geometry/remap
coverage does not qualify physical flow or production scaling.

### Stages 1–6 — Hierarchy, geometry and distributed migration

- [x] Immutable hierarchy/trials, certified geometry/remap, XY/Z and convex
      refinement, mixed-region closure and detached distributed migration/balancing.

See [scope and limits](docs/architecture/general_amr.md#qualification-status) and
[stage acceptance](docs/architecture/general_amr.md#implementation-stages-and-focused-acceptance).
Bounded bootstrap/catalog and seam costs remain; scalable planner storage/RSS,
sparse ALE and physical-flow qualification are separate gates.

### Stage 7 — General VoF and flow qualification

`SweptPartition` remains the default; immediate work targets shared numerical
blockers and opt-in GeometricFace qualification. The
[maintained regimes and gates](docs/verification/vof_solver_contracts.md#stage-7-scope-and-priorities)
define the existing mesh, material, Courant, horizon and resource limits. No
algorithm, test, numerical tolerance or supported backend changes follow from
this plan; generic reacting integration remains Stage 8.

Ordinary SweptPartition/GeometricFace default to cleanup at `1e-14`, with cumulative
caps `1e-13` of each original global phase inventory. Controlled algebraic fixtures
explicitly use zero unless their supported cleanup variant is selected. DonorCell
and non-Ordinary modes reject positive cutoffs. Record actual policy and retain
raw/cleaned accounting; raw passes do not qualify enabled cleanup or coupled remap.

#### Implemented foundation and recorded evidence

- [x] Source-pinned phase/momentum ledgers, collective rejection, compatible
      preparation, independent inventories, cache invalidation and atomic state;
      see [transport contracts](docs/architecture/general_vof_transport.md).
- [x] Ordinary homogeneous/interface-local transport, resolved aggregates,
      capillary surfaces and conditioned PLIC with strict offsets; see
      [capillarity](docs/modeling/general_vof_capillarity.md), [PLIC](docs/modeling/plic.md)
      and [provider boundaries](docs/architecture/general_vof_implementation.md).
      Unresolved aggregate approximations still reject.
- [x] Opt-in face candidates/shared limiting; separate fitting rings and consumer
      halos, optimizer evidence, wall traction, signed occupancy and canonical
      correction, interned hierarchy payloads and strict swept-face phase proofs.
      These repairs preserve the original gates and do not close long-flow accuracy.

The latest dated snapshot is `build/gcc/verification/algebraic_vof_repair_20261031/HANDOFF.md`,
with `stage-status.json` and the source applicability audit beside it. Q0/Q5
have scoped serial/MPI2/MPI4 evidence, including all Q5 field comparisons;
uniform N32 initialization now passes its unchanged cap. Q2/Q3/Q4 overall remain
incomplete. The separate `.125 s` motion baseline and older `.02 s` cylinder
result do not replace those campaigns. Preserve failed attempts and source
differences in these and the earlier [recorded results](docs/verification/vof_solver_contracts.md#recorded-stage-7-evidence).

#### Immediate numerical blockers — priority and dependency order

- [ ] **1. Shared pressure–velocity qualification.** The Ordinary compatible
      projection repair and independent energy/mass/occupancy regressions are
      implemented. Retain their original gates and complete item 4's full
      coupled-flow acceptance; face-continuity residuals alone are insufficient.
      Bounded and Exact projection remain separate supported scopes.
- [ ] **2. Faithful native reconstruction qualification.** Native-cell replays
      and constrained-merit/curvature/restart repairs are implemented; an affine
      surrogate alone remains insufficient. The later Q3 cell-36 candidate
      differs from its passing earlier fixture. Capture each remaining failure
      faithfully and distinguish optimizer failure from proven model
      incompatibility. Keep the original strict-volume/model/work gates and
      originating full-flow case.
- [ ] **3. Q2 refinement capacity and convergence.** On the smallest required
      failing setup, distinguish unique retained/live storage from duplication,
      cumulative charges and incorrect accounting before selecting a remedy.
      Uniform N32 GeneralVoF initialization now passes the original 64 MiB
      cap in serial/MPI2/MPI4 after measured allocation removal. Its original
      finite-rectangle motion still rejects a single-plane model; the independent
      all-plane lower bound is recorded in the Q2 repair evidence. N64
      authority/closure completes, but compact native-draft admission still
      rejects the expanded serialization proxy. Unblock the required N16/N32/N64
      sequence and separate temporal study under existing caps, or report the
      remaining capacity failure explicitly. Setup success is not convergence.
      Retain the original N16 model rejection, independent motion oracle, full
      endpoints and accuracy/order gates; omitting a failed level is not acceptance.
- [ ] **4. Q3 GeometricFace coupled motion — after the applicable blockers above.**
      Complete the required eight-second horizon and original displacement,
      redistribution, limiting and physical gates at the declared timestep
      fractions; report actual Courant separately. Recheck DonorCell where the
      shared projection/operators change. Its recorded eight-second completion
      still fails displacement and must not be relabelled a pass. Minimum evidence:
      affected coupled regressions, acceptable full serial cases, then their
      required MPI2/MPI4 counterparts with unchanged accuracy and resource criteria.
- [ ] **5. GeometricFace capillarity — after shared-flow prerequisites pass.**
      Complete the existing required Q4 physical horizon for the selected model,
      rechecking the recorded `.02 s` cylinder endpoint after relevant changes.
      Use the existing curvature/pressure/spurious-current and continuation
      contract; a geometry report or prior-source pass cannot qualify new flow
      operators. Minimum evidence: affected capillary regressions and the required
      full serial case, then MPI2/MPI4. Defer DonorCell curvature-support expansion
      and additional curvature models rather than broadening this repair.

#### Minimum validation for each repair

- One decisive faithful reproducer per defect, directly affected regressions and
  the relevant qualification case. Use short serial prefixes while developing;
  only the unchanged full serial endpoint establishes horizon acceptance. Run
  required MPI2/MPI4 after that serial result is acceptable.
- Repeat passed Q0/Q5, reconstruction, ABI and architecture checks only where
  changed code can affect their contracts. Select affected shared-path coverage,
  not blanket suites or unrelated AMR/repartition campaigns.
- Retain applicable conservation, boundedness, physical residuals, accuracy,
  cleanup accounting and work/storage limits. Report numerical, capacity and
  launcher outcomes separately. Preserve failures and existing safe rejection;
  completion, small continuity residuals or changed source provenance do not
  replace accuracy. Do not relax thresholds or add exhaustive permutations.

#### Separate SweptPartition default-policy qualification

These tasks remain open for the default backend and do not gate algebraic repair.
Shared changes still require directly affected default-path regression checks.

- [ ] Qualify automatic-cleanup full50: 50-cell skew/coarse-fine extrusion,
      density ratio 10, ten `.002` steps to `.02`, with original physical and
      cleanup-accounting gates; full serial first, then required MPI2/MPI4.
- [ ] Establish independent default-policy motion accuracy and mesh/time convergence,
      and applicable hydrostatic, capillary and viscous horizons. Keep extended
      swept diagnostics and the explicit-zero eight-step probe separate; their
      incomplete results neither qualify nor block the algebraic family.
- [ ] Quantify [cleanup sensitivity](docs/modeling/phase_cleanup.md#qualification-and-sensitivity)
      with matched cutoff/cap and mesh/time studies under the original inventory,
      accuracy and resource gates. Small cleanup totals alone do not bound dynamics.
- [ ] Apply the relevant [FP acceptance invariants](docs/verification/vof_solver_contracts.md#floating-point-acceptance-gates)
      to affected default paths, preserving independent oracles, MPI consistency,
      bounded resources and atomic rejection without an exhaustive mode matrix.

#### Explicitly deferred scope

- Explicit-zero microscopic full50 completion and wider plane representations.
  The recorded MPI2/MPI4 step-4 failure after `.006 s` remains failed; preserve its
  [capture/local replay](docs/verification/vof_solver_contracts.md#strict-mode-microscopic-replay)
  and safe rejection/state preservation without a new robustness campaign.
- Exhaustive arithmetic-mode, mesh, density, Courant, boundary and rejection/
  rollback permutations; unrelated AMR/repartition, balancing and ABI campaigns.
  The recorded `DistributedAdaptiveBalance.MeasuredOperationsDrivePinnedHierarchyMigration`
  failure remains open for that separately claimed capability.
- Cleanup-enabled coupled AMR/remap, automatic refinement/reinitialization,
  wider/high-contrast curved seams and nonzero-time regrid momentum. Keep current
  coupled-channel rejection; phase-only remap/ownership migration and raw Q5
  passes do not establish cleanup-enabled coupled support.
- DonorCell curvature-support expansion, additional curvature models and new
  long-time/static/dynamic studies beyond the existing selected Q4 requirement.
- Q6 performance/scaling and Q7 open-boundary/multistream extensions; periodic
  transport, within-step ALE, subcycling and reacting/representation integration.
- New coverage certificates, lazy moments, approximate aggregate admission and
  projection/curvature optimizations unrelated to the diagnosed defects. If later
  adopted, retain their proof/debit requirements and original numerical gates.

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

Every stage needs focused checks with actual empty-owner assertions; rank counts
must follow global cells or explicit sparse ownership, not root blocks. Keep run
evidence in local artifacts. Level subcycling, simultaneous ALE/regridding,
curved-cell refinement and rotated periodic transformations remain excluded;
correctness coverage alone does not qualify scaling or physical accuracy.

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
