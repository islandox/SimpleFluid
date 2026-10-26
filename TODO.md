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

**Documentation review date:** October 24, 2026

This refresh reconciles the current source, public APIs, and checked-in test
coverage. It does not represent a new runtime qualification or full-suite run.

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

The following infrastructure is implemented and should not be re-planned as
active roadmap phases unless regressions are found. The open numerical and
verification items below mean the foundation as a whole is not yet complete.

- [x] Mesh and geometry infrastructure
  - [x] CRTP mesh hierarchy
  - [x] Cartesian 3D mesh
  - [x] Orthogonal cylindrical 3D mesh
  - [x] Semi-structured XY×Z prism mesh
  - [x] STK/Exodus adapter for `HEX_8` and `WEDGE_6` meshes
  - [x] Runtime type-erased `MeshHandle`
  - [x] Owned/ghost cell decomposition
  - [x] Zoltan2/ParMETIS partitioning
- [x] Field infrastructure
  - [x] Cell-centered scalar, vector, and tensor fields
  - [x] Face-centered scalar and vector fields
  - [x] Boundary-face fields
  - [x] Tpetra-backed distributed storage
  - [x] Ghost-value synchronization
  - [x] Mesh-aware `FieldStored` abstraction
- [x] Finite-volume operators
  - [x] Cached least-squares scalar/vector gradient reconstruction
  - [x] Gauss-linear scalar/vector gradient reconstruction
  - [x] Orthogonal scalar/vector diffusion
  - [x] Explicit non-orthogonal correction
  - [x] Fully implicit non-orthogonal diffusion
  - [x] Hybrid implicit/explicit non-orthogonal treatment
  - [x] MPI partition-face implicit gradients with synchronized remote halves
  - [x] Scalar/vector transport systems
  - [x] Boundary-condition-aware diffusion in transport systems
  - [x] First-order upwind convection
  - [x] Backward Euler time integration
  - [x] Opt-in constant-step BDF2 for legacy/native mapped weighted-scalar and
        native mapped physical-temperature transport, with a required older field
  - [x] Opt-in bounded linear-upwind scalar convection through a conservative
        deferred correction
- [x] Momentum and pressure-velocity coupling
  - [x] Momentum equation assembly
  - [x] SIMPLE
  - [x] PISO
  - [x] PIMPLE
  - [x] Fully coupled block-Krylov pressure-velocity solver
  - [x] Rhie–Chow collocated-grid stabilization
  - [x] Slip-wall momentum transport without spurious boundary diffusion
  - [x] Residual reporting for momentum, pressure, and continuity
- [x] Physical solver baseline
  - [x] Transient incompressible Navier–Stokes
  - [x] Boussinesq buoyancy
  - [x] Temperature transport
  - [x] Natural-convection driver
  - [x] Compact solid-cell subdomains with synthetic interface boundaries
  - [x] Selected-first mesh reordering with range-backed solid membership
  - [x] Transient conservative solid heat conduction
- [x] Native runtime-mesh solver path
  - [x] Run `FluidSolver` and `BoussinesqSolver` directly on supported
        `MeshHandle` backends without reconstructing a legacy mesh
  - [x] Select `FieldStored` scalar, vector, and tensor cell storage plus
        scalar/vector face storage through `MeshFieldTraits`
  - [x] Support mapped diffusion, convection, pressure projection,
        Rhie–Chow fluxes, and coupled pressure–velocity assembly
  - [x] Carry turbulence, wall distance/treatment, prescribed heating, the
        typed reacting source, boiling, void, material feedback, linear
        source/decay scalars, and steady-state search
        through the native field path
  - [x] Preserve exact legacy `Mesh` identity, public solver hooks, and
        established behavior through a distinct compatibility path
  - [x] Cover Cartesian and cylindrical serial/distributed execution with
        focused regressions
  - [x] Add count-balanced, width-aware Cartesian/cylindrical tensor partitioning
        and contiguous owned cell/face algebraic maps
  - [x] Replace orthogonal local source-ID tables with compact halo-aware topology
        and arithmetic lookups
  - [x] Cover native unstructured execution directly in serial and through
        `MeshPartitioner`/`PartitionedMesh` in MPI
  - [x] Support native `SemiStructuredXY_Z` serial/MPI ownership with Zoltan2
        XY graph partitions crossed with Z slabs, `MeshReorderingFactory` XY
        ordering, compact decomposed XY/Z topology, contiguous algebraic cell/face
        maps, adjacency ghosts and axial ALE, including empty owners
        (see [scope and checks](docs/semi_structured_parallel.md))
- [x] I/O and utilities
  - [x] VTU output for ParaView
  - [x] STK Exodus II mesh input for `HEX_8` and `WEDGE_6` volume elements
  - [x] Typed key-value `Database`
  - [x] `RandomAccessView`
  - [x] `vec3`
  - [x] Concept-based Tpetra type configuration
- [x] Verification suite
  - [x] Lid-driven cavity transient smoke cases, Re = 100 and 1000
  - [x] Poiseuille flow
  - [x] Manufactured incompressible Navier–Stokes solutions
  - [x] Skewed-mesh diffusion
  - [x] Natural-convection square cavity
  - [x] External OpenFOAM comparison workflow and case-definition check
  - [x] Gradient/divergence reconstruction tests
  - [x] Pressure projection tests
  - [x] Rhie–Chow checkerboard-suppression tests
  - [x] Slip-boundary momentum-diffusion regression tests

#### Known foundational gaps

- [x] Align `verification/environments.sh` with the current preset output
      directories (`build/gcc` and `build/llvm`).
- [x] Add and verify opt-in constant-step BDF2 for legacy/native mapped
      weighted-scalar and native mapped physical-temperature transport while
      preserving Backward Euler defaults.
- [x] Add and verify opt-in bounded linear-upwind deferred correction for
      scalar transport while preserving first-order-upwind defaults.
- [ ] Add quantitative lid-driven-cavity profile checks against Ghia et al.
- [ ] Bundle validated OpenFOAM centerline profiles and enforce tolerances in
      an automated test.

---

### Compact region composition

The current annular factory and swept-provider APIs are documented in
[annular meshes](docs/annular_meshes.md). Their checked-in regressions cover
geometry, transfer, and motion contracts; they do not establish physical
pool-flow validation.

- [x] Separate topology and geometry contracts with shared Cartesian topology
      and independent coordinate providers.
- [x] Add static conforming serial composition through `MeshHandle`, canonical
      seam faces, explicit mixed HEX_8/prism-side interfaces, and namespaced BCs.
- [x] Replace normal implicit cell-face materialization with independent ranges;
      retain explicit compatibility CSR and preserve native ownership/halo queries.
- [x] Add assembled FVM/flow regressions, diffusion example, VTU region metadata,
      and an opt-in capacity/timing benchmark. Support and numerical gates are
      maintained in [region mesh contracts](docs/region_meshes.md); physical
      validation remains separate.
- [x] Verify the merged coupled block-composite backend on Cartesian and mixed
      HEX/prism regions, including all operator/workspace choices, both pressure
      gradients, backend switching and zero mesh-connectivity materialization.
- [x] Compact implicit-region MPI: multiple regions per rank, split regions,
      unique canonical face ownership, halos, owned-cell output and conservative
      two/four-rank tests. Explicit global constituents remain rejected.
- [x] Nested planar coarse/fine interfaces with coverage/overlap validation,
      canonical subfaces and signed conservative flux transfer.
- [x] Composite common axial affine ALE using existing GCL, epoch, lease,
      cache refresh and rollback machinery, including solver-integrated tests.
- [x] Cylindrical composition, physical angular closure and explicit translated
      periodic patches with image-correct geometry and fluxes.
- [x] Independent extruded geometry providers sharing immutable base topology,
      with analytic geometry, native parity and serial/MPI solver tests.
- [x] Validated execution views, geometry read leases, per-region interface
      directories and Cartesian/extruded descriptor validation without volume sweeps.
- [x] Nonlinear/coarse-fine, periodic-through-seam and decomposition-invariance
      qualification; diagnostic N/R/P baselines with peak RSS. See
      `docs/region_execution.md` and `docs/region_scaling.md` for measured scope.
- [x] Converge lagged MPI gradients in the actual implicit steady diffusion
      helper; require a fresh full-equation residual before success, retaining
      native/mapped paths and the original native overload symbols.
- [x] Resolved-face and transformed region-native scalar diffusion assembly,
      with a generic reference and motion/periodic/coarse-fine parity checks.
- [x] Remove per-cell exact-reserve allocation from explicit compatibility CSR.
- [x] Build graded polygonal annuli with independently sized radial wall rings,
      fixed-boundary Delaunay bulk triangles, hex wall layers, and conforming
      bottom/top prism stacks.
- [x] Coarsen eligible bottom wall intersections with compact swept R–Z
      mitered regions, preserving native volumes, canonical interfaces,
      distributed ownership, and common affine Z motion.
- [x] Distributed explicit region packets and region-aware load balancing via
      `CompositeMeshSource`/`MeshPartitioner`; native topology and canonical
      interfaces retained. Source union and graph construction remain centralized
      on the source rank; see `docs/composite_partitioning.md`.
- [ ] General mortar/AMI intersections, curved/nonconvex face subdivision,
      wider polygonal volume/output types and rotated periodic field transforms.
- [ ] Nonaffine composite motion and moving periodic lengths: require consistent
      shared interface geometry, exact GCL and full transactional solver tests.

### Two-equation RANS turbulence

This program was added after the numbered multiphysics roadmap. It is tracked
separately so its implemented scope and remaining validation work are not
mistaken for foundation or Phase 9 completion.

#### Implemented scope

- [x] Add runtime selection for six transported two-equation closures:
  - [x] standard k-epsilon
  - [x] RNG k-epsilon
  - [x] realizable k-epsilon
  - [x] standard k-omega
  - [x] Menter BSL k-omega
  - [x] Menter SST k-omega
- [x] Preserve a laminar selection that allocates no turbulence state.
- [x] Couple eddy viscosity to momentum and gradient-diffusion turbulent
      transport to temperature.
- [x] Add resolved low-Re SST, resolved standard/realizable k-epsilon, and
      standard high-Re k-epsilon wall treatments.
- [x] Add smooth/sand-grain high-Re momentum wall laws, constant-Prandtl and
      Jayatilleke thermal wall laws, and resolved-wall compatibility checks.
- [x] Add distributed Poisson wall distance with configurable non-orthogonal
      and linear-solver controls; BSL/SST automatically include every no-slip
      wall.
- [x] Add globally reduced wall-y+ diagnostics and a rebuild-only
      boundary-layer adaptation driver with mesh-quality rejection.
- [x] Couple signed OpenFOAM-style Boussinesq production/destruction into both
      transported turbulence equations.
- [x] Add formula, option-validation, transport, wall-treatment, solver, and
      MPI consistency tests.
- [x] Add the transient `pitz_daily` standard-k-epsilon example and the
      OpenFOAM profile-comparison workflow.
- [x] Add a transient isothermal incompressible solver that advances RANS
      without allocating or solving a temperature equation, and use it for
      `pitz_daily`.
- [x] Add the matched 1000 W Gaussian heated-solution-tank SST comparison,
      including 2 mm R-Z spacing, wall-layer refinement, and R-Z error figures.
- [x] Document that the current RANS variables use single-continuum, full-cell
      transport rather than phase-volume-fraction-weighted equations.

#### SST-1994 plus SAS source extension

- [x] Implement opt-in `SSTKOmegaSAS` without changing ordinary SST-1994.
- [x] Add bounded local algebra, physical-time cap, explicit omega source,
      unit-coefficient vector Laplacian, lazy work fields and accepted diagnostics.
- [x] Add pointwise reference, derivative, source-weighting, disabled-equivalence,
      wall/configuration, native/legacy and serial/MPI transaction checks.
- [x] Add a small isothermal transient activation example without IF97.
- [x] Qualify native cylindrical SAS with Cartesian vector components and
      integrated radial-face normals, including serial/MPI transient swirl.
- [x] Qualify serial SemiStructuredXY_Z SAS on skewed triangular extrusions.
- [x] Qualify mixed slip SAS boundaries with normal-foot reconstruction and
      serial/MPI affine, curved-source and transient tests.
- [x] Qualify native Cartesian periodic SAS and legacy translational pairs,
      including wrapped derivatives, flux geometry and output images.
- [x] Qualify Gauss-linear SAS with coordinate-moment consistency correction
      on native/legacy, curved, skewed, slip and periodic paths.
- [x] Add rollback-safe SAS material feedback.
- [x] Add rollback-safe SAS scalar void diffusion/collapse and mixture feedback.
- [x] Add rollback-safe SAS linear source/decay scalar inventories and balance diagnostics.
- [x] Add rollback-safe SAS legacy ideal/two-population gas models and single-gas ledgers.
- [x] Add rollback-safe SAS bulk/wall boiling and phase-change state.
- [x] Add rollback-safe fixed-grid SAS free-surface/liquid-mass budgets,
      including lazy initialization and accepted history.
- [x] Support NOX velocity-pressure coupling for native isothermal and physical
      Boussinesq SAS, with frozen coefficients, one turbulence advance per
      physical step, accepted-report restoration and retry after rejection.
- [ ] Establish spatial/temporal resolution and sustained three-dimensional
      scale-resolution evidence beyond source activation.
- [ ] Perform qualified whole-flow external comparisons with explicit parent-SST
      and discretization differences, then quantitative physical validation.

The [SAS model document](docs/modeling/sst_sas.md) distinguishes
implemented, component-tested, solver-regression-tested, externally compared
and physically validated status. The parent closure and pending pitzDaily gates
are unchanged.

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

---

## Core infrastructure and source phases — Phases 9 to 12

### Phase 9 — Performance and solver-regression benchmarks

Keep this phase independent of the new physics so solver performance changes can be diagnosed cleanly.

- [x] Add a repeatable benchmark harness.
- [x] Record nonlinear iteration counts.
- [x] Record Krylov iteration counts.
- [x] Record wall time.
- [x] Record memory usage.
- [x] Record continuity residuals.
- [x] Measure iteration count versus mesh non-orthogonality.
- [x] Compare diffusion/momentum treatments:
  - [x] explicit non-orthogonal correction + AMG
  - [x] implicit non-orthogonal matrix + AMG-preconditioned GMRES
  - [x] hybrid non-orthogonal treatment + AMG-preconditioned GMRES
  - [x] coupled Krylov + Schur preconditioner
- [x] Add MPI scaling benchmarks.
- [x] Ensure benchmark output is machine-readable, for example CSV or JSON.
- [x] Store versioned benchmark baselines keyed by case, preset, mesh, MPI
      size, non-orthogonality, treatment, coupling mode, and preconditioner.
- [x] Check deterministic iteration ceilings and a deliberately generous
      wall-time ceiling in CTest after warmup and repeated measured runs.
- [x] Reuse compatible coupled-system graphs, static pressure-gradient
      geometry, Schur-product storage, Ifpack/MueLu numeric state, Belos state,
      and preconditioner scratch vectors under an explicit rebuild policy.
- [x] Move coupled preconditioner packing, gradient updates, and unpacking to
      Kokkos device-local kernels.
- [x] Resolve ghost-cell ownership through a distributed Tpetra directory
      rather than replicating every owned mesh GID on every rank.
- [x] Cache mesh-only least-squares transport geometry and boundary-face
      locations while continuing to materialize dynamic boundary data.
- [x] Batch dynamic-source collective validation and skip disabled/static
      source and material imports.
- [x] Cache VTU topology and support appended binary rank pieces with a PVTU
      index.
- [ ] Migrate the remaining host-side finite-volume row assembly loops to
      backend-portable Kokkos kernels after defining a device assembly API.
- [ ] Revisit scoped geometry execution in
      `PlanarALEMeshMotion::compute_trial_state` only with stronger
      application-level performance evidence. The October 11, 2026 review of
      external host candidate C2 reverted the trial-read lease and its two added
      tests because the measured benefit did not clear the predeclared
      performance gate, not because of a correctness failure. On the
      267,652-cell, six-rank, 0–0.1 s startup fixture, warmed ABBAAB runs passed
      exact parity for all ten selected output files and matched native phase
      call counts; focused serial/MPI and rejection/replay checks also passed.
      Rank-zero inclusive ALE-geometry median time fell from 33.47 to 28.80 s
      (about 14%), while whole-run median time fell from 292.05 to 284.55 s
      (2.57%). Observed ranges overlapped: control 269.73–292.99 s, candidate
      277.49–287.13 s. The acceptance rule required at least 1% lower median
      wall time **and nonoverlapping observed ranges**; the application benefit
      therefore remained uncertain. Warmups and an earlier block contaminated
      by a competing build were excluded. Preserve arithmetic, traversal order,
      collective failure handling, and physical gates in any requalification;
      do not treat the inclusive phase reduction as an established wall-time
      speedup. The patch and evidence remain in the local, untracked external host
      directory `build/next-optimization-20261010/`: `C2.patch`,
      `C2-comparison.json`, `C2-decision.json`, and `report.json`.

The timing ceiling is a smoke-regression guard, not a claim of comparable
performance across different machines. Release scaling and profiling remain
manual measurement workflows.

- [x] Add opt-in coupled block application without monolithic CRS allocation,
      with retained-generation ownership and serial/two-/four-rank algebra tests.
- [x] Add independent streamed distributed-product assembly retaining final
      stabilization/Schur matrices; record small separate-process memory results.
- [x] Qualify coupled backend choices across current solver classes, native/legacy
      mesh paths and supported ALE motion; align interpolation and pressure-gradient
      reconstruction, and refresh MueLu numeric state during reuse.
- [ ] Add native scalar face-based orthogonal diffusion and frozen upwind/BE
      transport, with independent operator/diagonal/preconditioner inputs.
- [ ] Qualify coupled memory/performance at larger converged sizes and on
      accelerator builds; separate preconditioner/Krylov allocation accounting.

#### Phase 9 acceptance criteria

- [x] Benchmarks can be run from CTest or a documented command.
- [x] At least one small benchmark is safe for local Debug builds.
- [x] At least one larger benchmark is suitable for Release/RelWithDebInfo profiling.
- [x] Results include enough metadata to compare solver configuration, mesh size, MPI size, and compiler mode.
- [x] Debug CTest fails when a matching stored baseline exceeds its configured
      iteration or wall-time ceiling.

**Status:** measurement harness, a deterministic Debug smoke-regression gate,
and the highest-frequency setup/allocation/metadata bottlenecks are addressed.
Cross-machine qualification and general device-side matrix assembly remain
open.

---

### Phase 10 — Source-term and material-property infrastructure

Before adding reacting-gas sources or boiling, make the solver accept explicit source fields and updateable material properties cleanly.

- [x] Add a generic volumetric scalar source interface for transport equations.
- [x] Add source-term support to the temperature/energy equation.
- [x] Add optional explicit source fields to VTU output.
- [x] Add a reusable material-property interface for:
  - [x] density-like feedback field
  - [x] viscosity-like feedback field
  - [x] thermal diffusivity or conductivity if already supported by the energy equation
- [x] Add `Database` parsing helpers with defaults, validation, and unit comments.
- [x] Add robust field initialization from programmatic configuration.
- [x] Preserve existing Boussinesq behavior when new models are disabled.

#### Phase 10 acceptance criteria

- [x] Existing natural-convection examples reproduce their old behavior when all new model switches are off.
- [x] A constant heat source produces the expected one-cell analytic temperature change when diffusion/advection are disabled.
- [x] Invalid source/property parameters fail with clear error messages.
- [x] Source fields can be written to VTU.

---

### Phase 11 — Prescribed and reaction heat-source coupling

Provide thermal sources independently of the reacting-gas production law.
Prescribed heating can drive convection without producing gas; an accepted
reaction extent can release or absorb heat without requiring a power-yield law.

**Status:** prescribed non-negative profiles, imported cell fields, normalized
total power, time multipliers, energy-equation coupling, and VTU output exist.
The neutral `PrescribedHeatSource` publishes `qdot_external` under
`prescribed_heat_*` controls. Retired source names and keys have no aliases
and explicit old keys reject. General reaction/source kernels and their field
and native-flow adapters retain separate chemical/external thermal ledgers.
Broader physical material and multiphysics qualification remain open.

- [x] Support constant and Gaussian prescribed heating and caller-supplied
      non-negative cell fields.
- [x] Normalize spatial profiles to configured total power on Cartesian and
      cylindrical meshes and couple the result to the temperature equation.
- [x] Cover zero heating and analytic one-cell temperature rise in focused tests.
- [x] Introduce neutral source names and controls as a deliberate API break,
      with no aliases and explicit rejection of retired keys.
- [x] Evaluate reaction heat from the same accepted extents used for species
      production and donor consumption, matched to the stored thermal potential
      and its excluded chemical/formation-energy terms.
- [x] Distinguish prescribed external heat, reaction heat, phase-change latent
      heat, and transferred species enthalpy so each is counted once.
- [x] Verify exothermic/endothermic reactions, zero reaction with independent
      heating, donor-limited heat release, and rejected-step energy rollback.

---

### Phase 12 — General reacting gas source in solution

**Status:** immutable species/reaction schemas, supplied rates or integrated
extents, finite donor accounting and independent thermal inputs are implemented.
The [source design](docs/modeling/reacting_gas_sources.md) and
[species contract](docs/architecture/reacting_species_contract.md) define units,
volume bases and supported adapters. Numerical qualification uses manufactured
laws; no universal chemistry or physical material calibration is implied.

- [x] Supply stable species IDs, molar masses, elemental composition and phase
      membership without reserved H2/donor/solvent slots.
- [x] Accept alternative rate and integrated-extent inputs; convert actual
      liquid/bulk volume and timestep exactly once and retain source location.
- [x] Support production, reverse reactions, explicit external reservoirs and
      separate heat under a declared formation/sensible-energy convention.
- [x] Reserve competing raw donor events against their actual predecessor stock;
      reject or recompute failed steps without clipping inventories.
- [x] Use accepted events for species, element, mass, thermal and history ledgers,
      retaining tiny representable material and count endpoints through replay.
- [x] Integrate the weak field adapter and conservative native low-Mach controller
      with collective configuration/failure handling and immutable accepted state.
- [x] Remove enabled null-mechanism/yield fallbacks, old source selectors and
      misleading fixed-species fields; explicit retired keys reject.
- [x] Resolve the reproduced native low-Mach exactly-empty-gas rejection under
      nonuniform heating at small timesteps through componentwise conservative
      transport checks and bounded refinement within the original scalar iteration
      budget. Focused serial and two-/four-rank mixture tests cover tiny physical
      arrivals, signed fixed/ALE energy, absent gas and budget rejection/rollback;
      [the transport contract](docs/architecture/low_mach_reacting_flow.md#componentwise-transport-consistency)
      retains the original true-residual, physical-work and empty-phase gates.
      This does not qualify untested source, mesh or thermodynamic combinations.
- [ ] Improve or explicitly bound large-volume/small-timestep EOS-minus-advected-volume
      cancellation while retaining independent physical continuity and occupancy gates.
- [ ] Add a separately qualified stiff-chemistry/Jacobian integration backend.
- [ ] Calibrate supplied reaction and property data and verify physical
      pressure/temperature/composition ranges with spatial/time convergence.
- [ ] Qualify additional mesh, inlet, source and multiphysics combinations before
      relaxing the documented support/rejection matrix.

---

## Active multiphysics phases — Phases 13 to 20

### Phase 13 — Bulk and wall boiling source model

Add a simple energy-consistent boiling model. This is a placeholder for a later RPI-like wall heat-flux partitioning model.

**Status:** partial. The explicit bulk/wall source, latent-energy coupling, and
focused conservation tests are implemented; the RPI heat-flux-partitioning
interface remains open.

#### Bulk superheat model

For cells with:

$$
T > T_{\mathrm{sat}} + \Delta T_{\mathrm{act}}
$$

compute a requested energy rate, capped so one explicit step cannot remove
more than the cell's sensible superheat:

$$
\dot e_{\mathrm{avail}} = \rho_l c_{p,l}
\frac{\max(T - T_{\mathrm{sat}},0)}{\max(\tau_b,\Delta t)}
$$

then:

$$
\dot m_b = \frac{\dot e_{\mathrm{avail}}}{h_{\mathrm{fg}}}
$$

$$
S_{\alpha,\mathrm{boil}} = \frac{\dot m_b}{\rho_g}
$$

and latent heat sink:

$$
Q_{\mathrm{latent}} = \dot m_b h_{\mathrm{fg}}
$$

Before the temperature update, limit aggregate bulk and wall boiling to the
void capacity remaining after the current single-gas source and optional
timestep-realizable collapse. Scale `S_alpha_boil` and `Q_latent` together so
latent energy is removed only for vapor admitted into the authoritative scalar
`alpha_g` field.

#### Wall boiling placeholder

For heated boundary faces:

$$
\dot m''_w = f_{\mathrm{evap}}\frac{q''_w}{h_{\mathrm{fg}}}
$$

Distribute to owner cells:

$$
\dot m_{w,\mathrm{vol}} = \dot m''_w \frac{A_f}{V_P}
$$

#### Tasks

- [x] Add `BoilingSourceModel` or equivalent.
- [x] Add configurable parameters:
  - [x] `enable_bulk_boiling`
  - [x] `enable_wall_boiling`
  - [x] `saturation_temperature`
  - [x] `boiling_activation_delta_t`
  - [x] `boiling_time_scale`
  - [x] `latent_heat`
  - [x] `gas_density`
  - [x] `wall_evaporation_fraction`
- [x] Compute `S_alpha_boil` as a cell field.
- [x] Compute `latentHeatSink` as a cell field in W/m³.
- [x] Subtract `latentHeatSink` from the temperature equation.
- [x] Write `S_alpha_boil` and `latentHeatSink` to VTU.
- [x] Keep the implementation explicit and bounded.
- [ ] Leave a clean interface for future wall heat-flux partitioning:
  - [ ] convective heat transfer
  - [ ] quenching heat transfer
  - [ ] evaporative heat transfer

#### Phase 13 acceptance criteria

- [x] Below threshold, boiling source is zero.
- [x] Above threshold, boiling source is positive.
- [x] `latentHeatSink == mDot * latentHeat` within tolerance.
- [x] Zero latent heat or invalid density fails with clear error.
- [x] Wall boiling distributes face source to owner cells conservatively.
- [x] Existing cases are unchanged when boiling is disabled.

---

### Phase 14 — Low-order void-fraction field update and optional transport

Introduce `alpha_g` as the first unresolved gas-state field for source-driven
gas generation and boiling workflows. This phase is the low-order scalar path:
The legacy Phase 12 source produces `S_alpha_reaction`, Phase 13 provides boiling sources, and
Phase 14.1 can replace the scalar update with a two-population bubble model
that reconstructs `alpha_g` from bubble inventories.

The legacy ideal-gas mode publishes `S_alpha_reaction` and keeps its model-local
`alpha_g` unchanged. The implemented Phase 14 path owns the aggregate
`S_alpha_total` bookkeeping, bounded low-order scalar update, collapse, and
optional diffusion.

**Status:** partial. Aggregate bookkeeping and the bounded source/collapse
update are implemented. Low-order advection and slip transport remain open;
the operational slip setting belongs only to Phase 14.1 today.

#### Standalone scalar update

$$
\alpha_g^{n+1} = \operatorname{clamp}\left(\alpha_g^n + \Delta t\,S_{\alpha,\mathrm{total}},\alpha_{\mathrm{min}},\alpha_{\mathrm{max}}\right)
$$

where:

$$
S_{\alpha,\mathrm{total}} = S_{\alpha,\mathrm{reaction}} + S_{\alpha,\mathrm{boil}} + S_{\alpha,\mathrm{wall}} - S_{\alpha,\mathrm{collapse}}
$$

Here `reaction` identifies the neutral gas-source output. The current
standalone scalar source is supplied independently of the species model.
The generic species path owns its finite material and accepted void publication;
bounding a separate scalar void alone cannot conserve species or energy.
Any gas removed by collapse needs an explicit dissolved, headspace or external
sink ledger. Implementing those contracts remains open.

The existing collapse term is bounded over the explicit step:

$$
S_{\alpha,\mathrm{collapse}} = \min\left(\frac{\alpha_g}{\tau_c},
\frac{\alpha_g-\alpha_{\mathrm{min}}}{\Delta t}\right).
$$

#### Optional transport upgrade

If the existing scalar transport API allows it cleanly, add:

$$
\frac{\partial \alpha_g}{\partial t}
+ \nabla\cdot(U_l \alpha_g)
+ \nabla\cdot(U_{\mathrm{slip}}\alpha_g)
=
\nabla\cdot(D_\alpha\nabla\alpha_g)
+ S_{\alpha,\mathrm{total}}
$$

#### Tasks

- [x] Allocate and initialize gas `alpha_g` and `alpha_l`.
- [x] Allocate and initialize `S_alpha_reaction`.
- [x] Allocate and initialize `S_alpha_total`.
- [x] Add bounded explicit update for the low-order scalar model.
- [x] Derive mirrored `S_alpha_total` from the published scalar state change
  instead of copying a provider's internal source history.
- [x] Add conservative bounded diffusion for the low-order scalar model.
- [ ] Connect accepted general reaction sources and conservative collapse/sink
      ledgers without using post-update clipping to satisfy inventory gates.
- [ ] Add low-order scalar advection with liquid face fluxes.
- [ ] Add configurable upward slip velocity aligned opposite gravity.
- [x] Add configurable `alpha_diffusivity`.
- [x] Add optional simple collapse/removal time scale for gas disengagement tests.
- [x] Ensure gas-model fields are ghost-synchronized after model advance.
- [x] Write legacy gas-model `alpha_g`, `alpha_l`, and `S_alpha_reaction` to VTU behind output switches.
- [x] Write `S_alpha_total` to VTU after aggregate source bookkeeping exists.

#### Phase 14 acceptance criteria

- [x] One-cell constant-source update matches the analytic result.
- [x] `alpha_g` never leaves `[alphaMin, alphaMax]`.
- [x] Legacy ideal-gas mode does not advance `alpha_g` before the low-order update path exists.
- [x] With all sources disabled, `alpha_g` remains unchanged.
- [x] Scalar diffusion conserves total gas inventory with zero-flux boundaries.
- [ ] Advective transport conserves total gas inventory up to boundary flux and sources.
- [x] VTU output can include legacy gas-model `alpha_g` and `S_alpha_reaction`.
- [x] VTU output includes `S_alpha_total`.

---

### Phase 14.1 — Dissolved-gas transport and bubble populations

**Status:** general species fields and named populations own finite material and
bubble counts. The field adapter retains its weak sensible-temperature model;
[the native low-Mach solver](docs/architecture/low_mach_reacting_flow.md) owns
conservative phase internal energy, EOS occupancy and physical pressure work.
`ReactingGasMode::Species` requires supplied chemistry/material setup.

- [x] Transport Liquid/Dissolved species and each named gas/count inventory once
      with its declared carrier/slip flux; retain accepted physical escape events.
- [x] Use supplied mixture EOS, species diffusivity/solubility, partial pressures
      and capillary pressure for bounded bubble-radius reconstruction.
- [x] Keep constant, reconstructed and prescribed-history absolute pressure
      separate from the incompressible pressure correction. Retired inertial
      pressure selection rejects in the general model.
- [x] Reserve finite material transfers jointly, preserve every species through
      population conversion and dissolution, and require paired count events.
- [x] Retain exact accepted raw event components before reporting reduction;
      gross withdrawals cannot borrow same-stage incoming material or counts.
- [x] Couple configured local interphase heat/material stages to native pressure
      recovery with partial molar enthalpies and explicit ordered donors.
- [x] Preserve typed fixed-grid surface/headspace and constrained planar-ALE
      inventory, geometry, history and rollback transactions.
- [x] Retain weak RANS/SAS-plus-species coupling and native/legacy mesh identity.
- [ ] Qualify material/rate, nucleation, film, slip and boiling closures against
      physical gas-release/bubble-column data and mesh/timestep convergence.
- [ ] Add a combined RANS/reacting-gas user example and phase-aware turbulence.
- [ ] Add implicit stiff kinetics and nonlocal pressure-dependent interphase
      transfer; qualify simultaneous implicit interface laws separately.
- [ ] Extend allocated local updates toward allocation-free device execution.
- [ ] Qualify live reacting repartition/remap and general interface representation
      changes. VoF and bounded-FP/exact-rational paths remain deferred.

Prescribed slip is not a gas momentum equation. Population size/number closure
and composition are independent inputs; a scalar radius or aggregate gas amount
cannot stand in for per-species inventories. Current numerical tests do not
establish a physically validated turbulent multiphase flow model.

---

### Phase 15 — Temperature- and void-dependent feedback properties

Add temperature- and void-dependent property feedback for reacting solutions.

**Status:** partial. Density feedback and constant-viscosity feedback with
safety floors are implemented. Temperature/void viscosity extensions and a
natural-convection stability acceptance case remain open.

#### Density feedback

Support a conservative default:

$$
\rho_{\mathrm{mix}}(T,\alpha_{\mathrm{g}})
= (1 - \alpha_{\mathrm{g}})\rho_{\mathrm{l}}(T) + \alpha_{\mathrm{g}}\rho_{\mathrm{g}}
$$

and a Boussinesq-compatible form:

$$
\rho_{\mathrm{fb}}(T,\alpha_{\mathrm{g}})
= \rho_{\mathrm{ref}}\left[1 - \beta(T - T_\mathrm{ref})\right](1-\alpha_{\mathrm{g}})
+ \rho_{\mathrm{g}}\alpha_{\mathrm{g}}
$$

#### Viscosity feedback

Start with:

$$
\mu_\mathrm{eff}(T,\alpha_g) = \mu_l(T) f_\mu(\alpha_g)
$$

where $f_\mu(\alpha_g) = 1$ by default until a validated correction is added.

#### Tasks

- [x] Add `MaterialFeedbackModel` or equivalent.
- [x] Add configurable density model:
  - [x] constant
  - [x] Boussinesq temperature-only
  - [x] Boussinesq + void fraction
  - [x] mixture density
- [x] Add configurable viscosity model:
  - [x] constant
  - [ ] temperature-dependent if existing support is available
  - [ ] optional void correction
- [x] Add safety floors:
  - [x] `min_density`
  - [x] `min_viscosity`
- [x] Couple density feedback into buoyancy/body-force calculation.
- [x] Couple viscosity feedback into momentum diffusion only if existing equation design allows it cleanly.
- [x] Write `rhoFeedback` and `muFeedback` to VTU.

#### Phase 15 acceptance criteria

- [x] Increasing `alpha_g` lowers density for liquid-like systems.
- [x] Increasing `T` lowers density under Boussinesq settings.
- [x] Density and viscosity never become negative.
- [x] Existing Boussinesq cases remain unchanged when feedback models are disabled.
- [ ] Natural-convection benchmark remains stable with temperature-dependent density enabled.

---

### Phase 16 — Reacting-solution tank demonstration

**Status:** `reacting_solution_tank_demo` and `constant_power_cylinder_vessel`
supply manufactured finite-reactant/two-gas mechanisms independently of heating.
The cylinder retains its bulk and wall vapor behavior through the finite
Boussinesq reference closure. H2/IF97 benchmark applications keep their explicit
chosen reference material and external-reservoir source laws.

- [x] Supply ordinary species, stoichiometry, finite stocks, population counts,
      material laws and independent heat through the generic API.
- [x] Retain cylindrical geometry, boundary heat, material feedback and accepted
      species/event output with explicit source and energy ownership.
- [x] Document manufactured versus matched-reference material/source provenance.
- [ ] Verify output opens in ParaView.
- [ ] Verify disabled reactions/boiling recover the independently heated
      natural-convection baseline over a complete physical run.
- [ ] Add matched serial/MPI and mesh/timestep convergence before presenting
      any example as physical validation.

---

### Phase 17 — Unit, regression, and conservation tests for reacting gas and boiling

**Status:** focused generic numerical regressions cover supplied chemistry,
phase/count transport, finite thermal admission, pressure/volume work and atomic
acceptance. Serial-first Debug runs and registered two/four-rank cases qualify
only their documented configurations; skipped or omitted tests are not passes.

- [x] Cover multiple species, forward/reverse rates, source-volume bases,
      independent heat and external reservoirs, zero source and complete donors.
- [x] Cover raw material/count cancellation, small occupied phases, gross donor
      competition and explicit full-disappearance count ownership.
- [x] Cover supplied EOS/film/property failures, actual empty owners, physical
      boundary escape, source location and partition-independent identities.
- [x] Retain standalone boiling/void and finite bulk/wall latent-energy checks.
- [x] Test fixed/ALE conservation, actual pressure work, complete rollback,
      stale retained candidates and callback-free commit/checkpoint replay.
- [x] Audit public exports and disabled-family boundaries with exact API anchors.
- [ ] Qualify broader material, toolchain, solver and mesh combinations and hosted
      coverage without treating local focused tests as a full-suite/CI result.
- [x] Qualify the 1008-cell scale-1 bottom-convection NOX startup through two
      serial physical steps with GMRES and BiCGStab corrections and pressure
      Schur AMG/SGS, retaining the prescribed sources, timestep, forcing controls,
      and every residual and physical gate. This bounded startup qualification
      does not establish developed-flow, longer-run, or MPI application behavior.
- [ ] Complete independent physical validation and convergence campaigns.

---

### Phase 18 — Documentation and configuration examples

Document the general reacting-gas source contract, low-order scalar path and
population path separately from a full Euler–Euler formulation.

**Status:** the general design is documented in
[reacting gas sources in solution](docs/modeling/reacting_gas_sources.md).
Generic API/configuration names, explicit species ownership and the migration
mapping are implemented without aliases. The support/rejection matrices separate
weak Boussinesq, native conservative low-Mach and deferred VoF paths. A complete
key/default/unit/validity reference remains open.

- [x] Document governing quantities, storage bases, conservation gates and
      limitations of the existing numerical foundation.
- [x] Separate weak RANS-plus-bubble coupling from phase-aware turbulent
      Euler–Euler flow and distinguish proposed design from current APIs.
- [ ] Complete the [external host companion migration](docs/architecture/external_host_reacting_flow.md),
      including finite species/isotope feedback and a supported bulk Winter-diffusion
      owner before claiming parity with the removed reacting APIs.
- [ ] Document every supported configuration key with default, SI unit,
      validity range and migration mapping; retired source keys must reject.
- [x] Document supplied rate/stoichiometry/property interfaces, finite donor
      limits, heat/phase-change ownership, restart state and rejection behavior.
- [ ] Provide runnable zero-reaction, closed-batch and convecting-solution
      examples with auditable source and property inputs.
- [ ] Ensure a developer can configure each supported model from the user
      reference without consulting implementation headers.

---

### Phase 19 — Solution-species transport and reaction inventories

**Status:** Liquid/Dissolved compartments share a physical solution phase while
retaining their individual species amounts. Native and weak field adapters use
declared transport carriers and accepted source events. The standalone
`ScalarSourceDecayTransport` utility retains generic analytic scalar source/decay
and transport; it supplies no chemistry mechanism or power-yield coupling.

- [x] Supply explicit species/composition/property hooks and shared accepted
      reaction/transfer events with Phases 12 and 14.1.
- [x] Transport finite solvent, solutes and gas-forming reactants/products within
      the supported closed/escape boundary matrix.
- [x] Preserve joint finite donors, raw signed event receipts, history/schema
      identity and atomic failure/replay across actual empty MPI owners.
- [x] Use conservative accepted-old/trial-new storage and relative flux for
      supported fixed-topology planar ALE; retain independent absolute continuity.
- [x] Retain separate generic scalar source/decay conservation coverage.
- [ ] Qualify prescribed inlet compositions, wider boundary exchange and
      transport/reaction convergence for physical material laws.
- [ ] Qualify generalized field-adapter bulk-mixture diffusion and broader
      nonlocal thermal transport. Pure/native face kernels do not establish
      support in every adapter.
- [ ] Add reacting conservative remap/repartition and broader ALE only with
      species, mass, energy, momentum and count closure.

---

### Phase 20 — External reaction/source coupling and feedback maps

Map solution temperature, absolute pressure, composition and phase inventories
to an external reaction/source provider and import rates/heat with explicit
units, spatial support and time intervals.

**Status:** an in-memory heat-exchange scaffold, feedback registry, mapped
snapshots and placeholder outer loop exist. Reusable cell transfer supports
nearest/inverse-distance interpolation and conservative supported overlaps,
including explicit partial coverage and intensive/extensive quantities; see
[mesh-to-mesh transfer](docs/mesh_to_mesh_transfer.md).
The scaffold's neutral external-heat field is `qdot_external`. Typed in-memory
reaction/source callbacks now bind units, schema, source location, interval and
accepted receipts. Spatially mapped external chemistry and a production transport
protocol remain separate work.

- [x] Retain volume-averaging utilities, deterministic feedback snapshots and
      constant-field/integral-preservation mapping tests.
- [x] Retain scalar/vector/tensor cell transfer, MPI donor exchange,
      coverage/inventory diagnostics, and geometry/map invalidation.
- [x] Retain conservative Cartesian/coaxial cylindrical overlaps and supported
      cylindrical-to-XY/Z or swept R–Z overlaps with native-volume accounting.
- [x] Retain imported heat fields and the existing callback/subcycle scaffold.
- [x] Define typed in-memory provider contracts for rate basis, species ordering,
      stoichiometry, sign, timestamps and accepted source intervals.
- [ ] Exchange pure-liquid composition/occupancy and absolute pressure without
      substituting void-reduced mixture density or gauge pressure.
- [ ] Conserve species and heat over spatial maps and subcycles, including
      partial coverage, provider rollback and exact-once accepted source use.
- [ ] Validate zero-source behavior, rate-basis conversions, stale maps,
      species-order changes and rejection/replay in serial/MPI.
- [ ] Select and validate a production transport/protocol only when a concrete
      external reaction/source provider is selected.

---

### Phase 20.1 — Planar free-surface volume budget and constrained ALE

Add liquid-level bookkeeping and a deliberately constrained planar moving-mesh
path without confusing either with a resolved interface method.

**Status:** the supported Milestone-A fixed-grid component path is implemented
and focused-tested, including accepted-step Boussinesq integration, automatic
accepted-state history with an explicit CSV writer, opt-in VTU fields, an explicitly
approximate cell-centre occupancy with volume error, and a checked-in analytic
verification driver. Species-bound liquid following and catalogued gas/headspace
events are implemented. Wider vapor transport and phase-aware mapping remain
open. Milestone B has a constrained solver-integrated planar-ALE core for native
mutable structured/extruded geometry, constant-area vessels, Backward Euler,
laminar temperature, species-bound liquid mass, vented headspace and supported
multispecies population transport/escape. Broader model combinations fail closed. This is
conservation-focused implementation coverage, not quantitative physical
validation. Milestone C does not yet provide conservative pool occupancy or
cross-mesh mapping.

#### Milestone A — implemented fixed-grid components

- [x] Add analytic constant-area and monotone tabulated vessel maps with
      guarded inversion, area queries, endpoint/range diagnostics, and explicit
      `error` or `clampAndReport` handling.
- [x] Parse and validate the flat `free_surface_*` model, vessel, fill,
      headspace, and nonlinear-coupling keys without changing disabled default
      behavior.
- [x] Add the explicitly limited `globalConstantMass` liquid inventory with
      MPI reductions, accepted evaporation/condensation, mass residuals, and
      bubble-free `rhoLiquid` evaluation.
- [x] Add opt-in `cellMassInventory` transport on the fixed reference mesh
      with the projected single-continuum face flux, transactional trial/commit,
      local boiling/condensation sources, pure-density volume evaluation,
      strict solver-independent conservation residuals, and serial/MPI
      partition tests. Require error depletion and zero physical-boundary flux
      until phase-acceptance and inlet-composition contracts exist.
- [x] Bind finite species/population inventories and supplied EOS/Laplace laws
      to non-mutating candidate-pressure volume evaluation.
- [x] Consume actual accepted species and count escape events; keep dissolved
      boundary discharge distinct and never infer escape from rounded cumulative
      totals or changes that can include internal phase transfer.
- [x] Give boiling one accepted phase-change path: cap-consistent evaporation
      and latent energy, rejected-vapor diagnostics, submerged steam ownership,
      and explicit condensate return after scalar-void collapse.
- [x] Reject removal or replacement of free-surface tracking while boiling
      still owns nonzero submerged steam rather than stranding that inventory.
- [x] Add vented constant-pressure and closed lumped ideal-gas headspaces with
      by-species transfer, positive absolute-pressure/volume guards, and a
      safeguarded nonlinear pressure/level closure.
- [x] Report old/current clear and pool levels, volume closure, overflow/dry-out,
      headspace state, nonlinear convergence, fixed-grid validity, and
      per-species gas closure through immutable diagnostic snapshots.
- [x] Add optional stable feedback registry names for `rhoLiquid`,
      `clearLevel`, `poolLevel`, `headspacePressure`, and `poolOccupancy` while
      preserving the disabled/legacy field contract.
- [x] Keep optional physics and acceptance in `MultiphysicsSolverCoupler`,
      with one species/thermal/transport owner, exact accepted escape, supplied
      headspace closure and accepted diagnostic publication.
- [x] Restore geometry, fields, model ledgers, time and histories atomically
      for the supported checkpoint/ALE transaction; publish accepted states only.
- [x] Add opt-in VTU output for `rhoLiquid`, `clearLevel`, `poolLevel`,
      `headspacePressure`, and a clearly labeled cell-centre `poolOccupancy`;
      report its signed global volume error.
- [x] Retain initialization and accepted-step history with named species and
      compartments, headspace, thermal and closure diagnostics; generic output
      does not label all gas as hydrogen.
- [x] Add the deterministic `planar_free_surface_verification` driver for
      uniform heating, gas generation, complete escape, closed headspace, and
      boiling liquid-mass loss, with analytic tolerances and CSV stdout.
- [x] Verify the supported fixed-grid, geometry and solver scope through
      focused serial and actual registered MPI conservation/replay cases.
- [x] Document equations, ownership, all current `free_surface_*`
      keys/defaults/SI units,
      diagnostics, tests, and fixed-grid limitations in
      `docs/modeling/planar_free_surface_volume_budget.md`.

#### Milestone A — remaining integration and acceptance

- [x] Retain automatic global history for liquid mass/volume, gas compartments,
      clear/pool levels, headspace state, and normalized closure residuals
      without repeating reductions, with an explicit rank-zero CSV writer.
- [x] Complete the current full GCC RelWithDebInfo rebuild after the geometry-substrate and
      cellwise-inventory changes.
- [x] Add a cellwise conservative effective-liquid-mass inventory without
      claiming composition-preserving evaporation.
- [x] Retain solvent and individual solute extensive inventories in the typed
      surface follower. This alone does not qualify every evaporation/inlet law.
- [ ] Add conservative steam transport and an exact steam escape transfer
      before sending boiling vapor to the headspace/vent.
- [ ] Couple boiling saturation to absolute headspace pressure before allowing
      active boiling with a closed headspace; retain the explicit setup
      rejection while saturation temperature is fixed.

#### Milestone B — constrained solver-integrated planar ALE core

`PlanarALEMeshMotion` supplies the geometry/GCL authority for Cartesian
X/Y/Z and cylindrical or `SemiStructuredXY_Z` axial motion in serial/MPI. It retains topology/IDs/maps, owns
transactional trial/accept/rollback, old/new volumes, exact structured swept
face flux, quality rejection, shared geometry epochs, and cellwise GCL checks.
The integrated path adds an opt-in mutable native-solver seam, conservative
old/new-volume and mesh-relative transport, one generalized continuity target,
moving-top boundary ownership, complete refresh/invalidation for the supported
solver state, and whole-step accept/rollback. It remains restricted to the
matrix in `docs/modeling/planar_free_surface_volume_budget.md`; unsupported
optional models are rejected collectively. Within that matrix, temperature
conserves $V_c m_{l,c}^*c_{p,c}T_c$ with accepted/trial cellwise liquid-mass
density; pure-liquid density remains the liquid-volume conversion and
void-reduced mixture density does not own sensible-energy storage.

- [x] Add transactional fixed-topology geometry motion for the explicitly
      supported structured/extruded families, with MPI-conforming partition
      faces, controller ownership, and quality rollback.
- [x] Store old/new local cell volumes and exact structured swept-face mesh
      flux, then verify the cellwise geometric conservation law in serial/MPI.
- [x] Demonstrate constant-field preservation after transport operators consume
      old/new volumes and mesh-relative flux.
- [x] Add shared geometry epochs and stale detection/refresh for the core
      gradient, transport-geometry, and Rhie--Chow caches while preserving
      topology-safe graphs and partitioning.
- [x] Extend epoch refresh/invalidation to every solver-owned geometry-dependent
      cache and numeric preconditioner state in the enabled ALE matrix; keep
      other model combinations rejected.
- [x] Generalize every extensive transient/convection operator in the enabled
      ALE matrix to old/new volumes and mesh-relative flux; do not post-correct
      fixed-volume results.
- [x] Add an owned low-Mach volume-source field and generalized continuity RHS
      for SIMPLE, PISO, PIMPLE, Rhie-Chow, and coupled Krylov paths, with one
      common target-residual definition, bounded pressure-only refinement, and
      accepted per-outer-corrector residual histories.
- [x] Implement the moving-surface kinematic/dynamic and exact bubble-escape
      boundary contracts, including a moving-top slip condition that preserves
      tangential liquid flow while fixed-flux pressure ownership enforces the
      exact mesh-normal kinematics.
- [x] Add geometry-substrate serial/MPI GCL, expand-contract, shared-face, controller-alias,
      stale-cache, and mesh-quality rollback tests.
- [x] Add constant-field, nonzero-divergence, and conservative ALE liquid/gas/
      energy tests with the solver-integrated operators, including
      liquid-mass-weighted sensible energy and a checked-in solver-integrated
      verification driver; verify that standalone source/decay scalars and other unmigrated
      transports fail closed.
- [x] Snapshot and restore supported fields, model ledgers, geometry, time,
      history, and output state so a rejected outer trial leaves one accepted
      state and commits escape exactly once.
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

---

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

---

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

- [ ] Generalize the legacy single-gas substrate to the species/rate and shared
      donor/heat contracts in [Phase 12](#phase-12--general-reacting-gas-source-in-solution)
      and [Phase 14.1](#phase-141--dissolved-gas-transport-and-bubble-populations).
      Verify each species through dissolved/dispersed/resolved/headspace storage,
      source acceptance, representation changes, adaptation and MPI exchange.

- [x] Keep resolved gas `f`, dispersed fractions `alpha_d,k`, and liquid
      `alpha_l = 1 - f - sum(alpha_d,k)` distinct. Canonical conservative
      physical arrays store extensive amounts per owned cell. Published
      population number/molar densities are per bulk volume. Dissolved
      concentration `C_H2` is per liquid volume; the legacy bulk inventory
      density `I_H2 = alpha_l*C_H2` corresponds to cell amount `V*I_H2`.
- [x] Add resolved-gas molar inventory and a single-gas ledger covering
      dissolved, dispersed, resolved, produced, and escaped gas. Every molecule
      belongs to one inventory; representation changes create no gas inventory.
- [x] Extend conservative phase mass/momentum/thermal transport to low-Mach
      gas EOS and material-volume sources, keeping absolute thermodynamic and
      gauge pressures distinct and counting capillary pressure once.
- [x] Keep supplied solution properties and legacy material correlations
      external. Use validated domains for all selected transfer/slip laws.
- [x] Retain external shared ownership of the flow solver. Checkpoint phase
      inventories and cumulative physical-transfer ledgers; publish accepted
      states only.
- [x] Extend checkpoints to persistent birth identities, pending conversion
      reservations, and accepted representation-change events. Instantaneous
      component labels are not a material-lineage tracker through advection,
      breakup or coalescence.

### Stage 1 — Uniform-mesh geometric VoF

The first executable baseline is fixed, uniform Cartesian, isothermal,
source-free flow with two incompressible phases of different densities. It
establishes the geometric/conservative flow core before gas EOS, thermal and
reaction-source coupling is introduced. Supported boundary and parallel contracts
must be explicit and unsupported modes rejected.

- [x] Add compiled geometric PLIC reconstruction and conservative swept-volume
      advection with bounded fractions and shared phase face fluxes.
- [x] Add a variable-density VoF flow solver with consistent mass/momentum
      transport, variable-coefficient pressure projection, balanced surface
      tension, and computed interface curvature.
- [x] Bound the timestep by advection, explicit capillarity, and applicable
      viscous limits; reject invalid input or a failed step without publishing
      partially advanced fields/time.
- [x] Expose native mesh/field results and focused reproducible verification
      cases without changing the existing Boussinesq path.
- [x] Verify geometric plane volumes, translation/deformation transport,
      boundedness, and phase-mass conservation.
- [x] Verify static Laplace pressure, spurious-current behavior, capillary
      oscillation, and transport/projection at the intended density ratios.
- [x] Document algorithms, units, boundary/parallel scope, and numerical
      qualification separately from future physical validation.

See the [solver guide](docs/modeling/vof.md) and the dedicated
[VoF verification contracts](docs/verification/vof_solver_contracts.md).
The evidence includes density ratio 10,000, a complete capillary-wave period,
and matched-time static bubbles. Short inviscid tests do not establish
long-time parasitic-current limits, viscous rising-bubble accuracy, or coupled
reacting-solution qualification.

### Stage 2 — Conservative three-dimensional AMR

The implemented backend uses immutable Cartesian blocks and one shared
timestep in serial and MPI. Its native mesh/field transaction is the seam for
the later physical coupling controller.

- [x] Add a dyadic block hierarchy with 2:1 neighboring refinement, including
      edges/corners, and canonical conservative coarse/fine subfaces. Match
      periodic face levels and retain exact partial coarse-patch coverage.
- [x] Refine interface/curvature stencils, predicted motion buffers,
      curvature-driven thin-neck indicators, nearby interfaces/walls, supplied
      wake indicators, and imminent promotion regions. Restrict coarsening to
      uniform pure-phase siblings outside protected neighborhoods.
- [x] Expose a configurable 24–32-cell minimum-diameter target (0.31–0.42 mm
      for 10 mm bubbles), with an explicit maximum-level resolution limitation.
- [x] Prolong PLIC geometry geometrically and transfer phase masses, dissolved
      and population amounts, signed momentum, and thermal amounts
      conservatively. These quantities are passive extensives until Stage 3
      supplies their physical equations and closures.
- [x] Publish mesh/state changes atomically; freeze topology while a replay
      checkpoint is active. Rebuild native maps, halos, graphs, and fields;
      expose topology/partition generations and reject stale refinement plans
      so external transfers can be invalidated at accepted coupling boundaries.
- [x] Verify refinement/coarsening and three-dimensional interface passage
      across coarse/fine and MPI boundaries, capillary startup against a uniform
      fine reference, viscous stress, pressure balance, rejection, and replay.
- [x] Verify conservative implicit passive-momentum bounds, microscopic phase
      donor budgets and passive-carrier conditioning without inventory clipping.
- [x] Share immutable adaptive geometry and reuse compatible transport/projection
      workspaces, while refreshing numerical coefficients and protecting replay.
- [x] Add active curved-interface seam passage at density ratio ten and a stated
      finite-duration static-bubble check with a uniformly fine interface band.
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

See the [adaptive solver guide](docs/modeling/adaptive_vof.md) and the dedicated
[VoF verification contracts](docs/verification/vof_solver_contracts.md) for the
supported contracts, reproducible checks, and remaining qualification limits.

### Stage 3 — Reaction-source and interface coupling

**Numerical implementation present; physical calibration and application
qualification remain open.**

- [x] Retain the legacy microbubble substep through shared reactions-only
      kernels. Add fixed-molar-pivot upper populations, conservative birth/
      growth/dissolution/breakup/coalescence event ledgers, explicit underflow
      policy, and externally supplied rates with declared validity domains.
      Physical collision kernels use the actual carrier volume, while stored
      population densities remain bulk-based.
- [x] Add resolved-interface dissolved single-gas flux using uniquely owned PLIC or
      aligned-face area, one-sided single-gas partial pressure and Henry equilibrium.
      Implement a prescribed stagnant-film closure with exact carrier-geometry
      averaging and analytic planar/oblique checks; thickness and its physical
      validity remain supplied inputs.
- [x] Couple liquid/gas caloric EOS, thermal transport, finite chemical-donor
      and liquid-mass accounting, and dispersed slip to one material-volume
      constraint. Close global volume with thermodynamic pressure and local
      volume with a compressibility-aware pressure solve; retain actual mass,
      physical momentum and internal-energy ledgers.
- [x] Extend geometric phase transport and its source splitting consistently
      to nonzero material-volume sources; preserve boundedness and matching
      phase/momentum fluxes rather than bypassing the Stage 1 divergence guard.
- [x] Capture dispersed gas from interface-relative collision flux and an
      explicit coalescence efficiency; retain uncaptured inventory in liquid.
      Use actual molecular Stefan flux and relative slip; EOS-only motion
      does not count as a collision flux.
- [x] Define unresolved drainage/rupture through supplied efficiency/rate
      laws and retain geometric VoF coalescence at resolved grid contact.
- [x] Define fixed-domain headspace ownership by accepted fully gas cells in
      supplied regions. Record only actual vented legacy gas/inert amounts, enthalpy and
      momentum as escape, and publish cumulative ledgers only on commit.
- [x] Verify production, dissolution/growth, capture and escape against
      independent manufactured rates and single-gas/liquid/thermal budgets.
      Cover periodic momentum, source rejection, mixed-phase remap,
      retained tiny-phase inventories, stale trials and checkpoint replay,
      with serial, two-rank and empty-rank cases.
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

See the [current source/VoF coupling contract](docs/modeling/reacting_vof.md) and the separate
[VoF verification contracts](docs/verification/vof_solver_contracts.md) for the
implemented contracts, supplied inputs and reproducible verification scope.

### Stage 4 — Bidirectional representation changes

**Numerical implementation and focused verification are present; physical
localization/calibration and application convergence remain open.**

- [x] Add a checkpointed birth registry with caller-supplied persistent spatial
      IDs, eligible fixed-molar-bin donors, and one whole bubble per promotion.
      Bind reservations to state/topology revisions and require the insertion
      and guard region to be sufficiently resolved at `max_level`. Void
      fraction alone is not a birth criterion.
- [x] Use configurable physical-size and resolution hysteresis. Initial
      diameter defaults are 10 mm and 4 mm, with 24 cells across promotion and
      a 16-cell demotion resolution threshold. Demotion inspects actual cell
      widths, not an unused finer configured level.
- [x] Identify complete gas components through shared PLIC traces across
      coarse/fine, periodic and MPI faces. Retain attached ligaments,
      wall/headspace-connected components, and inert-containing gas. Current
      conversion stencils must fit the domain coordinate box; wrapped events
      are explicitly deferred.
- [x] Transfer typed gas inventories, internal energy and physical momentum
      atomically. Redistribute the complete remaining carrier, including
      liquid, dissolved/donor stock and other dispersed populations, within
      disjoint event supports. Preserve fixed-pivot number/molar moments,
      including tiny nonzero bins; reject undefined generic passive transfers.
- [x] Keep conversion at fixed physical time with retained phase-volume,
      pressure and energy histories. The next requested physical step performs
      EOS/capillary relaxation; a private predictor initializes coupled source
      correctors without duplicate time, heat or production. Report actual
      surface-area/energy and represented-number changes separately.
- [x] Verify whole-bubble and round-trip budgets, local displacement, donor
      exhaustion, hysteresis/refinement guards, distributed component/event
      ownership, rejection, stale plans and checkpoint replay. Cover serial,
      two-rank and active sparse conversion with an empty-owned rank.
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

See the [representation guide](docs/modeling/bubble_representation.md) and the
separate [VoF verification contracts](docs/verification/vof_solver_contracts.md)
for the implemented API, reproducible checks and qualification limits.

---

## Phase 23 — AMR for native extruded, polyhedral, and mixed-region meshes

**Status: Stages 1–6 implemented and verified within the stated scope; Stage 7
implemented and under qualification; Stages 8–9 remain proposed.** See the
[general AMR design](docs/architecture/general_amr.md) for the algorithms,
ownership/API contracts, admission rules, and focused acceptance matrix.
Geometry refinement and VoF/physical-coupling qualification are separate gates.
Preserve the corrected Cartesian [transport/state contracts](docs/architecture/vof_transport.md)
and [numerical verification gates](docs/verification/vof_solver_contracts.md)
throughout; each non-Cartesian algorithm also needs its own focused and physical gates.

### Stage 1 — Common hierarchy contracts and Cartesian adapter

- [x] Introduce immutable provider-specific ancestry behind an
      `AdaptiveHierarchy` seam, preserving the `CartesianAMR` API and optimized
      backend. Keep physical region/material identity separate from fragments,
      canonical GIDs, and MPI/local field order.
- [x] Track topology, partition, geometry, and accepted-state/control revisions
      independently. Pin plans and trials to exact source observations and
      preserve collective preflight, atomic publication, and stale-plan rejection.
- [x] Separate canonical inventories/history and immutable trial ledgers from
      derived fields and mutable workspaces. Require exact collective geometry
      and numerical-operator cache compatibility; schema identity is independent
      of mesh identity, and reinitialization invalidates old trials.
- [x] Verify unchanged corrected Cartesian geometry/numerics, native traversal,
      independent repartition stamps, retained-trial isolation after failure,
      cache refresh/reuse, and rollback without compatibility CSR.

Stage 1 support and qualification limits are maintained in
[general AMR qualification summary](docs/architecture/general_amr.md#qualification-status). Native partition
bindings qualify hierarchy, FVM and solver construction; live accepted-state
repartition/migration uses the separate detached Stage 6 path.

### Stage 2 — Geometric kernels and ancestry-based overlap

- [x] Add certified actual-cell halfspace clips, sections, moments, face
      apertures, and nested ancestry overlap records. Preserve positive tiny
      pieces; reject invalid/sliver geometry instead of dropping volume.
- [x] Factor reusable sparse transfer semantics while retaining physical
      carrier geometry. Use polygon-area times Z-overlap for bulk products and
      actual 3D clipped intersections for tilted PLIC phase volumes.
- [x] Verify analytic cube/tetrahedron/prism/skew cuts, tangencies, complements,
      represented coverage under scale/translation, first moments, and signed
      and nonnegative extensive conservation. Use provider-specific arithmetic
      bounds on actual clipped queries; retain real-gap rejection without
      nominal-spacing/origin-scaled tolerances or coverage normalization.
- [x] Preserve positive phase remnants hidden by rounded pure fractions,
      moving paired inventories by consistent overlap weights only when that
      retained phase volume lies within its consistency bound; reject material
      carrier mismatches.

Stage 2 support and qualification limits are maintained in
[certified geometry and remap scope](docs/architecture/general_amr.md#qualification-status).
This utility stage does not enable general-mesh VoF or a new refinement provider.

### Stage 3 — Whole-product XY/Z refinement

- [x] Add separate base-cell/edge and axial-interval ancestry, deterministic
      triangular base refinement with conformity closure, and common Z
      bisection. XY refinement initially covers whole columns.
- [x] Add quality-checked convex quadrilateral/polygon templates with complete
      inverse families. Preserve original polygonal walls and patch labels.
- [x] Verify XY-then-Z/Z-then-XY key and geometry identity, factor-specific
      sibling coarsening, exact parent recovery, compact O(base + layers)
      storage, and the native composite serial/MPI route. Initial extrusions
      are nonperiodic; new periodic descriptors need separate qualification.

Stage 3 support and limits are maintained in
[whole-product construction and remap scope](docs/architecture/general_amr.md#qualification-status).
Compact provider/hierarchy storage does not imply scalable distributed remap
catalog construction; bounded complete-family catalogs remain a bootstrap.
General-mesh VoF/flow and local/mixed product adaptation remain later stages.

### Stage 4 — Convex polyhedral refinement

- [x] Clip the actual convex parent with its AABB midpoint planes; record a
      variable set of nonempty octant children. Certify convexity, closed
      topology, representability, child quality, containment, and disjoint union.
- [x] Construct canonical parent-face subdivisions from both incident split
      patterns and propagate new edge vertices through every incident loop.
      Restore saved parent geometry on complete-family coarsening while
      retaining finer-neighbor face tiles.
- [x] Verify variable child counts, unrelated neighboring cuts, face/edge
      closure, signed incidence, volume/first moments, sliver rejection,
      transactional failure, and exact parent-shell recovery.

Stage 4 support and limits are maintained in
[convex hierarchy and exact explicit-provider scope](docs/architecture/general_amr.md#qualification-status).
This is bounded replicated canonical planning with sparse native resident
providers, not scalable distributed hierarchy storage. The separate detached Stage 6 path qualifies exact sparse shard reassembly. Stage 4 alone does not qualify mixed-region closure
(Stage 5) or general VoF/flow.

### Stage 5 — Mixed-region closure and local XY/Z products

Implemented and verified within the scope recorded in
[local/mixed qualification scope](docs/architecture/general_amr.md#qualification-status). The boundary includes
shared exact base/axis/run views, typed multiple adaptive/static participants,
stable physical-patch joins, bounded physical-span closure and closed face/edge/
vertex admission (including one-generation comparable factor traces), exact multi-provider
native tiles, grouped complete native-patch membership, and independent sparse
patch cardinality catalogs. Explicit O(interface tiles) seam metadata remains a
bounded bootstrap cost rather than a universal base-plus-layers bound. Contacts retain
separate native domains. Mixed storage reports are explicitly nonadditive and
do not establish complete planner/RSS or distributed scaling bounds. Native
field/remap qualification does not enable a general VoF/flow solver.

- [x] Represent local extrusion refinement as shared base patches times Z
      runs, with immutable shared base topology/metrics and compact seam
      descriptors. Account for unique templates and local-product overhead.
- [x] Close refinement across region joins using physical scales and common
      interface ancestry; preserve same-fluid pressure connectivity and
      distinct material/contact laws. Keep unsupported providers static.
- [x] Initially require nested supported joins. Add validated multi-provider
      subface descriptors for local products after qualifying coverage and
      ownership; crossed anisotropic tilings must close compatibly or reject.
- [x] Verify one canonical integrated flux per tile, complete unique coverage,
      patch inheritance, periodic support limits, contact isolation, bounded
      closure, and compact repeated seam storage.

### Stage 6 — Distributed planning and migration

Implemented and verified within the bounded scope of the design. Detached
owner-local Cartesian/product/convex/Mixed kernels use typed ancestry and exact
registered face/edge/node contact authorities, sparse actual native maps,
full requested ghosts and outer metric neighbors. Named accepted/history frames
preserve independent gas/carrier amounts and pair metadata. Ownership-only
migration has a distinct whole-domain frozen-native route for unsupported
accepted geometry; any topology request on that route rejects globally.
Measured compute work and final prepared native-plus-state application packet
bytes govern balance acceptance before owner exchange. Payload, capacity
estimates, packet scratch and normalized process RSS are separate reports.
Qualification limits are maintained in [the general AMR design](docs/architecture/general_amr.md#qualification-status).
The initial complete-source bootstrap remains bounded; this stage does not
qualify general VoF, sparse ALE, production scaling or physical applications.

- [x] Replace root-assembled AMR planning with distributed closure, ancestry
      lookup, canonical face/edge decisions, and owner-to-owner geometry/state
      migration. Use the present root assembly only as an explicitly bounded
      bootstrap for preceding stages.
- [x] Replace Stage 2's global-maximum-width padded native certificate bootstrap
      with variable-length sparse owner packets; bound heterogeneous-shell memory
      by actual local/imported payload and add a sharded typed ancestry directory.
      Include distributed seam/contact proofs and explicit membership payloads,
      planner ancestry, partition maps and actual per-rank memory accounting.
- [x] Balance measured reconstruction/interface/population work and migration
      cost; retain sparse native fragments and reconstruction halos.
- [x] Verify partition-equivalent keys and budgets, split sibling families,
      observed/asserted empty owners, unique ownership, collective failures
      without deadlock, and per-rank memory bounds without global explicit
      geometry or solution copies. Separate metadata payload from normalized
      process RSS in bounded correctness and work-accounting checks. Production
      scaling and matched performance qualification remain open.

### Stage 7 — General VoF and flow qualification

The opt-in [conservative face transport backend](docs/architecture/conservative_face_vof.md)
implements independent donor-cell streams, measure-only geometric candidates,
shared endpoint/export limiting, and reusable finite momentum preparation. Keep
`SweptPartition` as the default while the coupled physical gates remain open.

- [x] Implement the opt-in source-free fixed-mesh conservative face contract.
- [x] Separate scalar PLIC measurements from explicit closed-surface construction.
- [x] Reuse finite momentum routing/operator/preconditioner across component RHSs.
- [x] Expose explicit face-aligned surfaces with bounded halo planar support.
- [ ] Qualify high-density-ratio drop translation and rotation/deformation accuracy.
- [ ] Unify mixed-cell and face-aligned curved surface fitting and qualify capillarity.
- [ ] Qualify open boundary interfaces and multiple physical streams per face.
- [ ] Complete relevant coupled-flow gates before switching the ordinary default.


**Experimental source-free execution:** explicit VoF enablement permits focused finite geometry/flow development. Reacting-VoF schema migration and full-flow qualification remain open; historical implementation records are not current validation.

**Floating-point arithmetic is the primary VoF implementation and development
priority.** `ArithmeticMode::Ordinary` is the general-VoF default, with explicit
`BoundedCertification` and `Exact` selections. Focused numerical coverage is
available for the source-free, closed, nonperiodic convex-native backend. Overall Stage 7 physical
qualification remains incomplete. The [general AMR design](docs/architecture/general_amr.md#qualification-status)
maintains the scope and limits. Checked items below mean implementation plus
the specified focused coverage, not production or application qualification.

- [x] Implement polyhedral PLIC, canonical unsplit swept-volume transport,
      independent complementary phase ledgers and finite accepted histories.
      Publish independent binary64 gas/carrier amounts once per step/remap with
      counted roundoff bounds; retain conservative inventory accounting without
      clipping. Arithmetic range support alone does not qualify interface resolution.
- [x] Expose mode-typed PLIC geometry, points, measures and results, with checked
      runtime adapters for configured solver policies. Preserve publication
      uncertainty, provenance and resource caps; keep simple members inline and
      heavier conversions in compiled providers. Cover type separation and
      finite analytic geometry without changing the numerical acceptance gates.
- [x] Remove the native exact-geometry prerequisite and legacy geometry aliases.
      Use ordinary PLIC cells, faces, extrusion recipes and atomic complexes in
      AMR, with optional Exact conversions behind public PLIC interfaces. Retain
      source support ancestry and geometric coverage gates, independently validate
      bounded imports, and test zero exact work in native/FVM/solver finite paths.
- [x] Implement source-free compatible implicit momentum, physical-coordinate
      pressure/capillary/viscous operators, geometry-based timestep controls,
      attainable inner solves and original physical-residual acceptance under
      shared iteration/resource limits.
- [x] Balance missing-direction normal momentum with physical wall-pressure
      traction on qualified closed slabs. Check pressure offsets, cell momentum,
      face flux and divergence in both finite modes, including empty MPI owners;
      retain separate full hydrostatic and general-flow qualification gates.
- [x] Verify focused reconstruction, donor budgets, microscopic/subnormal
      inventories, stale identity, rejection, rollback, regrid, policy and
      empty-owner contracts in serial and MPI.
- [x] Share immutable topology and geometry preparation with identity/epoch/map
      invalidation. Reuse numerical operators only for matching preparation.
      The optional rational-route implementation/history and backlog are owned
      by [ExactVoF](docs/architecture/exact_vof.md#recorded-implementation-and-remaining-work),
      [ExactTransport](docs/architecture/exact_transport.md) and
      [ExactGeometry](docs/architecture/exact_geometry.md).
- [x] Implement a necessary scale-invariant interface-support/model screen
      and typed refinement requests, retaining small grazing cuts when their
      neighborhood is supported. Reject without changing accepted inventory.
      This does not establish full physical thickness/curvature resolution.
- [ ] Qualify physical resolution and convergence, including caller/controller
      refinement or reinitialization, reevaluation and exhausted/unavailable
      refinement. Use neither an absolute phase-volume cutoff nor arbitrary
      precision as a substitute for spatial resolution. Mandatory arbitrary
      subcell-phase fit success is retired; preserve accepted mass throughout.
- [ ] Complete full-flow qualification of the implemented default
      [finite arithmetic policy](docs/architecture/general_amr.md#primary-floating-point-policy).
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
- [x] Retain generated source-edge incidence and authenticated construction replay,
      with focused finite predicate, planar-admission, optional PLIC trial and
      serial/MPI transport coverage. Account shared immutable allocations once
      without increasing construction caps. Full physical horizons remain open.
- [x] Add private Ordinary split drafts and joint measurement admission in shared
      geometry. Reuse recovered source identities, validate connected directed
      boundaries, preserve contact sections and shared caps, and retain inside-only
      operations without requiring an unused sibling measure.
- [x] Add one deterministic Ordinary reconstruction retry for failed measured
      strict tetrahedral cuts when every parent coordinate can be reframed
      without loss. Rebuild shared crossings from the original parent/cut and
      retain replay and resource gates. This does not qualify tiny-feature relative
      accuracy or the complete flow horizon; strict cuts cannot become contacts.
- [ ] Admit contact-like computational-fragment coalescence only through explicit
      conservative bounds and shared existing face/cell/phase budgets. Keep face
      streams, donor/recipient inventories, momentum and replay consistent; retain
      independent moment/section requirements for surface-sensitive consumers.
- [ ] Replace full retained-donor and route scans with conservative spatial
      candidates and donor indexing; reuse overlay buffers and measure local work
      scaling without changing cumulative construction-budget semantics.
- [ ] Prepare finite momentum routing and its operator once per three-component
      update, retaining independent physical-residual and iteration gates.
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

Periodic images, material-volume/species sources, low-Mach thermodynamics,
population drift and representation coupling remain outside this backend;
Stages 8–9 retain their separate gates. Per-rank memory guards are engineering
admission limits, not physical constraints.


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

---

## Global acceptance criteria

**Overall status: not met.** The core configure/build/test criteria are
satisfied, but an auxiliary verification-launcher path, open model work,
turbulence validation, and documentation criteria remain.

- [x] `cmake --preset GCC-ninja-multi` succeeds.
- [x] `cmake --build --preset GCC-Debug` succeeds.
- [x] `ctest --preset GCC-Debug` passes, including registered MPI tests, in the
      supported local development environment.
- [x] Require a representative GCC Debug configure/build/test job in CI while
      retaining GCC and LLVM Release coverage.
- [x] Any MPI failures caused by sandbox/PMIx restrictions are reported separately from code failures.
- [x] Existing verified cases remain stable when new physics is disabled.
- [x] New physics fields are bounded and finite.
- [x] New source terms are dimensionally documented.
- [x] VTU output supports the implemented solver feedback fields.
- [x] Supported runtime model selectors and options can be changed from
      `Database` without recompilation.
- [x] Documentation clearly states which models are engineering placeholders and which are verified numerical capabilities.
- [ ] Complete the Phase 18 key/default/unit reference.
- [ ] Satisfy every acceptance criterion in the non-deferred roadmap phases.
- [ ] Close the remaining quantitative RANS validation criteria.
