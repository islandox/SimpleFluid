# TODO: SimpleFluid Multiphysics Roadmap

**Documentation review date:** October 24, 2026

This refresh reconciles the current source, public APIs, and checked-in test
coverage. It does not represent a new runtime qualification or full-suite run.

**Near-term goal:** generalize reacting gas sources in solution while retaining
the implemented RANS, single-gas, boiling, void-fraction and thermal-feedback
foundation. Close the reaction/species/donor/heat contracts and the remaining
physical-reference, example and configuration gaps. General chemistry and
production external-source coupling remain proposed; the current flow model
is not a full Euler–Euler two-fluid solver.

## Roadmap dashboard

The numbered phases record dependency and implementation history; they are not
the current execution order. Work from the priorities below, and use each
phase's unchecked tasks and acceptance criteria as its completion contract.

### Current priorities

Resolved-interface VoF, general AMR, and resolved/dispersed representation
conversion are excluded from this source snapshot. Their implementation and
qualification remain outside the retained reacting-flow support matrix.

1. Implement the general reaction-rate/species contract in Phases 12 and 14.1,
   including shared donor limits and accepted reaction heat.
2. Qualify the authenticated pitzDaily acceptance manifest with converged,
   checked-in OpenFOAM and SimpleFluid profiles and evidence-based tolerances.
3. Decide whether Phase 14's optional scalar-void transport/slip path is still
   needed now that Phase 14.1 owns bubble transport; implement it or explicitly
   defer it.
4. Add Phase 16 demo validation checks for ParaView-readable output and
   disabled-reaction/boiling baseline behavior after adding the general demo.
5. Complete the Phase 18 user-facing configuration reference.
6. Add quantitative Phase 14.1 physical validation and a checked-in combined
   RANS-plus-bubble user example.
7. Physically validate and harden the constrained Phase 20.1 planar-ALE path,
   then add conservative steam transport/escape, composition-resolved liquid
   transport, and phase-aware conservative coupling before widening its support
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
| Phases 10–12, sources/materials/reactions | Prescribed heat and single-gas substrate implemented | General rates, stoichiometry, donor limits and reaction-heat coupling |
| Phase 13, boiling | Partial | Wall heat-flux partitioning interface |
| Phase 14, scalar void | Partial | Advection and disposition of the low-order slip path |
| Phase 14.1, bubble populations | Single-gas substrate with combined regression | General species/closures, physical validation and combined example |
| Phases 15–16, feedback/demo | Partial | Property extensions and demo acceptance checks |
| Phase 17, focused tests | Existing specialized scope covered | General reaction/species/donor/energy regression gates |
| Phase 18, documentation | Partial | Complete key/default/unit/validity reference |
| Phase 19, solution species | Linear source/decay scalar substrate implemented | General composition, stoichiometry and shared reaction/transfer inventories |
| Phase 20, external reaction/source maps | In-memory heat scaffold and reusable conservative cell transfer implemented | General source/species exchange and external-provider protocol |
| Phase 20.1, planar free surface | Fixed-grid path plus constrained solver-integrated planar ALE | Physical validation, steam/composition transport, wider model support, geometric occupancy, and coupled inventory mapping remain open |
| Phase 21, Euler–Euler | Deferred | Requires validated lower-order models first |
| Phase 22, resolved/dispersed bubbles | Excluded from this snapshot | Separate implementation and physical qualification required |
| Phase 23, general native AMR | Excluded from this snapshot | Separate geometry, migration, and flow qualification required |

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
        legacy single-gas source, boiling, void, material feedback, linear
        source/decay scalars, and steady-state search
        through the native field path
  - [x] Preserve exact legacy `Mesh` identity, public solver hooks, and
        established behavior through a distinct compatibility path
  - [x] Cover Cartesian and cylindrical serial/distributed execution with
        focused regressions
  - [x] Cover native unstructured execution directly in serial and through
        `MeshPartitioner`/`PartitionedMesh` in MPI
  - [x] Cover `SemiStructuredXY_Z` in serial and explicitly reject its
        unsupported multi-rank construction
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
The current specialized provider retains the legacy compatibility identifiers
`PrescribedHeatSource` and `qdot_external`; general reaction-energy coupling is
proposed, not implemented by renaming that provider.

- [x] Support constant and Gaussian prescribed heating and caller-supplied
      non-negative cell fields.
- [x] Normalize spatial profiles to configured total power on Cartesian and
      cylindrical meshes and couple the result to the temperature equation.
- [x] Cover zero heating and analytic one-cell temperature rise in focused tests.
- [ ] Add general source names and an explicit compatibility adapter.
- [ ] Evaluate reaction heat from the same accepted extents used for species
      production and donor consumption, matched to the stored thermal potential
      and its excluded chemical/formation-energy terms.
- [ ] Distinguish prescribed external heat, reaction heat, phase-change latent
      heat, and transferred species enthalpy so each is counted once.
- [ ] Verify exothermic/endothermic reactions, zero reaction with independent
      heating, donor-limited heat release, and rejected-step energy rollback.

---

### Phase 12 — General reacting gas source in solution

Build the source contract in
[reacting gas sources in solution](docs/modeling/reacting_gas_sources.md).
This low-order branch supplies an unresolved gas-volume source; it does not
solve a separate gas momentum equation or establish general chemistry support.

**Status:** the existing single-gas, prescribed-energy-yield source supplies
ideal-gas volume conversion, bounded scalar source application, and focused
temperature/pressure/zero-source checks. The implementation retains
`ReactingGasModel` and `S_alpha_reaction` as legacy compatibility identifiers.
Arbitrary species, stoichiometry, reaction-rate providers, and coupled donor/
energy acceptance remain open.

#### Proposed source contract

Let $r_j$ be reaction extent per liquid volume and time, in
mol/(m³ liquid·s), and $\nu_{sj}$ signed moles of species $s$ per mole of
reaction $j$. For bulk-volume source fields:

$$
\dot n_s = \alpha_l \sum_j \nu_{sj} r_j,
\qquad
\dot q_{\mathrm{reaction}}
= \alpha_l\sum_j e_j r_j.
$$

$\dot n_s$ has units mol/(m³ bulk·s), and $\dot q_{\mathrm{reaction}}$ has
units W/m³ bulk. The signed $e_j$ is thermal energy per mole of reaction,
positive when added to the selected stored thermal potential. Derive it from
a declared enthalpy or internal-energy convention and only chemical/formation
terms excluded from that storage. Do not insert $-\Delta H_j$ into an
internal-energy equation or count composition-dependent sensible energy,
pressure-volume work or transported species energy again.
A provider already supplying bulk-volume rates must declare
that basis and must not receive another $\alpha_l$ factor. Heating is an
independent input unless a selected reaction law explicitly couples it.
Only accepted net molar transfer into the gas phase contributes to the
ideal-gas source $S_{\alpha,\mathrm{reaction}}=(RT/p)\sum_s\dot n_s^{g}$
in the fixed-temperature/fixed-absolute-pressure approximation; dissolved
production remains in the liquid inventory until a declared release or
transfer closure moves it.

#### Implemented numerical foundation

- [x] Convert an admitted single-gas molar production rate to ideal-gas volume
      using temperature and positive absolute pressure.
- [x] Limit the legacy scalar source by its configured rate and void capacity
      and publish its source field for the authoritative void update.
- [x] Cover zero input, prescribed-rate scaling, temperature/pressure scaling,
      and scalar-void bounds in the existing source tests.

#### Generalization tasks and acceptance gates

- [ ] Define species identity, molar mass, solution composition, concentration
      basis, reaction stoichiometry, rate units and validity domains explicitly.
- [ ] Accept prescribed cell/time rates or a supplied kinetic callback using
      local temperature, absolute pressure and composition.
- [ ] Permit production and consumption through signed stoichiometry; validate
      elemental/mass closure or declare explicit external material reservoirs.
- [ ] Allocate products to dissolved and gas inventories once using an explicit
      release/partition model rather than equating total production with void.
- [ ] Limit competing reactions jointly by available donor amounts, gas/phase
      capacity and the energy contract; retain rejected requests diagnostically.
      Reject or subcycle a failed step instead of clipping accepted inventory.
- [ ] Use the same accepted extent for donor consumption, product generation,
      phase mass change and reaction heat.
- [ ] Verify closed-batch stoichiometric depletion, competing donor requests,
      multiple gases, reversible consumption, independent heat input and
      rate-basis equivalence.
- [ ] Verify species/elemental, phase-mass and energy residuals in serial and
      MPI, including empty ranks, failure propagation and accepted-state replay.

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

#### Existing legacy scalar update

$$
\alpha_g^{n+1} = \operatorname{clamp}\left(\alpha_g^n + \Delta t\,S_{\alpha,\mathrm{total}},\alpha_{\mathrm{min}},\alpha_{\mathrm{max}}\right)
$$

where:

$$
S_{\alpha,\mathrm{total}} = S_{\alpha,\mathrm{reaction}} + S_{\alpha,\mathrm{boil}} + S_{\alpha,\mathrm{wall}} - S_{\alpha,\mathrm{collapse}}
$$

Here the subscript `rad` is the legacy source identifier, not the general
reaction-rate contract. The generalized path must admit a consistent source
before updating void; bounding void alone cannot conserve species or energy.
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

Extend the Phase 12 source contract to dissolved species, microbubble and
large-bubble populations, pressure-dependent equilibrium, interphase transfer,
growth/dissolution, rise and escape. The general design is specified in
[reacting gas sources in solution](docs/modeling/reacting_gas_sources.md).

**Status:** a specialized single-gas two-population implementation has focused
conservation, transport and serial/MPI coverage, including permanent serial
and exact-two-rank RANS-plus-bubble regressions. Its runtime selector
`reactingBubbleModel`, value `sheng2024TwoPopulation`, and dissolved field
`C_H2` remain legacy compatibility identifiers. These tests do not establish
generic species support, general reaction kinetics or physical validation.

#### Implemented numerical foundation

- [x] Keep thermodynamic absolute pressure separate from incompressible
      pressure correction, with constant, prescribed-history, reconstructed,
      and experimental inertial-pressure options.
- [x] Store dissolved concentration per liquid volume and population number/
      gas amounts per bulk volume, with explicit diagnostic inventory totals.
- [x] Apply a documented Henry-law convention and Laplace pressure, positive
      radius/pressure guards and focused equilibrium monotonicity tests.
- [x] Solve the bubble EOS with a guarded positive-root method and explicit
      empty-population behavior; reconstruct population and aggregate void.
- [x] Use paired conservative transfer for category conversion, dissolution
      and supersaturated growth in the existing single-gas closure.
- [x] Transport dissolved inventory through the scalar transport infrastructure
      and advect bubble number and moles with the same population velocity.
- [x] Support zero slip, prescribed upward slip and the existing correlated
      rise closure within its stated validity range.
- [x] Record boundary escape from the accepted transported state and localize
      rate fields to boundary-adjacent owner cells.
- [x] Split transport and local kinetics; use analytic linear decays, bounded
      local transfers and lifetime-limited subcycling.
- [x] Preserve single-gas inventories in source-free closed tests and balance
      imposed production and escaped amounts in source/transport tests.
- [x] Test radius residuals, dissolution, conservative conversion, growth,
      pressure response, void reconstruction and substep convergence.
- [x] Reduce inventory/event diagnostics across ranks and verify finite-Courant
      serial/MPI escape, including the combined RANS and density-feedback path.

#### General species and population contract

For dissolved species $s$ and bubble class $k$, use bulk-volume inventories
$\alpha_l C_s$ and $M_{s,k}$, with class number density $N_k$. Dissolved
transport, supplied reaction production/consumption and interfacial transfer
must balance species-wise:

$$
\frac{\partial(\alpha_l C_s)}{\partial t}
+\nabla\cdot(\alpha_l U_l C_s)
=\nabla\cdot(\alpha_l D_s\nabla C_s)
+\dot n_s^{l}-\sum_k J_{s,k}.
$$

Each $J_{s,k}$ is added once to its gas inventory and subtracted once from
dissolved storage. Mixture equilibrium uses species partial pressure and the
selected Henry convention. Bubble EOS uses the sum of species moles; number
density and the representative-size closure are separate inputs. ALE storage
uses accepted-old/trial-new volumes and mesh-relative transport in the
explicitly supported solver matrix.

- [ ] Generalize species fields, rates, property selection and output names
      without presenting legacy aliases as an implemented new API.
- [ ] Supply species-specific solubility, diffusivity, molar mass, caloric
      properties and validity bounds; reject unsupported mixtures/conditions.
- [ ] Introduce configurable nucleation and growth/dissolution closures with
      documented units, calibration domains and donor-limited seed production.
- [ ] Preserve every species through population conversion, escape and
      representation changes; include dissolved and headspace inventories.
- [ ] Couple reaction consumption and interfacial transfer to one donor budget,
      with explicit disposition at population and void bounds.
- [ ] Verify two or more species, partial-pressure equilibrium, dissolution to
      depletion, shared donor competition and conservation under repartition.
- [ ] Add source-Jacobian hooks and evaluate an implicit local kinetics option.
- [ ] Keep the generalized local update Kokkos-compatible and allocation-free
      per cell; report substeps, failed closures and inventory residuals.
- [ ] Add a checked-in combined RANS/reacting-gas example and quantitative
      gas-release/bubble-column validation using declared reaction/property
      inputs and mesh/timestep convergence.
- [ ] Quantify uncertainty for selected nucleation, transfer and slip closures.
      Two representative radii remain a reduced distribution; prescribed slip
      is not a gas momentum equation. Phase-aware turbulence remains Phase 21.

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

Provide a small cylindrical-tank example with independently supplied reaction
rates and heating, an explicit solution composition, gas release, optional
boiling, and complete accepted species/mass/energy diagnostics.

**Status:** the existing prescribed-source examples build and have short-run
coverage. `reacting_solution_tank_demo` is a legacy executable name, and
`constant_power_cylinder_vessel` also uses the specialized source path; neither
is an implemented general reacting-solution demo.

- [x] Retain cylindrical mesh, quiescent initialization, prescribed spatial
      heating, optional boiling and temperature/velocity/pressure/void output
      as an existing numerical/example foundation.
- [x] Retain focused Debug/Release build and short serial-run coverage for the
      existing examples.
- [ ] Add a generally named example with user-supplied reaction rates,
      species/stoichiometry, donor concentrations and independent external heat.
- [ ] Output species concentrations and phase inventories, requested/accepted
      reaction extents, reaction heat, gas/boiling sources, latent heat, density
      feedback and conservation residuals with explicit SI units.
- [ ] Document rate/property provenance, source-basis conversions and the
      supported transport/pressure/parallel scope.
- [ ] Verify output opens in ParaView.
- [ ] Verify disabled reactions/boiling recover the independently heated
      natural-convection baseline.
- [ ] Verify donor depletion stops production, permitted gas release produces
      bounded void, and species/mass/energy budgets close.
- [ ] Add matched serial/MPI and mesh/timestep convergence checks before
      presenting the example as physical validation.

---

### Phase 17 — Unit, regression, and conservation tests for reacting gas and boiling

Collect the new physics tests into a focused verification suite.

**Status:** tests cover the existing specialized Phase 12–15 implementation.
General reaction/species/donor/heat tests remain open; this documentation
revision does not claim a new runtime run or close physical-validation gates.

- [x] Legacy single-gas zero-source test.
- [x] Legacy prescribed-yield source linearity with respect to heating.
- [x] Legacy prescribed-yield source linearity with respect to yield.
- [x] Ideal-gas temperature scaling test.
- [x] Ideal-gas pressure scaling test.
- [x] Alpha boundedness test.
- [x] Boiling threshold test.
- [x] Boiling latent-heat conservation test.
- [x] Wall-boiling owner-cell distribution test.
- [x] Density feedback monotonicity test.
- [x] Viscosity floor test.
- [x] One-cell alpha update analytic test.
- [x] `Database` parsing/defaults test.
- [x] Regression test that all models disabled leaves old solver behavior unchanged.

- [ ] Add the generalized Phase 12/14.1 species, donor and energy acceptance
      tests, including competing rates, phase transfer, MPI and rollback.

#### Phase 17 acceptance criteria

- [x] All new tests are deterministic.
- [x] Tests are safe in serial and do not require MPI launch unless explicitly marked.
- [x] The Debug test suite, including registered MPI tests, passes in the
      supported local development environment.
- [x] Tolerances are documented and physically meaningful.

---

### Phase 18 — Documentation and configuration examples

Document the general reacting-gas source contract, low-order scalar path and
population path separately from a full Euler–Euler formulation.

**Status:** the general design is documented in
[reacting gas sources in solution](docs/modeling/reacting_gas_sources.md).
Current code still exposes legacy single-gas names and closures; a complete
key/default/unit/validity reference and implementation migration remain open.

- [x] Document governing quantities, storage bases, conservation gates and
      limitations of the existing numerical foundation.
- [x] Separate weak RANS-plus-bubble coupling from phase-aware turbulent
      Euler–Euler flow and distinguish proposed design from current APIs.
- [ ] Document every supported configuration key with default, SI unit,
      validity range and compatibility mapping.
- [ ] Document supplied rate/stoichiometry/property interfaces, coupled donor
      limits, heat/phase-change accounting, restart state and rejection behavior.
- [ ] Provide runnable zero-reaction, closed-batch and convecting-solution
      examples with auditable source and property inputs.
- [ ] Ensure a developer can configure each supported model from the user
      reference without consulting implementation headers.

---

### Phase 19 — Solution-species transport and reaction inventories

Use conservative liquid-fraction-weighted transport as the basis for solution
composition and reaction donors/products.

**Status:** linear source/decay scalar transport is implemented and focused-tested
under the legacy compatibility class `ScalarSourceDecayTransport`. It supplies
configurable groups, projected-liquid-flux advection, optional diffusion,
analytic local source/decay integration, globally reduced balance diagnostics
and collective validation. It is not a general stoichiometric chemistry solver.

For species $s$, the target equation is:

$$
\frac{\partial(\alpha_l C_s)}{\partial t}
+\nabla\cdot(\alpha_l U_l C_s)
=\nabla\cdot(\alpha_l D_s^{\mathrm{eff}}\nabla C_s)
+\dot n_s^{l}-\sum_k J_{s,k}.
$$

- [x] Preserve existing zero-source, exact source/decay, transport conservation
      and bounded-concentration regression coverage for the legacy scalar path.
- [x] Retain source, boundary-outflow, positivity-adjustment and before/after
      inventory diagnostics with collective input and rank-consistency checks.
- [ ] Add general species identities, composition-dependent property hooks and
      accepted reaction/transfer sources shared with Phases 12 and 14.1.
- [ ] Transport solvent, solutes and gas-forming reactants/products with an
      explicit mass/composition basis and supported boundary compositions.
- [ ] Couple competing reactions and phase transfer to the same local donor
      inventory; do not independently limit each consumer against the same pool.
- [ ] Verify species/elemental closure, depletion, inflow/outflow composition,
      transport/reaction splitting convergence and accepted-state rollback.
- [ ] Extend transport to ALE only through conservative old/new-volume storage
      and mesh-relative flux, retaining rejection for unsupported combinations.

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
The scaffold's legacy heat field is `qdot_external`; general reaction-rate
exchange and a production external-provider protocol remain proposed.

- [x] Retain volume-averaging utilities, deterministic feedback snapshots and
      constant-field/integral-preservation mapping tests.
- [x] Retain scalar/vector/tensor cell transfer, MPI donor exchange,
      coverage/inventory diagnostics, and geometry/map invalidation.
- [x] Retain conservative Cartesian/coaxial cylindrical overlaps and supported
      cylindrical-to-XY/Z or swept R–Z overlaps with native-volume accounting.
- [x] Retain imported heat fields and the existing callback/subcycle scaffold.
- [ ] Define an external-provider contract for rate basis, species ordering,
      reaction stoichiometry, sign, timestamps and accepted source intervals.
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
verification driver. Cellwise effective-liquid-mass transport is supported;
composition-resolved solvent/solute transport and steam transport/escape
remain open. Milestone B now has a constrained solver-integrated planar-ALE
core for native mutable structured/extruded geometry, constant-area vessels,
Backward Euler, laminar physical temperature, thermal-only pure-liquid density,
cellwise liquid mass, vented headspace, and optional supported legacy
single-species two-population gas transport/escape. Broader model combinations fail closed. This is
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
- [x] Extend the legacy single-gas two-population model with MPI-reduced dissolved,
      microbubble, and large-bubble inventories; unbounded EOS bubble volume;
      void-cap discrepancy; and non-mutating candidate-pressure evaluation
      through its existing surface-tension/Laplace-radius physics.
- [x] Define bubble escape as the exact before/after submerged micro-plus-large
      mole decrement and keep dissolved boundary outflow separate.
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
- [x] Wire one authoritative accepted-step sequence through
      `BoussinesqSolver`: post-temperature bubble/void and legacy source/decay scalar updates,
      material refresh, accepted boiling mass/condensate once, pure-density
      refresh, exact single-gas escape transfer, level/headspace closure, next-step
      pressure offset, and diagnostic publication.
- [x] Leave headspace, liquid phase-change ledgers, and history uncommitted on
      a rejected closure and fail-stop the solver after any later step error;
      full-field rollback is not available, so unsafe whole-step retries are
      rejected explicitly.
- [x] Add opt-in VTU output for `rhoLiquid`, `clearLevel`, `poolLevel`,
      `headspacePressure`, and a clearly labeled cell-centre `poolOccupancy`;
      report its signed global volume error.
- [x] Retain initialization and per-accepted-step global history and provide a
      rank-zero fixed-schema CSV writer with liquid, separate single-gas population,
      headspace, boiling-energy, and normalized closure diagnostics.
- [x] Add the deterministic `planar_free_surface_verification` driver for
      uniform heating, gas generation, complete escape, closed headspace, and
      boiling liquid-mass loss, with analytic tolerances and CSV stdout.
- [x] Verify the focused fixed-grid, geometry-substrate, core, and solver scope
      in serial and with all related MPI registrations under host networking;
      run the complete
      Debug non-MPI/MPI suites.
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
- [ ] Split solvent and each reacting/nonreacting solute into extensive
      inventories before claiming composition-preserving evaporation or
      boundary transport.
- [ ] Add conservative steam transport and an exact steam escape transfer
      before sending boiling vapor to the headspace/vent.
- [ ] Couple boiling saturation to absolute headspace pressure before allowing
      active boiling with a closed headspace; retain the explicit setup
      rejection while saturation temperature is fixed.

#### Milestone B — constrained solver-integrated planar ALE core

`PlanarALEMeshMotion` supplies the geometry/GCL authority for Cartesian
X/Y/Z and cylindrical axial motion in serial/MPI plus serial
`SemiStructuredXY_Z` axial motion. It retains topology/IDs/maps, owns
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
      verification driver; verify that legacy source/decay scalars and other unmigrated
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

Full VoF interface capturing and macroscopic surface breakup are deferred
beyond this snapshot; Euler–Euler gas momentum remains Phase 21.
Completing this planar phase must not imply those capabilities.

---

## Deferred architecture — Phase 21

### Phase 21 — Full Euler–Euler two-fluid extension

This phase is deferred until the generalized scalar-source and population
paths are implemented, verified and physically qualified.

**Status:** deferred by design; none of its implementation or validation
criteria are claimed complete.

The ability to configure RANS and legacy single-gas bubble populations in the same
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
