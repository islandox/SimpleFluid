# TODO: SimpleFluid Multiphysics Roadmap

**Status date:** August 19, 2026

**Near-term goal:** finish validation and integration of the implemented RANS,
radiolytic-gas, boiling, void-fraction, and thermal-feedback stack; close the
remaining physical-reference, example, and configuration-documentation gaps;
and avoid prematurely presenting the current in-memory coupling scaffold as
production neutronics integration or the flow model as a full Euler–Euler
two-fluid solver.

## Roadmap dashboard

The numbered phases record dependency and implementation history; they are not
the current execution order. Work from the priorities below, and use each
phase's unchecked tasks and acceptance criteria as its completion contract.

### Current priorities

1. Qualify the authenticated pitzDaily acceptance manifest with converged,
   checked-in OpenFOAM and SimpleFluid profiles and evidence-based tolerances.
2. Decide whether Phase 14's optional scalar-void transport/slip path is still
   needed now that Phase 14.1 owns bubble transport; implement it or explicitly
   defer it.
3. Add Phase 16 demo validation checks for ParaView-readable output and
   disabled-radiolysis/boiling baseline behavior.
4. Complete the Phase 18 user-facing configuration reference.
5. Add quantitative Phase 14.1 physical validation and a checked-in combined
   RANS-plus-bubble user example.
6. Close the remaining foundational validation gaps: Ghia profile checks and
   bundled OpenFOAM centerline tolerances.

### Status map

| Workstream | State | Principal open gate |
| --- | --- | --- |
| Foundation, Phases -1 to 8 | Implemented with verification gaps | Ghia and bundled OpenFOAM profile validation |
| Two-equation RANS | Implemented and focused-tested | Quantitative external validation and phase-aware bubbly-flow scope |
| Phase 9, performance | Substantially complete | Backend-portable device assembly API |
| Phases 10–12, sources/materials/radiolysis | Implemented | No open phase acceptance items |
| Phase 13, boiling | Partial | Wall heat-flux partitioning interface |
| Phase 14, scalar void | Partial | Advection and disposition of the low-order slip path |
| Phase 14.1, bubble populations | Broad implementation with combined regression | Physical reproduction, checked-in combined example, and production hardening |
| Phases 15–16, feedback/demo | Partial | Property extensions and demo acceptance checks |
| Phase 17, focused tests | Complete for supported scope | Reopen when model scope expands |
| Phase 18, documentation | Partial | Complete key/default/unit/validity reference |
| Phase 19, precursors | Implemented and focused-tested | No open items for the supported transport scope |
| Phase 20, TH/neutronics map | In-memory scaffold implemented and focused-tested | Production external-neutronics protocol and validation |
| Phase 21, Euler–Euler | Deferred | Requires validated lower-order models first |

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
- [x] Native runtime-mesh solver path
  - [x] Run `FluidSolver` and `BoussinesqSolver` directly on supported
        `MeshHandle` backends without reconstructing a legacy mesh
  - [x] Select `FieldStored` scalar, vector, and tensor cell storage plus
        scalar/vector face storage through `MeshFieldTraits`
  - [x] Support mapped diffusion, convection, pressure projection,
        Rhie–Chow fluxes, and coupled pressure–velocity assembly
  - [x] Carry turbulence, wall distance/treatment, fission, radiolysis,
        boiling, void, material feedback, precursors, and steady-state search
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
- [x] Add the matched 1000 W Gaussian fissile-solution-tank SST comparison,
      including 2 mm R-Z spacing, wall-layer refinement, and R-Z error figures.
- [x] Document that the current RANS variables use single-continuum, full-cell
      transport rather than phase-volume-fraction-weighted equations.

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

The timing ceiling is a smoke-regression guard, not a claim of comparable
performance across different machines. Release scaling and profiling remain
manual measurement workflows.

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

Before adding radiolysis or boiling, make the solver accept explicit source fields and updateable material properties cleanly.

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

### Phase 11 — Fission power density and heat-source coupling

Add the thermal source used by the criticality-accident model, initially without a neutronics solver.

- [x] Add a `FissionPowerSource` or equivalent model.
- [x] Support at least these power-density modes:
  - [x] constant volumetric power density
  - [x] Gaussian spatial profile
  - [x] user-provided cell field, if the existing field API makes this straightforward
- [x] Add total-power normalization for spatial profiles.
- [x] Add source field `qdot_fission` with units W/m³.
- [x] Couple `qdot_fission` into the temperature equation.
- [x] Add optional time-dependent multiplier for transient tests.
- [x] Write `qdot_fission` to VTU.

#### Phase 11 acceptance criteria

- [x] Integrated power equals configured total power within tolerance.
- [x] Zero power gives zero thermal source.
- [x] Constant power in a closed one-cell thermal problem gives the expected temperature rise.
- [x] Gaussian profile is centered and normalized correctly on Cartesian and cylindrical meshes.

---

### Phase 12 — Radiolytic gas generation model

Implement the first conservative unresolved gas-generation model. This phase produces an `alpha_g` source but does not solve a gas momentum equation.

#### Model variables

- `qdot_fission` — fission power density, W/m³
- `T` — liquid temperature, K
- `p_ref` or local pressure if available, Pa
- `alpha_g` — gas void fraction
- `alpha_l = 1 - alpha_g` — liquid volume fraction

#### Baseline model

$$
\dot n_g = \eta_g\,G_g\,\dot q_\mathrm f
$$

$$
\dot V_g = \dot n_g \frac{\mathrm R T}{p}
$$

$$
S_{\alpha,rad} = \alpha_l \dot V_g
$$

where:

- $G_g$ is the configurable gas yield in mol/J
- $\eta_g$ is a release efficiency
- $\mathrm R$ is the ideal-gas constant
- $S_{\alpha,rad}$ has units 1/s

#### Tasks

- [x] Add `GasGenerationProperties` or equivalent configuration struct.
- [x] Add `RadiolyticGasModel` with small testable pure functions.
- [x] Add configurable parameters:
  - [x] `enableRadiolysis`
  - [x] `radiolysisGasYieldMolPerJ`
  - [x] `gasReleaseEfficiency`
  - [x] `referencePressure`
  - [x] `gasConstant`
  - [x] `alphaMin`
  - [x] `alphaMax`
  - [x] `maxSourceAlphaRate`
- [x] Compute `S_alpha_rad` as a cell field.
- [x] Clamp or limit source terms so `alpha_g` remains bounded.
- [x] Write `S_alpha_rad` to VTU.

#### Phase 12 acceptance criteria

- [x] `qdot_fission = 0` gives `S_alpha_rad = 0`.
- [x] Doubling `qdot_fission` doubles `S_alpha_rad`.
- [x] Doubling `radiolysisGasYieldMolPerJ` doubles `S_alpha_rad`.
- [x] Higher temperature increases ideal-gas volume source.
- [x] Higher pressure decreases ideal-gas volume source.
- [x] Source is non-negative and bounded by `maxSourceAlphaRate`.
- [x] Source vanishes or is limited as `alpha_g` approaches `alphaMax`.

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
T > T_{sat} + \Delta T_{act}
$$

compute a requested energy rate, capped so one explicit step cannot remove
more than the cell's sensible superheat:

$$
\dot e_{avail} = \rho_l c_{p,l}
\frac{\max(T - T_{sat},0)}{\max(\tau_b,\Delta t)}
$$

then:

$$
\dot m_b = \frac{\dot e_{avail}}{h_{fg}}
$$

$$
S_{\alpha,boil} = \frac{\dot m_b}{\rho_g}
$$

and latent heat sink:

$$
Q_{latent} = \dot m_b h_{fg}
$$

Before the temperature update, limit aggregate bulk and wall boiling to the
void capacity remaining after the current radiolytic source and optional
timestep-realizable collapse. Scale `S_alpha_boil` and `Q_latent` together so
latent energy is removed only for vapor admitted into the authoritative scalar
`alpha_g` field.

#### Wall boiling placeholder

For heated boundary faces:

$$
\dot m''_w = f_{evap}\frac{q''_w}{h_{fg}}
$$

Distribute to owner cells:

$$
\dot m_{w,vol} = \dot m''_w \frac{A_f}{V_P}
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
radiolysis and boiling workflows. This phase is the low-order scalar path:
Phase 12 produces `S_alpha_rad`, Phase 13 provides boiling sources, and
Phase 14.1 can replace the scalar update with a two-population bubble model
that reconstructs `alpha_g` from bubble inventories.

The radiolytic ideal-gas mode publishes `S_alpha_rad` and keeps its model-local
`alpha_g` unchanged. The implemented Phase 14 path owns the aggregate
`S_alpha_total` bookkeeping, bounded low-order scalar update, collapse, and
optional diffusion.

**Status:** partial. Aggregate bookkeeping and the bounded source/collapse
update are implemented. Low-order advection and slip transport remain open;
the operational slip setting belongs only to Phase 14.1 today.

#### Minimum required update

$$
\alpha_g^{n+1} = \operatorname{clamp}\left(\alpha_g^n + \Delta t\,S_{\alpha,total},\alpha_{min},\alpha_{max}\right)
$$

where:

$$
S_{\alpha,total} = S_{\alpha,rad} + S_{\alpha,boil} + S_{\alpha,wall} - S_{\alpha,collapse}
$$

with collapse bounded over the explicit step:

$$
S_{\alpha,collapse} = \min\left(\frac{\alpha_g}{\tau_c},
\frac{\alpha_g-\alpha_{min}}{\Delta t}\right).
$$

#### Optional transport upgrade

If the existing scalar transport API allows it cleanly, add:

$$
\frac{\partial \alpha_g}{\partial t}
+ \nabla\cdot(U_l \alpha_g)
+ \nabla\cdot(U_{slip}\alpha_g)
=
\nabla\cdot(D_\alpha\nabla\alpha_g)
+ S_{\alpha,total}
$$

#### Tasks

- [x] Allocate and initialize radiolytic `alpha_g` and `alpha_l`.
- [x] Allocate and initialize `S_alpha_rad`.
- [x] Allocate and initialize `S_alpha_total`.
- [x] Add bounded explicit update for the low-order scalar model.
- [x] Derive mirrored `S_alpha_total` from the published scalar state change
  instead of copying a provider's internal source history.
- [x] Add conservative bounded diffusion for the low-order scalar model.
- [ ] Add low-order scalar advection with liquid face fluxes.
- [ ] Add configurable upward slip velocity aligned opposite gravity.
- [x] Add configurable `alpha_diffusivity`.
- [x] Add optional simple collapse/removal time scale for gas disengagement tests.
- [x] Ensure radiolytic gas fields are ghost-synchronized after model advance.
- [x] Write radiolytic `alpha_g`, `alpha_l`, and `S_alpha_rad` to VTU behind output switches.
- [x] Write `S_alpha_total` to VTU after aggregate source bookkeeping exists.

#### Phase 14 acceptance criteria

- [x] One-cell constant-source update matches the analytic result.
- [x] `alpha_g` never leaves `[alphaMin, alphaMax]`.
- [x] Ideal-gas radiolysis mode does not advance `alpha_g` before the low-order update path exists.
- [x] With all sources disabled, `alpha_g` remains unchanged.
- [x] Scalar diffusion conserves total gas inventory with zero-flux boundaries.
- [ ] Advective transport conserves total gas inventory up to boundary flux and sources.
- [x] VTU output can include radiolytic `alpha_g` and `S_alpha_rad`.
- [x] VTU output includes `S_alpha_total`.

---

### Phase 14.1 — Pressure-coupled two-population radiolytic bubble model

Implement the pressure-sensitive hydrogen-bubble model from Sheng et al. (2024) as an advanced, runtime-selectable refinement of the Phase 12 ideal-gas source and the Phase 14 scalar void-fraction path.

Phase 12 remains the low-cost source model. Phase 14 remains the lower-order scalar `alpha_g` path. Phase 14.1 adds dissolved hydrogen, microbubble and large-bubble populations, pressure-dependent nucleation, interphase mass transfer, bubble growth/dissolution, rise/escape, and optional inertial-pressure feedback.

**Status:** broad implementation with focused conservation, transport, and MPI
tests. End-to-end reproduction of the paper results, extrapolation warnings,
source-Jacobian hooks, an allocation-free Kokkos-compatible local update, and
a checked-in combined user example remain open. Permanent serial and
exact-two-rank combined RANS-plus-bubble regression coverage is implemented.

#### Model selection and scope

- [x] Add a runtime selector:

  ```yaml
  radiolyticBubbleModel: idealGasSource | sheng2024TwoPopulation
  ```

- [x] Keep `idealGasSource` as the default until the advanced model is verified.
- [x] Treat H₂ as the only radiolytic gas in the initial short-pulse implementation.
- [x] Supply temperature, absolute liquid pressure, liquid velocity, and fission power density from thermal-hydraulic fields.
- [x] Do not treat the incompressible pressure-correction field directly as absolute thermodynamic pressure.
- [x] Add an explicit absolute-pressure reconstruction/interface.
- [x] Support:
  - [x] prescribed constant pressure
  - [x] prescribed pressure history
  - [x] reconstructed local absolute pressure
  - [x] experimental inertial-pressure coupling

#### State fields

Add cell fields with documented SI units:

- [x] `C_H2` — dissolved hydrogen concentration, mol/m³ liquid
- [x] `N_micro` — microbubble number density, 1/m³ bulk volume
- [x] `M_micro` — H₂ moles in microbubbles, mol/m³ bulk volume
- [x] `N_large` — large-bubble number density, 1/m³ bulk volume
- [x] `M_large` — H₂ moles in large bubbles, mol/m³ bulk volume
- [x] `r_nucleation`, `r_micro`, `r_large` — m
- [x] `C_critical`, `C_equilibrium` — mol/m³
- [x] `K_L` — liquid-side mass-transfer coefficient, m/s
- [x] `alpha_g_micro`, `alpha_g_large`, and total `alpha_g`
- [x] diagnostic production, conversion, growth, dissolution, escape, and inventory fields

#### Henry-law equilibrium and critical concentration

Use:

$$
C_{eq}(p_l,r_b)=H_{H_2}\left(p_l+\frac{2\sigma}{r_b}\right)
$$

$$
C_{crit}(p_l,r_0)=H_{H_2}\left(p_l+\frac{2\sigma}{r_0}\right).
$$

- [x] Add configurable/property-model inputs for `H_H2`, surface tension, and absolute pressure.
- [x] Document the selected Henry-law convention and units explicitly.
- [x] Guard against very small radius and non-positive pressure.
- [x] Verify pressure and Laplace-pressure monotonicity.

#### Pressure-corrected nucleation radius

Implement Eqs. (11)–(15) of the article in a dedicated property class.

- [x] Compute the atmospheric-pressure nucleation radius from temperature, uranyl-nitrate concentration, mean LET, and H₂ yield.
- [x] Apply:

$$
f^P_{corr}
=5.165\times10^{-5}\left(\frac{p_l}{p_{atm}}\right)^4
-1.732\times10^{-3}\left(\frac{p_l}{p_{atm}}\right)^3
+0.02245\left(\frac{p_l}{p_{atm}}\right)^2
-0.1554\left(\frac{p_l}{p_{atm}}\right)
+1.134
$$

$$
r_0=f^P_{corr}r_0^*.
$$

- [x] Require explicit concentration and temperature units at the API boundary.
- [ ] Add validity-range checks and warnings for extrapolation.
- [x] Add tabulated regression tests evaluated directly from the published equations.

#### Dissolved-hydrogen equation

Implement the paper-faithful short-pulse baseline:

$$
\alpha_l\frac{\partial C}{\partial t}
=Q_{M1}+Q_{M2}+\nabla\cdot(\alpha_lD\nabla C).
$$

Add an optional CFD extension:

$$
\frac{\partial(\alpha_lC)}{\partial t}
+\nabla\cdot(\alpha_lU_lC)
=\nabla\cdot(\alpha_lD\nabla C)+Q_{M1}+Q_{M2}.
$$

- [x] Add configurable or temperature-dependent H₂ diffusivity.
- [x] Retain no-advection mode for short-pulse reproduction.
- [x] Use existing scalar-transport infrastructure for advective mode.
- [x] Conserve dissolved-plus-bubble H₂ up to fission production and boundary escape.

#### Bubble-population balances

For `i=1` (microbubbles) and `i=2` (large bubbles), implement:

$$
\frac{\partial N_i}{\partial t}
=P_{Ni}+S_{Ni}-Q_{Ni}-\nabla\cdot(U_iN_i)
$$

$$
\frac{\partial M_i}{\partial t}
=P_{Mi}+S_{Mi}-Q_{Mi}-\nabla\cdot(U_iM_i).
$$

The article uses axial transport; use the general finite-volume divergence form while retaining an axial compatibility mode.

##### Fission-fragment production

- [x] Implement:

  \[
  P_{M1}=G\dot q_f,
  \qquad
  P_{N1}=\frac{P_{M1}}{\zeta_0},
  \qquad
  P_{M2}=P_{N2}=0.
  \]

- [x] Convert H₂ yield to mol/J consistently.
- [x] Compute nucleation-bubble molar content `zeta_0` from the bubble equation of state.

##### Microbubble dissolution

- [x] Implement:

  \[
  Q_{N1}=\frac{N_1}{\tau_1},
  \qquad
  Q_{M1}=\frac{M_1}{\tau_1}.
  \]

- [x] Make `tauMicro` configurable; document approximately 10 μs as the article default rather than a universal constant.

##### Pressure-dependent micro-to-large conversion

For `C>C_critical`, implement:

$$
S_{N1}
=-FN_1p_l\left(\frac{C}{C_{crit}}-1\right)
\Theta(C-C_{crit})
$$

$$
S_{M1}=\zeta_0S_{N1}.
$$

- [x] Enforce category conservation:
  - [x] `S_N2 = -S_N1`
  - [x] `S_M2 = -S_M1`
- [x] Make `F` configurable and mark it as an empirical calibration parameter with units 1/(Pa·s).
- [x] Support exact and optionally smoothed Heaviside functions.

##### Large-bubble growth and dissolution

Implement the article's sign convention:

$$
Q_{N2}=\frac{N_2}{\tau_2}\Theta(C_{eq}-C)
$$

$$
Q_{M2}
=-K_LA_2(C-C_{eq})\Theta(C-C_{eq})
+\frac{M_2}{\tau_2}\Theta(C_{eq}-C)
$$

$$
A_2=4\pi N_2r_2^2.
$$

- [x] Make `tauLarge` configurable; document approximately 50 μs as the article default.
- [x] Verify that supersaturation grows `M_large` without creating bubble number.
- [x] Verify that undersaturation reduces both bubble moles and, through dissolution, bubble number.

#### Interfacial mass transfer

Implement the Hughmark correlation used in the article:

$$
K_L=\frac{ShD}{2r_2},
\qquad
Sc=\frac{\mu_l}{\rho_lD},
\qquad
Re=\frac{2\rho_l|U_l-U_g|r_2}{\mu_l}.
$$

$$
Sh=
\begin{cases}
2+0.6Re^{1/2}Sc^{1/3}, & 0\le Re<776.06,\\
2+0.27Re^{0.63}Sc^{1/3}, & Re\ge776.06.
\end{cases}
$$

- [x] Check the published validity range, especially for `Sc`.
- [x] Evaluate liquid properties locally.
- [x] Depend on a bubble-velocity interface rather than a hard-coded rise law.

#### Bubble equation of state and radius solve

For each category solve:

$$
\frac{4}{3}\pi r_i^3
\left(p_l+\frac{2\sigma}{r_i}\right)
=\zeta_iR_gT,
\qquad
\zeta_i=\frac{M_i}{N_i}.
$$

- [x] Implement a robust positive-root solver using bracketed Newton or bisection.
- [x] Define empty-population behavior without dividing by tiny `N_i`.
- [x] Add minimum and maximum radius guards.
- [x] Verify residuals and monotonic trends with temperature, pressure, bubble moles, and bubble count.

#### Void fraction and characteristic radius

Compute:

$$
\alpha_g
=\frac{4\pi}{3}\left(N_1r_1^3+N_2r_2^3\right),
\qquad
\alpha_l=1-\alpha_g
$$

and:

$$
r_c
=\frac{r_1^3N_1+r_2^3N_2}
       {r_1^2N_1+r_2^2N_2}.
$$

- [x] Report microbubble and large-bubble void separately.
- [x] Enforce void bounds without silently deleting H₂ inventory.
- [x] Record clipped inventory if a safety bound is reached.
- [x] Integrate total void volume:

  \[
  \Delta V_{void}=\int_V\alpha_g\,dV.
  \]

#### Bubble transport and free-surface escape

- [x] Add `BubbleRiseVelocityModel`.
- [x] Provide:
  - [x] zero-slip mode
  - [x] constant upward-slip mode
  - [x] article-compatible radius correlation after verifying the cited Celata relation and units
  - [ ] future Euler–Euler gas-velocity mode
- [x] Advect `N_i` and `M_i` with the same category velocity.
- [x] Add a conservative free-surface escape boundary condition.
- [x] Track escaped H₂ moles and bubble count.
- [x] Evaluate implicit-outflow escape from the accepted transported state and
  localize escape-rate fields to boundary-adjacent owner cells.

#### Optional inertial-pressure coupling

Implement this behind an experimental switch and do not destabilize the incompressible pressure solver.

- [x] Add runtime switch for inertial-pressure coupling:

  ```text
  radiolytic_pressure_mode = constant | prescribedHistory | reconstructed | inertial
  ```

- [x] Implement:

$$
\kappa
=\kappa_l(1-\alpha_g)
+\frac{\alpha_g}{p_l+4\sigma/(3r_c)}
$$

$$
\beta
=\beta_l(1-\alpha_g)
+\frac{\alpha_g}{T}
 \frac{p_l+2\sigma/r_c}
      {p_l+4\sigma/(3r_c)}
$$

$$
\frac{dp_l}{dt}
=\frac{\beta}{\kappa}\frac{dT}{dt}
+\frac{1}{\rho_l\kappa}\frac{d\rho_l}{dt}.
$$

- [x] Add a separately named absolute-pressure evolution field.
- [x] Couple density change to velocity divergence and bubble compression consistently with the project's continuity treatment.
- [x] Do not replace the pressure-projection equation without a design review.
- [x] Initially use a local or weakly coupled equation-of-state pressure update.
- [x] Support constant-pressure and inertial-pressure modes for comparison.

#### Numerical integration and stiffness

The microsecond dissolution time scales can be far shorter than the CFD time step.

- [x] Use a positivity-preserving local kinetics update:
  - [ ] implicit Euler
  - [x] analytic linear decays plus bounded local transfer
  - [x] lifetime-limited local subcycling with `max_radiolytic_subcycles`
- [x] Use operator splitting between transport and local bubble kinetics.
- [ ] Add source-Jacobian hooks for future monolithic coupling.
- [x] Preserve non-negativity of `C_H2`, `N_i`, `M_i`, and radii.
- [ ] Keep the local update Kokkos-compatible and allocation-free per cell.
- [x] Report substeps, clipping, radius-solver failure, and inventory error.

#### Configuration

- [x] `hydrogenYieldMolPerJ`
- [x] `henryCoefficient` and convention
- [x] `surfaceTensionModel`
- [x] `uraniumConcentration`
- [ ] `nitricAcidConcentration`, where required
- [x] `atmosphericPressure`
- [x] `microbubbleLifetime`
- [x] `largeBubbleDissolutionTime`
- [x] `microToLargeConversionCoefficient`
- [x] `hydrogenDiffusivityModel`
- [x] `bubbleRiseVelocityModel`
- [x] initial `C_H2`, `N_micro`, `M_micro`, `N_large`, and `M_large`
- [x] radius/population/concentration floors and ceilings
- [x] local ODE tolerance and maximum subcycles

#### Verification tests

- [x] Henry-law pressure monotonicity.
- [x] Laplace-pressure/radius monotonicity.
- [x] Nucleation pressure-correction regression.
- [x] Bubble-radius equation residual.
- [x] Microbubble exponential dissolution.
- [x] Micro-to-large conversion conservation.
- [x] Large-bubble supersaturated growth.
- [x] Large-bubble undersaturated dissolution.
- [x] H₂ inventory conservation without production or escape.
- [x] H₂ balance with known fission production.
- [x] Population-transport conservation.
- [x] Void reconstruction from prescribed populations and radii.
- [x] Characteristic-radius calculation.
- [x] Constant-pressure versus prescribed-pressure-pulse regression.
- [x] Stiff-source convergence versus substep size.
- [x] Serial/MPI consistency of global H₂ and void inventories.
- [x] Finite-Courant serial/MPI escape balances and rate-field integrals.
- [x] Globally reduce advanced-model event counters across MPI ranks.
- [x] Add a permanent serial integration test with nonzero liquid shear, an
      active RANS closure, advected dissolved/bubble inventories, bubble slip,
      fission production, and void-dependent density feedback; assert causal
      turbulence, dissolved-advection, and density-feedback changes.
- [x] Add an exact-two-rank MPI counterpart with the same conserved-inventory,
      bounded-field, and causal acceptance checks.

#### Phase 14.1 acceptance criteria

- [x] The advanced model is runtime-selectable and does not change Phase 12 results when disabled.
- [x] All state variables remain finite, non-negative, and bounded.
- [x] H₂ inventory is conserved after accounting for fission production and boundary escape in focused tests.
- [x] Micro-to-large conversion conserves bubble number and gas moles according to the model equations.
- [x] Bubble-radius solves converge over the documented operating range.
- [x] Increasing pressure raises `C_critical` and delays large-bubble conversion.
- [x] Supersaturation grows large bubbles and increases void fraction.
- [x] Constant-pressure and inertial-pressure options run from the same case definition.
- [ ] Component behavior reproduces the equations and qualitative pressure trends reported by Sheng et al.
- [ ] Validate full SILENE power/pressure behavior after point-kinetics or
      neutronics coupling becomes available; this remains deferred.

#### Reference and known limitations

Primary reference:

- H. Sheng, J. Gou, B. Zhang, J. Shan, and G. Liu, “Effect of inertial pressure on criticality excursion and radiolytic gas bubbles for fuel solution system,” *Annals of Nuclear Energy*, 206, 110668, 2024.

Known limitations:

- [ ] H₂-only chemistry is intended for short pulses; O₂ and N₂ chemistry are deferred.
- [ ] Two representative radii are used instead of a full size distribution.
- [ ] The empirical conversion coefficient requires calibration and uncertainty analysis.
- [ ] Bubble rise velocity strongly affects late-time local void fraction.
- [ ] Axial/prescribed bubble motion is not a replacement for a gas momentum equation.
- [ ] A later Euler–Euler phase should solve gas velocity and use validated drag closures.
- [ ] Current RANS transport is not weighted by liquid volume fraction and has
      no bubble-induced-turbulence or interphase turbulence-transfer closure.
- [ ] No checked-in user-facing example currently exercises RANS and the
      advanced bubble model together in a nontrivial flow; permanent serial
      and exact-two-rank integration regressions do cover the combined path.

---

### Phase 15 — Temperature- and void-dependent feedback properties

Add property feedback fields for neutronics-oriented thermal-hydraulics.

**Status:** partial. Density feedback and constant-viscosity feedback with
safety floors are implemented. Temperature/void viscosity extensions and a
natural-convection stability acceptance case remain open.

#### Density feedback

Support a conservative default:

$$
\rho_{mix}(T,\alpha_g)
= (1 - \alpha_g)\rho_l(T) + \alpha_g\rho_g
$$

and a Boussinesq-compatible form:

$$
\rho_{fb}(T,\alpha_g)
= \rho_{ref}\left[1 - \beta(T - T_{ref})\right](1-\alpha_g)
+ \rho_g\alpha_g
$$

#### Viscosity feedback

Start with:

$$
\mu_{eff}(T,\alpha_g) = \mu_l(T) f_\mu(\alpha_g)
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

### Phase 16 — Fissile-solution tank demo case

Add a small demonstration case that exercises the new source models without requiring a neutronics solver.

**Status:** partial. The examples build and have short-run coverage, but
ParaView readability and the disabled-radiolysis/boiling baseline are not yet
acceptance-tested.

- [x] Add example executable:
  - [ ] `radiolytic_bubble_cylinder`, or
  - [x] `fissile_solution_tank_demo`
  - [x] `constant_power_cylinder_vessel`
- [x] Use a cylindrical tank mesh.
- [x] Initialize quiescent liquid.
- [x] Use configurable central or axial fission power deposition.
- [x] Enable radiolytic gas generation.
- [x] Keep boiling disabled by default, with a documented switch to enable it.
- [x] Add a constant-power cylinder variant with active radiolytic gas and
  boiling coupling.
- [x] Output at least:
  - [x] `T`
  - [x] `U`
  - [x] `p`
  - [x] `qdot_fission`
  - [x] `alpha_g`
  - [x] `S_alpha_rad`
  - [x] `S_alpha_boil`
  - [x] `S_alpha_total`
  - [x] `latentHeatSink`
  - [x] `rhoFeedback`
- [x] Add a short README section or example note explaining how to run and inspect in ParaView.

#### Phase 16 acceptance criteria

- [x] Example builds in Debug and Release.
- [x] Example runs for a few time steps in serial.
- [ ] Output files open in ParaView.
- [ ] Disabled radiolysis/boiling gives a standard natural-convection-like result.
- [x] Enabled radiolysis produces nonzero bounded `alpha_g` in the powered region.

---

### Phase 17 — Unit, regression, and conservation tests for radiolysis/boiling

Collect the new physics tests into a focused verification suite.

**Status:** implemented for the currently supported Phase 12–15 scope. This
does not close the unchecked model and validation work in those phases.

- [x] Radiolysis zero-source test.
- [x] Radiolysis linearity test with respect to power density.
- [x] Radiolysis linearity test with respect to gas yield.
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

#### Phase 17 acceptance criteria

- [x] All new tests are deterministic.
- [x] Tests are safe in serial and do not require MPI launch unless explicitly marked.
- [x] The Debug test suite, including registered MPI tests, passes in the
      supported local development environment.
- [x] Tolerances are documented and physically meaningful.

---

### Phase 18 — Documentation and configuration examples

Document the low-order scalar model and the two-population radiolytic bubble model clearly so future work does not confuse either path with a full Euler–Euler method.

**Status:** partial. The model overview and limitations are documented, but a
complete user-facing key/default/unit/validity reference is still missing;
those details remain distributed across implementation headers and model docs.

- [x] Add `docs/modeling/radiolytic_bubble_boiling.md`.
- [x] Document governing equations and principal field units.
- [ ] Document every public configuration key with its default, SI unit, and
      validity range in a user-facing reference.
- [x] Document limitations:
  - [x] no separate gas momentum equation
  - [x] no drag-law closure yet
  - [x] scalar `alpha_g` mode has no population balance
  - [x] Phase 14.1 has two representative populations, not a full size distribution
  - [x] no detailed radiolysis chemistry
  - [x] no full RPI wall-boiling partitioning
  - [x] no neutronics solver yet
- [x] Document intended neutronics feedback fields:
  - [x] `T`
  - [x] `alpha_g`
  - [x] `rhoFeedback`
  - [x] delayed-neutron `C_i` and `S_C_i` fields
- [x] Add a short section to `README.md` linking the new model document and demo.
- [x] Document the weakly coupled RANS-plus-bubble scope and distinguish it
      from phase-aware turbulent Euler–Euler bubbly flow.

#### Phase 18 acceptance criteria

- [ ] A new developer can configure every supported model from documentation
      alone without consulting implementation headers.
- [ ] All dimensional configuration parameters include units and defaults in
      user-facing documentation.
- [x] Future full Euler–Euler work is clearly separated from the current scalar-void model.

---

### Phase 19 — Delayed-neutron precursor scalar transport

After velocity, temperature, and void feedback fields are available, add precursor transport on the liquid phase.

**Status:** implemented and focused-tested for the currently supported
transport scope. Conserved liquid-fraction-weighted inventories use the
projected liquid face flux for advection, optional diffusion, exact source and
decay integration, globally reduced balance diagnostics, and collective input
validation.

#### Target equation

For precursor group `i`:

$$
\frac{\partial (\alpha_l C_i)}{\partial t}
+ \nabla\cdot(\alpha_l U_l C_i)
=
\nabla\cdot(\alpha_l D_i^{eff}\nabla C_i)
+ S_i
- \lambda_i \alpha_l C_i
$$

#### Tasks

- [x] Add configurable number of delayed-neutron precursor groups.
- [x] Add per-group fields `C_1 ... C_N`.
- [x] Add per-group decay constants `lambda_i`.
- [x] Add per-group source fields from fission power or imported neutronics data.
- [x] Advect precursors with the projected liquid face flux, not a gas or
      mixture-transport surrogate.
- [x] Include `alpha_l = 1 - alpha_g` weighting.
- [x] Add optional effective diffusivity.
- [x] Write precursor fields to VTU.
- [x] Add globally reduced source, decay, boundary-outflow, transport
      positivity-adjustment, before/after inventory, and balance diagnostics.
- [x] Validate timestep, options, liquid fraction, optional flux/power fields,
      and rank-consistent selections collectively.

#### Phase 19 acceptance criteria

- [x] Zero source and zero initial concentration remains zero.
- [x] Pure decay one-cell test matches analytic exponential decay.
- [x] Constant source plus decay one-cell test reaches correct asymptotic value.
- [x] Advective transport conserves precursor inventory up to decay, boundary
      flux, source, and recorded positivity adjustment.
- [x] Precursor fields remain finite and non-negative under bounded test cases.

---

### Phase 20 — TH/neutronics feedback-map interface

Add data structures for mapping CFD feedback fields to a coarser neutronics mesh or external neutronics code.

**Status:** the in-memory scaffold is implemented and focused-tested. It
provides volume averaging, power import, a standard feedback-field registry,
deterministic mapped snapshots, and a callback-driven placeholder outer loop
with thermal-hydraulic subcycles and power exchange. This is not a production
external-neutronics interface or a neutronics solver.

- [x] Add feedback field registry:
  - [x] `T_liquid`
  - [x] `alpha_g`
  - [x] `rhoFeedback`
  - [x] `C_i`
- [x] Add volume-averaging utilities from CFD cells to feedback cells.
- [x] Add conservative mapping tests for scalar fields.
- [x] Add import path for externally supplied fission power density.
- [x] Add deterministic in-memory snapshot export for mapped
      thermal-hydraulic feedback.
- [x] Add a simple outer-coupling driver stub:
  - [x] receive or compute `qdot_fission`
  - [x] advance TH by one time step or configured subcycles
  - [x] map feedback fields
  - [x] call placeholder neutronics update
  - [x] repeat and import the returned power
- [x] Keep the interface file-based or in-memory depending on the existing code style; do not force an external dependency.

#### Phase 20 acceptance criteria

- [x] Constant CFD field maps to the same constant feedback field.
- [x] Volume-weighted integral is preserved for mapped scalar fields.
- [x] File or in-memory coupling path is documented.
- [x] Coupling driver can run with a placeholder neutronics callback.
- [ ] Define, implement, and validate a production external-neutronics
      transport/protocol when a target solver is selected.

---

## Deferred architecture — Phase 21

### Phase 21 — Full Euler–Euler two-fluid extension

This phase is explicitly deferred until the scalar void-fraction source model and the two-population radiolytic bubble model are verified.

**Status:** deferred by design; none of its implementation or validation
criteria are claimed complete.

The ability to configure RANS and radiolytic bubbles in the same
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
- [ ] Validate against a bubbly-plume or bubble-column benchmark before applying to criticality accident analysis.

#### Phase 21 acceptance criteria

- [ ] Scalar `alpha_g` model remains available as a lower-order option.
- [ ] Two-fluid equations conserve phase volume/mass under controlled tests.
- [ ] Drag-law unit tests reproduce expected limiting behavior.
- [ ] Phase-aware turbulence equations recover the current RANS solutions as
      `alpha_g` approaches zero.
- [ ] Bubble-induced-turbulence and interphase-transfer closures reproduce
      selected reference limits.
- [ ] Full model is validated against at least one non-neutronic bubbly-flow benchmark.

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
