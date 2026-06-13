# TODO: SimpleFluid Multiphysics Roadmap

**Status date:** June 12, 2026  
**Near-term goal:** add conservative radiolytic-gas, boiling, void-fraction, and thermal-feedback infrastructure for fissile-solution-tank criticality-accident simulations, without prematurely implementing a full Euler–Euler two-fluid solver.

## Completed foundation — squashed Phases -1 to 8

The following capabilities are considered complete and should not be re-planned as active roadmap phases unless regressions are found.

- [x] Mesh and geometry infrastructure
  - [x] CRTP mesh hierarchy
  - [x] Cartesian 3D mesh
  - [x] Orthogonal cylindrical 3D mesh
  - [x] Semi-structured XY×Z prism mesh
  - [x] STK/Exodus adapter for arbitrary polyhedral meshes
  - [x] Runtime type-erased `MeshHandle`
  - [x] Owned/ghost cell decomposition
  - [x] Zoltan2/ParMETIS partitioning
- [x] Field infrastructure
  - [x] Cell-centered scalar and vector fields
  - [x] Face-centered scalar and vector fields
  - [x] Boundary-face fields
  - [x] Tpetra-backed distributed storage
  - [x] Ghost-value synchronization
  - [x] Mesh-aware `FieldStored` abstraction
- [x] Finite-volume operators
  - [x] Least-squares gradient reconstruction
  - [x] Orthogonal scalar/vector diffusion
  - [x] Explicit non-orthogonal correction
  - [x] Fully implicit non-orthogonal diffusion
  - [x] Hybrid implicit/explicit non-orthogonal treatment
  - [x] Scalar/vector transport systems
  - [x] First-order upwind convection
  - [x] Backward Euler and Crank–Nicolson time integration
- [x] Momentum and pressure-velocity coupling
  - [x] Momentum equation assembly
  - [x] SIMPLE
  - [x] PISO
  - [x] PIMPLE
  - [x] Fully coupled block-Krylov pressure-velocity solver
  - [x] Rhie–Chow collocated-grid stabilization
  - [x] Residual reporting for momentum, pressure, and continuity
- [x] Physical solver baseline
  - [x] Transient incompressible Navier–Stokes
  - [x] Boussinesq buoyancy
  - [x] Temperature transport
  - [x] Natural-convection driver
- [x] I/O and utilities
  - [x] VTU output for ParaView
  - [x] STK Exodus II mesh input
  - [x] Typed key-value `Database`
  - [x] `RandomAccessView`
  - [x] `vec3`
  - [x] Concept-based Tpetra type configuration
- [x] Verification suite
  - [x] Lid-driven cavity, Re = 100 and 1000
  - [x] Poiseuille flow
  - [x] Manufactured incompressible Navier–Stokes solutions
  - [x] Skewed-mesh diffusion
  - [x] Natural-convection square cavity
  - [x] OpenFOAM comparison case
  - [x] Gradient/divergence reconstruction tests
  - [x] Pressure projection tests
  - [x] Rhie–Chow checkerboard-suppression tests

---

## Phase 9 — Performance and solver-regression benchmarks

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
  - [x] implicit non-orthogonal matrix + GMRES
  - [x] hybrid non-orthogonal treatment
  - [x] coupled Krylov + Schur preconditioner
- [x] Add MPI scaling benchmarks.
- [x] Ensure benchmark output is machine-readable, for example CSV or JSON.

### Phase 9 acceptance criteria

- [x] Benchmarks can be run from CTest or a documented command.
- [x] At least one small benchmark is safe for local Debug builds.
- [x] At least one larger benchmark is suitable for Release/RelWithDebInfo profiling.
- [x] Results include enough metadata to compare solver configuration, mesh size, MPI size, and compiler mode.

---

## Phase 10 — Source-term and material-property infrastructure

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

### Phase 10 acceptance criteria

- [x] Existing natural-convection examples reproduce their old behavior when all new model switches are off.
- [x] A constant heat source produces the expected one-cell analytic temperature change when diffusion/advection are disabled.
- [x] Invalid source/property parameters fail with clear error messages.
- [x] Source fields can be written to VTU.

---

## Phase 11 — Fission power density and heat-source coupling

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

### Phase 11 acceptance criteria

- [x] Integrated power equals configured total power within tolerance.
- [x] Zero power gives zero thermal source.
- [x] Constant power in a closed one-cell thermal problem gives the expected temperature rise.
- [x] Gaussian profile is centered and normalized correctly on Cartesian and cylindrical meshes.

---

## Phase 12 — Radiolytic gas generation model

Implement the first conservative unresolved gas-generation model. This phase produces an `alpha_g` source but does not solve a gas momentum equation.

### Model variables

- `qdot_fission` — fission power density, W/m³
- `T` — liquid temperature, K
- `p_ref` or local pressure if available, Pa
- `alpha_g` — gas void fraction
- `alpha_l = 1 - alpha_g` — liquid volume fraction

### Baseline model

$$
\dot n_g = \eta_g\,G_g\,\dot q_f
$$

$$
\dot V_g = \dot n_g \frac{\mathrm R T}{p}
$$

$$
S_{\alpha,rad} = \alpha_l \dot V_g
$$

where:

- $G_g$ is the configurable gas yield in mol/J
- $eta_g$ is a release efficiency
- $\mathrm R$ is the ideal-gas constant
- $S_{\alpha,rad}$ has units 1/s

### Tasks

- [ ] Add `GasGenerationProperties` or equivalent configuration struct.
- [ ] Add `RadiolyticGasModel` with small testable pure functions.
- [ ] Add configurable parameters:
  - [ ] `enableRadiolysis`
  - [ ] `radiolysisGasYieldMolPerJ`
  - [ ] `gasReleaseEfficiency`
  - [ ] `referencePressure`
  - [ ] `gasConstant`
  - [ ] `alphaMin`
  - [ ] `alphaMax`
  - [ ] `maxSourceAlphaRate`
- [ ] Compute `S_alpha_rad` as a cell field.
- [ ] Clamp or limit source terms so `alpha_g` remains bounded.
- [ ] Write `S_alpha_rad` to VTU.

### Phase 12 acceptance criteria

- [ ] `qdot_fission = 0` gives `S_alpha_rad = 0`.
- [ ] Doubling `qdot_fission` doubles `S_alpha_rad`.
- [ ] Doubling `radiolysisGasYieldMolPerJ` doubles `S_alpha_rad`.
- [ ] Higher temperature increases ideal-gas volume source.
- [ ] Higher pressure decreases ideal-gas volume source.
- [ ] Source is non-negative and bounded by `maxSourceAlphaRate`.
- [ ] Source vanishes or is limited as `alpha_g` approaches `alphaMax`.

---

## Phase 13 — Bulk and wall boiling source model

Add a simple energy-consistent boiling model. This is a placeholder for a later RPI-like wall heat-flux partitioning model.

### Bulk superheat model

For cells with:

$$
T > T_{sat} + \Delta T_{act}
$$

compute an available energy rate:

$$
\dot e_{avail} = \rho_l c_{p,l}\frac{\max(T - T_{sat},0)}{\tau_b}
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

### Wall boiling placeholder

For heated boundary faces:

$$
\dot m''_w = f_{evap}\frac{q''_w}{h_{fg}}
$$

Distribute to owner cells:

$$
\dot m_{w,vol} = \dot m''_w \frac{A_f}{V_P}
$$

### Tasks

- [ ] Add `BoilingSourceModel` or equivalent.
- [ ] Add configurable parameters:
  - [ ] `enableBulkBoiling`
  - [ ] `enableWallBoiling`
  - [ ] `saturationTemperature`
  - [ ] `boilingActivationDeltaT`
  - [ ] `boilingTimeScale`
  - [ ] `latentHeat`
  - [ ] `gasDensity`
  - [ ] `wallEvaporationFraction`
- [ ] Compute `S_alpha_boil` as a cell field.
- [ ] Compute `latentHeatSink` as a cell field in W/m³.
- [ ] Subtract `latentHeatSink` from the temperature equation.
- [ ] Write `S_alpha_boil` and `latentHeatSink` to VTU.
- [ ] Keep the implementation explicit and bounded.
- [ ] Leave a clean interface for future wall heat-flux partitioning:
  - [ ] convective heat transfer
  - [ ] quenching heat transfer
  - [ ] evaporative heat transfer

### Phase 13 acceptance criteria

- [ ] Below threshold, boiling source is zero.
- [ ] Above threshold, boiling source is positive.
- [ ] `latentHeatSink == mDot * latentHeat` within tolerance.
- [ ] Zero latent heat or invalid density fails with clear error.
- [ ] Wall boiling distributes face source to owner cells conservatively.
- [ ] Existing cases are unchanged when boiling is disabled.

---

## Phase 14 — Void-fraction field update and optional transport

Introduce `alpha_g` as the first unresolved gas-state field.

### Minimum required update

$$
\alpha_g^{n+1} = \operatorname{clamp}\left(\alpha_g^n + \Delta t\,S_{\alpha,total},\alpha_{min},\alpha_{max}\right)
$$

where:

$$
S_{\alpha,total} = S_{\alpha,rad} + S_{\alpha,boil} + S_{\alpha,wall} - S_{\alpha,collapse}
$$

### Optional transport upgrade

If the existing scalar transport API allows it cleanly, add:

$$
\frac{\partial \alpha_g}{\partial t}
+ \nabla\cdot(U_l \alpha_g)
+ \nabla\cdot(U_{slip}\alpha_g)
=
\nabla\cdot(D_\alpha\nabla\alpha_g)
+ S_{\alpha,total}
$$

### Tasks

- [ ] Allocate and initialize `alpha_g`.
- [ ] Allocate and initialize `S_alpha_total`.
- [ ] Add bounded explicit update.
- [ ] Add optional scalar-transport path if low-risk.
- [ ] Add configurable upward slip velocity aligned opposite gravity.
- [ ] Add configurable `alphaDiffusivity`.
- [ ] Add optional simple collapse/removal time scale for gas disengagement tests.
- [ ] Ensure ghost synchronization before neighbor-based operations.
- [ ] Write `alpha_g` and `S_alpha_total` to VTU.

### Phase 14 acceptance criteria

- [ ] One-cell constant-source update matches the analytic result.
- [ ] `alpha_g` never leaves `[alphaMin, alphaMax]`.
- [ ] With all sources disabled, `alpha_g` remains unchanged.
- [ ] Optional transport conserves total gas inventory up to boundary flux and sources.
- [ ] VTU output includes `alpha_g` and source fields.

---

## Phase 15 — Temperature- and void-dependent feedback properties

Add property feedback fields for neutronics-oriented thermal-hydraulics.

### Density feedback

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

### Viscosity feedback

Start with:

$$
\mu_{eff}(T,\alpha_g) = \mu_l(T) f_\mu(\alpha_g)
$$

where $f_\mu(\alpha_g) = 1$ by default until a validated correction is added.

### Tasks

- [ ] Add `MaterialFeedbackModel` or equivalent.
- [ ] Add configurable density model:
  - [ ] constant
  - [ ] Boussinesq temperature-only
  - [ ] Boussinesq + void fraction
  - [ ] mixture density
- [ ] Add configurable viscosity model:
  - [ ] constant
  - [ ] temperature-dependent if existing support is available
  - [ ] optional void correction
- [ ] Add safety floors:
  - [ ] `minDensity`
  - [ ] `minViscosity`
- [ ] Couple density feedback into buoyancy/body-force calculation.
- [ ] Couple viscosity feedback into momentum diffusion only if existing equation design allows it cleanly.
- [ ] Write `rhoFeedback` and `muFeedback` to VTU.

### Phase 15 acceptance criteria

- [ ] Increasing `alpha_g` lowers density for liquid-like systems.
- [ ] Increasing `T` lowers density under Boussinesq settings.
- [ ] Density and viscosity never become negative.
- [ ] Existing Boussinesq cases remain unchanged when feedback models are disabled.
- [ ] Natural-convection benchmark remains stable with temperature-dependent density enabled.

---

## Phase 16 — Fissile-solution tank demo case

Add a small demonstration case that exercises the new source models without requiring a neutronics solver.

- [ ] Add example executable:
  - [ ] `radiolytic_bubble_cylinder`, or
  - [ ] `fissile_solution_tank_demo`
- [ ] Use a cylindrical tank mesh.
- [ ] Initialize quiescent liquid.
- [ ] Use configurable central or axial fission power deposition.
- [ ] Enable radiolytic gas generation.
- [ ] Keep boiling disabled by default, with a documented switch to enable it.
- [ ] Output at least:
  - [ ] `T`
  - [ ] `U`
  - [ ] `p`
  - [ ] `qdot_fission`
  - [ ] `alpha_g`
  - [ ] `S_alpha_rad`
  - [ ] `S_alpha_boil`
  - [ ] `S_alpha_total`
  - [ ] `latentHeatSink`
  - [ ] `rhoFeedback`
- [ ] Add a short README section or example note explaining how to run and inspect in ParaView.

### Phase 16 acceptance criteria

- [ ] Example builds in Debug and Release.
- [ ] Example runs for a few time steps in serial.
- [ ] Output files open in ParaView.
- [ ] Disabled radiolysis/boiling gives a standard natural-convection-like result.
- [ ] Enabled radiolysis produces nonzero bounded `alpha_g` in the powered region.

---

## Phase 17 — Unit, regression, and conservation tests for radiolysis/boiling

Collect the new physics tests into a focused verification suite.

- [ ] Radiolysis zero-source test.
- [ ] Radiolysis linearity test with respect to power density.
- [ ] Radiolysis linearity test with respect to gas yield.
- [ ] Ideal-gas temperature scaling test.
- [ ] Ideal-gas pressure scaling test.
- [ ] Alpha boundedness test.
- [ ] Boiling threshold test.
- [ ] Boiling latent-heat conservation test.
- [ ] Wall-boiling owner-cell distribution test.
- [ ] Density feedback monotonicity test.
- [ ] Viscosity floor test.
- [ ] One-cell alpha update analytic test.
- [ ] `Database` parsing/defaults test.
- [ ] Regression test that all models disabled leaves old solver behavior unchanged.

### Phase 17 acceptance criteria

- [ ] All new tests are deterministic.
- [ ] Tests are safe in serial and do not require MPI launch unless explicitly marked.
- [ ] Debug test suite passes except for known environment-blocked MPI launches.
- [ ] Tolerances are documented and physically meaningful.

---

## Phase 18 — Documentation and configuration examples

Document the simplified model clearly so future work does not confuse it with a full Euler–Euler method.

- [ ] Add `docs/modeling/radiolytic_bubble_boiling.md`.
- [ ] Document model equations and units.
- [ ] Document configuration keys and defaults.
- [ ] Document limitations:
  - [ ] no separate gas momentum equation
  - [ ] no drag-law closure yet
  - [ ] no population balance
  - [ ] no detailed radiolysis chemistry
  - [ ] no full RPI wall-boiling partitioning
  - [ ] no neutronics solver yet
- [ ] Document intended neutronics feedback fields:
  - [ ] `T`
  - [ ] `alpha_g`
  - [ ] `rhoFeedback`
  - [ ] future delayed-neutron precursor fields
- [ ] Add a short section to `README.md` linking the new model document and demo.

### Phase 18 acceptance criteria

- [ ] A new developer can enable/disable the model from documentation alone.
- [ ] All dimensional parameters include units.
- [ ] Future full Euler–Euler work is clearly separated from the current scalar-void model.

---

## Phase 19 — Delayed-neutron precursor scalar transport

After velocity, temperature, and void feedback fields are available, add precursor transport on the liquid phase.

### Target equation

For precursor group `i`:

$$
\frac{\partial (\alpha_l C_i)}{\partial t}
+ \nabla\cdot(\alpha_l U_l C_i)
=
\nabla\cdot(\alpha_l D_i^{eff}\nabla C_i)
+ S_i
- \lambda_i \alpha_l C_i
$$

### Tasks

- [ ] Add configurable number of delayed-neutron precursor groups.
- [ ] Add per-group fields `C_1 ... C_N`.
- [ ] Add per-group decay constants `lambda_i`.
- [ ] Add per-group source fields from fission power or imported neutronics data.
- [ ] Advect precursors with liquid velocity, not mixture velocity.
- [ ] Include `alpha_l = 1 - alpha_g` weighting.
- [ ] Add optional effective diffusivity.
- [ ] Write precursor fields to VTU.
- [ ] Add group inventory diagnostics.

### Phase 19 acceptance criteria

- [ ] Zero source and zero initial concentration remains zero.
- [ ] Pure decay one-cell test matches analytic exponential decay.
- [ ] Constant source plus decay one-cell test reaches correct asymptotic value.
- [ ] Advective transport conserves precursor inventory up to decay, boundary flux, and source.
- [ ] Precursor fields remain finite and non-negative under bounded test cases.

---

## Phase 20 — TH/neutronics feedback-map interface

Add data structures for mapping CFD feedback fields to a coarser neutronics mesh or external neutronics code.

- [ ] Add feedback field registry:
  - [ ] `T_liquid`
  - [ ] `alpha_g`
  - [ ] `rhoFeedback`
  - [ ] `C_i`
- [ ] Add volume-averaging utilities from CFD cells to feedback cells.
- [ ] Add conservative mapping tests for scalar fields.
- [ ] Add import path for externally supplied fission power density.
- [ ] Add export path for mapped thermal-hydraulic feedback.
- [ ] Add a simple outer-coupling driver stub:
  - [ ] receive or compute `qdot_fission`
  - [ ] advance TH by one time step or subcycle
  - [ ] map feedback fields
  - [ ] call placeholder neutronics update
  - [ ] repeat
- [ ] Keep the interface file-based or in-memory depending on the existing code style; do not force an external dependency.

### Phase 20 acceptance criteria

- [ ] Constant CFD field maps to the same constant feedback field.
- [ ] Volume-weighted integral is preserved for mapped scalar fields.
- [ ] File or in-memory coupling path is documented.
- [ ] Coupling driver can run with a placeholder neutronics model.

---

## Phase 21 — Full Euler–Euler two-fluid extension, deferred

This phase is explicitly deferred until the scalar void-fraction source model is verified.

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
- [ ] Add optional population-balance model.
- [ ] Validate against a bubbly-plume or bubble-column benchmark before applying to criticality accident analysis.

### Phase 21 acceptance criteria

- [ ] Scalar `alpha_g` model remains available as a lower-order option.
- [ ] Two-fluid equations conserve phase volume/mass under controlled tests.
- [ ] Drag-law unit tests reproduce expected limiting behavior.
- [ ] Full model is validated against at least one non-neutronic bubbly-flow benchmark.

---

## Global acceptance criteria

- [ ] `cmake --preset Local` succeeds.
- [ ] `cmake --build --preset Debug` succeeds.
- [ ] `ctest --test-dir build -C Debug --output-on-failure` passes for non-MPI tests.
- [ ] Any MPI failures caused by sandbox/PMIx restrictions are reported separately from code failures.
- [ ] Existing verified cases remain stable when new physics is disabled.
- [ ] New physics fields are bounded and finite.
- [ ] New source terms are dimensionally documented.
- [ ] VTU output supports all new feedback fields.
- [ ] Solver configuration can be changed from `Database` without recompilation.
- [ ] Documentation clearly states which models are engineering placeholders and which are verified numerical capabilities.

## Suggested immediate Codex task order

1. Implement Phase 12 radiolytic gas source.
2. Implement Phase 14 explicit bounded `alpha_g` update.
3. Implement Phase 15 density feedback.
4. Add Phase 17 unit tests for the implemented subset.
5. Add Phase 16 demo case.
6. Implement Phase 13 boiling only after source/latent-heat plumbing is tested.
7. Document everything in Phase 18.
