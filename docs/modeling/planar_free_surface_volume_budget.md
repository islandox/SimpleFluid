# Planar Free-Surface Volume Budget

This note describes the fixed-grid planar volume-budget components introduced
for thermally and bubble-driven liquid-level evolution. The implemented scope
is the currently supported portion of Milestone A: vessel-volume maps, global
and cellwise liquid-mass inventories, conservative gas-compartment bookkeeping, vented
and closed ideal-gas headspaces, a
pressure/level closure, accepted-step `BoussinesqSolver` integration, and
opt-in diagnostic fields. It is a weak geometry-feedback model. It does not
move the CFD mesh, change the liquid-domain boundary, alter the existing
flow/temperature control volumes or continuity equation, or implement a
resolved interface. The opt-in cellwise liquid inventory adds its own
fixed-grid finite-volume transport equation as described below.

Full Milestone B planar ALE and the conservative part of Milestone C feedback
are not supported. A standalone B0 geometry-motion/GCL substrate is available,
but it is deliberately disconnected from the free-surface solver until
transport and pressure coupling are conservative. The remaining blockers are
recorded below so a reported level cannot be mistaken for moving-mesh physics.

## Applicability and State Ownership

The model is intended for a single planar pool surface whose vessel geometry
can be represented by a monotone volume map

$$
\mathcal V(h)=\text{vessel volume below elevation }h.
$$

Ownership is deliberately split along existing model boundaries:

- `LiquidMassInventory` owns total liquid mass, accepted evaporation and
  condensation totals, the optional cellwise `liquidMassInventory` field, and
  the derived pure-liquid volume. `globalConstantMass` retains fixed reference
  mass fractions; `cellMassInventory` owns the transported local state.
- `RadiolyticGasModel` remains authoritative for dissolved hydrogen,
  microbubble moles/count, large-bubble moles/count, bubble thermodynamics,
  and transport escape. Its raw EOS-derived bubble volume, rather than the
  bounded hydrodynamic void alone, is used for inventory closure.
- `BoilingSourceModel` owns accepted phase-change mass, rejected vapor demand,
  submerged steam mass/volume, scalar-void collapse assigned to steam, and
  condensate returned to the liquid inventory.
- `VentedHeadspaceModel` or `ClosedIdealGasHeadspaceModel` owns escaped gas
  after a single committed transfer, by species. `PlanarFreeSurfaceModel`
  retains the corresponding committed-transfer watermark.
- `PlanarFreeSurfaceModel` owns only accepted old/new levels, the planar volume
  closure, headspace closure state, and a reporting snapshot. It neither
  recomputes bubble kinetics nor material properties.

`BoussinesqSolver` owns the optional model and liquid inventory. Its accepted
post-temperature order is: advance the Sheng bubble/void state when selected;
advance precursors; refresh material properties and pure-liquid density;
preview accepted boiling mass and condensate without committing the liquid
ledger, including a conservative cellwise transport solve when selected; read
the radiolytic model's authoritative cumulative H2 ledgers; solve
the level/headspace closure; commit the liquid ledger and exact pending escape;
update the constant or reconstructed radiolytic absolute-pressure offset; then
publish diagnostic fields and append the accepted history record.

This is an accepted-step, one-way-lagged fixed-grid coupling. The momentum
solver retains its existing dynamic-pressure role. A pressure-sensitive bubble
update for the current step uses the previously accepted thermodynamic offset;
the newly closed pressure is the offset for the next step. Boiling contributes
its already accepted mass/latent-energy result and is not re-solved at the new
headspace pressure. There is no same-step outer pressure/level corrector, so
this path must not be described as fully coupled.

## Clear Liquid and Bubble-Displaced Pool Levels

The two reported volumes and levels have different meanings:

$$
V_l=\text{volume occupied by liquid material},
$$

$$
V_b=\text{volume of unresolved gas-phase bubbles still submerged},
$$

$$
V_{\mathrm{pool}}=V_l+V_b.
$$

The corresponding planar levels are

$$
h_{\mathrm{clear}}=\mathcal V^{-1}(V_l),\qquad
h_{\mathrm{pool}}=\mathcal V^{-1}(V_l+V_b).
$$

`clearLevel` is the hypothetical level after dispersed bubbles disappear.
`poolLevel` is the bubble-displaced mixture-surface level. In Milestone A it is
a global diagnostic, not a boundary location used by the flow solver.

For a constant-area vessel with bottom elevation $h_b$ and area $A$,

$$
\mathcal V(h)=A(h-h_b),\qquad
h=h_b+\frac{V}{A}.
$$

`ConstantAreaVesselVolumeMap` implements this analytic relation.
`TabulatedVesselVolumeMap` uses piecewise-linear interpolation of strictly
increasing heights and nondecreasing volumes, and a bracketed monotone inverse.
The first table volume must be zero. A zero-volume plateau is accepted as a
geometric table feature, but requests whose inverse would be non-unique and
area queries on a zero-area segment fail explicitly.

The accepted volume balance reported by `PlanarFreeSurfaceModel` is

$$
E_V=\mathcal V(h_{\mathrm{pool}})-V_l-V_b.
$$

Out-of-range requests either throw (`error`, the default) or return a clamped
value together with an explicit overflow/underflow amount
(`clampAndReport`). Closed headspace operation still rejects overfill because
it requires a positive headspace volume. When a configured initial clear level
is outside the map, its separate level underflow/overflow in m remains in every
accepted diagnostic snapshot rather than being lost after conversion to a
clamped volume.

## Liquid Mass and Pure-Liquid Density

Liquid volume must use material, bubble-free liquid density
$\rho_l(T,c,p)$, not `rhoFeedback` or another void-reduced mixture density.
Using mixture density here would count bubble displacement once through
density and again through $V_b$. `MaterialFeedbackModel::pure_liquid_density`
provides the supported temperature-dependent material-density evaluation, and
the liquid inventory publishes it as `rhoLiquid` in kg/m3.

The current material evaluator covers the configured constant,
Boussinesq-temperature, and configured pure-liquid branch of mixture feedback.
It does not yet supply composition-dependent density or a liquid
compressibility law. The planar model accepts a pressure-dependent liquid
volume callback so those laws can be added without changing its ownership.

The `globalConstantMass` fallback retains total liquid mass $M_l$ and fixed
reference mass fractions $w_c$ initialized from
$\rho_{l,c}^0 V_c$ over owned cells. It evaluates

$$
\sum_c w_c=1,\qquad
\bar v_l=\sum_c\frac{w_c}{\rho_{l,c}},\qquad
V_l=M_l\bar v_l.
$$

All sums are communicator-wide reductions over owned cells. Changing void
fraction alone cannot change $V_l$ because neither mixture density nor
`alpha_g` is accepted by this API. At an accepted phase-change update,

$$
M_l^{n+1}=M_l^n+m_{\mathrm{cond}}-m_{\mathrm{evap}},
$$

and the reported mass residual is

$$
E_M=M_l^0+m_{\mathrm{cond,cum}}-m_{\mathrm{evap,cum}}-M_l.
$$

Evaporation beyond available liquid either throws or is accepted only up to
the available mass while recording a dry-out deficit under
`clampAndReport`.

`globalConstantMass` is not a locally conservative liquid-mass transport
scheme. The fixed $w_c$ distribution is a mass-weighted specific-volume
approximation.

`cellMassInventory` instead stores the conservative fixed-control-volume
inventory $m_{l,c}^*$ in kg/m3. Its implemented cell equation is

$$
\frac{V_c\left(m_{l,c}^{*,n+1}-m_{l,c}^{*,n}\right)}{\Delta t}
+\sum_{f\in c}\phi_f m_{l,f}^{*,\mathrm{upwind},n+1}
=V_c\left(\dot m_{\mathrm{cond},c}-\dot m_{\mathrm{evap},c}\right),
$$

with the existing backward-Euler/upwind scalar transport operator, unit
storage/advection weights, zero diffusivity, and the accepted projected
single-continuum face-volume flux $\phi_f$ in m3/s. This is not a separate
phase velocity or liquid-fraction-weighted flux. The phase-change rates use
kg/(m3 s). The trial field and diagnostics remain uncommitted until the planar
volume/headspace closure succeeds. Liquid volume is then evaluated directly:

$$
M_l=\sum_c V_cm_{l,c}^*,\qquad
V_l=\sum_c V_c\frac{m_{l,c}^*}{\rho_{l,c}}.
$$

The initial field is distributed without a cell-centre pool mask:

$$
m_{l,c}^{*,0}=M_l^0\frac{\rho_{l,c}^0}
{\sum_j\rho_{l,j}^0V_j}.
$$

When mass is inferred from fill volume this is a uniform smeared reference-
volume fill fraction and reproduces the configured global volume exactly; it
is not a resolved pool interface. Until a liquid inlet/composition boundary
contract exists, cellwise mode collectively rejects nonzero physical-boundary
liquid flux. It also requires `error` depletion handling: locally clipping a
sink after boiling has accepted vapor would break phase conservation. The
accepted history and per-step mass residuals use a strict normalized tolerance
of `max(4096 * machine epsilon, 1e-10)`, independent of the linear-solver
tolerance. The current state is effective total liquid mass, not separate
solvent and fissile inventories, so composition-preserving evaporation remains
deferred.

## Bubble Volume, Boiling, and Exact Escape

Only gas-phase inventories that remain submerged contribute to $V_b$.
Dissolved hydrogen is retained in the gas conservation ledger but occupies no
bubble volume. For the two-population radiolytic model,
`global_submerged_bubble_volume()` integrates the unbounded, EOS-derived
`alpha_g_raw` from the microbubble and large-bubble states. The separately
reported `global_unrepresented_bubble_volume()` is the raw volume hidden by an
`alpha_g` upper bound; capping a representational field therefore does not
destroy moles or silently reduce the volume inventory.

Candidate-pressure evaluation reuses the radiolytic model's accepted
temperature and population states, configured surface-tension correlation,
and existing Laplace-pressure bubble-radius solve. Conceptually, each
population follows

$$
p_b=p_{l,\mathrm{abs}}+\frac{2\sigma}{R_b}+p_{\mathrm{inertia}},
\qquad
V_b=\sum_c\left(\alpha_{g,\mathrm{raw},c}V_c\right),
$$

with the actual implemented radius/EOS relation remaining authoritative in
`RadiolyticGasModel`. The free-surface closure does not contain a second ideal
gas or Laplace formula.

Bubble escape is measured as the exact globally reduced before/after
decrement of the transported microbubble and large-bubble mole inventories.
`submerged_bubble_hydrogen_escaped` is their sum and is the H2 quantity to
transfer to a vent/headspace. Dissolved boundary outflow is reported
separately; the legacy total `hydrogen_escaped` includes both for the existing
radiolytic balance. The free-surface update accepts
`escaped_moles_this_step`, previews it during the closure, and commits it once
only after a successful solve. A failed closure therefore cannot duplicate or
lose the transfer ledger: the pending amount remains the radiolytic model's
authoritative cumulative submerged-bubble escape minus the planar model's
committed-transfer watermark. Every positive decrement is retained, including
values small relative to a large remaining inventory. Because upstream flow,
temperature, gas, boiling, void, and precursor states are not generally
transactional, `BoussinesqSolver` fail-stops after such a step error and rejects
a full-step retry; reconstruct the solver from an accepted state instead.

The boiling path likewise has one owner. Accepted
`phaseChangeMassRate = latentHeatSink / latent_heat` removes liquid mass; vapor
rejected by the scalar-void cap removes neither mass nor latent energy and is
reported through `rejectedVaporMassRate`. After the scalar-void update,
`complete_void_fraction_update()` assigns realizable collapse only to steam
already tracked at the beginning of the step; newly accepted evaporation
cannot be used to explain collapse of the old aggregate void. It returns that
condensate to liquid, adds the matching latent heat through
`condensationLatentHeatRelease`, and reports any remaining non-steam collapse.
This steam/condensate ownership mode is selected explicitly by the
`BoussinesqSolver` only while a free-surface liquid inventory is active; its
default-off completion path preserves pre-feature boiling and latent-sink
behavior.
Because steam ownership is currently lumped, the accepted release is
distributed in proportion to bounded local collapse: it is globally
conservative but not a local steam-fraction model. The current boiling path does not
yet transport steam as a separate species to the free surface, so a steam
escape transfer to headspace must not be inferred from scalar-void collapse.
The existing solver also rejects simultaneous Sheng two-population radiolysis
and boiling, so the integrated $V_b$ callback currently uses either raw H2
bubble volume or owned submerged steam volume, never an unowned aggregate.
Vented boiling liquid-mass/steam-volume accounting is supported. Closed
headspace with active boiling is rejected because the current boiling model
uses a configured fixed `saturation_temperature`; it does not yet evaluate
pressure-dependent saturation from the absolute headspace pressure. Once this
coupling owns nonzero submerged steam, `remove_free_surface_model()` also
rejects removal or replacement until that inventory has returned to zero; no
implicit discard or transfer to the default-off path is allowed.

## Absolute Pressure and Headspace Models

All gas EOS evaluations and headspace states use Pa absolute. Dynamic/gauge
pressure remains a momentum variable and is not passed directly to the gas
closure. For constant and reconstructed radiolytic pressure modes,
`set_absolute_pressure_offset()` changes the thermodynamic offset while
preserving reconstructed gauge-pressure variation. Candidate bubble-volume
queries are non-mutating. Prescribed-history and inertial pressure modes reject
an externally coupled offset because their pressure ownership is different.
`BoussinesqSolver` therefore rejects those Sheng pressure modes when the
free-surface budget is enabled; constant and reconstructed pressure are the
supported combinations. It also rejects `idealGasSource`, which has no
conservative gas inventory, and nonzero scalar void that cannot be matched to
an authoritative submerged-gas owner.

Two headspace choices are implemented:

- `vented` fixes $p_h=p_{\mathrm{ambient}}$. Escaped species accumulate in
  `ventedMoles`; headspace moles remain empty. Its reported geometric volume
  is the nonnegative unused internal vessel volume.
- `closed` retains escaped species in a lumped ideal-gas inventory. Its
  positive headspace volume and pressure obey

  $$
  V_h=V_{\mathrm{vessel,total}}-V_l(p_h)-V_b(p_h),
  $$

  $$
  p_h V_h=n_h ZRT_h.
  $$

  The closure brackets absolute pressure between configured positive bounds
  and the collective lower-pressure domain reported by a coupled bubble
  callback, then uses safeguarded secant/bisection iterations. Acceptance uses
  the stricter of the configured pressure tolerance and the internal ideal-gas
  state-consistency tolerance. A nonpositive headspace volume, missing positive
  gas inventory, root outside the bracket, or failed
  convergence is an error. Under-relaxation applies only to this nonlinear
  pressure solve; it never modifies an inventory to conceal a residual.

For `fixed`, the configured initial temperature is retained. For `prescribed`,
the configured time/temperature table is strictly increasing in time and is
linearly interpolated without extrapolation; the owning timestep integration
passes that finite positive value to the closure. For `bulkLiquid`, the caller
must supply a finite positive representative liquid temperature on each
update. The initial closed-headspace moles may be supplied by species, as a
single `gas` inventory, or inferred from the initial pressure, temperature,
and available volume. Explicit moles are closed-only, mutually exclusive with
inference, and must reproduce that configured initial thermodynamic state.
`restrictedVent` is reserved and deliberately rejected;
there is no validated orifice law.

## Flat Database Configuration

The parser follows the repository's flat underscore-separated `Database`
convention. All dimensional values are SI. Keys marked "required when used"
have no meaningful default.

### Selection and vessel map

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_enabled` | `false` | boolean | Disabled construction returns no model and preserves existing behavior. |
| `free_surface_model` | `fixed` | `fixed`, `planarVolumeBudget`, `planarALE` | Enabling currently requires `planarVolumeBudget`; `planarALE` fails because its standalone geometry/GCL substrate is not connected to conservative ALE transport or solver coupling. |
| `free_surface_gravity_axis` | `z` | `x`, `y`, `z` | Records the planar elevation axis; Milestone A does not reorient or move the mesh. |
| `free_surface_validity_warning_relative_level_change` | `0.05` | dimensionless, >= 0 | Warn when `abs(poolLevel - initialPoolLevel) / vesselHeight` exceeds this value. |
| `free_surface_overflow_policy` | `error` | `error`, `clampAndReport` | Applies to vessel range and liquid depletion handling. |
| `free_surface_vessel_model` | `constantArea` | `constantArea`, `tabulated` | Selects the monotone map. |
| `free_surface_bottom_elevation` | required for constant area | m | Finite and below the top elevation. |
| `free_surface_top_elevation` | required for constant area | m | Finite and above the bottom elevation. |
| `free_surface_cross_section_area` | required for constant area | m2 | Finite and positive. |
| `free_surface_height_table` | empty | m array | At least two finite, strictly increasing values for `tabulated`. |
| `free_surface_volume_table` | empty | m3 array | Same size as height table, starts at zero, is nondecreasing, and ends positive. |
| `free_surface_total_internal_volume` | mapped usable volume | m3 | Must be positive and no smaller than the mapped usable volume; it is the single vessel/headspace capacity. |

### Initial fill and liquid inventory

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_liquid_volume_model` | `globalConstantMass` | `globalConstantMass`, `cellMassInventory` | Cellwise mode transports kg/m3 reference-volume inventory and currently requires `error` depletion plus zero physical-boundary liquid flux. |
| `free_surface_initial_liquid_mass` | not set | kg | Finite and nonnegative. |
| `free_surface_initial_liquid_volume` | not set | m3 | Finite, nonnegative, and within the vessel policy range. |
| `free_surface_initial_clear_level` | not set | m | Converted through the vessel map. |

At least one of initial mass, volume, or clear level is required when the
volume-budget model is enabled; the model never assumes a completely full
vessel. If both volume and level are provided, they must agree through the map
within the setup consistency tolerance. If mass is omitted, the liquid
inventory infers it from the selected initial volume and the global
volume-weighted pure-liquid density. If mass and a volume/level are both
provided, `BoussinesqSolver` verifies that the initialized pure-density volume
is consistent rather than silently choosing one.

### Headspace and nonlinear closure

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_headspace_model` | `vented` | `vented`, `closed` | `restrictedVent` is rejected; `closed` is incompatible with active fixed-saturation boiling. |
| `free_surface_ambient_pressure` | `101325` | Pa absolute | Active vented pressure; required positive in every mode. |
| `free_surface_initial_pressure` | `101325` | Pa absolute | Active closed initialization pressure; required positive in every mode. |
| `free_surface_headspace_temperature_model` | `fixed` | `fixed`, `prescribed`, `bulkLiquid` | Non-fixed modes require a supplied temperature on every update. |
| `free_surface_initial_temperature` | `293.15` | K | Positive; retained by `fixed`. |
| `free_surface_headspace_temperature_times` | empty | s array | Required by `prescribed`; finite and strictly increasing. |
| `free_surface_headspace_temperature_values` | empty | K array | Required by `prescribed`; same nonzero length as times and all positive. |
| `free_surface_headspace_gas_constant` | `8.31446261815324` | J/(mol K) | Positive. |
| `free_surface_headspace_compressibility_factor` | `1.0` | dimensionless | Positive constant $Z$. |
| `free_surface_initial_headspace_moles` | not set | mol | Closed only; when absent, the closed model infers inventory. Optional single-species (`gas`) inventory; mutually exclusive with per-species arrays. |
| `free_surface_headspace_species` | empty | string array | Must accompany the mole array; names must be unique. |
| `free_surface_initial_headspace_species_moles` | empty | mol array | Closed only. Nonnegative, same nonzero length as species; must match the configured initial thermodynamic state. |
| `free_surface_coupling_max_correctors` | `100` | integer | Positive. |
| `free_surface_coupling_absolute_tolerance` | `1e-8` | Pa | Positive pressure residual tolerance. |
| `free_surface_coupling_relative_tolerance` | `1e-10` | dimensionless | Nonnegative. |
| `free_surface_coupling_relaxation` | `1.0` | dimensionless | In `(0, 1]`, closed closure only. |
| `free_surface_minimum_absolute_pressure` | `1.0` | Pa absolute | Positive lower bracket. |
| `free_surface_maximum_absolute_pressure` | `1e9` | Pa absolute | Positive and greater than the lower bracket. |
| `free_surface_volume_closure_absolute_tolerance` | `1e-12` | m3 | Nonnegative absolute acceptance tolerance. |
| `free_surface_volume_closure_relative_tolerance` | `1e-10` | dimensionless | Nonnegative; scales with the pool/map volume. |
| `free_surface_gas_closure_absolute_tolerance` | `1e-12` | mol/species | Nonnegative per-species acceptance tolerance. |
| `free_surface_gas_closure_relative_tolerance` | `1e-10` | dimensionless | Nonnegative; scales with the species ledger. |

A minimal fixed-grid, vented constant-area setup is conceptually:

```text
free_surface_enabled = true
free_surface_model = planarVolumeBudget
free_surface_vessel_model = constantArea
free_surface_bottom_elevation = 0.0
free_surface_top_elevation = 1.2
free_surface_cross_section_area = 0.5
free_surface_initial_clear_level = 0.8
free_surface_headspace_model = vented
free_surface_ambient_pressure = 101325.0
```

Solver integration also requires the dimensional physical-model
`BoussinesqSolver` construction path; the legacy/default constructor rejects an
enabled free surface rather than inventing SI material properties. Install any
custom material updater before initializing the free-surface liquid-mass
reference state. The initializer evaluates that updater at the accepted initial
temperature before inferring mass.

The examples in this repository configure `Database` programmatically; this
block documents keys and is not a promise of a general file-backed case
parser.

## Diagnostics, Output, and Feedback Fields

`FreeSurfaceDiagnostics` is an immutable-by-value accepted-state snapshot. It
contains:

- accepted time and timestep in s;
- liquid, submerged-bubble, and pool volumes in m3;
- old/current clear and pool levels in m, their accepted-step rates in m/s,
  and current surface area in m2;
- overflow/dry-out volume in m3 and configured initial-level under/overflow in m;
- headspace pressure (Pa absolute), volume (m3), temperature (K), and total
  moles;
- nonlinear iteration count and pressure residual in Pa;
- absolute and normalized volume-closure residuals and the fixed-grid validity warning;
- generated, dissolved, submerged, just-escaped, documented-sink, headspace,
  and vented gas moles by species;
- per-species gas-inventory closure residuals in mol, their normalized values,
  and the summed/max-normalized gas residuals.

The gas ledger uses

$$
E_{g,s}=n_{s,0}+n_{s,\mathrm{generated}}
-n_{s,\mathrm{dissolved}}-n_{s,\mathrm{submerged}}
-n_{s,\mathrm{headspace}}-n_{s,\mathrm{vented}}
-n_{s,\mathrm{other\ sinks}}.
$$

`LiquidMassInventoryDiagnostics` separately reports initial/current mass,
cumulative accepted evaporation and condensation, dry-out mass deficit,
mass-weighted specific volume, liquid volume, and absolute/normalized
accepted-history and per-step mass residuals. `BoilingPhaseChangeDiagnostics` separates requested, accepted, and
rejected evaporation; steam and non-steam collapse; condensate; submerged
steam; and phase-change residuals. The boiling output registry exposes
`phaseChangeMassRate`, `condensationMassRate`, `rejectedVaporMassRate`, and
`condensationLatentHeatRelease` alongside the existing `S_alpha_boil` and
`latentHeatSink` fields. The radiolytic output retains raw,
bounded, and excess void/inventory diagnostics.

The feedback-field registry has stable optional names `rhoLiquid`,
`clearLevel`, `poolLevel`, `headspacePressure`, and `poolOccupancy`, and can
collectively require that set without changing the older standard-field
contract. `BoussinesqSolver` publishes `rhoLiquid` cellwise and replicates the
three global scalars over cells. `poolOccupancy` is exactly 1 when the cell
centroid coordinate on the configured gravity axis is at or below `poolLevel`
and 0 otherwise. It is a visualization/feedback approximation, not a cut-cell
fraction. `pool_occupancy_volume_error()` reports

$$
E_{\mathrm{occupancy}}=\sum_c C_cV_c-V_{\mathrm{pool}},
$$

so its geometric error is explicit.

`SolutionOutputOptions::include_free_surface_fields` defaults to `false`.
When enabled, VTU output includes `rhoLiquid`, `clearLevel`, `poolLevel`,
`headspacePressure`, and `poolOccupancy`; cellwise mode additionally writes
`liquidMassInventory`. The default preserves the existing
output schema. `BoussinesqSolver::free_surface_history()` automatically retains
the initialization snapshot and one record per accepted step without repeating
the underlying reductions. `write_free_surface_history_csv()` writes that
history on mesh rank zero with fixed SI-unit columns for liquid mass and
volume, separate H2 populations/compartments, levels, headspace state,
overflow/dry-out, optional boiling mass/energy, and absolute/normalized closure
residuals. Steam moles are intentionally absent because the current boiling
model owns steam mass and volume, not a molar species inventory. The analytic
verification executable also emits its own compact CSV stdout.

## Verification Scope

Focused tests cover the implemented component contracts:

- analytic constant-area forward/inverse behavior, tabulated interpolation,
  malformed/zero-area tables, endpoint handling, and range policies;
- fixed-density, temperature-dependent, phase-change, pure-versus-mixture
  density, and distributed liquid-inventory behavior;
- cellwise initialization, transactional local phase change, internal
  advection, dryout/boundary rejection, density/stale-token rejection, strict
  solver-independent mass closure, and partition-face transport;
- vented constant pressure, closed ideal-gas analytic closure, nonlinear
  pressure-sensitive liquid/bubble volume, exact species transfer, gas closure,
  overfill, and convergence failures;
- two-population raw bubble volume, dissolved-gas exclusion, pressure and
  temperature response through the existing radius/EOS solve, volume hidden by
  the void cap, absolute-pressure offset handling, and serial/MPI reductions;
- accepted/rejected boiling mass, latent-energy consistency, submerged steam,
  condensation, and balance residuals;
- deterministic registration/export of the optional free-surface feedback
  field names;
- Boussinesq disabled-baseline equivalence, thermal expansion from pure-liquid
  density, initialization/update ordering, accepted boiling mass in both
  inventory modes, positive cellwise condensate return exactly once, zero-flow
  cellwise preservation, guarded removal with retained steam, exact H2 bubble
  escape to the vent, pressure-mode/ownership rejection, opt-in VTU fields, and
  the reported cell-centre occupancy error.

`PlanarFreeSurfaceModel_2procs` checks a nonuniform pure-density liquid volume
against an independently reduced value and verifies that liquid mass, volume,
level/rate, escape, and closure diagnostics are replicated across two ranks.
The existing two-rank radiolytic regression separately checks global raw
bubble volume, population moles, cap discrepancy, and conservative finite-
Courant escape.

The standalone B0 motion tests cover Cartesian motion on every axis,
cylindrical and semi-structured axial motion, buffered/repeated motion,
transaction accept/rollback, quality rejection, exclusive shared-geometry
ownership, unchanged topology/IDs/maps/boundary tags, per-cell GCL, MPI
partition-face swept-flux agreement, and collective target/timestep/action
validation. Geometry-cache tests verify stale epoch rejection and analytic
coefficient/gradient recovery after explicit refresh.

The final post-fix focused selection passed 73/73 tests; seven MPI-only cases
were intentionally skipped by their single-rank guards. Six related
host-network MPI registrations also passed: `PlanarALEMeshMotion_2procs`,
`RadiolyticGasModel_2procs`, `BoilingCollectiveValidation_2procs`,
`PlanarFreeSurfaceModel_2procs`, `BoussinesqFreeSurfaceConfig_2procs`, and
`BoussinesqFreeSurfaceStepPreflight_2procs`.

A fresh GCC Debug registry contained 842 tests. These two disjoint full
executions covered that registry:

- `ctest --test-dir build/gcc -C Debug -LE mpi -j 4` passed 803/803 with 29
  expected single-rank skips;
- the host-network `ctest --test-dir build/gcc -C Debug -L mpi -j 2` run passed
  39/39.

After the final formatting pass, the GCC Debug and RelWithDebInfo builds
completed 198 and 253 scheduled steps, respectively; repeated builds reported
no work. The documentation target also completed successfully.

The `planar_free_surface_verification` executable provides five deterministic
constant-area analytic cases. Run all cases or one named case with:

```bash
./build/gcc/bin/Debug/planar_free_surface_verification
./build/gcc/bin/Debug/planar_free_surface_verification uniformHeating
./build/gcc/bin/Debug/planar_free_surface_verification gasGeneration
./build/gcc/bin/Debug/planar_free_surface_verification completeEscape
./build/gcc/bin/Debug/planar_free_surface_verification closedHeadspace
./build/gcc/bin/Debug/planar_free_surface_verification boilingMassLoss
```

The corresponding five `verification` CTests and a no-argument integration
smoke are registered. The driver emits CSV rows containing time, liquid and
bubble volumes, clear/pool levels, absolute headspace pressure, nonlinear
pressure residual, volume residual, and gas residual. The cases check:

- reciprocal-density uniform heating at constant liquid mass;
- H2 and steam bubble-volume displacement at vent pressure;
- exact complete H2/steam transfer to a vented ledger;
- a closed H2/air headspace against the positive root of an independent
  ideal-gas/Laplace-pressure quadratic;
- the clear-level decrement from an accepted 18 kg liquid-mass loss. This last
  case does not synthesize steam moles or steam escape; those remain
  unsupported in the production boiling path.

General level, volume, and species checks use an absolute-plus-relative
tolerance of `5e-12`. The closed-pressure check uses `5e-7 Pa + 2e-12`
relative, its $pV-nRT$ check uses `2e-6 J`, and its nonlinear pressure residual
limit is `5e-8 Pa`. A representative all-case GCC Debug run reported maximum
absolute volume closure `4.440892099e-16 m3`, maximum absolute gas closure
`0 mol`, and closed nonlinear residual `2.910383046e-11 Pa`.

These are analytic component/conservation verifications, not a physical
validation of pool dynamics. Separate B0 tests verify geometry-only GCL; they
do not validate an ALE transport or free-surface solve.

## Fixed-Grid Limitations

Milestone A leaves the computational geometry and all transport volumes
unchanged. The flow equations continue over the configured domain and bubble
escape continues at the existing actual boundary. The model does not:

- deactivate cells above `poolLevel`; the published cell-centre
  `poolOccupancy` mask does not alter flow or transport;
- move the physical escape boundary to the reported level;
- compute partially cut-cell fluxes or a conservative virtual escape plane;
- alter pressure-velocity continuity for thermal, phase-change, or bubble
  volume sources;
- use mesh-relative fluxes or remap extensive state;
- resolve surface shape, curvature, capillary waves, splash, foam, or overflow
  hydrodynamics.

Cellwise liquid mass is conservative on the fixed reference mesh, but its
initial fill is deliberately smeared over that modeled domain and physical
boundary mass flux is not yet supported. It uses the projected
single-continuum face flux rather than a liquid-fraction-weighted phase flux.
No separate solvent, solute, or fissile-mass field exists, so local total-liquid
conservation must not be described as composition-preserving evaporation.

The `validity_warning` threshold is therefore an applicability diagnostic, not
an error correction. A small reported level change can still be inaccurate if
the fixed top boundary or unmodified flow domain materially changes escape or
circulation. `clampAndReport` exposes a bookkeeping excess; it does not model
overflow fluid.

## Milestone B0 Geometry/GCL Substrate and Remaining B/C Blocker

`PlanarALEMeshMotion` now provides a standalone, transactional geometry layer.
It owns reference coordinates, accepted/trial surface elevation, old/new local
cell volumes, and owner-oriented swept-volume face rates. A trial applies the
configured full-column or buffered mapping, evaluates

$$
E_{\mathrm{GCL},c}=
\frac{V_c^{n+1}-V_c^n}{\Delta t}-\sum_f\phi_{m,f},
$$

checks every owned-cell residual and the distributed mesh-quality gate, and is
then explicitly accepted or rolled back. Rejected trials restore coordinates;
topology, global IDs, Tpetra maps, boundary tags, and partitioning do not
change. A geometry epoch and exclusive controller lease live on the shared
concrete geometry so alias `MeshHandle`s observe the same revision.

The truthful B0 support set is:

- Cartesian X/Y/Z motion, serial and MPI;
- cylindrical axial-Z motion, serial and MPI;
- `SemiStructuredXY_Z` axial-Z motion, serial only.

B0 is constructed programmatically and is not selected through the
free-surface `Database`. `PlanarALEMeshMotionOptions` defaults to axis `z`, a
full-column deformation beginning at the reference bottom, no per-trial level
limit, `1e-12 m3/s` absolute plus `1e-10` relative GCL tolerance, and the
existing `MeshQualityLimits` defaults. Supplying a deformation-start elevation
creates a monotone upper deformation zone above a fixed lower region;
`maximum_level_change` is an optional positive pre-mutation rejection limit.

Const-backed handles, non-axial cylindrical/semi-structured motion, general
unstructured/partitioned geometry, and STK/legacy meshes fail explicitly.
Direct raw mutation through a retained concrete pointer or `visit_mutable()`
cannot publish an epoch and remains unsupported after fields or caches exist.

`CellGradientCache`, `TransportGeometryCache`, and both Rhie--Chow face-flux
workspaces capture the geometry epoch, reject stale access, and expose explicit
numeric refresh. This is not yet a centralized refresh of every solver-owned
cache.

`free_surface_model = planarALE` therefore remains rejected. The remaining
solver-level blockers are:

- `FluidSolver` still owns a const `MeshHandle`, and no accepted multiphysics
  step owns geometry trial/accept/rollback together with fields and inventories;
- transient/convection operators still use one current control volume rather
  than `(V_new q_new - V_old q_old) / dt` and do not subtract mesh flux;
- wall distance/y+, pressure matrices, coupled-system numeric state, VTU
  points, and other solver caches do not yet share one epoch refresh path;
- pressure projection, SIMPLE/PISO/PIMPLE, and coupled Krylov paths do not all
  accept and report one generalized nonzero continuity target;
- moving-top kinematic/dynamic and exact bubble-escape boundary conditions do
  not exist.

Consequently constant-field preservation under motion, ALE liquid/gas/
precursor/energy conservation, and a moving free-surface solve are not claimed.
Merely connecting the reported level to this geometry layer before those
operators exist would still be false ALE.

Milestone C currently stops at stable feedback names and an explicitly
nonconservative cell-centre occupancy visualization with a measured volume
error. Conservative pool occupancy needs geometric cut fractions or a mapper
whose approximation error is part of its acceptance contract; the current
Heaviside field is not that mapper. Moving-mesh and cross-mesh feedback also
need conservative handling of liquid/fissile mass, gas species, precursor
inventory, and energy, plus reported mapping residuals. Those capabilities
remain unsupported until the remaining Milestone-B solver and ALE transport
contract exists. Full VOF and Euler-Euler phase momentum remain separate
deferred model families, not implicit future claims of this planar budget.
