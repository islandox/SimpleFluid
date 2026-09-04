# Planar Free-Surface Volume Budget and Constrained ALE

This note describes two related planar free-surface paths. The Milestone-A
`planarVolumeBudget` path retains the fixed CFD grid while closing liquid,
submerged-gas, level, and headspace inventories. The constrained Milestone-B
`planarALE` path connects that same ownership to `PlanarALEMeshMotion`,
conservative old/new-volume transport, mesh-relative convection, a generalized
low-Mach volume-continuity target, and an atomic accepted-step transaction.

The ALE path is deliberately narrow: native mutable structured/extruded
geometry, a constant-area vessel, Backward Euler, laminar dimensional
Boussinesq temperature transport, pure-liquid thermal-density feedback,
`cellMassInventory`, vented headspace, and either no radiolysis or the supported
Sheng two-population H2 transport/escape configuration. It is a flat moving
boundary, not an interface-capturing method, and its verification is currently
conservation- and solver-contract-focused rather than physical validation of
pool dynamics. Composition-resolved liquid transport, conservative steam
transport/escape, and conservative cross-mesh or cut-cell mapping remain open.

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

`BoussinesqSolver` owns the optional model and liquid inventory. In fixed-grid
mode its accepted post-temperature order is: advance the Sheng bubble/void
state when selected;
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
`poolLevel` is the bubble-displaced mixture-surface level. In fixed-grid mode it
is a global diagnostic, not a boundary location used by the flow solver. In
`planarALE` it is the accepted target elevation of the named planar top patch.

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

`cellMassInventory` instead stores the conservative liquid-mass density
$m_{l,c}^*$ in kg/m3 of the active control volume. On a fixed grid its cell
equation is

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

For `planarALE`, the same inventory uses accepted-old and trial-new volumes and
the mesh-relative carrier flux:

$$
\frac{V_c^{n+1}m_{l,c}^{*,n+1}-V_c^nm_{l,c}^{*,n}}{\Delta t}
+\sum_{f\in c}\phi_{rel,f}m_{l,f}^{*,\mathrm{upwind},n+1}
=V_c^{n+1}\left(\dot m_{\mathrm{cond},c}
-\dot m_{\mathrm{evap},c}\right),
$$

where $\phi_{rel,f}=\phi_{abs,f}-\phi_{m,f}$. The first ALE support matrix
rejects boiling, so the right-hand side is zero there; retaining the source form
states the contract that boiling must satisfy before that rejection can be
relaxed. A constant intensive inventory is preserved under pure mesh motion by
the cellwise GCL, while the extensive liquid mass is evaluated with
$V_c^{n+1}$ only after the trial is accepted.

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
transactional in the fixed-grid path, `BoussinesqSolver` fail-stops after such a
step error and rejects a full-step retry; reconstruct the solver from an
accepted state instead. The constrained ALE path instead snapshots its complete
supported state and restores it together with the geometry on rejection.

With ALE, bubble populations are transported by the liquid carrier flux
$\phi_{rel}$ plus the configured category slip flux. Escape is evaluated on the
named moving top patch from that relative bubble flux, so the mesh motion is not
counted as gas leaving the liquid domain. The accepted before/after decrement
remains the authoritative H2 transfer and is committed exactly once; the
reconstructed `alpha_g` cap still cannot delete gas.

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

For `planarALE`, the vent owns the absolute thermodynamic pressure $p_h$, while
the CFD pressure remains the gauge/dynamic field $p'$. The moving top requires
a zero-gauge Dirichlet physical-pressure condition. During pressure correction,
the exact prescribed absolute face flux is enforced as a correction-Neumann
condition rather than allowing a pressure correction to change the kinematic
boundary flux. The radiolytic reconstruction subtracts the volume-weighted
gauge mean, so the ALE coupling supplies $p_h+\overline{p'}$ as its internal
offset:

$$
p_{l,\mathrm{abs},c}=
(p_h+\overline{p'})+p'_c-\overline{p'}=p_h+p'_c.
$$

At the zero-gauge top this gives $p_{l,\mathrm{abs}}=p_h$; the headspace
pressure is neither omitted nor added twice. The flat boundary has no
curvature-pressure jump. That is an explicit planar approximation, not
surface-tension shape resolution.

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

## Conservative Planar ALE Contract

All face-volume fluxes use m3/s and the mesh owner normal. The existing
projected field remains the absolute liquid flux

$$
\phi_{abs,f}=\int_f\mathbf u\cdot\mathbf n_o\,dA.
$$

`PlanarALEMeshMotion` supplies the exact swept-volume rate $\phi_{m,f}$ for the
same face and timestep. ALE convection uses a separate synchronized field

$$
\phi_{rel,f}=\phi_{abs,f}-\phi_{m,f}.
$$

The absolute field is never overwritten with relative values. Pressure,
kinematic-boundary enforcement, and continuity diagnostics consume
$\phi_{abs}$; momentum, temperature, liquid inventory, enabled H2 transports,
and the ALE Courant diagnostic consume $\phi_{rel}$. Specifically,

$$
Co_c=\frac{\Delta t}{2V_c}\sum_{f\in c}|\phi_{transport,f}|,
\qquad
\phi_{transport}=\begin{cases}
\phi_{abs},&\text{fixed mesh},\\
\phi_{rel},&\text{planar ALE}.
\end{cases}
$$

The non-owning `ALEControlVolumeState` carries
accepted-old and active-trial-new volumes in mesh-local cell order, mesh fluxes
in mesh-local face order, timestep, concrete geometry identity, and old/new
epochs. It is valid only while the originating motion trial is active and
validates sizes, finiteness, positivity, identity, epoch, and the cellwise GCL.

For a conservative intensive unknown $q$ with storage weight $a$, advective
weight $b$, and volumetric source $S$, the Backward-Euler assembly is

$$
\frac{V_c^{n+1}a_c^{n+1}q_c^{n+1}
      -V_c^na_c^nq_c^n}{\Delta t}
+\sum_{f\in c}\phi_{rel,f}(bq)_f^{\mathrm{upwind},n+1}
=\mathcal D_c^{n+1}+V_c^{n+1}S_c^{n+1}.
$$

The first enabled path uses unit storage for momentum and local liquid/gas
inventories. Physical temperature instead uses the conservative cellwise
liquid-mass density $m_{l,c}^*$ and heat capacity $c_{p,c}$:

$$
E_T^n=\sum_c V_c^n m_{l,c}^{*,n}c_{p,c}^nT_c^n,
$$

and its moving-volume transient term is

$$
\frac{V_c^{n+1}m_{l,c}^{*,n+1}c_{p,c}^{n+1}T_c^{n+1}
      -V_c^nm_{l,c}^{*,n}c_{p,c}^nT_c^n}{\Delta t}.
$$

The accepted-old inventory supplies the old storage density and a conservative
trial inventory preview supplies the new storage and advective density. This
does not use pure-liquid density or void-reduced mixture density over the
bubble-displaced pool volume: $\rho_l$ remains the conversion from liquid mass
to material volume. New-time diffusion and source geometry use the trial mesh.
Only Backward Euler is enabled: fixed-grid BDF2 support does not imply a valid
moving-volume history and is rejected at ALE setup.

### Material-volume source and generalized continuity

`VolumeContinuityTarget` stores one immutable integrated target $Q_{V,c}$ in
m3/s for each owned cell. Every segregated SIMPLE, PISO, and PIMPLE pressure
correction and the coupled Krylov path uses the same constraint and residual:

$$
\sum_{f\in c}\phi_{abs,f}-Q_{V,c}=0.
$$

It is an integrated cell rate, not a volumetric source in 1/s, so pressure
assembly must not multiply it by cell volume. Target identity, geometry epoch,
generation, rank parity, and finiteness are validated before collective
assembly; predictor and numeric caches include those values in their reuse
contract.

For the supported liquid-plus-bubble material ledger, let

$$
W_c=\frac{m_{l,c}^*}{\rho_{l,c}}V_c
    +\alpha_{g,\mathrm{raw},c}V_c.
$$

Liquid mass uses its implicit trial-new upwind value. Each bubble population
retains a post-transport/pre-kinetics raw volume fraction, with the upwind cell
selected by the same combined $\phi_{rel}+\phi_{slip,k}$ sign used by its
number/mole transport. The resulting carrier and slip material-volume face
fluxes are

$$
\Phi_{carrier,f}=\phi_{rel,f}
  \left(\frac{m_l^*}{\rho_l}\right)_f^{n+1,up(\phi_{rel})}
  +\sum_k\phi_{rel,f}\alpha_{b,k,f}^{tr,up(\phi_{rel}+\phi_{slip,k})},
$$

$$
\Phi_{slip,f}=\sum_k\phi_{slip,k,f}
  \alpha_{b,k,f}^{tr,up(\phi_{rel}+\phi_{slip,k})}.
$$

The final $W^{n+1}$ is evaluated after local gas kinetics/EOS reconstruction,
so subtracting these exact transport-state fluxes leaves generation,
dissolution, conversion-volume, thermal, and pressure effects in the physical
material source rather than advecting them retroactively. The per-cell
material-volume rate and liquid-carrier target are

$$
Q_{mat,c}=\frac{W_c^{n+1}-W_c^n}{\Delta t}
+\sum_{f\in c}\Phi_{carrier,f}
+\sum_{f\in c}\Phi_{slip,f},
$$

$$
Q_{V,c}=Q_{mat,c}-\sum_{f\in c}\Phi_{slip,f}.
$$

Thus bubble slip changes the material-volume ledger without being imposed as
liquid-carrier divergence. Internal owner/neighbour face terms cancel
pairwise. The global acceptance check is

$$
\sum_cQ_{mat,c}=
\frac{V_{pool}^{n+1}-V_{pool}^n}{\Delta t}
+\dot V_{b,escape}+\dot V_{other,out},
$$

within the configured physical volume tolerances. `volumeSourceRate`,
`bubbleSlipVolumeRate`, and `continuityTarget` publish the accepted cellwise
terms; trial values remain private.

### Moving top and accepted-trial sequence

The configured top batch must exist globally, be coplanar, face outward along
the selected gravity axis, and have area and enclosed mesh volume consistent
with the constant-area vessel map. Its liquid kinematic condition is imposed
exactly as

$$
\phi_{abs,f}=\phi_{m,f},\qquad \phi_{rel,f}=0,
$$

The trial velocity cache marks this patch as `Slip`, so its face value retains
the owner-cell tangential velocity and contributes no wall-normal diffusive
constraint. A configured `Slip` or zero-velocity `Dirichlet` marker is accepted
at setup for compatibility, but the active trial cache uses `Slip`. The exact
mesh-normal absolute flux is owned separately by the fixed-flux pressure
boundary, so tangential freedom cannot undo the kinematic condition. Bubble
slip may still carry H2 outward relative to that moving face. The zero-gauge
physical-pressure condition supplies the flat dynamic boundary.

One accepted ALE step proceeds as a single logical transaction:

1. Snapshot accepted flow/temperature fields, old absolute flux, material
   properties, liquid and optional H2 state, free-surface/headspace ledgers,
   diagnostics, time, and history position.
2. Start one geometry trial from the accepted surface to the current relaxed
   level candidate. Retain both volume states and the exact mesh flux.
3. Refresh every geometry-dependent equation, boundary, pressure/coupled
   numeric cache, preconditioner, and VTU point cache for the trial epoch.
4. Restore the accepted physical snapshot at each outer corrector, solve the
   selected pressure--velocity algorithm, and apply the bounded pressure-only
   refinements needed by the strict continuity gate. Then derive $\phi_{rel}$,
   solve the enabled equations, and preview the liquid, H2 escape,
   level/headspace, and $Q_V$ ledgers without publishing them.
5. Repeat until level, continuity-target, material-property, and gas-primary
   state changes satisfy their tolerances. Then require the actual trial mesh
   volume to match the closure-implied pool volume and require mesh quality,
   GCL, pressure continuity, pool/source, liquid mass, H2 inventory, and
   adiabatic sensible-energy closure.
6. Commit rollback-capable ledgers and publications exactly once, finalize time
   and one history record while the geometry trial is still reversible, and
   accept the geometry last.

Any failed solve or acceptance gate rolls back the active geometry, restores
all supported field/model snapshots, refreshes accepted-epoch caches, and
leaves time, history, and transfer watermarks unchanged.

## Flat Database Configuration

The parser follows the repository's flat underscore-separated `Database`
convention. All dimensional values are SI. Keys marked "required when used"
have no meaningful default.

### Selection and vessel map

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_enabled` | `false` | boolean | Disabled construction returns no model and preserves existing behavior. |
| `free_surface_model` | `fixed` | `fixed`, `planarVolumeBudget`, `planarALE` | `fixed` is the disabled sentinel and is rejected when `free_surface_enabled=true`; `planarVolumeBudget` is fixed-grid, while `planarALE` is accepted only for the support matrix below. |
| `free_surface_gravity_axis` | `z` | `x`, `y`, `z` | Planar elevation/motion axis; cylindrical and semi-structured ALE are axial-Z only. |
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
| `free_surface_liquid_volume_model` | `globalConstantMass` | `globalConstantMass`, `cellMassInventory` | On a fixed grid, cellwise mode retains its kg/m3 fixed-reference-volume interpretation. In `planarALE`, the same output field is a current-control-volume mass density whose conserved product is `V m_l^*`; it requires `error` depletion plus zero physical-boundary liquid flux. |
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

### Planar ALE controls

These keys are parsed in every mode but are active only for `planarALE`.

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_ale_top_boundary` | empty | boundary name | Required; the only moving liquid/headspace patch and, with Sheng H2, the only escape patch. |
| `free_surface_ale_deformation_start_elevation` | unset | m | When unset, deform the full column from the reference bottom; otherwise keep points at and below this elevation fixed. A configured value must be at least the reference bottom and strictly below the reference top. |
| `free_surface_ale_maximum_level_change` | unset | m | Optional positive accepted-to-trial displacement limit. |
| `free_surface_ale_gcl_absolute_tolerance` | `1e-12` | m3/s | Nonnegative cellwise absolute GCL tolerance. |
| `free_surface_ale_gcl_relative_tolerance` | `1e-10` | dimensionless | Nonnegative cellwise relative GCL tolerance. |
| `free_surface_ale_maximum_correctors` | `12` | integer | Positive limit used independently for the outer geometry/physics/source Picard trials and for strict pressure-only continuity refinements within each outer trial. Enabled Sheng H2 requires at least two outer trials. |
| `free_surface_ale_level_absolute_tolerance` | `1e-10` | m | Nonnegative absolute level-corrector tolerance. |
| `free_surface_ale_level_relative_tolerance` | `1e-10` | dimensionless | Nonnegative tolerance scaled by the level/elevation magnitude. |
| `free_surface_ale_relaxation` | `0.5` | dimensionless | Outer level/source relaxation in `(0, 1]`. |
| `free_surface_ale_maximum_growth_ratio` | `5` | dimensionless | Finite mesh-quality limit >= 1. |
| `free_surface_ale_maximum_non_orthogonality_degrees` | `75` | degrees | Finite nonnegative mesh-quality limit. |
| `free_surface_ale_maximum_skewness` | `4` | dimensionless | Nonnegative mesh-quality limit. |
| `free_surface_ale_maximum_aspect_ratio` | `1e6` | dimensionless | Finite mesh-quality limit >= 1. |

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

A minimal ALE selection, in addition to a matching dimensional Boussinesq
problem and material configuration, is conceptually:

```text
free_surface_enabled = true
free_surface_model = planarALE
free_surface_gravity_axis = z
free_surface_vessel_model = constantArea
free_surface_bottom_elevation = 0.0
free_surface_top_elevation = 1.2
free_surface_cross_section_area = 0.5
free_surface_initial_clear_level = 0.8
free_surface_liquid_volume_model = cellMassInventory
free_surface_overflow_policy = error
free_surface_headspace_model = vented
free_surface_ambient_pressure = 101325.0
free_surface_ale_top_boundary = top
```

The initial pool level, actual top-patch elevation, total current mesh volume,
and constant-area vessel map must already agree within the configured closure
tolerances. Configuration alone does not create or trim a liquid domain. The
physical boundary set supplies zero-gauge Dirichlet pressure and either a
`Slip` or zero-velocity `Dirichlet` marker on the named top; during each ALE
trial that velocity marker is replaced by `Slip`, while the pressure solver's
fixed-flux boundary supplies the exact mesh-normal volume flux.

### Enabled and rejected ALE combinations

| Concern | Initially supported | Rejected until separately migrated and tested |
| --- | --- | --- |
| Geometry | Mutable native Cartesian X/Y/Z, serial/MPI; mutable native cylindrical axial-Z, serial/MPI; mutable native `SemiStructuredXY_Z` axial-Z, serial only | Const-only handles, legacy mesh compatibility, general unstructured/partitioned-unstructured/STK motion, non-axial cylindrical or semi-structured motion, topology change, remeshing, repartitioning |
| Vessel/range | `constantArea`; `error` range and depletion policies | Tabulated vessel ALE and `clampAndReport` |
| Time/flow | Backward Euler; laminar dimensional Boussinesq; SIMPLE, PISO, PIMPLE, or coupled Krylov using the common target; gravity zero or inward along the top axis | BDF2/other moving-volume histories, transverse/outward gravity, and RANS/wall-distance/wall-law paths |
| Thermal/material | Physical temperature transport with accepted/trial $V m_l^*c_pT$ storage; nonzero `BoussinesqTemperatureOnly` pure-liquid expansion; adiabatic boundaries; rollback-safe static volumetric sources | Legacy nondimensional temperature, solids/conjugate heat transfer, nonadiabatic boundaries, pressure/composition material laws, dynamic material or source callbacks |
| Liquid | `cellMassInventory`; every nonmoving patch has a closed velocity condition and absent or homogeneous-Neumann pressure, giving zero physical-boundary liquid flux | `globalConstantMass`, liquid inlet/outlet composition, separate solvent/solute/fissile inventories |
| Gas/void | No gas model, or `sheng2024TwoPopulation` with a uniform geometry-invariant fission-power source, advective dissolved H2, general bubble transport/slip, constant or reconstructed absolute pressure, and exactly the moving top as escape patch | Sheng without fission power, Gaussian/spatial fission profiles, `idealGasSource`, axial compatibility transport, prescribed/inertial pressure, scalar-void-only evolution, any unowned gas volume |
| Headspace/phase change | Fixed-temperature vented headspace with prescribed positive ambient absolute pressure | Closed/restricted headspace, non-fixed headspace-temperature policies, boiling, steam/condensation transport or escape |
| Other transported physics | None beyond the supported liquid, temperature, momentum, and optional H2 equations | Delayed-neutron precursors and any optional extensive equation without ALE storage/flux/cache/rollback tests |

Setup and step preflight validate the applicable choices collectively. An
unsupported mixture fails before rank-divergent assembly rather than running
some equations on new volumes and others on fixed volumes.

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

`BoussinesqSolver::planar_ale_diagnostics()` adds the latest accepted old/new
geometry epochs and mesh volumes, actual mesh-to-closure-pool mismatch, maximum absolute and
normalized GCL residual, mesh-quality metrics, outer-corrector count, level and
pressure, material-state, and gas-primary-state residuals, sensible-energy and liquid/H2 inventory residuals, the
material-volume/continuity diagnostics above, and cumulative rejected-
transaction count plus the last rejection reason. These are acceptance and
debugging diagnostics; they are not measurements of physical free-surface
accuracy.

The same accepted diagnostic snapshot retains
`level_residual_history`, `target_change_history`,
`continuity_maximum_history`, `material_state_residual_history`, and
`gas_state_residual_history`, with one entry per completed outer Picard trial.
It is copied into the corresponding accepted `free_surface_history()` record;
a rejected attempt restores the preceding accepted histories and appends no
record. The CSV writer reports accepted final diagnostics rather than
flattening these variable-length sequences.

GCL and level convergence use the configured absolute-plus-relative ALE
tolerances. Target change and generalized-continuity maximum residual use
`free_surface_volume_closure_absolute_tolerance / dt` plus the configured
relative tolerance scaled by `max(1 m3/s, integrated-target scale)`. A
succeeding Picard trial must also change material coefficients and gas primary
inventories by no more than `1e-10` in the reported scale-normalized maximum.
Actual trial mesh volume versus closure-implied pool volume uses the configured
volume absolute-plus-relative tolerance with `max(1 m3, mesh/pool-volume
scale)`. The adiabatic sensible-energy gate applies to
$\sum_c V_cm_{l,c}^*c_{p,c}T_c$ and uses
`4096 * machine epsilon + 1e-9` relative to the largest of 1 J, old energy, new
energy, and added volumetric heat. Liquid mass, H2 inventory, and pool/source
closure retain their model-specific physical tolerances. None of these
conservation gates is relaxed when a Krylov solve is configured with a looser
algebraic tolerance.

The source-to-pool diagnostic is a rate in m3/s, so its absolute term is
`free_surface_volume_closure_absolute_tolerance / dt`; the relative term uses
the reported rate scale. Moving-patch geometry checks do not compare unlike
units: the same configured volume tolerance is divided by the global patch
area for coplanarity and by the vessel height for cross-sectional-area
agreement before the relative term is added.

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
`liquidMassInventory`. In `planarALE` mode it also writes the cellwise rates
`meshVolumeRate = (V_new-V_old)/dt`, `volumeSourceRate = Q_material`,
`bubbleSlipVolumeRate = -div(Phi_b,slip)`, `continuityTarget = Q_V`, and
`continuityResidual = sum(phi_abs)-Q_V`, all as signed cell rates in m3/s.
Their underlying face fluxes use owner-normal signs. The default preserves the
existing output schema. `BoussinesqSolver::free_surface_history()` automatically retains
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

The geometry-motion tests cover Cartesian motion on every axis,
cylindrical and semi-structured axial motion, buffered/repeated motion,
transaction accept/rollback, quality rejection, exclusive shared-geometry
ownership, unchanged topology/IDs/maps/boundary tags, per-cell GCL, MPI
partition-face swept-flux agreement, and collective target/timestep/action
validation. Geometry-cache tests verify stale epoch rejection and analytic
coefficient/gradient recovery after explicit refresh.

Additional ALE-focused tests exercise stationary identity against the fixed
operator, expansion/contraction constant-field preservation, old/new scalar
storage and liquid-mass-weighted temperature energy, conservative round trips,
BDF2 and stale/wrong-descriptor rejection, and the equation-layer momentum and
physical-temperature forwarding seams. Pressure tests exercise nonzero
integrated targets, PISO reuse, bounded pressure-only refinement,
fixed-boundary flux ownership, coupled/segregated residual agreement, and
generation/epoch cache invalidation. Moving-boundary and material-volume tests
check planar/area/vessel validation,
$\phi_{abs}=\phi_m$, zero carrier-relative top flux, thermal expansion, bubble
slip/escape subtraction, moving-top tangential-flow preservation,
mesh-relative Courant evaluation, and strict pool/source closure. Solver-level
tests retain the disabled-path baseline and
exercise collective setup rejection, accepted residual histories, and
accepted/rejected ALE transaction behavior on the supported serial/MPI paths.

Exact build and test commands and their results belong in the change report;
this modeling note describes coverage rather than freezing a machine-specific
test count.

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

`planar_ale_verification` is the separate solver-integrated driver. It runs the
native mutable mesh, pressure--velocity coupling, old/new-volume transports,
volume-source ledger, moving boundary, and whole-step transaction, and prints
the accepted GCL, continuity, liquid-mass, gas, mesh/pool, source/pool, and
energy residuals:

```bash
./build/gcc/bin/Debug/planar_ale_verification
./build/gcc/bin/Debug/planar_ale_verification aleUniformHeating
./build/gcc/bin/Debug/planar_ale_verification aleGasGeneration
./build/gcc/bin/Debug/planar_ale_verification aleCompleteEscape
./build/gcc/bin/Debug/planar_ale_verification aleFailureRollback
```

The four named `verification;ale;conservation` CTests check the analytic
constant-mass thermal level, uniform-power H2 production, isolated repeated-
step near-complete micro/large-bubble decrement and exact-once vent transfer
with a falling level, and restoration
plus safe retry after forced outer nonconvergence. These remain deterministic
conservation checks, not quantitative validation of real surface dynamics.
The escape case starts from known nonzero micro- and large-bubble inventories,
suppresses conversion and dissolution kinetics, uses zero fission/thermal
source, and repeats stable implicit high-slip steps until the remaining bubble
moles are no more than `1e-18 mol + 1e-10` times the initial bubble inventory.
It checks each population decrement, cumulative and vent transfer, and the
accepted pool/raw-bubble-volume drop independently.
Per-step level monotonicity allows roundoff of eight machine epsilons times
the initial level, since the last bubble-volume decrements are smaller than
the level's floating-point resolution. Raw bubble volume must still decrease
on every step, and both an individual and the net level drop must exceed that
roundoff allowance.

General level, volume, and species checks use an absolute-plus-relative
tolerance of `5e-12`. The closed-pressure check uses `5e-7 Pa + 2e-12`
relative, its $pV-nRT$ check uses `2e-6 J`, and its nonlinear pressure residual
limit is `5e-8 Pa`. A representative all-case GCC Debug run reported maximum
absolute volume closure `4.440892099e-16 m3`, maximum absolute gas closure
`0 mol`, and closed nonlinear residual `2.910383046e-11 Pa`.

These are analytic component/conservation verifications, not a physical
validation of pool dynamics. The ALE regressions verify discrete algebra,
boundary ownership, transactional restoration, and conservation for the
constrained matrix; they do not validate physical surface trajectories,
circulation, or radiolytic-bubble experiments.

The paired [`planarALE` OpenFOAM verification cases](../../verification/openfoam/planarALE/README.md)
add a transient uniform-heating expansion and a source-off steady equilibrium.
`planar_ale_comparison` runs the production SimpleFluid ALE solver; the
OpenFOAM reference independently solves the corresponding reduced uniform
thermal-expansion problem using moving-mesh finite volumes. The comparison
checks matched physical times, temperature, geometry, mass, energy, and
conservation residuals. Its scope is the uniform expansion limit; it does not
extend the supported physics matrix or validate general free-surface momentum.
The water cases require the optional IF97 library for the reference density,
heat capacity, viscosity, conductivity, and thermal expansion at 300 K and
101325 Pa absolute. They retain constant reference coefficients and the
built-in linear thermal density law. The energy contract remains `M cp T`;
neither nonlinear IF97 enthalpy transport nor a dynamic ALE material callback
is enabled by this verification update.

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

## Constrained ALE Limitations and Deferred Mapping

`PlanarALEMeshMotion` remains the geometry authority. It owns reference
coordinates, accepted/trial surface elevation, old/new local cell volumes, and
owner-oriented swept-volume face rates. Every trial checks

$$
E_{\mathrm{GCL},c}=
\frac{V_c^{n+1}-V_c^n}{\Delta t}-\sum_f\phi_{m,f}
$$

and the distributed mesh-quality gate before the solver may accept it.
Rejected trials restore coordinates; topology, global IDs, Tpetra maps,
boundary tags, and partitioning do not change. A shared geometry epoch and
exclusive controller lease make the revision visible through alias handles.

The solver-integrated scope is exactly the matrix in the configuration section.
It should not be generalized by analogy: geometry-only support in
`PlanarALEMeshMotion` does not make an optional transport ALE-safe, and
fixed-grid support for an optional model does not permit it in `planarALE`.

In particular, the current ALE path does not:

- resolve a nonplanar interface, curvature, capillary waves, splash, foam,
  breakup, sloshing, or overflow hydrodynamics;
- perform topology changes, remeshing, repartitioning, or arbitrary
  unstructured deformation;
- provide composition-resolved solvent, solute, or fissile transport, or claim
  composition-preserving evaporation or boundary flow;
- transport steam conservatively or transfer steam across the moving surface;
- support closed-headspace pressure coupling, pressure-dependent boiling
  saturation, RANS/wall laws, precursor transport, solids, or another optional
  extensive equation outside the listed matrix;
- make the cell-centre `poolOccupancy` indicator a cut-cell volume fraction or
  use it to modify the ALE equations; or
- provide conservative cross-mesh feedback mapping.

Milestone C therefore remains open. Conservative pool occupancy needs
geometric cut fractions or a mapper whose approximation error is part of its
acceptance contract. Cross-mesh feedback must separately preserve
liquid/fissile mass, gas species, precursor inventory, and energy and report
mapping residuals across supported MPI partitions. Same-topology ALE
conservation does not satisfy that requirement.

Full VOF/interface capturing and Euler--Euler gas momentum remain separate
deferred model families. The present conservation tests establish the discrete
contract for a flat moving boundary; they do not constitute quantitative
physical validation.
