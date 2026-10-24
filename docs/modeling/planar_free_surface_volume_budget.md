# Planar Free-Surface Volume Budget and Constrained ALE

The [species/source contract](reacting_gas_sources.md) and
[species-bound surface adapter](../architecture/reacting_species_surface.md)
define the general inventory path. Enabled gas setup selects
`ReactingGasMode::Species` and supplies a catalog/mechanism, finite per-species
inventories, named population counts, mixture EOS and independent source/heat laws.
There is no preferred H2/inert slot, heat-to-gas yield default or inferred donor.

This guide distinguishes four thermal/geometry owners:

| Path | Owned approximation and boundary |
| --- | --- |
| Weak supplied-EOS `planarVolumeBudget` | Fixed CFD geometry; species-derived liquid/gas volumes close the reported level and prescribed-temperature headspace. Boussinesq sensible-temperature storage remains the thermal owner. |
| Weak supplied-EOS `planarALE` | The same inventory owner plus constrained planar motion, old/new-volume storage, mesh-relative transport and an atomic supported ALE transaction. It is not local compressible phase-occupancy closure. |
| Fixed-grid finite-solvent/vapor `cp*T` | Explicit `BoussReferenceVolume`/coordinated sources own finite solvent, gas/vapor, admitted latent/sensible exchange and fixed reference-density scalar storage. This path cannot be combined with the weak supplied-EOS free-surface/ALE adapter or an independent boiling/phase-U owner. |
| Native low-Mach phase-U | `LowMachMixtureSolver` and its external `ReactingMixtureCoupler` own conservative phase internal energy, EOS/material-pressure recovery and configured local interphase transfer. Their closed-mesh/prescribed-motion contract is separate from this surface controller. |

The constrained surface ALE path remains narrow: mutable native supported
geometry, a constant-area vessel, Backward Euler, laminar dimensional Boussinesq
temperature transport, the declared pure-liquid expansion/feedback configuration,
`cellMassInventory`, vented headspace and no gas or the supported schema-owned
solution/population transports. It is a flat moving boundary, not interface
capture. Numerical conservation and solver-contract checks do not physically
validate pool trajectories or an application chemistry. VoF and bounded
floating-point/exact-rational VoF/general-AMR geometry remain deferred.

## Optional ALE relaxation acceleration

`PlanarALEOptions::aitken` defaults to `false`. The fixed `relaxation` factor
(default 0.5) also supplies the fallback when acceleration is enabled.
`initial_relaxation` defaults to 1.0, `minimum_relaxation` to 0.1 and
`maximum_relaxation` to 1.0. Factors must be finite and in `(0,1]`, bounds must
be ordered, and enabled initial/fallback factors must lie within those bounds.

The accelerated iterate is the owned-cell continuity-target vector. Residuals
use the fixed scale `dt/accepted_cell_volume` and accepted-cell-volume weights;
collective dot products count each owned cell once. The global integral of the
relaxed target determines the free-surface geometry. Valid vector-secant factors
are bounded; invalid or degenerate estimates or residual-norm growth above two
use the fallback and retain the current observation as the new secant baseline.
History persists across a substep's internal correctors and starts afresh for
each new substep or outer replay. Unrelaxed convergence checks,
volume continuity and physical acceptance retain their existing definitions.

The parameter-reader keys are `free_surface_ale_aitken`,
`free_surface_ale_relaxation`, `free_surface_ale_initial_relaxation`,
`free_surface_ale_minimum_relaxation` and `free_surface_ale_maximum_relaxation`.

external host exposes these settings as `Coupling/ALEAitken`, `ALERelaxation`,
`ALEInitialRelaxation`, `ALEMinimumRelaxation` and `ALEMaximumRelaxation`.
Its outer `Coupling/Aitken` and inner radiation acceleration are independent.

## Applicability and State Ownership

The model is intended for a single planar pool surface whose vessel geometry
can be represented by a monotone volume map

$$
\mathcal V(h)=\text{vessel volume below elevation }h.
$$

Ownership is deliberately split along existing model boundaries:

- `ReactingGasModel` owns catalog-keyed Liquid, Dissolved, Resolved and named
  Dispersed material and explicit population counts. Liquid and Dissolved share
  one physical solution mass/volume. Accepted source and physical escape events
  remain distinct; EOS volume is not reconstructed from an aggregate gas slot.
- For Species coupling, `LiquidMassInventory` is a cellwise reporting follower
  of that already transported solution mass and pure-liquid EOS density. It
  consumes signed source/outflow budgets and performs no second transport.
  The no-species reference path separately retains global constant mass or its
  own conservative cellwise liquid-mass equation.
- `BoilingSourceModel` remains a separate standalone/reference phase-change
  owner. Its lumped steam/collapse model cannot coexist with the Species EOS
  surface path. The finite-solvent/vapor `cp*T` adapter is another explicitly
  bounded fixed-grid owner, not an extra sink to add to this surface closure.
- `VentedHeadspaceModel` or an explicitly supplied catalog-bound closed closure
  owns accepted escaped Resolved/Dispersed species after one committed transfer.
  `ClosedMixtureHeadspaceModel` supplies the prescribed-temperature mixture-EOS
  pressure closure; no unnamed air or inferred typed initial inventory is added.
- `PlanarFreeSurfaceModel` owns accepted levels, the planar volume/headspace
  closure and its reporting snapshot. It consumes events; it does not rerun
  chemistry, population kinetics, material transport or heat integration.

`MultiphysicsSolverCoupler` owns the optional models, controller/motion state and
coupled advancement and retains shared ownership of its `BoussinesqSolver`.
The solver owns its core equations, fields, mesh and clock. The caller's native
`MeshHandle`/`FieldStored` identity is preserved, as is legacy `Mesh` identity on
the supported fixed-grid compatibility path; neither is silently converted into
the other. Mutable solver-integrated ALE requires the native path.

Normal typed ordering advances conservative material transport and supplied
chemistry/population events before temperature, applies their accepted sensible
thermal source once, then refreshes thermodynamics without evaluating sources
again. The surface preview consumes that accepted event, follows actual solution
mass, solves level/headspace closure and commits exactly its pending physical
escape. Constant/reconstructed thermodynamic pressure is refreshed from the
closed result before publication. Heat is independent of reaction extent; a
supplied application law may read a prescribed driver, but must declare its
material reservoir and energy instead of using a library yield fallback.

Fixed-grid surface feedback remains one-way lagged. Flow keeps its dynamic-pressure
role; current source/transport uses the previously accepted thermodynamic offset,
and the newly closed offset informs the next step. No same-step outer flow/level
solve is claimed. A surface/headspace preview is non-mutating. Enabled typed
Species fixed-grid steps snapshot flow, temperature, material, species/count state, surface/follower
accounting, accepted receipts and histories, and restore them atomically on
rejection. The nontransactional no-species reference path can still fail-stop
after upstream advancement; reconstruct its complete accepted solver state
rather than replaying a partially advanced step. The constrained ALE path also
restores its complete supported state together with geometry on rejection.

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

Liquid volume must use the actual bubble-free solution density
$\rho_l(T,p,\mathbf x)$, never `rhoFeedback` or another void-reduced hydrodynamic
density. Otherwise bubble displacement is counted through both density and gas
volume.

### Species-bound solution follower

For the general path, the catalog's Liquid and Dissolved amounts define

$$
M_{l,c}=\sum_i W_i(n_{i,\mathrm{Liquid},c}+n_{i,\mathrm{Dissolved},c}),
\qquad V_{l,c}=M_{l,c}/\rho_l(T_c,p_c,\mathbf x_{l,c}).
$$

`ReactingGasModel::liquid_mass_density()` stores this mass per current control
volume and `pure_liquid_density()` supplies its EOS density.
`LiquidMassInventory::initializeFromSpecies` binds these authoritative fields;
`previewSpeciesMass` follows their already transported trial endpoints and raw
signed solution source/boundary budgets. Chemical, external and interphase mass
changes are counted once. Raw signed packets, not differences of separately
rounded gross counters, retain a small net change. The follower never performs
an additional solvent/solute transport or re-applies a source.

Species contexts explicitly use `ReactionVolumeClosure::ReferenceControlVolume`.
The actual EOS liquid volume sets liquid-based reaction rates; the geometric CV
sets bulk-based rates/heat. Their values may differ, including EOS liquid volume
larger than the prescribed CV. The adapter does not clamp that measure or claim
local compressible occupancy. Supported native phase-U flow has a different
owner which must close its material volume before acceptance.

For Species surface setup, select `cellMassInventory` with `error` depletion.
A configured initial fill/mass must agree with actual initialized species volume;
it cannot overwrite finite composition. No guessed gas-equivalent donor replaces
missing solvent or product material. The weak Boussinesq thermal equation still
owns its stated sensible-temperature approximation, not independent phase energy.

### No-species reference inventory

The separate no-species path evaluates pure-liquid density through
`MaterialFeedbackModel::pure_liquid_density` and publishes `rhoLiquid` in kg/m3.
That reference material evaluator covers the configured constant,
Boussinesq-temperature, and configured pure-liquid branch of mixture feedback.
It does not yet supply composition-dependent density or a liquid
compressibility law. The planar model accepts a pressure-dependent liquid
volume callback so those laws can be added without changing its ownership.

The no-species `globalConstantMass` approximation retains total liquid mass $M_l$ and fixed
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
+\sum_{f\in c}\phi_{\mathrm{rel},f}m_{l,f}^{*,\mathrm{upwind},n+1}
=V_c^{n+1}\left(\dot m_{\mathrm{cond},c}
-\dot m_{\mathrm{evap},c}\right),
$$

where $\phi_{\mathrm{rel},f}=\phi_{\mathrm{abs},f}-\phi_{m,f}$. The first ALE support matrix
rejects boiling, so the right-hand side is zero there; retaining the source form
states the contract that boiling must satisfy before that rejection can be
relaxed. A constant intensive inventory is preserved under pure mesh motion by
the cellwise GCL, while the extensive liquid mass is evaluated with
$V_c^{n+1}$ only after the trial is accepted.

Before returning a cellwise preview, liquid transport checks the owned-cell
balance using old-volume mass, new-volume phase-change amounts, and trial mass.
If the initial linear solve leaves a defect above `64 * machine epsilon` of
the transported mass scale, at most two residual corrections solve the same
matrix equation with the configured solver policy and reused preconditioner.
The refinement bound includes a `64 * denorm_min` floor for subnormal totals.
This prevents small same-sign solve errors from accumulating against the
fixed initial-mass gate. Neither mass normalization nor clipping is applied.
Both existing physical mass gates still run, and only then are trial ghosts
published. Reported iterations include the corrections; the reported relative
residual uses the original right-hand side. Failure diagnostics print masses,
residuals, and tolerances in scientific notation.

Within a planar ALE corrector, temperature first consumes a conservative mass
preview. Subsequent material feedback changes pure-liquid density, which affects
derived liquid volume but does not enter that mass transport equation.
`LiquidMassInventory::refreshCellwiseDensity()` takes a currently valid cellwise
preview, updates pure density and recomputes its trial volume from the same
transported mass. It returns a new token without another solve. The original
token becomes stale even when density is unchanged; an ordinary density update
cannot revive an older token through this API. Owner, generation, nonce,
geometry epoch and any live ALE descriptor are checked before reuse. Mass and
phase ledgers retain the first solve's unchanged conservation gates, while the
derived volume uses the new density and trial-new control volumes. Accepted
mass is still published only by `commitPhaseChange()`.

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
tolerance. This standalone inventory is effective total liquid mass, not separate
solvent and reactive-solute stocks. The Species path above owns those finite
components, but does not thereby permit boiling or composition-preserving
evaporation in this surface/ALE combination.

## Gas Volume, Standalone Boiling, and Accepted Escape

Only gas assigned to submerged Resolved/Dispersed compartments contributes to
$V_b$. Dissolved species remain in the full species/solution ledger but do not
occupy gas volume. A `Resolved` inventory label in this field adapter is not a
resolved free interface or a VoF qualification. The supplied gas-mixture EOS uses
each actual composition. Named dispersed bins additionally use their explicit
counts and the configured bounded radius/capillary closure:

$$
p_b=p_{l,\mathrm{abs}}+\frac{2\sigma}{R_b},\qquad
V_b=\sum_{c,a\in\mathrm{submerged\ gas}}V_{a,c}(\mathbf n_a,T_c,p_a).
$$

The radius/EOS provider, bounds and derivatives remain authoritative; the surface
closure contains no second ideal-gas, inertial-pressure or Laplace formula.
`global_submerged_bubble_volume()` integrates the physical EOS volume.
Candidate-pressure queries retain actual accepted T/composition/counts and are
non-mutating. Derived void fields are reports, not a way to destroy a finite
inventory. Generic overcapacity rejects the physical trial; any optional admission
law must explicitly own its accepted extent and retained external demand. A raw
or unrepresented-volume observer cannot authorize hidden material clipping.

### Exact accepted transport events

`physical_escape_events()` retains actual accepted owner-oriented transport
packets keyed by species, phase and population. Their gross outflow is not
recovered by subtracting large before/after inventories or cumulative totals.
Only accepted physical Resolved/Dispersed escape is eligible for the
vent/headspace. This field backend's Resolved bucket is stationary; changing its
volume creates no physical escape packet. Named dispersed populations provide
the active carrier/slip escape. Dissolved
boundary outflow is a separate solution/gas-subset sink. External reservoir
withdrawal is not physical boundary escape.

`species_event_receipt()` binds the accepted interval, schema, initialization
generation, revision and publication nonce. The controller consumes exactly the
next event once. Initialization and thermodynamic refresh create no new event.
`population_escape_receipt()` separately retains actual per-bin outward counts
in configured population order and shares that accepted event identity; counts
are not inferred from escaped moles. The lumped headspace owns escaped species,
not a transported bubble-number phase.

Snapshots retain physical events and their consumption watermark. Restore and
re-advance cannot impersonate an already consumed event because publication
nonces do not rewind. Tiny positive accepted packets remain represented or the
required endpoint rejects; finite-precision local/MPI reporting is not an
arbitrary exact-arithmetic claim. Supplied carried-enthalpy and carrier-momentum
escape traces remain traces: this Boussinesq adapter has no owned phase-U or
independent bubble-momentum field to debit a second time.

With ALE, every inventory uses accepted-old/trial-new CVs. Solution advection uses
the mesh-relative carrier; each named dispersed species and its count use the
same carrier-plus-slip donor/flux. On the moving top, only physical transport
relative to that face is escape. Swept mesh volume is not escaped material.
The exact accepted packets are transferred once after all supported gates pass.

### Separate boiling approximations

The retained standalone `BoilingSourceModel` uses an explicit fixed-saturation,
scalar-void/latent approximation. In its no-Species fixed-grid surface mode,
accepted `phaseChangeMassRate = latentHeatSink / latent_heat` removes liquid;
rejected vapor demand removes neither mass nor latent energy and remains in
`rejectedVaporMassRate`. `complete_void_fraction_update()` may assign realizable
collapse only to steam already tracked at step start, return that condensate and
its matching declared latent heat, and report non-steam collapse separately.
Newly accepted evaporation cannot explain disappearance of an old unrelated gas.
This globally conservative lumped steam ownership does not provide a local steam
species transport or a steam escape event. Without the explicit completion owner,
the standalone default does not invent condensate heat.

That reference mode cannot combine with the supplied-EOS Species surface adapter.
It is vented/fixed-grid only for active fixed-saturation boiling: closed headspace
requires a pressure-dependent saturation contract, and surface ALE rejects its
steam/boiling transport. Retained nonzero steam also prevents removing/replacing
its free-surface owner without an explicit disposition.

The distinct fixed-grid finite-solvent/vapor `cp*T` adapter instead transfers one
actual catalogued solvent between declared liquid/vapor compartments, reserves
joint finite donors/capacity and applies accepted latent/sensible energy once to
its fixed reference-density thermal owner. It cannot be combined with this weak
EOS free surface/ALE, an independent standalone boiling sink, or native phase-U
thermal ownership. Native [pressure-conditioned interphase transfer](../architecture/reacting_interphase_thermal.md)
has a separate conservative phase-energy/pressure-work contract; neither native
coverage nor standalone boiling tests qualify the other approximation.

## Absolute Pressure and Headspace Models

All EOS and headspace states use Pa absolute. Dynamic/gauge pressure remains a
momentum variable. `set_absolute_pressure_offset()` supports Constant and
Reconstructed ownership and preserves the latter's volume-weighted gauge
variation. PrescribedHistory is a separate supplied pressure owner and cannot
accept an externally closed offset here. The retired specialized inertial mode
is not an active alternative. Nonzero scalar void without a finite gas owner is
rejected.

For `planarALE`, the vent owns thermodynamic pressure $p_h$, while CFD owns $p'$.
The moving top requires zero-gauge physical-pressure Dirichlet data. Pressure
correction enforces its exact absolute kinematic flux through correction-Neumann
data. Since reconstruction subtracts the gauge mean, the internal offset is
$p_h+\overline{p'}$:

$$
p_{l,\mathrm{abs},c}=(p_h+\overline{p'})+p'_c-\overline{p'}=p_h+p'_c.
$$

At the zero-gauge top the absolute liquid pressure is $p_h$. No pressure is
omitted or added twice. The planar surface has no curvature jump; this does not
resolve capillary surface shape or supply capillary energy conservation.

`vented` fixes $p_h=p_{\mathrm{ambient}}$ and accumulates accepted escaped species
in `ventedMoles`; headspace moles remain empty. `closed` is supported by the
fixed-grid Species surface only through an explicit supplied prescribed-temperature
mixture closure, such as `ClosedMixtureHeadspaceModel`. It satisfies

$$
V_h=V_{\mathrm{vessel,total}}-V_l(p_h)-V_b(p_h),\qquad
\mathcal V_{\mathrm{EOS}}(\mathbf n_{h,0}+\mathbf n_{\mathrm{escape}},T_h,p_h)=V_h.
$$

An ideal supplied mixture reduces this to $p_hV_h=n_hRT_h$; general nonideal
headspace volume is evaluated by its supplied EOS. Positive bounded pressure,
positive available headspace volume, explicit finite amounts and converged
closure are required. The surface's safeguarded pressure root respects both its
configured bracket and supplied EOS/coupled-volume domains. Under-relaxation
changes a pressure iterate, never an inventory.

Use `configure_free_surface(options, headspace)` to inject a fresh closure with
matching catalog fingerprint, nonempty identity, positive revision and finite
T/p validity bounds. `options.headspace.initial_moles` names actual allowed
catalog species; set `infer_initial_moles=false`. No unnamed `gas`, inferred air
or extra background inventory is supplied for typed closed coupling. The
injected dynamic type is preserved, but clone-based checkpoint/restore may replace
the object with an independent equivalent clone; pointer identity after restore
is not promised.

Headspace callbacks are MPI-free. The controller fences local exceptions and
validates replicated metadata, bounds, inputs and candidate outputs before ranks
can diverge in a root branch, including empty owners. Fixed, prescribed or supplied
representative liquid temperatures remain pressure-closure policies, not a
headspace energy equation. Supported ALE uses fixed vent temperature or its
explicit typed external-temperature participant. Closed/adiabatic headspace
energy and `restrictedVent`/orifice dynamics are not provided by this adapter.

`ClosedIdealGasHeadspaceModel`, constant $Z$ and initial-mole inference remain
explicit standalone component/reference capabilities. They are not defaults for
a catalog-bound Species closed headspace. Their analytic tests establish that
reference closure, not general reacting-mixture or adiabatic-headspace support.

## Conservative Planar ALE Contract

All face-volume fluxes use m3/s and the mesh owner normal. The existing
projected field remains the absolute liquid flux

$$
\phi_{\mathrm{abs},f}=\int_f\mathbf u\cdot\mathbf n_o\,dA.
$$

`PlanarALEMeshMotion` supplies the exact swept-volume rate $\phi_{m,f}$ for the
same face and timestep. ALE convection uses a separate synchronized field

$$
\phi_{\mathrm{rel},f}=\phi_{\mathrm{abs},f}-\phi_{m,f}.
$$

The absolute field is never overwritten with relative values. Pressure,
kinematic-boundary enforcement, and continuity diagnostics consume
$\phi_{\mathrm{abs}}$; momentum, temperature, liquid inventory, enabled species/count transports,
and the ALE Courant diagnostic consume $\phi_{\mathrm{rel}}$. Specifically,

$$
Co_c=\frac{\Delta t}{2V_c}\sum_{f\in c}|\phi_{\mathrm{transport},f}|,
\qquad
\phi_{\mathrm{transport}}=\begin{cases}
\phi_{\mathrm{abs}},&\text{fixed mesh},\\
\phi_{\mathrm{rel}},&\text{planar ALE}.
\end{cases}
$$

`ALEControlVolumeState` carries
accepted-old and active-trial-new volumes in mesh-local cell order, mesh fluxes
in mesh-local face order, timestep, concrete geometry identity, and old/new
epochs. It is valid only while the originating motion trial is active and
validates sizes, finiteness, positivity, identity, epoch, and the cellwise GCL.
Repeating a trial with the same target and time step still changes its geometry
epoch, so a state retained from the earlier trial must be rebuilt.

The generic `make_ale_control_volume_state` factory returns a borrowed view and
performs full validation on every call, including checks after external
callbacks. `make_validated_planar_ale_control_volume_state` accepts only the
concrete final `PlanarALEMeshMotion` controller. It copies the old/new volumes
and swept fluxes into shared immutable storage and validates those copied
values before enabling reuse. Boussinesq's planar solver uses this snapshot
for each new geometry trial; copies retain ownership of the same data.

Snapshot consumers still collectively check mesh identity, active trial,
dimensions, static constituents, old/new geometry epochs, timestep and
tolerances. Only the immutable-data scan and full cellwise GCL walk are reused.
Every rank participates in selecting the path: a borrowed view on any rank
forces full validation on all ranks. Acceptance, rollback, restoration and
new trials invalidate retained snapshots through active-state/epoch checks.
The motion controller must still outlive its views. This is not an epoch-only
cache of mutable spans, and it does not change material/provider validation.

The ownership member changes the C++ ALE state layout. FVM, Solvers, affected
consumers and their tests must be rebuilt against matching headers and
libraries; the material C ABI is unchanged.

Validation combines rank-local state failures before any rank throws. The GCL
pass similarly combines invalid volume or mesh-flux data, mesh traversal
failures, and cellwise residual failures, so all ranks receive the same exception
category and message. Mesh execution is held only while local faces are read;
the lease is released before the collective result is checked. Rank-local
traversal failures are reported collectively.

For a conservative intensive unknown $q$ with storage weight $a$, advective
weight $b$, and volumetric source $S$, the Backward-Euler assembly is

$$
\frac{V_c^{n+1}a_c^{n+1}q_c^{n+1}
      -V_c^na_c^nq_c^n}{\Delta t}
+\sum_{f\in c}\phi_{\mathrm{rel},f}(bq)_f^{\mathrm{upwind},n+1}
=\mathcal D_c^{n+1}+V_c^{n+1}S_c^{n+1}.
$$

This weak Boussinesq path uses unit momentum storage and conservative bulk-CV
densities for dispersed moles/counts. Solution transport uses its concentration
unknown with the frozen liquid fraction as storage/advection weight, then
publishes mol/current-CV; it is not a second solution-mass equation. The
no-species liquid reference equation has unit inventory storage. Physical
temperature instead uses the conservative cellwise liquid-mass density $m_{l,c}^*$ and heat capacity $c_{p,c}$:

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
\sum_{f\in c}\phi_{\mathrm{abs},f}-Q_{V,c}=0.
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

For Species coupling, let $\Phi_{l,f}^{\mathrm{tr}}$ be
`transported_liquid_material_volume_flux()`: the canonical relative face-volume
packet captured from the same implicit post-transport/pre-source Liquid/Dissolved
molar fluxes, frozen transport void fraction and supplied EOS. It evaluates the
actual donor composition and T/p, not a final-cell mixture density. Its complete
`liquid_transport_volume_receipt()` must match the accepted event and active
geometry epoch; a stale or partial-carrier packet cannot drive the target.

Each gas population retains its own post-transport/pre-source raw fraction with
the upwind donor selected by $\phi_{\mathrm{rel}}+\phi_{\mathrm{slip},a}$, exactly as in
that population's species/count equations. The carrier and slip material packets
are

$$
\Phi_{\mathrm{carrier},f}=\Phi_{l,f}^{\mathrm{tr}}
 +\sum_a\phi_{\mathrm{rel},f}
 \alpha_{b,a,f}^{\mathrm{tr},\mathrm{up}(\phi_{\mathrm{rel}}+\phi_{\mathrm{slip},a})},
$$

$$
\Phi_{\mathrm{slip},f}=\sum_a\phi_{\mathrm{slip},a,f}
 \alpha_{b,a,f}^{\mathrm{tr},\mathrm{up}(\phi_{\mathrm{rel}}+\phi_{\mathrm{slip},a})}.
$$

The owning transport may publish the already combined exact carrier/slip packets.
The field backend's Resolved compartment is presently stationary: it uses zero
carrier and conservative ALE storage, with no inferred number/slip moment or
resolved-interface transport. No
post-source fraction is retroactively used as an advected label, and face
fractions are not normalized to conceal counterflow. The no-species reference
inventory uses its documented implicit upwind mass/density fallback; it is not
a replacement for the Species receipt.

The final $W^{n+1}$ is evaluated after local sources/EOS reconstruction,
so subtracting these exact transport-state fluxes leaves generation,
dissolution, conversion-volume, thermal, and pressure effects in the physical
material source rather than advecting them retroactively. The per-cell
material-volume rate and liquid-carrier target are

$$
Q_{\mathrm{mat},c}=\frac{W_c^{n+1}-W_c^n}{\Delta t}
+\sum_{f\in c}\Phi_{\mathrm{carrier},f}
+\sum_{f\in c}\Phi_{\mathrm{slip},f},
$$

$$
Q_{V,c}=Q_{\mathrm{mat},c}-\sum_{f\in c}\Phi_{\mathrm{slip},f}+Q_{\mathrm{mesh},c},
$$

$$
Q_{\mathrm{mesh},c}=\frac{V_c^n}{V_{\mathrm{mesh}}^n}
\frac{V_{\mathrm{pool}}^n-V_{\mathrm{mesh}}^n}{\Delta t}.
$$

The mesh correction removes the accepted global pool/mesh volume defect over
the current timestep, distributed in proportion to accepted cell volumes.
It is fixed throughout all outer trials of a step and recomputed after every accepted
step or checkpoint restore, including when the timestep changes. It changes
the continuity constraint and swept mesh volume together; it does not change
liquid mass, gas inventories, energy, or the physical material source.
`VolumeContinuityModel` accepts this signed global correction rate explicitly;
its default is zero for standalone material-ledger use.

Thus bubble slip changes the material-volume ledger without being imposed as
liquid-carrier divergence. Internal owner/neighbour face terms cancel
pairwise. The global acceptance check is

$$
\sum_cQ_{\mathrm{mat},c}=
\frac{V_{\mathrm{pool}}^{n+1}-V_{\mathrm{pool}}^n}{\Delta t}
+\dot V_{b,\mathrm{escape}}+\dot V_{\mathrm{other},\mathrm{out}},
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
\phi_{\mathrm{abs},f}=\phi_{m,f},\qquad \phi_{\mathrm{rel},f}=0,
$$

The trial velocity cache marks this patch as `Slip`, so its face value retains
the owner-cell tangential velocity and contributes no wall-normal diffusive
constraint. A configured `Slip` or zero-velocity `Dirichlet` marker is accepted
at setup for compatibility, but the active trial cache uses `Slip`. The exact
mesh-normal absolute flux is owned separately by the fixed-flux pressure
boundary, so tangential freedom cannot undo the kinematic condition. Bubble
slip may still carry gas species and counts outward relative to that moving face. The zero-gauge
physical-pressure condition supplies the flat dynamic boundary.

One accepted ALE step proceeds as a single logical transaction:

1. Snapshot accepted flow/temperature fields, old absolute flux, material
   properties, solution species and optional population/count state, free-surface/headspace ledgers,
   diagnostics, time, and history position.
2. Start one geometry trial from the accepted mesh height, deriving its new
   height from the relaxed global volume-source target and top area. Retain
   both volume states and the exact mesh flux.
3. Refresh every geometry-dependent equation, boundary, pressure/coupled
   numeric cache, preconditioner, and VTU point cache for the trial epoch.
4. Restore the accepted physical snapshot at each outer corrector, solve the
   selected pressure--velocity algorithm, and apply the bounded pressure-only
   refinements needed by the strict continuity gate. Then derive $\phi_{\mathrm{rel}}$,
   solve the enabled equations, and preview the liquid follower, accepted species/count escape,
   level/headspace, and $Q_V$ ledgers without publishing them.
5. Repeat until level, continuity-target, material-property, and gas-primary
   state changes satisfy their tolerances. Then require the actual trial mesh
   volume to match the closure-implied pool volume and require mesh quality,
   GCL, pressure continuity, pool/source, solution mass, every species/count budget, and
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
| `free_surface_liquid_volume_model` | `globalConstantMass` for no-species reference | `globalConstantMass`, `cellMassInventory` | Species surface requires `cellMassInventory/error` as a follower of already transported solution mass. No-species cellwise mode owns its own kg/m3 transport; ALE always uses current-CV extensive storage. |
| `free_surface_initial_liquid_mass` | not set | kg | Finite and nonnegative. |
| `free_surface_initial_liquid_volume` | not set | m3 | Finite, nonnegative, and within the vessel policy range. |
| `free_surface_initial_clear_level` | not set | m | Converted through the vessel map. |

At least one of initial mass, volume, or clear level is required when the
volume-budget model is enabled; the model never assumes a completely full
vessel. If both volume and level are provided, they must agree through the map
within the setup consistency tolerance. If mass is omitted, the liquid
no-species inventory infers it from the selected initial volume and the global
volume-weighted pure-liquid density. A Species follower instead retains actual
catalogued mass and checks the configured fill against its EOS volume. If mass and a volume/level are both
provided, `MultiphysicsSolverCoupler` verifies that the initialized pure-density volume
is consistent rather than silently choosing one.

### Planar ALE controls

These keys are parsed in every mode but are active only for `planarALE`.

| Key | Default | Unit / allowed values | Notes |
| --- | --- | --- | --- |
| `free_surface_ale_top_boundary` | empty | boundary name | Required; the only moving liquid/headspace patch and, with Species populations, the only gas escape patch. |
| `free_surface_ale_deformation_start_elevation` | unset | m | When unset, deform the full column from the reference bottom; otherwise keep points at and below this elevation fixed. A configured value must be at least the reference bottom and strictly below the reference top. |
| `free_surface_ale_maximum_level_change` | unset | m | Optional positive accepted-to-trial displacement limit. |
| `free_surface_ale_gcl_absolute_tolerance` | `1e-12` | m3/s | Nonnegative cellwise absolute GCL tolerance. |
| `free_surface_ale_gcl_relative_tolerance` | `1e-10` | dimensionless | Nonnegative cellwise relative GCL tolerance. |
| `free_surface_ale_maximum_correctors` | `12` | integer | Positive limit used independently for the outer geometry/physics/source Picard trials and for strict pressure-only continuity refinements within each outer trial. Enabled Species gas requires at least two outer trials. |
| `free_surface_ale_level_absolute_tolerance` | `1e-10` | m | Nonnegative absolute level-corrector tolerance. |
| `free_surface_ale_level_relative_tolerance` | `1e-10` | dimensionless | Nonnegative tolerance scaled by the level/elevation magnitude. |
| `free_surface_ale_relaxation` | `0.5` | dimensionless | Outer source relaxation in `(0, 1]`; trial height follows that same relaxed integrated source. |
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
| `free_surface_headspace_temperature_model` | `fixed` | `fixed`, `prescribed`, `bulkLiquid`; typed `external` | Temperature is supplied, not a headspace energy solve. ALE admits fixed vent T or a configured typed external-temperature participant. |
| `free_surface_initial_temperature` | `293.15` | K | Positive; retained by `fixed`. |
| `free_surface_headspace_temperature_times` | empty | s array | Required by `prescribed`; finite and strictly increasing. |
| `free_surface_headspace_temperature_values` | empty | K array | Required by `prescribed`; same nonzero length as times and all positive. |
| `free_surface_headspace_gas_constant` | `8.31446261815324` | J/(mol K) | Standalone ideal reference only; supplied mixture EOS owns its gas law. |
| `free_surface_headspace_compressibility_factor` | `1.0` | dimensionless | Standalone ideal reference only; no constant-$Z$ fallback for a nonideal typed headspace. |
| `free_surface_initial_headspace_moles` | not set | mol | Optional unnamed `gas`/inferred inventory belongs only to standalone ideal reference closure; rejected for typed Species closed coupling. |
| `free_surface_headspace_species` | empty | string array | Must accompany the mole array; names are unique actual catalog species for typed coupling. |
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

A minimal **no-species reference** fixed-grid, vented constant-area setup is
conceptually:

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

`MultiphysicsSolverCoupler` owns the optional models, ALE trials, and interval
checkpoints. Construct it with a shared solver pointer using
`MultiphysicsSolverCoupler<Pack> coupling(solver)` and use its configuration,
initialization, step, and output APIs; see the
[coupling ownership and transaction design](../architecture/multiphysics_solver_coupler.md).

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

### Explicit Species setup

Gas configuration is a supplied C++ schema, not a database heat-yield switch.
Conceptually, after constructing the physical solver and shared external coupler:

```cpp
ReactingGasOptions gas;
gas.mode = ReactingGasMode::Species;
gas.reaction_mechanism = mechanism;       // Immutable catalog and balanced reactions.
gas.liquid_thermodynamics = liquid_eos;   // Actual Liquid+Dissolved composition.
gas.gas_thermodynamics = gas_eos;         // Composition, fugacity/domain and caloric convention.
gas.population_ids = population_names;
gas.initial_species_state = finite_initial_material;
gas.initial_population_numbers = finite_initial_counts;
gas.species_properties = per_species_properties;
gas.species_source = independent_reactions_and_reservoirs;
gas.population_process = supplied_paired_material_and_number_events;
gas.solution_transport = ReactingTransportMode::Advective;
gas.bubble_transport = BubbleTransportMode::General;
gas.pressure_mode = ReactingPressureMode::Reconstructed;
gas.free_surface_patches = {moving_top_name};
coupling.configure_reacting_gas(gas);
```

The names above denote application-supplied laws/values, not built-in defaults.
Every species has a supplied molar mass and legal phase membership; populated bins
have explicit finite compatible N and gas. The source context includes actual
T/p/volumes and canonical `ReactionLocation` (geometry ID, current epoch, centroid,
ephemeral owned row), so spatial laws need no rank-local hidden species convention.
Callbacks are immutable/reentrant and MPI-free. Collective metadata and local
failure fencing include empty owners before later communication. Source extents
and heat carry their declared volume/time basis exactly once.

Configure Species before binding the surface follower, then initialize through
the external coupler. For typed closed fixed-grid operation, additionally pass a
fresh `ClosedMixtureHeadspaceModel` with the matching catalog gas EOS, provider
identity/revision and actual named initial headspace amounts to
`configure_free_surface(surface, std::move(headspace))`. Set inference off; the
surface's configured fill must match actual material. A database cannot synthesize
those objects. Do not configure the finite-vapor reference owner or an independent
boiling owner on this supplied-EOS surface path.

### Enabled and rejected ALE combinations

| Concern | Supported constrained surface contract | Rejected until separately owned and qualified |
| --- | --- | --- |
| Geometry | Mutable native Cartesian X/Y/Z; cylindrical and `SemiStructuredXY_Z` axial Z, serial/MPI; compatible compact multi-region common affine Z | Const-only/legacy motion, general unstructured deformation, nonaxial cylindrical/XY-Z motion, topology change/remeshing/repartitioning |
| Layout/identity | Original native mesh/provider identity, distributed owned/halo maps and canonical owner-normal faces; XY graph partitions crossed with Z slabs and actual empty owners participate | Conversion to replicated legacy mesh; treating contiguous algebraic GIDs as geometry ordinals; stale map/epoch/face packets |
| Vessel/range | Constant area; error range/depletion; actual initial mesh/pool/map agreement | Tabulated moving vessel, clamp-and-report motion, invented initial material to fill a mesh |
| Time/flow | Backward Euler, laminar dimensional Boussinesq; SIMPLE/PISO/PIMPLE or supported coupled Krylov with the same target; zero/inward axial gravity | BDF2 moving history, transverse/outward gravity, RANS/wall-law ALE or other unqualified extensive equations |
| Thermal/material | Conservative accepted/trial solution-mass `cp*T` storage; declared nonzero pure-liquid Boussinesq-temperature feedback; supplied liquid EOS for actual composition volume; adiabatic boundaries or an explicit typed thermal participant with immutable candidate heat-flux snapshots | Native phase-U ownership through this controller, arbitrary dynamic material/temperature callbacks, solids or unowned nonadiabatic boundaries |
| Liquid | Species: complete Advective Liquid+Dissolved transport and `cellMassInventory/error` follower; no-species: its conservative cellwise reference mass transport | Second transport of followed solution mass, selectively frozen solution species, liquid inlet/outlet composition or an unowned phase change |
| Gas/source | No gas, or explicit Species EOS/material/count laws; General carrier-plus-slip transport, Constant/Reconstructed pressure, exactly the moving top as escape patch; at least two outer trials | Null-mechanism/yield fallback, retired kinetics/inertial/axial selectors, prescribed-pressure conflict, scalar-void-only gas or duplicate material source |
| Heat input | Independent supplied source heat with the supported snapshot contract; uniform geometry-invariant prescribed heat or explicit extensive interval energy | Treating a heat budget as automatic gas amount; unrefreshed spatial heat profiles or callbacks without rollback ownership |
| Headspace/phase change | Vented positive ambient absolute pressure; fixed T or explicit typed external-T participant | Closed/restricted vent ALE, adiabatic headspace-energy claim, standalone boiling/steam transport, fixed-grid finite-vapor reference adapter |

Geometry support is not qualification of every row combination. See
[XY/Z distribution](../semi_structured_parallel.md) and
[compact region contracts](../region_meshes.md). Setup and step preflight validate
choices collectively, including ranks with no cells/faces; no rank may continue
with a different model or geometry epoch.

A `ThermalStepParticipant` is an explicit external thermal transaction, not an
arbitrary pointwise source callback. It owns all external thermal inventories,
immutable candidate outward boundary-flux snapshots, restore/begin-substep,
residual and commit state, and an optional cavity temperature for External mode.
Its methods must be collective-safe and fence their own local stages before
internal collectives. Candidate fluxes use the current geometry and the liquid
energy gate includes their signed outward energy exactly once. Ordinary
unowned nonadiabatic/dynamic callbacks remain rejected; this participant does not
turn the liquid sensible-temperature equation into native phase-U storage.

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
- gross generated/consumed and separately accumulated signed net source, dissolved,
  submerged, just-escaped, documented-sink, headspace and vented moles by species;
- per-species gas-inventory closure residuals in mol, their normalized values,
  and the summed/max-normalized gas residuals.

For each species the general gas-subset ledger is

$$
E_{g,s}=n_{s,0}+\Delta n_{s,\mathrm{source,signed}}
 -n_{s,\mathrm{dissolved}}-n_{s,\mathrm{submerged}}
 -n_{s,\mathrm{headspace}}-n_{s,\mathrm{vented}}
 -n_{s,\mathrm{other\ sinks}}.
$$

The signed source comes directly from actual chemical/external/interphase events;
separately rounded gross production and consumption are provenance, not a way to
reconstruct a small net source. Gas-subset and solution-subset membership overlap
at Dissolved and have distinct source/outflow budgets. The surface consumes only
its accepted physical escape subset, never a rounded cumulative difference.

`LiquidMassInventoryDiagnostics` separately reports initial/current mass,
cumulative accepted evaporation and condensation, dry-out mass deficit,
mass-weighted specific volume, liquid volume, and absolute/normalized
accepted-history and per-step mass residuals. `BoilingPhaseChangeDiagnostics` separates requested, accepted, and
rejected evaporation; steam and non-steam collapse; condensate; submerged
steam; and phase-change residuals. The boiling output registry exposes
`phaseChangeMassRate`, `condensationMassRate`, `rejectedVaporMassRate`, and
`condensationLatentHeatRelease` alongside the existing `S_alpha_boil` and
`latentHeatSink` fields in the standalone boiling path. Species histories and
physical/count escape receipts are schema keyed; raw EOS void is a derived
physical-volume observer, not a hidden inventory after clipping.

`MultiphysicsSolverCoupler::planar_ale_diagnostics()` adds the latest accepted old/new
geometry epochs and mesh volumes, actual mesh-to-closure-pool mismatch, maximum absolute and
normalized GCL residual, mesh-quality metrics, outer-corrector count, level and
pressure/material and full-schema gas-state/thermal-participant residuals,
sensible-energy and solution/species inventory residuals, the
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
succeeding Picard trial must also change material coefficients and the complete
schema gas/source state by no more than `1e-10` in the reported normalized maximum.
`gas_state_residual_history` and the observer's `blocking_mask` are the
authoritative typed Picard/Aitken gate. It covers actual species/count state,
applied source and physical molar escape; the retired five-field H2 slots are
zero/not applicable. Passive enthalpy/momentum traces are finite-checked but do not
create an extra nonlinear phase-energy gate. The subtractive
`material_volume_source()` report likewise cannot replace the exact material
face-packet/level/continuity acceptance gates.
The first gas comparison uses the accepted old-time state as its reference;
later comparisons use the preceding new-time trial. Thus stationary gas can
converge on the first pass, while a physical time increment above tolerance
requires a subsequent trial comparison. The accepted gas reference is captured
again after checkpoint restore or a timestep change.
Each trial starts from the actual accepted mesh height and derives its height
from the same target supplied to pressure:
`h_trial = h_mesh_old + dt * sum_owned(target) / top_area`.
This relation applies to both the next-step predictor and relaxed outer
correctors. The thermodynamic pool-level diagnostic can differ from mesh
coordinates within its tolerance. The continuity target includes the accepted
volume defect correction above, so geometry and pressure use the same rate.
Without that correction, the converged update preserves the old defect and
accepted iteration residuals can accumulate. With it, the next defect is set
by the current iteration and physical-ledger residuals instead of inheriting
the previous defect. The predictor uses the previous physical source and slip
with the newly computed correction, avoiding reuse of the old step's correction.
Pressure retains its global compatibility check for each trial, and the physical
ledger retains the existing nonlinear acceptance tolerances. Accepted diagnostics
report `global_mesh_volume_correction` separately from `global_material_source`;
the ALE history CSV includes `ale_mesh_volume_correction_m3_per_s`.

Actual trial mesh volume versus closure-implied pool volume uses the configured
volume budget also for the level check when
`free_surface_ale_level_tolerance_from_volume` is enabled: the level tolerance
is the same absolute-plus-relative volume tolerance divided by the moving top
area. This policy applies at initial configuration and during every trial.
Explicit level tolerances remain available when the policy is disabled.

The pool/mesh volume check uses the configured
volume absolute-plus-relative tolerance with `max(1 m3, mesh/pool-volume
scale)`. The supported sensible-energy gate applies to
$\sum_c V_cm_{l,c}^*c_{p,c}T_c$ and uses
`4096 * machine epsilon + 1e-9` relative to the largest of 1 J, old energy, new
energy, added volumetric heat and explicitly owned outward boundary energy.
Its signed balance is `E_new - E_old - E_added + E_boundary,out`; an installed
thermal participant also has its own convergence/commit gate. These are the weak
Boussinesq sensible-energy
gates, not native phase-U or arbitrary caloric-reference qualification. Solution
mass, per-species/count budgets and pool/source
closure retain their model-specific physical tolerances. None of these
conservation gates is relaxed when a Krylov solve is configured with a looser
algebraic tolerance.

Each general schema transport checks the candidate inventory against accepted-old
inventory and physical boundary outflow before publishing fields. If the
initial linear solve misses the existing `64 * machine epsilon` inventory
balance, at most two residual corrections solve the same transport equation
with its existing preconditioner. The unchanged inventory gate must pass before
fields and escape ledgers are updated. This is relevant for consecutive weakly
heated intervals whose retained species amount is small but nonzero.
The new residual check has a `64 * denorm_min` absolute floor for subnormal
roundoff; the existing escape and donor-inventory gates are unchanged.

The source-to-pool diagnostic is a rate in m3/s, so its absolute term is
`free_surface_volume_closure_absolute_tolerance / dt`; the relative term uses
the reported rate scale. Moving-patch geometry checks do not compare unlike
units: the same configured volume tolerance is divided by the global patch
area for coplanarity and by the vessel height for cross-sectional-area
agreement before the relative term is added.

The feedback-field registry has stable optional names `rhoLiquid`,
`clearLevel`, `poolLevel`, `headspacePressure`, and `poolOccupancy`, and can
collectively require that set without changing the older standard-field
contract. `MultiphysicsSolverCoupler` publishes `rhoLiquid` cellwise and replicates the
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
Their underlying face fluxes use canonical owner-normal signs. Native XY/Z and
other decomposed meshes separate canonical geometry IDs from rank-contiguous
algebraic field IDs; reports and callbacks retain the correct map/geometry frame.
`MultiphysicsSolverCoupler::free_surface_history()` retains initialization and one
record per accepted step without repeating reductions. Catalogued CSV history is
keyed by actual species/population and records event identity, signed/gross source,
separate outflow, levels, headspace and closure residuals with SI units and the
prescribed-temperature pressure-only headspace contract. Standalone/no-gas
reference CSV has a separate schema; its old placeholder gas columns are not
Species observers. The standalone steam model owns mass/volume rather than an
inferred molar catalog, so steam moles must not be manufactured for its output.

## Verification Scope

Source coverage and current local execution are different evidence. Focused tests
exercise map/inverse/range behavior, liquid reference transport, species-bound
mass following, raw signed source accounting, exact physical escape/event
consumption, supplied closed-mixture metadata/EOS failure, independent film and
count budgets, and atomic surface/ALE failure/replay. Native and legacy fixed-grid
identity are separate tests. Distributed geometry/operator/solver cases include
XY graph times Z slab ownership, shared-face signs, compact topology, real empty
owners and collective failures; registered two- and four-rank groups must actually
run before claiming that configuration passed.

ALE component/integration tests cover old/new-volume storage, relative convection,
mesh/epoch/map/GCL validation, volume-source/continuity and fixed-flux boundary
ownership, moving-top tangential flow, conservative temperature storage,
Picard/Aitken full-schema gates, cache refresh and complete rollback. This is not
physical validation of reacting pools, a turbulent two-fluid model or an arbitrary
combination of independently tested components. Exact commands, ranks, skips and
fresh results belong in the change report/build artifacts rather than historical
machine-specific maxima in this modeling contract.

### Retained analytic reference drivers

`planar_free_surface_verification` intentionally remains five standalone
component/reference cases, not a typed reacting-inventory qualification:

```bash
./build/gcc/bin/Debug/planar_free_surface_verification
./build/gcc/bin/Debug/planar_free_surface_verification uniformHeating
./build/gcc/bin/Debug/planar_free_surface_verification gasGeneration
./build/gcc/bin/Debug/planar_free_surface_verification completeEscape
./build/gcc/bin/Debug/planar_free_surface_verification closedHeadspace
./build/gcc/bin/Debug/planar_free_surface_verification boilingMassLoss
```

Their reference purposes remain reciprocal-density thermal expansion,
ideal/reference gas and steam displacement, complete prescribed escape to a vent,
a named H2/air ideal-gas/Laplace quadratic, and clear-level loss from an accepted
18 kg liquid withdrawal. These named reference materials and scalar events are
explicit manufactured inputs, not active H2/yield defaults for Species. The last
case synthesizes no steam molar transport/escape. General level/volume/species
reference checks retain `5e-12` absolute-plus-relative tolerance; the quadratic
check retains `5e-7 Pa + 2e-12` relative pressure, `2e-6 J` in $pV-nRT$, and
`5e-8 Pa` nonlinear residual. These reference tolerances do not replace the
separate general source/EOS acceptance contract.

The production-path `planar_ale_verification` retains its four named modes:

```bash
./build/gcc/bin/Debug/planar_ale_verification
./build/gcc/bin/Debug/planar_ale_verification aleUniformHeating
./build/gcc/bin/Debug/planar_ale_verification aleGasGeneration
./build/gcc/bin/Debug/planar_ale_verification aleCompleteEscape
./build/gcc/bin/Debug/planar_ale_verification aleFailureRollback
```

The migrated gas cases supply a finite two-gas catalog, actual solution mass,
named external reservoir feed and explicit incoming `cp*T*dm` law under their
weak approximation. That heat is supplied physical provenance, not internal
compensation for a failed budget or a gas-yield default. The original analytic
thermal level, source integration, near-complete independent bin escape/vent,
falling-level and forced-nonconvergence replay oracles remain. The escape case
starts from known finite bins, disables local production/conversion/dissolution
and checks actual accepted transport and exactly-once transfer until retained
moles are below `1e-18 mol + 1e-10` of initial. Level monotonicity retains its
8-epsilon initial-level roundoff allowance, while raw bubble volume must actually
decrease and net/individual level drops exceed that allowance. Source migration
alone is not fresh runtime evidence for these drivers.

These gas cases and the ALE mode of `coupling_kernel_benchmark` use
[`FiniteGasVerificationSupport.hh`](../../src/examples/FiniteGasVerificationSupport.hh).
Its manufactured Gibbs-consistent liquid EOS preserves the reference
`rho(T) = rho_ref * (1 - alpha * (T - T_ref))` exactly along the selected constant
`p_ref` path. The separate off-reference compressibility and caloric completion
have positive volume and caloric response over the declared EOS bounds. They do
not replace Boussinesq `M*cp*T` storage or the explicitly supplied material,
escape and feed energy laws, and introduce no compensating heat. This weak
verification closure does not qualify native phase-U energy recovery or a
physically calibrated liquid EOS.

The paired [planarALE OpenFOAM cases](../../verification/openfoam/planarALE/README.md)
compare uniform transient thermal expansion and source-off equilibrium against
an independently solved moving-mesh reduced reference. They match physical times,
T, geometry, mass, sensible energy and conservation. Their water reference may
use optional IF97 properties at 300 K and 101325 Pa with constant coefficients and
the configured linear thermal-density law. The energy model remains `M*cp*T`,
not nonlinear IF97 enthalpy transport or dynamic ALE material callbacks. That
external comparison limit does not qualify general surface momentum, reacting
species, native phase-U or either vapor approximation.

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

The no-species cellwise reference fill is smeared over the modeled domain and
uses the single-continuum carrier flux. The Species path instead transports
actual Liquid/Dissolved composition and follows that mass, but its reported pool
level still does not mask the computational domain or move a fixed escape plane.
It closes global material volume under a weak Boussinesq approximation, not local
compressible phase occupancy, separate gas momentum or phase internal energy.
Species-bound mass alone does not authorize evaporation or a liquid inlet law;
those require their own compatible composition/energy/boundary contract.

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
  fragmentation, sloshing, or overflow hydrodynamics;
- perform topology changes, remeshing, repartitioning, or arbitrary
  unstructured deformation;
- claim composition-preserving evaporation, a general liquid inlet/outlet law,
  independent gas momentum or conservative phase-U ownership merely because
  finite solvent/solute/reactant advection is present;
- transport steam conservatively or transfer steam across the moving surface;
- support closed-headspace pressure coupling, pressure-dependent boiling
  saturation, RANS/wall laws, standalone scalar transport, solids, or another optional
  extensive equation outside the listed matrix;
- make the cell-centre `poolOccupancy` indicator a cut-cell volume fraction or
  use it to modify the ALE equations; or
- provide conservative cross-mesh feedback mapping.

Conservative geometric pool occupancy and feedback mapping remain open here. They need
geometric cut fractions or a mapper whose approximation error is part of its
acceptance contract. Cross-mesh feedback must separately preserve
liquid mass, reactive and gas species, and energy and report
mapping residuals across supported MPI partitions. Same-topology ALE
conservation does not satisfy that requirement.

VoF, its bounded floating-point geometry and exact-rational geometry routes,
and general AMR/regridding are deferred outside this refactor. The
separately scoped VoF notes are not an enabled reacting backend or an
extension of this ALE support matrix. Same-topology geometry/mapping utilities
retain their own lower-level contracts; they do not supply this coupled
species/energy feedback transaction. The present checks establish a flat-boundary
discrete conservation contract, not quantitative surface dynamics.

## External coupling intervals

The external `MultiphysicsSolverCoupler` retains its shared solver and owns the
supported laminar planar-ALE/PISO interval transaction. It exposes
`create_coupling_checkpoint()`, `restore_coupling_checkpoint(checkpoint)`,
`validate_coupling_checkpoint_acceptance(checkpoint)` and
`accept_coupling_checkpoint(checkpoint)`, collectively on the mesh communicator.
The move-only token belongs to that coupler, not to an independently advanced
solver. Only one checkpoint is live; destruction performs no MPI work and does
not undo state. Configure models, supplied laws and sources first. A topology,
ownership, callback or solver-configuration change invalidates the token; dt may
change for subcycling and is restored on retry.

Restoration recovers accepted geometry, fields, actual species/count histories,
source and surface consumption receipts, liquid/headspace ledgers, clock and
history. Geometry epochs, validation generations and publication allocators remain
monotonic, so old views do not revive when coordinates are restored. Refresh
numeric transport/pressure caches. No checkpoint operation writes output; publish
after acceptance. The const preflight repeats ownership/configuration, failed-state
and interval-end checks without reserving the candidate. An external controller
may preflight, commit its own source state, then accept without intervening
SimpleFluid mutation; acceptance repeats the checks.

`set_coupling_interval_energy(owned_joules, duration)` is an independent prescribed
thermal budget. An existing `PrescribedHeatSource` and the supported idle ALE
interval owner are required. The caller maps extensive joules before setting it;
on each candidate geometry the source uses
`q = E / (duration * current_volume)`. Stable cells retain those extensive budgets
through every substep/corrector. Out-of-interval advancement rejects before state
changes; replay consumes no extra energy. This remains a first-order,
uniform-in-time material-following approximation; pulse applications need temporal
refinement. Unrefreshed spatial heat profiles and arbitrary dynamic material/
temperature-source callbacks are not made ALE-safe by this interface.

General chemistry/population amounts are separate inputs. Rates use their actual
declared liquid/bulk volume and dt once; integrated extents are not multiplied
again. A named external reservoir law may deliberately read the current prescribed
field through the supplied `ReactionLocation` and define a dose-linked test or
application correlation. It must still return actual species, associated
mass/energy/momentum and explicit compatible count events. That correlation is not
a default yield, and it cannot invent a finite donor deficit to hide mass.
Independent heat is added once by its thermal owner; gas-driving data is not
another heat source. Coupled retries repeat pure event evaluation from accepted
stocks without mutating provider histories.

Use actual catalogued solvent, reactants, products and external reservoirs for
mass/element/source budgets. The removed H2-equivalent donor-deficit field and
five-slot aggregate histories are not a general conservation substitute.
[Source ownership](reacting_gas_sources.md),
[species surface receipts](../architecture/reacting_species_surface.md) and
[native phase-energy flow](../architecture/low_mach_reacting_flow.md) retain their
distinct owner and acceptance contracts.
