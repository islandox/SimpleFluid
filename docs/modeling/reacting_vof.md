# Reacting gas: resolved and dispersed coupling

**VoF disabled by default:** VoF solvers are unqualified and excluded from the
supported general multispecies refactor. `SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF=ON`
opts into source-free VoF development and its qualification fixtures. The
unmigrated `ReactingVoFCoupler` and fixtures that construct it remain excluded;
the reacting contracts below are retained design history, not enabled behavior.
Reusable lower-level FVM and geometry mathematics remain available.

The [general reacting-gas design](reacting_gas_sources.md) separates supplied
reaction rates, species stoichiometry, finite reactants, and net reaction
heat from gas transport and representation. Reactions add or remove species;
dissolution, capture, promotion, and demotion transfer existing material.
Every transfer must move its species, mass, momentum, and energy together.

The neutral `Reacting*`, `PrescribedHeat*`, and `BubblePopulationKinetics`
interfaces replace the retired application-specific names without aliases.
The runtime two-population selector is `twoPopulation`. Existing H2 fields
and the energy-yield source closure remain transitional; this migration does
not establish arbitrary-species transport or a general chemistry interface.
See the [migration table](reacting_gas_sources.md) for exact API and key changes.

## General extension contract

The planned reaction boundary supplies integrated species changes from a
declared reaction extent and stoichiometry, including each consumed solution
species and each generated gas species. Rate volume bases and the destination
of new material must be explicit. Finite reactants are budgeted jointly across
all reactions, and a trial that exhausts a reactant is rejected or recomputed
at a smaller physical timestep. External feed is separately accounted for.

Heat input is already net under a common chemical/caloric reference. The same
reaction event cannot contribute both a separate reaction heat and duplicate
heat deposition. Composition-aware EOS and interfacial laws must recover
phase volumes and partial pressures from conserved species. Source integration
then participates in the existing local volume, global pressure, energy,
transport, and transaction acceptance gates. None of these extensions permits
inventory clipping or weaker tolerances. The detailed contracts below describe
the H2 implementation from which those general interfaces must be developed.

## Current single-gas implementation: scope and ownership

The opt-in low-Mach mode of `AdaptiveVoFSolver` transports liquid mass,
resolved H2 and inert gas, dissolved H2, chemically bound donor stock,
dispersed bubble number and H2, physical momentum, and internal energy.
`ReactingVoFCoupler` owns a shared pointer to that solver and adds reaction,
interface-transfer, capture, and explicitly designated headspace laws. Its
trial and checkpoint operations include the controller's cumulative ledgers.

The supported geometry is the fixed Cartesian AMR mesh described in
[adaptive VoF](adaptive_vof.md). The original source-free solver remains a
separate initialization path. This coupling does not enable moving boundaries
or automatic headspace detection. Explicitly enabled
[representation trials](bubble_representation.md) add whole-bubble promotion
and complete-component demotion; their birth localization remains supplied
by the application.

## Current single-gas implementation: state, pressure, and volume

All conservative arrays use extensive amounts per owned cell. In a cell of
volume `V`, resolved gas occupies `f V`, dispersed gas occupies `alpha_d V`,
and true aqueous liquid occupies `alpha_l V`, where

```
f + alpha_d + alpha_l = 1.
```

The carrier of dissolved species and dispersed bubbles is
`(alpha_l+alpha_d) V`, mathematically `(1-f) V`. Its numerical storage uses
the actual phase volumes rather than subtracting nearly equal fractions.
Dissolved concentration for Henry equilibrium and kinetics is instead
`n_dissolved / (alpha_l V)`. These volumes must not be interchanged.
Additional passive `GasVolume`, `LiquidVolume` (carrier), and `BulkVolume`
inventories retain their geometric carriers. The legacy passive `MixtureMass`
carrier uses constructor densities and is rejected in low-Mach mode; use the
typed physical mass and species fields for EOS-dependent quantities.

The liquid caloric EOS is supplied through `VoF::LiquidThermodynamics`, with
explicit temperature and absolute-pressure bounds, positive density and
caloric derivative, and `h=e+p/rho`. `make_linear_compressible_liquid` is a
Gibbs-consistent local approximation using explicitly supplied coefficients;
it is not a correlation for a particular aqueous solution. Resolved H2 and
inert gas use supplied ideal-gas caloric properties. Dispersed populations
contain H2. There is one local temperature for all phases in a cell.

The pressure is a uniform thermodynamic offset `P0` plus a mechanical field.
For the volume-averaged mechanical pressure `pi` and reconstructed curvature
`kappa`, one-sided pressures are

```
p_l = P0 + pi - f sigma kappa,
p_r = P0 + pi + (1-f) sigma kappa.
```

The resolved EOS and Henry law use these one-sided pressures; they do not add
a second Laplace jump. A dispersed bubble uses `p_b=p_l+2 sigma/r`, with its
radius obtained from its moles, number, temperature, and gas EOS. The
mechanical pressure has a volume-mean-zero gauge, with the corresponding
offset retained in `P0`.

For every trial, a closed-domain pressure root closes the global material
volume, while the pressure projection and source-aware VoF transport close
each cell's material volume. The integrated phase increments are obtained
from endpoint EOS volumes plus the divergence of their actual transported
volumes. Relative dispersed volume flux is included once in the target for
the continuous velocity. Global pressure closure alone is insufficient.
Only a reported correction bounded by floating-point accumulation error is
permitted in the pressure null mode. Acceptance checks the residual against
the original physical target.

The pressure iteration includes the positive volume susceptibility
`C=-dV_material/dpi` at fixed conservative inventories and internal energy.
Its integrated pressure matrix is `A+C/dt^2`; the positive susceptibility
removes the constant-pressure null mode. Liquid density partials are evaluated
within the supplied EOS bounds, and Gibbs consistency supplies the caloric
pressure response. Resolved ideal gas and dispersed Laplace/EOS volumes use
the same local temperature response. Intermediate phase increments match this
linearized pressure solve, while acceptance uses the original nonlinear EOS
increments and continuity residual. This avoids a compressibility feedback
instability at small timesteps.

`VoF::PhaseVolumeIncrement` carries integrated resolved-gas and carrier
volume changes. Geometric transport verifies that their sum matches the
integrated face divergence. The directional compression indicator remains
frozen at the old fraction. The final physical correction is
`Delta V_g - chi (Delta V_g + Delta V_c)`; directional and final fraction
bounds remain acceptance gates. Inventory clipping and removal of a physical
mean expansion source are not permitted.
The combined source and swept-volume update must leave nonnegative phase
storage. A source/advection combination that exhausts a thin phase rejects
the trial. Transport timestep limits do not replace these reaction and
phase-volume admissibility checks; a retry with a smaller physical timestep
must also recompute the integrated source energies for that timestep.
The mass/species transport capacities are reconstructed independently from
their conservative phase-volume equations. A positive carrier smaller than
one ulp of the cell volume can then retain its material inventory even when
the stored resolved fraction rounds to one.

## Current single-gas implementation: mass, momentum, and thermal accounting

The donor field is finite chemically bound H2-equivalent stock within the
liquid mass. The implemented H2 production consumes this stock. For internal
chemical or interfacial exchange,

```
Delta m_l = W_H2 (Delta n_donor + Delta n_dissolved).
```

Thus donor-to-dissolved conversion leaves liquid mass unchanged,
donor/dissolved-to-gas conversion removes the corresponding mass from liquid,
and gas dissolution returns it. The legacy `donorH2Deficit` diagnostic is not
used as a liquid-mass model. Dilute dissolved diffusion uses a counterdiffusive
zero-net-mixture-mass convention and is excluded from the chemical liquid-mass
change. This dilute approximation omits the solute/solvent partial-enthalpy
diffusion term. Venting removes gas mass from the domain and returns no mass
to liquid.

If `u` is the continuous/liquid velocity and `w_k` the supplied relative
velocity of dispersed population `k`, physical cell momentum is

```
M = m_total u + sum_k(m_k w_k).
```

Transport uses the same canonical phase-volume fluxes for phase inventories
and their mass, energy, and momentum fluxes. Material-volume ledgers use the
recovered candidate density, temperature, phase pressures, and transported
number/molar traces. A dispersed slip-volume flux can remain nonzero when
carrier advection cancels the net bubble flux. Dispersed momentum travels with
`u+w_k`. Interface and capture exchange between cells has equal and opposite
momentum and enthalpy amounts; a transfer within one mixed cell cancels in
that cell. Pressure, capillarity, viscosity, and gravity act in the flow core.
For zero surface tension, periodic momentum acceptance includes a reported
forward-error bound from the actual assembled impulse scales and arithmetic
operation count, in
addition to the configured physical tolerance. It does not correct the
domain's momentum to conceal a residual.

Internal energy is inverted through the caloric EOS, transported by phase,
and coupled to conduction and phase pressure work. Thermal face fluxes use
the current nonlinear iterate. Acceptance checks recovered temperature,
advected liquid specific energy, and mixture conductivity against the values
used in those fluxes, as well as pressure-work consistency. For stationary
liquid with constant properties this converges to the backward-Euler discrete
heat equation; a numerical starting guess cannot select a different thermal
time level. The timestep still obeys the thermal iteration stability bound.
`net_thermal_energy` is the caller's already net heat deposition.
`reaction_driver_energy` supplies
the chemical yield and is never added to heat a second time. All caloric
references must be consistent. The caller owns chemical formation-energy and
closure heat conventions. The thermal ledger is not a claim of conservation
of kinetic plus surface plus chemical energy: subgrid coalescence/breakup
changes interfacial area, and its surface-energy release or cost requires an
application closure. Viscous conversion into heat and an independently evolved
surface-energy reservoir are not included.
Prescribed slip also needs an external relative-kinetic-energy/drag-heating
closure if that energy conversion matters; momentum conservation alone does
not supply the missing heat.

## Current single-gas implementation: source substeps without duplicate transport

`BubblePopulationKinetics` exposes pure cell kernels. The legacy
`ReactingGasModel` delegates its microbubble source, decay, and conversion
substep to that seam while keeping its existing transport and upper-population
behavior. The VoF controller never calls the legacy full `advance` method.

The current energy-yield closure produces `eta G E_abs` moles of H2 from
absorbed energy `E_abs` per cell and yield `G` in mol/J. This is a retained
specialization, not the general reaction-rate contract. Its input is already
energy absorbed by the liquid: no second liquid-volume fraction is applied.
The hybrid path uses explicit source substeps, including the existing
microbubble production-before-decay ordering.
Active micro kinetics use `LegacySubcycled`, with
`nsub=max(1,ceil(dt/(0.2 tau_micro)))`; exceeding `max_subcycles` rejects.
Bulk dissolved diffusion runs once, kinetics/film/capture run in each source
substep, and venting runs once at the end.
Every corrector replays from the same accepted inventories. Interface patches
and carrier apertures stay at the accepted geometry for this first-order
source/transport split; thermodynamic coefficients use the current trial.
The flow core subsequently transports each inventory once. This ordering
requires time refinement when source motion becomes appreciable.
Integrated reaction and interface sources use shared compensated event
ledgers, so a small transfer is not recovered by subtracting two large stored
inventories. Momentum-source convergence measures its velocity impulse in
units of cell width per timestep and uses the flow corrector's existing
accuracy; conservation checks still use the actual signed source amounts.

Upper populations use fixed pivots in **moles per bubble**. For pivot `xi_k`,
the canonical moment is `n_k=xi_k N_k`; transported number and H2 must agree
with that relation. Births and size changes are distributed between adjacent
pivots to retain both number and moles. Coalescence counts identical pairs
once. A physical collision kernel `K_ij` in m3/s acts on carrier-based number
density: with bulk densities `N_i`, its bulk event rate is
`K_ij N_i N_j / (alpha_l+sum_k(alpha_d,k))` (halved for identical pairs).
The kernel receives the actual EOS carrier fraction, including positive
carrier below the precision of `1-f`; callers must not divide by it a second
time.
Growth and breakup retain their per-bubble rates. Breakup checks the daughter
molar moment. Dissolution returns H2 to the
aqueous inventory. Below-grid amounts either reject or explicitly return to
the micro population; above-grid amounts reject. Exhausted number, dissolved,
gas, or chemical donors reject the trial without clipping.
When below-grid gas is returned to an active micro population, its number and
moles must also satisfy the inherited micro-conversion closure's domain. That
closure uses `M_to_upper=min(M_micro,n_nuc N_converted)`; both retained and
converted reservoirs must have paired positive number/moles or paired zeros.
An incompatible residual is not repaired by deleting number or gas.
Passive underflow return alone does not qualify simultaneous active conversion.

Growth, coalescence, and breakup rates are externally supplied callbacks with
a model name, validity description, and domain validator. Empty callbacks add
no rate. Property and rate callbacks are deterministic value evaluations;
uncheckpointed external history or random draws must not change during replay.
The legacy upper-bubble lifetime is not silently applied to the new
bins. Diffusivity, nucleation radius, surface tension, and liquid properties
likewise remain supplied material inputs. The implemented resolved force uses
constant surface tension; a nonconstant supplied value rejects until
Marangoni stresses are implemented.
The controller regenerates every population's drift, including microbubbles,
from one selected zero-slip, constant-slip, or Celata mode. Constant slip points
opposite the supplied gravity. Celata uses each population's actual radius and
rejects outside its experimental domain. A broad micro/upper spectrum is not
automatically qualified by that shared selection; the low-Mach core itself
accepts independently supplied drift fields for each population.

## Current single-gas implementation: resolved H2 transfer and the prescribed film

Unique interface patches include mixed-cell PLIC polygons and exactly aligned
pure-gas/pure-carrier faces, including coarse/fine and periodic faces. Sparse
sampling and scattering give each patch one owner and equal/opposite species
exchange. Dissolved bulk diffusion uses the intersection of both incident
carrier traces on a face, so disconnected liquid pieces do not exchange
through a spurious open area.

The implemented subgrid option is a restricted **prescribed stagnant-film**
closure. A physical thickness `delta` and a local-planarity/mixing/calibration
validator are required. Distance `s` is measured from the reconstructed
interface into its carrier region. The assumed profile is

```
C(s) = C_s + (C_b-C_s) min(s/delta, 1),
C_s = H p_H2,
ell = average_carrier[min(s,delta)],
dot n_H2 = A D (C_bar-C_s) / ell.
```

`C_bar` uses the true aqueous volume. The homogeneous dispersed-carrier
assumption makes the geometric carrier average applicable to that aqueous
concentration. `p_H2` is the resolved H2 partial pressure, including an inert
gas fraction; total headspace pressure cannot substitute for it. The sign of
the flux permits both growth and dissolution.

The geometry integrates `ell` over the actual carrier polyhedron. In a planar
layer of depth `L`, it gives `L/2` for `delta>=L`, and
`delta (1-delta/(2L))` for `delta<=L`. There is no additional mesh-distance
resistance in series with this closure, and the numerical timestep is not
interpreted as interface age. Transfer is explicit within source substeps;
donor exhaustion requires a smaller step.

This closure does not reconstruct and correct every neighboring transport
flux in the published boundary-layer SGS algorithms. Those algorithms also
correct bulk-face fluxes near the interface; see
[Schwarzmeier et al., section 3.3](https://arxiv.org/html/2505.17036v1#S3.SS3).
Qualification here is limited to the declared profile and supplied physical
film law. High-Peclet centimetric-bubble transfer still requires application
data, mesh/time refinement, and an appropriate boundary-layer model.

## Current single-gas implementation: capture, coalescence, and headspace

Let the patch normal point from resolved gas into carrier, and let `j_H2` be
the signed molecular H2 mass flux into resolved gas per area. The point-bubble
collision approach speed is

```
v_approach,k = max(0, j_H2/rho_l - w_k dot n).
```

It follows from the liquid-side interface speed and is unchanged by a common
translation. EOS-only compression contributes no molecular Stefan flux.
The collision rate is `A v_approach,k N_k/V_carrier`, multiplied by an
explicit efficiency in `[0,1]`. Only captured number/moles are removed;
uncaptured dispersed gas remains in the carrier. Radius/approach/temperature
validity and the efficiency are supplied by the application.

Capture efficiency and population kernels represent unresolved encounter and
film-drainage/rupture assumptions. There is no resolved lubrication-film or
rupture-time model. Resolved interfaces that contact at grid scale retain
ordinary geometric VoF coalescence; grid-dependent coalescence must be
qualified for the intended application.

Headspace is resolved gas in explicitly supplied spatial regions. Joining it
does not count as escape. Ownership uses accepted fully gas cell centroids
inside the supplied boxes and remains fixed throughout a trial. The optional
validated vent law withdraws H2 and
inert proportionally from designated fully gas cells and records actual
species, mass, enthalpy, and momentum loss. Vent enthalpy is signed under the
chosen common caloric reference. This fixed-domain reservoir law is
not a resolved open-boundary nozzle model. A planar-ALE free surface remains
a separately qualified extension.

## Current single-gas implementation: transactions and remaining qualification

`preview` computes a candidate without publishing fields or cumulative
ledgers. `commit` accepts only a current trial belonging to this controller;
repeated or stale commits reject. Checkpoint/restore includes the flow state,
physical inventories, and controller statistics. Regridding is allowed at
accepted boundaries and conservatively remaps the physical state before
rebuilding geometry and closures. Thermal remap separates phase-carried
internal energies rather than treating temperature as a passive scalar.
Momentum transfer is conservative before projection; the final projection
preserves periodic momentum, while wall-normal momentum can exchange an
impulse with the impermeable boundary.
If a native phase is exactly absent only because its positive EOS volume is
below geometric resolution, its bounded residual inventory is transferred by
cell overlap and recombined with the ordinary phase transfer. Number, species,
mass, energy, and momentum move together; nothing is discarded or assigned
invented interface geometry.
If mixing remapped phases under a nonlinear EOS cannot close the local volume
at conserved energy and retained pressure history, regridding rejects and
keeps the accepted topology. It does not normalize the physical inventories.

The unit and distributed tests use synthetic declared materials and closure
laws with independent solutions. Calibrated aqueous rates, predictive film
thickness, centimetric-bubble rise/shape/wake and transfer convergence,
resolved coalescence sensitivity, and coupled-application qualification remain
application gates. The separate [representation guide](bubble_representation.md)
describes bidirectional conversion, its checkpointed registry, retained
pressure histories, and additional eligibility limits.

The molar-pivot discretization follows the two-moment principle of
[Kumar and Ramkrishna's fixed-pivot method](https://engineering.purdue.edu/ramkiites/wp-content/uploads/2019/10/125_1996_Kumar_On-the-Solution-of-Population-Balance-Equations-by-Discretization-I..pdf).
The tracked [VoF verification contracts](../verification/vof_solver_contracts.md)
identify reproducible assertions and measurement workflows. Dated errors and
source-specific results remain local build artifacts.

The [transport and state-ownership contract](../architecture/vof_transport.md)
defines compatible implicit momentum transport, canonical inventories and cache
lifetimes. The verification index lists reproducible assertions; local dated
measurements are not available as tracked qualification reports.
