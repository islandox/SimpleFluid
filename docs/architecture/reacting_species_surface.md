# Species-bound surface and headspace coupling

The typed Boussinesq path couples `ReactingGasModel` species inventories to
`PlanarFreeSurfaceModel` through accepted event receipts. `MultiphysicsSolverCoupler`
owns optional models and advancement; its shared solver still owns flow and
sensible-temperature fields. This path uses the existing reference-density,
liquid sensible-energy approximation. It does not own the native low-Mach
phase-energy equations or a headspace energy equation.

The field source context explicitly uses
`ReactionVolumeClosure::ReferenceControlVolume`. Its actual EOS liquid volume
sets liquid-based reaction rates; its prescribed geometric cell volume sets
bulk-based rates and heat. Transport or composition/thermal expansion may make
the former exceed the latter. The adapter neither clamps that physical measure
nor claims local phase occupancy. This differs from the kernel's strict
`Accepted` context and the native flow's `ThermodynamicPredictor`, whose owner
must close occupancy before acceptance. Finite volumes and donor limits still
apply in every mode.

## Accepted events and inventory ownership

Initialization binds the ordered species catalog and creates no accepted source
interval. Each successful model advance publishes one `SpeciesEventReceipt`
containing schema, initialization generation, interval, revision and publication
nonce. The surface controller consumes exactly the next receipt once. A pressure
or temperature refresh creates no new event. Physical snapshots restore receipts
and inventories; the nonce allocator never rewinds, so model-only restore and
re-advance cannot impersonate an interval already consumed by the controller.

Gas inventory includes Dissolved, Resolved and every named Dispersed population.
Stationary Resolved gas and all named Dispersed populations also contribute to
submerged volume. Candidate surface/headspace pressure queries evaluate Resolved
gas with its mixture EOS at that pressure, and Dispersed gas with its existing
Laplace/EOS radius closure; zero dispersed populations do not remove Resolved
occupancy. Dissolved gas remains part of the liquid EOS volume.
Solution mass includes Liquid and Dissolved. These overlapping subsets have
independent signed source packets. Raw accepted keyed chemical changes,
individual reservoir transfers and paired interphase events feed these packets
before narrowing; subtracting rounded gross production/consumption cannot recover
small net changes. Gross reports remain provenance. Cumulative species net history
also uses signed accepted events independently of its gross reports.

Local accumulation retains the reaction kernel's `long double` receipt precision;
public packets and MPI reductions retain their supported scalar representation.
This is not an exact-arithmetic claim across arbitrary event magnitudes or MPI
cancellation. Nonfinite or unrepresentable required packets reject rather than
being silently removed.

Only accepted physical Resolved/Dispersed escape packets enter the headspace or
vent. Dissolved outflow is a separate physical sink and contributes to solution
mass outflow. Supplied carried enthalpy and momentum remain Boussinesq escape
traces; the adapter does not invent an additional phase-energy or phase-momentum
debit. `LiquidMassInventory` follows the already transported solution species and
its pure-liquid EOS density without performing a second transport.

The typed ALE nonlinear gate includes species/count states, applied source inputs
and physical molar outflow. Passive escape enthalpy/momentum traces are retained
and checked for finite values but do not impose an extra nonlinear source gate;
this model owns neither a headspace energy equation nor separate bubble momentum.
The subtractive `material_volume_source()` diagnostic is also finite-checked but
does not drive the ALE target or its nonlinear source gate. The independent
level, exact material-volume packet, continuity and conservation gates remain
mandatory.

The mirrored `S_alpha_total` is `(alpha_g_final - alpha_g_step_start)/dt` over the
whole accepted interval, including transport, accepted sources, temperature and
headspace-pressure refreshes. Intermediate mirrors serve material coupling and do
not replace that baseline. Both fixed-grid source schedules retain this contract;
constrained ALE retains its before-temperature schedule and accepted-state replay.
This diagnostic is not a separate physical material source or a substitute for
exact material-volume transport receipts.

## Supplied headspace

The `configure_free_surface(options, headspace)` overload accepts an explicit
headspace model. Typed closed coupling requires `ClosedMixtureHeadspaceModel` or
an equivalent supplied provider with matching catalog fingerprint, nonempty
identity, positive revision and explicit temperature/pressure bounds. Initial
headspace amounts name actual catalog species; inferred air or unnamed initial
moles are rejected. The model solves pressure from the supplied mixture EOS at
its prescribed temperature. Adiabatic headspace energy conservation is outside
this contract.

Headspace laws are MPI-free. The controller fences local exceptions and checks
replicated metadata, maps, scalar closure inputs and raw headspace outputs before
any rank can take a different pressure-root branch. Finite rank divergence is an
error. Configuration retains the injected object's dynamic type. Existing
clone-based checkpoint/restore or failed-commit recovery may replace the object
with an independent clone carrying the same provider and physical state; pointer
identity across restore is not promised.

## Constrained Boussinesq planar ALE

The typed adapter retains the existing laminar, backward-Euler, constant-area,
vented planar-ALE boundary and motion restrictions. It requires complete
advective transport of the solution. Selectively freezing dissolved species is
insufficient for a nonideal liquid-volume packet and is rejected by this path.
VoF stays disabled; these transactions do not qualify independent boiling,
capillary, general moving-wall or native low-Mach ALE combinations.

Each liquid material face-volume packet comes from the same corrected upwind
Liquid/Dissolved molar transport, frozen transport void fraction and mesh-relative
carrier flux. The supplied EOS evaluates that actual face composition and donor
temperature/pressure before sources. The packet is neither a reconstruction from
final cell mass/density nor a second species transport. Its receipt binds the
accepted event and active geometry epoch; geometry changes require a fresh
transport preview.

Material source/thermal expansion uses accepted-old and trial-new EOS volumes
and the exact relative face packet. Numerical GCL corrections remain distinct
from physical material inventory and heat. Trial motion, species, liquid follower,
flow fields, source history, surface/headspace and clock commit or roll back
together. Failed previews preserve accepted state and permit a conservative retry.

## Focused verification

The owning solver tests cover signed subset accounting, real raw-chemistry
cancellation, supplied closed-headspace EOS and finite-rank divergence, exact
pre-source liquid-volume packets, typed surface consumption/escape, ALE rollback
and replay. Registered two- and four-rank cases include actual empty owners.
Consult fresh local test output for executed ranks and skips; source coverage and
roadmap checkmarks alone do not establish runtime or physical validation.
