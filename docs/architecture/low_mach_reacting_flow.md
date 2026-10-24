# Conservative low-Mach reacting flow

`Reacting::LowMachMixtureSolver` owns native mapped flow fields and conservative
physical phase inventories. `ReactingMixtureCoupler` retains shared ownership of
an initialized solver and holds its exclusive advancement lease. The controller
owns optional chemistry; the flow solver owns transport, momentum, thermodynamic
pressure, mechanical pressure, and the accepted clock.

## Physical state and pressure

Each owned cell stores species moles and internal energy for every declared
physical phase. Liquid and dissolved chemical compartments share one physical
solution phase. Gas phases use supplied mixture EOS providers. Density is the
sum of actual phase masses divided by the control volume, rather than a
Boussinesq reference density or a void correction to a fixed liquid mass.

The thermodynamic pressure closes total material volume at the supplied energy
and composition. The mechanical pressure correction enforces local volume
continuity using the variable-density face coefficient. Accepted phase volumes
come from the EOS; neither a separately advected volume label nor a numerical
pressure compatibility correction may change physical inventories.

Mechanical pressure correction includes the local caloric volume response
`chi = -dV/dp` at fixed transported/source inventories and advected-volume
packets. Its linear system adds `chi/dt` to the usual pressure mobility matrix.
This removes the explicit pressure/volume feedback that becomes unstable at
small time steps in dense liquid/gas mixtures. The correction enforces its
linearized continuity equation, while final flow acceptance still requires
nonlinear EOS occupancy and the original physical continuity residual. Removing
the mechanical gauge transfers the same uniform pressure into thermodynamic
pressure, preserving absolute phase pressures. Zero susceptibility recovers the
original conservative projection.

`volume_flux()` and `Trial::owned_volume_flux()` are absolute carrier fluxes.
Physical continuity uses the sum of the phase material fluxes: each phase's
carrier-plus-slip flux is multiplied by the same upwind EOS fraction used for
its conservative transport. In counterflow, different phases can select
different donors, so the sum of these face fractions can exceed one. The
material pressure operator retains that sum in its mobility and includes the
same slip-volume packet; it never normalizes fractions or repeats the slip
subtraction. Its source target uses current EOS volume minus the exact advected
material volume. Both cell pressure gradients are synchronized before momentum,
velocity correction and face prediction read their local/ghost values.

`Trial::owned_phase_transport()` retains the exact final transport-stage carrier,
phase velocity fluxes and upwind material fractions in owned-face order. The
stored stage carrier precedes the last pressure correction; the trial's ordinary
owned flux is the final corrected carrier. These two receipts support separate
checks of advected material volume and final physical continuity. Momentum
budgets expose the actual discrete `pressure_impulse` and `external_impulse`
separately from `momentum_error`; pressure work remains a phase energy term.

Species and phase energy use the same conservative upwind face transport.
Momentum uses the resulting physical mass flux in kg/s. Multiplying this flux by
density again is incorrect. Dynamic viscosity belongs in the stress operator;
non-Laplacian deviatoric stress terms require a supplied contribution when used.
Pressure work uses the phase-volume trial and the same transported volume packet
as the continuity budget.

## Transport-first reaction and heat integration

`MixturePhaseBinding` explicitly maps catalog species compartments to physical
energy phases. The controller consumes supplied reaction rates or integrated
extents through the shared finite-donor reaction kernel. Every transport
corrector starts from the same accepted state, then invokes the source stage
with the newly transported inventory. Chemistry never reserves donors from a
source-depleted previous corrector or reuses an accepted-state source after
transport has changed the available material.

`TransportSnapshot` owns the pre-source species moles, internal energy and
advected material-volume packet for each physical phase, together with old/new
control volumes, accepted time/revision/generation, geometry epochs, schema
fingerprint and source-stage metadata. This snapshot precedes chemistry, ordered face diffusion/Fourier sources and
pressure work.
The `advected_material_volume` is a continuity/work packet, not a physical donor
volume or a concentration denominator. Rates use actual transported composition
and the current thermodynamic T/p guess. The liquid-volume basis is recomputed
from all transported Liquid+Dissolved moles, actual molar masses and the supplied
liquid EOS; bulk-volume rates use the snapshot's new control volume.

The kernel's default `ReactionVolumeClosure::Accepted` requires liquid volume
not to exceed the geometric bulk volume. The flow controller explicitly marks
its intermediate EOS state as `ThermodynamicPredictor`: this permits a trial
phase volume before pressure/occupancy convergence, without changing the actual
EOS concentration basis or geometric bulk volume. Finite positive carrier and
finite-donor checks remain mandatory. The flow owner must pass the original
occupancy and conservation gates before any physical state is accepted.
The separate weak Boussinesq field adapter explicitly uses
`ReferenceControlVolume`; that mode makes no local occupancy claim and is not
used to bypass this native flow's acceptance gates.

`CoupledReactionContext` exposes both the raw transported inventory and a
thermodynamic rate state with that composition. For an arriving occupied phase
whose thermal guess is empty, separate-temperature mode recovers temperature
from that phase's transported internal energy and EOS. It does not borrow a
different phase's temperature. Equilibrium mode can use the shared temperature.
An exactly empty rate-state phase retains zero temperature and energy.

`preview_with_sources` uses one shared transport/source/pressure corrector loop.
Source callbacks are not invoked during pressure-root evaluations. The provider
has an immutable identity, positive revision and matching catalog fingerprint.
The controller forwards `ReactingMixtureOptions::source_revision`,
`source_relative_tolerance` and `maximum_correctors` into that provider.
A positive provider tolerance overrides the solver's source tolerance; zero
inherits it. A positive evaluation cap must be at least two and is bounded by
the solver's `maximum_correctors`; zero inherits that limit. There is no nested
controller loop multiplying the transport budget.

At least two source evaluations and convergence of source packets, transported
inventories, thermal state and any enabled gross-event channel are required.
The reaction controller always supplies gross events; a generic source-stage
provider may explicitly disable that channel with an empty outer vector.
Diagnostics report each residual normalized by its declared gate; acceptance
requires every channel
to be at most one, alongside occupancy and flux convergence. Gross events retain
forward and reverse reaction extents separately, plus signed-channel magnitudes
for external material, transfer energy, impulse and heat. Equal net source
packets cannot hide changing forward/reverse turnover. Event dimensions and
ordering remain fixed across correctors; ranks with no owned cells still take
the same collective sequence.

The internal source-stage provider may perform collective mesh work. User rate,
EOS, equilibrium, partial-molar-enthalpy and heat laws remain deterministic,
MPI-free and free of external history mutation. Local evaluation errors are
fenced before later halo exchanges or reductions.

Independent external heating or cooling is signed energy, separate from material
transfer energy and chemical heat. With formation energy in the caloric EOS,
chemistry does not also supply reaction heat. Separate-temperature reactions that
cross physical phases require explicit energy partitioning. The partition may
redistribute phase energy with zero net heat; it cannot introduce an undeclared
net source. The controller accumulates heat and material-transfer energy
together with compensation before narrowing each phase packet. Its
`net_thermal_energy` diagnostic retains the resulting interval energy when large
opposite reservoir channels cancel; adding the individually rounded channel
totals is not an equivalent energy budget.

## Spatial source context

`ReactionEnvironment::location` optionally supplies an integer canonical geometry
cell ID, geometry epoch, actual centroid in meters and a local owned-row index.
The owned row is valid only within that callback and is not persistent identity;
contiguous algebraic GIDs do not substitute for canonical geometry IDs. Pure
reaction kernels may omit location. Mesh-bound field and native source executors
supply the actual current trial frame. SSPRK2 retains this frame during its local
source substeps while advancing callback time; it does not invent a trajectory.

## Supplied mixture diffusion and Fourier contacts

`ReactingMixtureOptions::face_transport` is an ordered list of explicit physical
phase contacts. Each entry supplies immutable identity, revision and catalog
fingerprint, and enables a diffusion law, a Fourier law, or both. Conductivity in
an EOS alone does not enable heat transport. The initial face adapter supports
closed, nonperiodic native meshes; compact native periodic metadata rejects
before user laws or imports. Physical boundary face packets are zero.

A diffusion law supplies catalog-ordered molar conductances `K_i` in m3/s,
selected primary species and an explicitly selected finite counter-species in
the same physical carrier. Species ID zero is an ordinary valid selection;
omission is an error. The canonical owner-to-neighbor primary packet is
`dt*K_i*(C_owner-C_neighbor)`. Concentrations use actual post-chemistry moles and
EOS phase volume `m/rho(T,p,x)`. Neither the geometric control volume nor the
advected pressure-work volume substitutes for that concentration denominator.
Contact area and distance belong in the supplied conductance; the adapter does
not multiply by area, volume or phase fraction again.

One counterflux cancels the sum of primary species mass fluxes. Every incident
face reserves its outgoing primary and counter donors before any incoming credit
is applied. Insufficient donors reject the trial without clipping, capping or
borrowing incoming material. Canonical owned faces are evaluated once and their
identical integer-indexed packets imported to incident ranks. Each owned cell
applies its own signed side. Optional stages consume the actual finite endpoint
of their predecessor, never independently reserve one shared original state.

Within a source evaluation, the controller retains the high/correction parts of
`ReactionLocalReceipt::postSourceAmounts` through ordered face and interphase
stages. Each actual cell owner reserves original outgoing primary and counter
packets against those local stocks before incoming credit. EOS concentrations
and canonical MPI face packets retain their existing double representation;
the local predecessor parts are never serialized to halo ranks.

The explicit raw-predecessor overloads of `preview_species_diffusion` and
`compile_interphase_plan` validate schema, dimensions and projection to the
represented state. The original overloads still accept an explicitly represented
initial state. A frozen interphase plan retains its raw predecessor for every
pressure-domain and caloric replay; reconstructing it from a rounded stage
endpoint would lose earlier donor debits. `FrozenInterphasePlan` exposes an
appended member, so downstream consumers must rebuild against matching headers.

The donor partial-molar enthalpy of every signed primary and counter transfer is
explicitly supplied at actual donor T/p/composition. No ideal-gas or bulk-mixture
enthalpy surrogate is inferred. Identical raw molar/enthalpy terms enter the two
incident cells with opposite signs. The flow adapter uses a declared barycentric
zero-net-mass diffusion convention with zero mixture momentum contribution;
relative diffusion inertia and stress are outside this adapter. The equation
layer packet utility separately retains its explicitly supplied donor-velocity
convention. These are distinct declared uses of the same finite-donor kernel.

A Fourier law supplies contact conductance `G` in W/K. Its one paired packet is
`Q = dt*G*(T_owner-T_neighbor)`. Contact between dispersed bins, cross-phase
transport, boundary reservoirs and latent heat are not inferred. Phase pressure
work remains owned by the flow solver and is never added by a face law. Positive
post-chemistry inventory needs a valid candidate T/p; no absent phase temperature
or EOS state is fabricated.

The source sequence is conservative transport, local chemistry, ordered face
stages, one final source packet, and flow caloric/pressure acceptance. Chemistry
provides its local unrounded accepted source receipt before double reporting;
rounded extent reports cannot reconstruct small net chemical heat. Raw chemical,
external material, allocated heat, donor enthalpy and Fourier terms stay in one
compensated accumulator until final narrowing. A separate declared thermal sum
survives the entire sequence. Final phase packets must reproduce it on an
endpoint-relative representation scale; large gross turnover cannot hide a lost
joule after cancellation. A custom double heat allocation that loses such a term
rejects rather than weakening the energy gate.

The final signed molar packets must also replay the actual sequential donor
endpoint from transported stocks, with exact empty/nonempty consistency and an
endpoint-relative representation bound. A large depletion followed by a finite
credit can leave a valid donor endpoint whose net double source is not
representable; that publication rejects instead of erasing the remaining stock.
Pure diffusion keeps initial stocks and individual events in one accumulator.
Rounded per-face energy and net material ledgers are reporting views and must
not seed subsequent accumulation.

Receipts retain their producing adapter, physical phase, invocation and geometry
identity. The controller validates each receipt against its immutable ordered
adapter and the exact source/transport/gross-event packets retained by the flow
trial. Local chemistry, face laws, packing and final composition have failure
fences before their next collective, including on empty owners. Commit and
restore invoke no law. The net thermal diagnostic sums final applied phase
packets locally with compensation and then uses the supported scalar MPI sum;
it does not claim exact arithmetic for arbitrary cancellation across ranks.

## Explicit population numbers

`LowMachMixtureSolver::Options::phase_roles` distinguishes bulk liquid, bulk gas
and named dispersed gas. A chemical compartment name or gas EOS alone does not
enable number ownership. Each declared dispersed role has a unique identity and
an explicit finite nonnegative extensive `CellState::phase_numbers` entry. Bulk
entries are disengaged. Zero number and exactly empty gas inventory must coincide;
there is no minimum count, automatic birth or automatic disappearance. Number is
a continuum moment and may be fractional.

Every transport corrector restarts from accepted counts. It transports `N/CV`
using the same phase carrier-plus-slip flux and upwind operator as that bin's
moles, with no extra phase fraction. `TransportSnapshot` retains the actual
pre-source count. An independent local canonical-face balance checks transport,
and the final per-bin budget compares counts with integrated number sources.
Signed cell residuals are accumulated before MPI reduction; subtracting separately
rounded large global totals would misreport an exactly balanced small cell.
Number source and transport residuals have no absolute count floor.

The controller's optional immutable `population_events` provider receives actual
post-material species, transported counts and the current thermodynamic guess.
It returns explicit named nonnegative births and deaths. The role identity must
match its dispersed chemical binding. Provider identity, revision, schema and
presence agree across ranks before callbacks, including on empty owners. The law
is MPI-free; no separate-phase temperature is invented just to evaluate it.

Gross deaths reserve the original transported stock before any birth credit.
Raw events feed separate signed-change, final-stock and donor-margin accumulators;
rounded positive totals are reporting views. Final count packets must replay the
actual endpoint without erasing a finite residual number. Complete gas removal
needs an explicit death, and new gas needs an explicit birth. Fixed-material
coalescence or breakup may change count while leaving gas occupied; count events
infer no material, heat, momentum, radius or surface-energy change.

`Trial::local_population_events()` retains original events, provider/stage metadata,
transported/final counts and phase-aligned birth/death/change packets. Gross birth
and death convergence is checked independently of net count. Diagnostics expose
phase-aligned accepted interval totals. Exact source matching includes the number
packets; commit and restore never invoke the provider. Retained valid trials and
all counts survive later rejected previews and checkpoint replay.

## Prescribed closed planar motion

The explicit `preview_ale` and `preview_ale_with_sources` operations accept a
live `PlanarALEMeshMotion` bound to the exact same native mesh. Preview begins
one geometry trial, assembles on its actual geometry, then restores the accepted
geometry before returning an immutable candidate. `commit_ale` requires that
same live motion authority and restages the retained old/new volumes and swept
rates exactly. It publishes without invoking source, EOS or target laws. The
ordinary fixed-grid commit rejects moving candidates.

Geometry epochs invalidate numerical caches on motion and rollback. A separate
logical accepted-geometry identity lets earlier valid candidates survive a
later failed preview; it does not authorize arbitrary external geometry edits.
Moving checkpoint/restore requires the explicit motion authority and restores
geometry, all physical fields, accepted absolute flux, diagnostics and time.
Trials retain owned geometry certificates, not a borrowed active ALE view or a
lease that outlives the controller.

Species, phase internal energy, population number and momentum use accepted-old
and trial-new control-volume storage with mesh-relative phase advection. The
absolute carrier and phase-flux queries keep their existing meaning.
`PhaseFaceTransport::mesh_volume_flux` records swept volume rate, and
`relative_phase_volume_flux` records actual phase advection. Physical boundary
absolute carrier flux equals the prescribed swept rate, with zero relative
normal phase flux. The transport snapshot captures the actual trial geometry
and its pre-source inventory.

Let `Wold`, `Wadv` and `Wnew` be accepted, advected and final physical material
volumes, `CVnew` the trial geometric cell volume, `w` the canonical swept face
rate and `theta` the summed upwind material fraction. The absolute pressure target
is `(Wnew-Wadv)/dt + (Wold-CVnew)/dt + D(theta*w)`. Retaining these operands keeps
physical material changes separate from geometric conservation. Accepted occupancy
and discrete GCL reduce the geometric term to `D((theta-1)*w)`, without requiring
`theta=1`. Phase pressure work remains `-p*(Vnew-Vadv)`; swept volume is not an
additional source of material or heat. At zero motion, retained accepted EOS
occupancy roundoff can therefore produce a small pressure correction relative
to the fixed-grid route. Cross-route numerical equivalence is bounded by the
volume arithmetic and EOS susceptibility; accepted-state rollback is exact.

`ReactingMixtureOptions::planar_motion` supplies an immutable exact-mesh target
provider with identity and revision. The controller owns its motion authority
and evaluates the target once per preview at accepted time and proposed `dt`.
Source and face laws see actual trial coordinates and volumes. Face receipts
are validated while that geometry is live; exact retained source, transport and
gross-event equality binds their proof after preview rolls back. Commit and
restore invoke no target or source callback.

This path accepts closed, nonperiodic, orthogonal Cartesian, cylindrical and
`SemiStructuredXY_Z` native meshes with inviscid/slip walls. The swept normal
rate does not supply tangential viscous wall velocity or traction, so nonzero
viscosity is rejected. Reservoir boundaries, free-surface target selection,
capillarity, nonorthogonal and general composite motion require separate
contracts and qualification. Fixed-grid capability does not establish support
for those combinations.

## Acceptance and replay

The global Krylov residual does not by itself bound a small local arriving
population. Each population must also close its physical cellwise count/face
balance. Callers select sufficient linear accuracy and a suitable supported
solver policy; an underresolved small phase rejects without a count floor or a
relaxed conservation tolerance.

A source preview retains the final transport snapshot, integrated species,
energy and momentum packets, and gross events. The controller verifies exact receipt
equality with those retained by the flow trial before publishing its trial
ledgers; a tolerance-based resemblance is insufficient. Commit and restore
never call a source again. Retained trials share immutable provider/engine
lifetime independently of the controller's exclusive advancement lease: keeping
a trial alive does not keep that lease active after controller destruction.

A successful preview is an immutable candidate. Other successful or rejected
previews from the same accepted generation do not invalidate it. Commit,
initialization, restore, or controller ownership changes invalidate incompatible
candidates. Fields, clock, diagnostics and source ledgers publish together.

Mass, species, internal energy, momentum, EOS occupancy, and volume-continuity
checks are independent of Krylov convergence. Energy tolerance uses a caloric
scale that does not grow with an arbitrary additive mass-energy reference.
Compensated sums retain source/work/transport contributions, and a separately
reported operand-roundoff bound must remain a small part of the physical
accuracy budget. An ill-conditioned caloric reference is rejected rather than
used to weaken that budget.

The equilibrium mode retains a single recovered temperature. Separate-phase
mode recovers each occupied phase from its own internal energy. Exact empty
phases keep zero temperature and energy. Exact incompressible liquids are
supported when the supplied derivatives define that limit; an incompatible
closed-volume change with zero susceptibility rejects.

## Field-adapter histories and candidate pressure

The separate `ReactingGasModel` field adapter exposes catalog-ordered accepted
`species_histories()`, compartment/population-keyed `species_step_events()`,
and `physical_escape_events()`. Chemical production and consumption, external
supply and withdrawal, internal phase transfer, and actual transport outflow
retain distinct provenance. Step events are integrated events, not differences
of cumulative rounded totals. Private compensated local counters survive
snapshots and rejected/replayed steps; public reduced reports are rounded views
and must not seed subsequent accumulation.
An independent compensated signed net counter supplies the inventory residual;
subtracting separately rounded gross reports can erase a small real transfer.
MPI reporting uses ordinary scalar reductions, so this does not claim exact
arithmetic across arbitrarily cancelling rank totals.

The field adapter's physical thermal source uses `ReactionAdvance::localReceipt`
and individual external-transfer energy packets before one division by `dt*V`.
It does not reconstruct the applied source from rounded heat reports. For example,
a volumetric heat channel and carried-energy withdrawal can leave a finite cooling
packet even when their separately rounded reports cancel to zero. The receipt is
transient local data and has no long-double MPI serialization contract.

Its collective `evaluate_liquid_volume(candidate_offset)` query uses current
Liquid+Dissolved inventories exactly once, retained temperature and the supplied
mixture EOS. `evaluate_submerged_bubble_volume(candidate_offset)` likewise uses
the current gas compositions and population closure. They require initialized
constant/reconstructed pressure modes and identical finite candidate offsets on
all ranks. These queries replace the uniform pressure offset while preserving retained local pressure variation;
they evaluate no source and publish no pressure, fields or history. Pressure
publication validates occupied-phase EOS states and capacity before writes and
restores the accepted snapshot if publication fails. The
[species-bound surface adapter](reacting_species_surface.md) additionally owns
event consumption, liquid following and atomic surface/headspace advancement;
these candidate queries alone do not establish that coupling.

A Boussinesq field model does not own extensive phase energy or independent
dispersed momentum. Its supplied partial-molar carried-enthalpy and continuous-
carrier momentum escape values are explicitly traces, with
`phase_energy_debited=false` and `phase_momentum_debited=false`. The momentum
trace excludes bubble slip. Neither trace establishes a conservative phase
debit; an energy/momentum-owning controller must apply the corresponding debit
exactly once or reject that coupling. External reservoir withdrawal is distinct
from physical escape. See the
[species and thermodynamics contract](reacting_species_contract.md) for inventory
ownership and caloric conventions.

## Current support boundary

The solver supports closed, orthogonal native mapped meshes. Prescribed planar
motion is available through the explicit transaction above. Focused
geometry fixtures include Cartesian, cylindrical, and orthogonal
`SemiStructuredXY_Z` meshes with contiguous algebraic maps. The geometry check
rejects nonorthogonal incidences; geometry-dependent caches track the active
epoch and are refreshed by the motion owner.
An exactly zero interior material-mobility fraction requires strictly positive
EOS susceptibility in every cell. Other possibly disconnected pressure graphs
reject explicitly; this sufficient anchoring rule does not insert diagonal
floors. All public pressure operations validate linear solver options
collectively, including empty owners. Final thermodynamic pressure must remain
inside its declared bracket after the mechanical gauge is separated.

Prescribed phase drift is an explicit closure. Dispersed number ownership is
opt-in through explicit roles and events; bulk phases retain disengaged counts.
The present model uses common phase pressure and zero surface tension. Number
transport does not qualify a capillary, radius or slip correlation. Old/new-volume
fields in a source snapshot alone do not authorize motion; use the explicit ALE
transaction and its support boundary.

Supplied same-carrier species diffusion and Fourier contact laws use the explicit
ordered face adapter described above. Configured local interphase heat and
material stages use [pressure-conditioned thermal ownership](reacting_interphase_thermal.md)
with frozen amounts, supplied partial-molar enthalpies and the same physical work
and acceptance gates. Upstream complete depletion and nonlocal pressure-conditioned
face transfer need their own pre-transfer state/Jacobian contracts and remain rejected.
VoF solvers
remain disabled by default and excluded from the supported reacting-flow scope
while their qualification remains incomplete.

Thermodynamic pressure uses a safeguarded solve within supplied EOS bounds.
A typed caloric range failure at the initial pressure now starts bounded feasible
seed discovery. Explicit `EnthalpyPressureDependence::PressureIndependent`
metadata permits an exact affine pressure-domain calculation; the ideal-gas
factory supplies that capability, while a generic gas EOS does not imply it.
The old EOS constructor retains unspecified capability. Each evaluation checks
the declared derivative identity. Common-temperature mode treats the complete
physical phase group together; separate-temperature mode constrains each phase.
The query retains mechanical offsets, actual transported/source energy and the
advected volume used in pressure work. An exactly empty phase contributes its
work/energy equation without calling an EOS or inserting an inventory floor.

For other EOS, an optional immutable `CaloricPressureDomainProvider` supplies a
certified connected feasible interval, a certified empty domain, or an unresolved
result with an optional search bracket. Its identity, positive revision and
presence agree across ranks. Hooks are MPI-free and local failures are fenced
before global interval intersection, including on ranks with no owned cells.
A connected certificate that contradicts actual caloric recovery rejects.
`maximum_caloric_seed_evaluations` bounds generic discovery (default 64, valid
range 1 through 4096). Exhausted sampling throws `CaloricDomainUnresolved`;
finite samples cannot prove that a narrow or disconnected feasible branch is
absent. Certified emptiness retains the original caloric failure. Real EOS,
derivative and provider errors remain hard failures, never search signals.

Discovery and pressure iteration do not invoke the reaction source. Finding a
calorically feasible seed does not establish a physical volume root: the same
pressure, occupancy, energy and continuity gates still decide acceptance.
Once a feasible state exists, later caloric range failures use safeguarded
backtracking within the retained pressure domain.

MPI configuration and source failures are validated collectively. Regression
fixtures cover empty owners, rejected-source replay, independent heat, two gas
products, nonuniform heating and caloric-reference shifts, with additional
transport-first receipt, gross-turnover and arriving-phase cases. Focused serial
and two-/four-rank checks exercise these contracts, including actual empty owners,
nonunit material fractions, linear-policy failures and independent velocity/flux
oracles. Manufactured numerical checks do not establish physical validation of
a chemistry law.
