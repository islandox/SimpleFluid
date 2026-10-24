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
fingerprint and source-stage metadata. This snapshot precedes local sources and
pressure work; its API also reserves this ordering for future Fourier heat.
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

## Acceptance and replay

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

Its collective `evaluate_liquid_volume(candidate_offset)` query uses current
Liquid+Dissolved inventories exactly once, retained temperature and the supplied
mixture EOS. `evaluate_submerged_bubble_volume(candidate_offset)` likewise uses
the current gas compositions and population closure. They require initialized
constant/reconstructed pressure modes and identical finite candidate offsets on
all ranks. These queries replace the uniform pressure offset while preserving retained local pressure variation;
they evaluate no source and publish no pressure, fields or history. Pressure
publication validates occupied-phase EOS states and capacity before writes and
restores the accepted snapshot if publication fails. These seams do not by
themselves enable the pending species-owned free-surface/headspace adapter.

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

The solver supports closed, fixed, orthogonal native mapped meshes. Focused
geometry fixtures include Cartesian, cylindrical, and orthogonal
`SemiStructuredXY_Z` meshes with contiguous algebraic maps. The geometry check
rejects nonorthogonal incidences; construction captures a fixed geometry epoch.
An exactly zero interior material-mobility fraction requires strictly positive
EOS susceptibility in every cell. Other possibly disconnected pressure graphs
reject explicitly; this sufficient anchoring rule does not insert diagonal
floors. All public pressure operations validate linear solver options
collectively, including empty owners. Final thermodynamic pressure must remain
inside its declared bracket after the mechanical gauge is separated.

Prescribed phase drift is an explicit closure. ALE is not enabled by the
old/new-volume fields in a source snapshot. Population-number ownership and
transport are not implemented by this backend; `population_number` remains
disengaged rather than inferred from molar inventory.

Bulk species diffusion, Fourier heat transport and interphase thermal exchange
are not wired into this backend. Supplying conductivity in a caloric provider
or using a separate equation-layer diffusion helper does not enable their
conservative flow adapters. Those paths require complete transport, donor,
energy and acceptance contracts before their support gates open. VoF solvers
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
