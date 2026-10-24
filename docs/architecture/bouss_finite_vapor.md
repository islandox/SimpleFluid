# Finite solvent/vapor with a Boussinesq scalar thermal owner

This integration candidate adds an explicit fixed-reference-volume Bouss
approximation. Its source is implemented in BoussFiniteVapor, with the field
model and MultiphysicsSolverCoupler owning transport, temperature and atomic
acceptance. No qualification is inferred from this contract or source tests.

The typed field path selects ReactingGasMode::Species; its catalog and explicit
population_ids define actual species and bins. No hydrogen, donor, inert or vapor
slot is reserved, and no enabled null-mechanism or yield fallback is supplied.

## Material and energy ownership

The caller supplies an actual catalog/mechanism and finite amounts/counts.
BoussReferenceVolume supplies constant liquid density, fixed thermal density/cp
and catalog-ordered additive gas molar volumes. These are declared approximate
laws, not a supplied compressible EOS, fabricated susceptibility or native
phase-energy owner. Physical liquid diagnostic/rate volume is actual
Liquid+Dissolved mass/rho_l and may differ from the scalar void complement;
ReferenceControlVolume declares that weak Bouss ownership explicitly.

Temperature stores fixed rho_ref*cp*T on the geometric control volume. The
mechanism must use SensibleEnergy. Raw chemistry/external/reservoir heat applies
once, as do each accepted latent/correction event. Material carried enthalpy
remains a trace. No formation-energy storage or phase-U debit is inferred.

Ordinary actual chemistry is strict and unscaled. A separate gas-request law
explicitly authorizes ordered admission. Bulk and wall solvent/vapor proposals
then share finite donors, remaining void capacity and scalar energy allowance
under ProportionalRequests. Incoming admitted material never finances another
same-stage gross donor. Condensation reserves the actual pre-admission vapor.

Every dispersed birth/death is an explicit supplied event, after accepted
material admission. The Bouss helper requires paired count/material endpoints
and does not infer a count death from arbitrary gas disappearance. The generic
field default dissolution/conversion operator likewise requires its own paired
occupied predecessor and declared events; its own complete dissolution may
supply the remaining-count death. Earlier chemistry or a user material packet
cannot trigger an automatic count reset.

## Exact wall heat owner

Selected wall patches need an actual prescribed boundary heat-flux snapshot or
Neumann temperature condition. Unknown Dirichlet/Robin/convection heat at a
selected patch rejects unless an explicit flux owner is supplied. Other
unselected boundary thermal conditions remain owned by the temperature equation.

After current material/turbulence updates, the controller certifies the exact
effective conductivity/boundary cache, patch flux, geometry epoch, interval and
immutable configuration. Owned canonical face data is imported for ghost
boundary entries. One immutable cache is installed into TemperatureDiffusionEquation.
The same producer identity/epoch and conductivity identity/values are checked
before temperature; the original cache is restored after success or failure.

The full prescribed wall heat enters through that boundary assembly once.
Its configured evaporation fraction proposes vapor; accepted latent/correction
is the sole corresponding volumetric thermal debit. The wall budget is never
added a second time as a source. The cylinder retains its original top q''/k
Neumann condition and radial exterior-convection choices.

## Configuration and acceptance

Ordinary material refresh preserves initialized finite inventories, including
the first timestep. Explicit primary-field initialization owns any requested
reset, so a rejected first step restores its original initialization authority.

Partial configure_bouss_sources replaces immutable laws without resetting
inventories, histories, accepted event revision/nonce or physical initialization
authority. A separate source-configuration generation invalidates model
snapshots captured under older laws. Old accepted receipts retain their actual
historical configuration provenance; the next successful advance publishes
the new law's receipt.

configure_reacting_gas stages option construction/copy inside an MPI-free
allocation fence, validates complete option/schema metadata collectively, and
fences all current surface, pressure, independent-boiling, thermal and retained
scalar-mirror support checks before field/model construction or mutation.
Rank-local valid but incompatible selections reject coherently, including
empty owners. The standalone Database parser remains MPI-free behind the
existing collective controller entry point.

## Scoped support and explicit exclusions

This fixed-grid path supports native MeshHandle and the existing legacy Mesh
caller identity/synchronization through mapped thermal/species storage. It
keeps mechanical density feedback separate from pinned thermal storage.
Closed material boundaries, constant cp/reference density, zero surface tension
and zero/prescribed slip are the declared scope. Variable-cp/material-updater
schedules, native phase-U thermal participants, supplied phase EOS/film laws,
free-surface/ALE and coupledNonlinear combinations reject.

Standalone boiling/no-gas paths remain separate. Independent boiling cannot
also own the same typed solvent/vapor latent inventory. Supplied-EOS surface/
headspace and native low-Mach thermal stages retain their own contracts; this
adapter is not evidence for those combinations.

VoF remains outside this reacting-flow support matrix.
Bounded-FP and exact-rational VoF/general-AMR paths remain deferred; lower-level
geometry behavior is unchanged. Focused serial and actual2/4-rank qualification remains required.
