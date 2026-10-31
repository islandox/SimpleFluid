# General-native VoF transport families and preparation

General-native VoF records the selected transport family independently of the
configured backend. The family describes how material
crosses a face; it is independent of `ArithmeticMode` and does not imply physical
qualification.

| Family | Backend | Transport construction |
| --- | --- | --- |
| Algebraic face-volume | `DonorCell` | Canonical integrated face volume with donor composition and endpoint/export budgets |
| Algebraic face-volume | `GeometricFace` | The same face-volume contract with measure-only PLIC candidates and conservative limiting |
| Algebraic face-volume | `SweptPartition`, Ordinary interface-free step | Canonical face-volume authority after collective homogeneous admission |
| Interface-local swept | `SweptPartition`, moving Ordinary mixed domain | Indexed geometric face candidates, algebraic bulk faces, one limited signed phase/momentum ledger |
| Geometric swept-tetrahedra | `SweptPartition`, BoundedCertification/Exact | Complete swept construction and positive recipient partitions |

`SweptPartition` remains the default. `GeometricFace` uses geometry to estimate
phase coverage, but does not construct swept tetrahedra. Neither selecting a
family nor preparing its ledger silently changes arithmetic policy. The configured
backend stays `SweptPartition` when its homogeneous specialization reports the
actual `AlgebraicFaceVolume` family. Face-volume execution currently admits Ordinary
arithmetic only; the swept path retains its
explicit finite and Exact providers. See the [face-volume contract](conservative_face_vof.md)
and [floating-point ownership contract](vof_transport.md).

Ordinary `SweptPartition` and `GeometricFace` enable
[budgeted phase cleanup](../modeling/phase_cleanup.md) by default. It runs after raw
transport acceptance and before candidate halo publication, carries cumulative
global signed/absolute corrections, and exposes raw versus cleaned prepared
mass and declared cleanup momentum. It is a physical numerical correction,
separate from geometry quality, roundoff and the strict PLIC fit gate.
The default cutoff is `1e-14`; global absolute caps are derived once from each
initial phase inventory and retained across restart/migration. Explicit cutoff
zero opts out; `DonorCell` and non-Ordinary arithmetic retain disabled defaults.

Optional [measured live balancing](general_vof_balancing.md) records unique
reconstruction and canonical face work. Its explicitly lagged forecasts can
drive Ordinary ownership-only solver repartition, including cleanup accounting,
without changing the physical clock or enabling cleanup-coupled regridding.

## Homogeneous specialization

`homogeneous_transport_eligible()` collectively checks the current source.
Algebraic transport is eligible only when
all retained cells explicitly declare pure gas or pure carrier, no retained mixed
surface exists, and no internal face joins opposite declarations. Periodic images
remain outside this specialization. Disconnected components may declare different
pure phases. Empty owners participate in the eligibility reduction.

The node-control `plan_transport` entry authenticates every supplied node velocity,
departure and face control before selecting the specialization. The explicit
`plan_homogeneous_transport(face_controls,dt)` entry rechecks eligibility and
authenticates face controls, timestep and canonical ownership. It exposes no
trajectory certificate. Exactly zero canonical flux retains the original accepted
source and its histories after those checks. The solver selects this entry before
attempting nodal interpolation, including in `stable_timestep()`.

A declaration is geometric evidence, not a phase-volume cutoff. Independently
positive traces remain transported material. An `Infer` cell or an opposite-pure
face excludes the whole-step homogeneous specialization even if its current
face flux is zero. A moving Ordinary mixed domain instead uses the interface-local
family below; the globally stationary node-control identity path stays unchanged.

For old independent inventories `G,L`, let `S=G+L`. A directed canonical bulk
transfer `q` carries the smaller phase as `q*min(G,L)/S`; the other phase completes
`q`. Retained gas and liquid are each their old inventory minus their own exports,
then incoming transfers are added independently. Only stream composition is
normalized. The accepted inventories are never reconstructed by complement or
renormalized, and the old occupancy discrepancy stays in the donor. Admission
checks `q` summed over all exports against both native volume and `S`, independent
phase exports, canonical continuity, boundaries and resource limits.

Both phase-history channels remain independent. For positive old phase inventory,
the inherited error follows the corresponding phase-transfer fraction. For an
exactly absent phase, its nonzero uncertainty follows `q/S`; the self-retained
share is `1-outgoing_bulk/S`, rather than `(native_volume-outgoing_bulk)/S`.
Shares, products and history sums round outward. Existing pure-fit evidence is
transferred with the minority-phase share (normalized bulk share when absent),
and publication uncertainty is added separately. This records transported evidence
without enlarging a physical tolerance or deriving a bound from a desired result.

The resulting momentum equation uses adjacent-cell implicit upwind coefficients
from those same phase mass transfers. This is a deliberate homogeneous momentum
discretization, including for nonuniform velocity. Canonical face volume is the
transport authority; there is no nodal subface-counterflow model on this path.
The solver retains its face-based advective, viscous and capillary timestep bounds,
canonical absolute-face-rate bound and outgoing-volume CFL. Fitted-node speed and
displacement gates remain specific to geometric swept execution. Supplied nodes
still undergo authentication through the original node-control entry, whose
globally zero-motion identity shortcut remains unchanged. No stationary trajectories
are fabricated. There are no swept tetrahedra, donor intersections, recipient
overlays or PLIC cuts on the homogeneous path.

## Interface-local swept transport

`InterfaceLocalSwept` is the Ordinary moving mixed-domain path. It retains the
configured `SweptPartition` backend and the complete authenticated node/face
controls. It replaces recipient-overlay publication with a single canonical signed
face ledger. Algebraic faces and geometric faces contribute to the same update;
independently constructed recipient partitions are not spliced together.

### Conservative reach and sparse geometry

A scalar sweep recipe includes current and departure vertices, their centers, and
the private corrected departure center that matches the canonical volume control.
That corrected center can extend beyond every node trajectory. A balanced AABB
index over retained full native cells selects donors using the complete corrected
hull. A second index covers the retained-domain frontier, including physical walls.
The frontier catalog is retained with immutable topology before native assembly
consumes its construction draft.

Each frontier retains its canonical face/tile identity, physical boundary or
retained-halo kind, retained cell incidence and owner, and the missing cell key
and owner where applicable. Its boundary classifier binds the immutable original
cell-face support, rather than rebuilding a plane from rounded tile vertices.
The cache observations and retained cell identities authenticate geometry revision
and ownership. A geometric `PlaneAuthority` key alone does not authenticate either.
Trial-pinned construction controls rebuild the node trajectories against that
current source authority before reach admission.

`PLIC::OrdinaryBoundary` distinguishes inside, construction contact, represented
coordinate contact, outside and unresolved arithmetic. A sealed point on the
same original support establishes construction contact even when its stored
coordinates have a nonzero plane residual. Wall trajectories, frontier admission,
aggregate donor cuts and source-simplex common-support tests share sealed support
membership. Coordinate mutation invalidates the seal; a nonzero private cap-center
displacement drops the inherited supports. Without matching construction evidence,
the original framed plane predicate determines the side. Numerical uncertainty
does not establish contact, and no orientation epsilon is introduced.

The supporting-halfspace screen remains a sufficient containment proof. It is
stronger than testing intersection with finite frontier polygons and can reject
valid reach in a nonconvex retained domain. An outside or unresolved recipe point
still rejects collectively, with its role and frontier identity reported. Physical
boundary crossings and missing-donor halo reach have distinct diagnostics; the
path does not silently expand the halo or assume unknown fluid is homogeneous.
Finite-polygon reach and constrained cap correction remain separate work.

`GeneralVoFBoundaryReach.Face17ReplaysOriginalWallConstructionBeforeDonorClipping`
reconstructs a small native source fan using the captured nodal velocities,
timestep and canonical face control. It preserves original support identities
through backtracing and checks the wall contact independently of donor clipping.
Companion checks retain rejection of stale observations, foreign geometry owners,
missing halo donors and an escaping flux-corrected cap center.

Any `Infer` donor, retained mixed surface, or conflict between pure declarations
among reachable donors/endpoints selects geometric work. Thus a face-aligned
interface requires geometry without an artificially mixed cell. Only selected
nonzero-effect recipes retain tetrahedral source witnesses. Zero **net** control does not establish
zero effect: opposing swept pieces remain significant. Pure bulk faces use the
admitted donor-cell stream model.

Only requested donor planes are reconstructed, at their owners with the existing
reconstruction stencil. Requests/replies bind source control, transaction,
topology/partition generation, cell revision and owner. Ghost phase records carry
an owner-authenticated retained-interface flag. Within one sparse import, an
owner first reuses its locally requested result, then an invocation-local map of
peer-only results. Every incoming request completes packet/context and
owner/revision validation before either lookup. A miss performs the unchanged
reconstruction; every request still produces a separately serialized, charged
reply. Peer-only entries never enlarge the returned local request set and are
destroyed before reply exchange. No fitted plane persists between invocations:
neighbor phase changes can change a normal even when target geometry is unchanged.
Receivers retain their independent binding checks and mixed-plane remeasurement.

Each geometric face cuts its
source witnesses against its indexed candidate set, preserving overlap and
coverage checks. No recipient overlay is constructed. Immutable donor/frontier
indexes are shared through the retained topology, with hierarchy observations and
retained cell identities checked before reuse. The cache holds geometry and keys,
not pointers into a previous publication's phase cells. Every trial rebuilds its
corrected sweep bounds and classifies the current phases. Cache storage remains
charged to the retained-state and querying-operation budgets; build/reuse counters
distinguish index construction from actual query work. Reuse still checks the
retained cell catalog and does not establish constant-cost preparation.

### Aggregate metric admission

`OrdinaryAggregateSource` consumes an `OrdinaryAggregateSourceBatch` containing
one point table and indexed tetrahedron recipes. Each tetrahedron retains four
typed local indices and its stream sign; a table slot supplies the source-vertex
identity, including the point's complete provenance. Equal coordinates in different
slots remain different identities. Construction validates the table and incidence
and allocates metadata before moving the buffers, so rejection leaves the input
unchanged. Successful construction transfers exclusive buffer ownership; callers
must relinquish mutable aliases. Const source views remain valid while their
source owner lives. This indexed interface replaces the point-carrying tetrahedron
span interface.

Source-facet lookup uses contiguous records ordered by stream sign, three sorted
point indices, and original facet identity. A private facet handle encodes the
tetrahedron and one of its four faces; an unmatched sentinel replaces separate
persistent cancellation flags. The integer-only prepass covers at most the first
effective pair allowance plus one facets. Orientations, signed volume sums, pair
admission and semantic incidence rejection still follow the original source/face
order. Source-stage diagnostics retain the source ordinal and leave `cut` absent.
Only an opposite-oriented pair is authenticated; a third incidence rejects at its
original occurrence. A structural partner cancels only if that partner survives
the same donor/sign/phase channel.

A computational cut is an unmeasured barycentric
witness; its rounded shell need not have a separately measurable positive volume.
Donor supports come from the retained geometry authority. Complementary cuts
retain outside witnesses until donor coverage is complete. Donor overlap,
missing coverage, inconsistent identities and exhausted construction work reject.

Cut classification uses residual-based construction arithmetic separately from
the aggregate metric quality estimate. Exact operations retain zero uncertainty.
When rounded barycentric placement cannot resolve a later cut, the provider
reevaluates its original defining supports. A zero numerator with no unresolved
arithmetic establishes contact, including crossings whose barycentric division
was inexact. Unresolved signs still reject; no coordinate
snapping or tolerance enlargement establishes incidence. Source orientations,
source/cut distances and per-witness integration coefficients are reused within
their immutable dependencies, with storage and executed work accounted.

Generated vertices retain their defining supports separately from cached
barycentric coordinates and numerical estimates. Original vertex `i` carries
the three source-face equations `lambda[j]=0` for `j!=i`; crossings retain the
new cutter and supports common to both endpoints. The original tetrahedron and
defining-support catalog supply the construction authority; no edge-parent
history is retained. Issued donor support identities select
one deterministic original framed expression, independent of donor order, while
the complementary side retains its own orientation. Authenticated original
source incidences remain structural zeros in the coefficient rows.

If coordinate evaluation remains ambiguous, the classifier selects
three independent defining supports and evaluates the queried distance as a ratio
of four-by-four determinants in the original source frame. Coefficients are
reevaluated from original framed points, normals, anchors and offsets, rather
than treating cached rounded distances as exact input. Error-free floating
expansions retain correlated residuals, with a cap of 2048 components per
expansion and the existing byte/operation limits. An exactly singular basis can
be replaced by another defining triple; exhausted capacity, range loss or an
unresolved predicate rejects. This is a bounded Ordinary predicate, with no
arithmetic-mode change or rational transport geometry.

The indirect sign and its metric distance have distinct obligations. The sign
can establish contact or a side, but interpolation still consumes the returned
distance and residual-derived error estimate. A resolved sign never authorizes
discarding metric uncertainty or changing the physical volume allowances.
Fallback storage and executed work are charged on success and failure, including
when a caller already has an active geometry budget.
Evaluations are cached only within one immutable source/measure operation, keyed
by the three defining support identities and queried support. Coverage, metric
cuts and complementary queries can reuse that result; complements reverse the
value and sign while retaining the same error. Cache entries remain charged to
the original budget. Reuse does not reduce the count of work already executed.

### Compact complementary witnesses and construction cost

Each child stores only vertices referenced by its surviving face loops. Compaction
preserves all four barycentric components, their numerical errors, every support
incidence, inherited source-face tags and the immutable source/cache references.
Child indices no longer identify original source leaves; the source tetrahedron
remains available independently for predicates and integration.

Coverage and gas/carrier cuts use one complementary split. It classifies each
active input vertex once and constructs each crossed edge once, publishing the
same crossing coordinates, errors and support evidence to both children. One cap
loop supplies opposite windings. An unchanged child moves its existing storage;
direct metric clipping requests only the inside child. Phase integration retains
the bulk witness and accounts for the copy needed before consuming its phase
split. Source-face cancellation and the structural donor-cover proof are unchanged.

Ordinary interface-local preparation preserves ordered scalar metadata for every
pending face. Reach, classification and zero-effect checks run before a scalar
face releases its recipe and candidates. Only geometric faces retain those owners;
each geometric slot is consumed into the current face scope. Its complete points
move into the source table in center/backcenter/current/departure order, its old
recipe buffers are destroyed, and candidates are released after donor preparation.
Face order remains aligned with optional work-capture rows.

Phase quotas retain the numerical donor-vector order. A separate sorted
permutation resolves full donor IDs to those local slots; donor IDs are never
treated as quota-vector indices. Duplicate donor admission remains the aggregate
validator's responsibility.

Spatial indexes, pending preparation charges, imported planes/shared geometry and
completed face contributions remain charged throughout preparation even when a
pending owner has already been destroyed. Current-face setup,
source bookkeeping, donor validation, support/distance/predicate caches, coverage
remainders, metric witnesses and temporary aggregate results have a face-local
scope. The default 64 MiB admission condition covers the retained preparation
allowance plus the current face's conservative scratch allowance.

Sparse plane import also uses a scoped reservation on that preparation allowance.
Authority-count storage and the complete requested result map, including adaptive
key storage, are reserved before import. Live request/receive packet capacities,
decoded-key scratch and the full mandatory reply envelope remain charged while
peer-only entries are admitted from the remainder. Exchanges receive only the
remaining allowance after caller-owned input storage. Optional admission failure
reconstructs and serializes without retaining the plane; it neither raises limits
nor suppresses original reconstruction or packet checks. All peer entries are
destroyed before their reservation is released. The scoped import reservation
expires after its packet containers, leaving the returned planes charged.

Within that face, aggregate charges remain cumulative, including temporaries
already destroyed. This conservatively bounds the complete face workspace rather
than substituting the narrower live-witness counter. Both aggregate results and
their copied output are charged while they coexist. Successful publication retains
only the output groups' surviving capacity and growth of the retained face vector.
On exceptions, unpublished output is destroyed before its pending charge is
released. The scratch reservation expires only after all its owning containers.
Short-lived reach-query and sweep-boundary buffers use the same lifetime discipline.
The setup allowance and source allowance both conservatively cover the moved point
and index buffers. Ownership transfer removes the buffer copy but does not transfer
or refund admission charges. These overlapping reservations and retired charges
are not a count of actual allocations or simultaneous live storage.
This is a preparation-storage allowance, not a bound on total process RSS or
the combined memory of every source, candidate, MPI runtime and caller.

Finite global-point consumers request owning exports against their preparation
ledger and existing byte ceiling. Handles keep borrowed point spans alive; the
ledger pins admitted owners until its scope ends. Temporary export spending stays
in cumulative preparation accounting but is excluded from persistent spatial and
surface catalog byte totals unless those objects retain an export handle.

Construction traffic remains monotone across faces, including retired capacity
and work admitted before a rejected trial. `maximum_face_geometry_operations` defaults to 67,108,864
counted aggregate scalar operations per geometric face, including source
construction, retries and nested providers. Zero rejects; this shared control
must agree across ranks and is bound into the construction-control identity.
The existing preparation pair limit and aggregate witness/complexity limits remain
independent guards. Total aggregate operations are reported separately from storage;
reach and setup predicate operations are outside that particular counter.
Storage rejection identifies retained/setup/aggregate bytes and the storage limit;
work rejection identifies its exhausted scalar, pair or witness allowance.
`OrdinaryAggregateStatistics::stage_charged_bytes` attributes admitted charges to
source setup, donor validation, coverage and metrics; their sum equals
`storage_bytes`. A rejected aggregate charge is reported separately and is not
added to that sum. Copy, active/stored-vertex and pending-remainder counters expose
representation cost without changing numerical acceptance.

Set `SIMPLEFLUID_VOF_AGGREGATE_TRACE=1` to emit one local JSON record for each owned
geometric face, including failures. Records distinguish retained entry/exit storage,
face scratch, combined reservation high-water, cumulative construction traffic,
total aggregate operations, stage attribution and a rejected aggregate request.
Counters saturate instead of wrapping if their representation is exhausted.
`witness_admissions` retains the resource guard's requested-child count,
including empty children. The live-witness high-water mark tracks Poly objects
and their owned buffer capacities, including concurrent parent/child and metric
witnesses. It excludes allocator metadata, maps, predicate scratch, source/support
caches and other retained geometry, so it must not be interpreted as total memory.
Tracing adds no collectives and leaves the original rejection path intact.

### Retained-domain coverage proof boundary

The aggregate API accepts arbitrary donor sets, so disjoint donor interiors do
not prove that their union covers a source. Complement subtraction remains
mandatory. A future internal alternative could use the native mesh complex,
but it must authenticate complete cell shells and cavity/halo frontiers, current
geometry/topology/partition observations, an in-domain source seed and containment
of every corrected source tetrahedron, complete conservative spatial queries,
and exact correspondence between queried cells and supplied donor geometry.
An approximate donor-volume sum cannot replace these obligations: gaps and
overlaps can cancel in that sum. The existing reach/index metadata supplies
ingredients for such a certificate, not a currently admitted coverage shortcut.

### Aggregate volume integration

Scalar integration groups only one canonical face, old donor, stream direction
and phase at a time. Shared internal boundaries cancel by source construction
identity before integration; equal coordinates alone do not authorize cancellation.
Opposing streams and distinct donors remain separate. No convex hull or recipient
redistribution replaces the witness collection. The resulting bulk, gas and
carrier measures feed the same phase labels and source-resolved face ledger.

Integration uses translated, power-of-two scaled frames and compensated sums.
Original support distances and source determinants use the Ordinary compensated
predicates. Barycentric placement error contributes to a separate numerical
quality estimate. Every positive aggregate is checked, including small physical
phase amounts. A sparse donor-owner exchange counts every requested face/sign
authority across ranks. Each independent old source-phase allowance is divided
by that count and intersected with the face allowance; a remote or nonadjacent
face cannot spend the same donor allowance again. Pure geometric labels use the
source bulk scale, then their independent inventory factors carry that accuracy
to retained traces. These are inventory-scaled budgets, not a new relative
accuracy requirement on every tiny counterstream. Failed quality permits one
reconstruction from immutable source supports; it never retries from a failed rounded child. The retry applies the
phase support before donor supports and evaluates the same aggregate through
closed witness integrals in translated source-barycentric frames, using the
original source Jacobian. Different local reference frames require retaining all
faces of each witness in this equivalent domain sum. Structural boundary
cancellation still authenticates the aggregate, but the retry is not an
exterior-only quadrature. All raw signed terms accumulate before the aggregate
positivity/quality gate; no witness needs its own admitted measure. A
still-unresolved or nonpositive candidate rejects collectively.

A resolved positive candidate can exceed an old-phase quality share while fitting
its face allowance. In that case an inward-rounded ratio further limits the one
shared face correction coefficient: the final contribution's high-order quality
estimate is `theta * E_high`. The existing endpoint/export limits may reduce that
same coefficient again. This cannot rescue unresolved geometry, and it does not
change the low-order arithmetic/publication gates. `quality_limited_faces` and
`minimum_quality_theta` distinguish this admission from unrestricted swept
correction. The estimates remain Ordinary numerical indicators, not enclosures.

Three accuracy obligations remain distinct:

- Computational pieces need structural containment, orientation and coverage;
  they do not each need an accurately measured closed solid.
- Physical phase aggregates need independent gas/carrier accuracy and the
  original face/cell allowances. A bulk-volume allowance does not authorize
  losing a minority phase. Phase labels preserve independently stored traces.
- Surface sections, first moments and curvature retain their separate measured
  geometry contract. Scalar aggregate admission supplies no surface accuracy.

The provider reports numerical error **estimates**, not certificates. Its explicit
conservative approximation field is separate and remains zero in the implemented
resolved path. The transport consumer rejects a nonzero approximation field until
an explicit shared ledger debit is supplied; unresolved witnesses are never
silently deleted. Existing physical conservation, export, occupancy and momentum
residual gates remain unchanged. BoundedCertification, Exact, remap and public
measured-polyhedron consumers retain their existing paths.

`swept_pieces` counts unmeasured source tetrahedra on this path. `aggregate_groups`,
`aggregate_witnesses`, `cancelled_boundaries` and `aggregate_retries` report local
geometry work. These are operation counts, not memory or timing qualification.

### Shared correction and signed source contributions

Let `L[f,d,p]` be the admitted donor-cell contribution and `H[f,d,p]` the signed
swept contribution, resolved by face `f`, old donor `d`, and bulk/gas/liquid channel
`p`. One coefficient is shared by every channel and provenance/history field:

$$
J_{fdp}=(1-\theta_f)L_{fdp}+\theta_f H_{fdp},\qquad 0\le\theta_f\le1.
$$

The face owner retains separate positive/negative donor groups. It never replaces
them with net face volume. Endpoint and old-donor demand exchanges select the same
`theta` on both incidences. Low-order admission is mandatory before geometric
candidates. Endpoint availability comes from compensated **unpublished** low-order
receipts, since binary64 publication can round that availability upward.

With outward face incidence `B[c,f]`, the physical source-resolved update is

$$
W_{cdp}=\delta_{cd}V^n_{dp}-\sum_f B_{cf}J_{fdp},\qquad
V^{n+1}_{cp}=\sum_d W_{cdp}.
$$

The bulk coefficient uses the finite donor geometry volume; gas and liquid use
the independently accepted inventories. Native metric discrepancy remains in
occupancy provenance. Individual `W` entries may be negative. They are signed numerical source
coefficients, not negative accepted inventories or positive material parcels.
The limiter bounds adverse endpoint corrections and increases in positive remote
source contributions. Its ratios round inward, and final values are independently
rechecked. The signed donor budget records

$$
E^+_{dp}=\sum_{c\ne d}\max(W_{cdp},0),\quad
E^-_{dp}=\sum_{c\ne d}\max(-W_{cdp},0),\quad
V^n_{dp}=W_{ddp}+E^+_{dp}-E^-_{dp},\quad E^+_{dp}\le V^n_{dp}.
$$

`old-positive_remote` is budget slack, not the diagonal retained coefficient.
Accepted gas and liquid are separately nonnegative and separately conserved.
Each final face bulk amount must meet its original canonical-face allowance.
Two-sided cell closure budgets also constrain differing face coefficients; signed
cancellations before limiting cannot justify a larger allowance afterward.

### Histories and occupancy evidence

Contributions from the same old source are combined before taking absolute values.
For a positive old phase, its history weight is `abs(W/old_phase)`. For an absent
phase on pure geometry, retain the normalized coefficient operator directly:
low-order shares use `q/(G+L)` and high-order shares use geometric `bulk/V`.
Outward coefficient-arithmetic bounds and new publication uncertainty are added.
An absent physical phase with retained mixed geometry rejects rather than guessing
its history normalization. Signed antidiffusion can amplify histories; those
histories never relax the independent physical phase-conservation gate.

Occupancy evidence requires a separate operator because normalized low-order
transport leaves the old occupancy discrepancy local. For high-order bulk share
`k=H_bulk/V_old`, define

$$
R_{cd}=\delta_{cd}-\sum_f B_{cf}\theta_f k_{fd}.
$$

Old occupancy and fitting evidence propagate with `abs(R)`, while signed
`PhaseLabel` allocation is retained separately. Raw phase-cut closure defects
are source-derived and charged alongside sweep/control defects to the **original**
cell geometry allowance. `signed_phase_cut_closure` exposes that extra diagnostic;
combined closure is canonical residual plus signed geometry difference plus phase
cut closure. Neither observed output discrepancy nor a physical minority amount
creates a new error allowance. Compensated publication preserves even microscopic
occupancy residuals. Pure-fit evidence and geometric-presence declarations remain
separate from inventory amounts.

### Physical momentum and numerical support

The final phase ledger defines mixture mass and the physical operator

$$
A_{cd}=M^{n+1}_c\delta_{cd}+\sum_f B_{cf}\sum_{p\in\{g,l\}}\rho_p J_{fdp}.
$$

Positive donor-cell routes are retained as explicitly named numerical support.
An admitted low-order ledger supplies those routes, donor budgets and unpublished
compensated endpoint capacities. Shared admission retains the original input,
export, history, occupancy and independent global phase checks. The mixed path
does not construct or import a discarded low-order candidate state; its final
limited physical ledger alone produces the candidate publication.
For `SignedFaceLedger`, GMRES solves the actual signed matrix, with positive row
scaling `D=M_new+outgoing_low`. It does not normalize by the physical diagonal,
which may be zero or negative, or require equality with the low-order operator.
Every correction still solves the physical matrix; the original long-double
physical-residual gate, shared iteration limit and refinement limit remain.
Matching component right-hand sides reuse one preparation. Legacy equivalent-route
solves retain their previous matrix, preconditioner and coefficient-agreement gate.

## One detached finite ledger

### Checked controls and prepared views

`GeneralVoF::transport_controls<Mode>(nodes, faces, dt)` collectively creates a
`GeneralVoFControlView<Mode>` pinned to the accepted source publication. It owns
its control arrays, so temporary input vectors do not leave dangling spans.
Retained control storage obeys the existing allocation cap. The explicit mode
must match the source even on ranks with no nodes, faces or owned cells. Optional
finite and Exact face-volume payloads must match that mode; incompatible payloads
reject before dispatch.

The typed `plan_transport`, `plan_homogeneous_transport` and `plan_face_transport`
overloads authenticate the pinned source and its native observations. Source
tokens agree across all ranks, including owners with no control records. Face-only
execution requires empty node controls and retains the homogeneous interpolation
bypass; supplied node controls cannot be silently ignored. Existing span-based
entry points remain checked compatibility adapters.
Constructing a control view does not establish geometric reach, physical timestep
admissibility or transport conservation; the selected planning path retains those
checks.

`Trial::prepare_transport_view<Mode>(density, stage)` returns an owning
`GeneralVoFPreparedTrialView<Mode>` for Ordinary or BoundedCertification. Its
factory checks mode collectively and binds the trial, source, densities and time
stage. Local getters expose the immutable ledger without communication. Exact
controls remain supported through their separate Exact trial authority; the finite
prepared view does not convert Exact inventories or coefficients.

### Ledger and solver consumption

`GeneralVoF::Trial::prepare_transport` binds a finite trial to positive constant
phase densities and a caller time-stage identity. Its immutable
`GeneralVoFPreparedTransport` retains the source pin, transaction, mesh identity,
selected backend/family and arithmetic mode, timestep, independently stored phase
records and histories, and owned old/new mixture-mass capacities. Empty owner
collections are valid and do not select an arithmetic policy or transport family.
The homogeneous swept trial retains phase streams privately until the caller binds
densities and time stage with `prepare_transport`; it exposes no dummy-density
preparation through `prepared_transport()`. The actual family is factory-authenticated
and is not inferred from an empty stream collection.
An Exact trial continues through its separate typed path.
The source and trial validation methods authenticate publication/context pins;
they are not an admission API for caller-mutated ledger copies. Numerical execution
consumes the immutable preparation produced by the trial factory.

Prepared transport also retains node trajectories, face-chain diagnostics,
cell-closure diagnostics and local transport work counters. These records share
the ledger's lifetime and allocation admission. Face-chain records may include
replicated retained tiles: deduplicate canonical face IDs before global diagnostic
sums. Counters describe local work, not physical accuracy or runtime scaling.

`diagnostics.owner_plane_requests` reports rank-local sparse-import demand,
local-result and peer-cache hits, admissions/denials, actual owner calls and their
elapsed duration. Branch counters distinguish mixed-cell fits, retained
interfaces, pure declarations and zero-phase offset calls. Geometry-volume and
intersection deltas measure kernel operations executed within these owner calls;
they are not offset/clip iteration counts. Peer-cache peak bytes describe its map
nodes and adaptive keys; workspace peak bytes include overlapping preparation and
import allowances, rather than process RSS. With zero admission denials,
`owner_reconstruction_calls == local_owned_requests + peer_cache_admissions`
counts the exact union of requested owned keys. Denied entries can be fitted
again; the diagnostic does not claim a distinct-key count on that path. Aggregate
counts and elapsed durations across ranks outside a timing region when comparing
work or imbalance. Serial request sets already deduplicate local donors, so reuse
primarily removes MPI owner fan-out work.

Canonical signed physical face contributions and positive donor-to-recipient
route support have different meanings. Swept contributions can name nonincident
geometric donors and can oppose each other on a face; they must not be reinterpreted
as adjacent upwind streams or collapsed to a net volume. Face-volume preparations
also retain their directed streams and limiter diagnostics. Interface-local
preparations expose `signed_receipts` and `signed_donor_budgets`, separately from
`numerical_support_transfers` and `numerical_support_budgets`. Their legacy
`positive_transfers`, `donor_budgets` and trial positive-record views are empty;
consumers must select the prepared family, not infer an identity update from empty
collections. Independent gas and liquid amounts remain authoritative, including
traces on declared pure geometry.

The finite solver uses mode-checked controls and a prepared view, reads trajectory
evidence through that preparation, and checks phase
publication and mass capacities, and derives momentum inputs from its retained
ledger. The finite momentum workspace validates routing and raw face agreement
when preparing the numerical operator, then reuses its immutable routing, imports,
coefficients and applicable preconditioner for matching component right-hand sides.
Every right-hand side still has its own input, iteration and original physical
residual checks. Equivalent-route families preserve their coefficient-agreement
requirement. Interface-local signed execution uses its explicitly different
numerical-support contract without changing the accepted physical equation.
The freshly checked prepared view can be consumed directly; the solver retains
its later source/publication checks without immediately repeating the factory's
complete collective authentication.

Only successful whole-step publication exposes the preparation, including its
momentum identity and build/reuse counts, through `GeneralVoFSolver::prepared_transport()`.
Failure leaves accepted phases, velocity, pressure, time, histories and the previous
preparation untouched. A new mesh or changed numerical coefficients invalidate
reuse; retained accepted preparations are immutable evidence, not mutable workspaces.

## Collective failure evidence

Finite VoF local-work boundaries retain bounded structured failure records. Each
record identifies the originating rank, stage and error category, original cause,
and the available cell/face/donor/recipient/construction context. Numerical scale,
budget and retry fields are reported only when known. Sweep construction can retain
its bounded tetrahedral coordinates; unavailable information is explicitly absent.

All participants first agree that a local operation failed, select a deterministic
origin, and distribute the bounded record before throwing. The originating local
exception remains accessible. Allocation/resource exhaustion, malformed metadata,
and typed unresolved geometry/arithmetic remain distinguishable. Legacy untyped
exceptions retain their original message; boundary categories are diagnostics,
not a new recovery or fallback policy. Diagnostic records do not introduce new
collectives inside rank-dependent local geometry traversal.
The helper does not change unrelated distributed AMR error handling. One compiled
FVM context provider shares the thread-local diagnostic stack across solver, FVM
and caller scopes; the export boundary permits only that exact bridge.

These records improve diagnosis; they do not make a failed geometric construction
acceptable. An unresolved final aggregate or required surface measurement still
rejects atomically; a computational witness is not itself a measured output.

## Qualification boundary

The [Stage 7 roadmap](../../TODO.md#stage-7--general-vof-and-flow-qualification)
separates mandatory default-policy qualification, capability extensions,
conditional optimization/performance and optional strict-mode completion.
[Qualification regimes](../verification/vof_solver_contracts.md#stage-7-scope-and-priorities)
record backend, cleanup, mesh, material, Courant and horizon explicitly.
Ordinary SweptPartition with automatic cleanup remains the baseline. Ordinary
interface-local preparation uses the storage-lifetime and explicit work policies
above; strict measured geometry required by other consumers is unchanged.

Focused checks cover analytic phase transfer, competing exports, zero-net signed
counterflow and correlated histories, native-owner agreement, empty owners and
locality counters. The coupled regression covers three accepted steps from both
aligned and mixed interfaces at density ratio 1000, with the original mass,
physical-residual, divergence and rollback gates. Quality-limited corrections are
reported separately from unrestricted swept candidates. This finite horizon does
not qualify complete interface-motion or capillary cycles. Uncertain or uncovered
corrected reach still rejects. Index/candidate counts measure local work, not wall
time, RSS or production scaling.

The shared [capillary geometry model](../modeling/general_vof_capillarity.md)
provides mixed/aligned patch identities and explicit binary filtering with
component-volume constraints, independent of the transport backend. It changes
no accepted inventory, history or transport declaration. Complete motion/capillary
horizons and accuracy/convergence for the claimed regime remain core gates;
production scaling is a separate conditional performance claim. Focused preparation and failure tests
cannot establish those results.

### Independent prescribed-motion probe

`testGeneralVoFMotionAccuracy` includes an independent polygon-area oracle and an
opt-in eight-step skew/coarse-fine transport probe. In reference coordinates
`X=x-y/4`, `Y=y`, its closed streamfunction is
`psi=U*f(X)*Y*(1-Y)`, where `f=4X` below `X=1/4`, `f=1` in the central half,
and `f=4(1-X)` above `X=3/4`. The initial interface `X=1/2` remains in the
central shear region, so its exact evolution is `X+2UtY=1/2+Ut`.
Canonical face fluxes follow streamfunction differences; independent polygon
clipping and area integration supply gas and carrier reference volumes.

Each accepted step reports volume-normalized phase L1 error, maximum cell-fraction
error, conservation, passage and the final geometric-face theta histogram. Bin 0
records zero correction, bins 1 through 8 record `floor(8*theta)+1` for
`0<theta<1`, and bin 9 records full correction. These are canonical-owner face
counts and can be summed across ranks. This prescribed transport reference is
not an analytic coupled pressure/velocity solution, and no new accuracy threshold
is inferred from its reports.

The swept probe is opt-in while its full reconstruction and motion horizon
remains unqualified. The provisional small-phase
policy in
[PLIC reconstruction](../modeling/plic.md#initial-normal-and-angular-objective)
uses a supported gradient normal without angular optimization below its configured
length ratio; it retains directional admission and target-volume gates. Run the
current probe with `SIMPLEFLUID_RUN_ALGEBRAIC_MOTION=1` and
`--gtest_filter=GeneralVoFMotionAccuracy.SingleInterfaceClosedShearSweptPartition`.
It explicitly disables cleanup; the old `DISABLED_RepeatedClosedMotion...` name
is historical and is no longer a registered test.
Successful preliminary steps do not qualify its complete horizon. The
50-cell solver test and its `GeneralVoFFinitePassage_2procs`/`4procs` registrations
remain separate qualification gates for physical residuals and repeated passage.
The current named full50 helper leaves cleanup automatic (its Boolean argument
records balance work). It retains the original raw numerical gates. An explicitly
configured zero-cutoff microscopic variant is separate optional robustness work.
The historical capture used the then-disabled policy; preserve its failed MPI
horizon, local replay and collective state-preserving rejection without making
successful completion a default-policy prerequisite.

The algebraic variants in the same executable enter through
`plan_face_transport` with explicit `DonorCell` or `GeometricFace`. They have a
separate [preregistered campaign](../../verification/vof/algebraic_motion_manifest.json)
and [gate checklist](../verification/vof_solver_contracts.md#algebraic-face-volume-qualification).
Their independent polygon/prism oracle covers complete translation, rotation and
deformation-return motions, spatial and temporal refinement, and actual directed
stream limiting/fallback. Midpoint excursions distinguish physical motion from a
stationary return endpoint. Controlled algebraic fixtures explicitly disable
cleanup unless their documented switch is selected; the separate single-interface eight-step fixtures
remain explicit-zero. Default-policy independent accuracy and cleanup-sensitivity
evidence must therefore be identified separately. Algebraic results do not qualify
swept reach/geometry, and swept failures do not establish algebraic failure.
