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
an owner-authenticated retained-interface flag. Each geometric face cuts its
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

`OrdinaryAggregateSource` retains immutable tetrahedron recipes and explicit
source-vertex identities. A computational cut is an unmeasured barycentric
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
new cutter and supports common to both endpoints. Their parent records also
retain parent indices and the original cutter expression; redundant cached
endpoint evaluations are not copied through descendant histories. Issued donor support identities select
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
separates the shared foundation, homogeneous specialization, interface-local signed
ledger and aggregate metric admission from unified capillary surfaces and checked
preparation interfaces. The
configured default backend and resource caps are unchanged. Aggregate transport
does not change strict measured geometry required by other consumers.

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
horizons, accuracy/convergence targets and production scaling remain separate
qualification obligations. Focused preparation and failure tests
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

The complete probe is disabled in routine discovery while its reconstructed phase
neighborhood lacks sufficient support. The provisional small-phase policy in
[PLIC reconstruction](../modeling/plic.md#initial-normal-and-angular-objective)
uses a supported gradient normal without angular optimization below its configured
length ratio; it retains the phase-support and target-volume gates. Run the probe
explicitly after building the
target with `--gtest_also_run_disabled_tests` and
`--gtest_filter=GeneralVoFMotionAccuracy.DISABLED_RepeatedClosedMotionReportsIndependentPhaseErrorAndTheta`.
Successful preliminary steps do not qualify its complete horizon. The unchanged
50-cell solver test and its `GeneralVoFFinitePassage_2procs`/`4procs` registrations
remain separate qualification gates for physical residuals and repeated passage.
