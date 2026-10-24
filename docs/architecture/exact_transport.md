# ExactTransport: optional rational transport and flow arithmetic

The primary transport path uses bounded floating-point arithmetic under the
[VoF transport contract](vof_transport.md). This document owns the explicitly
selected rational implementation. Its exact equalities, rational pressure
fallback and storage leases are not requirements or fallback behavior of the
primary FP path.

See [ExactVoF](exact_vof.md) for policy/state selection and
[ExactGeometry](exact_geometry.md) for the geometry substrate. Common physical
conservation, positivity, residual tolerances and atomic rejection still apply.

## Swept geometry and donor coverage

`ExactSweptGeometry` uses canonical face triangulations, shared-node traces and
signed swept simplices corrected to the canonical projected integrated face
volume. This shared-vertex/signed-simplex/private-face-correction direction is
informed by [Owkes and Desjardins, sections 3.2–3.5](https://ctflab.mae.cornell.edu/papers/Owkes%2C%20Desjardins.%202014.%20A%20computational%20framework%20for%20conservative%2C%20three-dimensional%2C%20unsplit%2C%20geometric%20transport%20with%20application%20to%20the%20volume-of-fluid%20%28VOF%29%20method.pdf).
Their structured-mesh results do not qualify this general native backend.
Recipient-local positive pullback proofs and owner-routed donor cover
checks establish positive physical transfers independently of the signed raw
face ledger. Opposite signed contributions remain distinct before donor
accounting. An optional certified departure fan can replace the recipient's
signed overlay with positive tetrahedra only after authenticating the original
sweep chains, closed oriented manifold topology, a strict interior kernel,
pairwise interior disjointness, and independent exact volume and first moments.
Every resulting donor intersection still passes the original raw-face coverage
and owner-routed donor-column proofs. A missing derived fan certificate falls
back to the original overlay using the same remaining work budget. Corrupt
original inputs and exhausted resource limits reject rather than selecting a
fallback; the path diagnostic is informational, not an admission switch.
The exact zero-separation and wall-trajectory certificates used by these
operations are specified in [ExactGeometry](exact_geometry.md#separation-and-wall-trajectories).

## Phase remap and fixed-clock handoff

`ExactPhaseTransfer` denotes exact-geometry transfer of finite phase volumes,
optional independent phase mass, signed momentum and named histories through
ownership migration, split and coarsen. Exact overlap weights and transient
certificates establish geometric coverage; independently accumulated finite
phase outputs retain derived rounding, occupancy and fitting bounds. Optional
bounded mass-weighted momentum and pressure-volume products may remain exact
only within an authenticated remap transaction, freshly formed from current
finite phases and physical fields. This preserves individually sub-range
products until aggregation and final finite velocity/pressure publication;
these products are never adopted into phase/error state or reused as an evolving
physical-step history. Same-key ownership copies preserve represented values;
topology changes publish once-rounded target phases with bounded per-source
conservation and target occupancy checks. No rational phase remainder survives
adoption. Solver regrid
and repartition rebuild candidate geometry-dependent operators without advancing
the clock; a failed step or adaptation retains the complete accepted publication.

A multi-plan transfer names each actual native-bound candidate explicitly. The
next plan must originate from that candidate's observed maps and geometry;
an equivalent rebuilt view or a retargeted earlier plan is not a substitute.
The bounded chain authenticates every intermediate observation, then transfers
physical ledgers directly from the original accepted source to the final target,
without generating intermediate physical histories. Existing ancestry-factor
overlap restrictions still apply.

Adoption validation reads the actual accepted identity, control and owned finite
capsules without exporting or reconstructing snapshots. A weak source witness,
source control, transfer transaction and adopted target control bind the opaque
handoff to that publication. A separately constructed look-alike or a later
transported state cannot stand in for the adopted target. Native geometry and
map observations and physical phase/error provenance remain independently checked;
the witness does not retain an old accepted-state chain.

`ExactFaceTransfer` transfers accepted owner-canonical exact face rates at a
fixed clock. Source-free exact rate `q` is independent of cell momentum `M*u`
and pressure history `pV`; transferring those cell quantities cannot recreate
it. Exact oriented common-refinement area fractions preserve inherited boundary
rates. Only new internal face degrees of freedom on that common refinement are
extended from declared physical velocity seeds and closed by a deterministic
local divergence construction. Product intersections use their actual common
refinement, not a nested-refinement shortcut. Source/target coverage, exact
boundary-rate preservation and exact target divergence are checked against the
actual accepted source and adopted target publications. This operation performs
no pressure projection and changes neither momentum nor pressure histories.
These fixed-clock and optional-fan paths remain subject to Stage 7 qualification.

## Pressure, momentum and arithmetic admission

The pressure iterate retains exact primary owner-canonical face rates, including
updates below a represented velocity's ulp. Its defect RHS is normalized exactly
before narrowing. The fixed pressure reference comes from the initial exact
rate divergence; the original absolute/relative threshold remains fixed during
corrections. Physical pressure and force increments recover the same exact
normalization scale, and momentum impulses use the independent physical phase
mass. These bounded step-local certificates are not persistent phase-history
remainders; a new projection starts from the represented physical predictor.

Projection arithmetic intervals derive only from performed operations and exclude
Krylov and model error. A persistent sparse forest adjusts only faces with
positive arithmetic radius, checks exact component compatibility and checks
final divergence on every face. Actual bounded-forest feasibility shares the
original pressure correction and iteration budget; failure does not widen any
arithmetic bound. A zero-radius residual can invoke bounded rational CG only
when every represented tangential pressure offset is exactly zero and the
canonical face mobilities are positive. The same exact pressure supplies the
face-rate update, interior pressure force and physical WLS wall reaction before
any scalar view is published. This fallback does not qualify a nonorthogonal
operator. Unresolved arithmetic feasibility, breakdown or resource exhaustion
rejects the candidate.

The accepted exact face rate is immutable and scales exactly by subsequent
timesteps without another rounded volume-flux closure. It introduces no physical
source. Explicit Input/Scratch/Result admission bounds application exact storage.
Persistent scalar caps and a separate temporary operation lease share the same
configured per-rank budget, whose default is 64 MiB; allocating a temporary lease
does not enlarge that total. This is a configurable per-phase engineering
admission limit, not a physical constraint or a universal production sizing.
Changing it does not change numerical tolerances or make a failed physical gate
acceptable. Immutable exact compaction can remove unused
arithmetic allocation capacity before retaining a scalar/vector value, but must
preflight overlapping source and destination storage within its operation lease.
It preserves the exact rational value and the source object; the compact result
must still satisfy its retained capacity cap. It is not a rounded conversion,
a larger budget or a change to phase, model, convergence or physical gates.
The final donor phase accounts immutable exact payloads by live identity and
reference-counted leases, including source slots and pending replacements.
Scoped raw, paired and correction temporaries release their charges only after
their last retained handle is gone; proof and self-route aliases remain charged.
Its arithmetic arena shares the same phase cap. The per-call label-map cap
cannot exceed the captured construction limit and covers its inputs and
intermediates without changing exact coefficients. This does not claim to
preflight preceding geometric clipping internals.
Raw-face/positive-transfer binding likewise reserves synchronous arithmetic
from its remaining ledger, keeping decoded values, original face terms and
communication buffers accounted through their actual lifetimes. The arithmetic
growth envelope includes its returned result: after synchronous work ends,
that result is immediately admitted to the retained ledger before another
growing allocation. Separate tracking-metadata headroom remains reserved; a
freed operation arena is not permanently charged again as retained payload.
The exact original physical-norm reduction divides the caller's remaining
capacity into pending scalar, operation/compaction and communication arenas.
Codec temporaries die before exchange; exported packets, receive buffers and
sender authentication fit the communication arena concurrently. The exact
squared residual and original zero-RHS reference determine acceptance before
any diagnostic square-root view, including when a tiny positive defect rounds
away in that view. Legacy unleased calls keep their existing contracts. A failed
operation, communication or retained-result admission rejects collectively;
these leases cannot widen physical residual or phase-capacity bounds. Their
implementation alone does not establish physical-flow qualification.
Backend, allocator and Krylov allocations remain outside that accounting. Nested sparse owner queries and overlap services
receive bounded transient caps from the caller's remaining service reservation;
their live inputs, returned records and communication scratch must fit the
caller's simultaneous application-storage accounting. Such a cap neither
changes the retained hierarchy's configured budget nor enlarges the caller's
total. Omitting an optional service cap preserves the service's existing default
admission policy. These are phase-local application-capacity bounds, not a claim
about process RSS, allocator overhead or the combined accepted/candidate peak.

## Authenticated preparation reuse

Exact momentum preparation authenticates ledger payloads, capacities, routes,
face identities, donors, amounts, maps and budgets. Identical rounded views do
not identify identical ledgers. Cache hits rerun RHS, physical-residual,
iteration, positivity and final-publication gates; rejected preparation cannot
leave a partially valid cache. Pressure proposes bounded-significand dyadic
rates within existing arithmetic allowances, followed by exact closure or the
original exact fallback. Final exact width also depends on exponent span and
closure structure. A small dyadic inner normalization does not change the
original physical residual or shared iteration budget.

## Focused verification

Explicit reference targets include `testExactSweptGeometry`, `testDepartureFan`,
`testExactPhaseTransfer`, `testExactFaceTransfer` and
`testGeneralVoFExactPressure`. The registered `ExactPhaseTransfer_(2|4)procs`,
`ExactFaceTransfer_(2|4)procs` and `GeneralVoFExactPressure_(2|4)procs` cases
exercise their stated MPI contracts. Check available registrations and actual
rank counts before claiming coverage. These are optional-route checks, separate
from the primary floating-point acceptance campaign.

The complete physical horizons and policy-specific historical limits are listed
in [ExactVoF](exact_vof.md#verification-and-historical-qualification). A short
reference run or an exact algebraic result does not establish physical accuracy,
long-horizon arithmetic growth, production runtime or process-memory bounds.
