# General VoF implementation boundaries

The public `GeneralVoF` and `GeneralVoFSolver` interfaces retain their phase,
geometry, conservation and rollback contracts. Implementation modules separate
operations from arithmetic selection. Ordinary is the default; Bounded and
Exact remain explicitly selected paths with distinct numerical evidence.

## Phase transport and remap

`src/FVM/details/GeneralVoF/Transport` owns moving transport. `Remap` owns
finite hierarchy adaptation and ownership migration. Each has an independent
compiled provider with Ordinary and Bounded explicit instantiations. Exact
transport and Exact transfer retain their separate implementations and support
contracts; finite remap does not imply an Exact remap specialization.

`PhaseAllocation` owns the shared physical phase labels, signed fitting
redistribution and inherited uncertainty calculations. `Wire` owns shared
record encoding, without selecting communication order. Operation-specific
records and codecs stay with their operation.

Shared state support has separate compiled providers within FVM:

| Provider | Responsibility |
| --- | --- |
| `GeneralVoFState` | Accepted-state validation, inventories, catalog refresh, and source observation checks |
| `Wire` | Scalar, phase, and interface codecs; its declaration header does not include the accepted-state layout |
| `Reconstruction` | Samples, stencils, fitting, and observed owner reconstruction |
| `RetainedStorage` | Retained allocation accounting and admission |
| `OperationAgreement` | Transaction identities and collective option agreement |
| `HaloExchange` | Complete canonical node fans and owner phase exchange |
| `PlaneExchange` | Full and requested-owner plane exchange, including reply-cache admission |

Each provider owns its supported explicit instantiations. Reconstruction work
observation surrounds fitting itself; reply-codec calibration remains outside
that measurement. Moving these responsibilities does not move collective stages
or change their failure boundaries.

`PhasePublication` owns finite owned-cell array assembly and the shared
eight-value independent phase-conservation reduction summary and check. Each
transport coordinator retains its accumulation and narrowing policy, collective
reduction, and failure context. The check selects its policy at compile time,
with private Ordinary and Bounded instantiations owned by FVM. Ordinary physical
allowances and bounded arithmetic allowances remain distinct, including their
intermediate precision.
Publication copies already computed cell inventories and evidence in topology
order; it does not derive either phase from the other, import ghosts, or apply
cleanup. Finite remap uses the same assembly operations while retaining its own
coverage, channel, and history-preservation checks. Exact publication remains
with its separate authority.

The operation coordinators retain collective failure boundaries and owner
exchanges in their original order. Rank-local work must complete its collective
error propagation before the next exchange, including on empty owners. Local
preparation helpers do not silently initiate collectives.

Transport authenticates mode-checked source-pinned controls, selects algebraic
bulk or swept interface work, authenticates donor allocations, exchanges amounts
and validates the candidate. Prepared views retain the selected family's ledger
and diagnostics without inferring policy from nonempty records. Remap agrees
the schema and target, authenticates overlap requests, transfers contributions,
certifies recipient coverage and installs phase inventories and their evidence
together. A source-pinned trial remains responsible for observation validation.

After common authentication and the no-motion path, moving Ordinary transport
selects homogeneous face or interface-local transport. The remaining finite
partition body is compiled only for BoundedCertification. Local limiter ratios
and translated-face phase integrals have named computation boundaries; their
coordinators retain packet exchange, recovery, diagnostic accounting, and final
publication order.

## Solver transaction ownership

`src/solvers/details/GeneralVoF/State.hh` declares accepted/candidate fields and
arithmetic evidence. `SolverImpl.hh` declares the private coordinator, which
owns a `Resources` instance and the accepted state. `Resources` owns the native
mesh, immutable operation controls, geometry metrics, projection workspace and
momentum workspace. It has no accepted-state pointer. The public solver
implementation forwards operations to the coordinator. Substantial definitions
compile privately, while simple members remain in headers and the
caller-dependent `Execution` collective callback is defined in `.ipp`.

Initialization and advancement prepare a detached state. Remap prepares a
replacement resource owner and detached state. Both finite and Exact preparation
complete their physical checks and final phase observation validation before
returning. Only the coordinator installs the returned accepted snapshot; failed
preparation leaves the original publication and clock unchanged. Final remap
publication swaps the prepared resources and state without allocation or
additional numerical work. Old resources remain alive through those swaps.
Destruction performs no collective work.

`Initialization<Mode>`, `Advance<Mode>` and `Remap<Mode>` compose the narrow
components below. The coordinator selects a mode once per operation; selected
numerical loops call compiled typed implementations directly. Initialization
shares its operation sequence across all three modes. Advance and remap have
separate Exact specializations under `exact/`; finite remap and Exact transfer
retain their different protocols. The old unrestricted Exact context adapter
and aggregate implementation fragments are removed.

## Solver numerical components

The private `GeneralVoF/` directory contains independently compiled components:

| Component | Inputs and responsibility |
| --- | --- |
| `Execution` | Borrowed native mesh/maps; collective failure propagation, reductions and imports |
| `NodeVelocity` | Phase support metadata, cell velocity and byte cap; authenticated canonical node exchange |
| `Curvature<Mode>` | Phase interfaces, native geometry and capillary controls; distributed patch assembly |
| `PhaseStorage<Mode>` | Phase authority and material densities; mixture storage and publication validation |
| `Diagnostics<Mode>` | Phase authority, mass and velocity; physical inventory, momentum and energy reductions |
| `FaceRates` | Phase authority, canonical rate evidence and flow workspace; rates, integrated volumes and topology |
| `AdaptedFaceTransfer` | Source/target geometry and remap evidence; common-refinement face transfer and closure |
| `OwnedFaceTransfer` | Source/target canonical identities and rates; ownership-only face transfer |
| `StateTrace` | Borrowed finite fields and diagnostics; lazy rank-local advancement trace emission |

These components have no access to the solver coordinator or its accepted-state
pointer. Face transfer borrows pinned source/target views and returns detached
face fields, projection evidence and divergence. Phase storage returns separate
finite mass/error or rational mass results. Exact storage and diagnostics use
explicit specializations compiled in `exact/`; their shared rational reduction
retains the existing codec and storage admission checks.

`StateTrace` receives a borrowed view rather than the private accepted-state or
resource bundle. Derived trace values and cell/face output are evaluated inside
the existing opt-in trace guard; trace emission introduces no collectives.

The architecture check rejects component includes of the coordinator and
rejects whole-state/resource dependencies in the narrow numerical components.

Curvature assembly delegates patch fitting to the existing curvature kernel.
Projection and implicit momentum remain owned by the existing flow workspace
and conservative transport components. The extraction does not create another
numerical backend or broaden their support matrices.

`GeneralVoFProjectionSupport` shares the finite wide reductions, local arithmetic
guard, and local occupancy-target calculations. The guard retains the flow
workspace's Boolean reduction and `runtime_error` protocol. The workspace owns
the high/low/exponent face imports and consumes the owned prefix of its input.
Ordinary compatible projection and bounded finite projection retain their
separate operators, arithmetic policies, and admission rules.

Occupancy calculations borrow geometry, volumes, incident rates, and signed
inventory discrepancies. Their result reuses the existing target buffer and
holds its diagnostics. Each projection coordinator keeps the guards, reduction
sequence, zero-capacity branch, compatibility checks, and decision to freeze the
target after provisional physical projection. Bounded projection still rejects
transport readiness. The extraction preserves existing scratch-allocation and
collective placement; it does not extend allocation-failure guarantees.

## Compilation and resource contracts

The motion-accuracy executable compiles its independent polygon oracle, native
fixtures, qualification cases, and captured reconstruction replay as separate
sources. Test-only headers declare their named interfaces; the oracle uses STL
geometry calculations rather than production PLIC cuts or measures. The main
test source owns the single environment registration. Qualification cases retain
their existing names, horizons, gates, and opt-in execution controls.

Small general and finite test fixtures share hierarchy and control builders,
while their callers supply domain identities and independent phase values.
FVM-only test membership is an explicit CMake list, preserving the link boundary
against dependencies on the solver library.

Private `.hh` files declare interfaces and retain simple one-line members.
Heavy templates live in `.tcc` providers under `details/GeneralVoF/`. Their
supported modes are instantiated by `.cc` units under each owning component's
`explicit_instantiation/GeneralVoF/` directory. FVM transport/remap dispatch and
trial methods remain in ordinary `details/GeneralVoF/*.cc` sources. Exact
specialization method definitions remain in `details/GeneralVoF/exact/`; they
are ordinary compiled implementations, not explicit-instantiation statements.
Declaration and provider includes precede
code; only `.ipp` definitions may be included at the end of a header. Public
headers never include compiled providers.

The state-support providers retain their existing concrete template definitions
and explicit instantiations together in their owning `.cc` files.

New numerical translation units inherit the owning component's strict floating
point, PGO, PCH and unity-build policy. Internal functions stay hidden inside
their existing library; FVM must not depend on the solver library.

The component CMake files name general-native source-free provider groups
separately from adaptive and common VoF sources. The solver also separates
low-Mach and representation providers. These lists organize existing sources;
they do not create new libraries or alternate numerical backends. FVM providers
retain their existing unconditional registration, while solver VoF providers
retain the `SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF` gate. Explicit instantiations,
strict arithmetic flags, PGO exclusions and PCH/unity settings stay with their
existing translation units.

Splitting a function does not grant another resource budget. Existing admission
scopes, operation counters, packet lifetimes and geometry leases remain in
force. Intermediate records borrow pinned data or move owned buffers; additional
copies or overlapping lifetimes require explicit accounting. Ordinary physical
allowances must not be relabeled as bounded certificates.

The immutable FVM topology owns each retained native tile once. Surface and
frontier ranges, construction controls and finite wall identities refer to that
owner by checked index or canonical face ID. The public owned-incident tile
order is unchanged. Exact wall records retain their original identity payloads
without acquiring the finite construction catalog. Adjacency and complete node
fans borrow full cell keys from one topology-owned intern pool; iteration and
membership use key values, never pointer ordering. Accepted publications pin
the topology for every such borrow.

Retained admission charges tile owners, reference capacities, interned keys,
graph rows and geometry payloads. Temporary growth and encoding buffers retain
their own admission charges while live. Geometry admission deduplicates shared
geometry, provenance-support and typed identity owners across retained roots;
the ledger pins those owners and includes its bookkeeping in admission.
Allocation measurements report their scope and exclusions separately from these
conservative admission estimates; neither metric is process RSS.

The hierarchy publishes immutable `AdaptiveGeometrySnapshot` roots. Cells and
FVM catalogs select source entries from those roots; they do not construct a
second authoritative geometry catalog. A snapshot owns detached binding pages,
coordinates, original supports and represented geometry. It holds the exact
layout identity weakly. Retained views and PLIC construction records keep the
required source version alive after a mesh handle is destroyed.
Equal coordinates in independently constructed meshes do not identify one owner.

Binding pages share provider and recipe descriptors across rows. Cartesian and
static pages encode regular ancestry and ordinals as progressions. Product
pages select separate base and axial records, and explicit convex cells retain
their original keys and certificate owners. Captured native metric columns
preserve binary64 values, including signed zero, across later query environments.
Common evaluation policy flags are stored once per page. A separate ordered
selection preserves request order, and an expanded cache can share prior
binding and explicit shell pages. Reordered PLIC catalogs share shell pages
and procedural sources directly.

An Ordinary State cache expansion of up to four requested cells can prepare
the remaining owned Cartesian cells in the same recipe-relative page of at
most 64 ordinals. The requested snapshot and its overlap are admitted first.
Preparation uses the remaining operation, hierarchy and residency capacity and
shares the exact native recipe owner already pinned by its demand binding.
Tight-cap requests, foreign rows and value-equal recipes with separate owners
follow the demand path. A later selection authenticates the current owned row,
the exact layout and construction controls, and the raw endpoint bits before
reusing its prepared source slot. OriginalSources builds fresh raw contexts
over the selected rows of prepared State pages. Expanded snapshots pack
partial binding and shell pages while sharing full pages and the original
geometry owners. A successor layout may reuse exact selected source bindings
through a weak predecessor. Each successor authenticates its own layout
identity.

Provider-input acquisition and authenticated native-publication acquisition are
distinct. The private publication seam checks its source observation, complete
bindings and actual selected draft rows. A published explicit cell already owns
its mapped geometry and native metrics; that owner is referenced directly rather
than passed through the provider mapping again. Compact rows still use the
provider's original construction path. Publication-derived representations do
not replace provider-input cache entries.

Native publication stages the output manifest, source manifest, and their paired
owner-union replacement before committing. Each arithmetic channel stages one
ordered row per touched physical owner, with its original and final total and
typed reference counts. The first block lives in the editor and overflow blocks
are admitted before allocation. Receipt cleanup reads those rows while they still
include owners that entered and left the transaction. The manifest consumes only
changed counts and shares untouched index pages. The leaf merge uses a bounded
row window and admits branch scratch from the child pages the edit actually
produces. Actual allocations retire with their owners. Cumulative index charges
remain spent on success or rejection.

Ordinary procedural Cartesian cells retain a shared source context and a catalog
slot. Its immutable base owns the evaluated source axes, compact interval
selectors and original support facts. Canonical support slots are indexed by
axis coordinate, with sparse entries for supplied per-face exceptions.
Numerical evaluation caches are separate from that base.
Snapshot reuse requires the actual source binding and supported named arithmetic
controls; floating-point status flags and padding bytes are not cache keys.
State publication primes moment data at the same boundary as its original import
path. It has no backpointer to the hierarchy or phase state, general plane
owners, frozen cell topology or export cache. The native adapter authenticates
the actual provider capability, source recipe, typed ancestry and recipe-relative
ordinal before selecting this representation. Raw geometry bounds use the original
provider coordinate expression under the current construction controls. Stored
recipe axes retain their separate native binary64 metric role. A Cartesian
provider label alone is insufficient.
Constructed, rebased, numerical-only and explicit native roots use existing
materialized owners. BoundedCertification retains its separately certified
geometry; Exact retains its existing import and arithmetic workflow.

State publication also requires a sufficient exact representation check. Under
supported construction controls it checks raw-coordinate subtraction and scaling
witnesses, signed coordinate roundtrips, and identical source/import frames and
normalized corners. A failed proof preserves the original and separately imported
materialized representations. It neither rejects the cell nor forces moment
preparation. This matters for large coordinate ratios where a raw endpoint or its
error witness cannot survive the original import unchanged.

Accepted cells store catalog slots and phase amounts. The topology owns the
catalogs, with owned cells first and retained support cells afterwards. Public
owned-geometry ranges, remap common cells, reconstruction patches, support cells and phase samples
yield immutable `CellGeometryView<Mode>` values, or their runtime counterpart.
Ranges, iterators and returned cell/face values pin the catalog independently.
They provide checked indexed and range access without an implicit shell or
pointer conversion. Reordering builds a flat catalog of source references rather
than a chain of previous catalogs. Recipe-relative slots establish neither
canonical identity nor transaction authority.

Routine volume, moment, support, bounds, face and occupancy queries use these
views. Native binary64 mesh metrics remain separate from represented PLIC
metrics. Shared fixed-box and planar evaluators preserve the original frame,
corner/face order, integration, support and error arithmetic. Original source
supports are distinct from reconstructed canonical face coordinates; wall and
frontier constraints retain those original facts. Complete source-point fans
include hanging points on coarse edges and face interiors.

Active clipping, donor/overlap work and reconstruction explicitly call
`materialize_cell(view, workspace)` or the ordered `materialize_cells(range,
workspace)` batch. The operation-local workspace admits shell
construction, scratch and temporary ownership and tracks remaining arithmetic
and pair work. Temporary shells never enter an accepted cache ledger simply
because an operation needed them. A clipped result, snapshot or replay dependency
that survives the operation receives complete retained admission before
publication. Early destruction does not refund cumulative charges. Exact views
forward their existing owners explicitly and do not acquire a finite workspace.

Native materialization publishes a validated immutable source body independently
of its temporary geometry wrapper. Cut and rebase records retain that shared
source body and the necessary numerical certificates, rather than retaining the
temporary native-cell implementation. Replay binds already validated values;
it does not repeat source construction arithmetic. Original cutter-plane identity
and the registry's minimum-key selection rule remain unchanged. Weak body caches
do not turn routine source queries into persistent full-cell materialization.

Raw source first moments still use the first querying caller's controls. Fresh
raw evaluations and generated fallback imports therefore receive independent
lazy moment caches, even when their immutable geometry is shared. An explicitly
supplied original geometry keeps its existing ownership and cache semantics.
The initial and expanded catalogs within one constructor reuse their evaluation
owners only under the matching operation, catalog, snapshot, arithmetic-control
and source-binding proof. This preserves the original preparation order while
keeping independent constructors isolated. Captured readiness policies describe
facts established during source construction; later template queries cannot
seed another evaluation's lazy cache.
Snapshot-cache growth and old/new publication overlap are admitted before
publication, including hierarchy residency leases and caller ledger bookkeeping.
Rejected acquisition preserves published roots and spent cumulative charges.

Adapted face transfer obtains complete remote target support through the
collective `fetch_native_sources` facade. Its immutable result owns the fetched
draft and an original-source geometry range in owned-then-ghost order. A returned
view can outlive that result; a borrowed draft requires the result to remain alive.
The existing native fetch keeps its hierarchy construction cap. Returned draft
ownership is admitted into the caller's operation budget before allocating the
new catalog and gathers. The reported admission includes spent scratch and is
not a live-memory or RSS measurement. These geometric views establish neither
handoff authority nor permission to omit a remote fan or coverage proof.
Fresh raw source contexts retain the original box's lazy first-moment behavior:
the first successful moment or centroid query fixes its integration cache under
that caller's controls. Constructor volume remains separate; later origin shifts
use the current caller. Materialization copies an established cache or leaves an
unprepared raw result lazy. Inline cache publication is synchronized and a failed
query can retry without publishing a partial result.

Ordinary aggregate donors can carry an original cell view. Pure labels read its
represented bulk measure; ordered cut facts and framed bounds come from the same
original plane data. This avoids building a general cell solely to read its
planes. Existing owning donors keep their original path, and an unresolved
overlap still uses the established intersection kernel and complete donor checks.
Reconstructing formerly stored facts is charged as preparation work and reported
separately from the unchanged numerical integration work. It is never refunded.

Ownership-only remap reuse additionally requires exact source recipe and
represented geometry agreement, including ordered corners, original supports,
frame and face facts. Equal keys, volumes or centroids alone do not establish
that proof. A missing or failed proof uses the existing geometric coverage
checks. The final retained audit visits actual surviving catalogs, contexts,
support payloads, fallback geometry and tracking storage.

The allocation ledger stores a strong pin for each kind-qualified owner address.
Payload and bookkeeping totals belong to the admission scope rather than to
individual map entries. Node charges use the actual entry representation;
admission still precedes publication and a failed scope remains poisoned.

Finite published geometry uses immutable flat topology. Face offsets and vertex
indices, vertex-to-plane incidence, and surface plane slots/orientations share
one topology owner with a unique owning plane table. Plane slots follow first
encounter in vertex/incidence order and deduplicate actual plane objects only;
equal coefficients or authority values do not merge independent plane objects.
Checked actual counts and terminal offsets select 32-bit arrays or their 64-bit
alternative. Global identities, wire indices and provenance limits are separate.

Construction uses mutable drafts. Publication freezes only after validation,
recovery and measurement have completed, and prepares both requested clip children
before publishing either. The ledger accounts shared topology once, including
capacity, while construction admission covers draft/frozen overlap and lookup
scratch. Numerical rebases and certificate-only copies share topology; physical
translation and native-support replacement establish new immutable bindings.

The finite `faces()` API returns borrowed immutable `FaceRange` and
`FaceVertexRange` values. They are sized and indexable, with vertex indices yielded
as `size_t`, and do not expose contiguous `size_t` storage. A row keeps its own view
state after the temporary outer range disappears; a geometry owner must still
remain alive. Traversal allocates nothing. Nested-vector construction inputs remain
supported, range inputs are accepted, and owning numerical shells explicitly copy
their face loops. There is no nested-vector compatibility cache.

Finite clipping compacts each child through local used flags and an old-to-new
index array over the completed source/crossing vertex domain. Scanning old IDs in
ascending order preserves point numbering. The arrays are reused only between
children of one split and destroyed before final validation or recursive retry.
Ordinary face-distinctness marks are local to one validation call and cleared
along each successfully checked face. Aggregate cap membership and directed
successors similarly use local arrays, preserve the minimum-ID ring start and
winding, and retain duplicate/open/disconnected-ring rejection. Sparse edge and
support lookups remain separate mechanisms.

Scratch admission covers actual capacities, including whole replacement buffers
when growth overlaps an earlier allocation. Reuse and destruction do not refund
cumulative charges. These local indices never allocate by a maximum global ID or
introduce a dense vertex-pair table.

Finite construction history retains shared immutable recipes. Each recipe stores
only its active operation payload: a source geometry owner, a shared clip operation
and side, or an exclusively owned Rebase payload with its parent and full input
halfspace. The large Rebase payload is allocated only for that operation and is
charged once with its recipe owner. Shared clip records retain one point context
and compact source-point references;
the general recipe node has no retained-point vector. Both finite modes and
authenticated replay use the same creation and accounting rules. Construction
packet version 5, root ordering and replayed shells are independent of this private
storage layout.

Finite point provenance is a mode-typed immutable snapshot, independent of
geometric identity and of the geometry replay graph. Native-source references,
generated recipes, export aliases, numerical-only values and optional error
witnesses have separate arrays. A vertex retains a compact local point reference
and a typed incidence range; the geometry pins their context once. Native rows
reference shared coordinate fact blocks, rather than retaining an old geometry
or copying its construction graph. Generated edges reference the records they
need. Older frozen contexts are pinned once per dependency table, making the
ownership graph acyclic without a global interner.

Existing typed identities and canonical keys still name geometric entities.
Privately issued `PrivateAuthorityRef<Mode>` values name registered authority
entries in their context. Registration creates no point membership: separate
incidence rows come only from validated source topology or trusted construction
proofs. Ordinary evidence does not certify a Bounded source. Public boundary
tokens remain authenticated by their supplied registries. Local indices never
enter wire packets and cannot be rebound as public point tokens in a foreign
context. Failed builders are poisoned, and freezing prevents further mutation.

Owning `Point<Mode>` values retain a typed context and exact role/index token;
`DynamicPoint` erases mode only at the explicit runtime boundary. Changing a
point's numerical fields invalidates its proof but does not release its owner.
`PointSnapshot<Mode>` stores one context pin and compact ordered references,
without another numerical point array. The FVM canonical-node catalog uses such
a snapshot with local node-to-slot mappings. Materializing one owning point is
allocation-free and preserves its token. Source fans are validated in their
existing per-request numerical scopes while batch storage admission accounts
shared snapshot arrays and overlapping query storage.

Finite global point exports are explicit owners. `export_points(ledger, used,
limit)` admits a `DynamicPointExport` or `PointExport<Mode>` into the caller's
budget and returns a handle whose lvalue `points()` view borrows that payload.
The handle or ledger pin must survive the view. `vertex_count()` and numerical
queries need no global export; `local_points()` retains its authority-free
contract. Native planar and Exact point APIs are separate.

Geometry owns only fixed cache metadata: a mutex, weak export roots, a weak
provenance snapshot and fixed local alias slots. Actual dynamic and typed point
buffers, provenance and supports are allocated on request and admitted before cache publication. Typed
and dynamic exports own independent point arrays and share the immutable provenance
snapshot. A typed request does not materialize or pin a dynamic array. Requesting
both representations admits both arrays and counts their shared snapshot once.
Independent ledgers can share one
published payload; a ledger itself is not shared across threads. Geometry-only
admission ignores weak cache entries and has the same cost before and after an
export. Export-specific resource failures occur at the explicit request, while
malformed geometry, support and coordinate-range checks remain at construction.

After all export handles and ledger pins expire, later queries rebuild a payload.
Any copied point or descendant that remains alive pins its complete provenance
snapshot. Re-export reuses that snapshot and the same alias slots, preserving
identity even for uncertain bounded coordinates. This deliberately replaces
individual per-vertex expiry with publication-snapshot lifetime; it does not pin
the original geometry payload through point provenance.
Failed admission publishes neither new roots nor provenance slots and preserves
ledger poisoning. Scratch spending and retained owners share the caller's limit;
cumulative charges are never refunded when an owner expires.

Numerical rebases share the original export control and retain one flattened
canonical export-source dependency. Deferred export therefore uses the original
frame, coordinates, errors and sealing policy. The ledger charges that strong
dependency. Certificate-only copies share the existing control and dependency;
physical translations and changed shells detach. These edges remain acyclic.

Numeric-only finite consumers use a synchronous coordinate visitor over stored
vertex order. It applies the same local frame transform and arithmetic policy as
`local_points()`, without point ownership, provenance publication or temporary
point arrays. Coordinates are borrowed only for the callback invocation; callbacks
must preserve the floating-point environment. Repeated indexed reconstruction
queries retain one admitted numeric coordinate buffer for their invocation.

Accepted GeneralVoF state owns retained reconstruction results in a sorted sparse
table keyed by canonical cell key. A cell reserves no result storage when it has
no retained evidence. Records preserve the complete mode-specific result and are
copied with candidate state. Ownership migration transfers them through the existing
authenticated record; geometric remapping and changed phase publications invalidate
them at the same acceptance boundary as the corresponding cells. Cleanup and
face-transport proposals do not erase accepted evidence before collective admission.
Accounting includes table capacity and owned key ancestry. Published Exact result
enclosures are inline values; rational phase inventories retain their separate
ownership accounting.

Finite sweep preparation, materialization and zero-effect checks visit triangles
directly from the recipe's current/departure points and private backcenter. The
canonical triangle sequence, node-ID diagonal rule and construction ordinals are
unchanged. The traversal owns no triangle array and retains existing determinant,
pair-work and geometric construction charges.

Ordinary aggregate construction keeps one arena binding per fragment and stores
owning slot indices in local vertex order. Copies retain those slots; evidence
extensions preserve independent sibling evidence. Arena compaction changes only
the slot-to-record mapping, never local vertex numbering or numerical traversal.

Metric construction freezes all completed channels for one source tetrahedron
before advancing to the next source. Each compact record owns all four weights
and errors, channel-local numerical-row mappings, and ordered flat face rings with
source-face tags. Sharing uses live arena identity within that tetrahedron only.
Pure channels alias bulk; the opposite channel is empty. Records retain no arena,
support, chart or construction-cache pointers. First-attempt coefficient pairs
remain a separate attempt-owned array, absent on retry. The construction arena is
destroyed before channel-major integration. The two integrators consume borrowed
views without changing arithmetic or reference-selection order; compact records
are never replay provenance. Support catalogs and defining-expression caches keep
their measure lifetime.

Storage admission precedes each compact buffer and freeze lookup allocation and
covers capacity replacement and construction/frozen overlap. Descriptor capacity
uses the actual sign-matching source count. The combined geometry observation
includes arena records, slots, incidence, connectivity, compact records and freeze
scratch simultaneously. Failed freezes discard the attempt's compact collector
and release live observations while preserving spent admission and reusable source
inputs. Earlier destruction can reduce live storage while increasing cumulative
construction admission; those quantities must be reported separately.

The private retention census borrows quiescent application roots and allocation
ledgers, including pending pins left by failed walks. It distinguishes physical
strong-owner closure from required point/replay-record closure, keyed by owner,
mode, role and slot. It does not copy pins, lock weak caches, publish exports,
serialize identities, invoke numerical operations or alter production accounting.
Caller-bounded diagnostic scratch owns detached scalar reports; exhaustion makes
the report unavailable without poisoning a production ledger. Live payload is
recomputed from owners and capacities, separately from current ledger tracking,
cumulative admission and external RSS. Inclusive root-group totals overlap and
must not be summed. Required-record bytes are an optimization lower bound, not
reclaimable memory.

The FVM adapter covers explicitly supplied accepted, candidate, prepared, copied
point/trajectory and work roots. Unknown opaque prepared pins make its coverage
incomplete. Application-boundary checkpoints do not observe transient planning
peaks. Mesh/partition fields, native recipes, non-PLIC application buffers,
trajectory container storage, allocator control blocks, MPI storage and unprovided
external roots are outside that report. A private same-image aggregate callback
can inspect borrowed work roots during metric freezing; it permits no numerical
work or reentry. Neither probe establishes global retention without a complete
root manifest. Snapshot paging and ledger-pin ownership changes remain separate
decisions.

Owner exchange bounds transport staging independently of the total serialized
stream. Large exchanges use synchronized cyclic peer rounds, including ranks with
no local work, and keep received bytes in exact blocks. The original Distributor
sender order and each sender's packet order define the result; arrival order does
not. Small or tighter previously admitted exchanges can retain the contiguous path.
Both paths preserve the existing packet framing and total MPI count limits.

Every received stream is audited collectively before allocating decoded payloads.
Segmented decoding then creates the existing independently owned packet vectors
and destroys exhausted receive blocks. Admission includes the whole next packet,
partially consumed blocks, descriptor capacities and replacement overlap; a record
may cross either a length-word or payload boundary. Keeping independent output
packets preserves canonical-authority processing's ability to release a complete
support group's encoded inputs while constructing its geometric replies.

Borrowed input remains intact throughout exchange. The consuming adapter releases
its input only after collective success, so a rejected attempt can retry the same
source. Caller phase/plane imports destroy request, incoming and outgoing owners
at their last authenticated use. Short-lived framed fields can borrow a checked
reader span; data retained beyond the message lifetime is explicitly copied.
Temporary admission is released only after the corresponding owners disappear.
Canonical edge authorities reserve reply slots from the authenticated subscription
count. They retire each completed line in the existing key order after its index,
query buffers and endpoint borrows die. Replies own their encoded bytes; retiring
line storage reduces live admission without refunding cumulative construction.
Retirement diagnostics accumulate over propagation; the optional edge-decode
peak/cumulative fields describe the latest propagation iteration.

Exchange statistics separate planning, transfer and decode peaks from caller-owned
input and MPI/allocator memory. A lower phase peak does not imply lower cumulative
allocation traffic or RSS. Bounded rounds change the communication schedule but
not numerical traversal, publication or rollback. No exchange improvement alone
qualifies a complete cubic-grid motion run or resolves a separate closure-storage
limit.
