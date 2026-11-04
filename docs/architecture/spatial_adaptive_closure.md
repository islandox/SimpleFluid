# Spatial adaptive closure

Closure grading uses complete closed-contact neighborhoods: shared faces, edges,
vertices, nonconforming edge endpoints and supported periodic images. A spatial
broad phase only identifies candidates; it does not establish contact or authorize
refinement by itself. Connected-domain provenance keeps coincident independent
meshes separate.

The discovery foundation lives below native publication. Hierarchy construction
requires closure before a native halo exists, and the published face-neighbor halo
alone does not establish edge/vertex completeness. The provider contract declares
contact normalization and conservative represented-cell image bounds. Unsupported
capabilities are explicit. Cartesian logical intervals are used for additional
culling only when their complete frame matches and their physical embedding is
proved injective. Periodic images retain their physical-to-contact displacement.
Mixed periodic routes retain their existing rejection boundary.

The provider-bounds foundation `ClosedContactIndex` owns compact
key/domain/revision/owner/image metadata and a local height-balanced tree. It does not retain expanded geometry or a global leaf
catalog. Startup performs one provider bounds evaluation per cell and O(N log N)
metadata insertion; queries and mutations operate on the tree. Closed overlap keeps
face/edge/vertex equality. Results are deduplicated in canonical key order and
borrow entries until erasure/replacement. Returned vectors belong to the caller,
which must account for overlapping outputs in its enclosing budget. An optional
scratch cap bounds simultaneous result construction. Index live capacity and
construction overlap are capped independently of monotonically accumulated
admission observations. The directory cover uses a bounded frontier of actual
tree subtrees: it splits every subtree containing more than one owner or domain,
then refines homogeneous subtrees toward a target of eight fragments. Mandatory
owner/domain splits can produce more than eight. Covers are conservative; they
may identify extra candidates, while their owner and domain remain exact. A
homogeneous tree does not require a leaf scan to construct this frontier.
Operations preserve caller rounding and exception flags.

Ordinary coordinate bounds use the immutable canonical world-coordinate view,
including the original pre-rebase source and its captured arithmetic environment.
They enclose the represented convex hull without display-coordinate conversion or
point-export publication. They are not, by themselves, a proof of every contact
accepted by the legacy rounded edge-index arithmetic. Contact discovery must retain
complete canonical incidence or establish the corresponding conservative acceptance
bound before using this broad phase to exclude candidates. A false completeness
claim must never replace a missing remote response or unresolved geometric predicate.

The shared closure-record provider preserves the existing certificate validation,
lossless axis spans, trace provenance, physical-point membership and grading rules.
It supports both the legacy wire codec and direct owning record preparation. The
codec retains field order and scalar bytes. Local preparation admits owned fields
before allocation and shares the existing nested geometry-admission boundary;
outer live-byte observations are not substitutes for those construction caps.
Point-independent trace inventories share immutable owners only when every
ordered stem, depth, split and terminal field agrees. Point-restricted face traces
keep their existing ownership. Records still expose complete trace evidence, and
encoding selects that evidence without changing packet contents. Mutable record
metadata uses stable pages; key lookup borrows each page's complete cell key,
which avoids a second retained ancestry copy.

A new distributed hierarchy bootstraps complete physical contact incidence through
the canonical support/edge machinery and retains its accepted contact session.
The index used for exclusion is built from the actual normalized incidence points,
not merely the provider hull. This retains legacy edge-index admissions while the
provider hull interface remains a broad-phase building block. Production halo
ownership discovery uses `ContactCoverIndex`: it borrows the distinct normalized
point coordinates and interned domains owned by point groups and stores separate
flat leaf and branch metadata. Its bounded frontier consists of actual subtrees;
mixed-domain subtrees split regardless of the soft fragment target. Candidate
copies rebind their borrowed handles to their own point and domain owners before
use. The provider-bounds index remains a separate foundation, not this halo index.

An adaptation plan forks candidate-private mutable indexes from the accepted
session. Immutable identity, geometry-record and incidence bodies may be shared;
the candidate is independently admitted and cannot mutate the accepted session.
The one-time metadata fork and candidate hierarchy map copy may inspect the
accepted local state. Thereafter the hierarchy routes the initial user changes and
closure splits as retired key/revision and inserted-key deltas. A sparse owner
and revision overlay sits over the accepted ancestry directory. Its tombstones
never fall through to an obsolete base entry; addresses are not used as ancestry.
The final state is repacked and registered after convergence rather than once per
refinement pass.

Authority membership always comes from `kernel.surfaces(source_leaf)` in a
transient copy. A retained polyhedral leaf can already have physical support
records whose keys and boundary aliases differ from the mapped authority view;
the cache must preserve those source records and must not bypass the mapping
because the retained surface list is nonempty.

A changed support is reconstructed from its complete, authenticated owner
membership. Dirty supports are fetched in byte-bounded pages and preserve member
arrival order. During replay, each affected tile first retains compact line slots, physical
endpoint slots and receipt bits. A full working layer is materialized when a
reply differs from the ordered endpoint baseline. Endpoint-only replies still
undergo the original polygon, support and error checks, including translated-image
validation and the original inverse-translation arithmetic. Cached point
candidates are assembled in the legacy authority, line-key and image/edge order.
Completed staging owners retire before new incidence rows are allocated.

Changed tiles and their line subscriptions drive layered line replay;
the resulting rows identify all affected owned cells, including an unchanged
coarse leaf whose newly incident point changes its row without changing its leaf
revision. Leaf revision and incidence generation are separate observations.
Unchanged records can therefore remain shared while changed point neighborhoods
and the associated pair requirements are recomputed. Empty owners participate in
the same collective phases. A failed plan discards the candidate and its overlay,
leaving the accepted hierarchy, cache and source observations intact.

The incidence identity pools store full values in pages of sixteen records. Byte,
cell and point keys use templated hashing policies and flat hash-slot lookup with
full-value collision checks; plane, normal and image keys retain tree lookup. Point hashing isolates and
restores the caller's floating-point environment while emitting canonical words. Hash buckets and stable pool slots are private
references, not numerical or canonical ordering. Forks share existing page owners
and copy lookup metadata, then append to a private page instead of changing a
shared page.
Hashing and dynamic-payload accounting are template policies; forks preserve
their configuration. Shared plane payload remains in the separate identity ledger.
Fork overlap includes every current-source pool credit object. A credit required
by an inherited row is transferred from source-only overlap to persistent
borrowed admission, rather than charged twice. Credits originating in an earlier
ancestor remain distinct allocations even when they pin the same immutable page.
These pool pages are independent of PLIC point-snapshot paging. Per-cell point
membership uses sorted pool-slot reference counts, so removal of one contributing
incidence does not remove a point still supplied by another. Canonical contact
rows retain checked 32-bit references, or a 64-bit alternative, into a frozen
directory shared by each changed-row batch. The directory pins only referenced
point pages and their arena credits, independently of the mutable pool lookup.
Logical insertion slots and captured references are separate: a fork may seal a
partially filled page, so insertion slots cannot be decoded by division by the
page size. Rows preserve full physical-identity ordering and the existing
normalization and duplicate checks. Normalized references share the physical
array while every pair is equal; a separate reference array is retained only
when normalization differs. Standalone/manual rows retain their owning adapter.

Each arena independently admits shared point pages once. Published directories
retain the originating arena until their last row/view is destroyed. Forked
sessions separately admit borrowed directory and originating credit-object
metadata, without charging their page payload twice. Failed row publication
destroys partial buffers before returning live admission; cumulative spending
remains unchanged. An unchanged delta constructs no frozen directory.

Resident `PooledTile` records keep only the cell roles local to that resident,
while complete authenticated membership remains in the authority registries.
Normals and image vectors refer to pooled full-precision values that preserve
signed zero; they do not establish authority. Support groups retain their
generation and zero, one or two members inline, with an immutable shared spill for
larger groups. Empty groups retain generation tombstones. The owner-local reverse
support index likewise stores up to two cell slots inline and shares immutable
larger spills. Full member fields and canonical member ordering are preserved.

Persistent tile bodies occupy immutable pages of sixteen records with ordered
flat rings. Actual slots and terminal offsets select checked 32-bit storage or
the 64-bit alternative; cell presence is separate from its index, and emission
ordinals retain their full width. The construction draft is temporary. A complete
pending page batch is sealed before replay can read its bodies.
Candidate-private ordered map entries hold a page reference, initialization state
and optional layer history. Layerless tiles allocate no history owner. Forks
share immutable body pages, independently admit their arena credits, and rebind
entry references. Histories detach only on mutation; changed-entry snapshots
retain the old bodies and histories until contribution removal completes.
Vector-backed replay inputs explicitly pin their layer owner through replacement
or pruning; page-backed inputs retain their page reference.

Retirement releases a private page when no tile/view needs it. If an accepted
source still retains the immutable body, its candidate body credit moves to a
weak-owner table until physical expiry. Re-adoption transfers that existing
credit back without charging the body twice. Failed metadata admission leaves
the original page owner intact. Fork overlap counts distinct source credit
objects and directory metadata; shared history bodies are admitted once by the
candidate rather than added again as source-only metadata. Diagnostics separate
occupied/live records, page/ring capacities, history storage, credit objects and
deferred body storage.

Line replay streams edge edits and batches tile delivery in bounded pages.
Stored subscribers use checked 32-bit fields for support and endpoint slots,
emission and edge when representable, with a full-width spill otherwise; the
wire retains its 64-bit fields. A query materializes the subscribers of one
complete line transiently and preserves the existing arithmetic and ordering. A
certified layer containing only its intrinsic endpoints can be represented
implicitly; its subscriptions and output are reconstructed from the intrinsic
loop when needed. A private v2 line wave checks the complete subscriber identity,
rejects duplicate completions and verifies the completion count for each source.
A completed page alone is not proof that the full wave arrived.

Ownership migration rebinds the candidate session by retiring moved memberships at
old owners and inserting them at new owners, including when the key and revision
are unchanged. Owner routing and dirty support/line state follow the new partition.
A native partition binding may rebind an equivalent cache only after the existing
per-cell native geometry check and with the canonical leaf, provider and support
identities unchanged.

A distributed binary rank-cover tree stores bounds only. It returns overlapping
owner fragments; complete normalized point keys then go directly to those owners.
Exact matches return all associated cell/revision/physical-point incidences, or an
explicit empty completion. Cell records are fetched by matched owner, key and
revision and reused while valid in the candidate session. Private protocol v2
binds messages to geometry epoch, topology, partition, transaction, closure round,
query and page; source identity and the complete geometry observation are also
pinned and checked by the hierarchy plan. Sender, member, completion and terminal
counts are authenticated. Missing messages cannot become empty neighborhoods.
Pages and decoded records share a live admission budget with the retained index,
fans, pair table, overlay and cached records.

The evaluator builds unique cell pairs before numerical comparison. Its candidate
table stores exact paired local record IDs compactly when both fit 32 bits and
uses full-width storage otherwise; hashes never establish pair identity. A fully
admitted sorted buffer restores the original ascending pair order before any
numerical comparison. Dirty-point receipts borrow stable point-map keys and are
retired as their pair IDs are gathered. Empty-local point groups can then retire,
while their old record IDs remain available for requirement cleanup. Owned/owned
pairs contribute both local requirements; owned/remote pairs contribute only the
owned requirement. Point-independent span/trace work is reused across shared
contacts; physical-point-dependent trace membership is still checked at each
required incidence. Actual performed comparisons consume the contact work cap for
that closure round, while logical point-incidence pairs are reported separately. A new geometry
refinement pass begins a new round; paging within a round never resets its work
cap or cumulative admission counters.

These are private representation changes; public exports and reconstruction
behavior are unchanged. The candidate arena admits actual capacities and the
one-time source/candidate metadata overlap and
retains charges for shared bodies once in that candidate. The source-overlap
reservation is released at publication, while the accepted candidate retains its
own live charges. Packet, decoded-record and returned-result lifetimes are charged
while they overlap. Temporary decoded planes use a scoped ledger; canonical
planes adopted by the incidence pools use the session ledger. Retired immutable
bodies that remain pinned by an accepted state keep their admitted credit through
weak-owner tracking. Expired owners are reaped at operation boundaries, so live
statistics and storage can remain conservative until the next reap; cumulative
admissions are never refunded. Typed-container node charges use the actual stored
types; correcting those charges is an accounting change, not evidence of fewer
allocations. Existing opaque provider and PLIC geometry construction and codec
operations retain their own nested admission policies; the outer accounting is
not a bound on all allocator activity or process RSS. `storage()` includes the
retained contact cache in its owned-byte estimate without adding the bound Data
admission a second time.

The old closure implementation remains an explicit private test oracle, never a
production fallback. The standalone closure call still creates a complete
bootstrap for that invocation. The hierarchy session and its incremental replay
are separate from that standalone invocation; neither is evidence of broader
numerical or physical qualification.
