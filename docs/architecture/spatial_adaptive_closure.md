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

`ClosedContactIndex` owns compact key/domain/revision/owner/image metadata and a
local height-balanced tree. It does not retain expanded geometry or a global leaf
catalog. Startup performs one provider bounds evaluation per cell and O(N log N)
metadata insertion; queries and mutations operate on the tree. Closed overlap keeps
face/edge/vertex equality. Results are deduplicated in canonical key order and
borrow entries until erasure/replacement. Returned vectors belong to the caller,
which must account for overlapping outputs in its enclosing budget. An optional
scratch cap bounds simultaneous result construction. Index live capacity and
construction overlap are capped independently of monotonically accumulated
admission observations. Operations preserve caller rounding and exception flags.

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

The production closure route bootstraps complete physical contact incidence once
through the existing canonical support/edge machinery, then evaluates a spatial
halo. The index used for exclusion is built from the actual normalized incidence
points, not merely the provider hull. This retains legacy edge-index admissions
while the provider hull interface remains a broad-phase building block.

A distributed binary rank-cover tree stores bounds only. It returns overlapping
owner fragments; complete normalized point keys then go directly to those owners.
Exact matches return all associated cell/revision/physical-point incidences, or an
explicit empty completion. Cell records are fetched once per matched owner, key,
revision and session. Bounded application pages, authenticated context/query/page
identifiers and terminal counts prevent missing messages from becoming empty
neighborhoods. Pages and decoded records share one live budget with the retained
index, fans, pair table and cached records.

The evaluator builds unique cell pairs before numerical comparison. Owned/owned
pairs contribute both local requirements; owned/remote pairs contribute only the
owned requirement. Point-independent span/trace work is reused across shared
contacts; physical-point-dependent trace membership is still checked at each
required incidence. Actual performed comparisons consume the contact work cap,
while logical point-incidence pairs are reported separately. Communication and
work placement intentionally differ from the previous hash-authority algorithm;
transport paging never resets admission counters.

The old closure implementation is an explicit private test oracle, never a
production fallback. The standalone production call currently creates a complete
bootstrap for that invocation. Persistent canonical-incidence deltas and hierarchy
frontier updates are the subsequent integration stage; no incremental traversal
claim follows from spatial packet routing alone.
