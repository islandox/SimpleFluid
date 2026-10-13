# Batched mesh traversal

`geometry/MeshTraverse.hh` provides reusable selections for numerical operations
on cells and faces. Generic plans retain selection and ownership metadata;
structured plans derive ranges from grid dimensions. Execution dispatches a
typed operation outside each homogeneous numerical loop.

```cpp
#include "geometry/MeshTraverse.hh"

auto plan = SimpleFluid::make_mesh_traverse_plan(mesh);
{
    auto traverse = plan.acquire();
    traverse.cells().all().apply(cell_operation);
    traverse.cells().interior().apply(interior_cell_operation);
    traverse.cells().boundary().apply(boundary_cell_operation);
    traverse.faces().all().apply(face_operation);
    traverse.faces().interior().apply(interior_face_operation);
    traverse.faces().boundary().apply(boundary_face_operation);
    traverse.faces().boundary(batch_id).apply(batch_operation);
}
```

## Selection and references

`CellBatchProxy` and `FaceBatchProxy` borrow a `MeshTraverseExecution`, which
borrows its `MeshTraversePlan`. The mesh must outlive all of them. Plans and
executions cannot be moved or copied. Neither the factory nor the plan
constructor accepts a temporary mesh; execution selectors require an lvalue
execution scope.

All selectors default to owned entities. `.local()` includes locally visible
ghost entities; `.owned()` selects owned entities explicitly. These options
are independent of physical boundary classification.

| Selection | Meaning |
| --- | --- |
| Interior faces | Two cells in the underlying topology, including periodic, stitched, coarse/fine and partition interfaces. |
| Boundary faces | Physical exterior faces, including those without a boundary tag. |
| Boundary cells | Cells touching at least one physical exterior face. Corners and patch intersections are counted once. |
| Interior cells | Cells with no physical exterior face. |

Thus all is the disjoint union of interior and boundary for the chosen
ownership scope. Boundary here is deliberately broader than the older
`is_boundary_face()` predicate, which additionally requires a tag. A boundary
operation that needs a prescribed condition must validate the tag separately.
Unknown or locally absent patch IDs select an empty range. Named patch
membership alone does not turn a periodic or stitched face into a boundary.

References expose `id()` in the existing field-local ordinal space, even for
native structured meshes. Cell references provide `volume()` and `centroid()`.
Face references provide `owner()`, `area()`, `centroid()` and `normal()`;
interior references additionally provide `neighbor()`, and boundary references
provide `boundary_id()`. `geometry()` exposes the typed, borrowed provider
adapter for additional metric queries. The compile-time member `is_interior`
permits specialized numerical code:

```cpp
traverse.faces().all().apply([&](const auto& face) {
    using Face = std::remove_cvref_t<decltype(face)>;
    if constexpr (Face::is_interior)
        internal_flux(face.id(), face.owner(), face.neighbor(), face.area());
    else
        boundary_flux(face.id(), face.owner(), face.boundary_id(), face.area());
});
```

An interior face remains interior when a thin halo omits its neighbour.
Adjacency access then returns the mesh's invalid local sentinel. Call
`.require_local_adjacency()` before a kernel that needs both adjacent cell
values. This is local validation: an MPI numerical operation must coordinate
rank-local failures before later collectives. Traversal itself never performs
communication or publishes field values.

## Batches and ordering

`apply(operation)` calls the operation once per selected entity.
`visit_batches(visitor)` instead passes typed `MeshTraverseBatch` views, which
provide `size()`, `empty()`, `is_interior`, and `apply(operation)`. Batch and
entity references expire with their callback. Move-only operations and visitors
are supported without per-entity type erasure.

Nested incidence operations can reuse the outer typed geometry access:

```cpp
traverse.cells().all().apply([&](const auto& cell) {
    traverse.cell_faces(cell.id()).all().apply(cell.geometry(), incidence_operation);
});
```

The two-argument `apply(access, operation)` and
`visit_batches(access, visitor)` overloads validate the same mesh and reuse the
enclosing callback's geometry lease without repeating provider dispatch for
each cell. The borrowed access and any nested views must remain inside that
callback.

By default, all-category traversal follows existing local ordinal order.
Subsets retain their order within that sequence. Homogeneous half-open ranges
avoid storing one ID per entity on structured meshes. Irregular category,
ownership, or patch transitions create additional ranges; metadata can grow
linearly for highly fragmented selections. Plans retain no geometry snapshots
or implicit mesh connectivity materialization. `storage_bytes()` reports an
estimate including retained selections, excluding allocator bookkeeping.

`.independent()` explicitly permits regrouping interiors together and boundary
faces by patch. It does not start threads. Use it only when aliasing, callbacks,
and accumulation order permit reordering. Existing numerical kernels can rely
on earlier writes being visible to later rows; grouping those rows would change
their behavior. Face-to-cell scatter also requires an explicit race/accumulation
policy before parallel execution.

## Structured range specialization

Native `OrthogonalCartesian3D` and `OrthogonalCylindrial3D` providers specialize
the plan layout when public ordinals cover the complete provider in canonical
order. This includes their serial identity `MeshHandle` views. The descriptor
retains only three dimensions and three periodicity flags. Plan construction
does not scan entities or allocate cell, face, or incidence schedules, and
retained plan storage is independent of mesh size.

Cell selections produce arithmetic row/slab ranges. Face selections produce
ranges within orientation blocks, with the normal coordinate varying fastest.
Periodic axes contribute no physical boundary ranges. Ordered traversal,
ascending patch grouping for `.independent()`, thin axes, and the established
owner orientation remain unchanged.

Cell-face traversal derives the six lower/upper I, J, K incidences into bounded
callback-local storage. Classification selects homogeneous runs before the
numerical operation executes; no per-cell heap storage is retained. Geometry
is still read through the leased provider, so the same plan survives supported
geometry-only ALE updates and rollback.

The selected layout remains part of the borrowed access type in nested
callbacks. Canonical owner and neighbour queries use the structured indexer and
topology directly; metric formulas still come from the native provider.

`uses_structured_ranges()` reports whether this specialization is active.
Reordered handles, distributed maps, composite providers, and other mesh
families retain the generic field-local schedules. A structured provider alone
does not imply that public field ordinals have the canonical structured layout.

## Layered XY/Z range specialization

Native `SemiStructuredXY_Z` providers and their serial identity `MeshHandle`
views retain a distinct layered layout. `uses_layered_ranges()` identifies it;
`uses_structured_ranges()` continues to mean the orthogonal six-face layout.
The shared immutable `layered_layout()` contains flat XY cell-edge incidence,
side ownership and patch metadata, and compressed XY runs. Its retained storage
is independent of the number of Z layers. Cells, axial faces and side faces use
their respective native ordinal layouts and repeat arithmetically over Z.

Cell incidence retains flat XY side descriptors (edge IDs, orientation, physical
category and patch) and homogeneous side runs once. Row-local run indices retain
interior order and stable ascending patch order. Borrowed views decode base/layer
coordinates once, then combine the arithmetic lower/upper axial prefix with these
runs, preserving maximal adjacent category/patch coalescing. Independent traversal
uses each row's sorted run indices; patch lookup uses binary search, without
scanning global patch IDs or rediscovering per-face groups during execution. There is no fixed polygon degree,
per-cell allocation, or shared mutable scratch in traversal. Patch grouping
uses boundary IDs rather than incidence slots, including repeated noncontiguous
patches. Exterior classification follows neighbour absence even when a boundary
value slot is inactive. Nested traversal retains the geometry execution lease.
Reordered handles and composite providers retain the generic fallback. This
specialization adds neither semi-structured MPI distribution nor public periodic
geometry support. Geometry-only axial motion may reuse the plan; XY topology
changes require a replacement plan and cache.

### Typed interior cell-face ranges

`execution.cell_faces(cell_ref)` preserves the physical-interior proof carried
by an enclosing typed cell callback. Canonical layered interior cells select a
small direct incidence range: `all()`, `interior()` and `independent()` retain
native face order, while boundary selections are empty. Ordinary operations run
directly in that range; member pointers retain their invocation semantics.
Boundary cells, other providers and the ordinal-only API retain their existing
selection paths. `cells().all()` continues visiting owned ordinals in native
order. Layered LS interior-only geometry construction explicitly builds interior
cells first and boundary cells second, since those rows have no live boundary
callbacks and are independent. Each row retains native incidence arithmetic;
owned-ID row mapping preserves evaluation and legacy ordering. The boundary-aware
pass retains native boundary-cell callback order.

The layout retains compressed interior XY cell runs once and repeats eligible
Z layers arithmetically. Cell references carry their originating plan identity;
ordinary global face references and batches retain their previous storage layout.
Layered incidence references carry resolved adjacency separately. Typed range
creation rejects foreign plans/meshes, invalid IDs and plans without
`AllWithCellFaces`. Plan/access validation remains active on range use, including
empty selections and adjacency preflight. These references and ranges borrow the
enclosing execution/access scope and must remain inside its callback lifetime.
Generic fallback and supported geometry-motion/invalidation policies are unchanged.

The initial typed-range qualification is recorded in
`build/verification/layered-xy-interior-range-20261013/REPORT.txt`.
That qualification found no statistically clear traversal improvement over the
preceding retained-XY implementation on the large cases. LS refresh took about
5–7% longer; Gauss refresh differences were not statistically clear. The report
also retains the original-control comparison and exact correctness gates.
The split-only intermediate is preserved in
`build/verification/layered-xy-split-cells-20261013/STATUS.txt`.

## Cell-face incidence

Classifying a boundary cell does not classify each face of its stencil. An
explicit `MeshTraverseEntities::AllWithCellFaces` plan additionally retains
ordered, locally visible cell-face incidence:

```cpp
auto plan = SimpleFluid::make_mesh_traverse_plan(
    mesh, SimpleFluid::MeshTraverseEntities::AllWithCellFaces);
auto traverse = plan.acquire();
traverse.cell_faces(cell).all().apply(incidence_operation);
traverse.cell_faces(cell).interior().apply(interior_incidence_operation);
traverse.cell_faces(cell).boundary(batch_id).apply(boundary_incidence_operation);
```

These references also provide `cell_id()`, `orientation()` and
`outward_normal()`. The default preserves native incidence order and canonical
fine-subface expansion. An incidence traversal includes unowned incident faces
needed by the selected row; it does not filter by face ownership. Optional
incidence storage for generic plans is proportional to locally visible
incidences. Structured plans derive it on demand. Ordinary `All`, `Cells`, and
`Faces` plans do not allocate incidence storage.

## Geometry lifetime and invalidation

Acquire current geometry for each execution scope and release that scope before
ALE trial or rollback. Geometry-only motion does not rebuild selections.
Composite execution uses the existing thread-affine geometry lease. Direct
guarded native meshes are also pinned across the execution scope; native meshes
behind runtime handles or partition wrappers are guarded during callbacks.
Other legacy mutation continues to require external synchronization.

Rebuild the plan after topology edits, periodic pairing changes, reordering,
repartitioning, or rebinding. `invalidate()` explicitly rejects further use.
Count, backend and map replacement are checked at acquisition and proxy use.
Same-count in-place topology changes have no general revision counter and must
be invalidated by the caller. `geometry_epoch()` is not a topology revision.
Execution retains runtime provider ownership until its borrowed leases release,
so replacing a handle rejects subsequent proxy use without dangling its lease.

Runtime handle provider dispatch occurs once per traversal. Composite metric
queries retain the existing canonical geometry path, including transformed ALE
coordinates and periodic images; its internal region resolution is unchanged.
This API removes repeated selection dispatch, while numerical conditions and
existing provider-specific geometry costs remain part of their kernels.

## Direct numerical batches

`geometry/DirectBatchRange.hh` pairs two equally ordered, contiguous immutable
arrays. `checked(topology, metrics)` validates their lengths once. `apply(op)`
passes `(const Topology&, const Metrics&)` to the operation using a local
induction variable. No mesh query, provider visitation, geometry-index vector,
lease acquisition, or selection predicate occurs in this loop. The owner must
outlive the range. `subrange(begin, count)` has the same valid-bounds precondition
as an unchecked span subview.

The stored pressure-weighted face-flux workspace uses these ranges directly:

```cpp
auto execution = workspace.acquire_execution();
execution.interior().apply(interior_operation);
for (const auto& group : execution.boundary_groups())
    execution.boundary(group).apply(boundary_operation);
execution.zero_flux_faces().apply(zero_operation);
```

Boundary descriptors passed to `boundary(group)` must originate from that
execution's `boundary_groups()`. Field-local face/cell IDs and original boundary
slots remain in the topology records; the paired-array position never addresses
field storage. Normal execution acquires and validates one snapshot, retains
its ownership and native/composite geometry leases, and reads the arrays without
entering generic mesh traversal again. Numerical boundary types and values are
resolved once per group on each flux call, preserving missing/short arrays,
Slip slot availability, and independent pressure-Dirichlet corrections. The
regrouped output must not alias input fields.

`topology_groups()` is shared and retained across geometry-only ALE changes.
Initial construction classifies/counts owned faces, fills final interior and
boundary-group arrays, and retains inactive/untagged exteriors in a separate
zero-output array. Geometry refresh fills category-specific metric arrays from
those retained IDs, then publishes a replacement immutable snapshot only on
success. Shared older snapshots retain their own metrics. Rollback creates a
new geometry epoch and needs refresh even when coordinates return to an earlier
state. Replacement snapshots also retain their gradient cache's traversal plan;
the new snapshot owns independently rebuilt gradient weights.

Topology/partition/order edits require workspace reconstruction. Call
`invalidate_topology()` before an in-place edit; all snapshots sharing the same
topology then reject acquisition and refresh. Counts and backend/map identities
are checked at operation entry, but same-count edits still need explicit
invalidation. The pressure-flux entry aggregates input, topology, snapshot and
lease failures once before gradient synchronization or numerical writes; the
zero-pressure-coefficient fast path retains its previous behavior.

For source compatibility, `face_geometry()` lazily materializes the old ordered
vector only when explicitly requested. `face_batches()` and `traverse_plan()`
are also lazy compatibility paths. The generic plan, if requested, is retained
with topology across epochs. Normal pressure-flux execution uses none of these
accessors. `has_materialized_face_geometry()` exposes whether the ordered view
was requested; shared workspaces observe the same optional materialization.

The original batched comparison and profiling evidence remain in
`build/verification/face-flux-unbatched-20261013/`. Direct-range measurements use
an independent campaign directory and the same original unbatched control.

## Gradient construction and refresh

`CellGradientCache` and `GaussLinearGradientCache` retain an ordered
`AllWithCellFaces` plan from construction through geometry refresh and rollback.
Each rebuild uses owned cell batches and typed incident-face batches; interior
and physical-boundary sample selection is compiled separately. The nested face
operation reuses the current cell's typed access. Boundary slot lookup is
retained by these caches rather than rediscovered during their `refresh()`.

Least-squares normal matrices still accumulate directions in original mixed
face order. Independent inverse applications then fill separate interior and
boundary sample arrays, preserving each array's order. The boundary-aware
cache builds only boundary-cell rows; interior rows retain their established
empty boundary-cache representation. Boundary direction callbacks run in their
original order and remain live during refresh.

The shared least-squares, scalar affine (including Robin), and vector affine
stencil builders also accept a retained traversal plan. Their existing entry
points create an operation-local plan. `TransportGeometryCache` shares its
gradient plan between both stencil builders and across replacement geometry
snapshots. The pressure-flux workspace likewise shares the gradient plan when
it constructs replacement snapshots. Previous snapshots keep their own weights.

`traverse_plan()` exposes the retained plan for cache diagnostics. The
least-squares cache additionally supports a shared plan constructor and
`shared_traverse_plan()` for snapshot owners. Plans must belong to the exact
same mesh and contain cell-face incidence. Topology, partition, and local-order
changes require new caches; geometry-only refresh preserves the plan identity.

All rebuilds construct numeric replacements privately and capture the mesh
epoch before publishing them. A failed callback or geometry query leaves the
old cache data and epoch unchanged. These builders remain local and add no MPI
collectives or implicit field synchronization. Gradient application kernels
retain their existing owned/overlap reads, write order, and aliasing behavior.

Public construction and refresh entry points remain available. Cache object
layouts now include traversal state, so downstream users must rebuild against
the updated headers and libraries. Export-symbol checks do not establish
binary layout compatibility. Matched measurements of the generic migration
and the structured specialization are retained in
`build/verification/traverse-extension-20261013/REPORT.txt` and
`build/verification/structured-traverse-canonical-20261013/REPORT.txt`.
These separate initial cache construction, retained-plan refresh, and cached or
uncached gradient evaluation.

### Structured numerical gradient storage

When the traversal plan has a canonical structured layout, gradient caches also
use compact numerical rows. Least-squares interior rows store six weights and
six neighbour IDs in contiguous arrays. Boundary cells use separate flat arrays
for interior-only and boundary-aware reconstruction. Gauss interior rows retain
only areas, interpolation distances, and neighbour IDs; full boundary records
are stored only for boundary cells. Evaluation still visits owned cells in their
original order and accumulates every contribution in the established order.
Boundary callbacks remain live, and no interpolation reciprocal is precomputed.

The existing `interior_geometry()`, `boundary_geometry()`, `boundary_locations()`
and Gauss `geometry()` interfaces remain available. Structured compatibility
views are materialized on explicit request with `std::call_once`; normal cached
evaluation reads the compact rows directly. Multiple const readers may request
compatibility views concurrently. Cache refresh must remain separate from
concurrent readers, as before. Gauss copies share immutable numerical snapshots;
refresh replaces only the refreshed cache's snapshot. Least-squares caches retain
their move-only interface.

Refresh builds a replacement privately and holds the geometry lease through
epoch capture and publication. Failed callbacks leave the previous data and
epoch unchanged. Structured refresh reuses compact boundary locations rather
than retaining a dense per-face lookup; a requested least-squares compatibility
lookup can be shared across later snapshots. Reordered, distributed, composite,
and other unsupported layouts keep the generic numerical representation.

`uses_structured_storage()` identifies this representation, while
`legacy_geometry_materialized()` reports whether compatibility geometry has
been requested. `numerical_storage_bytes()` estimates native numerical storage,
excluding allocator overhead and explicitly requested compatibility arrays.
The matched cache-layout qualification is retained in
`build/verification/structured-gradient-cache-final-20261013/REPORT.txt`.
It separates fresh evaluation, evaluation after a fixed geometry refresh, and
cache construction/refresh with fixed iteration counts in both variants.

### Layered numerical gradient storage

Canonical XY/Z plans also select variable-degree flat numerical cache rows.
`uses_layered_storage()` and `layered_geometry()` expose this representation.
Least-squares rows retain interior-only weights once and add boundary-aware
weights only for cells with active samples. Gauss rows store compact interior
areas, neighbour IDs and interpolation distances. Gauss topology arrays (row
maps, offsets, neighbour IDs and ordered boundary metadata) are immutable and
shared across geometry refreshes and cache copies; snapshots replace metrics
only. Interior face ordinals are derived from retained XY connectivity without
a duplicate 3D ID array. Refresh visits retained owned rows directly, without
rebuilding incidence groups, payload counts or boundary-slot mappings. Both
kernels preserve owned ordinal order and original
floating-point accumulation and interpolation expressions. Boundary callbacks
remain live. Scratch used while building least-squares rows is reused across
cells and can grow to any supported polygon degree.

Legacy numerical geometry is materialized lazily with `std::call_once`.
Layered caches reuse provider boundary-slot metadata during refresh, including
inactive entries. Least-squares retains the shared dense lookup and includes
its capacity in `numerical_storage_bytes()`. Gauss accounting includes the shared
cache topology once, as well as metric payload. The provider rejects side patch names
`zmin` and `zmax`. Refresh publishes immutable snapshots
only after successful rebuilding under the geometry lease. The orthogonal
compact representation and mapped/composite fallback remain separate.
The matched XY/Z qualification and focused validation are retained in
`build/verification/semistructured-xy-z-20261013/REPORT.txt`.
The follow-up Gauss payload reservation comparison is retained in
`build/verification/layered-gauss-reserve-20261013/REPORT.txt`.
The retained Gauss topology qualification is recorded in
`build/verification/layered-gauss-retained-topology-20261013/REPORT.txt`.
The retained XY incidence/run comparison is recorded in
`build/verification/layered-xy-incidence-runs-20261013/REPORT.txt`.

Layered incidence references resolve owner/neighbour IDs directly from the
provider's existing `SemiStructMeshTopo::side_face()` entries using the decoded
XY edge ID; axial adjacency is arithmetic. The traversal plan stores no duplicate
owner/neighbour table or borrowed topology pointer. Both mixed batches and the
dedicated interior range obtain topology through the active geometry access.
LS directions
use the decoded current/opposite cell pair and the provider's live centroid
subtraction, so geometry values are never retained. Ordinary global face
references, orthogonal paths and mapped fallbacks keep their existing queries.
The split-cell and XY adjacency qualification is recorded in
`build/verification/layered-xy-adjacency-20261013/REPORT.txt`.
The follow-up removal of duplicate adjacency storage is recorded in
`build/verification/layered-xy-direct-topology-20261013/REPORT.txt`.

## Transport system construction

Scalar and vector transport, weighted scalar transport, physical temperature,
and physical momentum builders use the same ordered row/face traversal for
legacy and stored fields. Each row's retained incidence is split into contiguous
runs by interior/physical-boundary/untagged-exterior category and owner
orientation. Category and orientation dispatch occurs once per run; numerical
face callbacks see compile-time flags. Upwind, coefficient, boundary-condition,
and nonorthogonal-treatment choices remain numerical decisions.

`TransportGeometryCache::assembly_geometry()` retains the traversal plan and
shares immutable category schedules through `batches` across geometry-only
refreshes. Refresh copies the existing compatibility topology arrays into the
new snapshot and recomputes volumes; it does not rediscover incidence or rebuild
category schedules. Previous snapshots remain unchanged and stale epochs are
rejected before execution. Uncached builders create operation-local geometry.

Native/mapped geometry access dispatches outside the row traversal. Composite
meshes keep their typed region resolution: a reusable buffer, bounded by the
largest row, holds that row's resolved face metrics while its runs execute.
There is no persistent global resolved-metric table. Canonical coarse/fine
faces, periodic images, field-local IDs, repeated neighbours and numerical
accumulation order retain their existing meaning.

The legacy uniform scalar/vector builders still apply boundary conditions
after the row pass. Their preclassified boundary descriptors preserve the
original mesh boundary-batch iteration order and original in-batch slots,
including gaps removed by ownership or periodic filtering. This preserves
boundary callback order and face-field ownership behavior.

Stored-field explicit nonorthogonal and deviatoric transpose-gradient stress
contributions reuse the same row batches. Existing standalone legacy correction
operators and nonorthogonality detection retain their own execution paths.
Matrix maps, symbolic graph reuse, fixed rows, coefficient formulas and field
synchronization contracts are unchanged.

Assembly geometry selection catches local setup/cache errors and reduces one
status on every participating rank before subsequent assembly collectives or
cached-matrix mutation. This adds one setup reduction per builder on MPI
communicators; serial execution needs no reduction. A supplied geometry cache
does not let one rank skip that collective. This validation does not change
the enclosing operator's existing boundary/source failure policy.

No transport speedup is claimed without matched timing measurements. Exact
metric/coefficient, matrix/RHS, boundary order, retained topology, cache-reuse,
ALE/rollback and two-rank collective tests cover the migration.
