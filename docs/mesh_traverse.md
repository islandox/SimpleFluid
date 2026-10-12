# Batched mesh traversal

`geometry/MeshTraverse.hh` provides reusable selections for numerical operations
on cells and faces. Selection and ownership tests run when the plan is built;
execution dispatches a typed operation outside each homogeneous numerical loop.

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
incidence storage is proportional to locally visible incidences. Ordinary
`All`, `Cells`, and `Faces` plans do not allocate it.

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
state. The gradient cache retains its established refresh behavior; this change
does not eliminate its own topology-related work.

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
