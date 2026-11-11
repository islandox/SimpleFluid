# Mesh handle topology/geometry split

`geometry/MeshHandle.hh` formerly owned topology, geometry, distribution layout,
typed entity IDs, and the traversal access surface in one class. The class is now
a validated composition of two borrowed views over the same authoritative state:

| Component | Responsibility | Storage |
| --- | --- | --- |
| `MeshTopologyHandle<Pack>` | counts, ownership, entity-ID conversions (canonical ↔ field-local ↔ native/provider ↔ global), incidence (`faces`, `visit_cell_faces`, owner/neighbor/opposite), boundary membership/IDs/batches, decomposition views and permutations, Tpetra maps, region-local IDs | none; delegates to the owning `MeshHandle` |
| `MeshGeometryHandle<Pack>` | concrete geometry variant, controlled mutable visitation, geometry identity/epoch, coordinates: cell volume/centroid, face area/centroid/normal/area-vector (and outward variants), stored and image-aware displacements and distances | none; delegates to the owning `MeshHandle` |
| `MeshHandle<Pack>` | owns the concrete mesh variant, the distributed layout, the boundary batches, and the per-handle compatibility caches (lazy indexer, optional CSR) | sole owner |
| `detail::MeshBinding<Pack>` | immutable, validated snapshot: geometry identity, local/owned counts, borrowed map identities/sizes, layout generation, reorder/serial-identity flags | value snapshot; hidden visibility |

`MeshHandle::topology()`, `MeshHandle::geometry()`, and `MeshHandle::binding()`
construct these views on demand. They are cheap values (one back-pointer plus a
shared binding snapshot), so copying or moving a `MeshHandle` never stores a
dangling back-pointer. The views borrow the `MeshHandle`; they and every value
obtained from them must not outlive it.

## Validity

A binding is published by every constructor path and re-published after a local
cell reorder, which also advances `layout_generation`. `MeshBinding::validate()`
checks that the geometry identity exists, owned counts do not exceed local
counts, every owned/overlap/boundary map is present, and map local sizes agree
with the recorded entity counts. `validate(expected_generation)` additionally
rejects a view that predates a layout edit.

Two independent validity tokens are preserved and must not be merged:

- **Layout generation** changes on cell reordering, repartitioning, periodic
  pairing, map/backend replacement, or coarse/fine identity changes. Traversal
  plans and geometry caches must be rebuilt.
- **Geometry epoch** changes on coordinate/volume motion only. Fixed-topology
  traversal plans stay valid; metric caches refresh through their existing
  `refresh()` entry point.

## Compatibility

`MeshHandle` keeps its complete previous public API: every constructor,
`variant()`, `visit()`, `visit_mutable()`, maps, boundary queries, metrics,
region/execution accessors, and VTU export forward to the owning state or the
borrowed views with unchanged behavior. The geometry metric bodies now live in
`MeshGeometryHandle.ipp`; their periodic-image arithmetic, legacy STK branches,
composite region resolution, and compact orthogonal paths are unchanged.

Remaining migration work (tracked, not required for compatibility):

- topology query bodies are exposed by `MeshTopologyHandle` but still defined in
  `MeshHandle`; a later phase moves the storage itself into a shared layout state
  so aliases structurally share one revision counter;
- private helper coupling in `geometry/details/MeshTraverseAccess.ipp` still
  reads `MeshHandle` members directly;
- the constructor funnel still has one `initialize_*` body per family and can be
  collapsed into a request/builder funnel.

## Numerical assembly boundary

`FVM/details/TransportAssemblyGeometry.hh` now resolves one row of
`FVM/details/AssemblyFaceMetrics` per cell and hands the numerical callback a
borrowed `AssemblyFaceRow` instead of the concrete traversal access type. The
large transport kernels therefore instantiate once per `(Pack, MeshType)` rather
than once per concrete geometry backend, while run/category dispatch, ordering,
boundary slots, orientation specialization, and every provider metric formula
are preserved. See [mesh traversal](../mesh_traverse.md#stable-assembly-metrics).

## Measured effect

Measured on the GCC Debug/Release Ninja Multi-Config tree with identical flags
and PCH: `VariableDensityMomentumEquation.cc` drops from 78.3 s / 8.2 GB peak RSS
/ 10851 symbols to 8.0 s / 1.1 GB / 2121 symbols (Debug), and the supported
mapped `weighted_scalar_transport_system` body now compiles once in the FVM
library instead of in eight equation translation units. The receiving
`FVM/.../TransportSystem.cc` grows correspondingly; net measured compile cost
for the two units falls by about 30%. Full numbers and the reproduction recipe
live in the local `build/analysis/measurements.md` artifact.
