# Region Execution Contract

The region execution framework defines the boundary for computational kernels operating on multi-region mesh domains, ensuring type-safe, metadata-aware access to regional data.

## Execution Model

- **Lease-Based Boundary:** `acquire_execution_view()` establishes a thread-local lease on the mesh geometry, native providers, and topology caches. This boundary rejects concurrent geometric motion or structural mutations, ensuring consistency for the duration of the kernel operation.
- **Raw Constituent Access:** The framework provides visitor interfaces (`visit_regions()`, `visit_region_geometry()`) that dispatch once per region. These kernels operate on raw constituent coordinates and canonical offsets, abstracting the underlying complexity of the multi-region mesh.
- **Snapshot Resolution:** The framework resolves face metrics (e.g., area, normal) through `resolve_face()`, which decodes incident native geometry into an operation-local snapshot. These snapshots are immutable and bound to the geometry epoch of the mesh.

## Architectural Requirements

- **No Dense Lookups:** To maintain a minimal memory footprint, the framework explicitly avoids building dense per-cell or per-face native lookup tables. Spatial and topological queries are performed via compact regional directories.
- **Boundary Consistency:** Structured boundary orientations and irregular buckets are managed through region-specific directory spans. This ensures that canonical face recovery remains efficient, utilizing binary search over sorted native lists rather than global indexing.
- **Numerical Kernel Parity:** Numerical kernels (e.g., FVM diffusion, gradient reconstruction) remain implementation-agnostic regarding the underlying mesh family, relying on the resolved snapshots provided by the execution boundary.

## Integration Principles

- **Safety & Validation:** View acquisition performs pre-flight validation of the mesh and constituent providers, ensuring that all necessary connectivity metadata is present before kernels begin.
- **Exception Unwinding:** View acquisition and kernel execution honor C++ exception safety, ensuring that leases are correctly released even in the event of numerical failures or convergence errors.

For qualification and validation targets, see:
- [Region accuracy qualification](../verification/region_accuracy.md)
- [Batched mesh traversal](mesh_traverse.md)
