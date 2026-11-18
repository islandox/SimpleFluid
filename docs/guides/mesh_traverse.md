# Batched Mesh Traversal

The `MeshTraversePlan` framework provides efficient, metadata-aware traversal of mesh entities (cells, faces) across composite multi-region domains.

## Traversal Mechanisms

- **Structured Region Caching:** For regions with structured geometry, the framework caches metadata including origin, extents, strides, and cell counts. Traversals utilize direct quotient-remainder arithmetic, avoiding full topological lookups.
- **Irregular/Explicit Mapping:** Partial or irregular region mappings utilize a cached, sorted indexer. Lookup of faces and cells is performed via binary search over pre-indexed native entity lists, ensuring $O(\log R)$ scaling independent of total interface complexity.
- **No Global Indexing:** The framework deliberately avoids constructing global face-to-native arrays. All indexing operations are local to the region and the canonical entity metadata, minimizing the memory footprint.

## Core Interfaces

- **`visit_regions()`:** Dispatches visitor kernels once per region, providing access to raw constituent coordinates and canonical offsets.
- **`visit_region_geometry()`:** Applies composite affine transformations and resolves incident periodic images.
- **`resolve_face()`:** Resolves a canonical face ID to its constituent native metrics in a single decode, providing a snapshot that remains valid for the duration of the traversal kernel.

## Integration Principles

- **No Implicit Publication:** Scalar/vector gradient and transport kernels use bulk views acquired through the execution boundary. Output publishing requires explicit kernel-level publication to ensure conservation properties.
- **Lease-Based Traversal:** Execution views acquire read leases on the composite and its constituents, rejecting geometry motion or topological mutations until the traversal completes.
- **Memory Efficiency:** Metadata and traversal caches are optimized for rank-local storage. No supplemental persistent connectivity tables are stored outside the provider's native records.

For implementation examples, see the verification test suite in `testRegionExecution` and `testRegionInterfaceIndex`.
