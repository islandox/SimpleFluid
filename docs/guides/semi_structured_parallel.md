# Semi-Structured XY-Z Parallel Execution Contract

The `SemiStructuredXY_Z` framework enables efficient parallel distribution of extruded polygonal meshes, utilizing a hybrid decomposition strategy based on base-cell XY graphs and axial Z slabs.

## Parallel Architecture

- **Hybrid Decomposition:** Ownership is determined by crossing a Zoltan2/ParMETIS-based partition of the XY base-cell graph with balanced axial Z slabs. This avoids the construction of a full 3D graph, maintaining a compact topological footprint.
- **Compact Geometry Metadata:** The XY graph layout and axial Z coordinates are replicated across ranks, while local fields and algebraic maps are distributed based on the partition plan.
- **Deterministic Reordering:** To ensure efficient memory access, the framework computes a deterministic XY-base permutation per rank, grouping XY partitions and applying reverse Cuthill-McKee ordering within the induced graph.

## Configuration & Ownership

- **Factor Selection:** The framework supports flexible XY and Z factor splitting. With automated factor selection, it optimizes for nonempty tensor-product owners while minimizing the total interface cross-section.
- **Empty Rank Support:** The framework accommodates configurations that leave owners empty, provided the `allow_empty_partitions` policy is explicitly enabled.
- **Local Indexing:** Owned cells and faces are assigned dense local IDs, with algebraic GIDs assigned contiguously per rank. Topology-ID mapping is handled through a temporary distributed directory, avoiding a persistent global 3D table.

## Integration Principles

- **Field Synchronization:** Fields bind directly to the native owned/overlap maps. Ghost cell synchronization is handled via standard Tpetra import/export operations, adhering to the canonical ownership established at construction.
- **ALE Stability:** Planar ALE motion is supported for axial Z deformation. The framework enforces collective GCL checks and atomic rollback, ensuring that field storage and geometry remain synchronized through the transaction.
- **Boundary Preservation:** Empty ranks retain global boundary IDs and metadata, participating fully in solver and ALE collectives.

For implementation examples, see the verification test suite in `testSemiStructuredDistributed`.
