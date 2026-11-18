# Multi-Region Mesh Contract

The `MultiRegionMesh` framework provides a composite structure for assembling disjoint regional meshes (e.g., orthogonal, structured extrusions, swept R-Z) into a unified computational domain.

## Architecture & Construction

- **Composite Topology:** The framework manages the composition of native regional meshes, preserving their local topological integrity. It supports non-orthogonal connectivity, structured interface stitching, and periodic interface identification.
- **Collective Construction:** The mesh must be constructed collectively on the shared Tpetra communicator. Each rank participates in the assembly, ensuring that ownership lists, boundary metadata, and topological seams are consistent across the parallel domain.
- **Provider-Based Ownership:** The mesh retains ownership of constituent regional providers. Native provider records (topology, cell/face incidence) are accessed directly through the composite seaming interface.

## Ownership & Topology

- **Interface Integrity:** Canonical interfaces contribute graph edges and halo adjacency. Periodic images and coarse/fine subfaces preserve orientation and flux rules. A face possesses a single owning rank, identified by its canonical owner cell.
- **Partitioning:** The framework supports topology-aware partitioning, respecting logical connectivity of regional units. Partition cuts retain remote adjacency information, allowing for consistent halo exchange across regional boundaries.
- **Stability:** Canonical global IDs remain stable across ownership policies, ensuring that field data and operators remain consistent after repartitioning.

## Integration Principles

- **Field Binding:** Fields are constructed against the finalized partition. The framework separates region-specific topology from field storage, allowing for compact memory layouts.
- **Geometry Epochs:** The framework enforces geometry epochs to manage cached operator invalidation. Any modification to constituent regional geometry (e.g., via motion) invalidates dependent numerical caches, necessitating a refresh.
- **Implicit Traversal:** The mesh provides allocation-free iteration ranges for structured and region-based traversal, minimizing the need for persistent auxiliary connectivity arrays.

For details on partitioning and execution views, see:
- [Topology-preserving composite partitioning](composite_partitioning.md)
- [Region execution and numerical qualification](region_execution.md)
