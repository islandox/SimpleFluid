# Topology-Preserving Composite Partitioning Contract

The `MeshPartitioner` framework facilitates domain decomposition for composite meshes, ensuring that regional topological connectivity and canonical entity incidence are preserved across parallel partitions.

## Partitioning Philosophy

- **Topology Awareness:** The partitioner respects the logical connectivity of constituent regions. Orthogonal, structured extrusion, and explicit polyhedral regions maintain their native topological ordering during distribution.
- **Atomicity:** The partitioning process is a collective operation. The source geometry (regional providers) must remain intact until the distribution is finalized, ensuring global consistency in entity ownership.
- **Independence:** The distributed partition owns its native geometry independently of the source provider, enabling decoupled field binding and solver operations.

## Construction Requirements

- **Collective Plan:** The partitioner requires a global partitioning plan constructed across the target communicator. This ensures that the union of regional shards is mapped exactly once to the parallel ranks.
- **Authoritative Source:** The source rank acts as the aggregator for regional topology. All participating ranks must contribute authoritative canonical cell IDs, ensuring the union is globally unique.
- **Distribution Consistency:** The process propagates failures collectively, preventing dangling field maps or inconsistent metadata across the MPI communicator.

## Integration Principles

- **Field Maps:** Repartitioning produces new algebraic maps. Existing fields and operators must be explicitly rebuilt against the new partition; the framework does not perform implicit data migration.
- **Storage Compactness:** Compact region descriptors (axis/base templates) are replicated to preserve topology, while explicit polyhedral cell records are distributed only to destination ranks, maintaining memory efficiency.
- **Persistence:** Canonical global IDs remain stable across repartitioning events, facilitating the management of persistent field states and diagnostic records.

For implementation details and validation, see:
- [Region mesh contracts](region_meshes.md)
