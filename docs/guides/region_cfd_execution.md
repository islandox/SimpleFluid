# Region-Native CFD Execution Principles

This document defines the architectural principles for implementing CFD kernels (transport, gradients) that operate natively on multi-region mesh domains.

## Principles of Native Execution

- **Canonical Geometry Authority:** Kernels must respect canonical field layouts, geometry leases, and existing MPI ownership/halo behaviors. They do not introduce additional retained face-metric tables or global face-to-native arrays.
- **Bulk View Access:** Stored-field CFD operators (gradients, transport) are implemented using bulk views provided by the execution framework. Neighbor/stencil lookups retain original overlap semantics, ensuring that cross-rank halo data remains consistent.
- **Reuse of Canonical Incidence:** CFD kernels reuse existing cached face IDs and incidence orders established by the regional providers. This reuse ensures that numerical consistency is maintained across different mesh families.

## Performance Requirements

- **Local Metadata:** Kernels must maintain a minimal memory footprint. Metadata (e.g., removed structured patches) should be managed as rank-local descriptors, utilizing direct quotient-remainder selection rather than expanded connectivity arrays.
- **Transformed Geometry:** Region-specific kernels (e.g., Gauss kernels) are responsible for applying coordinate transformations and resolving periodic images based on the snapshots provided by the region-native execution boundary.

## Validation & Parity

- **Numerical Invariance:** Implementations must demonstrate numerical invariance under domain decomposition (e.g., identical results for one-region vs. multi-region configurations).
- **Physical Consistency:** New region-native implementations must pass the standard set of CFD conservation gates (mass, species, momentum) and reproduce established transport/gradient operator results.

For implementation benchmarks and qualification records, see:
- [Region performance measurement contract](../verification/region_scaling.md)
