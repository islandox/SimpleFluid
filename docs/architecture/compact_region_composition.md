# Compact Region Composition Contract

`MultiRegionMesh` is a compact serial/MPI backend of `MeshHandle`. 

## Architectural Principles
- **Canonical Seam-face Identity:** Preserve the canonical seam-face identity and owner-oriented flux across every field and operator path.
- **Provider Integrity:** Native region providers retain geometry without copying it; child identity/layout/revision checks must run through the existing epoch cache seam.
- **Epoch Management:** Do not treat an unchanged maximum child epoch as an unchanged composite. Topology region identity is separate from material membership and MPI ownership.
- **Storage Efficiency:** Distributed composites replicate only compact axis/base/interface descriptions and construct rank-local field maps. Global explicit constituents are rejected.
- **Consistency:** Coarse/fine seams keep canonical fine faces; coarse cells traverse all covering subfaces, and integrated flux restriction uses signs without multiplying by area fractions again.
- **Periodic Identity:** Periodic geometry must use incident-cell image vectors in both reconstruction and Rhie–Chow stabilization to prevent duplicated or misweighted interface flux.

For implementation, see the `MultiRegionMesh` source and the [region mesh contracts](../guides/region_meshes.md).
