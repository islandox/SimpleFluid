# Annular Mesh Factory Contract

The `AnnularSemiStructuredMeshFactory` provides a programmatic interface for generating graded annular polygonal meshes, specifically designed for geometries requiring boundary-layer resolution (e.g., pipe flows, heated annular vessels).

## Mesh Architecture

- **Zonal Decomposition:** The factory produces meshes consisting of triangular-prism bulk cells and hexahedral radial-wall cells. The resulting domain is structured as a `MultiRegionMesh`, facilitating hybrid topological management.
- **Graded Boundary Layers:** The model supports independent radial and axial boundary-layer refinement. Boundary widths follow a geometric series ($w_n = w_0 \cdot r^n$), enabling high-fidelity resolution near solid boundaries.
- **Corner Topology:** To avoid degenerate mesh cells at wall-to-wall intersections, the framework implements optional mitered corner coarsening, ensuring that the local tensor-product overlap is replaced by conforming, swept quadrilateral-prism topology.

## Construction Requirements

- **Domain Geometry:** The domain is defined by finite inner/outer radii and axial boundaries. The framework requires strict adherence to valid geometric ranges ($0 < R_{inner} < R_{outer}$ and $Z_{bottom} < Z_{top}$).
- **Interface Consistency:** The factory preserves the boundary segmentation supplied by the user, ensuring that wall-attachment interfaces remain consistent for boundary-condition binding.
- **Coordinate System:** All spatial inputs must be consistent with the SI units of the solver (metres). The factory maintains polygonal volume integrity; it does not normalize volumes to an analytic annulus.

## Integration Principles

- **Field Binding:** Mesh handles are constructed collectively on the communicator. Fields and boundary conditions must be bound to the produced handles after distribution, adhering to the standard mesh-handle API.
- **Compatibility:** While the generated mesh is polygonal, it is fully compatible with standard CFD transport and gradient operators. However, custom wall-width or spacing choices must remain within the solver's non-orthogonality and aspect-ratio quality limits.
- **Validation:** Programmatic geometry generation is subject to immediate structural validation. The factory rejects unrepresentable coordinates, overlapping stacks, or insufficient clearance to ensure solver stability.

For workflow details and regression coverage, see:
- [Region mesh contracts](region_meshes.md)
