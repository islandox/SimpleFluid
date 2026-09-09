# Compact region meshes

`Meshes::MultiRegionMesh` is a static, serial FVM mesh assembled from conforming
regions. `MeshHandle`, `FieldStored`, diffusion/transport assembly, gradient
reconstruction, Rhie–Chow interpolation and the existing pressure–velocity
solvers use its canonical cell/face identities. There is no separate physics or
solver stack. No matrix-free FVM operator is present in this checkout.

## Providers and ownership

`TopologyProvider` describes ordinal incidence, node loops, cell types and the
owner-outward face convention. `GeometryProvider` supplies physical metrics and
an exact compatible ordinal layout. `IsoRegion<T,G>` retains shared const
providers with a unique region name. Compatibility checks include counts,
indexing extents, mesh family, and explicit/factored template identity.
Callers supply compatible numbering; there is no isomorphism discovery.

`RectilinearTopology` observes an immutable shared `OrthoMeshTopo`.
`RectilinearGeometry` owns only three coordinate arrays and an arithmetic
indexer. Different geometries can share the same topology. Geometry parameters
are never expanded into cell/face metric arrays. `native_region()` wraps an
existing Cartesian, `SemiStructuredXY_Z`, or `UnstructuredMesh` in a zero-copy
adapter retaining its exact native object. Extruded incidence and boundary
queries use base connectivity plus axial indices; metrics retain existing
base-plane and axis caches. An explicit adapter observes the original arrays.
The current independent geometry construction example is rectilinear; native
extruded/explicit adapters require identical native template identity.

Topology regions are independent of material labels, field values, equation
coefficients, and MPI ownership. Changing material does not require a new
region. Provider/range observations require their owning region or mesh to stay
alive. Computed entity IDs and metrics are returned by value, never by reference
to scratch storage. Ranges carry independent query state and support nested
queries and random access.

Composite geometry is immutable. Every native child's identity, layout and revision is
checked against its own construction snapshot, including through the existing
`mesh_geometry_epoch()` cache seam. Replacing or moving native geometry now
advances its geometry revision, so an external mutable alias cannot keep a
composite cache apparently current. This strengthens raw replacement detection;
controlled single-region ALE still uses `PlanarALEMeshMotion`. Composite handles
retain no mutable backend and reject stale constituents rather than refreshing
a partially changed composite. Rebuild the composite, fields and operators
after changing constituents. Child epochs are never reduced with `max()`.

## Interfaces and canonical identities

`StructuredPatchInterface` joins rectangular Cartesian exterior patches.
`StructuredPatch` stores the region, native boundary ID, tangential begin/extents,
axis order and reversals. Boundary ID is `2*normal_axis + upper_side`.
Zero extents select the remaining extent of a whole native patch. The interface
maps its first parameter domain to its second with a signed permutation;
`map()` and `inverse()` check bounds and shape. A parameter permutation does
not rotate physical vectors or tensors. Regular correspondence occupies a
fixed-size descriptor, with no per-face pairing table.

`ExplicitConformingInterface` supplies complete one-to-one native ordinal pairs
for two declared boundary patches. It supports irregular/mixed connections,
including a HEX_8 face joined to a triangular-prism side. It stores the input
pairs plus two sorted lookup orders, all proportional to interface size.
Duplicate faces, missing pairs, interior faces, noncoincident shapes and
subdivisions are rejected before the mesh is available for fields.

Default tolerances are an absolute length of `1e-12` in coordinate units and a
relative tolerance of `1e-10`. Position checks scale with face size (the larger
of square-root area and vertex distance from centroid); area checks use area
and absolute-length times face size. Unit outward normals must be opposite to
the relative tolerance. Cyclic/reversed vertex-loop matching checks edges and
shape in addition to areas/centroids. No inconsistent geometry is averaged.

Each stitched pair retains the first native face's geometry and orientation.
Its native owner is the composite owner; the second face's native owner is the
composite neighbor. Both cells reference the same logical face and the same
face-field degree of freedom. A positive integrated flux contributes `+Q` to the
owner and `-Q` to the neighbor. Volume-normalized divergence therefore conserves
its volume-weighted sum. A seam has no physical boundary condition.

Cell IDs use region prefixes. Face IDs use prefix counts with arithmetic
rank/select through removed rectangular ranges, or sorted explicit removals.
There is no cell-to-region array or face-to-region hash table. Nodes remain
region-qualified for visualization, including duplicated coordinates at seams.
Default physical names are `region/native_patch`; intentional name merging
requires `BoundaryNamePolicy::MergeMatchingNames`. Namespaced string collisions
(including ambiguous separators or repeated native names) are rejected. Connected regions form one
pressure problem. Disconnected composites and disconnected native constituents
are rejected in this milestone.

## Storage and traversal

`MeshHandle::faces()` returns `EntityRange`, a sized/indexable range with no
allocation or mutable scratch. Native structured and extruded paths never build
CSR cell-face offsets or expanded face lists, including construction and FVM
execution. Unstructured incidence is borrowed from native stored connectivity.
`materialize_cell_faces()` explicitly allocates compatibility CSR;
`materialized_faces()` is the separately named span accessor. Ordinary field,
operator and solver paths do not request it. The legacy `indexer()` accessor
explicitly requests compatibility lookup tables; ordinary serial handles use
arithmetic IDs, and distributed handles reuse their Tpetra maps after setup.
MPI construction still uses transient native indexing tables while establishing
existing ownership/maps. It never expands implicit cell-face connectivity.

Physical boundary batches in `MeshHandle` remain stored for the existing BC
API. Native extruded boundary batches are factored by base edges and layer
indices. Arbitrary native cell reordering is explicit and allocates permutation
or index tables; it no longer rebuilds implicit connectivity. Arbitrary
composite reordering is rejected. `visit_regions()` provides a batch dispatch
seam for later native-region kernels. Current generic FVM queries still perform
runtime dispatch and range resolution; this milestone does not claim a speedup.

`storage_report()` reports allocated vector capacities and object sizes for
native/shared topology, geometry, interface descriptors, indexing/permutations,
compatibility connectivity, and boundary batches. Shared native topology is
counted once by identity. Allocator metadata, shared-pointer control blocks,
string allocations and container nodes/buckets are excluded from the measured
subtotal. Explicit index-tree storage is separately estimated from payload and
three pointers per node. Required-map estimates report an ID-payload equivalent;
Tpetra may encode contiguous IDs arithmetically and has additional opaque
allocations. These estimates are not RSS measurements or full solver memory.
Fields, sparse matrices, histories, Krylov vectors and preconditioners remain
O(N).

VTU connectivity is constructed only on output request and may be held by the
existing output cache. It is not a persistent solve requirement. Output carries
cell values in handle order and a `topology_region` ordinal; region names follow
construction order. The example reports temporary topology capacities separately
from solve storage. Writer field arrays and serialization buffers are additional
output-only allocations.

## Example, tests and benchmark

```sh
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug --target region_diffusion testMultiRegionMesh testMultiRegionOperators testMultiRegionFlow region_mesh_scaling
ctest --preset GCC-Debug -R '^(RegionProvidersTest|MultiRegionMeshTest|MultiRegionOperatorsTest|MultiRegionFlowTest)\.|^region_diffusion_smoke$'
build/gcc/bin/Debug/region_diffusion /tmp/region_diffusion.vtu
build/gcc/bin/Debug/region_mesh_scaling 8
ctest --preset GCC-Debug -R '^MultiRegionMeshSerialOnly_2procs$'
```

The example solves unit diffusion with analytic `phi=x` and Dirichlet data on
remaining physical boundaries, reporting maximum error and total outward
boundary flux. Tests use least-squares linear reconstruction, assembled
diffusion, backward-Euler/upwind transport and the existing non-orthogonal
correction. Linear exactness tolerances are `2e-12` for reconstruction and
`2e-10` for a converged diffusion solution; they do not assert convergence order
or bitwise identity. Mixed-mesh and pressure–velocity comparisons match physical
centroids instead of assuming face ordering.

The opt-in CSV benchmark doubles the axis resolution through its argument
(default 8, maximum 256). It compares a single native Cartesian block, equivalent
compact regions, and the native block with explicitly materialized compatibility
connectivity/indexing. It reports construction, traversal and assembled diffusion
time, capacities, estimates, and a checksum. No wall-time or RSS threshold is a
unit-test gate. Materialized mode uses spans for its timed traversal; operator
assembly uses the common range API in all three modes. Large cases are never
launched by CTest.

## Deliberate limitations and later milestones

Composite support is serial, static, conforming and limited to Cartesian,
straight extruded triangles/quads and native HEX_8/WEDGE_6 unstructured regions.
Single-region cylindrical/periodic and MPI/ALE paths retain their established
scope; cylindrical and periodic composition are unavailable. General polygonal
native extrusion still supports arbitrary cell-face counts, but composite VTU
and cell-type support currently require triangular or quadrilateral bases.

Later composite MPI must support several regions per rank and a region split
across ranks, with unique canonical face ownership, halos, distributed field
ordering and conservative seam flux tests. Multi-rank construction currently
fails consistently before map setup or communication. Later composite ALE
requires exact child-revision transactions, interface coincidence preservation,
GCL, cache refresh and rollback tests. Nonconforming/mortar/AMI, overset,
automatic extraction, refinement/remeshing, rotated periodic transformations and
new intermaterial laws remain separate work. Unsupported combinations never
flatten silently or run as disconnected subproblems.

`MeshHandle::faces()` now returns a value range instead of a span, and the
handle/provider object layouts have changed. Rebuild downstream C++ consumers;
the export-symbol regression does not certify class-layout ABI compatibility.
