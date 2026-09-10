# Compact region meshes

`Meshes::MultiRegionMesh` presents one logical finite-volume domain through
`MeshHandle`, `FieldStored` and the existing FVM/solver stack. It supports compact
MPI ownership, controlled composite axial ALE, conforming and nested coarse/fine
interfaces, cylindrical regions, explicit translated periodic patches, and
independent Cartesian/extruded geometry. Topology regions, materials and MPI
partitions are separate concepts.

Bulk execution, per-region interface indexing, provider-aware validation and
analytical/scaling qualification are described in
[region_execution.md](region_execution.md) and [region_scaling.md](region_scaling.md).

## Providers and ownership

`TopologyProvider` supplies ordinal incidence, node loops, cell types and the
owner-outward convention. `GeometryProvider` supplies physical metrics in a
compatible ordinal layout. `IsoRegion<T,G>` retains const providers, a region
name, and exact identity/layout/revision snapshots. Counts, indexing extents,
mesh family, periodic dimensions and explicit/factored template identities must
match. Callers supply compatible numbering; no isomorphism discovery occurs.

`RectilinearTopology` shares an orthogonal template. `RectilinearGeometry` owns
three coordinate arrays and computes metrics on demand. `ExtrudedTopology`
owns an immutable `SemiStructMeshTopo` base template, with incidence factored
through layers. `ExtrudedGeometry` retains that template but owns independent
XY coordinates and axial edges. It caches base-cell and base-edge metrics once;
it never expands them through layers. Both geometry types can share one topology
with other independently located/scaled regions. Composite extrusion cells are
currently triangles or strictly convex quadrilaterals in the base plane (WEDGE_6/HEX_8).

`native_region()` also provides zero-copy adapters for Cartesian, cylindrical,
`SemiStructuredXY_Z` and `UnstructuredMesh` objects. A native adapter retains the
exact object. Native explicit/factored topology and geometry adapters must refer
to the same template; the independent extrusion API removes that restriction by
sharing an explicit immutable template object.

Provider and range owners must remain alive while observations are used. Metrics
and computed IDs return by value; ranges carry independent query state and
support nested/random-access traversal without mutable scratch. Sharing topology
does not share coordinates, materials, fields or equation coefficients.

## Canonical conforming interfaces

`StructuredPatchInterface` joins coordinate-normal rectangular Cartesian or
cylindrical exterior patches. `StructuredPatch` stores region, boundary ID,
tangential begin/extents, axis order and reversals. Boundary ID is
`2*normal_axis + upper_side`; cylindrical logical axes are R/theta/Z. Zero
extents select the remaining whole-patch extent. `map()`/`inverse()` validate
bounds and signed permutations. Correspondence remains descriptor-sized.

`ExplicitConformingInterface` supplies a complete one-to-one list for two named
exterior patches. It supports mixed volume-cell types with matching faces,
including HEX_8/prism-side connections. It retains the pairs and sorted lookup
orders, proportional to interface size.

A conforming pair retains the first native face and its outward normal. Its
native owner becomes the logical owner and the second native owner becomes the
neighbor. Both cells reference one face-field degree of freedom; the second face
is absent from logical iteration. A shared integrated flux contributes `+Q` to
the owner and `-Q` to the neighbor. No physical BC acts on a seam.

Interfaces reject invalid/exterior IDs, incomplete or duplicate mappings,
multiple stitching, mismatched area/centroid/orientation, noncoincident vertex
loops, and unsupported subdivisions. Default tolerances are `1e-12` in coordinate
units and `1e-10` relative. Position tolerance scales with face diameter; area
checks use relative area plus absolute-length times face size. Unit normals must
be opposite within the relative tolerance. Inconsistent geometry is never
averaged.

## Nonconforming coarse/fine interfaces

`NonconformingInterface` declares coarse/fine region and boundary IDs and a list
of `CoarseFineFaceMapping {coarse_face, fine_faces}`. The declared mappings must
cover both complete boundary patches. Every coarse polygon must be tiled by
convex planar fine polygons with opposite outward normals. Projected polygon
clipping checks containment and pairwise overlap, and summed areas check gaps.
Polygon areas and centroids must agree with native geometry.

Each fine face becomes the canonical interior face, oriented out of its fine
cell, with the coarse cell as neighbor. The original coarse face is removed;
the coarse cell's computed face range visits every covering fine face. This
creates no cell-face CSR or dense neighbor replacement. The existing gradient,
diffusion, transport, pressure correction and coupled assembly consume the
actual subface centroids, areas and adjacent cells.

`logical_faces(native_face)` returns all logical subfaces. `canonical_face()`
rejects a subdivided coarse face because it has multiple degrees of freedom.
`face_flux_weights()` supplies signed area fractions, normalized by total
canonical subface area, to prolong a uniformly distributed native integrated
flux even when geometry agreement is only within matching tolerance. `restrict_face_flux()` sums canonical
integrated fluxes with the native outward sign. Area fractions are not applied
again during restriction. Interface storage is O(number of declared subfaces).

This supports nested whole-face refinement. Arbitrary mortar/AMI intersection,
fine faces crossing several coarse faces, nonconvex polygons and curved-face
refinement are rejected. Planar faces of cylindrical regions use the same
geometric checks; curved radial/annular faces are not approximated as polygons
to bypass them.

## Cylindrical and periodic composition

Cylindrical adapters preserve native analytic metrics and R/theta/Z indexing.
Their positive-inner-radius restriction remains. Composite angular cell widths
must be below pi so linear HEX VTU cells remain nondegenerate. Native cylindrical
periodic indexing is retained, including node wrapping; open sectors may also
be joined into a ring with conforming angular patches.

An explicitly engaged `StructuredPatchInterface::periodic_translation` declares
`first point + translation = second point`. Zero translation can identify a
physical angular closure. Translated/self-region patch pairs require distinct
incident cells. All surface validation still runs in the translated frame.
No vector/tensor rotation is inferred from a logical index permutation.

`face_center_vector()` and `cell_center_vector()` use the incident cell's
periodic image. Reconstruction, diffusion distances, mesh-quality checks and
Rhie–Chow geometry use those vectors. Periodic faces share one canonical signed
flux and receive no physical boundary condition. Rotated periodic field
transformations and time-dependent translations are outside this support.
Native cylindrical discretization/quadrature is preserved; matching a native
cylinder is not a new physical-accuracy claim.

## MPI ownership and output

On an MPI communicator, construct the same compact region/interface description
on every rank. Construction validates local inputs collectively and compares an
exact serialized compact description, excluding process-local pointers. A
changed constituent is rejected collectively when constructing a handle.
No maps are published after a failed preflight.

Raw `MultiRegionMesh` counts/IDs describe the global logical domain. `MeshHandle`
partitions contiguous canonical cell-ordinal ranges using communicator rank and
size. A rank can own several regions, and a region can span several ranks. The
handle builds local owned/ghost IDs and face maps, with one face owner selected
by its canonical owner cell. Halo construction follows canonical adjacency,
including coarse/fine and periodic seams. Positive ghost depth defaults to one;
manual rank/count overrides are rejected for composites.

Global compact axis/base/interface descriptions are replicated. Expanded 3D
geometry, connectivity, fields and operators are not replicated as a substitute
for distribution. Global explicit `UnstructuredMesh` constituents are therefore
rejected in multi-rank composites; existing single-region partitioned-unstructured
MPI remains supported. The direct `SemiStructuredXY_Z` handle keeps its serial
contract; wrap it in a composite to use compact MPI. Distributed explicit region packets and arbitrary
region-aware load balancing are later work.

Physical names default to `region/native_patch`; collisions are rejected.
Intentional name merging requires `BoundaryNamePolicy::MergeMatchingNames`.
Connected regions form one pressure problem. Disconnected constituents/domains
remain rejected. Region-qualified node IDs are distinct from canonical face IDs.

VTU topology is generated only when output is requested. Each MPI piece contains
owned cells in field order and only referenced output nodes. Cell values and
`topology_region` labels retain their association. Temporary node lookups and
connectivity are output-only; no persistent `UnstructuredMesh` conversion occurs.
The existing output cache may retain requested topology between writes.

## Controlled composite ALE

A mutable `MeshHandle` around a mutable composite can be passed to the existing
`PlanarALEMeshMotion`. It supplies common affine Z stretching/contraction about
the global lower plane on implicit Cartesian/cylindrical/extruded constituents.
Explicit unstructured constituents retain static composite support: general
warped-face geometry is not covered by the axial metric transformation.
Composite X/Y motion, buffered deformation starting
above that plane and changing a Z-periodic length are rejected. Fixed XY
translations and physical angular closures commute with this axial motion.

The composite owns its affine geometry parameters and `GeometryEpochState`.
It transforms volumes, centroids, area vectors and node coordinates while child
providers stay immutable. The existing exclusive motion lease, trial/accept/
rollback, exact swept flux, GCL and quality checks remain authoritative. Every
trial and rollback advances the composite epoch observed by all alias handles;
existing numeric caches must refresh. Exact child snapshots are still checked,
never reduced with `max(epoch)`.

Motion operations and distributed field/solver operations are collective.
Constituent replacement/mutation is outside the frozen-child contract: rebuild
the composite and its fields/operators after changing children. The supported
motion path changes only composite parameters, consistently on all ranks.

Tests exercise old/new-volume scalar transport, constant preservation, expansion
and contraction, rollback after quality rejection, read-only aliases and motion
ownership. Solver-integrated Boussinesq heating tests reuse the existing mass,
volume, continuity and moving-top checks for PISO and every coupled
backend/workspace combination in serial and MPI. No new physics is introduced.

## Storage, verification and commands

`MeshHandle::faces()` is an allocation-free sized/indexable range. Implicit
regions never eagerly create per-cell offsets or expanded face lists.
`materialize_cell_faces()` and `materialized_faces()` are explicit CSR/span
compatibility operations. `indexer()` explicitly requests compatibility tables;
normal serial IDs are arithmetic and distributed lookup reuses Tpetra maps.
Arbitrary composite cell reordering remains rejected.

`storage_report()` separates topology, geometry, interfaces, indexing,
compatibility, boundary batches and objects. It counts shared topology once and
reports vector capacities. Allocator metadata, shared-pointer control blocks,
string heaps and unordered-container nodes/buckets are excluded. Index-tree and
map ID-payload estimates are separate from measured capacities; none is total
RSS. Fields, histories, sparse blocks, Krylov vectors and preconditioners remain
O(N). Output topology and writer buffers are additional temporary storage.

The coupled solver supports assembled and block-composite operators with cached
or streamed product workspaces. Block-composite runs retain scalar sparse blocks
without building a monolithic coupled CRS matrix. This algebraic representation
is independent of geometric mesh composition; scalar face-based matrix-free
transport remains separate work.

```sh
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug --target testMultiRegionMesh testMultiRegionMeshMultiRank testMultiRegionOperators testMultiRegionTransportMultiRank testMultiRegionALE testCoupledSolverBackends testBoussinesqPlanarALE region_diffusion -j 4
ctest --preset GCC-Debug -R '^(RegionProvidersTest|MultiRegionMeshTest|MultiRegionOperatorsTest|MultiRegionALETest|MultiRegionFlowTest)\.|CoupledSolverBackendsTest\.MultiRegion|BoussinesqPlanarALESupportMatrixTest\.AcceptsCompositeAxialMotion' --parallel 4
ctest --preset GCC-Debug -R '^(MultiRegionMesh|MultiRegionTransport|MultiRegionALE|MultiRegionFlow|MultiRegionSolverALE)_(2|4)procs$'
build/gcc/bin/Debug/region_diffusion /tmp/region_diffusion.vtu
mpiexec -n 4 build/gcc/bin/Debug/region_diffusion /tmp/region_diffusion.vtu
build/gcc/bin/Debug/region_mesh_scaling 8
```

Run MPI tests with normal host networking when PMIx cannot open sandbox
interfaces. The serial diffusion example solves `phi=x` on two Cartesian blocks
and reports error, conservation and storage. The opt-in scaling benchmark retains
native compact, composite compact and explicitly materialized compatibility
paths. No wall-time/RSS threshold or percentage saving is asserted.

Linear reconstruction tests use tolerances around `2e-12`; diffusion solutions
use `2e-10`–`3e-10`. Coupled field/flux comparisons use `2e-8`, with integrated
cell continuity below `1e-8`. These establish discrete consistency and
conservation, not convergence order or experimental physical validation. See
[the extension record](region_mesh_extensions.md) for observed results and
[the original implementation record](region_mesh_implementation.md) for dated
initial measurements. Rebuild downstream C++ consumers after interface/layout
changes; symbol export checks do not certify class-layout ABI compatibility.
