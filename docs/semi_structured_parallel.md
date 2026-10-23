# SemiStructuredXY_Z in MPI

`MeshHandle` distributes a polygonal XY mesh extruded through nonuniform Z
layers. Construction retains the supplied geometry object. It does not convert
the mesh to the legacy `Mesh` backend or expand its entire 3D connectivity.

## Ownership and configuration

Construct the same compact geometry on every rank after initializing MPI/Tpetra.
The one-argument const and mutable constructors remain available. The options
overloads select halo depth, XY/Z factors, and the policy for ranks without cells:

```cpp
using Handle = SimpleFluid::MeshHandle<>;
Handle::DistributionOptions distribution;
distribution.ghost_layers = 2;
distribution.allow_empty_partitions = true;
auto mesh = std::make_shared<Handle>(geometry, distribution);
```

The source topology cell ordinal stays `xy_cell + cells_per_layer * z_layer`.
Ownership combines a Zoltan2/ParMETIS partition of the base-cell XY adjacency
graph with balanced contiguous Z slabs. Only XY edges shared by base cells
enter that graph. The same XY assignment is reused in every slab, with
`rank = z_slab * xy_partitions + xy_part`. No expanded 3D graph is built.

Set `distribution.xy_partitions` or `distribution.z_partitions` to choose a
factor; the other is inferred from communicator size. If both are supplied,
their product must equal communicator size. For example, four ranks with
`xy_partitions = 2` and `z_partitions = 2` split both directions. `(P, 1)`
selects XY-only ownership and `(1, P)` selects Z-only ownership.

With both factors omitted, construction considers divisors of communicator
size. It maximizes the number of nonempty tensor-product owners, favors a
true XY/Z hybrid that uses all ranks, then minimizes estimated maximum cell
load and an XY/Z interface estimate. Remaining ties favor the smaller XY
factor. The estimate uses counts, not measured physical areas or graph cuts.
Read the selected factors through `mesh->semi_structured_decomposition()`.
Single-layer meshes use XY partitions; a single XY column uses Z slabs.
Factorizations that leave empty owners require `allow_empty_partitions`, even
when the total cell count exceeds the rank count.

The graph solve runs on a prefix communicator with one or more XY vertices per
active rank. Only the compact XY labels are shared with the full communicator.
One XY part and one vertex per active part have trivial assignments. Backend
labels are canonicalized by minimum base-cell ordinal. Graph partition quality
and repeatability can depend on the backend version; no minimum-cut or
connected-part guarantee is implied.

After partitioning, `MeshReorderingFactory::partitioned_xy_order()` computes
one permutation of the compact XY topology per rank. It groups XY partitions
and applies deterministic reverse Cuthill-McKee ordering within each partition's
induced graph. Every Z layer shares that base topology and permutation; there
is no separate reordering per slab or copied topology per layer. Local owned
cell maps follow the shared XY order within each Z layer, and ghost maps follow
Z-layer order then the same XY order. Reordering occurs before fields or caches
bind to the handle. It changes local cell order only:
source topology cell/face indices, physical geometry, rank ownership, and
owner/neighbor incidence remain unchanged. Algebraic GIDs are assigned after
this local ordering is established. The one-rank view retains its identity ordering.

Halos expand actual cell adjacency by the requested positive number of layers.
Owned cells precede ghosts. A face is owned by the rank containing its canonical
geometry owner. Faces surrounding local/ghost cells remain visible, and local
nodes cover those cells. Partition interfaces remain interior faces.

Both explicit `partition` and `partitions` overrides must be supplied together
and match the communicator rank and size. Geometry, topology, boundary names,
epoch, halo depth, factor selection, and empty-rank policy are checked
collectively before maps are created. A null or inconsistent descriptor rejects
collectively. Empty owners retain every global boundary ID/name with an empty
local face vector.

`DecomposedSemiStructuredTopology` describes the local partition with one XY
layout and Z bands. Inspect it through
`mesh->decomposed_semi_structured_topology()`; the observer is null for the
serial identity view and other mesh families. Radius selections reuse the same XY layout across the bands;
side and axial faces retain their separate extrusion orientations. Exact halos
satisfy `XY graph distance + Z slab distance <= ghost_layers`. This excludes
corner cells that a full product of independent XY and Z halos would add.

The container translates local cells, faces and nodes to source XY/Z indices
and back for geometry lookup. It retains compact base selections and band
descriptors, without a persistent expanded 3D geometry-ID table. The supplied
geometry object remains shared, so geometry epochs and axial motion still refer
to that original object. The global compact XY topology and Z coordinates remain
replicated; local fields and algebraic maps scale with the partition and halo.

## Fields, numbering and output

Owned cells and faces have dense local IDs. Their algebraic GIDs are assigned
as `rank_offset + local_id`, with separate offsets for cells and faces. Owned
Tpetra maps use the contiguous constructor. Ghost maps follow those owner-assigned
GIDs and may be noncontiguous. Owner lookup uses a temporary distributed topology-ID
directory and integer IDs, without a replicated expanded 3D ownership table.

`cell_global_id()` and `face_global_id()` return algebraic IDs;
`cell_geometry_global_id()` and `face_geometry_global_id()` return the source
topology IDs. The optional compatibility `indexer()` describes topology IDs.
Source owner/neighbor relations and oriented normals define flux orientation;
renumbering algebraic IDs does not alter those relations.

The construction-time XY ordering precedes algebraic numbering. A later explicit
`MeshReorderingFactory::selected_cells_first()` operation preserves existing
algebraic IDs while changing local row order, so its resulting cell map may lose
contiguity. That opt-in operation retains its local permutation separately from
the compact decomposition. The topology observer retains construction order;
the handle's geometry accessors apply any later row permutation. Construct
fields and caches after finalizing order.
The tests check map contiguity and numerical correctness; they do not measure
operator speedup.

Fields bind to the native owned/overlap maps and synchronize ghosts through the
usual Tpetra imports. Face fields use the same canonical ownership as geometry.
The serial identity view retains its layered range optimization. Distributed
views use mapped traversal and rank-local metric caches, preserving the native
mesh identity; those caches may materialize compatibility metric vectors.

Distributed VTU topology contains points referenced by owned cells only, with
one consistent local node map for wedges, hexahedra and polygonal-prism face
streams. Empty owners produce zero-point, zero-cell pieces. Serial output keeps
its established node numbering. Geometry epochs and execution leases retain
their existing invalidation and controlled-mutation contracts.

## Flow and motion scope

The native non-orthogonal operator and coupled-flow paths use this ownership
view. Planar ALE supports the existing axial Z deformation, with conservative
old/new-volume storage, mesh-relative transport, absolute-flux continuity,
collective GCL checks, and accepted-state rollback. Optional local volume and
previous-target spans infer their shared selection from ranks that have rows;
zero-row ranks do not imply that a selected input is absent. XY geometry remains fixed.
Non-axial motion, topology changes, and repartitioning during an ALE step remain
unsupported. Distributed mesh support does not establish physical validation of
every turbulence, transfer, or optional multiphysics combination.

## Focused checks

Build the geometry, operator, flow and ALE fixtures with a compatible Debug tree:

```sh
cmake --build --preset GCC-Debug --parallel 4 --target \
  testSemiStructuredDistributed testSemiStructuredDistributedOperators \
  testMeshHandle testMeshReorderingFactory testDecomposedSemiStructuredTopology \
  testPlanarALEMeshMotion \
  testCoupledSolverBackends testBoussinesqPlanarALE testVolumeContinuityModel
ctest --preset GCC-Debug -N -R '^SemiStructured.*_[24]procs$'
ctest --preset GCC-Debug -R '^SemiStructured.*_[24]procs$' --output-on-failure
```

The registered two- and four-rank cases exercise canonical ownership, one/two
halo layers, skew polygons, nonuniform Z, explicit XY-only/Z-only/hybrid factors,
adversarial base-cell numbering, compact topology lookup, contiguous algebraic
maps, factory ordering, a single-layer XY partition, field
imports, analytic gradient/diffusion and conservative transport, coupled flow,
and ALE failure/retry. One-cell fixtures assert `ranks - 1` empty owners; the
two-cell four-rank ALE fixture asserts two empty owners. These cases do not skip
the empty-rank work. Run the corresponding serial tests first. Omitted MPI tests
(for example through `SIMPLEFLUID_MAX_TEST_PROCS`) are not passes.
