# Native mesh construction and STK input

`MeshFactory` publishes native `Meshes::UnstructuredMesh` geometry through
`MeshHandle`. STK/Ioss is a transient file-input dependency. Programmatic BOX,
CYLINDER, ANNULUS and SPHERE construction uses STK-free node/cell/face records.
The existing STK computational backend remains available while its callers migrate.
Native constructors and physical fields do not own STK data.

## Factory API

```cpp
auto mesh = MeshFactory(database).build_handle<Pack>();
auto mesh_on_comm = MeshFactory(database).build_handle<Pack>(comm);

auto imported = MeshFactory::from_stk_file<Pack>(filename, options, comm);
```

The configured `DomainType::EXTERNAL` route reads `external_mesh_file` and the
configured boundary names. `external_mesh_decomposition_method` optionally
selects the input layout policy. `STKMeshInputOptions` exposes the same policy
for the direct file entry point, explicit boundary-name/ID mappings, and the
automatic boundary-ID policy. No STK types appear in these public options.
An omitted direct-file communicator uses the default Tpetra communicator.
Every participant must enter construction with the same file/options and a
compatible communicator; the factory checks its dispatch and input signature.

Input policy is explicit:

- `decomposition_method = "auto"` preserves a complete existing Ioss
  `basename.N.rank` set for the supplied communicator. If no rank pieces exist,
  a single base file is read with Ioss LINEAR decomposition. A partial piece
  set is rejected rather than replaced by another input.
- An empty method selects Ioss decomposed-file lookup explicitly.
- A named Ioss method, such as `LINEAR`, selects single-file parallel input.

The generated rank suffix uses Ioss's processor-width convention. Input mode
is selected collectively. All Ioss/STK operations use the supplied communicator,
including split communicators, rather than an implicit world communicator.

The input support matrix remains `HEX_8`, `WEDGE_6`, and mixtures of those
volume types. Unsupported populated volume topologies fail collectively.
General native polyhedron support does not imply arbitrary Exodus polyhedron,
tetrahedron, pyramid, shell or high-order-element input support.

## Construction data and geometry authority

`UnstructuredMeshData` is a detached, STK-free construction value. Its records
use source IDs for connectivity, preserve cell and face traversal order, carry
fixed cell types and authoritative metrics, and include a complete boundary
catalog. Owned entities precede overlap entities. Optional explicit algebraic
IDs are distinct from source IDs. `UnstructuredMesh::from_data` validates the
records and remaps connectivity to compact local IDs without recomputing the
supplied centroids, areas, normals or volumes. Existing node/cell and polyhedral
constructors retain their own established metric derivation.

The private STK reader releases its broker, bulk data, metadata and entity
handles before native redistribution. Serial input retains its extraction
order and metrics. Parallel STK extraction can produce different local owners
and loop origins for a shared face. The reader elects a producer by the lowest
incident owned source-cell ID, then source rank, and transfers that producer's
**complete** face record unchanged. Other observers must agree on cyclic face
connectivity, incidence, boundary identity and node-coordinate representations.
This source-authority rule preserves the selected metric values and does not
promise bitwise equality across different input decompositions for warped
interfaces. The ordinary native-data distribution API still requires already
authoritative, mutually consistent shared records.

HEX/WEDGE generation uses the original side ordering and geometry formulas.
In particular, quad-loop origins determine the centroid triangulation. The
faceted cylinder and annulus remain faceted; they are not converted to analytic
cylindrical geometry. Source geometry changes require separate numerical gates.

## Distribution and identifiers

`MeshPartitioner::partition_native` accepts root-only sources or uniquely owned
cell shards, exchanges source records through mixed-hash directories, builds
balanced distributed graph rows, applies the existing ParMETIS policy, and
requests remote halo records sparsely. The small-global-graph fallback retains
the existing direct-placement rule when there are fewer cells than ranks.
It does not broadcast complete source geometry to every rank.

Source IDs, compact geometry IDs and algebraic GIDs are separate. All mapped
source/algebraic conversions are checked against the selected Tpetra ordinal
range. Explicit indexed-handle construction retains `EntityMapping` ordinals
through map construction and supported pre-field reordering. Native factories
request the established rank-contiguous **cell** algebraic numbering; face
identity and the current overlapping owned-face semantics are preserved.
The general distribution API preserves supplied algebraic IDs by default.

A physical partition face may be owned on both incident ranks under the
existing face-field contract. This migration does not replace that contract
with a globally one-to-one face map. Outer-halo interior faces retain their
physical classification even when their opposite cell has no local ID. Every
rank retains the full boundary catalog, including empty patches and empty
partitions. Maps and source associations belong to the wrapper/indexer rather
than duplicate source-ID arrays in native geometry.

Caller-owned local preparation and allocation are agreed before the new
factory/distributor/handle path advances to its next collective. Shared storage
is allocated before constructors enter map collectives. These boundaries do
not claim to repair failures internal to third-party collective implementations
or every pre-existing direct `PartitionedMesh` construction path.

`MeshHandle::export_vtu()` for native unstructured handles retains source IDs in `cell_gid`
and adds `algebraic_cell_gid` for the field-map association. Cell order follows
the field-local permutation. Rank suffixes use the mesh communicator, including
serial meshes created inside a larger MPI job.

## Lifetime and compatibility

Imported geometry is initially fixed. Construction, partitioning and permitted
reordering finish before fields/caches are bound. Reimport/repartition returns
a new mesh identity; it does not mutate existing fields or caches in place.
Geometry epochs and execution leases retain their existing responsibilities.
General unstructured ALE and live repartition/remap are separate capabilities.
Legacy manually paired translational periodicity is not silently translated
into distant ordinary neighbors; native image geometry would be required.
A closed annular seam sharing node IDs remains ordinary interior topology.

`MeshFactory::build()` explicitly materializes compatibility Mesh storage from the
common construction records. Ordinary native construction never allocates that
snapshot or legacy solver workspaces. Existing Mesh and STKMesh APIs and template
defaults remain available in this stage; their deprecation and runtime STK backend
retirement are separate changes. Factory/handle consumers and providers require a
matching rebuild for the added mapping and construction state.

STK/Ioss property construction stays inside the single static-Trilinos archive
owner through a narrow versioned I/O bridge. Reusing an older backend without
that bridge is rejected at configuration. Shared or unwrapped Trilinos builds
compile the same bridge with the I/O component. Making STK optional is a
separate dependency-policy change.
