# Topology-preserving composite partitioning

`MeshPartitioner` plans ownership over a composite's canonical cell adjacency
and distributes native region providers. Orthogonal regions retain procedural
connectivity, XY-Z regions retain their base topology and layer coordinates,
and explicit polyhedral regions retain ordered polygon loops and owner-oriented
incidence. A rank may own multiple fragments from multiple regions.

## Construction

Build a source on its source rank using an explicit serial communicator. This
avoids the legacy collective composite constructor, which deliberately rejects
globally replicated explicit regions in MPI. All ranks then call the partitioner
with the communicator that will own the fields:

```cpp
using namespace SimpleFluid;
using Partitioner = MeshPartitioner<DefaultTpetraTypes>;
auto comm = Tpetra::getDefaultComm();
std::shared_ptr<const Meshes::MultiRegionMesh> geometry;
if (comm->getRank() == 0) {
    auto serial = Teuchos::rcp(new Teuchos::SerialComm<int>);
    geometry = std::make_shared<Meshes::MultiRegionMesh>(
        regions, interfaces, serial);
}
CompositeMeshSource source(geometry, 0);
CompositePartitionOptions options;
options.ghost_layers = 1;
auto plan = Partitioner::make_plan(source, options, comm);
auto partition = Partitioner::distribute(source, plan);
auto mesh = std::make_shared<MeshHandle<>>(partition);
```

`partition(source, options, comm)` performs planning and distribution together.
Planning does not modify the source; applying a plan checks the original source
identity and geometry revision. Keep the source alive until distribution ends.
The completed partition owns its native geometry independently of the source.
Failures are propagated collectively before subsequent communication or map
publication. Every rank participates, including ranks with no owned cells.
All ranks must call each operation on the same communicator and collective
plan. Distribution rejects mixed plan IDs and differing source ranks before
starting payload exchange.

Already distributed geometry can supply a new partition directly:

```cpp
auto owned = partition.plan().owned_cells();
auto distributed = CompositeMeshSource::distributed(
    partition.geometry(),
    std::vector<uint64_t>(owned.begin(), owned.end()));
auto next = Partitioner::partition(distributed, options, comm);
```

Each rank supplies authoritative canonical cell IDs and local visible geometry.
Their lists must cover the global cell domain exactly once. The source rank
assembles and validates the union of explicit records, including agreement of
overlapping geometry and complete interface coverage. A distributed plan pins
each original shard's identity, geometry revision and supplied ownership list.
Keep those shards alive until distribution completes. Fields still require
explicit transfer to the new maps.

## Ownership policy

`TopologyAware` partitions a graph of logical orthogonal boxes, connected XY
base patches times layer intervals, and explicit cells. Work weights count the
cells in each graph unit. `target_cells_per_fragment` bounds structured unit
size; the planner also limits units relative to the communicator size. A rank
can receive several such units, so retaining topology does not imply one box
per rank or one rank per region.

`CellGraph` permits individual cell assignments while retaining native providers.
`ContiguousCanonical` supplies the previous arithmetic ownership as a comparison
policy. Both expose indexed selections for regions whose assignment is not a
logical box/product.

The plan reports measured balance and the lower bound imposed by indivisible
units. `require_balance` makes failure to meet `imbalance_tolerance` a collective
error. Otherwise the caller can inspect the reported balance, including cases
with more ranks than cells.

Canonical interfaces contribute graph edges and halo adjacency. Periodic images
and coarse/fine subfaces retain the composite's existing orientation and flux
rules. A face has one owning rank: the owner of its canonical owner cell.
Partition cuts retain remote adjacency and are never tagged as physical walls.

## Identity and storage

Region IDs and original native entity ordinals remain unchanged. Canonical
global IDs are stable across ownership policies; local field ordinals are
owned-first, followed by overlap entities. `cell_owner_ranks()` uses this local
cell order. Region execution visits actual owned selections, including gaps in
canonical IDs, and passes the correct field LID to the existing FVM kernels.

The partition's geometry retains global catalog counts. Explicit provider
records contain locally visible topology plus adjacent geometry needed to
resolve halo faces. Queries for unavailable explicit entities fail; the
partition's `MeshHandle` supplies local iteration. Construct that handle from
the partition result, since automatic arithmetic distribution of a resident
geometry would discard its ownership plan.

Compact descriptors are transmitted as axis/base data. Axis arrays and XY base
templates remain replicated compact metadata; shared immutable topology is
interned within each decoded composite while geometry stays independent.
Explicit geometry uses
versioned variable-length cell/face/node records; it is not replicated globally
on destination ranks. Transfer preserves ordered loops separately from identity
matching. The source rank currently constructs the complete planning graph and
destination metadata before distributing graph rows to Zoltan2/ParMETIS. This
root planning storage remains a scaling limit. Distributed source ingestion
also assembles its explicit union on that rank; it does not replicate the union
on destination ranks.

Fields are constructed against the completed partition. Repartitioning produces
new maps and handles; existing fields/operators are not mutated or migrated
implicitly. Explicit polyhedral composite motion retains its static contract.

## Polyhedral geometry

`UnstructuredMesh::PolyhedralTopology` supplies a cell count and ordered
`FaceDefinition` records. Every face loop points outward from its declared
owner; a neighbor consumes the reverse orientation. The cell-list input also
accepts `CellType::POLYHEDRON` with outward `face_node_ids`.

Faces are simple planar polygons, and cells are closed orientable manifolds
with non-self-intersecting surfaces and positive volume. Concave polygons and
cells use signed area/volume moments. Invalid references, duplicate incidence,
open or misoriented shells, degenerate faces and nonpositive volumes are rejected.
This does not add support for curved or self-intersecting faces.

VTU output includes actual polyhedron face streams in ASCII and binary formats.
FVM row capacities follow face incidence, including cells with more than seven
neighbors. Explicit region connectivity is native storage; structured regions
do not acquire compatibility CSR arrays during partitioning or assembly.

## Focused checks

The relevant targets are `testCompositePartition`,
`testMultiRegionPartitionCodec`, `testCompositePartitionOperators`,
`testUnstructuredMesh`, `testPolyhedralDiffusion`, `testVTUWriter`, and the
existing composite/native partition regressions. MPI registrations exercise
composite partitioning and operators on two ranks; the partition target also
has a four-rank registration for empty-rank and ownership coverage.
