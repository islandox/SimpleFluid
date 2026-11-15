# Native mesh input and assembly support

Native construction uses detached `UnstructuredMeshData`: source IDs and optional
algebraic IDs, ordered cell/face/node connectivity, authoritative geometry metrics,
owned-first counts, and a full boundary catalog. `UnstructuredMesh::from_data`
validates and remaps these records without recomputing the supplied metric values.
Existing native constructors retain their behavior.

`MeshPartitioner::partition_native` publishes root-only or owned-shard sources
through sparse mixed-hash directories, the existing ParMETIS graph policy, and
halo requests. It preserves supplied algebraic IDs by default; an explicit option
selects rank-contiguous cell IDs. Physical face identity and existing overlapping
face ownership remain distinct. Empty ranks retain the complete boundary catalog.
The indexed `MeshHandle` path preserves source/algebraic associations, supplied
communicators, and pre-field reordering. Outer-halo interior faces remain interior
when their opposite cell has no local ID. Mesh export retains source `cell_gid`
and separately exposes `algebraic_cell_gid` for native unstructured geometry.

The private STK/Ioss reader converts HEX_8/WEDGE_6 input to detached records. STK
source objects die before redistribution. Parallel readers elect a producer by
minimum incident owned source-cell ID then rank and transfer its entire face
record unchanged. Connectivity, node-coordinate representations and boundary
identity must agree. This preserves the chosen record; warped-interface metric
bits are not promised invariant across different input decompositions.

Input auto mode prefers a complete Ioss basename.N.rank set, otherwise reads a
single base file using LINEAR decomposition; partial sets fail. Empty method
selects decomposed input explicitly. Native factory adoption is a later stage;
the existing factory and STK computational backend remain available here.

Caller-owned preparation is agreed before the next collective in the new native
distribution/handle path. This does not repair dependency-internal collective
failures or every older direct PartitionedMesh constructor path. The I/O bridge
resides in the single static-Trilinos owner; older reused backends without it are
rejected. The structural concepts and compiled-provider boundaries remain intact.
