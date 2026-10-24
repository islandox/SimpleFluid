# Cartesian and cylindrical decomposition

Native Cartesian and cylindrical `MeshHandle` construction uses a tensor rank
grid. The source geometry object and its epochs remain shared. The decomposition
is fixed after construction, including during accepted/trial ALE motion.

## Rank grid and balanced cuts

`DistributionOptions::orthogonal_partitions` optionally supplies three positive
factors whose product equals the communicator size. Otherwise construction
considers factor triples automatically. Logical rank ordering is
`i_part + p_i * (j_part + p_j * k_part)`.

The selection priorities are, in order:

1. Maximize the number of nonempty owners.
2. Minimize the largest owned cell count.
3. Minimize the largest-minus-smallest count among nonempty owners.
4. Minimize the worst physical block aspect ratio.
5. Minimize an estimated cut-area cost, then choose the lexicographically
   smaller factor triple for a deterministic tie.

Cell counts take precedence over physical widths. Along each axis, block counts
are either `floor(N/P)` or `ceil(N/P)`. At each cumulative cut, construction
chooses the feasible count-balanced boundary closest to the corresponding
equal-physical-width target while preserving enough remainder cells for later
blocks. Exact ties choose the lower cut. This is a sequential tie rule, not a
claim of globally optimal geometric partitioning. Oversubscribed axes keep
nonempty one-cell intervals first and empty intervals trailing.

Cartesian physical widths use the coordinate edges directly. Cylindrical widths
use radial distance, angular arc length at the global midpoint radius, and axial
distance. The angular scale is a characteristic width for process-grid selection;
it does not change geometry, cell volumes, or numerical integration.

Use `mesh->orthogonal_decomposition()` to inspect the selected factors and
`axis_cuts(axis)`. Explicit factors are still subject to count and empty-owner
validation. A nonempty tensor factorization may not exist even when the total
cell count exceeds the rank count. Set `allow_empty_partitions` to admit empty
owners; otherwise such a configuration rejects collectively.

## Entity order and algebraic maps

Owned cells use I-fastest order within each block. Faces are grouped by
orientation and then use I-fastest coordinate order, with owned faces before
halo faces. Source owner/neighbor incidence and oriented normals define flux
signs; algebraic numbering does not change them.

On an actual MPI communicator, owned cell and face Tpetra maps use rank-contiguous
algebraic IDs. Integer directory queries resolve ghost IDs from the owner's
local row and rank offset. `cell_global_id()` and `face_global_id()` expose
these algebraic IDs; their `*_geometry_global_id()` counterparts expose source
topology IDs. The optional compatibility `indexer()` uses source topology IDs.

Fields and operators retain native geometry access and share its motion epoch. Calling
`MeshReorderingFactory::selected_cells_first()` later preserves algebraic IDs
attached to physical cells while changing local rows, so its cell map may lose
contiguity. Construct fields and caches after finalizing order.

One-rank identity construction retains its established ordering. Explicit
serial virtual partition/count overrides remain diagnostic compatibility views
with source-ID maps; they are not communicating MPI partitions and do not carry
the contiguous-map guarantee. In real MPI, rank/count overrides must match the
communicator.

## Compact local topology and geometry lookup

`mesh->decomposed_orthogonal_topology()` exposes the construction-local view
for decomposed Cartesian/cylindrical handles. It shares the source topology and
stores the owned block plus disjoint logical boxes for halo cells, faces and
nodes. It retains no per-entity source-ID table, hash map or tree. One-part
serial handles retain their established arithmetic path and return a null view.

| Query | Cost |
| --- | --- |
| Owned cell or face, local ID to source ID and back | O(1) coordinate/stride arithmetic |
| Halo cell/face or local node, local ID to source ID | O(log B) prefix-box selection, then arithmetic |
| Halo cell/face or local node, source ID to local ID | O(B) box membership, then arithmetic |

Here B is the number of relevant disjoint boxes, not the number of cells or
faces. Metadata does not scale with the owned block volume for a fixed halo
layout. Larger halo depths can increase the number of boxes and construction
cost. Required Tpetra maps, field storage, boundary rows and shared coordinate
arrays still scale according to their own contracts; this is not a claim of
constant total process memory.

Owned faces retain orientation-grouped I-fastest order, including periodic seam
faces. Ghost entities and nodes concatenate deterministic boxes. The public
box observers expose coverage; owned-face boxes do not define the flattened
owned-face order across a seam.

Ordinary geometry and traversal queries use this view without materializing
`indexer()` or cell-face CSR. Explicit `indexer()` requests create compatibility
source-ID arrays. Explicit selected-cell reordering retains an optional forward
and inverse local ordinal overlay; the compact view keeps its construction
order, and the handle's accessors apply the overlay. Faces and nodes keep their
order. Storage reports count compact metadata as topology and an optional
reordering overlay as compatibility storage.

The compact view aliases the concrete mesh's topology, so ALE trials and
rollback retain geometry identity, map identity and ownership. Numeric caches
must still track the shared geometry epoch. This local lookup path does not
expand the support matrix of specialized structured traversal ranges.

## Halos, collective inputs and support

Halos follow exact cell adjacency distance, including periodic wrapping. With
multidimensional cuts, this means the sum of axis distances is at most the halo
depth; a full product halo would incorrectly add diagonal corner cells.
Zero halo depth remains available for topology/diagnostic use. Numerical operators
that need neighbors require an adequate halo and retain their validation gates.
Local faces/nodes are the union incident on all local cells. An absent local
neighbor at the halo fringe does not create a physical boundary.

Geometry edges, periodicity, boundary names, geometry epoch, halo depth, factor
selection and the empty-owner policy are checked collectively before maps are
created. Empty owners retain global boundary metadata and participate in map
and field collectives. The cylindrical curved-face normal integral decodes
source face IDs, independently of the algebraic face numbering.

The dedicated geometry and operator regressions run at two, four and eight
ranks, including explicit 2x2 and 2x2x2 grids, nonuniform axes, periodic seams,
empty owners, exact ownership/halo oracles, field imports and conservative
transport. Run the corresponding serial fixtures first. These checks do not
constitute performance benchmarking or qualification of every coupled model.
