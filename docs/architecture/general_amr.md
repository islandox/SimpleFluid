# AMR for extruded, polyhedral, and mixed-region meshes

**Status: Stages 1–5 are implemented and verified within the stated scope; Stages 6–9 remain proposed.** This document
defines the implemented hierarchy, certified geometry and sparse remap boundaries and the remaining
implementation sequence and acceptance contracts. It does not extend the
supported mesh families of the existing solvers or record dated validation results.
Geometry adaptation and numerical VoF qualification are separate deliverables.

The existing [VoF transport and state-ownership contract](vof_transport.md) is
the numerical baseline for these extensions. Its corrected mass-compatible
momentum, microscopic carrier budgets, physical-residual acceptance, and
population identity rules are requirements to preserve, not work deferred until
the new geometry is available. The [durable verification contracts](../verification/vof_solver_contracts.md)
identify the retained regressions; [review follow-up evidence](../verification/vof_review_followup.md)
records measured results and qualification limits separately from this design.
The [Stage 1 verification record](../verification/general_amr.md) covers hierarchy
observations, native partition bindings, cache isolation and solver transactions.
The [Stage 2 verification record](../verification/general_amr_stage2.md) covers
certified convex queries and sparse conservative amount transfer.
The [Stage 3 verification record](../verification/general_amr_stage3.md) identifies
the product-provider implementation, focused evidence and qualification limits.
The [Stage 4 verification record](../verification/general_amr_stage4.md) identifies
the convex hierarchy/exact explicit-provider scope, focused evidence and limits.
The [Stage 5 verification record](../verification/general_amr_stage5.md) tracks
local/mixed provider evidence and its explicit qualification limits.

The objective is one immutable adaptive hierarchy with provider-specific
Cartesian, XY-base × Z-product, and polyhedral ancestry. All providers continue
through native `MultiRegionMesh`, `CompositePartition`, `MeshHandle`, and
`FieldStored`. The optimized Cartesian algorithms remain available through the
existing `CartesianAMR` interface. No new geometric-library dependency is
selected here; external algorithm references explain proposed approaches only.

## Existing support and proposed scope

| Mesh or operation | Existing native capability | Adaptive capability and gate |
| --- | --- | --- |
| Dyadic Cartesian blocks | Immutable balanced `CartesianAMR`, canonical subfaces, adaptive VoF/flow; Stage 1 common hierarchy seam and source-validated native partition bindings | Existing Cartesian behavior retained; Stage 2 adds separate certified geometry/remap utilities |
| Direct `SemiStructuredXY_Z` handle | Serial native execution; direct MPI construction rejects | Use the composite route for distributed adaptation, stage 3 |
| Composite native/independent extrusion | MPI ownership; triangle, quadrilateral, and polygonal prisms | Common-Z whole-column refinement, then local products, stages 3 and 5 |
| Explicit planar-faced polyhedra | Native incidence, variable face loops, partitioned FVM | Certified convex refinement/coarsening, stage 4 |
| Concave explicit cells | Representation of admissible closed polyhedral surfaces | Certified decomposition before adaptation/VoF, stage 9 |
| Mixed native regions | Conforming, nested planar coarse/fine, and translated periodic joins | Independent ancestry with cross-region closure, stage 5 |
| Analytic cylindrical / `SweptRZ` providers | Native geometry and existing supported motion | Retain as static topology participants; refinement or incompatible neighboring splits reject until provider-specific refiners/joins qualify |
| Distributed shard input | Root assembles geometry before partition planning | Distributed closure and owner-to-owner migration, stage 6 |
| Extruded/polyhedral VoF and radiolysis | No support implied by native geometry | General geometric transport/flow, then physical coupling, stages 7 and 8 |

The relevant seams are [`CartesianAMR::adapt`](../../src/geometry/CartesianAMR.hh),
[`MultiRegionMesh::Region`](../../src/geometry/mesh/MultiRegionMesh.hh),
[`MeshHandle::initialize_semi_structured`](../../src/geometry/MeshHandle.tcc), and
[`ExtrudedTopology::cell_type`](../../src/geometry/mesh/ExtrudedRegionProviders.cc).
The last already represents larger polygonal bases as `POLYHEDRON`; AMR still
needs its own simplicity, convexity, and child-quality admission checks.
The present [`AdaptiveVoF`](../../src/FVM/AdaptiveVoF.hh) and
[`AdaptiveVoFSolver`](../../src/solvers/AdaptiveVoFSolver.hh) explicitly retain
`CartesianAMR`; their box stencils and directional transport are not generic.

The initial extensions use a fixed physical domain, one shared timestep, and
geometry frozen during each physical step. Time subcycling needs conservative
flux registers; simultaneous ALE and topology change needs a separate contract.
Neither capability follows from this design.

## Native hierarchy, identities, and ownership

```mermaid
flowchart TD
    C[External physics controller] --> S[Adaptive solver and accepted fields]
    S --> H[Immutable AdaptiveHierarchy]
    H --> P[Cartesian / XY-Z / polyhedral providers]
    P --> M[Native MultiRegionMesh candidate]
    M --> D[CompositePartition and MeshHandle]
    D --> F[FieldStored and existing FVM operators]
    S --> R[Sparse conservative remap and candidate publication]
    R --> D
```

`AdaptiveHierarchy` is the abstract provider seam implemented in Stage 1.
Concrete Cartesian, whole-product and convex-polyhedral implementations own
immutable native geometry and ancestry. The Cartesian implementation retains its
optimized numerical backend. Stage 5 adds a mixed assembly over local product
states and supported hierarchy adapters. Provider
splits, inverse families and physical resolution metrics remain distinct from
solver refinement policy. Native region dispatch follows
[`RegionGeometryView`](../../src/geometry/mesh/MultiRegionMesh.hh), rather than
erasing every cell query.

Stable region identity, material membership, provider-fragment ordinal,
canonical field GID, and rank-local field order are distinct. Materials and
physical patch labels are inherited explicitly; neither region names nor MPI
ownership determine them. A physical region can contain several provider
fragments and span ranks. Unchanged fragments may share their const providers.
The current composite prefix ordinals can change when an earlier region grows,
so they cannot serve as persistent ancestry or physical boundary identity.

The implemented Stage 1 interface is declared in
[`AdaptiveHierarchy.hh`](../../src/geometry/AdaptiveHierarchy.hh):

```cpp
class AdaptiveHierarchy {
public:
    virtual ~AdaptiveHierarchy();
    virtual HierarchyObservation observation() const noexcept = 0;
    virtual const SP<const MultiRegionMesh>& native_geometry() const noexcept = 0;
    virtual AdaptiveCellKey cell_key(uint64_t cell) const = 0;
    virtual uint64_t cell_material(uint64_t cell) const = 0;
    virtual size_t cell_fragment(uint64_t cell) const = 0;
};
```

`HierarchyObservation` retains exact hierarchy, topology and partition identity
tokens independently of generation values, effective resident geometry identity
and epoch, and communicator identity. `AdaptiveCellKey` admits
`CartesianCellAncestry`, factor-specific `ProductCellAncestry` and explicit
`ConvexCellAncestry` provenance; the latter
identifies supplied convex cells without claiming a polyhedral refiner. Physical
region identity, material membership and fragment
ordinals remain separate. `CartesianAMR::plan_adaptation()` and
`apply_adaptation()` retain and validate the exact source observation.

`CartesianAMR::with_partition()` binds an actual completed native partition after
collective source/epoch/communicator validation. Its weak origin observer preserves
exact source identity without retaining global partition scratch. Bound hierarchy,
FVM and solver construction use native ownership; this is not live migration of
an existing solver's accepted state. Live repartition/migration remains Stage 6.
Both canonical and resident geometry must retain their captured epochs. Supported
resident ALE motion invalidates a binding; restoring coordinates does not revive
an earlier observation. The Cartesian numerical backend remains unchanged.

Product hierarchy candidates are implemented in Stage 3; general mixed-region
refinement and a solver `RegridTrial` remain later stages. Stage 2 accepts
caller-supplied convex provenance and explicit certified ancestry overlap relations. Their data must contain geometry/partition state separately from
solver inventories and controller revisions.

Plans pin the exact source identity, communicator, provider observations, and
topology/partition/geometry stamps. Solver trials additionally pin accepted
state and controller revisions. Population schema is an independent semantic
identity: unchanged hierarchy and field dimensions do not validate population
indices or a prepared source trial after solver reinitialization.
Ownership-only changes advance partition
generation independently of topology. Geometry epochs remain the numeric-cache
revision described by [`MeshHandle::geometry_epoch`](../../src/geometry/MeshHandle.hh).
Keys must have checked, collision-free equality; a hash alone is not identity.
Provider layout and revision checks retain the existing
[`IsoRegion::validate_static`](../../src/geometry/mesh/RegionProviders.hh) contract.

## Implemented certified geometry and sparse remap

[`ConvexCellGeometry`](../../src/geometry/ConvexCellGeometry.hh) retains an
immutable exact convex shell from represented binary64 inputs. It certifies
outward planar faces, convexity and closed manifold topology before clipping.
`GeometryHalfspace` means `normal dot (x-anchor) <= offset`; a local anchor and
separate finite offset retain translated microscopic cuts that a rounded point
on the plane cannot encode. Clips retain both exact sides, section geometry,
per-parent-face apertures and local volume/first moments. Positive pieces never
vanish through a geometric tolerance; invalid or unrepresentable geometry rejects.
Bulk product overlap factors into actual XY polygon area times Z overlap; tilted
phase cuts retain the full three-dimensional intersection.

`CertifiedGeometryValue` gives a finite converted value and outward error bound.
Positive volume must remain representable. A nonzero first moment below the output
range may have a zero-straddling outward interval with `exact_nonzero` retained;
the exact internal moment remains available for further geometry. Checked narrowing
rejects disappearance of a known nonzero value. Binary64-format moment queries
exercise this contract even when the host's `long double` has a wider range.

[`AdaptiveOverlapPlan`](../../src/geometry/AdaptiveOverlap.hh) takes explicit
source/target bindings and unchanged/refine/coarsen relations. It certifies actual
intersection geometry, per-cell disjointness and complete coverage exactly.
Catalogs may contain only the complete ancestry families needed locally. This is
not a global key directory or distributed refinement planner.

[`ConservativeAMRRemap`](../../src/FVM/ConservativeAMRRemap.hh) binds the exact
native endpoint identities, maps, epochs and plan observations. Sparse owner
exchange authenticates every supplied source and target certificate cell,
including inactive remote siblings: actual native shell geometry must agree
exactly, and region/material/typed ancestry metadata must match its owner's binding
by lossless bytes. Native shell extraction uses certified outward orientation and
exact edge-vertex propagation. Analytic curved providers and translated periodic
joins reject instead of being approximated by a planar node hull.

The caller supplies physical gas and carrier volumes independently, plus accepted
geometry declarations. `Plic` requires a valid mixed-cell plane; `PureGas` and
`PureCarrier` explicitly declare accepted rounded-pure geometry. Only an explicitly
empty geometric minority with positive physical volume within the actual-cell
consistency bound may use bulk-overlap weights. A missing mixed-cell plane rejects;
a resolved positive tiny clip retains its spatial placement. Paired number/moles
use identical carrier weights. Signed/nonnegative extensives, phase densities and
zero-carrier consistency remain checked, with no clipping, carrier floor or
coverage normalization. Output is replaced only after all collective gates pass.

Dimensionless ratios and checked mantissa/exponent products avoid intermediate
overflow and silent underflow. Global signed/L1 budgets use per-budget integer
binary exponents, supported double communication and deterministic compensated
scalar accumulation; diagnostics retain their scale instead of overflowing a
physical double total. Physical tolerances remain independent of remap scaling.

Native owner certificate exchange currently pads local/imported shells to the
global maximum shell payload width. It exchanges no global solution array, but
highly heterogeneous large shells can inflate storage. Variable-length packing
and a global ancestry directory remain Stage 6 gates. The Stage 2 utilities alone add no new refinement provider; Stage 3 supplies the
whole-product provider below. Neither stage adds a general-mesh VoF solver or
live accepted-state repartition. The optimized Cartesian numerical/remap path
remains separate.

## XY/Z products

An extruded cell key is `(BaseCellKey, AxialIntervalKey)`. The base key records
its original polygon and refinement template/path; the interval key records its
original Z interval and dyadic path. Side ancestry is base-edge segment × axial
interval; cap ancestry is base-cell leaf × axial plane. Shared coordinates come
from the same ancestral construction. XY-then-Z and Z-then-XY refinement must
produce identical final product keys and geometry.

The Stage 3 implementation is [`ExtrudedAMR`](../../src/geometry/ExtrudedAMR.hh).
Its `ExtrudedAMRGrid` supplies represented root XY nodes/base loops, boundary edge
labels, Z edges, optional compatible triangle reference edges, physical region and
material IDs, depth and quality limits. `adapt(refine_base, refine_z, coarsen_base,
coarsen_z)` and the plan/apply pair construct immutable candidates collectively.
`base_keys()`, `base_edge_keys()` and `interval_keys()` expose independent ancestry;
`BaseEdgeKey::Kind` separates `Root` edges from `TemplateInterior` edges, so
identical creator/ordinal values cannot alias a boundary edge and an interior
refinement seam. Child paths retain that namespace through further refinement.
`cell_geometry(gid)` queries an authoritative exact cell on demand. Root embedding,
conformity, complete immediate sibling families and stale source observations are
checked before publication. Failed candidates preserve the source hierarchy.

The existing `ExtrudedGeometry` now optionally retains
[`ExactExtrudedConstruction`](../../src/geometry/mesh/ExactExtrudedConstruction.hh).
Represented roots and exact affine mean recipes define base points and common Z
coordinates. Native cell/face volumes, areas and centroids derive from that exact
construction. Display/VTU nodes may round independently and never define overlap
geometry or change the parent domain. Admission still requires a valid distinct
display layout, exact and displayed strict convexity, and representable signed
native center-to-face/axial metric differences. These checks are explicit
construction-provider requirements; the exact constructor does not delegate its
geometry definition to the ordinary display-polygon constructor. Ordinary
binary64 providers keep their existing path.

`has_exact_construction()` identifies this capability, and
`certified_cell_geometry()` returns a retained actual cell on demand.
`ConvexCellGeometry::from_native_cell()` dispatches to that authoritative provider;
`native_cell_shell()` rejects constructed geometry rather than exporting its
rounded node hull. Versioned construction descriptors bind the full extrusion
topology, recipes and active addresses. Native partition packets and configuration
signatures retain those descriptors. The raw recipe API rejects invalid parents,
nonfinite roots, incompatible active layouts, malformed packets and foreign
retained topology. The raw `make()`/`decode()` contract certifies the DAG, layout,
individual convex cells, metric viability and packet framing; it assumes an
already valid, certified embedded base. It does not establish global embedding
of arbitrary raw provider input. `ExtrudedAMR` certifies root embedding explicitly
and preserves it through exact templates and conformity closure. Native packet
decode preserves that descriptor rather than independently admitting an arbitrary
mesh. These are direct provider contracts, separate from refinement.

`ExtrudedAMR::with_partition()` binds an actual completed native partition after
source/revision/communicator validation. Its observation identifies the effective
resident geometry. Fields, traversal and remap use that resident handle; manually
substituting a canonical geometry observation does not authenticate a partition.
This is a construction binding, not live accepted-state migration.

Owner authentication now tags represented native shells separately from exact
constructed certificates. `encode_exact_certificate()` and
`decode_exact_certificate()` losslessly carry retained rational shells without
public Boost types or display-node conversion. Product keys encode both factors
by exact bytes. Every supplied source/target certificate and inactive sibling
still authenticates against its owner's geometry and ancestry/material binding.
`MeshToMeshTransfer` and planar ALE reject constructed providers until their
node-hull/geometry-transform paths gain an authoritative construction contract.

The compact claim concerns provider, hierarchy and partition descriptor storage:
base/edge/point recipes plus common axial data, with exact cell certificates
created only on demand. `storage_counts()` and `storage_report()` separate those
retained allocations from maps and temporary certificates. The Stage 3 remap
bootstrap may explicitly assemble bounded complete-family catalogs for a query;
it does not retain a global base×layers shell cache or weaken Stage 2's exact
source/target coverage checks. Product-family distributed proofs, a global owner
lookup and variable-length certificate packing remain Stage 6 work. Repeated
refine/coarsen histories and layer-count changes have dedicated storage tests;
the verification record identifies their observed outcomes and limits.

Across two snapshots, product ancestry has a factor partial order: refinement
requires each base/Z factor to remain equal or descend, with at least one strict
change; coarsening reverses that relation. Explicit `ProductIntersection` handles
opposite strict factor directions, such as refined base/coarsened Z. It never
pretends those cells have a single scalar parent. Within one active catalog,
comparable base paths and comparable Z paths imply positive product overlap even
when their directions cross; those active combinations reject before geometric
surrogate shells can hide the conflict. Explicit ancestry pairs are enumerated
on demand. The target-selected overload treats an empty target list as zero
records; the unfiltered overload enumerates all requested snapshot relations.

Parents and sibling families are factor-specific. XY coarsening requires the
complete base sibling family over compatible axial intervals; Z coarsening
requires complete interval siblings over compatible base leaves. There is no
single octree-style parent for every independently refined product cell.

Stage 3 is limited to nonperiodic whole products with a conforming base repeated through
common Z layers. Deterministic triangle bisection uses compatible root refinement edges
and conformity closure. XY changes affect whole columns; Z bisection affects a
selected interval across the entire base. This directly retains the layout of
[`SemiStructuredIndexer`](../../src/geometry/mesh/SemiStructuredIndexer.hh).
Hanging base edges cannot be inserted into its whole-edge adjacency contract.
Strictly convex quads use four children: with edge midpoint `m_i` and the
deterministic mean `c` of the four vertices, child `i` is `(v_i,m_i,c,m_(i-1))`
in cyclic order. Construct shared boundary midpoints once; require every child
to pass convexity, positive-area, and quality checks. Larger strictly convex
bases triangulate from the boundary vertex with the smallest stable key, then
use triangle bisection. Save the original polygon, template, and complete
inverse family for coarsening. Template depth is not a claim that every
physical length has halved.

Stage 5 represents local refinement as connected base patches × contiguous Z
runs, sharing identical base templates and axial data and combining adjacent
runs with identical layouts. Reuse the product representation of
[`CompositeRegionFragment::ExtrudedProduct`](../../src/parallel/CompositePartition.hh).
Avoid one provider per 3D cell. Arbitrary layer-local patterns cannot promise
one global O(base + layers) representation; storage also includes unique local
templates, product descriptors, and their interface ancestry.
[`LocalProductAMR`](../../src/geometry/LocalProductAMR.hh) retains independent
base/interval keys and an interned axial selection for each active base leaf.
Connected compatible base components form templates; contiguous intervals form
run views. Factor marks name source product keys. XY or Z coarsening requires
complete immediate factor siblings on the unchanged other factor. Saved exact
construction recipes preserve the original domain through inverse families.
This implemented local-product extension does not change the Stage 3
whole-column adaptation contract.

`SemiStructMeshTopo` shares immutable base connectivity. Exact product providers
share base construction/metrics and an axial coordinate DAG independently of
axial selections, with lightweight layer/indexer and contiguous-run views.
`ProductKeyView` shares base/Z key vectors rather than copying ancestry into
every run. State-local interning and merging of compatible adjacent runs avoid
a global strong cache. Native packets catalog each unique base topology, exact
base, axial value table and axial selection once, then retain per-run references
and view descriptors. Display nodes do not replace authoritative constructions.

Refined wall vertices remain on the original polygonal boundary. Projection to
an analytic cylinder, domain-volume renormalization, and unrequested remeshing
would change the physical domain and are outside this contract.

## Convex polyhedral refinement and shared topology

Stage 4 admits planar-faced convex cells only after oriented-halfspace,
non-self-intersection, closure, and quality certification. Split the actual
parent with its three axis-aligned bounding-box midpoint planes, yielding up
to eight nonempty convex children. Store the occupied octants, construction
version, and parent/child ancestry; do not require eight children or replace
the physical cell by its box. Each child's coordinate spans are bounded by
half the corresponding parent box spans.

[`PolyhedralAMR`](../../src/geometry/PolyhedralAMR.hh) implements this stage for
one native region without joins and one physical region ID; per-root materials
and physical patch labels are retained independently. Input must already be a
valid embedded native planar-convex mesh with complete canonical root residency.
Root import certifies each actual
ordered atomic cell shell and declared shared-face incidence; it is not arbitrary
global mesh admission. Mixed native regions/joins remain Stage 5 and reject.

`PolyhedralAMRGrid` supplies the native snapshot, optional per-root physical
region/material arrays, depth/quality limits and explicit `maximum_cells` and
`maximum_face_tiles` bootstrap budgets. The cell budget checks conservative upper
refinement growth with overflow protection. The face budget bounds overlay pairs
within shared ancestral surfaces before large expansion. Exhaustion rejects the
whole candidate before publication. A final lossless exact canonical descriptor
comparison enforces collective root/candidate identity even when ordinary
explicit meshes have equal layout and different coordinates.

`adapt(refine, coarsen)` uses source leaf ordinals and persistent parent keys.
Only the complete recorded immediate family in the original source is eligible
for coarsening; grandchildren, chained eligibility and refine/coarsen conflicts
reject. Plan/apply retain the exact source observation. No-op calls retain the
snapshot, while successful changes publish the hierarchy and native geometry
together after every check succeeds. `with_partition()` binds the actual native
resident observation, separately advancing ownership without changing ancestry.

Canonical planning constructs a bounded replicated explicit hierarchy using a
local `SerialComm`. Distributed fields use `CompositeMeshSource`, native
partitioning and a real retained partition binding. The ordinary direct MPI
constructor's rejection of globally explicit regions remains intact. Sparse
resident providers do not prove scalable per-rank hierarchy storage: canonical
root/planning geometry is still replicated, under the explicit budgets above.

Closed-volume halfspace clipping is the relevant geometric primitive; the
[CGAL clipping contract](https://doc.cgal.org/6.1.1/Polygon_mesh_processing/group__PMP__corefinement__grp.html)
distinguishes clipping a bounded volume from merely splitting its surface.
The implemented cutter uses the repository's exact rational predicates and
constructions. That reference is algorithm background, not a new dependency or
qualification claim.

Construct one canonical subdivision of every parent face from both incident
cells' cut patterns before assembling children. Both sides reference exactly
the same atomic polygons with opposite incidence. Every new edge vertex must
be propagated into all incident face loops; coordinate-tolerance matching of
independently assembled children is insufficient. Coarse neighbors traverse
the resulting atomic subfaces. This internal conforming face overlay is new
explicit topology surgery, distinct from arbitrary mortar/AMI interpolation
and from extending the cross-provider interface API.

Certify positive finite child measures, planar simple faces, oriented closed
manifolds, containment, pairwise interior disjointness, parent volume and first
moment recovery, and complete parent-shell coverage. Discard only proven empty
intersections. A positive sliver that violates quality or representability
rejects the candidate; its volume cannot disappear. Exact rational midpoint
planes need not be binary64 display coordinates. Native display/metric viability
remains an explicit admission limit; no unspecified retry changes the split.

Coarsening restores the saved parent geometry after its complete recorded
immediate sibling set is present, without grandchildren or changed material
membership. Retain boundary tiles still required by finer neighbors. Never
recover a parent through a convex hull. This geometry hierarchy has no solver
refinement policy. A future solver integration must separately protect mixed
interfaces, reconstruction neighborhoods, motion forecasts, wakes and promotions.

[`ExactPolyhedralComplex`](../../src/geometry/ExactPolyhedralComplex.hh) groups
faces by stable `Root` or `Cut` surface ancestry, with canonical orientation and
physical patch declarations. A cut surface is identified by its creator parent
and axis, not a rounded plane or coordinate tolerance. Opposite incident patterns
contribute to one exact atomic overlay; same-side overlap, missing coverage or
inconsistent surface/patch declarations reject. Canonical tile identities ignore
loop start and redundant exact collinear edge segmentation. Every exact edge vertex
is propagated to incident loops, and the reconstructed atomic cell shell must
match its immutable canonical geometry. Coarsening therefore restores the saved
parent shape while retaining covering tiles required by a surviving fine neighbor.

`ConvexCellGeometry::from_native_cell_atomic()` imports authoritative atomic faces
in native traversal order, including constructed cells. It does not assume that a
canonical domain certificate's face count/order equals its tiled native topology.
The ordinary domain certificate remains the geometry used by conservative remap.

The optional exact `ExplicitRegionProvider` overload keeps the legacy constructor
and record representation available. Exact cells use `POLYHEDRON` connectivity
when atomic tiling changes face counts. The compiled adapter authenticates native
records against the exact complex; native face area comes from the authoritative
area conversion, not the norm of already rounded area-vector components. Legacy
record range checks remain explicit admission requirements.

Versioned native packets retain requested cells plus incident-owner geometric
support, their faces/nodes, and only metric records for remaining outer neighbors.
This geometric support closure does not enlarge field halos. Exact certificates are available only
for fully resident cells. Empty/sparse providers keep the global catalog and
boundary schema independently of residency; their public factory requires the
complete schema explicitly, even when the global schema is empty. Unnamed
exteriors retain invalid boundary IDs and empty names; topological exterior
status is independent of labelling. Omitting a sparse factory schema rejects. Exact descriptor bytes participate in configuration
identity. All supplied source/target remap catalog cells, including inactive
siblings, still authenticate against their actual native owners without coverage
normalization. Exact-shard distributed reassembly is explicitly rejected until
Stage 6 can preserve its authoritative semantics; ordinary legacy shard merge
continues through its existing path.

[`UnstructuredMesh`](../../src/geometry/mesh/UnstructuredMesh.hh) supplies ordered
face loops and manifold incidence, but its signed moments and face-loop checks
do not certify a new nonoverlapping refinement partition. Likewise,
[`evaluate_mesh_quality`](../../src/geometry/MeshQuality.hh) uses absolute normal
distances: add signed visibility/convexity checks and shell-intersection tests.
Positive volume alone is insufficient.

Use cell-local coordinates and dimensionally scaled absolute/relative length,
area, volume, and first-moment tolerances. Construct shared coordinates once
from canonical ancestry; use adaptive predicates or a certified fallback for
ambiguous signs. Inconclusive emptiness, intersection, or coverage rejects;
tolerances must neither erase positive measures nor renormalize coverage.

Coverage compares intersections with the actual represented, boundary-clipped
query geometry. A nominal spacing product or AABB volume is not the volume of
that query. Use local-coordinate differences and compensated positive summation.
New kernels need bounds for their own clipping/predicate arithmetic, not the
Cartesian box helper's bound transplanted unchanged or a tolerance proportional
to global origin. Retain rejection of genuine missing, doubled, or empty
coverage at large offsets.

## Mixed-region closure and boundaries

The verified Stage 5 implementation below is a
geometry/native-field/remap extension; it does not enable a general VoF or flow
solver. Same-fluid joins keep one connected native pressure domain. Contacts
retain separate native meshes and external coupling maps; coincident geometry
never creates a pressure join implicitly. Physical region, material, patch,
fragment and persistent ancestry identities remain separate.

[`MixedAMRGrid`](../../src/geometry/MixedAMR.hh) accepts zero or more
`LocalProductAMR` states, supported `CartesianAMR`, `ExtrudedAMR` and
`PolyhedralAMR` hierarchy adapters, and retained static native participants. An
optional first-product convenience entry is not a required dummy participant.
`MixedPatchJoin` names participants and physical patch names, so accepted
adaptation transactions rebuild joins from candidate native patches rather than
requiring callers to update changing run ordinals. Typed marks distinguish
product XY/Z factors, Cartesian block parents and saved convex parent families.
No-op transactions retain observations; real topology changes advance topology
and partition generations from the source. Plans pin the exact source
observation; rejected candidates leave the source unchanged.

Adaptive participant imports retain each exact hierarchy observation and its
canonical geometry identity/epoch. Both canonical and resident provider guards
must be valid before rewrapping regions; a fresh `IsoRegion` cannot revive a
moved source or discard its affine transform. Later participant motion makes
Mixed observations visibly stale and rejects geometry queries, plans and
adaptation. Returning to old coordinates through a new epoch does not restore
compatibility. This retained dependency metadata is O(participants) and is
included in the assembly metadata storage breakdown.

Each provider retains independent ancestry. Provider-local admission includes
closed face, edge and vertex neighbors. Comparable dyadic base-edge and axial
traces allow at most one generation difference, even when a larger physical
span ratio is configured; unrelated template levels are not compared as physical
lengths. Exact directional AABB spans enforce the configured physical ratio.
Whole and local product `contact_violations()` queries derive contacts from base-edge prefixes,
shared vertices/T-junctions and closed axial addresses without a persistent
expanded cell/certificate catalog. Mixed assembly rejects a violating local
state before native publication when provider-local closure is unsupported.
The initial sixteen-cell fixture retains its nineteen-cell local XY and
seventeen-cell local Z candidates. Repeated boundary-touching marks
can reject atomically instead of silently publishing excessive gradation.

Cross-participant closure compares exact physical AABB spans in each direction
against the configured ratio, using actual canonical same-fluid adjacency,
including low-level `grid.interfaces` joins; named joins are not a bypass gate. Supported coarse incident
cells refine under provider depth/quality checks and explicit mixed cell,
interface-tile and closure-round bounds. Native shared-node incidence checks
internal supported edge/vertex neighbors. Across different providers, a
conservative span envelope over each canonical joined face's closed AABB
neighborhood certifies possible corner scales. Authoritative exact cell/face
bounds select candidates; unrelated interior cells are excluded. If that
seam-local envelope cannot prove the bound, the transaction rejects. This can
reject an overcomplex contact requiring a sharper local seam proof. It does not weld
coincident independent/contact domains or claim arbitrary mixed-corner closure.
Contact incidence/pair work has an explicit bootstrap allocation/check budget. Conservative expansion preflight can
reject before a provider allocates a large candidate. Static refinement requests
and exhausted/incompatible closure reject the complete transaction. Static
canonical participants keep native provider identity/epoch, local ordinals and physical
membership. Their presence does not imply PLIC support for analytic curved or
nonconvex cells: conservative remap requires every participating cell to have a
qualified planar convex certificate, or rejects before publishing output.

The existing [`NonconformingInterface`](../../src/geometry/mesh/ConservativeInterface.hh)
retains its one-fine-region contract. The new
`MultiProviderNonconformingInterface` declares one participant per fine provider,
with an explicit list of distinct complete `FineNativePatch` records. A provider
may contribute several native patch IDs to a single coarse-face mapping; their
physical labels are not merged. Each record retains its native patch count and
complete face-ID membership. The descriptor maps each selected whole coarse face
to native `RegionFace` tiles across those providers. Exact face certificates prove coplanarity, containment, disjointness,
area and first-moment coverage and opposing orientation. A one-tile equality join
uses the same path for run caps. No mortar polygon or synthetic flux face is
created. Crossed XY/Z traces without a complete nested native tiling reject;
this implementation does not promise arbitrary anisotropic seam closure.
Even a locally graded state can have an unsupported crossed run grouping:
a fine native patch spanning several Z ranges may need partial use under
different coarse providers. Complete-patch rules still reject that grouping.
The raw contact diagnostic therefore covers some states that are not admissible
native assemblies; it is a diagnosis gate, not permission to bypass native
coverage or publish unjoined fields.
`RegionInterfaceDirectory` remains derived lookup data, never ancestry authority.

Every canonical fine tile stores one integrated flux, in its retained native
owner orientation. Coarse incidence visits all covering tiles.
[`restrict_face_flux`](../../src/geometry/mesh/MultiRegionMesh.hh) sums signed
integrated fluxes using retained native addresses, independently of area ratios
or floating-point vector dot products. Prolongation uses the same discrete sign
and positive area ratios; an unrepresentable ratio rejects explicitly. A
range-only `hypot` fallback rescues finite nonzero area vectors whose ordinary
squared norm underflows or overflows, preserving ordinary-range arithmetic.
A genuinely zero vector retains zero scalar area; this does not admit a
zero-area face to a new join.

Versioned native packets retain an independent per-region native patch-count
catalog captured from complete source topology. Interface declarations cannot
redefine a complete fine patch by omitting tiles and decrementing their own
counts. Decoding checks complete procedural topology counts, available explicit
face records and global entity bounds. Sparse explicit shards keep the catalog
without acquiring remote geometry; repacketization and configuration signatures
preserve it. Grouped fine-patch membership adds explicit O(interface tiles)
metadata to this bounded bootstrap; it is not compressed into a universal
base-plus-layers seam bound. Each native patch must be completely and uniquely
covered, and absent/duplicated patch or face declarations reject. A sparse packet
is a descriptor from a validated source, not an independent proof admitting an
arbitrary global embedded mesh.

Constructed joins are enabled only for interfaces qualified by the exact nested
path. Unrelated ordinary legacy joins do not invalidate qualified constructed
joins elsewhere. Authoritative canonical face certificates close joined cells,
including an ordinary coarse provider whose face is tiled by constructed fine
providers. Display-node hulls are not a fallback. Materials and physical patch
labels remain inherited rather than being inferred from region names or rank.
Ambiguous duplicate persistent keys reject collectively; the assembler does not
renumber physical regions to hide missing provider-root provenance.

`MixedContactDomains` retains separate connected assemblies and explicit contact
patch references. It validates references but supplies no contact constitutive
law or numerical exchange operator. Existing connected-fluid admission remains
in force. Structured translated-periodic joins keep their existing matching
subdivision and image-vector rules. New exact local-product/polyhedral joins
are nonperiodic; translated/rotated constructed and nonconforming periodic joins
remain unsupported until separately qualified. Analytic curved subdivision
remains unsupported.

Stage 5 canonical assembly and complete-family overlap catalogs are explicitly
bounded replicated bootstrap work. Ancestry-family indexes enumerate overlap
relations, including independent product-factor directions and empty target
selections; they do not search every source/target geometry pair. Native
partition providers remain sparse, but that does not qualify per-rank hierarchy
scaling. Exact explicit-shard reassembly and live migration remain Stage 6 gates.

`MixedAMR::storage_breakdown()` exposes uniquely accounted native assembly
providers/derived lookups and owned mixed metadata (participant arrays, joins
and interfaces) separately from retained participant reports. Those reports overlap
through shared providers and must not be summed. Local-product/whole-product
reports include their planner state; Cartesian/polyhedral reports explicitly
exclude private planner ancestry. Resident partition geometry is separate, and
partition maps/control-block/allocator overhead are not a complete accounting.
These are capacity estimates, not a whole-process retained-memory total or RSS
measurement. Stage 6 must qualify actual distributed payload and memory bounds.

## Conservative remap and physical resolution

Exploit unchanged leaves, ancestor/descendant containment, and complete sibling
unions to find donors. Do not run a global all-pairs intersection search. For
source cell `i`, target cell `j`, and old PLIC gas halfspace `H_i`, retain the
intersection geometry needed to compute

$$
V_{ij}=|C_i\cap C_j|,\qquad G_{ij}=|C_i\cap C_j\cap H_i|.
$$

Bulk XY/Z overlap factors into polygon intersection area × interval overlap.
A tilted PLIC phase cut requires the actual 3D clipped volume; multiplying a
single section area by height is not equivalent. Target gas volume is the sum
of `G_ij`; conservative amounts use the fraction of their declared carrier in
each intersection. Restriction sums extensives; intensive values derive from
transferred extensives and target measures. Detect missing/excess coverage,
absent carriers, and nonfinite weights rather than normalizing errors away.

Keep sparse donor residual closure, nonnegative inventory bounds, and signed
momentum from [`AdaptiveVoF::remap`](../../src/FVM/AdaptiveVoF.cc). Low-Mach remap
transfers separate liquid/resolved/dispersed mass or moles, population number,
dissolved/donor stock, phase energies, and physical momentum including drift.
Constant-density `MixtureMass` is not a replacement for that state. Recover EOS
variables only after conservative transfer and retain occupancy, energy,
inventory, and physical continuity acceptance gates. New interior face fluxes
require reconstruction/projection; retained integrated fluxes restrict by sign.

Canonical phase inventories can remain positive when the published fraction
rounds to zero or one. Compute low-Mach carrier storage from retained phase
volumes and accepted source/flux increments, never solely from `(1-f)*V`.
Preserve the existing bounded-remnant remap rule: when phase geometry cannot
locate a phase because the published fraction is pure (`f<=0` or `f>=1`), but
the corresponding retained phase volume is positive and within its consistency
bound, distribute its paired inventories through the same bulk-cell overlap
before recombination. Do not classify the
carrier as absent, give number/moles inconsistent weights, or use this fallback
for a material geometric mismatch. Each new provider must qualify that rule
with its own represented-geometry error bound and retain rejection above it.

Reuse sparse application, quantity semantics, coverage reporting, and stale-map
checks from [`MeshToMeshTransfer`](../../src/fields/MeshToMeshTransfer.hh).
Its present [setup](../../src/fields/MeshToMeshTransfer.tcc) supports neither
general polyhedron/polyhedron nor extrusion/extrusion intersections and
replicates source geometry. It is not the proposed geometry builder. A scalar
bulk-overlap transpose does not supply reverse PLIC transfer weights.

Use physical directional spans and face-normal distances to select resolution
and stability. `V^(1/3)` hides thin or long cells. A conservative cell-diameter
bound is acceptable for general polyhedra; XY and Z resolution remain separate.
Retain 24–32 cells across a 1 cm resolved bubble as an initial target, subject
to anisotropy and stencil guards. This is not DNS or application qualification;
measure shape, curvature, transport, and coupled-solution convergence.

## General VoF and flow backend

Stage 7 adds an independent backend using actual polyhedron clips, section
polygons, phase apertures, and physical-neighbor stencils. Solve each PLIC
offset from clipped volume. Qualify inferred normals against oblique planar
interfaces on skew cells; a volume-fraction least-squares gradient alone is
not assumed planar-exact. Reject insufficient or ill-conditioned stencils.
Keep the current Cartesian reconstruction and optimized directional algorithm.

The proposed unsplit transport prototype uses canonical face triangulations,
consistently traced shared vertices, signed swept-volume simplices, and a
correction matching the projected integrated face flux. Partition swept pieces
by actual donor cells and then by their PLIC planes. Shared swept surfaces must
agree; independently normal-extruded face prisms can double-count transport.
This follows the algorithm direction in
[Owkes and Desjardins, sections 3.2–3.5](https://ctflab.mae.cornell.edu/papers/Owkes%2C%20Desjardins.%202014.%20A%20computational%20framework%20for%20conservative%2C%20three-dimensional%2C%20unsplit%2C%20geometric%20transport%20with%20application%20to%20the%20volume-of-fluid%20%28VOF%29%20method.pdf);
their unstructured discussion is a starting point, not a qualified implementation.

Let `Phi_f` be the owner-oriented projected volume transported over one step
[m3], `s_cf` its outward incidence sign for cell `c`, and `DeltaG_c`, `DeltaC_c`
the signed physical gas/carrier source increments [m3]. One canonical flux
ledger must satisfy

$$
G_c^\mathrm{new}=G_c^\mathrm{old}-\sum_f s_{cf}F_{g,f}+\Delta G_c,
\qquad F_{g,f}+F_{c,f}=\Phi_f,
\qquad \Delta G_c+\Delta C_c=\sum_f s_{cf}\Phi_f.
$$

The carrier has the analogous update, so total occupancy remains the fixed
cell volume. Qualify source-free transport first; the existing Phase 22 Stage 3
nonzero physical volume sources need a verified geometric/source update satisfying
this ledger. Do not assign skew faces to a dominant Cartesian axis or reuse
the Cartesian Weymouth–Yue correction. The present directional backend retains
its frozen indicator and source correction; the unsplit backend must derive
its own source update. Fractions and phase inventories share the accepted
ledger, but geometric and momentum time integration remain distinct.

### Phase budgets and compatible momentum

Every donor must cover outgoing gas and complementary carrier with its actual
extensive inventory, including microscopic positive traces below PLIC offset
resolution. The Cartesian sweeps repair only roundoff-sized excess on the
shared canonical face partition, preserve total transported face volume,
recompute the complement, and recheck owned donor budgets collectively.
Cancellation-resistant minority-phase updates preserve the unexported tail;
there is no cell-fraction or inventory clipping. The unsplit backend needs its
own donor decomposition and capacity proof with an explicit arithmetic bound;
the directional helper alone does not qualify it. A material excess rejects.
Apply each corrected canonical subface once, with no second area weighting.

Retain the implicit upwind momentum contract in
[`ConservativeVoFTransport`](../../src/solvers/details/ConservativeVoFTransport.hh).
For the source-free, nondrifting case each component solves

$$
M_c^{\mathrm{new}}u_c^{\mathrm{new}}
+\sum_f s_{cf}F_{m,f}u_{\mathrm{upwind}(f)}^{\mathrm{new}}
=P_c^{\mathrm{old}}.
$$

`M_new` is final extensive mixture mass [kg] and `F_m` is signed integrated
face mass [kg]. Their identity `M_new + sum(s*F_m) = M_old` gives the implicit
equation its constant-velocity, passive-bound and conservation properties up to
the physical residual and roundoff. Reusing
phase swept volumes with explicit accepted-old donor velocity is incompatible
with the existing split phase update and is not a permitted adapter shortcut.

Incompressible flow derives mass from fixed phase densities and the corrected
phase-volume ledger. Low-Mach flow instead uses final liquid/species/population
inventories and their actual mass fluxes. Its velocity is the continuous/liquid
velocity, with physical momentum `P = M*u + sum(W_H2*n_k*w_k)`. Account for
drift face/cell impulses separately. Use physical source impulses, never species
molar flux inferred from geometric volume or source velocity inferred from a
tiny fraction. Passive velocity bounds apply to the source-free, force-free
fixture, not general forced or drifting flow.

This reuse retains first-order implicit upwind diffusion and the momentum solve
cost. Geometric PLIC accuracy does not establish second-order momentum/species
accuracy; qualify temporal and spatial convergence separately.

### Microscopic carriers and numerical acceptance

Passive quantities retain final carrier storage plus outgoing carrier flux in
their implicit equation. For positive physical rows, optional left
equilibration divides directly by the diagonal, without reciprocal formation,
a carrier floor, or inventory deletion. An attainable inner relative tolerance
and the unchanged Belos true-residual contract govern the scaled solve. Final
acceptance still uses the compensated residual of the original unscaled amount
and canonical face-flux equation at the requested physical tolerance. A global
maximum-diagonal bound must not become an unattainable mandatory inner target.

Retain bounded physical-defect corrections, overflow-safe norm/RHS scaling,
compensated accumulation and one shared initial/correction iteration budget.
Exhaustion or invalid data rejects. Positivity, global extensive budgets and
pressure/divergence acceptance remain independent; signed quantities use
absolute-amount conservation scales. Qualify subnormal carriers and forced
correction/exhaustion, not just ordinary capacities or scaled Krylov convergence.

### Curvature and flow qualification

Use physical-coordinate surface fitting for curvature with explicit conditioning
and stencil gates. Qualify nonorthogonal pressure correction, balanced surface
tension, hydrostatic equilibrium, viscosity, and geometry-based timestep bounds
before claiming flow support. Cartesian height columns cannot be manufactured
from cell boxes. Initially retain one timestep for all active leaves.

Use independently integrated mixed-cell fractions and finite-duration pressure,
phase-volume and parasitic-velocity checks; binary cell-center masks can have
zero curvature. Demonstrate actual phase passage across seams. Existing curved
moving-seam coverage is at moderate density contrast; high-contrast static
coverage protects a uniformly fine interface neighborhood. The high-contrast
curved-seam gate remains open: retain the disabled strict Cartesian oracle and
equivalent new-provider criteria without relaxing pressure or velocity gates.

## Transactions, radiolysis, and checkpoint boundaries

Prepare geometry, partition/maps, remapped state, operators, coupling maps, and
registry changes privately. Pin source stamps and revisions throughout; catch
rank-local failures and reduce status before subsequent collectives. Publish
only after all geometry, coverage, quality, conservation, EOS, and continuity
gates pass, using preallocated nonthrowing swaps. A failed candidate leaves the
accepted hierarchy, fields, time, ledgers, diagnostics, and output untouched.
It also leaves any earlier valid retained trial and its immutable face ledger
intact; later preview work must not overwrite that trial through scratch aliases.

Incompressible fraction and velocity remain accepted physical state. Low-Mach
extensive inventories and phase-pressure histories belong to private candidate
or immutable accepted snapshots; its temperature, fractions, densities and
population radius/volume are derived views. Retained phase volumes/pressures
during representation relaxation remain physical history, not recomputable
workspace. Native publication assigns every owned value before ghost
synchronization. Mutable solver workspaces never own or alias retained
snapshots, histories or immutable trial flux ledgers.

Topology changes rebuild graphs/maps/importers, gauges, reconstruction/curvature,
apertures, geometry caches, pressure operators/preconditioners, interface and
contact/transfer maps, and output topology. Ownership changes invalidate local
map-dependent state; geometry-only changes retain their existing numeric-cache
rules. Old immutable snapshots remain valid; stale plans/trials reject access
to a replacement snapshot. No rejected regrid produces a physical step/history.

Geometry metadata reuse requires the exact retained hierarchy identity, native
maps/layout, and topology/partition/geometry observations, with a cache hit on
every rank before skipping collective construction. Fraction-dependent geometry
stays call-local; exchange vectors are instance owned. Preserve sparse queries
and compute only consumed neighborhoods. New providers declare actual physical
stencil support; Cartesian patch radii are not universal halo rules. Refinement
guards cover those stencils plus motion across qualified seams/periodic joins.

Reuse a transport numerical operator only after exact collective equality of
final capacities, signed carrier fluxes and equilibration policy. A shared graph
is insufficient. RHS/source/positivity checks remain per quantity, and failed
refresh cannot leave a valid cache key. Projection may reuse graph/storage but
refreshes mobility and susceptibility and invalidates its preconditioner on
assembly. No workspace owns canonical inventories or phase-pressure history.

[`RadiolyticVoFCoupler`](../../src/solvers/RadiolyticVoFCoupler.hh) remains the
external shared owner of the solver and registry/kinetic ledgers. Geometry and
FVM layers acquire no reverse solver/controller ownership. Stage 8 replaces
box-based apertures, film integrals, birth support, donor selection, and bubble
resolution measures with actual polyhedra. Interface patches retain unique
ownership and sparse remote sampling/deposition. Regrid preserves birth/event
IDs and accepted ledgers but invalidates reservations, topology-dependent
component labels/origins, and cached geometry for fresh inspection.

Resolve required population names against the current low-Mach schema
collectively at each source-preparation boundary, including after same-mesh
reinitialization or remap. Publish the complete binding only on success;
reordered schemas retain semantic destinations, missing names reject before
indexing, and reinitialization invalidates prior trials independently of mesh
identity. Do not let a hierarchy cache retain stale population indices.

Preserve closure-specific support on native faces. `ConstantSlip` is a liquid
carrier closure, so empty-population liquid retains its specified drift and
pure resolved gas has zero drift. Radius-dependent closures evaluate only
paired positive number/mole moments; empty radius and volume remain zero.
At a populated/empty front use the populated side's evaluated drift; when both
sides are populated use the qualified distance interpolation. Compute post-source
moments as accepted moments plus signed source increments, then require that
result to be finite, nonnegative, and either both zero or both positive before
deriving support. This includes birth and complete depletion. Transport and
source-drift timestep selection use the same face rule and canonical geometry.
Source-created populations receive an evaluated closure during source
preparation; populations received by transport get their own local closure on
subsequent preparation.

Active checkpoints and pending Phase 22 Stage 4 representation relaxation continue to
block topology change, as in [`AdaptiveVoFSolver::Impl::regrid`](../../src/solvers/AdaptiveVoFSolver.cc).
[`LowMachState`](../../src/solvers/VoFLowMach.hh) carries liquid/resolved pressures
and dispersed pressure/volume history; the
[representation packets](../../src/solvers/AdaptiveVoFRepresentation.inc) retain
phase `V`, `pV`, energy, and momentum. Ordinary remap/EOS reconstruction must not
erase those histories or substitute pressure from new curvature. Physical
relaxation remains a physical step, with no synthetic heat or source event.

Exact same-cell repartition during pending relaxation is a later independent
gate: migrate every state/history field and pending flag without interpolation.
Mesh-inclusive checkpoints are another later gate because current checkpoints
save published fields, not the complete context. They must retain hierarchy,
partition/maps, context, controller registry/ledgers, and output/history cursors;
restore must invalidate prepared trials with a fresh monotone state revision.

## Distributed execution and memory bounds

The current [`MeshPartitioner::make_plan`](../../src/parallel/CompositePartition.tcc)
distributed-source branch gathers shards, calls `merge_partitions` at a root,
and plans that serial union. It may serve as a bounded bootstrap for earlier
geometry stages; it does not satisfy scalable AMR migration. Stage 6 is required
before describing the new adaptation/repartition path as distributed at scale.

Exchange marks and shared face/edge decisions through canonical authorities;
perform closure, owner lookup, and migration between ranks that need the data.
Partition weights include PLIC/reconstruction, interface, and population work;
the objective also includes interface communication and migration cost.
Report weights, measured imbalance, and ownership changes, beyond bare cell counts.
Transfer ancestry, sparse explicit geometry, carrier inventories, and histories
together. Rebuild halos from canonical adjacency and actual reconstruction
support. A partition cut never becomes a physical boundary. Empty ranks join
every collective; duplicate/missing owners, shard disagreement, mixed plans,
or stale observations reject collectively before publication.

Per-rank storage should scale with owned/ghost leaves, local ancestry branches,
resident explicit faces/vertices, needed stencils, and overlap records, plus
documented compact root/templates/interface metadata. Whole products retain
O(base + layers) geometry; local products add unique templates and run/seam
descriptors. Permit no global solution replication, whole-mesh flattening,
expanded global explicit topology, or retained global cell-to-rank arrays.
Account separately for maps, transient migration buffers, and shared objects;
measure memory on empty and unevenly loaded ranks as well as mean storage.
Separate metadata payload from process peak RSS, using the platform-normalized
probe and declared units. Performance needs a numerically matched baseline with
actual ownership, solver work and configuration recorded; Debug timings or cache
reuse alone do not establish production scaling.

## Certified nonconvex extension

Stage 9 first handles star-shaped cells using a certified interior kernel point
and canonical constrained triangulations of their boundary faces. Certify the
resulting positive, disjoint tetrahedral/convex pieces and retain the original
shell for coarsening. A centroid, vertex average, or signed integration fan is
not proof of a valid kernel or decomposition.

For non-star-shaped cells, add boundary-conforming constrained tetrahedral or
convex decomposition, allowing documented interior vertices and canonical
boundary subdivisions. Reject self-intersecting shells first. Verify union,
disjointness, shell equality, and volume/first moments before using the pieces.
[CGAL's convex-decomposition manual](https://doc.cgal.org/6.1.1/Convex_decomposition_3/index.html)
is a reference for decomposing bounded polyhedra, not a selected dependency or
a substitute for this boundary-preservation certificate. Unsupported cells
continue to reject until this entire path is qualified.

After either certified decomposition, clip every convex piece by the same
parent AABB midpoint planes. Within each octant, assemble connected volume
components using positive-area face adjacency, canceling paired canonical
internal-face tiles and validating each resulting boundary. Component counts
are variable; keys retain `(parent, octant, component provenance)`, with
deterministic provenance from sorted constituent piece keys. Retain certified
pieces for subsequent clipping and the saved parent shell for coarsening.
Unchanged decomposition faces are internal integration partitions, never
artificial material or flow boundaries. Reject nonmanifold or unresolved
assemblies. Multiple PLIC patches remain distinct for curvature/film transfer;
one area centroid cannot replace disconnected interfaces. Preserve the physical
shell without boundary projection or domain renormalization.

## Implementation stages and focused acceptance

Stages 1–5 are verified within their stated boundaries; Stages 6–9 remain
proposed. Existing tests are starting points for later extensions, not evidence
that they work. Run focused Debug serial tests,
then affected MPI cases at 2 and 4 ranks where they establish the intended seam
and ownership conditions. Choose additional ranks or deliberately sparse
partitions when needed for a specific ownership regime, and assert observed
empty-owner counts; rank labels or root-block counts do not prove emptiness.
The [durable MPI protocol](../verification/vof_solver_contracts.md#focused-debug-workflow)
includes fixtures with more ranks than global cells. Broader performance
matrices require a specific scaling question.

| Stage | Deliverable | Required focused gates |
| --- | --- | --- |
| 1. Common contracts / Cartesian adapter | Immutable hierarchy/provider seam and distinct stamps | Preserve corrected Cartesian numerics; separate schema/state/geometry identity; stable-key and stale-plan checks; independent partition generation; cache/snapshot isolation; no implicit connectivity materialization |
| 2. Geometry kernels / ancestry overlaps | Certified clips, sections, moments, and nested overlap records | Analytic cube/tetra/prism/skew cuts; tangency/complements; represented coverage at scale/translation extremes and rejection of real gaps; tiny positive remnants; bulk and tilted-PLIC conservation; invalid geometry rejection |
| 3. Whole-product XY/Z AMR | Common Z, whole-column triangles, then convex polygon templates | Split-order key identity; complete factor siblings; exact refine/coarsen recovery; patch inheritance; convexity/quality gates; O(base + layers) storage; native composite MPI |
| 4. Convex polyhedral AMR | AABB-plane cuts, canonical internal overlays, saved parents | Variable child counts; face/edge propagation; orientation and manifold closure; union/first moments; unrelated neighbor cuts; transactional sliver rejection; exact parent recovery |
| 5. Mixed-region / layer-local closure | Product runs, independent ancestry, nested joins, then validated multi-provider tiles | Crossed-trace closure/rejection; one canonical subface flux; periodic matching; contact isolation; physical gradation; no duplicated/gapped tiles; compact repeated seams |
| 6. Distributed migration | Distributed closure/planning and owner-to-owner topology/state migration | Serial/partition-equivalent keys and budgets; split siblings; asserted empty ownership; unique ownership; stale/rank-divergent failures without deadlock; bounded per-rank memory; no root union |
| 7. General VoF / flow | Polyhedral reconstruction, unsplit phase budgets, compatible implicit momentum and physical-residual solves | Oblique planes and skew/coarse-fine/rank transport; microscopic phase/complement budgets; constant velocity and passive momentum bounds; microscopic/subnormal carrier oracle and original residual; correction-budget rejection/reuse; hydrostatics and finite-duration capillarity with independent mixed fractions; retain open high-contrast curved-seam gate; viscous modes and regrid momentum |
| 8. Radiolytic / representation integration | Nonzero volume sources, closure/schema support, actual film/birth geometry, complete coupled transactions | Phase mass/species/energy/physical momentum and source-divergence closure; tiny retained carriers; empty/depleted front speed and timestep parity; same-mesh schema reorder/removal; remote deposition and donor competition; immutable retained trials after failure; pending-history rejection and checkpoint replay |
| 9. Certified nonconvex extension | Kernel-certified star cells, then constrained general decomposition | Invalid/self-intersecting shell rejection; disjoint union and shell/moment equality; canonical boundary splits; disconnected phase patches; saved-parent recovery; repeat geometry, VoF, and coupling gates on each admitted class |

Use [`testCartesianAMR`](../../src/geometry/unitTests/testCartesianAMR.cc),
[`testMultiRegionMesh`](../../src/geometry/unitTests/testMultiRegionMesh.cc),
[`testCompositePartition`](../../src/parallel/unitTests/testCompositePartition.cc),
and [`testCompositePartitionOperators`](../../src/FVM/unitTests/testCompositePartitionOperators.cc)
for native topology and ownership contracts. Numerical references include
[`testAdaptiveVoF`](../../src/FVM/unitTests/testAdaptiveVoF.cc),
[`testAdaptiveVoFThreeDimensional`](../../src/FVM/unitTests/testAdaptiveVoFThreeDimensional.cc),
[`testConservativeVoFTransport`](../../src/solvers/unitTests/testConservativeVoFTransport.cc),
[`testAdaptiveVoFSolver`](../../src/solvers/unitTests/testAdaptiveVoFSolver.cc),
[`testAdaptiveVoFLowMach`](../../src/solvers/unitTests/testAdaptiveVoFLowMach.cc),
[`testRadiolyticVoFCoupler`](../../src/solvers/unitTests/testRadiolyticVoFCoupler.cc),
and [`testBubbleRepresentation`](../../src/solvers/unitTests/testBubbleRepresentation.cc).
Retain the Phase 22 Stage 4 positive-surface-tension pressure-history oracle
when extending representation geometry. Keep dated results in validation records;
support claims change only after the corresponding stage gates pass.
