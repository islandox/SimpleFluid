# Annular meshes with graded wall layers

`AnnularSemiStructuredMeshFactory` builds a polygonal annular fluid domain
with triangular-prism bulk cells and hexahedral radial-wall cells. The result
is a `Meshes::MultiRegionMesh`, usable through `MeshHandle` in serial or MPI.
It also supports the composite's common affine Z motion. This does not change
the serial-only contract of a direct `SemiStructuredXY_Z` handle.

Include `geometry/AnnularSemiStructuredMeshFactory.hh` and link
`SimpleFluid::Mesh`. Construct the mesh collectively on the default Tpetra
communicator, with identical options on every rank, before creating fields.

## Construction

After initializing Tpetra/Kokkos, a small mesh can be built as follows:

```cpp
#include "geometry/AnnularSemiStructuredMeshFactory.hh"
#include "geometry/MeshHandle.hh"

using namespace SimpleFluid;
AnnularSemiStructuredMeshFactory::Options options;
options.inner_radius = 0.04;
options.outer_radius = 0.09;
options.bottom = 0.20;
options.top = 0.25;
options.xy_spacing = 0.01;
options.z_spacing = 0.02;
options.wall_layers = 3;
options.first_layer_height = 0.002;
options.growth_ratio = 1.5;

AnnularSemiStructuredMeshFactory factory(options);
auto result = factory.build();
auto mesh = std::make_shared<MeshHandle<>>(result.mesh);
// Bind boundary conditions and fields to this handle.
```

The physical boundary batches are `rmin`, `rmax`, `zmin`, and `zmax`.
Regional seams are internal faces with one canonical flux degree of freedom.
`rmin`/`rmax` identify polygonal walls; their local normals follow the planar
faces rather than a curved cylindrical surface.

All lengths use the mesh coordinate unit (metres for SI solver inputs).

| Option | Default | Contract |
| --- | --- | --- |
| `inner_radius`, `outer_radius` | `0`, `0` | Required: finite `0 < inner_radius < outer_radius`. |
| `bottom`, `top` | `0`, `0` | Required: finite `bottom < top`. |
| `xy_spacing`, `z_spacing` | `0.01`, `0.01` | Positive requested bulk spacings; XY spacing controls interior point placement. |
| `inner_angular_cells`, `outer_angular_cells` | `0`, `0` | Independently sized wall rings. Zero selects automatic sizing; an explicit count must be at least eight. |
| `wall_layers` | `8` | Layer count at each selected wall; zero disables all stacks. |
| `first_layer_height` | `0.002` | Positive first boundary-normal width. |
| `growth_ratio` | `1.25` | Successive layer-width ratio, at least one. |
| `refine_inner`, `refine_outer`, `refine_bottom` | `true` | Enable the corresponding graded stack. |
| `refine_top` | `false` | Enable a graded top stack. |
| `coarsen_bottom_corners` | `true` | Replace eligible fine radial/bottom intersections with swept mitered cells. |

Automatic angular counts are even and at least eight. Each wall stack is sized
using its own largest circumference; the inner count need not equal the outer
count. Explicit counts retain the supplied value, including odd counts.
The normal wall widths follow
`first_layer_height * pow(growth_ratio, layer)`, starting at layer zero.
Radial widths are measured between parallel polygon sides using their
apothems. Bulk Z intervals are split evenly to meet `z_spacing` independently
of those stacks. Overlapping stacks, insufficient polygon clearance,
unrepresentable coordinates, and overflowing mesh sizes are rejected.

## Bulk and corner topology

`FrontalDelaunay2D::triangulate_annulus()` connects the two fixed wall-interface
loops with triangular cells. Both input loops are strictly convex and
counter-clockwise, with the inner loop strictly inside the outer one and no
repeated closing vertex. The mesher preserves every supplied boundary segment
without adding or moving boundary points, allowing one-to-one wall attachment.
Requested XY spacing controls the advancing-front interior points; it does not
override the supplied boundary segmentation.

Radial stacks extrude to hex cells. Bottom and optional top stacks retain
triangular prisms beneath/above the triangular bulk, so axial interfaces match
face-for-face. Bottom corner coarsening selects consecutive wall widths
strictly below `0.5 * min(xy_spacing, z_spacing)`. It requires bottom refinement,
at least one refined radial wall, at least two eligible layers, and an actual
cell-count reduction. Otherwise the ordinary wall intersections remain.
Coarsening replaces that local tensor-product overlap with convex mitered R–Z
quadrilaterals swept into hex cells. It preserves native volume and conforming
interfaces. Top corners are not coarsened by this option.

`SweptRZRegion` stores a compact radius-height template and angular edges.
Its native template loops are clockwise, with radius in the node's X component,
height in Y, and zero Z. Triangles and strictly convex quadrilaterals are
supported, radii must be positive, and each angular interval must lie in
`(0, pi)`. Straight segments between angular planes give planar cell faces;
these are not analytic curved cylindrical cells. See
[region provider contracts](region_meshes.md#providers-and-ownership).

## Counts, volume, and motion

`planned_counts()` constructs the two-dimensional layout and returns exact
counts without allocating the three-dimensional extrusion. It still performs
triangulation and validation; it is not a constant-time size estimate.
`build()` returns the mesh, counts, reference coordinates, and area diagnostics.
Use `counts.cells`, `hex_cells`, `prism_cells`, `faces`, `nodes`, and `regions`
for actual totals. `radial_cells`, `coarse_xy_cells`, `radial_apothems`, and
`axial_fractions` describe reference guides/templates, not a uniform global
cell indexing scheme. The compatibility member `angular_cells` is the outer
angular count; use the separate inner/outer members for both rings.

The inner polygon circumscribes the inner circle and the outer polygon is
inscribed in the outer circle. The fluid domain therefore lies inside the
requested analytic annulus. `cross_section_area` is its actual polygonal area;
`analytic_cross_section_area` and `relative_area_deficit` expose the difference.
Native volumes are never normalized to the analytic vessel volume.

Use `PlanarALEMeshMotion` on the mutable composite handle for supported common
affine Z motion. Child coordinates remain immutable. Respect trial/accept/
rollback, geometry epochs, GCL, and mesh-quality checks as described in the
[maintainer guide](../MAINTENANCE.md#geometry-epochs-and-motion-transactions).
Positive volumes alone do not establish solver suitability: custom wall widths
or spacing can exceed the solver's non-orthogonality/quality limits.

For transfer to an analytic cylindrical mesh, use the
[conservative transfer API](mesh_to_mesh_transfer.md). The polygonal and
circular domains differ, so select `AllowPartial` and account for uncovered
inventory. Both straight XY extrusions and swept R–Z corner cells are supported
against a native cylindrical endpoint; weights are not renormalized.

## Focused regression coverage

The checked-in tests cover annular containment and native volume, independent
angular counts, Delaunay boundary preservation, graded widths, corner selection,
canonical interfaces, distributed ownership, affine motion, GCL, and rollback.
The default centimetre R100 geometry has a dedicated mesh-quality regression.
These are geometry and numerical contracts, not physical CFD validation.

To build and run this focused geometry coverage:

```bash
cmake --build --preset GCC-Debug --target \
  testAnnularSemiStructuredMeshFactory testFrontalDelaunay2D \
  testMiteredCornerPatch testSweptRZRegionProviders
ctest --preset GCC-Debug --output-on-failure \
  -R '^(AnnularSemiStructuredMeshFactoryTest\.|FrontalDelaunay2DTest\.|MiteredCornerPatchTest\.|SweptRZRegionProvidersTest\.|AnnularSemiStructuredMeshFactory_2procs$|SweptRZRegionProviders_2procs$)'
```

MPI cases require a configuration that enables two-process tests. The commands
are a reproducible workflow; they do not record a new test run.
