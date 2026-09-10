# Region execution and numerical qualification

This milestone retains `IsoRegion`, canonical conforming/coarse-fine faces,
`MeshHandle`, and the existing FVM/solver paths. It adds an execution boundary,
compact local interface indexing, provider-aware construction validation, and
accuracy/scaling qualification. The contiguous canonical-cell MPI partitioner
and current supported geometry/interface combinations remain authoritative.

## Execution contract

```cpp
const auto execution = mesh.acquire_execution_view();
assemble_or_apply(mesh);  // Existing MeshHandle queries and kernels.
```

Acquisition checks every constituent once and holds read leases on the composite,
mutable native geometry, native-provider objects and shared Cartesian topology.
The leases reject assignment, moves from leased objects, and controlled geometry
motion before coordinates or topology change. Copy construction creates an
independent unleased value. Existing stale-child rejection still runs when an
operation acquires a view, or when a checked public query runs outside a view.

While the view lives, queries retain bounds checks and omit whole-composite
validation on that thread. Nested kernels reuse the enclosing view. The view
borrows its mesh, cannot be copied or moved, and must be destroyed in scope order
on its acquiring thread. Concurrent readers are supported; writers still require
external synchronization, as in the existing motion contract. View acquisition is
local. Composite construction and MeshHandle preflight retain collective failure
handling before map creation. A read lease held on any rank rejects a collective
ALE trial or rollback on all ranks.

MeshHandle setup, gradient geometry construction, transport assembly, face-flux
kernels, and composite VTU topology generation enter this boundary. Native and
legacy mesh paths continue through their existing implementations. No per-cell or
per-face native lookup table is introduced.

`execution.visit_regions(visitor)` also dispatches once per region, passing its
ordinal, canonical cell offset, and original provider pair. This gives kernels
that already know their region access to arithmetic native indexing. These are
**raw constituent coordinates**: canonical mesh queries apply composite affine-Z
motion. Region visitation is not a substitute for that transform in moving-mesh
kernels.

The next production step adds `execution.visit_region_geometry(visitor)`. Its
typed region view applies the composite affine transform and resolves incident
periodic images. `visit_cell_faces()` walks native incidence once, expanding
coarse faces in the existing canonical fine-face order. Each callback receives
an operation-local `ResolvedFace` value with canonical incidence and the metrics
needed by numerical kernels. `execution.resolve_face(id)` provides the same
snapshot when only a canonical face is available, with one native decode.
Snapshots do not follow subsequent motion or extend the execution lease.

Scalar `FVM::diffusion_system()` now uses this traversal for composite handles
whose owned cells retain canonical order. The existing field maps, ghost IDs,
two-point coefficient formulas, CRS insertion order and boundary callback order
are retained. Reordered and other native/legacy handles use the generic path.
The generic scalar kernel remains available as an internal reference for parity
tests and same-binary performance comparisons. No global resolved-face or
connectivity table is stored. See [region_production.md](region_production.md)
for the production-path comparisons and durable reports.

## Compact indexing and construction

Each region has seven interface-directory spans: six structured boundary
orientations and one irregular bucket. There are two side descriptors per
interface. Boundary queries inspect the addressed region and boundary; irregular
correspondences retain their existing sorted tables. Canonical face recovery
translates directly for regions with no removed faces; other regions perform
rank/select using only their incident removal descriptors. Boundary-name lookup
also uses a per-region span. Remaining region-offset binary searches cost
`O(log R)`; this milestone does not claim constant-cost canonical ordinal recovery.

Cartesian validation examines coordinate arrays, native midpoint arithmetic and
the actual minimum/maximum volume cells. Straight-extrusion validation examines
base cells and axial coordinates. These paths avoid full tensor-product cell
sweeps while retaining nonfinite, overflow and underflow rejection. Explicit and
cylindrical cell validation remains entity-wise. Structured boundary counts and
remaining physical boundary coverage use arithmetic descriptor counts; interface
geometry continues to be validated face by face. Explicit/extruded boundary
fallback scans remain visible construction costs.

## Numerical studies

`testRegionQualification` uses the existing operators and Belos solves:

- Smooth nonlinear manufactured reaction-diffusion on ratios 2 and 3, interfaces
  at x=0.25 and x=0.5, three refinements, and different interface
  nonorthogonality. It reports volume-weighted global and interface-band errors,
  observed orders, converged correction count and true residual, with conforming
  references. MPI partition-face corrections are iterated to convergence before
  comparing errors.
- Sinusoidal periodic advection for two complete domain crossings, three
  refinements, inventory, phase and amplitude measurements. An independent
  backward-Euler/upwind Fourier symbol verifies the expected first-order damping.
  MPI cases assert that the periodic connection crosses ranks.
- A fixed 16x4x2 physical grid represented by 1, 2 and 8 regions, comparing
  operator action, converged scalar fields, production Rhie-Chow face fluxes and
  flux-divergence continuity operators in serial and MPI. This last measurement
  is decomposition invariance of the continuity operator; existing coupled-flow
  regressions cover convergence of the pressure-velocity solve.

`testRegionExecution` covers leases, aliases, moves, nesting, exception unwinding,
bounds, motion and rank-asymmetric lease rejection. `testRegionInterfaceIndex`
covers local directories, orientations, split patches, periodic/coarse-fine
correspondence, descriptor-sized storage and volume-range validation.

The independent N/R/P benchmark and peak-RSS methodology are documented in
[region_scaling.md](region_scaling.md). Timings are diagnostic, without an
arbitrary speedup threshold. Analytical qualification is distinct from physical
validation against an experiment.

## Executed checks

The 2026-09-10 working tree is based on `a1a5a7a`; these results apply to the
uncommitted execution/qualification changes, rather than the original commit.
The focused LLVM 22.1.8 Debug run passed 106 serial tests, with three MPI-only
skips. This includes native Cartesian/cylindrical/extruded/unstructured geometry,
region providers and operators, motion, the new execution/directory tests, and
12 selected composite coupled-backend cases. All eight selected two/four-rank
MPI registrations passed, including collective input/stale-child rejection,
ownership, halos, transport, affine ALE and rank-asymmetric execution leases.

The focused GNU 16.2.1 Release CTest run passed 59 registrations (including seven
MPI registrations), with 11 serial rank-conditioned skips and no failures. Its
scope is the region execution/directory, provider, geometry, operator, halo and
ALE tests; the wider native/coupled checks above were run under LLVM Debug.

The independent numerical runner passed all three studies at one, two and four
ranks in GCC Debug, GCC Release and LLVM Debug, with a largest serial/MPI diagnostic
difference of `1.0e-12`. Numerical values, acceptance bounds and configuration
details are in [region_accuracy.md](region_accuracy.md). OpenMPI 5.0.10 runs used
host networking; sandbox socket failures are not numerical results. The full
unrelated suite and hosted CI were not run for this milestone.

Selected MPI checks can be reproduced after building the named test targets:

```sh
ctest --test-dir build/llvm -C Debug --output-on-failure -j 1 \
  -R '^(MultiRegionMesh_[24]procs|MultiRegionTransport_[24]procs|MultiRegionALE_[24]procs|RegionExecution_2procs|PlanarALEMeshMotion_2procs)$'
```
