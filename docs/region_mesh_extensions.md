# Region mesh extensions

This implementation continues the compact region work on the current branch.
The existing coupled-backend tests and their uncommitted documentation updates
remain part of the checkout.

Implementation and verification order:

1. MPI ownership from arithmetic global cell ranges, rank-local maps and halos;
   retain compact global descriptors, reject explicit global unstructured
   regions, and output only owned cells. Check two/four ranks, split regions,
   multiple regions per rank, rank-divergent input rejection, and native regressions.
2. Extend the existing planar ALE transaction to a common axial affine motion
   of composite geometry. Require GCL, conservative transport, cache invalidation,
   exclusive motion ownership and rollback, retaining static child observations.
3. Add explicitly declared coarse/fine planar interface subdivisions with exact
   coverage and overlap validation, one canonical fine face per subface, and
   signed conservative face-flux restriction/prolongation.
4. Compose cylindrical regions, physical angular closures and translated
   structured periodic patches. Check geometry in the correct periodic image;
   vector/tensor component rotations remain unsupported.
5. Add independent straight-extruded geometry with shared base topology, analytic
   metrics and no layer-expanded connectivity. Exercise fields, diffusion and
   coupled/segregated flow through the existing solver stack.

Each support restriction is relaxed only with focused runtime coverage. The
implemented combinations and remaining limits are recorded here after testing.

## Implemented support

- MPI uses communicator-sized arithmetic cell ranges, local owned/ghost maps,
  canonical face ownership and adjacency-based halos. Two/four-rank checks cover
  split regions, several regions per rank, empty owned-cell ranks, rank-divergent
  descriptors, and rejection of stale children before map setup. Global explicit
  unstructured constituents remain rejected; implicit Cartesian/cylindrical and
  native/independent straight extrusions keep compact replicated descriptions.
- The existing planar ALE controller applies a common affine Z map to composite
  volumes, points and area vectors. Child providers remain frozen. Trials,
  leases, epochs, scalar old/new-volume transport, GCL, quality rejection,
  rollback and Boussinesq PISO/coupled heating paths run in serial/MPI.
- Nonconforming seams use declared planar convex coarse/fine tilings. They
  validate containment, native metric agreement, overlap and total coverage.
  Coarse cells traverse canonical fine faces without CSR materialization.
  Signed flux prolongation normalizes by total fine area; restriction sums
  integrated fluxes with signs, preserving supplied flux even within geometry
  matching tolerance.
- Cylindrical composition retains native analytic metrics and periodic indexing.
  Open angular sectors can close through explicit identity-periodic patches.
  Explicit Cartesian translations use incident-cell images in reconstruction,
  diffusion, mesh quality and Rhie--Chow geometry. The inherited bulk distance
  helper also dispatches through the derived periodic distance contract.
- Independent extrusion providers share one immutable base topology while
  retaining separate XY/Z geometry and base metric caches. Single/multi-cell
  templates match native incidence and analytic geometry, and execute existing
  scalar and coupled solvers in serial/MPI.

The expanded tests run all coupled operator/workspace choices with backend
switching and four timesteps. Existing scalar sparse operators and numerical
preconditioners are reused. No new physical model or solver stack was added.
The original pending backend-qualification edits were retained and extended.

## Deliberate limits

MPI currently replicates compact axis/base/interface descriptions, and uses
contiguous canonical-cell ranges rather than a region-aware graph partitioner.
It does not replicate full global unstructured interiors as a fallback.
Explicit distributed region packets remain future work.

Composite motion is common affine Z deformation of implicit Cartesian,
cylindrical and straight-extruded constituents about the global lower plane.
Explicit unstructured constituents remain static because warped-face metric
updates require a different geometry contract. Independent extrusion bases are
strictly convex. Nonaxial/buffered/nonaffine composite motion and changing an axial periodic
length are rejected. Static child replacement requires rebuilding the composite;
controlled MPI motion is collective.

Refinement is nested whole-face planar convex tiling. General mortar/AMI
intersection, nonconvex or curved-face subdivision, overset grids, remeshing and
rotated periodic vector/tensor transformations remain unsupported. Composite
cylindrical angular cells must be narrower than pi for nondegenerate linear HEX
output. Composite extrusion output supports triangular/quadrilateral bases.
These limits remain explicit rather than silently selecting a flattened path.

## Validation commands

The affected GCC Debug build used:

```sh
cmake --build --preset GCC-Debug --target \
  testMultiRegionMesh testMultiRegionMeshMultiRank testMultiRegionOperators \
  testMultiRegionTransportMultiRank testMultiRegionALE testCoupledSolverBackends \
  testBoussinesqPlanarALE testMultiRegionFlow testMeshHandle testPlanarALEMeshMotion \
  testFields testFieldStoredOperators testFvmOperators testMeshReorderingFactory \
  testSemiStructuredXY_Z testOrthogonalCylindrial3D testFluidSolver \
  testCellFluxBalanceCache testStoredPressureFaceFluxCache region_diffusion \
  region_mesh_scaling -j 4
```

Serial/affected native regression:

```sh
ctest --preset GCC-Debug --parallel 4 \
  -R '^(RegionProvidersTest|MultiRegionMeshTest|MultiRegionOperatorsTest|MultiRegionALETest|MultiRegionFlowTest|MeshHandleTest|MeshReorderingFactoryTest|PlanarALEMeshMotionTest|FieldStoredOperatorsTest|FvmOperatorsTest|FieldStoredTest|CellFieldTest|FaceFieldTest|BoundaryFaceFieldTest|VectorCellFieldTest|VectorFaceFieldTest|TensorCellFieldTest|SemiStructuredXY_ZTest|OrthogonalCylindrial3DTest|FluidSolverTest|CellFluxBalanceCacheTest|StoredPressureFaceFluxCacheTest|BoussinesqPlanarALETest|BoussinesqPlanarALESupportMatrixTest)\.|CoupledSolverBackendsTest\.MultiRegion|^(simplefluid_elf_export_boundary|region_diffusion_smoke)$'
```

MPI regression, with host networking for PMIx:

```sh
ctest --preset GCC-Debug \
  -R '^(MultiRegionMesh|MultiRegionTransport|MultiRegionALE|MultiRegionFlow|MultiRegionSolverALE|region_diffusion)_(2|4)procs$|^(PartitionedMeshHandleCommunicator|MeshReorderingFactory|PlanarALEMeshMotion|NonOrthogonalPartitionFaces|FieldStoredTransportValidation|CellFluxBalanceCache|StoredPressureFaceFluxCache)_2procs$'
```

The MPI diffusion example accepts the same filename argument as the serial
example and writes rank pieces plus a PVTU index:

```sh
mpiexec -n 4 build/gcc/bin/Debug/region_diffusion /tmp/region_diffusion.vtu
```

Output inspection verified two/four pieces, 32 unique owned-cell GIDs, correct
region labels, and maximum scalar error `6.661e-16` in both MPI output sets.
The example reports global error/net flux and clearly labels rank-zero storage.
The storage report excludes fields, sparse operators, preconditioners and
allocator metadata; no total-RSS or speedup claim is made. Initial benchmark
numbers in `region_mesh_implementation.md` remain historical snapshots.

## Observed delivery results (2026-09-10)

The affected-target GCC Debug build completed successfully. The serial command
selected 271 registrations: 262 passed and nine rank-conditioned cases skipped,
with no failures (17.95 seconds). This includes the ELF export audit, native
single-region ALE and FVM regressions, all new region/provider cases, both
coupled representations and solver-integrated composite heating.

The serial 32-cell example observed maximum error `4.4408920985e-16`, net outward
diffusive flux `9.7699626167e-15`, and zero compatibility-connectivity storage.
The current small Debug scaling observation at 1,024 cells was:

| Path | Measured mesh subtotal | Compatibility connectivity | Index vectors | Index-tree estimate |
| --- | ---: | ---: | ---: | ---: |
| Native compact | 6,176 bytes | 0 | 0 | 0 |
| Compact composite | 8,600 bytes | 0 | 128 bytes | 0 |
| Explicit compatibility path | 177,984 bytes | 32,776 bytes | 139,032 bytes | 695,160 bytes |

The separate map ID-payload estimate was 75,776 bytes for each serial path;
this is not Tpetra allocation or total RSS measurement. Regular correspondence
remained 232 bytes independent of the tested face counts. Timing observations
were single Debug runs, with no performance threshold or speedup claim. Fields,
sparse operators and preconditioners are excluded from mesh subtotals.

All 19 delivery MPI registrations passed with host networking in 53.84 seconds,
including two/four-rank geometry, reconstruction/diffusion, conservative
nonconforming transport, empty-rank diffusion, coupled flow, solver-integrated
ALE, VTU/PVTU example output and directly affected native MPI regressions.
Rank-specific subcases inside MPI executables retain their stated guards (for
example, the empty-rank case requires four ranks).

`git diff --check` and whitespace checks of the new source files passed. The
current branch remains `feature/mesh`; no commit or push was made. The complete
unrelated test suite, other compiler/configuration matrices, and large/RSS
benchmarks were not rerun for these extensions.
