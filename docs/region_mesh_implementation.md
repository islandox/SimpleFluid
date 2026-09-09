# Compact region mesh implementation record

The initial checkout stores Cartesian metrics by coordinate axis and extruded
metrics by base entity and layer. `MeshHandle::initialize_cell_faces()` eagerly
expands both into CSR rows, while `LocalGlobalIndexer` stores redundant serial
identity vectors and trees. FVM consumers require sized/indexable traversal;
the old handle connectivity test alone requires a span. The extruded topology
also reserves a full interior-cell list. Geometry revisions belong to concrete
meshes and are observed by existing FVM caches. The VTU visitors default to
orthogonal geometry for alternatives other than STK, semi-structured and
unstructured. No matrix-free FVM operator was found in tracked source.

Implementation sequence:

1. Replace implicit cell-face expansion with immutable indexed ranges and
   retain explicit compatibility materialization. Preserve halo traversal.
2. Add topology/geometry provider contracts and owning `IsoRegion` adapters.
3. Add validated structured/explicit interfaces and serial canonical composite
   identities, then integrate the backend into `MeshHandle` and output.
4. Exercise existing fields, assembled operators and flow solvers; add a
   diffusion example, structural storage checks and an opt-in scaling benchmark.
5. Run focused Debug tests, broader affected regressions and registered MPI
   checks; document measured storage and scope limits.

Baseline command (existing binaries, before edits):

```sh
ctest --preset GCC-Debug -R '^(MeshHandleTest|OrthoMeshTopoTest|SemiStructMeshTopoTest|UnstructuredMeshTest)\.'
```

Observed: 24 passed, one rank-conditioned test skipped, 2.03 seconds.
Build commands use `cmake --preset GCC-ninja-multi` and
`cmake --build --preset GCC-Debug --target <affected targets>`.

## Validation commands

The implementation uses GCC Debug, C++23, GNU 16.2.1 and the preset's existing
Trilinos/Kokkos installation. The first build command found a missing
`build/gcc/CMakeCache.txt`; `cmake --preset GCC-ninja-multi` restored the build
configuration without dependency changes.

The final affected-target build is:

```sh
cmake --build --preset GCC-Debug --target \
  testMultiRegionMesh testMultiRegionOperators testMultiRegionFlow \
  testMeshHandle testSemiStructMeshTopo testPlanarALEMeshMotion \
  testMeshReorderingFactory testOrthoMeshPartitioner region_diffusion \
  region_mesh_scaling testOrthogonalCartesian3D testOrthogonalCylindrial3D \
  testSemiStructuredXY_Z testUnstructuredMesh testLocalGlobalIndexer \
  testMeshPartitioner testFields testGhostSyncMultiRank \
  testFieldStoredOperators testFvmOperators testFluidSolver \
  testFluidSolverMultiRank testCellFluxBalanceCache \
  testStoredPressureFaceFluxCache testBoussinesqPlanarALE -j 4
```

Broader affected serial regression (includes the export check and example):

```sh
ctest --preset GCC-Debug --parallel 4 \
  -R '^(RegionProvidersTest|MultiRegionMeshTest|MultiRegionOperatorsTest|MultiRegionFlowTest|MeshHandleTest|MeshReorderingFactoryTest|OrthoMeshPartitionerTest|OrthogonalCartesian3DTest|OrthogonalCylindrial3DTest|SemiStructMeshTopoTest|SemiStructuredXY_ZTest|UnstructuredMeshTest|LocalGlobalIndexerTest|BoundaryFaceFieldTest|CellFieldTest|FaceFieldTest|FieldStoredTest|TensorCellFieldTest|VectorCellFieldTest|VectorFaceFieldTest|FvmOperatorsTest|FieldStoredOperatorsTest|CellFluxBalanceCacheTest|StoredPressureFaceFluxCacheTest|FluidSolverTest|PlanarALEMeshMotionTest|BoussinesqPlanarALETest|BoussinesqPlanarALESupportMatrixTest)\.|^(simplefluid_elf_export_boundary|region_diffusion_smoke)$' \
  --output-log /tmp/simplefluid_region_regression.log
```

MPI regression:

```sh
ctest --preset GCC-Debug \
  -R '^(MultiRegionMeshSerialOnly|SemiStructuredMeshHandleSerialOnly|PartitionedMeshHandleCommunicator|MeshReorderingFactory|PlanarALEMeshMotion|MeshPartitioner|GhostSyncMultiRank|FieldStoredTransportValidation|NonOrthogonalPartitionFaces|CellFluxBalanceCache|StoredPressureFaceFluxCache|FluidSolver|FluidSolverGeometryRefresh|BoussinesqPlanarALE)_(2|4)procs$' \
  --output-log /tmp/simplefluid_region_mpi.log
```

The initial five-test MPI subset failed before entering test bodies because
PMIx reported `socket() failed with errno=1` and no available interfaces in the
sandbox. The same command passed outside the sandbox. The broader MPI command
also uses host networking. No MPI launch failure is counted as a passing test.

Two existing raw-replacement ALE assertions now expect revisions 2 and 1 rather
than a reset to zero. Their controller-rejection, geometry-preservation and
rollback assertions remain intact. New scalar fixtures synchronize owned values
into the `FieldStored` overlap before calling the existing mapped operators.

Large benchmarks, the full unrelated test suite, and another compiler/configuration
matrix were outside the initial run. No matrix-free integration was added in
that pre-merge checkout; the later block-composite coupled backend is covered
by the follow-up below.

## Final observed results (2026-09-09)

- The affected-target GCC Debug build completed successfully.
- The broader serial command selected 268 tests: 259 passed and 9
  rank-conditioned tests skipped, with no failures (9.15 seconds).
  This includes the new provider/composite/FVM/flow checks, the runnable
  example, single-region geometry/ALE regressions and the ELF export audit.
- All 16 selected MPI registrations passed with host networking, including
  2- and 4-rank partition/halo checks and 2-rank transport, flow and ALE.
- `git diff --check` and whitespace checks of the new source files passed.
- The current branch is `feature/mesh`. No commit or push was made. The
  pre-existing untracked `docs/benchmarks/` directory was left untouched.

Example and small benchmark commands:

```sh
UCX_LOG_LEVEL=error build/gcc/bin/Debug/region_diffusion /tmp/region_diffusion.vtu
UCX_LOG_LEVEL=error build/gcc/bin/Debug/region_mesh_scaling 8
```

The 32-cell, 132-face example observed maximum solution error
`4.4408920985e-16`, net outward diffusive flux `9.7699626167e-15`, and zero
compatibility-connectivity bytes. Its measured mesh subtotal was 5,232 bytes;
the separate required-map ID-payload estimate was 3,200 bytes. Temporary VTU
topology occupied 4,496 bytes, excluding output field arrays and serialization
buffers. The example writes scalar `phi` and `topology_region` cell data.

The final Debug CSV is [region_mesh_scaling.csv](region_mesh_scaling.csv).
For the 1,024-cell case:

| Path | Measured mesh subtotal (bytes) | Compatibility CSR (bytes) | Index vectors (bytes) | Index-tree estimate (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Native compact | 6,176 | 0 | 0 | 0 |
| Compact two-region | 8,400 | 0 | 96 | 0 |
| Explicit compatibility materialization | 177,984 | 32,776 | 139,032 | 695,160 |

All three paths had the same 75,776-byte map ID-payload estimate and query
checksum 98. The estimate is not a measurement of Tpetra's actual allocations;
contiguous maps may use arithmetic IDs. The measured subtotal includes vector
capacities, provider/native object sizes and shared topology counted once,
but excludes allocator metadata, control blocks, string heap allocations and
unordered-container nodes/buckets. It excludes all field/operator/solver memory.

Regular interface correspondence stayed at 200 bytes from 16 to 1,024 cells.
Construction/query/assembly times for 1,024 cells were respectively
0.00121/0.03755/0.09975 seconds for native compact,
0.07187/0.07125/0.20622 seconds for two-region compact, and
0.01546/0.00818/0.10192 seconds for materialized compatibility. These are single
small Debug observations; no speedup, RSS threshold, expected percentage saving
or convergence-order claim is made.

## Merge follow-up (2026-09-10)

The follow-up started from clean `feature/mesh` at `af3128d`, after the region
implementation in `e339ec3` and the merge from `develop`. That merge added the
block-composite coupled operator and cached/streamed workspace policies, so the
initial audit's absence of that backend no longer describes the current tree.

Eight added parameterized cases in `testCoupledSolverBackends` cover Cartesian
and mixed HEX/prism region meshes using the existing qualification harness.
They test all four solver families and backend/workspace combinations, varying
timesteps, backend switching, both pressure gradients and mixed-mesh
explicit/implicit/hybrid non-orthogonal treatment. Field/flux parity, continuity,
production cache counters and zero mesh-connectivity materialization are checked
after each step. The existing solver implementation required no changes.

Focused commands and observed results:

```sh
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug --target testCoupledSolverBackends testMultiRegionMesh testMultiRegionOperators testMultiRegionFlow region_diffusion -j 4
ctest --preset GCC-Debug -R 'CoupledSolverBackendsTest\.SerialMultiRegion' --parallel 4
ctest --preset GCC-Debug -R '^(RegionProvidersTest|MultiRegionMeshTest|MultiRegionOperatorsTest|MultiRegionFlowTest)\.|^(region_diffusion_smoke|simplefluid_elf_export_boundary)$' --parallel 4
```

The focused build succeeded; the two test commands passed 8/8 in 3.87 seconds
and 27/27 in 3.57 seconds, respectively. Historical validation totals above
remain scoped to the original runs. Composite MPI, composite ALE and independent
extruded geometry providers remain future work.

The complete configured GCC Debug target set was then rebuilt successfully:

```sh
cmake --build --preset GCC-Debug -j 4
ctest --preset GCC-Debug -R '^(MultiRegionMeshSerialOnly|PartitionedMeshHandleCommunicator)_2procs$'
git diff --check
```

Both MPI registrations passed with host networking (2/2, 0.34 seconds).
The serial checks and MPI registrations total 37 passing focused registrations;
the full test suite and other build configurations were not rerun. No commit or
push was made for this follow-up.

## Subsequent region extensions

The later MPI, axial affine ALE, coarse/fine, cylindrical/periodic and independent
extruded-provider work is recorded in [region_mesh_extensions.md](region_mesh_extensions.md).
Its current support matrix supersedes the initial serial/static limitations;
all dated measurements and earlier commands above remain historical records.
