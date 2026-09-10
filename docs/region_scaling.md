# Region execution scaling qualification

The subsequent [production diffusion qualification](region_production.md)
compares production dispatch against the retained ordinal-query reference in
the same binary and repairs the compatibility materialization allocation
pattern. Measurements below remain the September 10 historical snapshot.

`region_scaling_benchmark` retains the native geometry, `MultiRegionMesh`,
`MeshHandle`, FVM diffusion assembly, and Belos solver paths. It is opt-in and has
no timing or memory threshold in CTest. Each process runs exactly one baseline,
so a previous representation cannot contaminate its peak resident memory.

The unit cube has `n^3` identical Cartesian cells in all cases. Artificial regions
are x-slabs, and `R` must divide `n`. The runner changes cell count `N`, region
count `R`, and MPI rank count `P` in separate sweeps, holding the other two fixed.
This is strong scaling for the P sweep, not weak scaling.

| Baseline | Storage and execution |
| --- | --- |
| `native` | One compact `OrthogonalCartesian3D` through `MeshHandle`. |
| `composite` | R compact Cartesian providers and R-1 conforming interfaces. |
| `materialized` | The same composite, followed by forced cell-face and compatibility indexer materialization. |
| `flattened` | Explicit global node coordinates, hexahedral cell connectivity, deduplicated faces, and cached geometry in `UnstructuredMesh`. |

Native and flattened baselines always represent one region; their metadata
distinguishes requested R from representation R. Flattened construction includes
building input nodes/cells and native unstructured face construction. It is a
different baseline from forcing compatibility connectivity on a compact provider.
MPI flattened cases are recorded as unsupported: the executable exits 77 if
requested, and the runner records the reason without substituting replicated
explicit geometry.

Construction includes descriptor validation, interfaces, partitioning, halos,
maps and boundary batches. Execution-view acquisition is timed separately from
owned-cell geometry/connectivity traversal. Assembly uses the existing diffusion
system with Dirichlet data for `1+x+2y-0.5z`; matrix application has one warm-up
before its timed repeats. The CG/Jacobi solve starts from zero and includes
preconditioner setup. A fresh true residual must be at most `1e-9`, and the maximum
solution error at most `1e-7`; these gates protect timing comparisons from failed
solves. Output measures native geometry VTU generation and writing, including
temporary output connectivity, without `fsync` or a solved field export.

Composite modes also measure `native_region_traversal_average`: `visit_regions`
intersects each provider's canonical cell interval with the owned interval, then
queries provider geometry and native face incidence directly. Its checksum must
match canonical traversal for these conforming Cartesian fixtures. It bypasses
canonical face recovery and illustrates the cost still paid by canonical handle
queries; it is not a complete alternative FVM assembly kernel. The current
contiguous composite partition is an explicit precondition of this diagnostic.

Every checkpoint reports maximum-rank elapsed time, maximum ghost/owned ratio,
owned/ghost counts, maximum-rank peak RSS and structural storage components.
Summed rank high-water marks are also retained, but they are not a simultaneous
application peak. RSS includes runtime libraries, fields, matrices, solver
workspace, allocator overhead and temporary allocations from all earlier phases.
Structural bytes exclude allocator metadata and opaque Tpetra allocations; map ID
payload and index tree values remain separately labelled estimates. Compact
baselines must still have zero compatibility connectivity after output, and must
not acquire a compatibility indexer that was absent after construction. Required
MPI local ownership/overlap indexing is present from construction.

The following commands use the repository's shared environment. Configure again
after adding the target. Set `SIMPLEFLUID_COMPILER=LLVM` to use the second compiler.

```sh
export SIMPLEFLUID_BUILD_CONFIG=Release
. verification/environments.sh
export_build_env "$PWD"
cmake --preset "$SIMPLEFLUID_CONFIGURE_PRESET"
simplefluid_build_target region_scaling_benchmark 4
python3 verification/run_region_scaling.py \
  --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-scaling-release \
  --cells 8,16,32 --regions 1,2,4,8,16 --ranks 1,2,4 \
  --fixed-cells 16 --fixed-regions 2 --fixed-ranks 1 \
  --samples 3 --repeats 5 --mpi-args '--bind-to core'
```

Use a new output directory for each run. The runner keeps every exact command,
source SHA and dirty status, a source diff and new source files, executable and
project shared-library SHA-256 digests, environment settings,
raw JSONL/stderr, output files, and CSV rows. `report.md` contains median timings
and memory; `summary.csv` also includes sample minima/maxima. New untracked code
under `src`, `cmake` and `verification` is copied into `source-new`; the executed
binary and library digests identify the executed project artifacts. CPU binding,
process count, compiler and MPI transport
should be kept consistent when comparing runs. MPI launcher flags are explicit,
and runtime/sandbox failures must not be interpreted as mesh failures.

Small runs reveal costs and storage trends, not production scalability. In
particular, this milestone keeps the contiguous canonical-cell partitioner and
whole-descriptor construction; benchmark setup costs can expose remaining global
work. Region-aware distributed layouts and distributed explicit packets require
separate qualification. Numerical convergence and decomposition-invariance
studies are independent of this linear benchmark's solve gate.

## Recorded Release qualification, 2026-09-10

The measured executable was built from reviewed head
`a1a5a7a77ab0ca96cf383d60ba1b64746e2eaa25` plus the region execution milestone
changes. All 42 fresh benchmark processes completed with the same executable
SHA-256 `0566a82b6bd1aa0bec34e3afaf31e2ef7d53affbc00645099b39faff70674550`.
The toolchain was GNU 16.2.1 Release, Open MPI 5.0.10, on an Intel i5-12500 with
12 logical CPUs under WSL2/Linux 6.18.33.2. MPI measurements used host defaults
and `--bind-to core`; serial processes inherited the available CPU affinity.
Builds and correctness tests had finished before these measurements.

The main matrix used one fresh sample at `n=16,32`, `R=1,2,8,32`, and `P=1,2,4`,
with fixed `n=32,R=2,P=1` for the independent sweeps. Four additional serial
samples covered `n=64,R=2`, one per baseline. A targeted repeat then used three
fresh composite samples for each `R=1,2,8,32` at fixed `n=32,P=1`. Each sample
averaged five traversal/application repetitions. All true relative residuals
were at most `9.97e-11`, and all analytical maximum solution errors were at most
`2.74e-9`. Compact storage checks and native/canonical traversal checksum checks
passed. The two flattened MPI matrix cases were explicitly unsupported; a
separate two-rank invocation verified the executable's rejection status 77.

[The recorded CSV](region_scaling_release.csv) retains every phase's median,
minimum and maximum, including storage and rank maxima. The `main` and `large`
series are single samples, so their minima and maxima are identical. The
following table reports the repeated region-count sweep; brackets contain the
three-sample minimum and maximum, in milliseconds.

| R | Provider traversal | Canonical handle traversal | Diffusion assembly |
| ---: | ---: | ---: | ---: |
| 1 | 2.268 [2.156, 2.514] | 54.492 [53.601, 55.039] | 113.835 [106.052, 125.595] |
| 2 | 2.172 [2.137, 2.544] | 81.312 [78.220, 85.520] | 381.681 [379.195, 384.943] |
| 8 | 2.271 [2.242, 2.307] | 105.930 [103.118, 116.638] | 568.208 [561.747, 592.513] |
| 32 | 2.220 [2.129, 2.295] | 115.582 [112.296, 115.584] | 594.177 [588.348, 607.677] |

Provider traversal stays near 2.2 ms across this R sweep. Median execution-view
acquisition grows from 0.290 to 1.513 microseconds. Canonical handle traversal
and production assembly still carry substantial overhead: at this fixed mesh,
the native baseline traversed in roughly 6.1 ms and assembled in roughly 31 ms.
The directory and execution scope remove whole-composite validation from the
bulk loop, but canonical face recovery remains an expensive path. This evidence
supports efficient provider traversal and identifies remaining operator work;
it does not establish inexpensive production canonical execution.

At 262,144 cells (`n=64,R=2,P=1`), the single-sample memory and construction
comparison was:

| Baseline | Structural mesh KiB | Peak RSS MiB | Construction seconds | Assembly seconds |
| --- | ---: | ---: | ---: | ---: |
| Native compact | 106.422 | 132.020 | 0.00135 | 0.333 |
| Compact composite | 109.625 | 150.449 | 0.14749 | 4.264 |
| Forced compatibility | 39,701.180 | 433.480 | 218.38394 | 3.557 |
| Flattened unstructured | 197,573.945 | 402.852 | 1.37663 | 0.294 |

The composite has a much smaller structural mesh payload than the explicit
reference, while its measured benchmark process peak is about 150 MiB and the native
compact peak about 132 MiB. Fields, matrices, solver state and output account for
memory beyond the structural report. Forced compatibility construction includes
the existing per-cell `reserve(current_size + face_count)` behavior and full
indexer materialization; its extreme setup time must be kept separate from
compact construction and must not support a general application speedup claim.

The fixed `N=32,768,R=2` composite rank sweep was also diagnostic:

| P | Construction ms | Canonical traversal ms | Assembly ms | Solve ms | Peak RSS max-rank MiB | Ghost/owned max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.701 | 82.947 | 379.409 | 24.202 | 65.023 | 0.0000 |
| 2 | 127.611 | 59.173 | 340.228 | 15.192 | 79.934 | 0.0625 |
| 4 | 90.997 | 36.226 | 212.953 | 14.094 | 70.109 | 0.1250 |

Construction does not improve with rank count at this size. The map/halo setup
and replicated descriptor work remain visible; this is not evidence of scalable
distributed construction. Solver work and operator application are similar
across representations, with the main CPU difference in mesh traversal and
assembly. Larger production workloads and alternative distributed layouts remain
outside this qualification.

The retained manifests, exact commands, source patches/new files, library hashes,
JSONL, geometry VTU output, and full CSV records are in:

- `/tmp/region-scaling-gcc-release-main`
- `/tmp/region-scaling-gcc-release-large`
- `/tmp/region-scaling-gcc-release-region-repeats`

To reproduce the measured axes, use the environment/build commands above and
invoke the runner with these arguments, choosing new output directories:

```sh
python3 verification/run_region_scaling.py --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-scaling-main-new --cells 16,32 --regions 1,2,8,32 --ranks 1,2,4 \
  --fixed-cells 32 --fixed-regions 2 --fixed-ranks 1 --samples 1 --repeats 5 --mpi-args '--bind-to core'
python3 verification/run_region_scaling.py --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-scaling-large-new --cells 64 --regions 2 --ranks 1 \
  --fixed-cells 64 --fixed-regions 2 --fixed-ranks 1 --samples 1 --repeats 5 \
  --modes native composite flattened materialized --timeout 300
python3 verification/run_region_scaling.py --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-scaling-repeat-new --cells 32 --regions 1,2,8,32 --ranks 1 \
  --fixed-cells 32 --fixed-regions 2 --fixed-ranks 1 --samples 3 --repeats 5 --modes composite
```
