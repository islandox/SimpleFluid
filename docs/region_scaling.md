# Region performance measurement contract

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
work. The [composite partitioner](composite_partitioning.md) provides a separate
graph/distributed-explicit path; this benchmark's contiguous-partition fixtures
do not qualify its scaling. Numerical convergence and decomposition-invariance
studies are independent of this linear benchmark's solve gate.

## Production versus reference assembly

The optional final `auto|reference` argument selects production region traversal
or the retained ordinal-query kernel. Use `--assemblies auto reference` in
`verification/run_region_scaling.py` to run each in separate processes with the
same executable, fields, matrix formulas and solve gates. This isolates dispatch
from source-version changes. Explicit materialization has amortized CSR growth;
its allocation regression is independent of timings.

Recorded Release studies completed the supported native/composite/materialized
cases and demonstrated lower production-assembly cost on their conforming
Cartesian fixtures. Flattened MPI remains an explicit unsupported result.
Correctness coverage separately includes motion/rollback, periodic/coarse-fine
parity and the [steady diffusion residual contract](steady_diffusion_convergence.md).
These studies do not establish scalable construction, general CFD speedups or
whole-process memory bounds. Recreate measurements with fresh artifacts rather
than treating old timings as the current executable's performance.

Report `T_reference/T_auto`, `(T_reference-T_auto)/T_reference`, and
`T_auto/T_native` separately. Medians of maximum-rank durations are not total CPU
cost; independently maximized timers are not additive. Compare identical build,
rank, binding, inputs and completed numerical gates, retaining failed runs and
unsupported cases outside successful timing cohorts.
