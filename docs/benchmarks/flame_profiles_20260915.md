# RelWithDebInfo flame profiles, 2026-09-14–15

[Open the interactive gallery](../../build/verification/flames-20260914/index.html)
or [download the chart bundle](../../build/verification/flames-20260914/flame-charts.zip).
The bundle contains 27 interactive SVGs, including six time-order perf charts,
full-symbol Speedscope JSON files, summaries and provenance. Raw profiler
traces remain in the local artifact directory and are excluded from the bundle.

## Captures

Source: `69a7572`, with no solver-source changes. SimpleFluid was built in
`build/performance-strict-gas-20260914`, configuration `RelWithDebInfo`:
`-O2 -g -DNDEBUG -fno-omit-frame-pointer`, with LTO, NOX and IF97 enabled.
The OpenFOAM reference executable was rebuilt with `-O2 -g` and frame pointers;
its installed optimized shared libraries were retained.

All runs use two MPI ranks bound to physical cores 0 and 1, one BLAS/OpenMP
thread, strict gas transport, and ten physical steps of 0.02 s. PISO uses
pressure PCG/DIC, NOX uses analytic Newton with GMRES/Schur and the existing
block-composite/cached-products backend, and scalar transport remains
BiCGStab/SGS. The same physical inputs and acceptance gates are retained.

| Tool and scope | Mesh | PISO | NOX | OpenFOAM |
| --- | ---: | ---: | ---: | ---: |
| perf, whole-process user CPU sampling | 98,304 | 14,699 samples | 38,259 samples | 2,997 samples |
| Callgrind, ten `BoussinesqSolver::step()` calls | 10,752 | 58.772 billion instructions | 195.656 billion instructions | — |

Counts above combine both ranks. Perf uses `cpu-clock:u`, 199 Hz, a 65,528-byte
DWARF stack dump and the monotonic clock. Time-order charts preserve sample
order and CPU-period weights; they exclude unsampled/off-CPU gaps. Aggregate
flame graphs merge repeated stacks.

Callgrind records instruction self costs and global-bus events, with cache and
branch simulation disabled. Its explicit `--separate-callers=16` contexts are
decoded directly into bounded stacks. Every chart's instruction weights equal
the Callgrind summary exactly. No proportional call-graph expansion or inferred
chronology is used. The flat self-cost charts are also available.

## Main findings

Native perf self-cost categories, summed across both ranks:

| Function family | PISO | NOX |
| --- | ---: | ---: |
| Mesh geometry, indexing and dispatch | 19.0% | 20.1% |
| Field and Tpetra map/view access | 10.0% | 16.2% |
| Sparse, BLAS and smoother functions | 26.3% | 15.4% |
| MPI and communication runtime | 16.0% | 2.1% |
| Named allocation/reference/type management | 3.7% | 5.5% |

These are exclusive self-cost groups based on function scope and library,
not argument-type keyword matches. Other named work and unresolved system
internals comprise the remainder. The following phase percentages are
**inclusive** and overlap those self-cost groups:

- **PISO:** pressure projection accounts for 45.7% of sampled CPU cost. The
  largest named self-cost functions include distributed DIC application and
  `KokkosSparse::Impl::SPMV::spmv`. MPI runtime is also substantial.
- **NOX:** native linearization accounts for 35.5%, and analytic coupled
  operator application for 14.7%. Repeated `Tpetra::MultiVector::getLocalLength`,
  `getData`, map lookups, `MeshBase::cell_id`/`face_id`, topology queries and
  `MeshHandle` variant dispatch are visible throughout these paths.
- **Gas update:** the named `RadiolyticGasModel::advance` context accounts for
  about 8.4% in PISO and 3.4% in NOX. The gas-subset charts select stacks with
  gas-named frames; inlining can limit that selection's coverage.
- **OpenFOAM:** DIC/Gauss–Seidel smoothing and LDU residual/matrix-vector work
  dominate the named hotspots. The complete reference graph is in the gallery.

Callgrind independently places map/view access, mesh indexing/dispatch, DIC and
sparse kernels among the high instruction-count functions. NOX executes about
3.33 times the profiled timestep instructions of PISO at 10k cells. This is an
instruction-count comparison, not a wall-time speed ratio.

The strongest code-review target is repeated mesh/index translation and
map/view acquisition inside operator applications and geometry traversals:
hold scoped host views and stable lookup results across each kernel loop,
and use cached geometry/face rows where the mesh epoch permits. PISO pressure
preconditioning and its communication should be evaluated separately. Changes
still require matched timings and numerical validation before claiming gains.

## Quality and limits

- Every accepted profile run passes the existing physical comparison and
  produces byte-identical fields, histories, turbulence and wall diagnostics
  versus its corresponding native RelWithDebInfo control. Strict gas counts
  remain 20 solves and 30 skipped auxiliary stages per SimpleFluid window.
- Accepted perf captures report no buffer loss. The initial OpenFOAM capture
  reported one lost chunk and was replaced with a 4 MiB-buffer capture; both
  are retained, but only the replacement is charted.
- Perf root coverage is 98.7–99.4%. Unresolved leaf cost is 18.1% for PISO,
  8.3% for NOX and 9.9% for OpenFOAM, confined to prebuilt system/MPI libraries.
  These frames retain their library labels and are not reassigned to named
  functions. SimpleFluid library leaf symbols resolve.
- Callgrind's caller limit applies to 40.9% of PISO instruction costs and
  80.2% of NOX instruction costs; affected stacks explicitly mark older callers
  as truncated. Perf retains deeper call chains for phase attribution.
- Automatic debug-symbol downloads stalled the initial Callgrind attempt.
  That attempt was terminated; final captures use local symbols with
  `DEBUGINFOD_URLS` empty. Valgrind's brk-growth notice was recorded, but both
  complete runs passed. Allocation/MPI-wait behavior under instrumentation
  must not be treated as native timing.
- Native controls also retain `perf stat` hardware counters. Only `cpu_core`
  events are used in summaries; unsupported/duplicated hybrid aliases are
  excluded. These measurements run under WSL2.
- No captures overlap. Saved-data postprocessing was pinned to separate cores.
  Profiled wall times are not compared with earlier Release benchmark medians.
  This remains a short startup window, not full-transient qualification.

## Reproduction

All scripts, commands, hashes, counter files and validation reports are under
`build/verification/flames-20260914`. Start with its [README](../../build/verification/flames-20260914/README.md),
[provenance](../../build/verification/flames-20260914/provenance.json), and
[machine-readable summary](../../build/verification/flames-20260914/summary.json).

The SVG renderer is the unmodified official
[FlameGraph](https://github.com/brendangregg/FlameGraph) implementation.
Callgrind collection and caller separation follow the
[Valgrind manual](https://valgrind.org/docs/manual/cl-manual.html).
