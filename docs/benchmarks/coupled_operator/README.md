# Coupled operator measurements (2026-09-09)

These are small reproducible CPU measurements of the dirty working tree based
on `57fdfe3370f3d7a18c7e97485437675d79ec19c4`, GCC 16.2.1 Debug, Trilinos 17.2,
Kokkos Serial. Each JSONL is a **separate process**, with identical closed-wall
fixed-grid orthogonal momentum physics, precision, tolerance `1e-9`, maximum
400 iterations, block size 4 and `Num Blocks=min(100,global cells)`. The
preconditioner is the unchanged Ifpack2 one-sweep Jacobi / MueLu Schur
preconditioner (full reuse). See each file's metadata for compiler and type sizes.

Each process assembles, warms apply once, times 20 applications, solves, then
performs three timestep updates (factor 1.1), resetting initial fields to the
same prescribed state before each updated solve. Old returned systems are
released before normal updates. This is an assembly/linear-solve benchmark,
not a time-accurate physical benchmark. No reference matrix is assembled in
composite processes. Benchmark processes run sequentially after builds/tests.

## Serial measurements

CRS payload is observed unique graph/value view storage after setup. It excludes
maps/importers, static host reconstruction stencils, allocator metadata,
preconditioner internals and Krylov storage. Scratch is separately reported in
JSON; composite one-RHS scratch is 5,120 bytes at 64 cells and 17,280 at 216.
All sizes below are bytes; timing is milliseconds. These are single-run timing
samples, not statistically qualified speedups.

| Cells | Operator / workspace | CRS payload | Setup ms | Apply ms | First solve incl. preconditioner ms | Peak process RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | assembled / cached | 214,408 | 37.511 | 0.00375 | 10.667 | 65,454,080 |
| 64 | assembled / streamed | 153,280 | 43.903 | 0.00477 | 14.523 | 65,249,280 |
| 64 | composite / cached | 140,412 | 43.773 | 0.13553 | 13.945 | 65,667,072 |
| 64 | composite / streamed | 79,284 | 43.272 | 0.12727 | 16.990 | 65,626,112 |
| 216 | assembled / cached | 821,480 | 137.360 | 0.01530 | 25.858 | 72,159,232 |
| 216 | assembled / streamed | 593,312 | 157.341 | 0.01485 | 23.780 | 71,663,616 |
| 216 | composite / cached | 531,420 | 181.109 | 0.56425 | 58.845 | 71,987,200 |
| 216 | composite / streamed | 303,252 | 134.296 | 0.31795 | 39.530 | 71,499,776 |

At 216 cells, composite removes 290,060 CRS bytes (35.31%) from the cached
baseline. Streaming removes another 228,168 bytes (42.94% of composite/cached
CRS storage). Combined CRS reduction is 518,228 bytes (63.08%); after including
17,280 composite scratch bytes, the tracked payload reduction is 500,948 bytes
(60.98%). These are **not total solver memory percentages**. Peak process RSS
falls by 659,456 bytes (0.91%) in this sample. At 64 cells it instead rises by
172,032 bytes (0.26%). Startup, library/preconditioner/Krylov storage and
allocator noise dominate such small RSS comparisons.

Peak process RSS observed by the end of operator setup at 216 cells was
57,929,728 (assembled/cached), 58,019,840 (assembled/streamed), 55,865,344
(composite/cached), and 55,726,080 (composite/streamed). Thus streaming lowers
persistent payload without consistently lowering setup high-water RSS at
these sizes. The direct streamed Schur accumulation has different transient
insertion/compression costs. The composite application is slower in this Debug
build: retain that tradeoff when choosing the backend. Timing variation
between the two composite policies, which use the same apply implementation,
is itself evidence that these tiny samples are not stable performance rankings.

All four policies have serial iteration sequences `13,16,19,19` at 64 cells
and `27,50,64,81` at 216 cells. The largest measured true relative residual is
`9.128144e-10`; maximum 216-cell algebraic continuity L2 is about `5.60435e-9`
m3/s. The final clear checkpoint retains one momentum matrix owned by the
momentum equation and zero composite scratch. Composite monolithic build
counters remain zero at every checkpoint. Tests separately assert a maximum
of one live application-owned streamed product and no live products after
setup; imported/symbolic workspace inside Trilinos is not included in that
counter.

## Two-rank measurements

216 global cells, 108 owned cells/rank. Payload is the sum of disjoint local
CRS views; times and peak RSS below use the maximum rank.

| Operator / workspace | CRS bytes summed across ranks | Setup ms | Apply ms | First solve incl. preconditioner ms | Maximum rank peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| assembled / cached | 818,200 | 92.007 | 0.01803 | 29.475 | 72,699,904 |
| assembled / streamed | 589,960 | 77.575 | 0.01544 | 23.912 | 71,749,632 |
| composite / cached | 530,724 | 76.601 | 0.23453 | 32.892 | 72,773,632 |
| composite / streamed | 302,484 | 77.285 | 0.21602 | 33.002 | 72,011,776 |

Every policy has iteration sequence `27,49,65,85` and maximum true relative
residual below `8.746e-10`. Partitioned CRS payload differs from serial because
local graph representations/capacities differ. Each checkpoint records both
rank-local current RSS and a synchronized sum of current RSS. **That sum is
not a sum of independent rank high-water marks.** High-water RSS comes from
Linux `getrusage`, current RSS from `/proc/self/statm`; Linux accounting can
make the latter slightly exceed the former at small checkpoints. No device
allocations were measured on this CPU-only build.

## Larger pilot and limits

`assembled-cached_products-8-serial-failed.jsonl` is an explicitly failed
512-cell pilot, not part of the successful comparison tables. With the same
400-iteration cap it converged initially but a later update stopped at a true
relative residual about `5.96e-5`. An earlier 100-iteration pilot also failed.
No claim is made that the new backend fixes that baseline convergence problem.
No million-cell test was enabled or run. The accepted 100^3 CLI size is opt-in
and remains unqualified here.

The measured removed monolithic payload is approximately 1,156 bytes/cell at
64 cells and 1,343 bytes/cell at 216 cells. Naively multiplying the latter by
one million gives **1.343 decimal GB** (about 1.25 GiB), but this is an inference,
not a million-cell measurement. Boundary fractions, actual extended
stabilization graph, explicit zero slots and view capacities explain why it
need not match the prompt's illustrative 1.1 GB estimate. The 3.2 GB broader
redesign and 36–84 MB/scalar-matrix models are not measured or accepted by this
coupled-only implementation.

Preconditioner setup remains inside the production `solve()` call, so its
separate checkpoint/time and detailed retained preconditioner/Krylov/import
allocation breakdown are unavailable. The benchmark labels the combined solve
time and unsupported metrics explicitly. Peak setup RSS is process high-water
RSS through operator setup, not an allocation-level trace of setup temporaries.
Current local CRS view spans are measured; spare capacity outside these views
and allocator overhead are unsupported metrics. Geometry/cache storage is
part of RSS but not fabricated as a byte estimate. Full device and larger
converged performance qualification remains future work.

## Reproduce

Build once:

```sh
cmake --build --preset GCC-Debug --target coupled_operator_memory -j 4
```

Run one backend/policy per fresh process (for every combination, use this
same invocation with `assembled|block_composite`, `cached_products|streamed_products`,
and sizes 4 and 6; use `-n 2` with size 6 for the distributed table):

```sh
mpiexec -n 1 build/gcc/bin/Debug/coupled_operator_memory assembled 6 cached_products
mpiexec -n 1 build/gcc/bin/Debug/coupled_operator_memory block_composite 6 streamed_products
python src/benchmarks/summarize_coupled_memory.py \
  docs/benchmarks/coupled_operator/assembled-cached_products-6-serial.jsonl \
  docs/benchmarks/coupled_operator/block_composite-streamed_products-6-serial.jsonl
```

The comparison script validates mesh/rank/build/type/solver metadata and rejects
incomplete runs, then emits per-checkpoint absolute and percentage differences.
`comparison-216-serial.json` and `comparison-216-mpi2.json` are captured outputs.
All other successful raw JSONL files include initial setup, warmed application,
first solve, all three updates, and post-clear state.
