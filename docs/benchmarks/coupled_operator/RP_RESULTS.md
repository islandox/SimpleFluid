# Coupled backend measurements after numeric-refresh fixes

Measured on clean commit `64059d787ff4fee246f1e255a1a54ae181d1129d`, GCC
16.2.1 Debug, Trilinos 17.2, Kokkos Serial, on 2026-09-09. All 16 independent
process runs completed the first solve and three numeric updates within the
unchanged tolerance `1e-9` and maximum 400 iterations. This includes the
previously failing 512-cell size and 1,728 cells on one, two and four ranks.

Settings, physics and warm-up are identical to the historical benchmark,
except MueLu now uses the corrected `RP` numeric-refresh policy. The benchmark
remains fixed-grid orthogonal Stokes assembly with a frozen zero convective
flux, reset initial fields and a timestep increase of 10% per update. It is
not a physical transient-validation case. Every backend/policy runs in a fresh
process, sequentially after builds/tests; no dual matrix construction occurs.

All types are measured in the metadata: scalar 8 bytes, local index 4, global
index 8 and offset 8. Times below are milliseconds; all memory figures are
bytes. CRS payload sums disjoint rank-local graph/value view spans and
explicitly excludes maps/imports, preconditioner/Krylov internals and allocator
metadata. Setup/apply/solve times and peak RSS use the maximum rank. These are
single-run samples, not statistically qualified speedups.

## Serial

| Cells | Operator / workspace | CRS payload | Setup ms | Apply ms | First solve incl. preconditioner ms | Peak process RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | assembled / cached | 2,065,608 | 358.229 | 0.0501 | 50.947 | 70,746,112 |
| 512 | assembled / streamed | 1,498,752 | 343.323 | 0.0231 | 47.821 | 70,184,960 |
| 512 | composite / cached | 1,329,084 | 336.041 | 0.7067 | 89.909 | 70,340,608 |
| 512 | composite / streamed | 762,228 | 333.436 | 0.6324 | 78.373 | 69,812,224 |
| 1,728 | assembled / cached | 7,376,648 | 1,173.450 | 0.1912 | 249.515 | 88,190,976 |
| 1,728 | assembled / streamed | 5,375,552 | 1,221.825 | 0.1356 | 136.271 | 86,605,824 |
| 1,728 | composite / cached | 4,722,684 | 1,090.736 | 2.6759 | 234.605 | 85,852,160 |
| 1,728 | composite / streamed | 2,721,588 | 1,285.746 | 3.0622 | 305.043 | 83,910,656 |

At 1,728 cells, composite removes 2,653,964 CRS bytes (35.98%) from the cached
baseline. Streaming removes another 2,001,096 bytes (42.37% of composite/cached
payload). Combined CRS reduction is 4,655,060 bytes (63.11%). Composite
one-RHS scratch is 138,240 bytes, leaving a net tracked-payload reduction of
4,516,820 bytes (61.23%). These percentages are not total solver memory savings.

Peak process RSS falls by 4,280,320 bytes (4.85%) from assembled/cached to
composite/streamed in the 1,728-cell sample. High-water RSS through operator
setup is 79,990,784 versus 62,107,648 bytes, respectively. Peak setup includes
transient insertion/compression allocations and startup state, so its
difference need not equal final CRS payload removal. Decimal MB means 10^6
bytes; MiB means 2^20 bytes. No million-cell extrapolation is presented as a
measurement.

Composite applications remain slower in Debug. The timing variation between
the two policies for the same operator implementation also shows why a single
small run cannot establish a stable performance ranking. Memory reduction,
converged updates and numerical equivalence are the established outcomes.

## Distributed, 1,728 global cells

| Ranks | Operator / workspace | CRS bytes summed across ranks | Setup ms | Apply ms | First solve incl. preconditioner ms | Maximum rank peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | assembled / cached | 7,363,000 | 701.992 | 0.1416 | 132.326 | 79,822,848 |
| 2 | assembled / streamed | 5,361,832 | 702.461 | 0.1187 | 135.060 | 78,823,424 |
| 2 | composite / cached | 4,719,396 | 634.942 | 2.1295 | 182.341 | 78,815,232 |
| 2 | composite / streamed | 2,718,228 | 610.774 | 1.3370 | 222.822 | 76,746,752 |
| 4 | assembled / cached | 7,335,704 | 430.934 | 0.0658 | 121.612 | 74,498,048 |
| 4 | assembled / streamed | 5,334,392 | 439.098 | 0.0584 | 76.482 | 74,158,080 |
| 4 | composite / cached | 4,712,820 | 383.903 | 1.1417 | 140.020 | 74,014,720 |
| 4 | composite / streamed | 2,711,508 | 433.466 | 2.3277 | 154.433 | 72,839,168 |

Rank-local owned counts are 864 and 432 for two and four ranks. Every raw
checkpoint separately records current per-rank RSS and synchronized summed
current RSS, as well as independent rank high-water RSS and its maximum.
The synchronized sum is not a sum of independent high-water marks. The
existing Linux `/proc/self/statm` versus `getrusage` accounting caveat applies.

## Convergence and release checks

All four backend/policy combinations have these iteration sequences:

| Cells / ranks | First solve and three updates |
| --- | --- |
| 512 / 1 | 36, 38, 38, 38 |
| 1,728 / 1 | 42, 42, 44, 45 |
| 1,728 / 2 | 37, 39, 40, 41 |
| 1,728 / 4 | 33, 35, 35, 36 |

Maximum true relative residual across all solves is below `9.014e-10`.
Maximum reported algebraic continuity L2 is below `1.270e-9` m3/s.
Every composite checkpoint has zero monolithic matrix builds. Every final
clear checkpoint retains only the momentum equation's one matrix and no
composite application scratch. The current tests separately bound streamed
product lifetime and validate physical reconstructed continuity for the full
solver paths.

The failed `full`-reuse pilot remains preserved as historical evidence. It was
not removed or relabeled as successful. The new runs show that refreshing
numeric preconditioning resolves that reproduced update failure.

## Reproduce and inspect

```sh
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug --target coupled_operator_memory -j 4

for n in 8 12; do
  for backend in assembled block_composite; do
    for workspace in cached_products streamed_products; do
      mpiexec -n 1 build/gcc/bin/Debug/coupled_operator_memory "$backend" "$n" "$workspace"
    done
  done
done
for ranks in 2 4; do
  for backend in assembled block_composite; do
    for workspace in cached_products streamed_products; do
      mpiexec -n "$ranks" build/gcc/bin/Debug/coupled_operator_memory "$backend" 12 "$workspace"
    done
  done
done

python src/benchmarks/summarize_coupled_memory.py \
  docs/benchmarks/coupled_operator/rp-assembled-cached_products-12-ranks1.jsonl \
  docs/benchmarks/coupled_operator/rp-block_composite-streamed_products-12-ranks1.jsonl
```

Captured local output names are `rp-<backend>-<workspace>-<n>-ranks<r>.jsonl`,
with four corresponding `rp-comparison-<cells>-ranks<r>.json` comparisons.
Generated JSON remains local under the repository's ignore rule; this written
report and the reproducible driver are committed. Comparison metadata checks
commit/build/ranks/types/tolerances/basis/preconditioner compatibility and
rejects incomplete runs. These measurements do not add accelerator or
million-cell qualification, separate preconditioner timing, or fabricated
preconditioner/Krylov/import allocation counts.
