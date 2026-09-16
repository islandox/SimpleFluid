# Solver settings and staged kernel optimization, 2026-09-16

This work follows the priorities in the
[refreshed profiles](flame_profiles_20260916.md). The starting checkout is
`7bd19e1`, including shared FVM kernels and the shared backend that owns static
Trilinos. Historical tuning results are retested on a fresh Release build.

The retained changes are the turbulence Gauss cache and coefficient/Schur-slot
reuse. The analytic packing prototype passed correctness checks but showed no
repeatable whole-loop benefit, so its code and binaries remain experimental;
the production analytic operator is restored to its previous implementation.

## Measurement contract

Each comparison uses 10,752 and 98,304 cells, three alternating before/after
repeats, two MPI ranks bound to physical cores 0 and 1, one BLAS/OpenMP thread,
and ten steps of 0.02 s. Strict gas transport keeps 20 active solves and 30
skipped inactive stages. Residual, continuity, conservation, and spatial-output
gates are unchanged. Fresh OpenFOAM runs provide physical references; their
executable is the retained optimized RelWithDebInfo reference from the profile
fixture. Only matched SimpleFluid Release pairs establish speedup claims here.

The isolated build is `build/kernel-optimization-20260916`, Release
(`-O3 -DNDEBUG`), GCC, NOX/IF97/LTO enabled, with the existing non-LTO static
Trilinos backend. Source comes from a frozen checkout under
`build/verification/kernel-optimization-20260916/source`; agents' working-tree
edits do not enter a stage until copied explicitly. Source manifests, patches,
new headers, executable/library hashes, dependency resolution, commands,
and output comparisons accompany each preserved stage.

Code stages are cumulative, with fixed solver settings: baseline → Gauss cache
→ coefficient slots → analytic application. Each code comparison is against
the preceding stage. The setting comparisons use identical baseline binaries
on both sides. These conditional results must not be added into a predicted
combined speedup, nor confused with overlapping flame percentages.

An initial forcing campaign overlapped external compiler jobs in another
checkout. It was interrupted and retained as `forcing/`, explicitly excluded
from timing claims. The pressure campaign ended 168.8 seconds before the
earliest observed external build. After all implementation builds and tests,
15 host samples spanning 42 seconds showed no external build/test activity.
The fresh campaigns run sequentially with lightweight host monitoring on
separate cores. This documents observed contention, not complete OS isolation.

The fresh forcing campaign also encountered a sustained editor-indexing burst
overlapping the third large-mesh baseline run. Its complete matched triplet is
excluded from the selected timing set. The replacement selection was recorded
before launching a fresh large-mesh triplet; all other original outputs remain
available. Brief ordinary editor activity and unexplained timing variation are
retained in the reported ranges.

## Pressure setting on the current baseline

All 18 runs qualified. Times are medians in seconds with observed ranges.

| Cells | PCG/DIC | PCG/MueLu | Reduction | Aggregate flow iterations, DIC → MueLu |
| ---: | ---: | ---: | ---: | ---: |
| 10,752 | 2.274 [2.219, 2.545] | 1.974 [1.944, 2.130] | 13.2% | 3,890 → 1,276 |
| 98,304 | 30.033 [30.019, 34.127] | 19.705 [18.817, 20.291] | 34.4% | 8,301 → 1,309 |

Both policies retain 70 aggregate flow solves and 20 gas solves. Gas iterations
remain 34 and 36 for the two meshes. At 98,304 cells the maximum flow
true-relative residual is `8.956e-12` for DIC and `8.628e-12` for MueLu.
Maximum sampled differences are `2.274e-13 K` in temperature and
`4.937e-17 m/s` in velocity. Physical exports and solver statistics are
deterministic within each option across its three repeats.

This confirms a benefit from `--pressure-preconditioner muelu` in the tested
window. Native `FluidSolver` already enables MueLu pressure preconditioning
and reuse; the performance runner's PISO control explicitly selects DIC.
No distributed DIC ordering or communication algorithm was changed.

See [pressure analysis](../../build/verification/kernel-optimization-20260916/pressure/analysis.json)
and [physical deviations](../../build/verification/kernel-optimization-20260916/pressure/physical-deviations.json).

## Initial NOX forcing on the current baseline

The selected cohort contains three matched repetitions on each mesh: all three
small-mesh triplets from `forcing-repeat`, large-mesh triplets 1 and 2, and the
predeclared replacement large-mesh triplet. All 18 selected runs qualified.
Both settings use identical executables and libraries; initial forcing changes
from `0.1` to `0.001`, while final acceptance gates remain unchanged.

| Cells | Default seconds [range] | Initial `0.001` seconds [range] | Median reduction | Nonlinear/Krylov counts |
| ---: | ---: | ---: | ---: | --- |
| 10,752 | 7.244 [7.205, 7.675] | 6.028 [5.783, 6.132] | 16.80% | 36/580 → 20/517 |
| 98,304 | 71.811 [70.949, 78.250] | 56.380 [56.339, 56.608] | 21.49% | 37/706 → 20/615 |

At 98,304 cells, median linearization decreases from 25.077 to 13.739 s and
linear solves from 23.819 to 21.182 s. Maximum sampled differences across all
selected pairs are `5.718e-9 K` in temperature, `1.678e-12 m/s` in velocity,
and `2.242e-15` in gas fraction. The small-mesh hydrogen-balance difference is
`8.47e-22 mol`; the large-mesh balance difference is zero. Repeated outputs are
deterministic within each option, including the replacement run.

The option remains explicit: `--nonlinear-forcing-initial 0.001` in the
executable or `--nox-forcing-initial 0.001` in the performance runner.
[Selection records](../../build/verification/kernel-optimization-20260916/forcing-selected/selection.json)
retain the original run IDs and the excluded triplet;
[selected analysis](../../build/verification/kernel-optimization-20260916/forcing-selected/analysis.json)
checks cross-campaign inputs, runtime artifacts, settings, and physical outputs.

## Kernel implementation

**Gauss geometry.** A separate `GaussLinearGradientCache` stores face-ordered
local incidences, neighbour indices, distances, area vectors, and cell volumes.
Cached scalar/vector overloads retain live boundary callbacks and the original
arithmetic order. Turbulence owns only the cache for its selected gradient
scheme, and refreshes Gauss geometry after motion or rollback. Direct use of
a stale cache still fails. The optimization targets the turbulence path
identified in the profiles.

**Coefficient locations and geometry blocks.** Coupled assembly replaces seven
per-cell hash accumulators with face-to-row coefficient slots and reusable
scratch, preserving structural zeros. Compatible gradient, divergence, and
stabilization blocks are retained after collective dependency checks, while
live boundary/RHS terms are evaluated. Cached Schur assembly replays source
contributions into final CRS value slots in the original summation order.
Its keys retain all source/destination graphs and the gauge. Momentum-dependent
products remain current; retained generations prevent mutation.

The existing MueLu `RP` policy is unchanged: transfer operators are retained and
smoothers recomputed, with graph stability required for reuse. This matches
[MueLu's reuse contract, section 5.8](https://trilinos.github.io/pdfs/mueluguide.pdf).
Streamed Schur accumulation keeps its established path. An unchanged
streamed-to-cached transition needs six additional intermediate matrices;
changing the timestep creates the other three geometry products, and returning
to streamed mode releases them.

**Analytic experiment, not retained.** The prototype combined velocity/pressure
imports into four columns, aliased pressure storage, and specialized directional
fluxes using immutable geometry and homogeneous boundary types. It retained
pressure-gradient/flux exchanges, base application, ordered donor accumulation,
and private scratch. Its isolated
[patch](../../build/verification/kernel-optimization-20260916/analytic-only.patch)
and preserved `artifacts/analytic` build make the experiment reviewable.

## Gauss cache: matched results

All 30 runs qualified, with byte-identical physical exports and identical
non-timing linear/gas/NOX statistics in all 12 implementation pairs. Stage
source reconstruction and runtime hashes passed. Only this cache stage differs
between each pair; pressure remains PCG/DIC and initial NOX forcing remains 0.1.

| Cells | Solver | Before seconds [range] | Gauss cache seconds [range] | Median reduction |
| ---: | --- | ---: | ---: | ---: |
| 10,752 | PISO | 2.187 [2.146, 2.302] | 1.840 [1.817, 2.069] | 15.88% |
| 98,304 | PISO | 31.346 [30.360, 32.751] | 28.325 [28.018, 33.788] | 9.64% |
| 10,752 | NOX | 7.917 [7.752, 8.297] | 7.921 [7.705, 9.499] | −0.05% |
| 98,304 | NOX | 75.816 [74.910, 77.726] | 77.805 [74.784, 82.717] | −2.62% |

The small PISO improvement is consistent across repetitions. Large PISO has
a lower median, but overlapping ranges and one reversed pair. NOX has no
demonstrated whole-loop improvement: its medians are flat/slower, with
overlapping ranges and unchanged nonlinear work. Negative reductions above
mean a higher median. No repetition was discarded because of its speed.
See the complete [Gauss analysis](../../build/verification/kernel-optimization-20260916/gauss/analysis.json).

## Coefficient slots: matched results

This comparison adds coefficient/Schur slots to the Gauss stage with unchanged
NOX settings. All 18 runs qualified; all six implementation pairs have
byte-identical physical exports and identical non-timing statistics.

| Cells | Before seconds [range] | With slots seconds [range] | Median reduction |
| ---: | ---: | ---: | ---: |
| 10,752 | 7.034 [7.014, 7.237] | 5.702 [5.550, 5.889] | 18.93% |
| 98,304 | 67.227 [67.043, 67.460] | 56.125 [55.547, 56.764] | 16.51% |

All three pairs improve on both meshes, with nonoverlapping timing ranges.
At 98,304 cells, median linearization falls from 25.127 to 15.114 s and native
setup from 7.193 to 5.289 s. Linear-solve time is similar at 23.230 versus
23.733 s. Nonlinear/Krylov counts remain 36/580 at 10,752 cells and 37/706
at 98,304 cells. Phase timers include nested work and independently maximized
rank times, so they are not an additive decomposition.

See the [coefficient-slot analysis](../../build/verification/kernel-optimization-20260916/slots/analysis.json).

## Analytic experiment: matched results

All 18 runs qualified and all six pairs preserve physical outputs and
non-timing statistics byte-for-byte.

| Cells | Slots stage seconds [range] | Analytic prototype seconds [range] | Median reduction |
| ---: | ---: | ---: | ---: |
| 10,752 | 5.592 [5.430, 5.761] | 5.597 [5.342, 6.630] | −0.08% |
| 98,304 | 52.658 [52.258, 53.200] | 52.492 [51.813, 52.961] | 0.31% |

These overlapping ranges establish no throughput benefit. Large-mesh
linear-solve medians are 21.582 versus 21.758 s, with unchanged nonlinear and
Krylov counts. The extra private flux implementation and aliasing machinery are
therefore not retained in production. The final source matches the measured
`slots` stage. Full results remain in the
[analytic analysis](../../build/verification/kernel-optimization-20260916/analytic/analysis.json).

## Focused validation

- Gauss stage: 38 serial tests; registered two-rank gradient/turbulence tests;
  the new motion/rollback regression also passed on two ranks.
- Coefficient stage: 25 coupled-assembly and 13 block-operator tests passed;
  42 nonlinear tests passed with two MPI-only skips. Registered block/nonlinear
  MPI checks and all four new coefficient-reuse parameter cases passed on two
  ranks.
- Analytic experiment: 43 nonlinear tests passed with two MPI-only skips; all 45
  tests passed in the two-rank registration. The generic-flux oracle includes
  least-squares/Gauss reconstruction, periodic and reordered meshes, Slip,
  multiple columns, noncontiguous views, scaling, and input/output aliasing.
- Solver and FVM export-boundary checks passed for each preserved code stage.
- After removing the analytic prototype, the production rebuild passed 42
  serial nonlinear tests (two MPI-only skips), all 44 two-rank nonlinear tests,
  and both export checks. All 435 source/build files, the executable, every
  shared library, and the build cache match the measured `slots` stage exactly;
  see [production finalization](../../build/verification/kernel-optimization-20260916/production-finalization.json).

Build and test logs are retained in
`build/verification/kernel-optimization-20260916/correctness/`. Caches add
geometry/slot metadata; no total-memory reduction is claimed. These are focused
regressions and short startup windows, not full-transient or scaling validation.

All 105 runs in completed campaigns passed the physical gates; 102 distinct
runs enter the selected timing cohorts after excluding the indexing-affected
triplet. All 24 code-stage pairs (including the unretained prototype) are exact.
The earlier interrupted forcing campaign remains separate. Runtime/input hashes
and reconstructed stage sources passed the final audits. The final host record
covers all 87 fresh recorded launcher intervals; no compiler/test activity
overlapped them, and sustained indexing was confined to the excluded triplet.
Brief editor activity and other timing variation remain visible.

Machine-readable results are in
[report-metrics.json](../../build/verification/kernel-optimization-20260916/report-metrics.json),
with [host quality](../../build/verification/kernel-optimization-20260916/fresh-host-quality.json)
and [interval correlation](../../build/verification/kernel-optimization-20260916/fresh-host-correlation.json).
