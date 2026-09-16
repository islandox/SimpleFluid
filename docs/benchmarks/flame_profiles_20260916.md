# Refreshed RelWithDebInfo flame profiles, 2026-09-16

[Open the interactive gallery](../../build/verification/flames-20260916/index.html)
or [download the chart bundle](../../build/verification/flames-20260916/flame-charts.zip).
The gallery includes aggregate flame graphs, per-rank on-CPU time-order charts,
full-symbol Speedscope files, gas subsets, and Callgrind instruction graphs.

Direct native CPU graphs:
[PISO](../../build/verification/flames-20260916/charts/perf-piso-combined.svg),
[NOX](../../build/verification/flames-20260916/charts/perf-nox-combined.svg),
[OpenFOAM](../../build/verification/flames-20260916/charts/perf-openfoam-combined.svg).

## Workload and build

Fresh source `17fec85990a9cb13f4ba071e138beb090cbae365`, including the kernel
optimizations and current shared FVM/static-Trilinos backend layout. The isolated
GCC 16.2.1 build is `build/profile-rwdi-20260916`, RelWithDebInfo, with NOX, IF97,
frame pointers and LTO enabled. The static Trilinos backend retains its required
non-LTO link. The OpenFOAM reference executable was rebuilt with `-O2 -g` and
frame pointers; installed optimized OpenFOAM libraries were retained.

The previous profiling controls are unchanged: two ranks bound to physical
cores 0 and 1; one OpenMP/BLAS thread; equal Z partitions; ten steps of 0.02 s;
strict gas transport. PISO uses PCG/DIC pressure and BiCGStab/SGS transport.
NOX uses analytic Newton, GMRES/Schur, block-composite/cached products, restart
80 and initial forcing 0.1. Opt-in MueLu pressure and tighter initial forcing
are not selected in this refresh.

| perf capture, both ranks | Cells | Samples | Accumulated sampled CPU seconds |
| --- | ---: | ---: | ---: |
| OpenFOAM | 98,304 | 2,858 | 14.362 |
| PISO | 98,304 | 12,939 | 65.020 |
| NOX | 98,304 | 30,207 | 151.794 |

Perf uses user CPU-clock sampling at 199 Hz, a 4 MiB buffer, 65,528-byte DWARF
stack dumps and the monotonic clock. These totals include both ranks and
whole-process setup; they are not elapsed times. Time-order charts retain
sample order and CPU-period weights while excluding unsampled/off-CPU gaps.

Callgrind captures ten `BoussinesqSolver::step()` calls at 10,752 cells:

| Solver | Instructions, both ranks | Direct instruction graph |
| --- | ---: | --- |
| PISO | 53,207,291,994 | [Open](../../build/verification/flames-20260916/charts/callgrind-piso-combined.svg) |
| NOX | 152,239,400,676 | [Open](../../build/verification/flames-20260916/charts/callgrind-nox-combined.svg) |

Cache and branch simulation are disabled; instruction self costs and global-bus
events are collected. Up to 16 actual callers are decoded from the recorded
contexts, with older callers explicitly marked as truncated. No call-graph
reconstruction or chronology is inferred. Instruction counts and instrumented
wall times are not native performance measurements.

## Current hotspots

Exclusive native CPU self-cost groups, using function scope and library:

| Function family | PISO | NOX |
| --- | ---: | ---: |
| Mesh geometry, indexing and dispatch | 19.5% | 17.4% |
| Field and Tpetra map/view access | 7.7% | 8.3% |
| Sparse, BLAS and smoother kernels | 27.2% | 17.4% |
| MPI and communication runtime | 15.5% | 2.4% |
| Named allocation/reference/type management | 2.3% | 5.4% |

Other named work and unresolved system code comprise the remainder. The
following inclusive percentages overlap those self-cost groups:

- PISO pressure projection: **46.7%**. Distributed DIC application is 26.0%
  inclusive and 13.7% self; sparse matvec is another 9.6% self. MPI CPU polling
  appears in the caller stacks and must not be interpreted as elapsed wait time.
- NOX linearization: **34.2%**; analytic coupled operator application:
  **16.4%**. Remaining leaf costs include mesh-ID translation, map access,
  pressure-weighted face fluxes and cached gradients. NOX retains 37 nonlinear
  iterations, 706 Krylov iterations, 57 residual evaluations and zero line-search
  rejections at 98,304 cells.
- OpenFOAM: DIC smoothing is 20.6% self, Gauss–Seidel smoothing 19.7%, LDU
  residual evaluation 10.8%, and LDU matrix-vector application 9.0%.

Each graph scales to its own total. Frame-width changes against older charts
are not speedup measurements. This refresh includes both kernel and library
layout changes since the original `69a7572` capture, so it does not isolate the
effect of either change.

## Native controls

These are single fresh RelWithDebInfo control runs, collected before profiling.
They establish numerical controls and descriptive timings, not repeated Release
benchmark medians. All physical history/conservation and spatial-output checks
passed.

| Cells | OpenFOAM loop seconds | PISO loop seconds | NOX loop seconds |
| ---: | ---: | ---: | ---: |
| 10,752 | 0.628 | 2.370 | 7.700 |
| 98,304 | 6.961 | 31.155 | 75.891 |

The fresh OpenFOAM fields and histories match the retained qualified reference
byte-for-byte on both meshes. Strict gas work remains 20 active solves in each
solver and 30 skipped inactive auxiliary stages in SimpleFluid. The reference
does not solve those inactive auxiliary equations.

## Validation

All 11 runs passed: six native controls, three perf captures and two Callgrind
captures. Every profiled run reproduces its corresponding native fields,
history, turbulence and wall diagnostics byte-for-byte, together with every
non-timing gas, linear and nonlinear statistic. Executable and SimpleFluid
library hashes remain unchanged. No perf buffer loss was reported.

All four Callgrind rank graphs exactly reproduce their raw instruction and
global-bus event totals. The caller-depth bound applies to 45.7% of PISO and
78.0% of NOX instruction costs; those stacks mark the missing older callers.
Raw addresses remain where dependency symbols are unavailable, and flat
self-cost views retain the module names. Valgrind reported its brk-growth
notice on each rank; both complete runs passed all output checks. Its
allocation and MPI polling behavior must not be interpreted as native timing.

The bundle contains 27 XML-validated interactive SVGs, including six per-rank
time-order charts, plus six full-symbol Speedscope JSON files. See the
[validation record](../../build/verification/flames-20260916/final-validation.json).

## Artifacts and scope

Commands and toolchain details are in the
[README](../../build/verification/flames-20260916/README.md),
[provenance](../../build/verification/flames-20260916/provenance.json), and
[summary](../../build/verification/flames-20260916/summary.json).
Raw traces, input hashes, binary hashes, per-rank outputs and validation reports
remain under `build/verification/flames-20260916`. The shareable archive omits
the raw traces. No solver implementation or default was changed for this task.

Profilers ran sequentially after builds finished. Saved-data processing used
separate physical cores. Dependency debug information was not rebuilt, and
automatic debuginfod downloads were disabled throughout. Perf's unresolved
leaf costs are 17.9% PISO, 10.0% NOX and 10.3% OpenFOAM, entirely in prebuilt
system/MPI libraries; SimpleFluid DSO leaves resolve. Root coverage is
98.8–99.2%. Remaining mangled C++ symbols are resolved using `llvm-cxxfilt`.
Hardware-counter summaries exclude duplicate/unsupported hybrid PMU aliases.
Measurements are under WSL2 and cover a short startup window, not the full
20-second transient.
