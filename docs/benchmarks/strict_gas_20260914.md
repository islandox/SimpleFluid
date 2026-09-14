# Strict active-gas workload comparison, 2026-09-14

The strict path now executes **20 gas linear-solver calls per ten-step window,
matching OpenFOAM**, while retaining identical SimpleFluid physical outputs.
Removing the 30 inactive-field calls reduces gas-update time by roughly one
third. At 98k cells the total runtime reduction is 4.1% for PISO and 1.7% for
NOX; the remaining OpenFOAM gaps are 4.40x and 12.59x respectively.

## What is aligned

Both applications transport microbubble number and moles with the liquid
carrier plus prescribed slip, then apply the same source and radius/void
closure for this reduced fixture. Dissolved and large-bubble populations start
at zero; conversion is disabled and dissolution is suppressed to working
precision. The general SimpleFluid path previously assembled and called the
linear solver for those three inactive fields as well.

The opt-in `--gas-transport skip-zero-auxiliary` policy checks each auxiliary
inventory for **exact global zero** using a collective MPI decision. On fixed
meshes these transport stages have zero source, homogeneous Neumann data and
no incoming physical-boundary flux, so zero is invariant during transport.
The skip occurs before matrix assembly and preconditioner setup. Kinetics
still executes; a subsequently populated field resumes normal transport.
Microbubble equations always execute, including their first zero-inventory
step. The default `full` policy and ALE behavior are preserved.

| Work over ten steps | SimpleFluid full | SimpleFluid strict | OpenFOAM |
| --- | ---: | ---: | ---: |
| Microbubble linear calls | 20 | 20 | 20 |
| Auxiliary linear calls | 30 | 0 | 0 |
| Auxiliary stages skipped after zero checks | 0 | 30 | N/A |
| Matrix assemblies | 30 | 10 | 20 |
| Gas iterations, 10k | 34 BiCGStab | 34 BiCGStab | 40 smoothing |
| Gas iterations, 98k | 36 BiCGStab | 36 BiCGStab | 50 smoothing |

SimpleFluid's number/moles pair already shares one operator. Thus equal
equations do not require equal assembly counts. Iteration types and residual
normalizations differ; neither iteration nor call counts are cost multipliers.
The statistics count each logical distributed solve once, without multiplying
by MPI rank count.

## Matched runs

Base source is `d0791e3`, including the preceding NOX workspace optimizations.
Full and strict paths use the **same instrumented executable and libraries**.
OpenFOAM is rebuilt with timers; its physical equations are unchanged.

- GCC Release/LTO, NOX and IF97 enabled, installed Trilinos 17.2 and OpenFOAM
  v2606; build directory `build/performance-strict-gas-20260914`.
- Two ranks bound to physical cores 0 and 1, one OpenMP/BLAS thread. No builds
  or profiles overlap timed executions.
- Same uniform `48 x 4 x 56` and `96 x 8 x 128` meshes, matched Z-slab ownership,
  material/source inputs, boundaries, initial states, and `dt=0.02 s`.
- Ten steps to `t=0.2 s`, with unchanged physical and residual acceptance gates.
  This is a startup window, not the full 20-second transient qualification.
- PISO retains pressure PCG/DIC; NOX retains analytic Newton, GMRES/Schur,
  block-composite/cached products and default forcing/refresh policies.
  Both retain scalar BiCGStab/SGS. OpenFOAM retains its existing solver settings.
- Three repetitions in alternating order, with a fresh OpenFOAM execution
  each repetition. All 30 runs passed. Five pilot runs are excluded from medians.

## Runtime results

All times are seconds for the ten-step window. Total loop time is the maximum
rank duration and includes per-step physics, diagnostics and CSV output.
Initial mesh/solver construction and launcher time are excluded.

| Cells | Mode | Full loop median | Strict loop median | Strict / OpenFOAM | Gas-update median, full → strict |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,752 | PISO | 2.366 | 2.211 | 3.74x | 0.495 → 0.349 |
| 10,752 | NOX composite | 8.720 | 8.758 | 14.82x | 0.515 → 0.353 |
| 98,304 | PISO | 29.352 | 28.158 | 4.40x | 4.534 → 3.039 |
| 98,304 | NOX composite | 81.951 | 80.566 | 12.59x | 4.507 → 3.105 |

OpenFOAM loop medians are **0.591 s at 10k** and **6.398 s at 98k**. Its
gas-update medians are 0.0403 s and 0.3963 s respectively.

At 98k, PISO's full/strict loop ranges are `[29.299,29.633]` /
`[27.964,28.482]` s; NOX's are `[81.945,82.241]` / `[80.483,80.757]` s.
The corresponding gas-update reductions are 33.0% and 31.1%.
At 10k, NOX loop ranges overlap substantially: full `[8.581,9.227]`, strict
`[8.434,9.186]` s. Its 0.4% higher strict median does not establish a regression
or a speedup; the gas-update reduction is clear, but the total change is below
the observed run variation. Full timings/ranges are in the
[unrounded CSV](strict_gas_20260914.csv).

These full/strict deltas isolate the new gas policy within this build. Changes
from older September 11/12 timings also include intervening code and host
timing variation and must not be attributed to inactive-field removal.

## Gas timing details

At 98k, the PISO-path medians show where the inactive-field cost occurred:

| Gas measurement | SimpleFluid full | SimpleFluid strict | OpenFOAM |
| --- | ---: | ---: | ---: |
| Matrix/RHS preparation | 1.710 | 0.306 | 0.0405 |
| Linear-solver calls | 0.248 | 0.205 | 0.0773 |
| Complete gas update | 4.534 | 3.039 | 0.3963 |

Solver-call timing includes preconditioner setup and residual checks. Removing
inactive calls mainly saves matrix/RHS preparation, while active Krylov work
is unchanged. Complete strict gas-update time remains about 7.7x OpenFOAM.
Most of SimpleFluid's remaining gas-update time lies outside those two measured
phases, in the combined property/kinetics, radius/derived-field reconstruction,
inventory accounting and publication work. Further phase timers or profiling
are needed to separate those contributions.

The phase boundaries are implementation-specific: SimpleFluid preparation also
includes per-population flux/weight/halo work; OpenFOAM prepares the common
bubble flux outside its per-equation timers, within complete gas-update time.
Times are maximum-rank local durations exported per call, and their separately
summed/median values are not an additive exclusive profile.

This comparison aligns active gas equations and removes inactive PDE setup.
It does not force identical carrier flux/temperature histories, matrix formats,
initial guesses, solver algorithms or residual normalization across applications.
Those fields are computed independently, and the shared physical comparison
gates remain the acceptance contract. It is therefore not an isolated
identical-matrix measurement of Trilinos overhead. Generic auxiliary-state
bookkeeping also remains in SimpleFluid's gas update.

## Correctness and validation

- All **12 full/strict pairs** (two meshes, two flow modes, three repetitions)
  have byte-identical fields, histories, turbulence and wall diagnostics.
  Flow/gas iteration and residual statistics also match, except the intentional
  gas-call count reduction. All 30 external comparison runs passed.
- OpenFOAM timer instrumentation preserves pilot fields/history/turbulence
  byte-for-byte against the retained uninstrumented reference outputs.
- All 39 existing serial gas tests passed. The two initial new policy tests
  passed serially; all 9 MPI gas tests passed on each of two ranks, including
  exact-zero skips, population activation on one rank, continued transport of
  populated auxiliaries, and fresh moles assembly after its number partner was
  skipped. Rank-divergent policy selection is rejected collectively.
- All 14 performance-runner checks and the shared-library ELF export boundary
  check passed. No full repository-wide suite was run.

## Reproduction and artifacts

The local artifact root is `build/verification/strict-gas-20260914`.
`analysis.json` contains all pair-equivalence checks; `summary.csv` contains
unrounded medians and ranges. `repeated/` holds per-run commands, source patch,
input/executable/library hashes, raw rank outputs, gas statistics and physical
comparison reports. Runtime hashes were checked before/after runs and again
after analysis; all remained unchanged.

The prepared fixtures reuse the previous matched meshes/decomposition, with
the newly instrumented OpenFOAM executable under `fixtures/bin`:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
python3 verification/openfoam/run_coupled_nonlinear_performance.py \
  --fixtures build/verification/strict-gas-20260914/fixtures \
  --build-dir build/performance-strict-gas-20260914 \
  --output build/verification/strict-gas-new --repeats 3 \
  --variants openfoam piso-gas-full piso-gas-strict \
    nox-composite-gas-full nox-composite-gas-strict
```

The next optimization target is the remaining gas property/reconstruction and
publication work, alongside the flow costs. The removed 30 calls do not explain
the overall OpenFOAM gap.
