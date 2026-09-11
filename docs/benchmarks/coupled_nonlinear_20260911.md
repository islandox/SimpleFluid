# Coupled nonlinear convection comparison, 2026-09-11

NOX now runs the physical bottom-heated bubbly-convection case. On the measured
startup window, it costs more than PISO and coupled Krylov. The existing
block-composite representation substantially reduces NOX's assembly cost, but
does not close the runtime gap to OpenFOAM.

## Implementation and scope

Source base: `e2d035e`, with the accompanying uncommitted Boussinesq NOX extension,
verification controls and runner. Each campaign directory records the source
patch, untracked sources, both executable hashes, SimpleFluid shared-library hashes, build cache hash,
input hashes, environment, CPU information, and exact commands. The requested
external Codex thread could not be addressed by available tools; the extension
was implemented and reviewed locally with an implementation agent.

The private timestep problem copies temperature, density, effective viscosity,
turbulent-pressure gradient and boundary viscosity. It retains native buoyancy,
stress and pressure normalization. Velocity and pressure are solved together
with trial-dependent convection and pressure-weighted face flux. Thermal, SST
and gas transports then advance once in their existing order. This is a
nonlinear **velocity-pressure** solve with frozen auxiliary physics, rather
than a monolithic solve of every transported field.

Failed flow solves preserve primary fields and accepted diagnostics and restore
supported gas/material snapshots. Boiling, precursors, free-surface/ALE, custom
momentum subclasses and unsupported boundaries fail closed. Caller-owned
updater side effects and failures after flow acceptance are outside an atomic
multiphysics rollback guarantee. See the
[solver contract](../architecture/coupled_nonlinear_solver.md).

## Measurement contract

- GCC Release, `-O3`, LTO, NOX and IF97 enabled; installed Trilinos 17.2 and
  OpenFOAM v2606. Build: `build/performance-nox-20260911`.
- Two MPI ranks, bound to physical cores 0 and 1; one BLAS/OpenMP thread.
  Timed jobs were sequential, with no concurrent builds or profiling.
- Identical uniform meshes: `48 x 4 x 56 = 10,752` and
  `96 x 8 x 128 = 98,304` cells. OpenFOAM uses the matching two Z slabs;
  IsoRegion also splits into two conforming Z regions.
- Identical properties, source support, boundaries, initial fields, timestep
  `0.02 s`, and ten physical steps. Outputs at `0` and `0.2 s`.
- Runtime is maximum-rank loop wall time, including per-step setup, physics,
  diagnostics and output. Initial mesh/solver construction and launcher cost
  are excluded. Whole-process elapsed times are retained in raw measurements.
- PISO: three pressure correctors in one outer sweep, pressure PCG/DIC.
  Scalar transports use BiCGStab/SGS for every SimpleFluid mode.
- Coupled Krylov uses the existing block GMRES/Schur path. NOX uses analytic
  Newton, Type 2 inexact forcing, independent GMRES corrections, restart 80,
  and the same native Schur preconditioner. Nonlinear tolerances retain their
  defaults: component relative `1e-8`, absolute `1e-10`, integrated maximum
  continuity `1e-10 m^3/s`; reference scales are one.
- These algorithms perform different convergence work. NOX's inner forcing
  tolerance is not the fixed scalar linear tolerance. Final nonlinear
  residuals, physical acceptance and matching physical time govern acceptance.
- OpenFOAM timings are fresh runs. Its solver source is unchanged from the
  retained reference executable; the binary hash is recorded.

This is a short startup performance comparison. It does not qualify the full
20-second transient or establish performance for fully developed turbulent
flow. OpenFOAM also retains the previously documented two gas equations versus
SimpleFluid's five gas transport calls per step.

## Three-repeat results

Every run below passed the existing global quantity/conservation comparison
and spatial-output validity checks. The runner alternated variant order.
There were no solver failures among these 30 runs.

| Mode | 10,752 cells, median [range], s | 98,304 cells, median [range], s |
| --- | ---: | ---: |
| OpenFOAM | 0.574 [0.564, 0.634] | 6.431 [6.227, 6.498] |
| PISO, native | 2.287 [2.274, 2.415] | 30.089 [28.664, 30.997] |
| Coupled Krylov, composite | 4.431 [4.363, 4.436] | 49.630 [47.792, 57.409] |
| NOX Newton, assembled | 13.444 [13.419, 13.825] | 136.304 [132.589, 139.810] |
| NOX Newton, composite | 11.325 [11.240, 11.581] | 103.725 [102.631, 104.150] |

At 98k cells, composite NOX costs **3.45x PISO**, **2.09x coupled Krylov**, and
**16.13x OpenFOAM**. PISO costs 4.68x OpenFOAM. At 10k cells the corresponding
composite-NOX ratios are 4.95x, 2.56x and 19.75x.

Composite reduces Newton runtime by 15.8% at 10k and 23.9% at 98k relative to
the assembled base operator. Both retain scalar blocks, Schur matrices and an
analytic advective Jacobian correction. Neither is a fully face-based,
assembly-free nonlinear operator. No process-memory reduction is inferred
from these timing measurements.

## Numerical comparison

All ten NOX steps converge, with 36 Newton corrections at 10k and 37 at 98k.
There are no rejected line-search trials. Maximum reported scaled residuals
are below `8.7e-11`; maximum integrated all-cell continuity residuals are below
`2.5e-15 m^3/s`. The gauge cell is included in the physical continuity check.

The following are relative velocity-vector L2 differences at `t=0.2 s` on the
exported uniform mid-depth XZ slice; they are not full-volume error norms or
proof that the OpenFOAM solution is exact.

| Comparison | 10,752 cells | 98,304 cells |
| --- | ---: | ---: |
| PISO versus OpenFOAM | 20.84% | 12.61% |
| Coupled Krylov versus OpenFOAM | 8.290% | 4.520% |
| NOX Newton versus OpenFOAM | 8.292% | 4.522% |
| Coupled Krylov versus NOX Newton | 0.0139% | 0.0207% |

Thus the additional nonlinear iterations change the coupled-Krylov result very
little on this window. The existing global velocity acceptance limits are
broader than these spatial differences, so passing them must not be described
as pointwise equivalence to OpenFOAM. Assembled and composite NOX velocity
fields agree within `1.2e-16 m/s` in maximum vector difference on this slice.

## Where NOX spends time

These are medians of sums over ten steps, using the backend's maximum-rank
timers. Timers include nested native work and independently maximized rank
durations; they are not an additive exclusive profile.

| Backend quantity | 10k composite | 98k composite | 98k assembled |
| --- | ---: | ---: | ---: |
| Nonlinear corrections | 36 | 37 | 37 |
| Krylov iterations | 580 | 706 | 706 |
| Reported residual evaluations, including final driver checks | 56 | 57 | 57 |
| Backend total, s | 7.083 | 66.501 | 90.917 |
| Linearization/preconditioner, s | 4.367 | 38.489 | 61.861 |
| Krylov solve, s | 2.354 | 24.766 | 25.643 |
| Residual callbacks, s | 0.202 | 1.722 | 1.822 |

Problem construction, snapshot work, remaining physics and output lie outside
backend total. Their contributions cannot be separated from these timers.
Line-search rejection is not responsible for the runtime gap.

Source inspection identifies two reuse limits:

1. `FluidSolver::solve_coupled_nonlinear()` creates a new private problem each
   physical step. It rebuilds the immutable affine coupled system and its Schur
   products, as well as native geometry/coupled caches. The retained NOX object
   reuses vector/Krylov allocations, but not those native timestep caches.
2. `NativeModel::evalModelImpl()` constructs the next linearization while the
   previous generation remains retained. The native coupled cache protects
   retained matrices/preconditioners from mutation, preventing ordinary
   in-place graph/numeric reuse at that point.

These ownership protections are required for correctness; optimizations must
retain immutable live generations. The baseline has one PISO outer sweep per
step, so there is no existing multi-outer-iteration sequence for NOX to collapse
in this workload.

## Picard, streamed products and IsoRegion diagnostics

These are **single runs per configuration**, separate from the repeated
Newton/PISO/Krylov campaign. All eight runs passed the physical and spatial
checks. The OpenFOAM references in this table are also fresh executions.

| Diagnostic | 10,752 cells, s | 98,304 cells, s |
| --- | ---: | ---: |
| OpenFOAM | 0.607 | 6.325 |
| NOX Picard, native composite/cached | 9.979 | 92.375 |
| NOX Newton, native composite/streamed | 11.071 | 103.760 |
| NOX Newton, two-region IsoRegion composite/streamed | 20.680 | 191.656 |

Picard uses the same nonlinear residual and acceptance gates as Newton. It
still needs 36/37 nonlinear corrections, with 582/708 Krylov iterations versus
Newton's 580/706. Avoiding the analytic advective Jacobian correction reduces
both linearization and solve time in this probe. The final Picard/Newton
velocity-vector differences are at most `3.6e-12 m/s` at 10k and
`4.2e-13 m/s` at 98k; temperature differences are below `1e-9 K`.
The lower timings are promising, but a single sample does not establish a
repeatable speedup. Even this Picard sample costs 14.6x its paired OpenFOAM run
at 98k cells.

Streamed and cached Newton exports are identical. Native and IsoRegion Newton
exports are also identical, with identical nonlinear/Krylov counts. IsoRegion
costs 1.87x native at 10k and 1.85x at 98k in these probes. At 98k, its reported
linearization time is 74.980 s versus native 38.372 s, and residual time is
8.299 s versus 1.683 s. Krylov solve time is similar: 25.239 s versus 24.937 s.
Problem construction and other work outside backend time are also larger.
Thus region overhead remains in the new NOX residual/setup path even though
the earlier region-native transport improvements are present in this build.

Repeated field exports are deterministic for every repeated configuration.
The field metrics and comparisons are retained in `analysis.json`.

## Optimization priorities

1. **Persist immutable geometry and reusable symbolic storage across timesteps.**
   Create fresh coefficient/history snapshots without reconstructing all native
   stencils and coupled caches. Avoid generating unused Schur products solely
   for the immutable affine residual operator. Include map, boundary,
   discretization and geometry-epoch changes in cache invalidation.
2. **Allow safe reuse after a linearization is retired.** Use explicit ownership
   boundaries or a pool of generations so live operators remain immutable while
   retired matrix/AMG storage can be reused. Then qualify adaptive preconditioner
   refresh or lagging with true residual and physical gates. Do not bypass the
   current generation protections or retain stale AMG numerical data.
3. **Tune nonlinear work for setup cost.** Picard deserves repeated evaluation
   on this weakly nonlinear window. Tighter initial inexact forcing or lagged
   Jacobians may trade more Krylov work for fewer expensive rebuilds; these
   alternatives have not been benchmarked here. Larger timesteps require a
   separate time-accuracy study before claiming a benefit.
4. **Use cached face rows or region traversal in the NOX residual.** In
   particular, continuity can reuse the private geometry rows instead of a
   separate public-mesh traversal on every residual evaluation. Extend this
   review to coupled setup on IsoRegion; the unchanged Krylov work shows where
   to focus.

For this startup workload, PISO remains the fastest SimpleFluid option. When
the coupled discretization is wanted, existing coupled Krylov delivers nearly
the same sampled solution as NOX at roughly half the 98k runtime. Composite is
the better measured Newton representation; neither NOX selection nor streamed
products alone removes the performance gap.

## Validation

| Focused check | Result |
| --- | --- |
| NOX enabled, Release, serial | 33 passed; 2 MPI-only skips |
| NOX enabled, Release, two ranks | 34 passed per rank; 1 serial-only skip |
| Enabled logical nonlinear tests exercised across serial/MPI | All 35 passed |
| NOX disabled, Debug, serial | All 17 passed; disabled target rebuilt |
| Existing Boussinesq solver regressions, serial | 35 passed; 3 MPI-only skips |
| Existing turbulent Boussinesq regressions, serial | 9 passed; 1 MPI-only skip |
| Existing performance-runner checks | All 8 passed |
| Shared-library ELF export boundary | Passed |
| Repeated comparisons | All 30 accepted |
| Additional diagnostic comparisons | All 8 accepted |

New tests cover physical residual equality with native assembled equations,
nonuniform density/viscosity, stress and buoyancy sources, analytic derivatives,
immutable trial coefficients, once-per-step heat/material updates, independent
NOX/scalar Krylov selection, and failed-flow state restoration followed by a
successful retry. An initially invalid gas test fixture was corrected; final
enabled and disabled nonlinear suite logs are all green. Initial failed logs
remain in the artifact directory for traceability.

The six pilot executions also passed but are excluded from all repeated
medians. No complete repository-wide suite was run, and the full 20-second
multiphysics transient was not rerun.

## Reproduction and artifacts

Prepared fixture directory: `build/verification/region-operator-20260911/cfd`.
It contains the two input sets, meshed/decomposed OpenFOAM templates and the
reference executable. The new runner validates fresh outputs and excludes
failed configurations from timing summaries.

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
python3 verification/openfoam/run_coupled_nonlinear_performance.py \
  --fixtures build/verification/region-operator-20260911/cfd \
  --build-dir build/performance-nox-20260911 \
  --output build/verification/nox-comparison-new \
  --repeats 3
```

Raw local artifacts are under `build/verification/nox-comparison-20260911`:
`repeated/measurements.json`, `repeated/summary.json`, per-run commands/logs,
rank CSVs, merged outputs, physical comparisons, and provenance. The six pilot
runs are retained separately and excluded from the repeated medians.
`analysis.json` holds detailed field differences and counters; `summary.csv`
holds the unrounded timings. A copy of that small summary is retained beside
this report as [CSV](coupled_nonlinear_20260911.csv).

The additional diagnostics can be reproduced with a fresh output directory:

```sh
python3 verification/openfoam/run_coupled_nonlinear_performance.py \
  --fixtures build/verification/region-operator-20260911/cfd \
  --build-dir build/performance-nox-20260911 \
  --output build/verification/nox-diagnostics-new --repeats 1 \
  --variants openfoam nox-picard-composite nox-composite-streamed \
    iso2-nox-composite-streamed
```
