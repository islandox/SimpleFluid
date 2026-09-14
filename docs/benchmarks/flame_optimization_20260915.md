# Flame-profile optimization, 2026-09-15

This change implements the follow-up recommendations from the
[RelWithDebInfo flame profiles](flame_profiles_20260915.md). Measurements and
reproduction artifacts are retained under
`build/verification/flame-optimization-20260915`.

With unchanged solver settings and byte-identical physical outputs, matched
Release medians improve by 22.3% for NOX at 10,752 cells and 19.8% at 98,304
cells. Corresponding PISO reductions are 10.0% and 4.5%. Separate tuning
campaigns measure initial forcing and pressure preconditioning below.

[Open the interactive before/after gallery](../../build/verification/flame-optimization-20260915/profiles/index.html)
or [download the chart bundle](../../build/verification/flame-optimization-20260915/flame-charts.zip).

## Implementation

- NOX scaling and convergence loops acquire host views and extents once.
  Component IDs are derived once from ordered global IDs and retained with the
  compatible vector cache, including signed and noncontiguous maps.
- `MeshHandle::visit_cell_faces()` dispatches once per cell traversal, retaining
  native face order and translating each visible face once. Gradient geometry
  gathers local incidences and metrics once into operation-local scratch before
  forming its coefficients. Native, reordered, ghost, and periodic mappings
  retain their established semantics.
- Coupled assembly lazily caches owned-cell face incidences and refreshes them
  on geometry-epoch changes. Unused coupled solvers do not allocate this cache.
- Pressure-flux workspaces and analytic Jacobians share immutable mesh gradient
  and face geometry. Each keeps private pressure-gradient and directional
  scratch. Refresh replaces the snapshot; retained generations cannot observe
  new metrics through an old reference. Boundary conditions remain inputs to
  each evaluation.
- Gas reconstruction, transport preparation, and momentum/stress kernels use
  scoped bulk field views. Owned face rows are mapped explicitly and views end
  before synchronization. Kinetics, correlation formulas, transport ordering,
  and strict-zero decisions retain their existing arithmetic and rules.

The performance runner also accepts candidate-only
`--piso-pressure-preconditioner muelu` for a matched pressure-preconditioner
probe. This retains PCG and its residual gates. DIC ordering and the default
solver policies are unchanged.

## Measurement contract

The baseline is source `69a7572`, with independent preserved Release and
RelWithDebInfo executables and libraries. The candidate uses the same configured
build tree, `build/performance-strict-gas-20260914`, with NOX, IF97, and LTO
enabled. Source snapshots and binary hashes accompany each campaign.

Release timings use two MPI ranks bound to physical cores 0 and 1, one
BLAS/OpenMP thread, and ten steps of 0.02 s on 10,752- and 98,304-cell fixtures.
Baseline/candidate execution order alternates over three repeats. Strict gas
transport retains 20 active microbubble solves and 30 skipped inactive
auxiliary stages per window. Fresh OpenFOAM references use the same fixtures
and existing physical gates. Builds and profilers do not overlap timed runs.

RelWithDebInfo profiles are a separate measurement: whole-process user CPU
sampling, two ranks, 199 Hz, DWARF stack unwinding and frame pointers. Profile
weights include MPI CPU polling but exclude off-CPU waits. They are not
elapsed communication times or Release speedup estimates.

## Matched Release results

All 30 runs qualified. Values below are median timed-loop seconds over three
repeats; brackets show the observed minimum and maximum. Reductions compare
medians from this campaign, not the previous day's timings.

| Cells | Solver | Baseline seconds [range] | Candidate seconds [range] | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 10,752 | PISO | 2.333 [2.329, 2.458] | 2.101 [2.084, 2.226] | 9.95% |
| 10,752 | NOX | 9.642 [9.415, 10.073] | 7.493 [7.341, 7.544] | 22.29% |
| 98,304 | PISO | 33.574 [33.372, 33.973] | 32.064 [31.314, 32.084] | 4.50% |
| 98,304 | NOX | 89.947 [89.554, 90.434] | 72.171 [72.170, 72.590] | 19.76% |

Fresh OpenFOAM reference medians were 0.653 s at 10,752 cells and 7.523 s at
98,304 cells. The optimized NOX path remains slower than PISO and the reference
in these short windows.

All 12 baseline/candidate pairs produced byte-identical `fields.csv`,
`history.csv`, `turbulence.csv`, and `wall_resolution.csv`, together with
identical linear statistics and every non-timing gas/NOX statistic. Executable
and library hashes remained unchanged. At 98,304 cells, both NOX builds used
37 nonlinear iterations, 706 Krylov iterations, 57 residual evaluations, and
no line-search rejections. Maximum continuity imbalance was
`2.7310411102488147e-16 m^3/s` in both builds.

Median phase diagnostics at 98,304 cells:

| Diagnostic | Baseline seconds | Candidate seconds |
| --- | ---: | ---: |
| NOX native setup | 9.068 | 6.783 |
| NOX backend total | 66.849 | 52.875 |
| NOX linearization | 33.246 | 23.042 |
| NOX linear solve | 30.954 | 27.765 |
| NOX residual callbacks | 1.018 | 1.034 |
| NOX gas update | 3.132 | 1.621 |
| PISO gas update | 3.115 | 1.620 |

The largest NOX phase reduction is linearization, about 30.7%. Complete gas
update time falls about 48% in both modes, while gas transport matrix assembly
and linear-solve times remain approximately unchanged. PISO's whole-loop
reduction is correspondingly much smaller than its gas-update reduction.
These diagnostics include nested work and independently maximized rank times;
their medians must not be added as an exclusive phase decomposition.

See the machine-readable
[analysis](../../build/verification/flame-optimization-20260915/matched-release/analysis.json)
and [summary CSV](../../build/verification/flame-optimization-20260915/matched-release/summary.csv)
for every measurement and equivalence result.

## Initial forcing tolerance

A separate 98,304-cell campaign compares the same optimized executable with
the default initial forcing tolerance `0.1` and opt-in `0.001`. This tightens
the initial linear correction target; final nonlinear, continuity, and
physical acceptance gates remain unchanged. All nine runs, including three
fresh OpenFOAM references, qualified and retained unchanged binary hashes.

| Initial forcing tolerance | Median seconds [range] | Nonlinear solves | Krylov iterations | Residual evaluations |
| --- | ---: | ---: | ---: | ---: |
| `0.1` | 74.830 [72.929, 75.581] | 37 | 706 | 57 |
| `0.001` | 61.648 [58.934, 64.781] | 20 | 615 | 40 |

The 17.62% median reduction is an additional tuning result measured within
this campaign. Linearization falls from 24.406 to 13.542 s, while linear
solves fall from 28.148 to 26.227 s. Both options have zero line-search
rejections. The tighter initial solve reduces outer work in this fixture;
it does not establish a better default for other problems.

Sampled fields differ by at most `4.893e-9 K` in temperature,
`1.439e-12 m/s` in a velocity component, and `2.242e-15` in gas fraction.
Hydrogen inventory, production, escape, and balance history columns are
identical. Other physical history columns have small differences. Maximum
dimensional continuity decreases from `2.731e-16` to `2.553e-17 m^3/s`, and
maximum scaled nonlinear residual from `8.625e-11` to `3.170e-12`.

The exported maximum *last-correction* true-residual ratio changes from
`0.002533` to `2.452e-5`. This export is not a maximum over every inner solve;
the linear adapter still checks each actual correction against its adaptive
forcing target. See the detailed
[analysis](../../build/verification/flame-optimization-20260915/forcing-release/analysis.json)
and [physical deviations](../../build/verification/flame-optimization-20260915/forcing-release/physical-deviations.json).
The option remains `--nox-forcing-initial 0.001` in the performance runner,
or `--nonlinear-forcing-initial 0.001` in the verification executable.

## Pressure preconditioner

A separate 98,304-cell campaign compares PCG/DIC with PCG/MueLu using the same
optimized executable and unchanged residual gates. All nine runs qualified;
a preceding three-run pilot also qualified but is excluded from these medians.

| Pressure policy | Median seconds [range] | Aggregate flow iterations | Maximum flow true-relative residual |
| --- | ---: | ---: | ---: |
| PCG/DIC | 30.076 [29.417, 30.371] | 8,301 | `8.956e-12` |
| PCG/MueLu | 18.051 [17.890, 18.109] | 1,309 | `8.628e-12` |

MueLu reduces timed-loop median by 39.98% in this window. Both policies retain
70 aggregate flow solves and 20 gas solves; gas iterations remain 36.
Sampled temperature changes by at most `2.274e-13 K` and velocity by at most
`4.937e-17 m/s`. Physical exports and linear statistics are identical across
the three repeats within each policy, and binary hashes remain unchanged.

Hierarchy reuse is verified from the source: `FluidSolver` enables pressure
preconditioner reuse, the verification option preserves that setting, the
pressure equation retains its fixed-mesh Poisson matrix, and `BelosLinearSolver`
retains the preconditioner for the same matrix pointer and preconditioner kind.
No runtime hierarchy-reuse counter is exported. DIC's distributed factor
ordering, staged halo exchanges, and validation are unchanged.

The baseline flame stacks separate DIC halo exchange (10.00% of all PISO user
CPU samples) from Krylov reductions (4.00%) and DIC collective validation
(0.286%). This attributes sampled CPU work, including MPI polling, rather
than measuring off-CPU communication latency.

Use `--piso-pressure-preconditioner muelu` in the performance runner or
`--pressure-preconditioner muelu` in the verification executable to select
this policy. The performance runner keeps its DIC baseline default; the native
`FluidSolver` already defaults to MueLu pressure preconditioning. See the
[analysis](../../build/verification/flame-optimization-20260915/pressure-release/analysis.json)
and [physical deviations](../../build/verification/flame-optimization-20260915/pressure-release/physical-deviations.json).

## Updated native profiles

Both optimized RelWithDebInfo captures passed the physical and strict-gas
checks and produced byte-identical outputs to their fresh native controls.
Neither capture reported lost buffers. PISO retained 12,970 samples and NOX
30,262 samples across the two ranks. The comparison uses the original
baseline profiles with the same configuration and controls; it is separate
from the matched Release wall-time campaign. No new Callgrind or memory
measurements were collected.

Exclusive self-cost shares, weighted by perf CPU periods:

| Function family | PISO baseline | PISO candidate | NOX baseline | NOX candidate |
| --- | ---: | ---: | ---: | ---: |
| Field and Tpetra map/view access | 10.03% | 7.42% | 16.17% | 7.36% |
| Mesh geometry, indexing and dispatch | 18.99% | 18.70% | 20.05% | 17.48% |

NOX leaf sample counts for the targeted functions:

| Function | Baseline | Candidate |
| --- | ---: | ---: |
| `Tpetra::MultiVector::getLocalLength` | 2,136 | 247 |
| `Tpetra::MultiVector::getData` | 1,074 | 385 |
| `Tpetra::Map::getLocalElement` | 1,725 | 713 |
| `MeshBase::cell_id` | 1,345 | 766 |
| `MeshBase::face_id` | 1,295 | 870 |

Inclusive NOX linearization falls from 35.54% to 32.23%, or 68.33 to 49.01
weighted CPU seconds across both ranks. Analytic operator application rises
in relative share, from 14.66% to 18.39%, while its absolute sampled weight
is nearly unchanged at 28.19 versus 27.97 CPU seconds. This profile does not
establish a speedup in analytic application itself. Named gas-advance contexts
fall from 8.38% to 4.94% for PISO and 3.38% to 2.27% for NOX.

With the DIC policy retained in both PISO captures, halo exchange remains
10.00% versus 10.37% of total sampled CPU cost and Krylov reductions 4.00%
versus 4.50%. These increasing relative shares reflect the smaller surrounding
workload; they are not evidence that communication latency increased. Function
families are exclusive, while the named phase contexts are inclusive and
overlap those families. Full weighted attribution is in
[profile-attribution.json](../../build/verification/flame-optimization-20260915/profile-attribution.json).

Root-stack coverage is 98.57% for PISO and 98.93% for NOX. Unresolved leaf
cost is 19.58% and 10.22%, respectively, confined to prebuilt system/MPI
libraries; all SimpleFluid leaf symbols resolve. No perf-script warnings were
reported. The [quality record](../../build/verification/flame-optimization-20260915/profiles/quality.json)
and [profile provenance](../../build/verification/flame-optimization-20260915/profiles/provenance.json)
retain collection commands, loss checks, coverage, and output-equivalence
evidence. Charts preserve period weights and label unresolved frames.

## Correctness checks

The focused Debug serial checks passed 153 tests, with six MPI-only skips,
across mesh traversal/reordering, gradient and geometry caches, stored
operators, pressure flux, coupled assembly, NOX ownership, and radiolytic gas.
All six affected registered two-rank CTest entries passed. The NOX map-cache
tests also passed on two ranks, as did the exact periodic/ghost traversal
fixture. The independent NOX-disabled configuration passed all 23 nonlinear
tests. All 16 performance-runner unit tests passed. Both final Release and
RelWithDebInfo libraries passed the ELF export-boundary check. Validation logs
are retained in the artifact directory's `correctness/` subdirectory.

New checks cover shared geometry with private scratch, boundary changes,
creator destruction, stale epochs and rollback, retained Jacobians, ordered
component-map reuse/invalidation, and cached assembly incidences. The native
boundary-distance conversion also preserves non-default scalar rounding.

These checks and the short benchmark windows do not qualify a full transient,
an MPI scaling matrix, or other mesh/physics configurations.
