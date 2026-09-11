# NOX setup optimization, 2026-09-12

The default NOX path now reuses native timestep storage, skips unused Schur
construction, and recycles retired linearizations safely. A matched three-run
comparison reduces native runtime by 24.4% at 10,752 cells and 22.8% at 98,304
cells, with the same nonlinear and Krylov iteration counts.

## Changes

- `CoupledNonlinearWorkspace` retains two native storage slots through the
  existing problem registry. Each active timestep owns a lease; old callbacks
  keep their history immutable. Compatible slots reuse face geometry, equation
  caches, matrix graphs, physical snapshot storage, and scratch fields.
- `CoupledAssemblyPurpose::ResidualOnly` preserves the full affine residual
  operator, stabilization, RHS and gauge, but creates no Schur products. The
  six historical assembly overloads remain and forward to full solve assembly.
- NOX retires its own Jacobian, preconditioner and dormant Belos references
  before requesting another linearization. Externally retained operators still
  force a fresh native generation. MueLu uses `RP` to refresh numeric data.
- Residual evaluation accumulates continuity and advection in one cached-face
  traversal. Physical gate and commit checks also use those cached rows.
- New per-step diagnostics distinguish native construction, graph reuse,
  Schur construction, preconditioner construction/numeric refresh, and setup
  time. BiCGStab no longer receives the GMRES-only explicit-scaling parameter;
  its independent true-residual gate remains active.

The default preconditioner refresh and forcing settings are unchanged. An
opt-in `PerTimeStep` preconditioner policy refreshes when residual progress
stalls or a matching residual evaluation is unavailable. Its experiment below
does not establish a speedup.

## Matched measurement contract

The baseline is the preserved Release executable and libraries from clean
source `3f5a37a`. Their hashes match the prior campaign, and reconstruction of
that campaign's source patch matches the current starting source. The candidate
is built from the accompanying changes with the same GCC Release/LTO,
Trilinos 17.2, NOX, and IF97 configuration.

Runs use two MPI ranks on physical cores 0 and 1, one BLAS/OpenMP thread, the
same meshes/properties/boundaries, ten steps of `0.02 s`, and unchanged final
nonlinear, continuity and physical comparison gates. Build order alternates
between repetitions. Builds, other tests, and profiling are excluded from the
timing window. OpenFOAM references are executed freshly for each repetition.

The reported time is maximum-rank verification-loop time, including timestep
construction, remaining physics and output. Mesh/solver startup is excluded.
This is the same short startup window as the earlier report, not full-transient
or developed-flow qualification. NOX couples velocity and pressure while
temperature, SST and gas remain sequential. No process-memory saving is claimed.

## Default native NOX: three repetitions

Both builds use analytic Newton with block-composite/cached products and
initial forcing `0.1`.

| Cells | Baseline median [range], s | Candidate median [range], s | Reduction |
| ---: | ---: | ---: | ---: |
| 10,752 | 12.543 [12.389, 12.614] | 9.483 [9.407, 9.733] | 24.4% |
| 98,304 | 117.373 [116.565, 122.248] | 90.595 [89.963, 91.241] | 22.8% |

All 18 runs in this campaign, including the fresh OpenFOAM references, passed.
Both builds use 36/37 nonlinear corrections and 580/706 Krylov iterations at
10k/98k cells, respectively, with no rejected line searches.

At 98k, median linearization/preconditioning time decreases from 43.015 to
32.933 s; residual time decreases from 2.004 to 0.991 s. Krylov time is nearly
unchanged, 30.325 versus 30.281 s. These independently maximized timers include
nested work and must not be added as an exclusive breakdown.

The candidate constructs two native workspaces and reuses them eight times.
Two slots are intentional: NOX still owns the previous callbacks while the
next immutable timestep is prepared. Four cold coupled operators serve the
affine and dynamic paths in the two slots; later assembly reuses their graphs.
At 98k, there are 43 graph reuses, 37 Schur builds, and two preconditioner builds
followed by 35 numeric refreshes. The ten affine residual constructions make
no Schur products. Baseline native counters were not recorded.

## Numerical agreement

Repeated exported fields, history and turbulence files are byte-identical
within each build/grid configuration. Baseline/candidate differences on the
final exported velocity slice are at most `1.00e-15 m/s` at 10k and
`9.14e-16 m/s` at 98k; relative vector L2 differences are `3.36e-12` and
`8.40e-12`. Maximum temperature differences are below `3.6e-12 K`. Hydrogen
production, escaped and total inventories agree exactly in the checked exports.
Near-zero diagnostic residuals are assessed by absolute differences rather
than misleading large relative ratios.

## Focused 10k tuning, three repetitions

These are separate candidate-only experiments using the same final acceptance
gates. Their default comparison is the candidate result above.

| Setting | Median [range], s | Nonlinear corrections | Krylov iterations |
| --- | ---: | ---: | ---: |
| Default Newton, forcing `0.1` | 9.483 [9.407, 9.733] | 36 | 580 |
| Newton, forcing `0.01` | 9.249 [9.214, 9.363] | 29 | 689 |
| Newton, forcing `0.001` | 7.508 [7.488, 7.587] | 20 | 517 |
| Picard, forcing `0.1` | 7.826 [7.752, 7.874] | 36 | 582 |
| Newton, preconditioner per timestep | 9.504 [9.498, 9.542] | 36 | 580 |

Forcing `0.01` trades fewer corrections for more Krylov work and gives little
net benefit. Forcing `0.001` lowers both on this weakly nonlinear startup case.
Its velocity difference from default Newton is below `1.7e-12 m/s`, with a
maximum temperature difference of `5.8e-9 K`; all acceptance checks pass.

Preconditioner lagging lowers Schur builds from 36 to ten but increases cold
preconditioner builds from two to ten and coupled operator builds from four to
14. Keeping P immutable pins an older matrix generation, limiting reuse of
the current native cache. The reduction in Schur work does not produce a
repeatable improvement here. Keep the default iteration refresh policy.

### Matched 98k forcing confirmation

A separate three-repeat campaign compares the **same optimized executable**
with initial forcing `0.1` and `0.001`, alternating their execution order.
Default forcing takes 91.318 [89.820, 98.658] s; `0.001` takes
76.013 [72.697, 77.808] s, a 16.8% median reduction. Corrections decrease from
37 to 20 and Krylov iterations from 706 to 615. All nine executions, including
three fresh OpenFOAM references, pass. The wider timing ranges in this later
campaign are retained rather than mixed into the code-only comparison.
The tuned/default final velocity difference is at most `1.44e-12 m/s`
(`1.36e-8` relative vector L2); temperature differs by at most `4.90e-9 K`
and gas volume fraction by `2.25e-15`. Hydrogen production and escape totals
are unchanged. These are stopping-path differences, not weakened tolerances.

For this startup workload, `--nonlinear-forcing-initial 0.001` is the best
tested setting. It changes the initial correction tolerance, not the final
nonlinear or physical gates. The library default remains `0.1`: these short
windows do not establish the best forcing for developed or strongly nonlinear
flow. Picard's larger-grid benefit was not retested in this optimization run.

## IsoRegion

At 10,752 cells, matched three-run composite/streamed medians decrease from
22.506 [22.401, 22.905] s to 14.793 [14.721, 14.838] s, a 34.3% reduction.
Residual time falls from 0.944 to 0.116 s, close to the native path. Linearization
still costs 7.497 s versus 3.749 s native, leaving coupled assembly as a further
region-specific target. Native and IsoRegion exports are byte-identical for
each build at this size.

At 98,304 cells, a single matched IsoRegion diagnostic decreases from
205.726 to 139.417 s, a 32.2% reduction. Residual time falls from 8.428 to
1.059 s; linearization falls from 79.107 to 67.626 s. Krylov time is unchanged
within this sample, 30.866 versus 30.854 s, and both builds use 37 corrections
and 706 iterations. This is one qualified pair, not a repeated median.

## Validation and artifacts

- Enabled nonlinear checks: 41 serial passes with two MPI-only skips; 42
  passes per rank with one serial-only skip. All 43 logical tests are exercised.
- Direct coupled assembly: 21 passes serially and on two ranks, including 16
  new residual-only/purpose-transition tests. Distributed target fixtures use
  owned-cell order and validate the intended compatibility failure.
- Generation retirement: both tests pass serially and on two ranks, including
  GMRES/BiCGStab, backtracking, and externally retained J/P cases.
- NOX-disabled nonlinear checks: all 22 pass.
- Release shared-library export check passes, independently anchoring old and
  new assembly overloads. No unrelated project-wide suite was run.
- All 14 runner checks pass (eight existing checks and six matched-runner
  checks). Every final timed execution passes its solver/statistics, physical
  comparison and spatial-output checks: 38 SimpleFluid and 25 OpenFOAM runs.

Local artifacts are under `build/verification/nox-optimization-20260912`:
source-equivalence proof, baseline source/binaries, per-campaign provenance,
commands, rank/merged CSVs, comparison reports, `analysis.json`, `summary.csv`,
and focused validation logs. Historical September 11 timings are context only;
the reductions above use the new matched baseline runs.

Reproduce the code-only comparison with a new output directory:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
python3 verification/openfoam/run_coupled_nonlinear_performance.py \
  --fixtures build/verification/region-operator-20260911/cfd \
  --build-dir build/performance-nox-20260911 \
  --baseline-build-dir build/verification/nox-optimization-20260912/baseline-build \
  --output build/verification/nox-optimization-rerun \
  --cells 10752 98304 --variants openfoam nox-composite --repeats 3
```

The runner's `--nox-forcing-initial` and `--nox-preconditioner-update` overrides
apply only to the candidate. For the matched forcing experiment, both build
arguments name the optimized build and the candidate receives
`--nox-forcing-initial 0.001`. Per-campaign scripts in the artifact directory
retain the exact executed commands.
