# Gas subcycles and ALE setup reuse, 2026-09-18

Baseline: `6e597b0193d293f245a7c3e056ae6673f17b62bd`, the clean `develop`
revision matching `origin/develop` after fetching. The candidate is the code in
this change. No numerical model, timestep schedule, solver tolerance, or physical
acceptance gate was changed.

## Implementation and reuse contract

- Gas kinetics computes its unchanged subcycle schedule and exponential decay
  coefficients once per advance. Production increments, the pressure prefactor,
  and the frozen liquid fraction are computed once per cell. Concentration,
  supersaturation, conversion, radius, slip, and mass transfer remain dependent
  on the evolving state and are recomputed inside each subcycle. Arithmetic
  grouping, including microbubble retention as `1 - (1 - exp(...))`, is retained.
- Kinetics, history, snapshots, liquid-mass diagnostics, and ALE material/energy
  loops acquire scoped bulk host views. Owned storage remains authoritative;
  overlap views supply ghosts only. Read/write views retain previous inventories,
  and all views end before synchronization, solves, or field mutation elsewhere.
- Pressure refresh stages current coefficients and collectively checks CRS row,
  column, domain and range maps, column support, and exact numerical equality.
  Compatible matrix structure, RHS storage and Krylov workspace survive ALE
  trials and rollback. Changed values are written through Tpetra's fill-active
  API, with replacement counts checked, and invalidate numerical preconditioners.
  Exactly unchanged coefficients may retain factors under the existing reuse
  option. Incompatible structure requires fresh assembly. Explicit
  `rebuild_matrix()` still forces a new matrix.
- Native pressure-flux owners share one immutable current geometry snapshot,
  with separate field scratch. A changed identity/epoch refreshes metrics;
  an already-current snapshot can be shared without rebuilding it again.
  Legacy meshes still honor unconditional explicit geometry refresh.

This does not cache a collection of historical factorizations. Returning to an
older geometry still refreshes its coefficients; changed numerical values
require new factors. Coupled Schur/MueLu reuse policies are unchanged.

## Focused verification

Incremental `GCC-Release` builds used the existing `build/gcc`, requesting only
these directly affected targets:

```sh
cmake --build --preset GCC-Release --parallel 4 --target \
  coupling_kernel_benchmark testRadiolyticGasModel testRadiolyticGasModelMultiRank \
  testBoussinesqPlanarALE testFluidSolver testPlanarFreeSurfaceModel \
  testPhysicalEquations testPressureProjectionMultiRank \
  testStoredPressureFaceFluxCache testBelosLinearSolver testALEEquations
```

The final focused run passed **159 CTest entries**, including both ELF export
boundary checks. The distributed gas executable also passed four serial cases
and skipped its six MPI-only cases. **71 two-rank cases passed**: gas (10),
pressure (7), new pressure refresh/foreign-map checks (2), shared face geometry
(4), ALE equations (4), Belos invalidation (4), and ALE/coupling replay (40).
The 18 legacy liquid-inventory fixtures passed serially; distributed liquid
coverage comes from the native ALE/coupling tests. An exploratory launch of the
serial legacy fixture under MPI was stopped and is excluded from these totals.

New regressions cover changing heterogeneous gas inputs and subcycle counts,
analytic production/decay and donor inventory, synchronized ghosts, unchanged
and changed pressure coefficients, a numerical change on only one rank,
expansion/rollback/restored geometry, backend/gauge changes, foreign maps,
preconditioner invalidation, and immutable shared-geometry lifetime. Existing
coupling tests exercise consecutive weak-heating intervals, partial/failed
replay, inventory closure, strict acceptance timing, and several coupling modes.

Exact executable filters, test names, outputs and results are retained in
[the validation artifacts](../../build/verification/coupling-kernels-20260918/validate.py)
and [the serial test list](../../build/verification/coupling-kernels-20260918/focused-serial-tests.txt).
No broad suite was run.

## Matched measurement

The new opt-in `coupling_kernel_benchmark` target uses public APIs and does not
add a CTest timing threshold. Both revisions use identical harness source,
compiled against their own headers and libraries. Baseline headers/libraries
were frozen before production edits. Both arms share the unchanged Trilinos
backend; isolated library paths prevent accidental cross-loading.

Configuration: GCC 16.2.1, Release `-O3 -DNDEBUG`, LTO on, NOX on, IF97 off,
Kokkos Serial, Intel i5-12500. Each process uses one OpenMP/BLAS thread.
`mpiexec --bind-to core --map-by core` places one or two ranks on physical cores.
Open MPI uses `pml=ob1`, `btl=self,sm`. Runs are sequential, without our builds or
correctness tests overlapping them. Normal editor activity was present; these
are workstation measurements, not isolated-machine guarantees.

For every fixture/rank count, one baseline/candidate warmup pair is excluded
before five measured pairs with alternating execution order. No measured
repetition was discarded. CSV output and additional verification are outside
the timers. The table reports maximum-rank wall time, median [minimum, maximum],
in seconds. [Raw timings](coupling_kernels_20260918.csv) also retain summed CPU
time and the excluded warmup pair.

| Fixture | Ranks | Baseline seconds | Candidate seconds | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| gas512 | 1 | 1.013 [0.995, 1.043] | 0.940 [0.927, 0.956] | 7.2% |
| gas512 | 2 | 0.589 [0.561, 0.598] | 0.530 [0.523, 0.563] | 9.9% |
| gas4096 | 1 | 1.979 [1.971, 2.017] | 1.830 [1.818, 1.833] | 7.5% |
| gas4096 | 2 | 1.078 [1.051, 1.099] | 0.979 [0.971, 0.985] | 9.1% |
| ale256 | 1 | 2.207 [2.154, 2.262] | 1.241 [1.219, 1.278] | 43.8% |
| ale256 | 2 | 1.711 [1.553, 1.745] | 1.076 [1.064, 1.134] | 37.1% |

Gas fixtures use 512/4096 cells, 40/10 repeated advances, 100 subcycles,
populated micro/large bubbles, and heterogeneous temperature/density. Each
advance restores the same initial state outside the timer. Restoration clears
transport setup in both revisions, so timings include transport assembly and
kinetics, rather than measuring only warmed local kinetics.

The ALE fixture uses a 4×8×8 full-annulus mesh, radii 0.03815–0.25 m, initial
liquid height 0.50852 m, gravity −9.81 m/s², 1 W shaped heating, gas transport,
and donor tracking. Four 0.01 s intervals each have two checkpoint trials.
Both arms use identical canonical endpoints and substep selection with an
initial 0.005 s cap: 18 total solver advances, ending at 0.04 s. The timed region
includes checkpoint creation, restore, energy assignment, substeps and accept.
The default gas subcycle policy is preserved. The reported ALE reduction is
for the combined implementation, not an attribution to pressure reuse alone.

## Numerical equivalence and limits

All **36 paired comparisons**, including warmups, have identical numerical
exports: every owned field value, geometry value, ALE trial history entry and
non-timing summary diagnostic. All field/history files are byte-identical
between revisions at a fixed rank count. Every benchmark run passed its
unchanged time, mass, hydrogen and replay checks; clipping and radius-solver
failure counters are zero.

Serial versus two-rank differences are also exactly the same for both
revisions. In the ALE fixture the largest differences are `7.96e-13 K` in
T, `2.37e-14 m/s` in velocity components, `6.72e-10 Pa` in pressure and
`1.87e-11 kg/m³` in liquid inventory. Final mass, energy and mesh top agree
exactly across rank counts. The largest observed continuity residual is
`3.83e-14`; maximum GCL residual is `2.52e-17 m³/s`. Gas fixture differences
are at rounding scale, with dissolved inventory differing by at most
`2.28e-13 mol/m³`. These rank-count differences already exist in the baseline.

These are solver-only CPU fixtures. No full Hydra-TF campaign, IF97 path, GPU
backend, large-rank scaling campaign or long-duration physical qualification
was performed. Exact-factor reuse helps only when the coefficients really are
unchanged; changing geometry still incurs metric and numerical setup work.

## Reproduction and evidence

Build the benchmark explicitly in each revision, using the same configuration
and this harness source. Give each process group a new output directory and
its matching library directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm \
  LD_LIBRARY_PATH=VARIANT_LIB:BACKEND_LIB \
  mpiexec -n 2 --bind-to core --map-by core VARIANT_BIN/coupling_kernel_benchmark \
  --mode gas --nx 8 --ny 8 --nz 8 --steps 40 --output NEW_DIRECTORY
```

For the larger gas fixture use `--nx 16 --ny 16 --nz 16 --steps 10`. For ALE use
`--mode ale --nx 4 --ny 8 --nz 8 --steps 4 --replays 2`. Repeat with `-n 1`.
A reused output directory is rejected.

The local evidence bundle is
[`build/verification/coupling-kernels-20260918`](../../build/verification/coupling-kernels-20260918).
It retains baseline/candidate binaries, libraries, source/patch hashes,
compiler commands, unchanged-backend hash, per-run commands, full field/history
CSVs, numerical comparison JSON, rank-parity deltas and test logs. The original
execution paths were under `/tmp/simplefluid-perf-6e597b0-mnolqvwu`; copied
runner scripts should be given new output locations before another campaign.
The checked-in CSV preserves all 72 raw timing records.
