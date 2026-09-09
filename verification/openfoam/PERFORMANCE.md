# Repeating the verification performance measurements

`run_performance.py` compares two existing SimpleFluid builds using the same
complete physical fixtures and solver policies. Select the cases explicitly;
the script does not build either executable or prepare OpenFOAM. It retains
commands, binary hashes, shared input snapshots, MPI bindings, per-rank timing,
CSV outputs, and comparisons in a new output directory.

The default meshes have since been enlarged tenfold in each dimension while
retaining cell widths; see [current mesh sizes](BOUNDARY_LAYER_MESHES.md).
The following policy was measured on the original scale-1 fixtures and has
not been requalified for the enlarged defaults. The runner snapshots current
inputs; rerunning it now measures the larger problems and can take much longer.

The opt-in `recommended` policy uses these historical fixture-specific settings:

| Fixture | Default ranks | Solver override |
| --- | ---: | --- |
| 1,008-cell bottom-heated convection | 2 | Pressure PCG/DIC; transport BiCGStab/SGS |
| 44-cell steady/transient bubbles | 1 | Transport BiCGStab/SGS |
| 12-cell steady/transient ALE | 1 | Pressure PCG/DIC |

These fixture-specific choices follow the September 9, 2026 reprofile at
`e6a492c`. Two-rank SGS convection used less CPU time than four ranks; four
ranks saved about 8% wall time while consuming 84% more CPU time. Pressure
PCG/DIC reduced these small ALE loop times by about fourfold. The steady
bubble SGS gain was modest, and the transient bubble timing ranges overlapped.
Use explicit rank overrides to evaluate latency on a different machine.

Prepare baseline and current IF97-enabled builds with the same compiler,
configuration, dependencies, and build options. Keep both binary directories
available and do not rebuild them during measurement. For example, after
building the corresponding targets in two separate checkouts:

```sh
python3 verification/openfoam/run_performance.py \
  --baseline-bin-dir /path/to/baseline/build/gcc/bin/Release \
  --current-bin-dir build/gcc/bin/Release \
  --output build/verification/performance-convection \
  --cases convection

python3 verification/openfoam/run_performance.py \
  --baseline-bin-dir /path/to/baseline/build/gcc/bin/Release \
  --current-bin-dir build/gcc/bin/Release \
  --output build/verification/performance-small-fixtures \
  --cases bubble-steady bubble-transient ale-steady ale-transient
```

When using copied executable snapshots, also preserve their shared libraries
and pass `--baseline-lib-dir /path/to/baseline/lib` and, when needed,
`--current-lib-dir /path/to/current/lib`. Each is prepended to that build's
`LD_LIBRARY_PATH`; hashes of the supplied shared libraries are retained and
checked before and after every run. Copying an executable alone can silently
load libraries from its original build tree after they have been rebuilt; the two runs must
use their respective libraries as well as their respective executables.

`--ranks 1 2 4` explicitly overrides the recommendation for every selected
fixture; a failure still stops the schedule and preserves its logs.
`--policy default` selects the executables' normal algorithms for both builds.
`--policy pressure-dic` keeps the former pressure-only experiment (normal gas
transport); `--policy dic-sgs` explicitly selects the recommended combination.
Rank counts and solver policies are independent: selecting `default` does not
change the chosen ranks. Three repeats run by default, sequentially, with the
baseline/current order alternating. Use `--repeats 1` for a focused check.
There is no shortened-step option: each accepted run must cover every physical
time in its unchanged manifest.

The launcher uses OpenMPI's `--timeout`, binds ranks to physical cores, and
sets OpenMP/OpenBLAS/MKL thread counts to one. `--mpiexec` and `--mpi-args`
select the installed OpenMPI launcher and its binding arguments. Record
equivalent dynamic-library environments for both builds, use idle cores, and
run profiling separately from timing. Binary hashes are checked before and
after each solve; the recorded environment and shared inputs help identify
changed dependencies, meshes, and policies when repeating the experiment.
Mesh/topology/geometry cache invalidation needs separate focused regression
tests; performance on these fixed inputs does not establish those contracts.

`summary.json` reports median/min/max verification-loop wall time (the slowest
rank), summed process CPU time, full launcher wall time, and median speedup.
Loop timing includes each driver's diagnostics and loop output, and excludes
mesh/solver setup and merging. Full launcher timing also includes startup and
shutdown; artifact hashing is outside both timing intervals. Every run's
`measurement.json` and the append-only
`measurements.jsonl` remain available when a later run or comparison fails.
Only configurations whose complete repeat set passes appear in the summary.

For each pair the script merges disjoint owned fields, verifies that global
histories agree across ranks, and applies the existing physical manifest's
quantities and independent conservation gates to both builds. The comparator's
historical `openfoam`/`simplefluid` report slots are explicitly labelled
**SimpleFluid baseline/current** in `comparison.json`; this workflow makes no
new OpenFOAM agreement claim. Complete field/time coverage, geometry, finite
values, and material/void bounds are also checked using the shared field
reader. Per-column maximum field changes are reported for review; those
changes do not have an additional same-engine tolerance gate.

To establish cross-engine agreement, use each case's `run_comparison.sh` and
its unchanged paired manifest. The independent OpenFOAM reference has a
smaller equation scope for the bubble and ALE fixtures. Future algebraic
audits should retain per-field absolute residuals, normalization denominators,
normalized residuals, iteration counts and convergence status for both engines
at the same accepted physical steps. A temperature normalization floor cannot
be inferred from normalized residuals alone. Match the full residual definition
and acceptance target before interpreting a cross-engine ratio as equal work.
