# OpenFOAM pitzDaily comparison

This fixture is derived from
`$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily` in OpenFOAM v2606.
It retains the tutorial's five-block backward-facing-step geometry, 10 m/s
inlet, `1e-5 m2/s` kinematic viscosity, standard k-epsilon closure, inlet
`k=0.375 m2/s2`, and inlet `epsilon=14.855 m2/s3`. The checked-in OpenFOAM
case adds vertical velocity samples at `x=0.05`, `0.10`, and `0.20 m`.

The SimpleFluid executable uses the same duct outline, boundary names, five
connected blocks, and OpenFOAM `simpleGrading`/`edgeGrading` distributions.
This includes wall-normal clustering through the upper duct and a graded shear
layer downstream of the step. The mesh is coarsened from the tutorial by a
factor of four in each in-plane direction by default. Set
`SIMPLEFLUID_PITZ_MESH_DIVISOR=1` for the tutorial cell counts and grading.

## Run

`run_comparison.sh` is the acceptance launcher. It first requires a qualified
physical tolerance manifest and then runs both cases on up to six ranks:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
verification/openfoam/pitzDaily/run_comparison.sh 6
```

The checked-in `reference/physical_acceptance.json` is intentionally marked
`pending_external_qualification`, so this command currently stops before
launching either solver. No matched, converged profile pair with sufficient
provenance was available to justify numerical CFD tolerances. This fail-closed
behavior prevents diagnostic differences from becoming accidental acceptance
limits. After qualification, the launcher will enforce every station/component
limit in that manifest. `SIMPLEFLUID_PITZ_ACCEPTANCE_FILE` may select another
manifest, but it must be qualified with scope `physical_reference`.

A qualified manifest must authenticate its declared retained source files with
SHA-256 digests. For physical scope it must also record qualified OpenFOAM and
SimpleFluid provenance plus exact positive `simplefluid_run` values for
`mesh_divisor`, `steps`, `dt_s`, and `mpi_ranks`. The launcher asks the
comparator to validate and print that contract before either solver starts,
rejects a different requested rank count, and exports the recorded mesh,
step-count, and timestep settings to the SimpleFluid run.

Run either side independently with:

```sh
verification/openfoam/pitzDaily/openfoam/Allrun 6
verification/openfoam/pitzDaily/run_simplefluid.sh 6
```

The SimpleFluid launcher sources `verification/environments.sh` and builds the
`GCC-RelWithDebInfo` preset by default, using
`build/gcc/bin/RelWithDebInfo`. Set `SIMPLEFLUID_COMPILER=LLVM` to use the
corresponding LLVM preset and `build/llvm/bin/RelWithDebInfo`, or set
`SIMPLEFLUID_BUILD_CONFIG=Debug` for a debug run. A custom, already-configured
build tree can be selected with `SIMPLEFLUID_BUILD_DIR`. Rank CSV and VTU
files are written under `profiles/`.
Runtime and resolution are configurable:

```sh
SIMPLEFLUID_PITZ_MESH_DIVISOR=2 \
SIMPLEFLUID_PITZ_STEPS=1000 \
SIMPLEFLUID_PITZ_DT=1e-5 \
  verification/openfoam/pitzDaily/run_simplefluid.sh 6
```

Set `SIMPLEFLUID_PITZ_STEADY_STATE=1` to run the adaptive pseudo-transient
steady-state search. During that search, momentum and turbulence transport
start with a relaxed `1e-6` linear tolerance and tighten monotonically to the
final `1e-9` tolerance as the physical update rate falls. The pressure
projection retains its independent strict tolerance throughout. Override the
transport schedule with:

```sh
SIMPLEFLUID_PITZ_STEADY_RELAXED_LINEAR_TOLERANCE=1e-5 \
SIMPLEFLUID_PITZ_LINEAR_TOLERANCE=1e-9 \
SIMPLEFLUID_PITZ_STEADY_FULL_ACCURACY_UPDATE_RATIO=10 \
SIMPLEFLUID_PITZ_STEADY_STATE=1 \
  verification/openfoam/pitzDaily/run_simplefluid.sh 6
```

The controller never loosens a tolerance after an accepted step, preserves it
across rejected attempts, and requires final-tolerance steps before the
steady-state acceptance window can advance. If the configured final tolerance
is looser than `1e-6`, the default relaxed tolerance is raised to match it.
Progress lines report the requested current and next transport tolerances
separately from the aggregate achieved `linear_tolerance`.

The comparison script selects the SimpleFluid cell layer nearest each OpenFOAM
sampling station, interpolates the OpenFOAM profile to the SimpleFluid cell
centres, and reports RMS and maximum errors for `ux` and `uy`. Optional
`--max-l2` and `--max-linf` arguments turn those metrics into pass/fail checks.
For an acceptance run, `--tolerances` loads station-specific `ux`/`uy` RMS and
maximum-error limits plus the maximum permitted station offset, minimum sample
count, and minimum covered fraction of the reference profile span. Inputs and
computed norms must be finite, and result samples cannot silently fall outside
the reference range. A manifest is accepted only when its qualification
status, scope, and authenticated sources satisfy the schema.

Until the physical manifest is qualified, run the two sides independently and
produce diagnostic metrics without acceptance limits:

```sh
verification/openfoam/pitzDaily/openfoam/Allrun 6
verification/openfoam/pitzDaily/run_simplefluid.sh 6
python3 verification/openfoam/pitzDaily/compare_profiles.py \
  --openfoam-case verification/openfoam/pitzDaily/openfoam \
  --simplefluid-glob \
    'verification/openfoam/pitzDaily/profiles/simplefluid_cells_rank*.csv'
```

## Deterministic comparator gate

`fixtures/` contains a checked-in synthetic OpenFOAM/SimpleFluid profile pair,
SHA-256 provenance, analytically calculated errors, and distinct tolerances for
both velocity components at all three stations. Its limits are rounded upward
with approximately ten percent headroom. The fixture qualifies only the
comparison machinery; it is explicitly not a physical pitzDaily reference.

Run the test directly with:

```sh
python3 verification/openfoam/pitzDaily/test_compare_profiles.py
```

When Python is available during CMake configuration, the same test is
registered as `pitz_daily_profile_comparator_fixture` with the `verification`
and `integration` labels. It checks recorded norms, authenticated hashes and
tamper rejection, a passing gate, deliberately tightened and incomplete-data
failures, non-finite input/metric rejection, physical-run-setting extraction,
CLI behavior, and the fail-closed pending physical manifest.

## Physical-reference qualification contract

Before changing `reference/physical_acceptance.json` to `qualified`, retain all
of the following evidence in a checked-in reference directory:

1. OpenFOAM.com v2606 source/case provenance, MPI rank count, solver log, and
   residual-control convergence for the three final raw profiles.
2. The exact SimpleFluid revision, compiler/build configuration, MPI rank
   count, `SIMPLEFLUID_PITZ_*` settings, and rank CSVs. Record the executable
   `mesh_divisor`, `steps`, `dt_s`, and `mpi_ranks` values under
   `reference_definition.simplefluid_run`; use the tutorial cell counts
   (`SIMPLEFLUID_PITZ_MESH_DIVISOR=1`).
3. Profile stationarity at the selected final SimpleFluid time, a time-step
   refinement comparison at that time, and the divisor-2 to divisor-1 mesh
   trend. The current 200-step, `1e-5 s` default is only a smoke/comparison
   setting and is not convergence evidence.
4. SHA-256 checksums for every retained raw profile and CSV under
   `qualification.source_files_sha256`; the comparator verifies these before
   accepting the manifest.
5. Observed `ux` and `uy` RMS/maximum errors at `x=0.05`, `0.10`, and `0.20 m`,
   along with per-metric headroom justified from the refinement/repeatability
   envelope rather than from one run.

Only then populate all six component entries and the station offset,
minimum-sample, and minimum-reference-span limits in the physical manifest.
This separates reproducibility and fail-closed authentication of the
comparator, which are automated now, from external CFD qualification, which
remains open.

## Interpretation

OpenFOAM solves a steady SIMPLE RANS problem and applies `kqRWallFunction`,
`epsilonWallFunction`, and `nutkWallFunction` at the walls. SimpleFluid advances
the same standard k-epsilon coefficients transiently with SIMPLE coupling and
selects its coordinated `standardHighReKEpsilon` treatment on `upperWall` and
`lowerWall`. That treatment supplies zero-gradient `k`, OpenFOAM.com v2606
stepwise `nutkWallFunction` face viscosity, and the `epsilonWallFunction`
adjacent-cell epsilon and production constraints with the v2606 default
`lowReCorrection false`. The front/back `empty` planes are
represented as slip planes in the one-cell-thick SimpleFluid mesh.

The wall model and block grading are now aligned, but the cases still differ in
mesh implementation, time integration versus a steady solve,
pressure-velocity iteration, and discretization details. This remains an
end-to-end comparison rather than a claim of identical numerical solutions.
OpenFOAM pressure is kinematic;
SimpleFluid uses a reference density of `1 kg/m3`, making pressure numerically
comparable, while the automated profile comparison intentionally evaluates
velocity only.
