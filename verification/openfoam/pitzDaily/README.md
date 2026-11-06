# OpenFOAM pitzDaily comparison

This fixture is derived from
`$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily` in OpenFOAM v2606.
It retains the tutorial's five-block backward-facing-step geometry, 10 m/s
inlet, `1e-5 m2/s` kinematic viscosity, standard k-epsilon closure, inlet
`k=0.375 m2/s2`, and inlet `epsilon=14.855 m2/s3`. The checked-in OpenFOAM
case adds vertical velocity samples at `x=0.05`, `0.10`, and `0.20 m`.

Generated study inputs, outputs, captured source/configuration snapshots, and
intermediate reviews are archived outside the implementation history. Restore a
reviewed bundle locally when needed; `reference/qualification_*/` is ignored.
The maintained acceptance manifest is an unqualified template with no measured
limits or historical dataset bindings. Maintained case inputs and deterministic
test fixtures remain in the repository.

The SimpleFluid executable uses the same duct outline, boundary names, five
connected blocks, and OpenFOAM `simpleGrading`/`edgeGrading` distributions.
This includes wall-normal clustering through the upper duct and a graded shear
layer downstream of the step. The mesh is coarsened from the tutorial by a
factor of four in each in-plane direction by default. Set
`SIMPLEFLUID_PITZ_MESH_DIVISOR=1` for the tutorial cell counts and grading.
`SIMPLEFLUID_PITZ_MESH_MULTIPLIER=2` doubles the in-plane tutorial counts;
supported multipliers are 1 and 2. Multi-grading section rounding matches
OpenFOAM v2606 on the coarse meshes as well as the tutorial grid.

## Run

`run_comparison.sh` is the acceptance launcher. It requires a qualified
physical tolerance manifest, runs SimpleFluid with its authenticated settings,
and compares against the exact retained OpenFOAM profiles bound by that manifest:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
verification/openfoam/pitzDaily/run_comparison.sh 3
```

The status and review rationale are recorded in
`reference/physical_acceptance.json`. An unqualified manifest stops the launcher
before simulation. A qualified manifest authenticates every declared reference
file and every executable setting before the run, then enforces every
station/component limit. `SIMPLEFLUID_PITZ_ACCEPTANCE_FILE` may select another
manifest with scope `physical_reference`. Acceptance uses the retained data;
regenerating OpenFOAM is a separate evidence-collection operation. This prevents
an unrelated fresh simulation from silently replacing the qualified reference.

OpenFOAM v2606 remains necessary for the common `cellPoint` sampler. The
SimpleFluid cell field is mapped bijectively to its own matching OpenFOAM mesh
and sampled at all 129 identical points at each station. Values are preserved;
the sampler checks coordinates, coverage and velocity boundary conditions.
`--use-retained-reference` selects this authenticated reference path in the
comparator CLI; it cannot be combined with a diagnostic `--openfoam-case`.

A qualified manifest must authenticate its declared retained source files with
SHA-256 digests. For physical scope it must also record qualified OpenFOAM and
SimpleFluid provenance plus an exact `simplefluid_run` contract. OpenFOAM
provenance is restricted to the named OpenFOAM.com v2606 tutorial, and the
SimpleFluid revision must be a full 40-digit Git revision. The three retained
OpenFOAM station paths are assigned by `openfoam_profile_files`; the ordered
`simplefluid_run.rank_csv_files` list must contain exactly one correctly named
CSV per declared MPI rank. Every path assigned either role must have a matching
SHA-256 entry. Nesting the CSV set with the mesh, step count, timestep, rank
count, steady-state switch, linear tolerance, and every adaptive controller
setting binds those retained outputs to one executable run contract.

The launcher validates and exports all contract values before either solver
starts, so ambient `SIMPLEFLUID_PITZ_*` solver controls cannot change a
qualified run. `SIMPLEFLUID_PITZ_OUTPUT_DIR` may select the output directory;
the launcher canonicalizes it, passes it to `run_simplefluid.sh`, and compares
the rank CSVs from that same directory.

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

`SIMPLEFLUID_PITZ_TRANSPORT_PRECONDITIONER=ilut` selects the existing ILUT
preconditioner for momentum and turbulence transport. The default is `none`;
pressure retains its independent MueLu policy. This changes linear algebra,
not the physical model or convergence tolerances. The physical run contract
and evidence plan must explicitly record `transport_preconditioner` as `none`
or `ilut`. The acceptance launcher locks that value against ambient settings,
and evidence verification checks the executable's reported selection. Older
diagnostic bundles retain their original collector for replay; new plans must
include the explicit setting.

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
When steady-state search is enabled, exhausting the step budget or retry budget
without reaching the configured criterion is a failed run and returns exit
status 1 after writing the diagnostic outputs.

The comparison script first selects the latest numeric OpenFOAM time containing
all three required station profiles, so an incomplete final write cannot be
mixed with older station data. It then selects the SimpleFluid cell layer
nearest each OpenFOAM sampling station, interpolates the common-time OpenFOAM
profile to the SimpleFluid cell centres, and reports RMS and maximum errors for
`ux` and `uy`. Optional
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

## Reproducible evidence collection

`collect_evidence.py` runs the existing `openfoam/Allrun` and
`run_simplefluid.sh` launchers independently of the acceptance gate. It uses
`compare_profiles.py` for sampling, interpolation and velocity error norms.
It never modifies the checked-in physical manifest or assigns tolerances.

Run on Linux (including WSL), in a separate worktree with the normal compiler,
Trilinos and MPI prerequisites installed:

```sh
git worktree add -b evidence/pitzdaily ../pitzdaily-evidence feature/multiphase
cd ../pitzdaily-evidence
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
python3 verification/openfoam/pitzDaily/collect_evidence.py template \
  --output ../pitzdaily-plan.json
# Complete all null settings in ../pitzdaily-plan.json after reviewing
# src/examples/pitz_daily.cc and the qualification contract below.
python3 verification/openfoam/pitzDaily/collect_evidence.py collect \
  --plan ../pitzdaily-plan.json --output ../pitzdaily-evidence-run001
python3 verification/openfoam/pitzDaily/collect_evidence.py verify \
  ../pitzdaily-evidence-run001 > ../pitzdaily-recomputed-measurements.json
python3 -m unittest discover -s verification/openfoam/pitzDaily -p 'test_*.py'
```

The template deliberately leaves the step budget and MPI rank count unset.
Other fields start from the retained run contract and require review. They are executable
controls, not evidence that a particular runtime is sufficient. Supply every
run-contract field explicitly; ambient `SIMPLEFLUID_*` settings are discarded.
Builds use the repository GCC/LLVM presets in this worktree, so another
checkout's cached executable cannot silently supply results. The collector
requires `WM_PROJECT_VERSION=v2606`. Use a fresh output path for every attempt;
it refuses to overwrite an existing directory or reuse old rank outputs.

The four named runs are:

| Role | Required relationship to baseline |
| --- | --- |
| `baseline` | Tutorial mesh divisor 1; explicitly chosen step budget and timestep |
| `stationarity` | Same settings, larger step budget; for adaptive runs, minimum steps exceed the baseline budget |
| `mesh` | Same settings and stopping rule, mesh divisor 2 |
| `timestep` | Same settings except smaller timestep and corresponding budget/bounds; fixed runs end at the baseline physical time; adaptive runs also reduce the maximum timestep and may lower the Courant target |

For a fixed-step study, double the timestep run's step budget when halving its
timestep, and choose a later stationarity endpoint on the same mesh and
timestep. For adaptive steady-state runs, all four must actually report
`reached=yes`; a zero exit status alone is insufficient. Compare their steady
endpoints rather than asserting equal pseudo-time. A reduced timestep cap can
remain inactive when the Courant controller sets the actual timestep. Lower
the timestep run's Courant target where needed; verification requires its
observed final timestep to be smaller than the baseline's, and rejects an
increased target. This is a numerical continuation control, not a physical
acceptance tolerance. A longer independent run
provides the stationarity comparison because the current executable retains
only its final rank CSVs. This does not substitute for a time history when the
physics requires averaging, nor does one coarse/fine delta prove asymptotic
mesh convergence. Additional studies and repeatability runs should use separate
bundles, with their rationale retained in the qualification review.

The bundle retains the exact plan, Git revision/status/diff, workflow and solver
source snapshots, compiler/MPI/OpenFOAM version probes and executable hashes,
build cache and solver-binary hash, command arguments, timestamps and exit
statuses. A copied OpenFOAM case contains its actual dictionaries, initial
conditions, mesh, logs and final profiles. Each SimpleFluid run retains its raw
rank CSVs and extracted station profiles. Solver logs include residual and
adaptive diagnostics. `checksums.json` authenticates retained files against
accidental changes; keep this index under version control or sign/archive it
through your normal trusted process. SHA-256 alone does not establish authorship
or physical accuracy. A dirty source tree is recorded and must be resolved or
fully accounted for before qualification.

`verify` checks checksums and required role bindings, rank completeness,
OpenFOAM residual-control convergence and profile time, SimpleFluid endpoint
summaries, and adaptive convergence diagnostics where enabled. It recomputes
all measurements from retained raw files, including:

* OpenFOAM versus baseline `ux`/`uy` RMS (`l2`) and maximum (`linf`) errors at
  all three stations, sample count, actual station position and covered span.
* Baseline versus later-time, divisor-2 and smaller-timestep profile deltas,
  including station positions and common-domain coverage. Mesh deltas use only
  the overlapping cell-centre range without extrapolation.

These are candidate measurements in m/s, not acceptance results. Collection
fails if OpenFOAM merely reaches its iteration limit, a solver command fails,
required data is missing, or a refinement contract is inconsistent. Failed
attempts keep logs, failure metadata and checksums for diagnosis and cannot
produce a usable candidate acceptance manifest. Verification may be rerun after
moving the entire bundle; it does not trust previously recorded error metrics.
After a failed SimpleFluid launcher, an available build cache and executable
hash are also retained. Their scope explicitly records availability after the
attempt; the receipt and log establish whether failure occurred before or
during solver execution.

On success, `candidate_acceptance.json` populates the existing manifest's
provenance, retained file roles and checksums, but remains
`pending_external_qualification` with an empty `stations` object. To construct
a qualified reference, archive the complete bundle outside Git and restore it
under an ignored reference directory when reviewing or executing acceptance,
review matching physics/configurations and all convergence diagnostics, repeat
or extend studies as necessary, and document per-metric tolerance headroom from
the observed refinement/repeatability envelope. Populate every station's
sampling constraints and both component limits using the existing manifest
schema, record the review rationale, and only then set `status` to `qualified`.
Do not turn measured error values directly into acceptance limits. Finally run:

```sh
python3 verification/openfoam/pitzDaily/compare_profiles.py \
  --tolerances /path/to/retained-bundle/candidate_acceptance.json \
  --required-scope physical_reference --print-simplefluid-settings
# This must fail until qualification is complete.
SIMPLEFLUID_PITZ_ACCEPTANCE_FILE=/path/to/retained-bundle/candidate_acceptance.json \
  verification/openfoam/pitzDaily/run_comparison.sh 6
```

Choose the MPI count recorded in the plan for the last command. Keep the bundle
paths relative to its manifest when archiving it. The original acceptance
launcher and its fail-closed pending physical manifest remain unchanged.

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
of the following evidence in a reviewed external bundle (or an ignored local
reference directory restored from that archive):

1. OpenFOAM.com v2606 source/case provenance, MPI rank count, solver log, and
   residual-control convergence for the three final raw profiles. List those
   profiles by station under `reference_definition.openfoam_profile_files`.
2. The exact SimpleFluid revision, compiler/build configuration, MPI rank
   count, `SIMPLEFLUID_PITZ_*` settings, and rank CSVs. Record the executable
   mesh/time, linear-solver, and adaptive steady-state values under
   `reference_definition.simplefluid_run`; use the tutorial cell counts
   (`SIMPLEFLUID_PITZ_MESH_DIVISOR=1`). Every field in the checked-in pending
   manifest is required when it is qualified. Record the ordered rank CSVs in
   that same run object, one `simplefluid_cells_rankN.csv` path for every rank.
3. Profile stationarity at the selected final SimpleFluid time, a time-step
   refinement comparison at that time, and the divisor-2 to divisor-1 mesh
   trend. The current 200-step, `1e-5 s` default is only a smoke/comparison
   setting and is not convergence evidence.
4. SHA-256 checksums for every retained raw profile and CSV under
   `qualification.source_files_sha256`; the comparator verifies the content
   and requires every role-specific profile/CSV path to name one of those
   authenticated files before accepting the manifest.
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

The nominal wall constants and block grading are aligned. Wall-production
evaluation differs, and the cases also differ in
mesh implementation, time integration versus a steady solve,
pressure-velocity iteration, and discretization details. This remains an
end-to-end comparison rather than a claim of identical numerical solutions.
The native default stores mechanical pressure and explicitly includes
`-2/3 grad(k)`. Matched qualification uses
`SIMPLEFLUID_PITZ_MODIFIED_PRESSURE=1`, which stores
`p_mechanical + 2*rho*k/3` and absorbs that force in its gradient. At rho=1,
the stored value and zero outlet match OpenFOAM modified kinematic pressure.
`SIMPLEFLUID_PITZ_OPENFOAM_OUTLET_STRESS=1` separately selects zero normal
velocity gradient with retained deviatoric transpose traction. The default
traction-free Neumann treatment is preserved for existing callers. Both options
are explicit authenticated run fields and are checked against executable logs.
The high-Re wall treatments also have a documented production-gradient
difference: SimpleFluid uses tangential velocity difference over its wall-model
distance, while OpenFOAM uses the patch `snGrad(U)` magnitude. Neither difference
has been isolated as the dominant contributor by the retained studies.

## Independent reference field checks and replay

An OpenFOAM residual-stop message is a convergence observation. Qualification
also reviews raw field histories. `reference_field_gate.py` records a separate
probe, detects scalar-only cycles even when velocity appears stationary, and
can perform bounded continuations before rechecking the original residual
criteria. Physical accuracy limits are never assigned by that protocol.

Set `reference_field_controls` explicitly in a collection plan to request the
protocol. Its fields are `schema_version`, `probe_steps`,
`required_consecutive_steps`, `scaled_update_tolerance`, `burnin_steps` and
`maximum_burnin_cycles`. The numerical studies used a 40-iteration probe,
20 consecutive accepted changes, and a global-scaled field-change criterion of
1e-7. At the 10 m/s scale this limits velocity changes to about 1e-6 m/s,
orders below measured mesh sensitivity. It also rejects the observed epsilon
cycles. A plan without these controls remains residual-only diagnostic evidence
and needs independent reference review before qualification.

Raw `k`/`epsilon` must be positive, eddy viscosity nonnegative, and every
retained physical field finite. Common sampling requires all physical CSV
columns. The source sampling dictionary is checked for exact station positions,
span, interpolation scheme, field and 129-point grid.

Use `collect_evidence.py verify BUNDLE --replay-sampling` with the recorded
OpenFOAM tool loaded to independently regenerate the retained common profiles
from raw cells. Replay checks the executable hash and rejects valid-shaped
profile corruption independently of the stored profile checksum. Offline
verification checks hashes, raw-field bindings, controls and recomputed norms.

The retained reference is finer than the operational native grid. The native
sampler constructs its own matched mesh from the tutorial geometry; the
comparison reads the exact authenticated reference files declared by the
manifest. No fresh unqualified OpenFOAM run can replace those files in acceptance.


### 48,900-cell target studies

`run_target_study.py` reuses `run_simplefluid.sh`, the physical-field checks,
matched-mesh sampler and comparator. It retains source/configuration provenance,
raw fields, actual loaded executable/library mappings, convergence diagnostics,
profiles and SHA-256 checksums. `target_study.py DIRECTORY` verifies that evidence.
The target uses mesh multiplier 2, divisor 1 and matched modified pressure/outlet
stress. Transport can use `bicgstab`/ILUT and pressure `cg`/`MueLuSymmetric`, with
unchanged residual tolerances. The symmetric preset uses symmetric smoothing,
Galerkin transfer and a direct coarse solve. The usual GMRES choices remain
available when those algorithms do not converge. Failed cases remain evidence,
never successful qualification data.

With OpenFOAM v2606 loaded and the previous authenticated parity campaign restored
beside this worktree, a fresh, finite campaign is reproducible with:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
python verification/openfoam/pitzDaily/run_target_campaign.py \
  --previous ../qualification-parity-20261022 \
  --root ../qualification-target48900-fresh
```

Run from a clean pinned checkout. The script uses two groups of three MPI ranks
on CPU groups `0,2,4` and `6,8,10`; adjust these explicit allocations for another
machine. Baseline/later-time budgets are 6000/8000 steps, with the latter prohibited
from stopping before step 6100. Time-step caps are 1e-4, 5e-5 and 2.5e-5 s, with
12000/26000-step budgets for the two smaller steps. Cases must converge rather
than merely reach those budgets. `--resume-starts` is only for the two already
running baseline/later-time cases. The script starts each smaller-step case only
after the corresponding first case passes its audit.

The campaign checks the actual endpoint time-step ratios, keeps unrelated controls
and executable hashes fixed, rechecks the previously selected reference's hashes
and independent full-field stationarity, and writes measured changes/errors.
It assigns no physical limits and never changes `physical_acceptance.json`.
A separate review is still required to justify spatial uncertainty and acceptance
limits. The legacy acceptance launcher's 25-setting contract explicitly pins its
original linear algorithms; ambient diagnostic overrides cannot change that run.
Future acceptance of the new algorithms must explicitly extend that authenticated
contract with the retained target settings.
