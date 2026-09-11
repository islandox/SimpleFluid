# OpenFOAM cavity comparison

This fixture matches the published `incompressibleFoam` lid-driven cavity:

- Repository: <https://github.com/ferrop/incompressibleFoam>
- Commit: `22823e134183b57e06b8c2a3e8656e12e55a54f9`
- Case: `tutorials/cavityFlow`
- Grid: `100 x 100 x 1`
- Reynolds number: `1000`
- Lid speed and cavity length: `1`
- Kinematic viscosity: `0.001`

The upstream case samples `U` on the horizontal and vertical centerlines.
Generate matching SimpleFluid centerline CSV files with the test-only
`VelocityProfileCsv` helper by running:

```sh
verification/openfoam/cavityFlow/run_simplefluid.sh
```

The default runs the existing PISO Re=1000 verification. Select the coupled
NOX solver explicitly after configuring a Trilinos build with NOX/Thyra
support:

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_NOX=ON
SIMPLEFLUID_PROFILE_OUTPUT_DIR="$PWD/profiles-nox-re1000" \
  verification/openfoam/cavityFlow/run_simplefluid.sh --coupling nox
SIMPLEFLUID_PROFILE_OUTPUT_DIR="$PWD/profiles-nox-re100" \
  verification/openfoam/cavityFlow/run_simplefluid.sh --coupling nox --reynolds 100
```

`--reynolds 100|1000` works with either coupling mode. Setting
`SIMPLEFLUID_CAVITY_COUPLING=nox` is equivalent to `--coupling nox`; the
command-line option takes precedence. A NOX run requires
`SIMPLEFLUID_ENABLE_NOX=ON` in the selected build tree's `CMakeCache.txt`.
The launcher checks this before building or replacing profiles and does not
reconfigure a build to enable NOX. It also requires both profile files after
the test, so a missing or skipped test cannot report a successful profile run.

The NOX cases run `VerificationCasesTest.LidDrivenCavityRe100Nox` or
`VerificationCasesTest.LidDrivenCavityRe1000Nox` on the native orthogonal
Cartesian mesh, with constant viscosity, a moving lid, stationary in-plane
walls, and physical slip on the two out-of-plane boundaries. This exercises
the supported fixed-mesh velocity-pressure problem; thermal, bubble,
turbulence, and ALE coupling are outside these cases. Configure the existing
coupled operator and workspace policies using:

```sh
SIMPLEFLUID_CAVITY_COUPLED_OPERATOR=block_composite \
SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE=streamed_products \
SIMPLEFLUID_PROFILE_OUTPUT_DIR="$PWD/profiles-nox-composite" \
  verification/openfoam/cavityFlow/run_simplefluid.sh --coupling nox
```

The accepted policy values are `assembled|block_composite` and
`cached_products|streamed_products`, respectively. These controls require
NOX coupling; the launcher rejects them for PISO.

The NOX enablement was checked with GCC Debug on 2026-09-11: all six cavity
configuration/solver checks passed serially, and both NOX Reynolds-number
cases passed on two MPI ranks. The three focused slip-boundary checks passed
serially; the two distributed checks also passed, with the rotated serial-mesh
fixture skipped under MPI. All 11 launcher checks passed. Real launcher runs
exported both profiles for Re=1000 NOX assembled/cached, Re=100 NOX
composite/streamed, and the default Re=1000 PISO selection. Each profile had
eight finite samples at the default resolution. A disabled-NOX selection was
also confirmed to fail before creating its output directory.

Without an output directory override, the launcher writes:

- `profiles/simplefluid_lineX.csv`
- `profiles/simplefluid_lineY.csv`

`SIMPLEFLUID_PROFILE_OUTPUT_DIR` changes the destination for both files.
Use separate directories when comparing coupling modes or Reynolds numbers.

For mesh-refinement studies, set `SIMPLEFLUID_CAVITY_MESH_CELLS` to the
number of cells in each in-plane direction. The default is `8`. Fine meshes
may also need a larger pressure-solver budget through
`SIMPLEFLUID_CAVITY_MAX_LINEAR_ITERATIONS`, whose default is `300`:

```sh
SIMPLEFLUID_CAVITY_MESH_CELLS=64 \
SIMPLEFLUID_CAVITY_MAX_LINEAR_ITERATIONS=2000 \
SIMPLEFLUID_PROFILE_OUTPUT_DIR="$PWD/profiles-64" \
  verification/openfoam/cavityFlow/run_simplefluid.sh
```

The launcher sources `verification/environments.sh` and builds the
`GCC-RelWithDebInfo` preset by default, using
`build/gcc/bin/RelWithDebInfo`. Set `SIMPLEFLUID_COMPILER=LLVM` to use the
corresponding LLVM preset and `build/llvm/bin/RelWithDebInfo`,
`SIMPLEFLUID_BUILD_CONFIG=Debug` for a debug run, or
`SIMPLEFLUID_BUILD_DIR` for a custom, already-configured build tree.

The helper samples the nearest cell-center line and volume-averages symmetric
ties, which handles even structured grids where the geometric centerline lies
between two cell rows.

After running the OpenFOAM case's `Allrun`, compare its raw centerline file
against the generated SimpleFluid CSV with:

```sh
python3 verification/openfoam/cavityFlow/compare_profiles.py \
  --openfoam postProcessing/line_plot/50/lineY_U.xy \
  --simplefluid profiles/simplefluid_lineY.csv \
  --component ux
```

OpenFOAM raw input is expected as `coordinate ux uy uz`. SimpleFluid input is
expected as `coordinate,ux,uy,uz`. The script linearly interpolates the
OpenFOAM profile to SimpleFluid sample locations and reports L2 and maximum
absolute differences. Add `--max-l2` and/or `--max-linf` to make the command
exit unsuccessfully when a project-approved tolerance is exceeded. No default
tolerance is imposed because the repository does not yet contain a validated
reference profile and acceptance threshold.

OpenFOAM is not bundled with SimpleFluid, so generating the external raw file
remains an environment-level step. The automated verification test checks
that the SimpleFluid Re=1000 case definition uses the same nondimensional
parameters and wall conditions. Both coupling modes use an 8 x 8 x 1 grid by
default and a short 12-step transient window. These runs check startup and
solver integration; they do not qualify a fully developed cavity profile or
claim numerical agreement with the external Re=1000 reference. Re=100 is
an additional solver verification, not a match to that reference case.
