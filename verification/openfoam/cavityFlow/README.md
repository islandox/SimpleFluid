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
SIMPLEFLUID_PROFILE_OUTPUT_DIR="$PWD/profiles" \
  build/bin/Debug/testVerificationCases \
  --gtest_filter=VerificationCasesTest.LidDrivenCavityRe1000
```

This writes:

- `profiles/simplefluid_lineX.csv`
- `profiles/simplefluid_lineY.csv`

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
absolute differences.

OpenFOAM is not bundled with SimpleFluid, so generating the external raw file
remains an environment-level step. The automated verification test checks
that the SimpleFluid Re=1000 case definition uses the same nondimensional
parameters and wall conditions.
