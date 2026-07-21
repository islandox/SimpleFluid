# OpenFOAM pitzDaily comparison

This fixture is derived from
`$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily` in OpenFOAM v2606.
It retains the tutorial's five-block backward-facing-step geometry, 10 m/s
inlet, `1e-5 m2/s` kinematic viscosity, standard k-epsilon closure, inlet
`k=0.375 m2/s2`, and inlet `epsilon=14.855 m2/s3`. The checked-in OpenFOAM
case adds vertical velocity samples at `x=0.05`, `0.10`, and `0.20 m`.

The SimpleFluid executable uses the same duct outline and boundary names. Its
mesh has the same five connected blocks but uses uniform block spacing and is
coarsened from the tutorial by a factor of four in each in-plane direction by
default. Set `SIMPLEFLUID_PITZ_MESH_DIVISOR=1` for the tutorial cell counts.

## Run

Load OpenFOAM, then run both cases on up to six ranks:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
verification/openfoam/pitzDaily/run_comparison.sh 6
```

Run either side independently with:

```sh
verification/openfoam/pitzDaily/openfoam/Allrun 6
verification/openfoam/pitzDaily/run_simplefluid.sh 6
```

The SimpleFluid launcher builds `RelWithDebInfo` by default and writes rank
CSV and VTU files under `profiles/`. Runtime and resolution are configurable:

```sh
SIMPLEFLUID_PITZ_MESH_DIVISOR=2 \
SIMPLEFLUID_PITZ_STEPS=1000 \
SIMPLEFLUID_PITZ_DT=1e-5 \
  verification/openfoam/pitzDaily/run_simplefluid.sh 6
```

The comparison script selects the SimpleFluid cell layer nearest each OpenFOAM
sampling station, interpolates the OpenFOAM profile to the SimpleFluid cell
centres, and reports RMS and maximum errors for `ux` and `uy`. Optional
`--max-l2` and `--max-linf` arguments turn those metrics into pass/fail checks.
No default tolerance is imposed because the numerical differences below keep
this fixture from serving as a validated reference solution.

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

The wall model is now aligned, but the cases still differ in mesh spacing,
time integration versus a steady solve, pressure-velocity iteration, and
discretization details. This remains an end-to-end comparison rather than a
claim of identical numerical solutions. OpenFOAM pressure is kinematic;
SimpleFluid uses a reference density of `1 kg/m3`, making pressure numerically
comparable, while the automated profile comparison intentionally evaluates
velocity only.
