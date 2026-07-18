# Shiri natural-convection comparison

This fixture implements the final closed-cavity configuration in
`NC_Tutorial_Shiri.pdf`. The heated vertical pipe is at 360 K; the outer,
bottom, and top walls are at 290 K; all physical walls are no-slip. The
OpenFOAM case retains the tutorial's 45-degree cyclic wedge and four-block
mesh. The SimpleFluid executable uses the same 45-degree annular sector with
symmetry (slip/zero-gradient) side planes, which is equivalent for the intended
axisymmetric solution and can be distributed over MPI.

The tutorial is internally inconsistent about pipe size: its prose says a
0.15 m radius, while every inner vertex in `blockMeshDict` is at 75 mm. This
fixture follows the executable geometry, hence an inner radius of 0.075 m.

## Run

Load an OpenFOAM installation that provides
`buoyantBoussinesqSimpleFoam`, then run both solvers on up to six ranks:

```sh
verification/openfoam/naturalConvection/run_comparison.sh 6
```

Run either side independently with:

```sh
verification/openfoam/naturalConvection/openfoam/Allrun 6
verification/openfoam/naturalConvection/run_simplefluid.sh 6
```

The SimpleFluid launcher builds `RelWithDebInfo` by default. Set
`SIMPLEFLUID_BUILD_CONFIG=Debug` only when a debug comparison is needed.

The OpenFOAM case defaults to 2000 steady SIMPLE iterations. SimpleFluid
defaults to a `40 x 20 x 100` annular mesh, 200 transient steps, and a 0.002 s
step size. Override the SimpleFluid resolution or duration with:

```sh
SIMPLEFLUID_SHIRI_NR=40 \
SIMPLEFLUID_SHIRI_NTHETA=20 \
SIMPLEFLUID_SHIRI_NZ=100 \
SIMPLEFLUID_SHIRI_STEPS=5000 \
SIMPLEFLUID_SHIRI_DT=0.002 \
  verification/openfoam/naturalConvection/run_simplefluid.sh 6
```

Results are written under `profiles/` as one VTU and one cell CSV per rank.
During the SimpleFluid solve, MPI rank zero prints one flushed convergence
summary per physical step by wrapping `std::cout` in a `ProgressStream` and
passing it through the solver's `run(steps, progress)` interface.
OpenFOAM samples `T` and `U` on an axial line at `r=0.080 m`. The comparison
script selects the nearest SimpleFluid radial cell layer, averages its sector
theta cells, interpolates the OpenFOAM profile to the SimpleFluid axial cell
centres, and reports L2 and maximum absolute differences.

## Interpretation

The PDF used compressible `buoyantSimpleFoam` with a k-epsilon RAS model. This
fixture deliberately uses laminar Boussinesq physics in OpenFOAM because that
is the model implemented by SimpleFluid. At the tutorial dimensions and 70 K
temperature difference the Rayleigh number is large, so these laminar results
are a code-to-code discretization check, not a validated turbulent prediction.
No pass/fail tolerance is imposed until a mesh- and time-converged reference is
established.

The OpenFOAM dictionaries target releases that still ship
`buoyantBoussinesqSimpleFoam`; newer modular OpenFOAM releases may require
solver-dictionary migration.

## Profile the SimpleFluid solve

The `RelWithDebInfo` build retains debug symbols and frame pointers. With
Valgrind installed, collect a serial Callgrind profile with:

```sh
verification/openfoam/naturalConvection/profile_simplefluid.sh
```

The profiling launcher defaults to a `6 x 3 x 12` mesh and two steps so it
finishes quickly while exercising repeated pressure corrections. It toggles
collection only while `BoussinesqSolver::step()` is running, excluding mesh
construction and result output from the instruction counts. The usual
`SIMPLEFLUID_SHIRI_NR`, `SIMPLEFLUID_SHIRI_NTHETA`,
`SIMPLEFLUID_SHIRI_NZ`, `SIMPLEFLUID_SHIRI_STEPS`, and
`SIMPLEFLUID_SHIRI_DT` overrides select a larger workload.

The raw data and an inclusive text report are written to
`profiles/callgrind/callgrind.out` and
`profiles/callgrind/callgrind.annotated.txt`. Open the raw data in a Callgrind
viewer, or regenerate the report with `callgrind_annotate`. Callgrind measures
executed instructions rather than elapsed time, so confirm an optimization
with a normal `RelWithDebInfo` run as well.
