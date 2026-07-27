# Shiri natural-convection comparison

This fixture implements the final closed-cavity configuration in
`NC_Tutorial_Shiri.pdf`. The heated vertical pipe is at 360 K; the outer,
bottom, and top walls are at 290 K; all physical walls are no-slip. The
OpenFOAM case retains the tutorial's 45-degree cyclic wedge and four-block
mesh. The SimpleFluid executable uses the same radial and axial block edges in
a 45-degree annular sector with symmetry (slip/zero-gradient) side planes.
Those planes are equivalent to the OpenFOAM cyclic pair while the computed
solution remains axisymmetric.

The tutorial is internally inconsistent about pipe size: its prose says a
0.15 m radius, while every inner vertex in `blockMeshDict` is at 75 mm. This
fixture follows the executable geometry, hence an inner radius of 0.075 m.

## Run

Load an OpenFOAM installation that provides
`buoyantBoussinesqPimpleFoam`, then run both solvers on up to six ranks:

```sh
verification/openfoam/naturalConvection/run_comparison.sh 6
```

Run either side independently with:

```sh
verification/openfoam/naturalConvection/openfoam/Allrun 6
verification/openfoam/naturalConvection/run_simplefluid.sh 6
```

The SimpleFluid launcher sources `verification/environments.sh` and builds the
`GCC-RelWithDebInfo` preset by default, using
`build-gcc/bin/RelWithDebInfo`. Set `SIMPLEFLUID_COMPILER=LLVM` to select the
corresponding LLVM preset, or set `SIMPLEFLUID_BUILD_CONFIG=Debug` only when a
debug comparison is needed. A custom, already-configured build tree can be
selected with `SIMPLEFLUID_BUILD_DIR`.

Both cases start from the same quiescent 290 K field and the tutorial's
`k=0.00375 m2/s2`, `epsilon=0.00075 m2/s3` turbulence state. They advance
200 transient steps with a 0.002 s step size to the matched comparison time
`t=0.4 s`. SimpleFluid defaults to the OpenFOAM mesh's `40 x 20 x 100`
radial, angular, and axial cell counts. Override a standalone SimpleFluid
resolution or duration with:

```sh
SIMPLEFLUID_SHIRI_NR=40 \
SIMPLEFLUID_SHIRI_NTHETA=20 \
SIMPLEFLUID_SHIRI_NZ=100 \
SIMPLEFLUID_SHIRI_STEPS=5000 \
SIMPLEFLUID_SHIRI_DT=0.002 \
  verification/openfoam/naturalConvection/run_simplefluid.sh 6
```

## Search for a steady state

The fixed-step transient remains the default. Enable adaptive
pseudo-transient continuation explicitly when the desired result is a steady
state rather than the matched `t=0.4 s` comparison:

```sh
SIMPLEFLUID_SHIRI_STEADY_STATE=1 \
SIMPLEFLUID_SHIRI_STEPS=5000 \
SIMPLEFLUID_SHIRI_DT=0.002 \
SIMPLEFLUID_SHIRI_STEADY_MIN_DT=0.000125 \
SIMPLEFLUID_SHIRI_STEADY_MAX_DT=0.05 \
SIMPLEFLUID_SHIRI_STEADY_TARGET_COURANT=0.8 \
SIMPLEFLUID_SHIRI_STEADY_TOLERANCE=1e-4 \
  verification/openfoam/naturalConvection/run_simplefluid.sh 1
```

`SIMPLEFLUID_SHIRI_STEPS` is the maximum search length and
`SIMPLEFLUID_SHIRI_DT` is the initial pseudo-time step. After every accepted
step, the controller computes the maximum cell Courant number from the
pressure-projected face flux and ramps the next step toward the target,
limited to 1.5 times the preceding step by default. The maximum and growth
factor can be overridden with `SIMPLEFLUID_SHIRI_STEADY_MAX_DT` and
`SIMPLEFLUID_SHIRI_STEADY_DT_GROWTH`. The minimum defaults to one sixteenth
of the initial step and can be set with
`SIMPLEFLUID_SHIRI_STEADY_MIN_DT`.

A rejected transactional momentum solve is retried at half the preceding
pseudo-time step, up to four retries. Configure those safeguards with
`SIMPLEFLUID_SHIRI_STEADY_DT_REDUCTION` and
`SIMPLEFLUID_SHIRI_STEADY_MAX_RETRIES`. Rejected attempts do not advance
pseudo-time, consume the accepted-step limit, or update the steady-state
confirmation window. Failures after a solver stage has published state are
not retried; they remain fatal rather than risk continuing from a partially
advanced solution.

Steady convergence uses actual volume-weighted changes in velocity,
temperature rise, `k`, and `epsilon`, not the linear solver's internal
residual. Each change is normalized by the current field scale and by the
accepted pseudo-time step, so the reported `update_rates` have inverse-time
units. The maximum rate must remain below
`SIMPLEFLUID_SHIRI_STEADY_TOLERANCE` for five consecutive steps after at
least 20 steps. Override those guards with
`SIMPLEFLUID_SHIRI_STEADY_CONSECUTIVE_STEPS` and
`SIMPLEFLUID_SHIRI_STEADY_MIN_STEPS`.

The executable writes the latest fields even when the maximum step count is
reached, reports `steady_state_search: reached=no`, and returns exit status 2
so an unconverged search cannot be mistaken for a steady solution. Adaptive
mode is deliberately rejected by `run_comparison.sh`, whose OpenFOAM
reference is a fixed-time transient.

The Shiri executable defaults transported equations to `bicgstab/jacobi` and
pressure projection to `bicgstab/MueLu`. Override either policy independently:

```sh
SIMPLEFLUID_SHIRI_LINEAR_SOLVER_BACKEND=gmres \
SIMPLEFLUID_SHIRI_LINEAR_PRECONDITIONER=jacobi \
SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_SOLVER_BACKEND=bicgstab \
SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_PRECONDITIONER=MueLu \
  verification/openfoam/naturalConvection/run_simplefluid.sh 1
```

Supported backend names are `gmres`, `bicgstab`, and `cg`; supported
preconditioners are `none`, `jacobi`, `ilu0`, `ilut`, and `MueLu`. Do not use
CG for this case: the pressure gauge is imposed by replacing one matrix row,
so the resulting operator is not symmetric.

The default mesh, step count, and step size are required for the supplied
comparison. The comparator requires the latest OpenFOAM profile at `t=0.4 s`,
the fixed OpenFOAM sample radius `r=0.0800000443 m`, and 80,000 unique
SimpleFluid cells forming the complete `40 x 20 x 100` mesh. The overrides are
for standalone experiments and are rejected by `run_comparison.sh`; a
different matched run requires updating both cases and the comparator
expectations. Because the SimpleFluid cell CSV does not encode time, invoke
the comparator through that launcher to guarantee a fresh matched-time pair.

Results are written under `profiles/` as one VTU and one cell CSV per rank.
During the SimpleFluid solve, MPI rank zero prints one flushed convergence
summary per physical step by wrapping `std::cout` in a `ProgressStream` and
passing it through the solver's `run(steps, progress)` interface. OpenFOAM
uses `cellPoint` to sample `T`, `U`, `k`, `epsilon`, `nut`, and `alphat` on
one axial line at the fixed radius and one sector angle. The comparator
theta-averages SimpleFluid cell-centred values on the two bracketing radial
layers, linearly interpolates them to that radius, and interpolates the
OpenFOAM line axially at the SimpleFluid cell centres. Radial, azimuthal, and
axial velocity components are all compared.

The reported `axial_rms` is a trapezoidal, axial-space-weighted RMS over the
overlapping line span; it is not a volume or full-domain norm. `linf` is the
maximum pointwise absolute difference. `relative_axial_rms` divides by the
corresponding axial RMS of the OpenFOAM field. For temperature that
normalization uses `T - 290 K`, so it measures error relative to the
temperature rise; the other fields are unshifted. Per-field maximum theta
spreads on both SimpleFluid radial layers and the maximum selected-layer
`|u_theta|` expose departures from the axisymmetry needed to compare a theta
average with OpenFOAM's single-theta sample. Both solvers also emit
wall-y-plus diagnostics. The comparator also reports the global SimpleFluid
temperature range and warns if it leaves the imposed 290--360 K bounds.

The interior turbulent thermal-diffusivity metric derives `nut/Prt` with
`Prt=0.85` on both sides after their respective sampling and interpolation; it
does not compare raw OpenFOAM `alphat`. The raw `alphat` boundary value remains
relevant to the Jayatilleke wall heat flux and is a separate wall-treatment
check.

## Interpretation

Both fixtures use the modern incompressible Boussinesq standard k-epsilon
model with `Cmu=0.09`, `C1=1.44`, `C2=1.92`, `sigmak=1.0`, and
`sigmaEps=1.3`. They use smooth `kqRWallFunction`,
`epsilonWallFunction`, and `nutkWallFunction` equivalents plus the
Jayatilleke thermal wall law at `Prt=0.85`. OpenFOAM's
`buoyancyTurbSource` and SimpleFluid's `OpenFOAMBoussinesq` option apply the
same temperature-gradient production to `k` and `epsilon`.

The continuum closures are aligned, but the segregated update order is not.
OpenFOAM advances velocity and temperature with lagged `nut`, corrects
pressure, then updates `epsilon` and `k` using the new temperature.
SimpleFluid corrects velocity and pressure, advances `k` and `epsilon` from
the accepted temperature state, then advances temperature with the updated
`nut`. Fixed-step differences therefore include first-order operator-splitting
error; timestep refinement is required before attributing them solely to the
spatial discretizations.

The PDF instead used the legacy compressible `buoyantSimpleFoam` closure, so
this fixture is not an exact reproduction of its density-based buoyant
k-epsilon model. Reported wall `y+` values on parts of this mesh fall below
the log layer required by the high-Re wall functions. The matched `t=0.4 s`
output is therefore a code-to-code transient agreement check, not a
physically resolved RANS prediction. No pass/fail tolerance is imposed until
temporal and mesh convergence are established.

The OpenFOAM dictionaries target releases that still ship
`buoyantBoussinesqPimpleFoam`; newer modular OpenFOAM releases may require
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
