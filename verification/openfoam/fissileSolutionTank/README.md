# Gaussian fissile-solution tank SST comparison

This case compares SimpleFluid with OpenFOAM.com v2606 for a prescribed,
Gaussian-heated cylindrical tank. It is a single-phase thermal-hydraulics
verification case: radiolysis, boiling, void feedback, and neutronics feedback
are disabled on both sides.

## Matched definition

| Quantity | Value |
| --- | --- |
| Tank | radius `0.1 m`, height `0.3 m` |
| Fission power | `100 W` over the full tank |
| Gaussian center | `R=0`, `Z=0.15 m` |
| Gaussian widths | `sigma_R=0.03 m`, `sigma_Z=0.075 m` |
| Initial and wall temperature | `300 K` |
| Density / heat capacity | `1000 kg/m3` / `4200 J/(kg K)` |
| Dynamic viscosity / conductivity | `1e-3 Pa s` / `0.6 W/(m K)` |
| Thermal expansion / gravity | `2.1e-4 1/K` / `-9.81 m/s2` in Z |
| Turbulence | SST k-omega, `k=1e-6 m2/s2`, `omega=1 1/s`, `Prt=0.85` |
| Time integration | Euler, `dt=0.1 s`, 1,200 steps, compare at `t=120 s` |
| Coupling / convection | two PISO correctors / first-order upwind |

All physical walls are no-slip and isothermal. SimpleFluid uses its resolved
low-Re SST treatment and exact cylinder wall distance
`min(0.1-R, Z, 0.3-Z)`. Its initial omega field is made wall-compatible with
`max(1, 6 nu/(beta1 y^2))`, and the case-specific lower bound is `0.5 1/s`.
The bound prevents a numerical omega undershoot from producing an unphysical
eddy-viscosity spike; OpenFOAM remains above approximately `0.54 1/s` in the
reported run.

The nominal R-Z subdivision is exactly `50 x 150`, or `0.002 m`. Five cells at
the radial wall and at both end walls replace base cells with widths
`1.000`, `1.180`, `1.3924`, `1.643032`, and `1.93877776 mm`, measured from
the wall inward. The remaining radial and axial cells are approximately
`2.0632 mm` and `2.0407 mm`.

OpenFOAM uses a one-cell, 5-degree wedge with 7,500 cells. Its coded source
therefore injects `100 * 5/360 = 1.3888888889 W`; revolving the wedge gives
the requested 100 W. SimpleFluid uses a full-cylinder triangular-prism mesh
with an independent `0.01 m` circumferential target and 495,450 cells. After
each accepted step, its state is volume-averaged over azimuth onto the same
50 x 150 R-Z bins and republished. This enforces the axisymmetry inherent in
the OpenFOAM wedge and sets azimuthal velocity to zero.

## Run

Load OpenFOAM and run both solvers on up to twelve ranks:

```sh
source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
verification/openfoam/fissileSolutionTank/run_comparison.sh 12
```

Run either side independently with:

```sh
verification/openfoam/fissileSolutionTank/openfoam/Allrun 1
verification/openfoam/fissileSolutionTank/run_simplefluid.sh 12
```

The SimpleFluid launcher uses `GCC-RelWithDebInfo` by default. The usual
`SIMPLEFLUID_COMPILER`, `SIMPLEFLUID_BUILD_CONFIG`, and
`SIMPLEFLUID_BUILD_DIR` overrides are supported. Mesh and time overrides are
accepted by `run_simplefluid.sh`, but `run_comparison.sh` rejects them because
they would no longer match the checked-in OpenFOAM case.

The programmatic cylinder generator constructs the global geometry only on
rank 0. After conversion to compact legacy arrays it releases the STK source,
scatters cell-adjacency rows for partitioning, and sends owned and ghost cell
packets to every rank. Cell data is serialized directly into exact-size MPI
buffers; the complete partition map exists only transiently on rank 0, while
neighbor ownership travels in each packet. Rebuilds release global vector
capacities promptly, host-only runs allocate Kokkos device mirrors only if
requested, and the solver's legacy `MeshHandle` views existing IDs and
connectivity instead of duplicating them. Rank 0 still needs the most headroom
during global construction and redistribution.

## Outputs

SimpleFluid rank CSV and VTU files are written under `profiles/`. The
comparison maps both solutions to identical R-Z bins and writes:

- `profiles/rz/comparison_summary.json`
- `profiles/rz/rz_matched.csv`
- temperature-rise distribution and signed/absolute-error figures
- velocity-magnitude distribution and signed/absolute-error figures
- fission-power-density distribution and signed/absolute-error figures

Each figure is emitted as SVG and, when `rsvg-convert` is installed, PNG and
PDF.

## Verified result

The following result was measured on 2026-07-29 with a four-rank
GCC-RelWithDebInfo SimpleFluid run and a serial OpenFOAM.com v2606 run:

| Metric at `t=10 s` | Result |
| --- | --- |
| Integrated power | OpenFOAM full equivalent `100 W`; SimpleFluid `100 W` |
| Source-distribution RMS error | `311.260 W/m3` (`0.3151%` of OpenFOAM peak) |
| Peak temperature rise | OpenFOAM `0.231258 K`; SimpleFluid `0.230424 K` |
| Temperature RMS / bias | `0.000737157 K` / `-0.000488037 K` |
| Normalized temperature RMS | `0.3188%` of OpenFOAM peak rise |
| Peak speed | OpenFOAM `0.00178129 m/s`; SimpleFluid `0.00173816 m/s` |
| Velocity-vector RMS error | `1.29279e-5 m/s` (`0.7258%` of peak speed) |
| Maximum wall y+ | OpenFOAM `0.2130`; SimpleFluid `0.1781` |

For the SST fields, the SimpleFluid/OpenFOAM geometric-mean ratios are
`0.8640` for k, `1.0872` for omega, and `0.7993` for turbulent kinematic
viscosity. The respective volume fractions within a factor of two are
`95.56%`, `98.01%`, and `92.73%`. These are useful closure diagnostics, not a
claim that every SST implementation detail is identical.

## Interpretation and failure modes

The final temperature and velocity fields agree well at this transient
horizon. The remaining systematic temperature and speed underprediction is
consistent with several known differences:

- OpenFOAM's current `kOmegaSST` uses its strain-based limiter, while
  SimpleFluid implements the original Menter-1994 vorticity limiter and a
  production cap of 20.
- OpenFOAM is an orthogonal wedge; SimpleFluid uses a polygonal full cylinder
  with unstructured triangular prisms and azimuthal volume averaging.
- The full-equivalent OpenFOAM volume is `0.00941282 m3`, the SimpleFluid
  volume is `0.00940916 m3`, and the analytic cylinder volume is
  `0.00942478 m3`. Both sources are normalized on their own discrete volumes,
  explaining a small local power-density difference despite exact total
  power.
- The solvers use different pressure variables, wall implementations,
  segregated update ordering, and linear algebra.

A more aggressive `0.2 mm` first radial layer was tested and rejected. With
the `0.01 m` circumferential target, polygon curvature reduced some actual
wall-face normal distances to `25 micrometres`; the resolved SST wall omega
condition then drove omega outside a useful numerical range, eddy viscosity
reached about `0.3 m2/s`, and max Courant exceeded 13 by `t=0.2 s`. The adopted
1 mm stack keeps wall-face distances between `0.292` and `0.500 mm` and all
layers at or below the 2 mm base size.

Likewise, an unconstrained full-cylinder run developed non-axisymmetric mesh
modes that the wedge reference cannot represent. At `t=1 s` its raw peak speed
was `0.122 m/s`, while its azimuthal R-Z mean was only `0.00637 m/s`.
Projecting each accepted state onto that mean removes this comparison-domain
mismatch. Consequently, this fixture verifies the matched axisymmetric
thermal/SST path; it is not evidence that arbitrary three-dimensional SST
tank runs are validated.
