# Gaussian fissile-solution tank SST comparison

This case compares SimpleFluid with OpenFOAM.com v2606 for a prescribed,
Gaussian-heated cylindrical tank. It is a single-phase thermal-hydraulics
verification case: radiolysis, boiling, void feedback, and neutronics feedback
are disabled on both sides.

## Matched definition

| Quantity | Value |
| --- | --- |
| Tank | radius `0.1 m`, height `0.3 m` |
| Fission power | `1000 W` over the full tank |
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
eddy-viscosity spike; both solutions remain above approximately `0.57 1/s` in
the reported run.

The nominal R-Z subdivision is exactly `50 x 150`, or `0.002 m`. Five cells at
the radial wall and at both end walls replace base cells with widths
`1.000`, `1.180`, `1.3924`, `1.643032`, and `1.93877776 mm`, measured from
the wall inward. The remaining radial and axial cells are approximately
`2.0632 mm` and `2.0407 mm`.

OpenFOAM uses a one-cell, 5-degree wedge with 7,500 cells. Its coded source
therefore injects `1000 * 5/360 = 13.888888889 W`; revolving the wedge gives
the requested 1000 W. SimpleFluid uses a full-cylinder triangular-prism mesh
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
verification/openfoam/fissileSolutionTank/openfoam/Allrun 12
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

The following result was measured on 2026-08-03 with OpenFOAM.com v2606 and
GCC-RelWithDebInfo SimpleFluid. Both used 12 MPI ranks and ran all 1,200 fixed
`0.1 s` steps at 1000 W:

| Metric at `t=120 s` | Result |
| --- | --- |
| Integrated power | OpenFOAM full equivalent `1000 W`; SimpleFluid `1000 W` |
| Source-distribution RMS error | `3112.60 W/m3` (`0.3151%` of OpenFOAM peak) |
| Peak temperature rise | OpenFOAM `4.66435 K`; SimpleFluid `4.65883 K` |
| Temperature RMS / bias | `0.0881249 K` / `-0.00798688 K` |
| Normalized temperature RMS | `1.8893%` of OpenFOAM peak rise |
| Peak speed | OpenFOAM `0.0217573 m/s`; SimpleFluid `0.0216733 m/s` |
| Velocity-vector RMS error | `0.000532123 m/s` (`2.4457%` of peak speed) |
| Maximum wall y+ | OpenFOAM `2.0272`; SimpleFluid `2.0395` |
| Peak maximum Courant | OpenFOAM `1.7587` at `19.2 s`; SimpleFluid sampled `3.7336` at `19 s` |
| Final maximum Courant | OpenFOAM `1.0663`; SimpleFluid `2.3088` |

For the SST fields, the SimpleFluid/OpenFOAM geometric-mean ratios are
`0.3251` for k, `1.1096` for omega, and `0.2825` for turbulent kinematic
viscosity. The respective volume fractions within a factor of two are
`46.81%`, `97.95%`, and `40.87%`. Thus the mean thermal and velocity fields
remain close at 120 s, but the long-horizon run does not verify agreement of
the two SST closures.

## Interpretation and failure modes

The final temperature and velocity distributions agree at the several-percent
level, and the matched circulation topology is visible in both R-Z fields.
Their errors are structured around the rising wall jet, upper return flow, and
thermal-front locations rather than appearing as random solver noise. The
remaining differences are consistent with several known implementation
differences:

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

The turbulence variables are a weaker result. Their volume-weighted mean
turbulent kinematic viscosities are `2.767e-6 m2/s` for OpenFOAM and
`1.615e-6 m2/s` for SimpleFluid, compared with a molecular value of
`1.0e-6 m2/s`. Turbulent transport is therefore material at 1000 W, and its
spatial mismatch should not be dismissed merely because the mean temperature
and velocity errors remain small. The SST variant, production limiter, wall
treatment, mesh topology, and segregated update ordering can all compound;
the present comparison does not isolate one of them as the sole cause.

Both Courant histories accelerate to a maximum at about 19 s and then relax.
SimpleFluid also shows a smaller late plateau near 2.5. The peak maximum
Courant is `3.7336` in the 1 s SimpleFluid samples and `1.7587` in OpenFOAM;
both final values remain above one. A matched smaller-time-step study is
therefore necessary before treating the remaining 2--3% field difference as
purely model-form error. Every accepted GMRES run step converged, and both
codes reach bounded final fields with maximum wall y+ near two.

An earlier 100 W, 12-rank SimpleFluid attempt used BiCGStab with ILUT for the
physical transport equations. It completed step 473 (`t=47.3 s`) and then
reported a momentum linear-solve nonconvergence on all ranks. GMRES with the
same ILUT preconditioner, `1e-9` tolerance, and 1,000-iteration limit passed
that point in both the 100 W and 1000 W runs and completed all 1,200 steps.
This identifies the old abort as a transport linear-solver robustness failure,
not evidence of a physical SST instability.

Finally, standard SST k-omega is a fully turbulent RANS closure, not a
laminar-to-turbulent transition model: it has no intermittency or transition
correlation. Extending the horizon can expose transient RANS behaviour and
cross-code closure drift, as it does here, but it cannot by itself establish a
physical transition onset. That would require a matched transition-capable
closure or a sufficiently resolved reference calculation.

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
