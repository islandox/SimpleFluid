# Matched SST with resolved viscous sublayers

The convection case now advances **Menter-1994 k–omega SST** on both sides.
SimpleFluid uses its production `SSTKOmega` model with `ResolvedLowReSST`.
The bundled OpenFOAM application independently assembles the same closure
with OpenFOAM finite-volume operators in `solver/ReferenceSST.H`. It does not
use stock SST-2003 settings or read SimpleFluid solution fields.

The formulation follows the [Turbulence Modeling Resource SST definition](https://tmbwg.github.io/turbmodels/sst.html):
the vorticity-based viscosity limiter, a 20 betaStar k omega production cap
in the k equation, standard blended coefficients, and signed cross diffusion.
Both turbulence equations use backward Euler, first-order upwind transport,
linear face diffusion coefficients and coefficients frozen at the old
k/omega state. Both use Gauss-linear turbulence gradients. The momentum
system includes effective viscosity, deviatoric transpose stress, and
-2/3 grad(k); the pressure conventions remain dynamic (SimpleFluid) and
kinematic (OpenFOAM).

## Wall treatment and mesh acceptance

“Resolved wall” here means integration through the viscous sublayer, with
**maximum wall-face y+ <= 1**, not a high-Re logarithmic wall law.
The no-slip patches are xmin, xmax and zmin. The top and y boundaries remain
impermeable slip, preserving the existing model geometry.

At each no-slip face, both implementations impose:

- k_wall = 0;
- omega_wall = 60 nu / (beta1 d_face^2), beta1 = 0.075;
- nut_wall = 0, hence molecular viscosity and conductivity at the wall.

`d_face` is the wall-normal distance from the adjacent cell centre to that
face. Omega is a **face Dirichlet value**, not the six-coefficient owner-cell
constraint used by some omega wall functions. The nearest-wall distance
inside this Cartesian box is min(x, W-x, z). Initial omega uses
max(1, 6 nu/(beta1 d_nearest^2)); initial k is 1e-8 m²/s², with initial nut=k/omega.
These are documented startup seeds, not measured turbulence inlet data.
The source-off rest control starts k at its 1e-12 floor to represent quiescent
water and retains its original stationary-flow tolerance.

Each timestep independently checks every sampled wall face through the
maximum over globally owned wall faces. `wall_resolution.csv` records this
maximum every step, and the run fails if it exceeds the target. The existing
shared cell widths are retained initially. A failing y+ check requires a
shared mesh refinement and a fresh run; it is not accepted by changing the
y+ threshold. Resolving y+ does not establish bulk mesh or time convergence.

## Buoyancy, heat and gas

Both sides use Prt=0.9 and the same frozen mixture density as the momentum
buoyancy closure. Direct turbulent buoyancy production is

    Gb = -nut/(Prt rho0) g dot grad(rho_mixture).

Positive production is explicit; negative production is an implicit sink.
The omega equation receives gamma Gb/(nut + machine epsilon), split in the
same way. Molecular nu is mu_reference/rho_reference. The effective dynamic
viscosity is mu_reference + rho_reference nut. Thermal conductivity is
k_reference + rho_mixture cp_reference nut/Prt.

Each step advances momentum/pressure, then SST, then temperature, then gas
and material feedback. The thermal budget uses the actual frozen heat
capacity, the new SST conductivity used during temperature transport, and
molecular conductivity at resolved walls. At the cold slip top, the
owner-cell effective conductivity is used. Existing mass, heat-source,
continuity and temperature-envelope bounds remain unchanged.

The dispersed gas still uses mean liquid advection plus prescribed slip.
SST does not add a separate gas momentum equation, turbulent bubble
dispersion, interphase drag/lift, or a bubble-induced turbulence source.
Density-gradient buoyancy feedback remains a weak single-continuum model.
No nonlinear IF97 or ALE capability is enabled by this change.

## Diagnostics and comparison

`history.csv` adds mean k, mean omega, maximum nut and maximum wall y+.
The manifest compares the three SST state diagnostics with a declared 10%
relative tolerance and small absolute floors. The wall-y+ column is an
independent bound on each solver, not merely agreement between them.
`turbulence.csv` exports k, omega, nut, wall distance and incident-wall y+
on the same central-y cell plane as the physical fields. Interior cells
have zero wall-y+ diagnostic; zero-reference relative errors remain undefined.
`plot_sst_fields.py` creates separate SST distribution and error galleries.

A paired comparison is accepted only when the existing physical and new SST
criteria all pass. In particular, similar temperature and velocity outputs
alone do not establish quantitative SST agreement. The full-run comparison
and y+ history, rather than startup alone, determine acceptance.

## Focused results and outstanding accuracy, 2026-09-09

GCC Release and OpenFOAM v2606 checks are retained under
`build/verification/sst-convection/`:

- The ten-step, 1,008-cell startup comparison passes all physical and SST
  limits; maximum y+ is 0.0767 (SimpleFluid) and 0.0746 (OpenFOAM).
- Both 1,008-cell full trajectories complete 1,000 steps through 20 s and pass
  the original physical-field/budget limits and wall-resolution checks. Their
  final maximum y+ values are 0.486 and 0.481.
- **The full small-fixture SST comparison fails its new 10% k/nut limits.**
  At 20 s, mean k differs by about 67.5% and maximum nut by about 66.7%.
  The existing 2,080-cell finer fixture does not resolve that discrepancy
  (mean k differs by about 116%). These are retained failures, not accepted
  SST validation. Mean-flow agreement does not remove this limitation.
- The full 790,560-cell mesh passes a five-step comparison on six MPI ranks,
  including all declared SST limits and global budgets. Maximum y+ is 0.0673
  in SimpleFluid and 0.0628 in OpenFOAM. This is startup evidence only.
- The source-off rest and graded-transport CTests pass. The original stationary
  control bound is retained; its turbulence seed is at the positive floor.

The new large SST run must complete its full trajectory and meet all declared
comparison limits before being called an accepted comparison. It will fail
closed if y+ exceeds one. Neither these startup checks nor the retained small
runs establish mesh independence, statistical stationarity, DNS resolution,
or experimental validation of turbulent bubbly convection.
