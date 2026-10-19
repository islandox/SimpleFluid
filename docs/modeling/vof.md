# Uniform-mesh volume of fluid

`VoFSolver` is the Stage 1 flow core of the resolved/dispersed reacting-bubble
roadmap in [TODO.md](../../TODO.md). It advances two incompressible, isothermal
phases on a fixed uniform Cartesian grid. Gas volume fraction `f=1` denotes gas;
`f=0` denotes liquid. Density differences enter momentum storage, transport, and
pressure projection. The existing Boussinesq solver remains a separate backend.

This stage supports serial execution, periodic coordinate directions, and
impermeable slip walls. A periodic direction requires at least two cells;
a nonperiodic one-cell direction can represent a two-dimensional verification
case. The solver rejects multi-rank construction before creating distributed
mesh or field state. The separate [adaptive Cartesian backend](adaptive_vof.md)
provides distributed AMR while retaining native fields. Gas compressibility,
temperature, reaction kinetics, dispersed populations, contact angles, and phase
change have separate roadmap acceptance gates.

## State and units

`VoF::UniformGrid` specifies the cell counts, origin in metres, spacing in
metres, and periodic directions. Cells use x-fastest indexing. A face array
normal to axis `a` has `cells[a]+1` entries along that axis and cell counts along
the other axes. Its velocity is positive in the positive coordinate direction;
duplicated periodic end faces carry the same value. Nonperiodic end faces have
zero normal velocity.

`VoFMaterial` supplies constant liquid/gas density in kg/m3, dynamic viscosity
in Pa s, and surface tension in N/m. Density is strictly positive; viscosity
and surface tension may be zero. Material defaults are illustrative solver
inputs, not a qualified aqueous-material correlation.

The accepted arrays contain gas fraction, velocity in m/s, gauge pressure in Pa,
and curvature in 1/m. `fraction_field()`, `velocity_field()`, and
`pressure_field()` expose the same accepted results through native
`MeshHandle`/`FieldStored` objects. Face velocity supplies the authoritative
finite-volume transport flux. References remain valid until a successful
initialization or step replaces the accepted snapshot.

Initialize `f` with cell-average geometric occupancy. A cell-center
inside/outside mask omits mixed cells and cannot supply curvature for a curved
surface. The [verification driver](../../src/examples/vof_verification.cc)
contains independent volume-integrated sphere and sinusoidal-interface
initializations; a flat grid-aligned interface correctly has zero curvature.

The constant-property density is

```text
rho = f*rho_g + (1-f)*rho_l.
```

Diagnostics report gas volume, each phase mass, total cell momentum, cell
kinetic energy, maximum velocity/divergence, pressure residual, and linear
iterations. There is no molar or thermal ledger in this source-free stage.
Maximum divergence is measured in 1/s. The reported pressure residual is
`max(abs(div(u_face)))/dt` in 1/s2, not a pressure difference in Pa; its absolute
tolerance uses the same units.

## Interface transport and surface tension

`FVM/GeometricVoF.hh` provides the compiled geometry layer independently of the
solver library. PLIC reconstruction chooses a physical normal pointing out of
gas and an offset matching the cell gas volume. Box/plane intersections provide
the gas volume swept through a face. The complementary liquid flux is computed
from the same total swept volume.

Directional transport uses a frozen Weymouth--Yue compression indicator. For
discretely divergence-free velocity, the directional compression terms cancel,
so the returned integrated gas/liquid face volumes account for the fraction
change. Momentum advection uses the resulting face masses, rather than an
independently interpolated density flux. The geometric layer rejects excessive
Courant number, inconsistent periodic/wall fluxes, or material violations of
fraction bounds. Roundoff-sized excursions are diagnosed without silently
clipping away inventory.

The flow step is first order in time and uses upwind momentum transport.
Geometric interface transport alone does not make the complete flow solver
second order. Spatial and temporal refinement are separate verification axes.

Curvature is computed from the fraction field using height columns where a
complete local interface stencil exists. Three-dimensional height functions
use transverse weights `(0.2,1,0.2)/1.4` on their second derivatives to suppress
an alternating numerical mode, preserving quadratic graphs and the invariant
two-dimensional limit. First and mixed derivatives retain their centered
stencils; see the [reference curvature formulation](https://basilisk.fr/src/curvature.h).
Incomplete columns use a local
quadratic graph fitted to the area centroids of neighboring PLIC polygons.
The fit uses physical tangent/normal scales, periodic image displacements,
and pivoted, reorthogonalized QR with rank and residual checks. A
normal-divergence estimate remains the final fallback when a local graph
cannot be resolved reliably.
The gas-outward convention gives positive curvature for a convex gas bubble.
Pressure and capillary forces use matching face differences. A spherical gas
bubble of radius `R` therefore has the equilibrium target `p_g-p_l=2*sigma/R`.
The discrete curvature error and resulting parasitic motion require numerical
verification; the solver does not accept an analytic bubble radius to supply
its curvature.

The split transport reference is Weymouth and Yue,
[Conservative Volume-of-Fluid method for free-surface simulations on
Cartesian-grids](https://doi.org/10.1016/j.jcp.2009.12.018).
[Basilisk's interfacial-force implementation](https://basilisk.fr/src/iforce.h)
explains the matching pressure/capillary stencil, and its
[momentum-conserving transport](https://basilisk.fr/src/navier-stokes/conserving.h)
illustrates the shared phase-flux principle. These are algorithm references;
the implementation here has its own explicit support and validation scope.

## Time advancement and failure semantics

Initialize gas fractions and either cell or face velocities. Initialization
projects the supplied velocity to a solenoidal face flux. `step(dt)` transports
phase volume and momentum, applies viscous/gravity/capillary forces, and performs
a variable-coefficient pressure projection. Viscosity uses conservative
symmetric stresses. Successful steps publish arrays, native fields, time,
and diagnostics together. Invalid inputs or a failed candidate leave the
previous accepted state intact.

`stable_timestep()` combines the advective Courant limit, explicit viscous
limit, and explicit capillary limit. The capillary bound is proportional to

```text
sqrt((rho_l+rho_g)*h_min^3/(2*pi*sigma)).
```

It is not an acoustic timestep or a gas-growth limiter. A motionless,
inviscid, single-phase case can return infinity; callers still provide a
positive finite step. Source/EOS restrictions will be added with those models.
The explicit viscous bound accounts for the larger phase viscosity acting on
the smaller phase inertia across an interface. It can be substantially more
restrictive than a bound formed from the two pure-phase kinematic viscosities.

Pressure projection uses the native Belos algebra with a symmetric pressure
gauge treatment. Up to four residual corrections use an attainable internal
Krylov tolerance and share one pressure-iteration budget. Corrections act on
the actual face-flux divergence and accumulate pressure/force increments,
avoiding repeated subtraction of a large static pressure offset.

Acceptance checks the physical pressure residual, the unpinned finite-volume
continuity residual, and the integrated divergence required by VoF transport.
Those limits retain the configured accuracy regardless of the internal Krylov
stopping flag. Nonfinite right-hand sides, corrections, or linear diagnostics
reject the trial. A finite correction that misses its internal Krylov target
can still contribute to a candidate, but the candidate is accepted only if
all final physical gates pass within the shared iteration budget.

## Numerical verification

The focused targets are `testGeometricVoF`, `testVoFSolver`, and the
`vof_verification` example. Geometry checks cover geometric volume identities,
transport, bounds, and conservation. Flow checks cover native fields,
variable-density projection/transport, failure rollback, static bubbles, and
time-dependent capillary oscillation. A two-rank test checks the explicit
distributed-construction rejection; it does not qualify an MPI flow solution.

```sh
cmake --build build/gcc --config Debug --target testGeometricVoF testVoFSolver vof_verification -j 2
ctest --test-dir build/gcc -C Debug -R '^(GeometricVoF\.|VoFSolverTest\.|VoFSolverParallelTest\.|VoFRejectsDistributedConstruction_)' --output-on-failure
build/gcc/bin/Debug/vof_verification --case sphere --resolutions 12,24 --density-ratio 10000 --end-time 0.025
build/gcc/bin/Debug/vof_verification --case wave --resolutions 24,48 --density-ratio 10 --periods 1
```

The driver writes measured summary quantities as CSV. `--trace` additionally
writes wave history to standard error. Increasing the sphere resolution keeps
the physical horizon fixed; short startup unit tests serve a separate purpose.

For a small-amplitude interface between two inviscid layers bounded by rigid
slip walls, the zero-gravity reference is

```text
omega^2 = sigma*k^3 / (rho_g*coth(k*h_g) + rho_l*coth(k*h_l)).
```

Here `h_g` and `h_l` are the undisturbed layer depths and `k` is the wave number.
The verification must measure the evolving Fourier amplitude or phase, not
only evaluate this formula. The finite-depth dispersion relation is also
given in [Fast capillary waves on an underwater superhydrophobic
surface](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821838/).

Use the focused numerical results to qualify this baseline. They do not by
themselves establish centimetric-bubble rise/wake accuracy, realistic gas
transfer, unresolved-film coalescence, or coupled reactor performance. Those
remain explicit later-stage studies in the roadmap.
