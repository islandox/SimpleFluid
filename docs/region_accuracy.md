# Region accuracy qualification

`testRegionQualification` exercises the existing stored-field FVM operators and
Belos solver on compact `MultiRegionMesh` geometry. Its three tests run in serial
and at two/four MPI ranks. Every case checks that compatibility connectivity and
the materialized indexer remain absent. The scope is spatial accuracy, transport
through translated periodic identification, and numerical invariance under an
artificial conforming region decomposition.

## Nonlinear coarse/fine refinement

The manufactured field is

$$
\phi(x,y)=1+\sin(\pi x)\sin(\pi y),\qquad
(5-\nabla^2)\phi=5\phi+2\pi^2(\phi-1).
$$

The unit square is extruded through one unit-thickness cell. Analytic Dirichlet
conditions apply on the X/Y boundaries; the Z boundaries have zero normal
gradient. The existing nonorthogonal implicit transport assembly represents the
reaction term with `dt=0.2`, fixed analytical old values, unit diffusivity and the
analytical diffusion source.

Each half has `normal_refinement*n` cells in X. The coarse half has `n` cells in
Y and the fine half has `ratio*n`, with `n=4,8,16`. The following cases vary one
geometric property at a time:

| Fine/coarse ratio | Interface X | Normal refinement | Maximum nonorthogonality |
|---:|---:|---:|---:|
| 2 | 0.5 | 1 | 26.565 degrees |
| 3 | 0.5 | 1 | 33.690 degrees |
| 2 | 0.25 | 1 | 26.565 degrees |
| 2 | 0.5 | 2 | 45 degrees |

The angle is measured between the face normal and adjacent cell-center vector.
Changing normal refinement changes that angle while preserving the supported
planar Cartesian tiling. This study does not extend the geometry support envelope.

Each level reports normalized, volume-weighted L2 error over the whole domain and
over the first cell layer on both sides of the interface. The latter band shrinks
with refinement and exposes interface errors that a domain norm can hide. A
conforming ratio-one reference is also solved at every level. It has coarser Y
resolution in the right half; the separate decomposition test below keeps every
physical cell fixed.

The current implicit nonorthogonal assembly places the remote half of an MPI
partition-face gradient on the RHS using `correction_field`. Therefore a single
linear solve is one partition correction. The qualification converges these
corrections to a global maximum update below `2e-12`, while holding the reaction
and source terms fixed. It then rebuilds the RHS using the final gradients and
requires a relative residual below `2e-10`. Correction counts and final residuals
are emitted with the errors. This distinction matters: comparing just one
correction initially produced rank-dependent interface errors despite all linear
solves reporting convergence.

Acceptance requires decreasing global and interface errors on both refinements,
a converged final equation, and global/interface rates above 1.7 on the finest
`8 -> 16` refinement. The broad rate floor is grounded in measured rates near
1.92--2.01 for these cases and catches interface degradation that simple monotonic
error reduction could miss. It qualifies this field, tiling family, treatment and
finite set of levels; it does not establish convergence on every supported mesh
family.

## Periodic advection

The test transports `1 + sin(2*pi*x)` at unit speed around the unit periodic
interval for two complete periods. It uses the existing backward-Euler,
first-order-upwind transport at CFL 0.5 on 32, 64 and 128 cells. Two regions meet
at the middle and through the translated periodic connection. At multiple ranks,
the test asserts that the periodic face joins an owned cell to a remote cell.

Volume-weighted Fourier moments provide amplitude and phase error, and the
volume-weighted scalar sum checks conservation of the nonzero mean. An independent
discrete Fourier symbol supplies the amplitude and phase expected from the
selected discretization:

$$
G=\frac{1}{1+C(1-e^{-i\theta})},\qquad
\theta=2\pi/N,\quad C=0.5,\quad \widehat\phi_{\mathrm{final}}=G^{4N}.
$$

The measured complex coefficient must agree within `2e-9`, inventory within
`2e-10`, and amplitude, phase and global L2 errors must decrease on refinement.
The selected first-order method has substantial expected damping after two
periods; agreement with the Fourier symbol distinguishes that damping from
periodic-interface errors.

## Fixed physical mesh, different regions

A fixed `16 x 4 x 2` Cartesian grid is represented with one, two and eight
regions. The scalar reaction-diffusion operator, converged scalar solution,
production Rhie-Chow face fluxes and resulting cell flux-divergence arrays are
compared by physical cell/face coordinates. Unique owned values are reduced
across MPI ranks and face signs are normalized to positive coordinate directions.

The comparison tolerances are `2e-12` for operator action, `2e-10` for solution and
integrated face flux, and `2e-8` for divergence. This establishes invariance of the
flux and continuity operators under region decomposition. Incompressible flow
convergence and its continuity tolerance remain covered by the existing coupled
solver regression tests.

## Qualification status

Recorded GCC Debug/Release and LLVM Debug campaigns passed these three
fixtures in serial and at two/four ranks, including independent runner parity
checks. Finest-grid global/interface rates exceeded the stated 1.7 gate,
periodic inventories and Fourier coefficients met their bounds, and region
decomposition preserved the fixed physical operator and fluxes. These are
bounded numerical observations, not a general mesh-family convergence proof
or a fresh test run during documentation consolidation.

## Reproduction

Build only the focused target using the chosen existing toolchain environment:

```sh
export SIMPLEFLUID_COMPILER=GCC
export SIMPLEFLUID_BUILD_CONFIG=Debug
source verification/environments.sh
export_build_env "$PWD"
simplefluid_build_target testRegionQualification 4
python3 verification/run_region_accuracy.py \
  --executable build/gcc/bin/Debug/testRegionQualification \
  --output /tmp/region-accuracy-gcc-debug
```

Use `SIMPLEFLUID_BUILD_CONFIG=Release` with the corresponding Release binary, or
`SIMPLEFLUID_COMPILER=LLVM` with `build/llvm/bin/Debug/testRegionQualification` for
the second compiler. Run MPI commands where OpenMPI is permitted to create its
local communication sockets. The runner executes only these three tests at one,
two and four ranks, retains full logs and writes `accuracy.json` with the binary
SHA-256, timestamp and diagnostics. It also requires serial/MPI global/interface
errors and periodic amplitude/phase/L2 diagnostics to agree within `1e-9` absolute.
Correction counts and residuals can differ across ranks and are reported separately.

CTest also registers `RegionQualification_2procs` and
`RegionQualification_4procs`; the Python runner adds comparison between the
separate serial/MPI processes. See [region scaling](region_scaling.md) for the
independent construction, execution and memory measurements.
