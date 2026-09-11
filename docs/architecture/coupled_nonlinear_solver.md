# Coupled nonlinear velocity-pressure solver

`PressureVelocityCoupling::CoupledNonlinear` selects the optional NOX backend.
Supported drivers are the native constant-kinematic-viscosity
`FluidSolver<DefaultTpetraTypes>` and physical `BoussinesqSolver<DefaultTpetraTypes>`
with frozen thermal, material, turbulence, and gas state during the flow solve.
Existing SIMPLE, PISO, PIMPLE, and
`CoupledKrylov` modes retain their algorithms.

## Build and select

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_NOX=ON
cmake --build --preset GCC-Debug --target testCoupledNonlinearSolver
```

The default is `SIMPLEFLUID_ENABLE_NOX=OFF`. Enabling it requires NOX built
with Thyra support, ThyraCore, and ThyraTpetraAdapters from the selected
Trilinos installation. NOX/Thyra implementation types stay inside the solver
library. The public callback interface uses the existing default Tpetra pack.

```cpp
SimpleFluid::TimeStepperOptions time;
time.pressure_velocity_coupling =
    SimpleFluid::PressureVelocityCoupling::CoupledNonlinear;
time.nonlinear.linearization =
    SimpleFluid::CoupledLinearization::AnalyticNewton;
time.nonlinear.maximum_iterations = 20;
time.nonlinear.linear_backend = SimpleFluid::LinearSolverBackend::Gmres;
time.nonlinear.continuity_tolerance = 1e-10; // integrated cell imbalance, m^3/s
time.coupled_operator_backend =
    SimpleFluid::CoupledOperatorBackend::BlockComposite;
time.coupled_workspace_policy =
    SimpleFluid::CoupledWorkspacePolicy::StreamedProducts;

// mesh is a native MeshHandle<DefaultTpetraTypes>.
SimpleFluid::FluidSolver solver(mesh, boundaries, time, linear_options);
solver.step();
const auto& nonlinear = solver.last_nonlinear_result();
```

`coupledNonlinear`, `CoupledNonlinear`, and `couplednonlinear` are accepted by
the existing coupling string adapter. Configuration uses typed options; this
change does not introduce a separate input-file parser.

## Supported equations and boundaries

The initial mode supports fixed orthogonal native finite-volume meshes,
backward Euler, first-order upwind convection, constant nonnegative kinematic
viscosity, prescribed Dirichlet/NoSlip or axis-aligned Slip velocity and
Neumann pressure on physical boundaries, and mesh periodic interfaces. The
selected native least-squares or Gauss-linear pressure reconstruction is retained.

Built-in physical/material isothermal solvers, custom momentum
subclasses, legacy mesh drivers, pressure outlets, extrapolated/oblique-slip velocity
boundaries, nonorthogonal geometry, free-surface, boiling and precursor models,
and moving-mesh contexts are unsupported.
They are rejected rather than silently losing their momentum sources or
stress terms. A monolithic nonlinear temperature/turbulence/gas solve and ALE
require separate trial residual and state contracts.

The physical Boussinesq extension copies accepted temperature, density,
effective dynamic viscosity, turbulent kinetic-energy gradient, and boundary
viscosity into the private step context. Its affine residual and Picard block
operator use the existing physical Boussinesq assembly, including density or
thermal buoyancy, turbulent pressure, and the native explicit deviatoric
transpose-gradient stress. That stress remains evaluated from accepted velocity;
this preserves the established split discretization. Material and turbulence
coefficients remain fixed through all trial residuals and linearizations.
Only advection and the pressure-weighted flow flux become nonlinear in `(u,p)`.

The driver refreshes accepted-state properties and sources once before flow,
then advances turbulence, heat, and gas in their existing order once after an
accepted flow solution. NOX callbacks never invoke these updates. A rejected
flow preserves primary fields, time, and accepted solver diagnostics and
restores material, material-feedback, gas, and void-fraction snapshots.
Source/property updater callbacks may have caller-owned side effects and are
not replayed or undone by residual evaluations. A later failure in the native
post-flow model updates prevents retrying the partially advanced driver; this
extension does not claim an atomic transaction over every multiphysics model.

Axis-aligned slip faces retain the native kinematic momentum treatment and
have identically zero normal flux for every trial and directional state.
This supports the original one-layer Re=100 and Re=1000 cavity fixtures
without changing their mesh or wall conditions. See the
[cavity launcher](../../verification/openfoam/cavityFlow/README.md) for NOX
selection and profile export. Oblique slip is rejected because its normal
projection does not share this exact zero-flux property in floating-point
arithmetic.

The standalone `CoupledNonlinearProblem` also accepts a positive reference
density and an immutable `VolumeContinuityTarget`. All-Neumann compatibility
is checked through the existing explicit target assembly path.

## Residual and analytic action

The state preserves the coupled global map, with entries
`4*cell_gid + {0,1,2,3}` representing `(ux,uy,uz,p/reference_density)`.
Public pressure is in Pa. Distributed owned cell IDs need not be contiguous.
`pack_initial()` removes the all-Neumann gauge pressure from the private initial
guess before division by reference density. This eliminates arbitrary common
pressure offsets without changing accepted fields or projecting residual trials.

At construction the problem copies accepted velocity, pressure, boundary
values, coefficients, and continuity target. Native zero-convection assembly
provides the immutable transient/diffusion/pressure affine operator and RHS.
The accepted velocity enters the backward-Euler history exactly once.
Residual callbacks add native upwind advection evaluated with trial velocity
and the existing pressure-weighted Rhie-Chow flux. They create no coupled
matrix, Schur products, or preconditioner.

Physical continuity is evaluated from the same reconstructed face flux:

\[
R_{c,c}=\sum_f s_{cf}\phi_f(u,p)-Q_{V,c}.
\]

The algebraic pressure-gauge row remains `q_g = 0`. The physical continuity
check includes the gauge cell independently of that row replacement.

`Picard` uses the native frozen-flux block operator as an approximate
Jacobian. `AnalyticNewton` adds the flux derivative to that operator:

\[
\delta(\phi u_{up})=\phi\,\delta u_{up}+u_{up}\,\delta\phi.
\]

Directional Rhie-Chow flux uses homogeneous directional boundary values.
There are no finite-difference residual calls in production Jacobian actions.
The donor branch is frozen within a linearization and recomputed for each
trial state. At exactly zero interior flux the tangent uses the canonical face
owner as the donor on both adjacent rows, preserving conservation. The native
zero-flux residual is unchanged; quadratic convergence is not claimed at a
switching point.

Both assembled and block-composite representations use the existing native
Schur preconditioner. Cached and streamed product policies remain separate
from nonlinear method selection. Retained operators and preconditioners keep
their numerical generation while later linearizations are built. Instances
and their reusable operator scratch are sequential-use objects.

## NOX and acceptance

`NOXNonlinearSolver` owns its callbacks, retained linearization, native linear
solver state, and reusable physical/scaled vector workspaces through a private
implementation. `solve()` resets nonlinear validity and convergence baselines
for each problem; `set_callbacks()` replaces the problem while retaining
compatible storage. `FluidSolver` retains one instance across physical steps.
The free `solve_nox()` helper provides a one-shot convenience interface.

The native problem registry retains a `CoupledNonlinearWorkspace` with two
storage slots. A new timestep copies its history and coefficients into a slot
only after the previous context releases it. Retained callbacks keep their
original state; retained Jacobians and inverses keep their numerical leases.
Extra live contexts remain valid without growing the pool beyond two cached
slots. Boundary or geometry changes invalidate dependent caches; timestep
and material changes refresh native numeric values and RHS terms. The affine
residual workspace uses `ResidualOnly` assembly, retaining stabilization and
the gauge without building Schur products. Continuity is accumulated with
advection in cached face order; gate and commit checks use the cached rows too.

Before requesting another Jacobian, NOX releases its retired internal
operator/preconditioner references while retaining vector and Krylov storage.
This permits native graph reuse and MueLu `RP` numeric refresh. External
references still prevent in-place mutation of their generations.

```cpp
SimpleFluid::NOXNonlinearSolver nonlinear_solver(problem.callbacks());
auto x = problem.pack_initial();
auto result = nonlinear_solver.solve(*x, nonlinear_options, linear_options);
// Rebind to another immutable timestep context on a compatible map.
nonlinear_solver.set_callbacks(next_problem.callbacks());
```

A private Thyra model presents the scaled equations
`Fhat(y) = Dr F(Dx y)` and `Jhat = Dr J Dx`. State scales apply to velocity and
normalized pressure. Residual scales apply to volume-normalized momentum and
continuity; the gauge uses normalized pressure scale. All scales are positive,
fixed for a solve, and checked collectively. NOX's in-place weighting is unused.

NOX uses Newton directions with Type 2 forcing. The private Thyra linear
solve adapter uses native Belos GMRES or BiCGStab and a right Schur
preconditioner transformed consistently with the model scales. It checks the
explicit scaled correction residual `norm(Jhat*d + Fhat)/norm(Fhat)` against
the requested forcing tolerance. `Rescue Bad Newton Solve` is disabled.
`krylov_restart` is independent of the four physical components.
`linear_backend` optionally selects GMRES or BiCGStab for nonlinear corrections
while leaving the segregated scalar transport policy unchanged. An unset value
inherits the supplied linear solver policy; PCG is rejected for this saddle-point
correction system, and effective controls are checked collectively.

`preconditioner_update` defaults to `EveryLinearization`. The opt-in
`PerTimeStep` policy retains the first inverse while assembling current
Jacobians without repeated Schur products. It refreshes the inverse when
the scaled residual fails to decrease by `preconditioner_stagnation_ratio`
(default `0.9`), or when no matching evaluated state is available. Only this
policy incurs the residual-state copy and norm reduction for its decision.
The preconditioner stays fixed within each linear solve; a failed explicit
forcing check still fails the solve.

Custom Armijo backtracking evaluates actual trial residuals. Invalid trials
are rejected, and exhausted backtracking fails with no recovery step.
Convergence requires separate absolute/relative RMS checks for all four
interleaved components and an independent physical continuity/gauge gate;
a small update alone cannot signal convergence.
The timestep driver then reevaluates the final residual and checks the
maximum dimensional all-cell continuity error.

`solve_nox()` changes its supplied state vector only on convergence. The
physical driver keeps all trials private and publishes velocity, pressure,
and projected flux only after final checks. Failed solves leave accepted
fields, time, step index, and accepted diagnostics unchanged. A successful
step advances time and the step index once. Existing momentum diagnostics
remain velocity-update norms; nonlinear equation diagnostics are separate.

`last_nonlinear_result()` reports nonlinear iterations, linear solves, Krylov
iterations, residual evaluations, line-search rejections, scaled residual,
and achieved linear tolerance. Direct problem statistics count residual
evaluations and linearizations. Existing coupled cache statistics remain
available through the native coupled solver interfaces.
The backend also reports maximum-rank total, residual-callback, linearization,
and linear-solve elapsed times. These include nested work and independently
maximized rank times, so they are not an additive exclusive phase breakdown.
Problem construction and the driver's final checks lie outside backend time.
The driver reports `native_setup_seconds` separately, together with counts
for native workspace builds/reuses, face-geometry builds, coupled operator
builds, coupled graph reuses, Schur builds, preconditioner builds, and numeric
preconditioner refreshes. Counts and durations are maximum-rank values per
accepted step; operator counts do not enumerate every scalar graph or AMG
allocation. These diagnostics are included in verification nonlinear CSVs.

## Validation and remaining qualification

Focused tests in `testCoupledNonlinearSolver` cover immutable history,
residual repeatability, analytic directional derivatives, retained operator
and preconditioner generations, pressure normalization, continuity targets,
gauge-cell conservation, nonlinear method/operator equivalence, scaling,
accepted-step advancement, rejected solves, unsupported scope, and collective
input validation. The same test binary has a two-rank CTest registration.
Physical Boussinesq tests additionally compare frozen-state residuals against
native physical assembly with nonuniform density and viscosity, verify the
analytic derivative and deep-copy isolation, check heat/model advancement once
per accepted step, and exercise material/gas restoration and retry after a
rejected nonlinear flow.

The initial implementation was validated on 2026-09-11 using GCC Debug and
the selected Trilinos 17.2 installation, before the physical Boussinesq
extension. That historical matrix was:

| Check | Result |
| --- | --- |
| NOX enabled, serial nonlinear tests | 26 passed; 2 MPI-only tests skipped |
| NOX enabled, two MPI ranks | All 28 logical tests passed on both ranks |
| NOX disabled, serial nonlinear tests | All 14 tests passed |
| Existing coupled-backend regressions | All 40 tests passed |
| Existing direct coupled, base-fluid, and isothermal tests | 5, 16, and 4 passed |
| Shared-library ELF export boundary | Passed |

Both enabled and disabled configurations built successfully. The enabled
tree is `build/gcc`; the independent disabled tree is `build/nox-disabled`.
The two-rank run used host networking because the sandbox cannot provide
the PMIx launcher sockets. No unrelated project-wide suite or performance
campaign was run.

The subsequent physical Boussinesq extension has a separate
[Release validation and 10k/100k performance report](../benchmarks/coupled_nonlinear_20260911.md).
It includes 35 nonlinear tests exercised across serial/MPI and matched physical
convection runs; the initial matrix above is not a claim that every old check
was repeated for the extension.

The nonlinear equations differ from a single frozen-convection timestep.
Compare converged nonlinear solutions against an equivalent residual and
physical accuracy target, rather than requiring bitwise identity with one
`CoupledKrylov` sweep. Runtime tests are component and solver regression
evidence; they are not external CFD validation or a performance qualification.

NOX does not remove region traversal costs or establish a stronger Schur
preconditioner. Performance evaluation must measure time per accepted,
equally accurate physical step, including residual evaluations and setup,
with failed solves excluded. No speedup is claimed by this implementation.
