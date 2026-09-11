# Coupled nonlinear velocity-pressure solver

`PressureVelocityCoupling::CoupledNonlinear` selects the optional NOX backend.
The first supported driver is the native, constant-kinematic-viscosity
`FluidSolver<DefaultTpetraTypes>`. Existing SIMPLE, PISO, PIMPLE, and
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
viscosity, prescribed Dirichlet/NoSlip velocity and Neumann pressure on
physical boundaries, and mesh periodic interfaces. The selected native
least-squares or Gauss-linear pressure reconstruction is retained.

Built-in physical/material isothermal and Boussinesq solvers, custom momentum
subclasses, legacy mesh drivers, pressure outlets, extrapolated/slip velocity
boundaries, nonorthogonal geometry, and moving-mesh contexts are unsupported.
They are rejected rather than silently losing their momentum sources or
stress terms. Temperature, turbulence, gas, material feedback, and ALE are
future extensions requiring their own trial residual and state contracts.

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

## Validation and remaining qualification

Focused tests in `testCoupledNonlinearSolver` cover immutable history,
residual repeatability, analytic directional derivatives, retained operator
and preconditioner generations, pressure normalization, continuity targets,
gauge-cell conservation, nonlinear method/operator equivalence, scaling,
accepted-step advancement, rejected solves, unsupported scope, and collective
input validation. The same test binary has a two-rank CTest registration.

Local validation on 2026-09-11 used GCC Debug and the selected Trilinos 17.2
installation:

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

The nonlinear equations differ from a single frozen-convection timestep.
Compare converged nonlinear solutions against an equivalent residual and
physical accuracy target, rather than requiring bitwise identity with one
`CoupledKrylov` sweep. Runtime tests are component and solver regression
evidence; they are not external CFD validation or a performance qualification.

NOX does not remove region traversal costs or establish a stronger Schur
preconditioner. Performance evaluation must measure time per accepted,
equally accurate physical step, including residual evaluations and setup,
with failed solves excluded. No speedup is claimed by this implementation.
