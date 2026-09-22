@page solver_algorithms Solver workflows

# Solver workflows

The API reference contains hand-authored activity diagrams for the major
implemented solver paths. They document control flow and field publication;
they are not class diagrams and do not replace the equation-level contracts.
Configure `SIMPLEFLUID_DOCS_ENABLE_DIAGRAMS=ON` to render them. Plain
documentation builds retain this overview and the explanatory API text but
omit the images.

## Diagram index

| Diagram | API entry point | Implementation source |
|---|---|---|
| Shared time/pseudo-time lifecycle | @ref SimpleFluid::FluidSolver "FluidSolver" `run()` / `step()` | `src/solvers/FluidSolver.tcc` |
| Coupling dispatch; segregated corrector detail | @ref SimpleFluid::FluidSolver "FluidSolver" `solve_pressure_velocity_coupling()` | `src/solvers/FluidSolver.tcc`, `src/equations/PressureProjectionEquation.tcc` |
| Monolithic assembly and Krylov solve | @ref SimpleFluid::CoupledPressureVelocitySolver "CoupledPressureVelocitySolver" | `src/solvers/CoupledPressureVelocitySolver.tcc` |
| Isothermal flow and optional turbulence | @ref SimpleFluid::IncompressibleIsothermalSolver "IncompressibleIsothermalSolver" `step()` | `src/solvers/IncompressibleIsothermalSolver.tcc` |
| Boussinesq timestep | @ref SimpleFluid::BoussinesqSolver "BoussinesqSolver" `step()` | `src/solvers/BoussinesqSolver.tcc` |
| Boussinesq optional-model phases | @ref SimpleFluid::BoussinesqSolver "BoussinesqSolver" model-update helpers | `src/solvers/BoussinesqSolver.tcc` |
| Adaptive pseudo-transient search | @ref SimpleFluid::AdaptiveSteadyStateController "AdaptiveSteadyStateController" and @ref SimpleFluid::SteadyStateFieldMonitor "SteadyStateFieldMonitor" | `src/examples/pitz_daily.cc`, `src/examples/natural_convection_shiri.cc`, `src/solvers/SteadyStateSearch.cc` |

## Important distinctions

- `FluidSolver::run()` owns the requested step loop. `step()` does not write
  files or perform external coupling; progress output is limited to overloads
  that receive a `ProgressStream`.
- `SIMPLE`, `PISO`, and `PIMPLE` use configured fixed corrector counts. These
  loops are not iterate-until-converged searches. Explicit non-orthogonal
  momentum correctors are distinct from the default implicit treatment.
- `CoupledKrylov` is a monolithic linear solve. Its delegated Belos iterations
  are not the outer nonlinear iterations used by `CoupledNonlinear`/NOX.
- @ref SimpleFluid::IncompressibleIsothermalSolver "IncompressibleIsothermalSolver"
  advances no temperature field. @ref SimpleFluid::BoussinesqSolver
  "BoussinesqSolver" adds buoyancy, temperature transport, and optional
  multiphysics in a defined sequential order.
- Adaptive steady-state orchestration belongs to example drivers. The
  controller adapts pseudo-time and counts accepted physical-update samples;
  it does not own a solver loop or general rollback checkpoint.
- Planar ALE has a specialized geometry-trial and acceptance loop in
  `BoussinesqSolver::step_planar_ale()`. The top-level Boussinesq diagram shows
  the dispatch boundary instead of compressing that large workflow into the
  fixed-grid multiphysics diagram.

## Maintenance

When a solver changes ordering, loop bounds, acceptance gates, field
publication, optional-model timing, or ownership of retry/rollback behavior,
review the associated diagram in the same change. Read the `.tcc`
implementation and driver code before updating a diagram; declarations and
textbook algorithms are not sufficient evidence of implemented behavior.
