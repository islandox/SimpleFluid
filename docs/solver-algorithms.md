@page solver_algorithms Solver workflows

# Solver workflows

The API reference contains hand-authored activity diagrams for the major
implemented solver paths. They document control flow and field publication;
they are not class diagrams and do not replace the equation-level contracts.
Configure `SIMPLEFLUID_DOCS_ENABLE_DIAGRAMS=ON` to render them. Plain
documentation builds retain this overview and the explanatory API text but
omit the images.

The major flow-chart phases also expose cumulative, rank-local wall timings
through the solver API. See [solver phase timings](solver_timings.md) for
`Teuchos::Time` instrumentation, caller queries, and rollback semantics.

## Diagram index

| Diagram | API entry point | Implementation source |
|---|---|---|
| Shared time/pseudo-time lifecycle | @ref SimpleFluid::FluidSolver "FluidSolver" `run()` / `step()` | `src/solvers/FluidSolver.tcc` |
| Coupling dispatch; segregated corrector detail | @ref SimpleFluid::FluidSolver "FluidSolver" `solve_pressure_velocity_coupling()` | `src/solvers/FluidSolver.tcc`, `src/equations/PressureProjectionEquation.tcc` |
| Monolithic assembly and Krylov solve | @ref SimpleFluid::CoupledPressureVelocitySolver "CoupledPressureVelocitySolver" | `src/solvers/CoupledPressureVelocitySolver.tcc` |
| Isothermal flow and optional turbulence | @ref SimpleFluid::IncompressibleIsothermalSolver "IncompressibleIsothermalSolver" `step()` | `src/solvers/IncompressibleIsothermalSolver.tcc` |
| Boussinesq multiphysics timestep | @ref SimpleFluid::MultiphysicsSolverCoupler "MultiphysicsSolverCoupler" `step()` (calls the referenced Boussinesq equation stages) | `src/solvers/MultiphysicsSolverCoupler.tcc` |
| Optional-model phases | @ref SimpleFluid::MultiphysicsSolverCoupler "MultiphysicsSolverCoupler" model-update helpers | `src/solvers/MultiphysicsSolverCoupler.tcc` |
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
  "BoussinesqSolver" adds buoyancy, temperature transport, and turbulence.
  An external `MultiphysicsSolverCoupler` orders optional physics around its
  equation stages.
- Adaptive steady-state orchestration belongs to example drivers. The
  controller adapts pseudo-time and counts accepted physical-update samples;
  it does not own a solver loop or general rollback checkpoint.
- Planar ALE has a specialized geometry-trial and acceptance loop delegated by
  `MultiphysicsSolverCoupler::step_planar_ale()` to the private `PlanarALEStep`
  schedule in `MultiphysicsSolverCouplerPlanarALE.tcc`. The top-level coupling diagram shows
  the dispatch boundary instead of compressing that large workflow into the
  fixed-grid multiphysics diagram.
  See [the coupling design](architecture/multiphysics_solver_coupler.md) for
  field ownership and the compatibility API.

## Corrected pressure projection

`PressureProjectionMethod::Anderson` is the default. `FullOperator` applies
the corrected operator `A-C` inside Krylov using a compact pressure matrix
and a reconstructed face-flux correction. `AssembledFullOperator` instead
stores that same operator as a distributed sparse matrix. The assembled mode
requires least-squares pressure gradients and a nonsymmetric Krylov solver
(GMRES or BiCGStab). Both corrected methods solve with `dt*(A-C)` and
independently recompute the corrected residual before accepting pressure.
Both methods skip Krylov when the integral RHS norm meets either the
predictor-scaled tolerance or the absolute roundoff floor. The assembled method
then also skips building `A-C`. The final check uses the same thresholds and
reports the actual residual, even for a zero correction.
Assembled ILUT factors the unscaled `A-C`. Reassembling its coefficients
invalidates those factors even when the graph is unchanged. Other matrix-based
preconditioners use unscaled compact `A`, as does ILUT with matrix-free
`FullOperator`.

With `AssembledFullOperator`, `MueLuSymmetric` uses compact `A` directly for
Dirichlet pressure correction. For a zero-pressure gauge it uses a separate
copy of the compact coefficients on the same graph and maps. The copy keeps
the identity gauge row and zeros the off-diagonal gauge column entries,
including on remote ranks, while preserving neighboring diagonals. The
assembled solve and compact residual still use the original coefficients.
Changes to compact coefficients or its graph invalidate the copy. RHS-only
changes can reuse preconditioner setup when enabled.

Assembly factors the operator into reconstruction and flux balance. The first
factor maps each scalar pressure to itself and the three components of its
homogeneous-increment gradient. The second combines the compact pressure row
with the signed, owner-published face corrections. Sparse multiplication imports
remote reconstruction rows, including pressure columns beyond the immediate
cell halo. The pressure gauge row receives no nonorthogonal correction.

Adjacent PISO corrections on the same trial geometry share the matrix: changes
to the predictor, target or prescribed flux value update the RHS, not its graph.
Motion and rollback rebuild the two temporary factors and refresh values in
the final product graph. The equation retains the reconstruction map and
face-import plan for its mesh and partition; a changed fixed-flux boundary
mask can require a new product graph. The assembled stencil is wider than the compact stencil,
so compare setup plus all solves on a geometry, not just time per matrix apply.

## Maintenance

When a solver changes ordering, loop bounds, acceptance gates, field
publication, optional-model timing, or ownership of retry/rollback behavior,
review the associated diagram in the same change. Read the `.tcc`
implementation and driver code before updating a diagram; declarations and
textbook algorithms are not sufficient evidence of implemented behavior.
