# Multiphysics Solver Coupling Contract

The `MultiphysicsSolverCoupler` framework establishes the architectural boundary between the core CFD solver and auxiliary multiphysics models (e.g., reacting gas sources, free-surface thermodynamics, mesh ALE motion).

## Coupling Architecture

- **Dependency Direction:** The coupler acts as an external timestep controller. It possesses shared ownership of the solver instance. The solver itself is agnostic to the coupler and does not contain any forwarding APIs or model-specific configuration.
- **Lease Mechanism:** A solver accepts exactly one external timestep controller at a time. This lease ensures that coupled physical advancement (e.g., source injection, ALE motion) and core CFD advancement remain synchronized and managed within a single transaction.
- **Modular Physics:** Optional models (boiling, prescribed heat sources, etc.) are injected into the coupling framework lazily. The coupler orchestrates the execution order of these physics models relative to the solver's internal stages.

## Transactional Advancement

- **Atomicity:** The coupler manages advancement as a series of atomic transactions. Each `step()` operation includes transport, source evaluation, pressure correction, and field publication. If any sub-stage (e.g., linear solver convergence) fails, the coupler ensures the solver state is rolled back to the pre-step checkpoint.
- **Checkpoint/Replay:** The coupler supports state snapshots for interval checkpoint/replay. This allows for transactional validation of multiphysics interactions without persistent alteration of the solver's history or diagnostic state until the entire step is accepted.
- **Ordering:** The advancement sequence follows a strict order:
  1. Transport (Advection/Diffusion)
  2. Source evaluation (Chemistry/Heat/Mass Transfer)
  3. Pressure/Continuity Projection
  4. Final Acceptance/Publication

## Integration Constraints

- **Field Consistency:** The coupler mediates between the solver's primary fields and optional model inventories. Data migration between solver and model representations is performed via explicit, authenticated field adapters that preserve physical conservation.
- **Namespace and Header Isolation:** The solver's public header does not include the coupler header. The coupling interface is exposed via friend-accessible equation-stage hooks, maintaining strict encapsulation of solver internals.

For specific physics-coupling contracts, see:
- [Reacting Gas Sources](../modeling/reacting_gas_sources.md)
- [Planar Free-Surface Budget](../modeling/planar_free_surface_volume_budget.md)
