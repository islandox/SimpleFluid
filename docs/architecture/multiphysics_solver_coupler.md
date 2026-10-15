# Multiphysics solver coupling

`MultiphysicsSolverCoupler<Pack>` is an external timestep controller constructed
with shared ownership (`SP<BoussinesqSolver<Pack>>`, an alias of
`std::shared_ptr`) of an existing solver. It calls the solver's flow and temperature stages and owns optional physics, ALE trials, diagnostics,
and interval checkpoint replay. `BoussinesqSolver` owns no coupler and has no
model-configuration forwarding APIs.

```cpp
auto solver = std::make_shared<BoussinesqSolver<Pack>>(
    mesh, boundaries, time, linear, material);
MultiphysicsSolverCoupler<Pack> coupling(solver);
coupling.add_prescribed_heat_source().initialize_constant(power_density);
coupling.configure_reacting_gas(gas_options);
coupling.configure_material_feedback(feedback_options);
coupling.initialize_heated_box(initial_temperature, initial_temperature);
coupling.configure_free_surface(surface_options); // Configure after dependencies.
coupling.step();
const auto& temperature = solver->temperature();
```

The coupler retains a strong shared owner, so the caller may release its solver
handle while coupled stepping and checkpoint replay continue. The solver is
destroyed when the coupler and every other shared owner release it. A null
solver is rejected. The coupler cannot be copied, moved, or rebound, and a
generic lifetime lease permits only one external timestep controller per solver. While that lease is active, direct `solver->step()` or
`solver->run()` rejects advancement rather than bypassing coupled physics.
Destroying the coupler releases the lease and its shared solver owner, without
initiating rollback or timestep collectives. Standalone stepping is then
available again if another shared owner retains the solver. Coupled applications
call `coupling.step()` or `coupling.run()`, including their progress-output
overloads.

## Ownership and equation inputs

The solver retains its mesh, primary fields, material storage, turbulence,
momentum and temperature equations, pressure/velocity algorithms, generic
heat-source registry, and clock. It can independently advance flow, heat, and
turbulence. The coupler owns supplied reaction sources, prescribed heat power, boiling, scalar void,
material feedback, scalar source-decay groups, free-surface mass accounting, planar mesh motion,
and thermal participants. Optional models remain lazily allocated.

The dependency is from the controller to the solver. A private friend interface
allows the controller to call equation stages and preserve existing accepted
state and rollback machinery. Borrowed numeric transport inputs specify the
active ALE control-volume state, relative-flux workspace, conservative thermal
storage fields, and extra heat sink/release fields. These inputs contain no
coupler pointer or model callbacks. Flow inputs are cleared on both success and
failure. Geometry refresh first calls the solver, then refreshes the optional
models owned by the coupler.

The solver's public header does not include the coupler header. Its only named
coupler declaration grants access to the equation-stage interface. Existing
pressure-boundary providers remain the numerical boundary-data seam for ALE;
they do not own or invoke timestep orchestration.

Use `coupling.initialize_*()` to initialize primary fields and dependent model
inventories together. The retained solver remains accessible through
`coupling.boussinesq_solver()`. Coupled output belongs to
`coupling.write_solution_vtu()` and `coupling.write_parallel_solution_vtu()`;
these combine core fields with optional-model fields in their established
order. The accepted ALE Courant diagnostic is
`coupling.maximum_courant_number()`; the standalone solver diagnoses its own
absolute projected flux.

## Advancement and transactions

Fixed-grid advancement preserves collective validation, interval-source
refresh, material/source refresh, flow, turbulence, lagged ideal supplied reaction sources and
boiling/void updates, temperature, Sheng supplied reaction sources and its void mirror,
scalar source-decay groups, material feedback, free-surface accounting, and accepted publication.
SST-SAS and nonlinear-flow rollback retain their existing scopes. The existing
planar ALE geometry-trial Picard loop, relative transport, GCL/quality gates,
energy/mass/gas/continuity acceptance checks, and rejection rollback belong to
the coupler. The supported physics combinations remain those in
[the planar free-surface model documentation](../modeling/planar_free_surface_volume_budget.md).

`step_planar_ale()` initializes the configured owners and delegates to a private
step-local `PlanarALEStep` in `MultiphysicsSolverCouplerPlanarALE.tcc`. Its short
corrector schedule calls separate geometry, flow, temperature, gas, ledger,
convergence, commit, and replay helpers. Accepted snapshots stay alive for the
whole step, and preview objects stay alive through each corrector's commit or
internal replay. The outward catch performs eager rollback; destruction does
not initiate a transaction or MPI work. The implementation is included only by
the compiled coupler `.tcc` and adds no persistent coupler state.

```cpp
auto checkpoint = coupling.create_coupling_checkpoint();
coupling.set_coupling_interval_energy(owned_energy_joules, interval_duration);
coupling.run(substeps);
// Another trial: restore the checkpoint, replace the budget, and replay.
coupling.validate_coupling_checkpoint_acceptance(checkpoint);
coupling.accept_coupling_checkpoint(checkpoint);
```

Checkpoint, ALE diagnostic, and history types now belong to the coupler.
Only one checkpoint may remain live. Restore replays the interval without
duplicating accepted history or gas/energy ledgers. Acceptance verifies owner,
configuration, health, and the complete interval endpoint before releasing the
token. Configuration and transactions retain their mesh-communicator contracts.

## Migration and compilation

Move optional-model configuration, coupled initialization, advancement,
checkpoint operations, coupled output, and ALE diagnostics from the solver to
an explicitly constructed coupler, passing a shared solver pointer instead
of a borrowed C++ reference. Generic field/material/turbulence queries
and heat-source APIs remain on the solver. The source and class-layout changes
require downstream consumers to migrate and rebuild.

The library compiles `DefaultTpetraTypes` through
`explicit_instantiation/MultiphysicsSolverCoupler.cc`; native and legacy solver
construction remain supported. Other packs require compiled instantiations of
the solver, coupler, and their dependencies. Public headers do not include
`.tcc` implementations. This is a Boussinesq backend; arbitrary solver adapters
and additional physics combinations are outside this change.

External-controller migration validation in GCC Debug on 2026-10-15 passed
287 serial/ELF checks, with 19
MPI-only cases skipped in serial, plus 15 two-rank groups, the four-rank
composite ALE regression, and nine bounded example checks. Both public headers
and owning implementation units compiled without PCH. The two optional IF97
example sources passed syntax checks; IF97 and NOX runtime configurations were
not qualified. MPI used host networking. This is focused validation, not a full
suite or performance qualification.

The subsequent ALE phase split passed 82 focused serial/ELF checks, with five
MPI-only serial skips, the two-rank ALE replay suite, and the four-rank composite
ALE check. Its headers and owning implementation units compiled without PCH.
All four verification-driver cases retained byte-identical printed physical
diagnostics compared with the implementation before the split. Private phase
helpers add no exported symbols.

Shared-solver ownership validation passed 289 focused serial/ELF checks, with
19 MPI-only serial skips, six MPI groups including the four-rank composite ALE
check, and nine bounded example checks. The lifetime regression releases the
caller's solver handle, advances and replays a checkpoint through the retained
owner, then verifies solver destruction when the coupler releases the last
owner. Null ownership and duplicate-controller rejection are also covered.
