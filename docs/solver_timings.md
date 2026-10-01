# Solver phase timings

`FluidSolver`, `BoussinesqSolver`, and `IncompressibleIsothermalSolver` expose `solver_phase_timings()` (an owned
snapshot of all stable entries), `solver_phase_timing(SolverPhase)` (one entry),
and `reset_solver_phase_timings()`. Each entry has `name`, `calls`, `seconds`, and
`depth`. Unused phases have zero calls and seconds. Timings start at construction
and accumulate until explicit reset. Reset throws `std::logic_error` while a
scope is active, including calls from a solver callback.

Scopes use native `Teuchos::Time::start/stop` timing (the scope constructor starts
the timer), with independent timer instances and no global TimeMonitor registry.
They record rank-local wall seconds without barriers or reductions. An entry is
updated when its scope ends, including exception unwinding. Snapshots exclude
unfinished scopes. Queries/reset require the same serialized access as the
solver. Diagnostic timings are outside physical snapshots: restoring a coupling
checkpoint, rejecting an ALE trial, and replaying a Picard attempt retain actual
attempted work and subsequent attempts accumulate it.

Times are **inclusive**: `sf.step` includes all work in a physical step, `sf.ale`
includes the complete planar ALE transaction, and `sf.ale_trial` includes an outer
corrector attempt. `sf.ale_geometry` covers moving the trial geometry, creating
control-volume state, and refreshing geometry-dependent caches. Pressure/velocity
coupling includes each momentum predictor and pressure correction for segregated
methods; coupled Krylov/NOX work belongs to `sf.pressure_velocity` and does not
pretend to have separate segregated solve times. Physical model refresh,
turbulence, pre-temperature models, temperature transport, post-temperature
models, radiolytic gas advance, and fixed-grid free-surface advance follow the
existing flow charts in `FluidSolver.hh`, `BoussinesqSolver.hh`, and
`IncompressibleIsothermalSolver.hh`. Isothermal steps record flow and turbulence
without a temperature phase. The collector cannot be copied or moved while
scopes retain references to it.

`depth` describes the logical flow chart for display; it is not a runtime nesting
trace. Gas can run under pre/post-temperature models or an ALE trial, and shared
pressure/temperature stages can run inside an ALE trial. Therefore, do not add
parent and child values or subtract logical children to infer exclusive time.
In particular, `sf.pressure_project` also includes ALE continuity-refinement
corrections outside `sf.pressure_velocity`; its accumulated time can exceed
that row. Phase timings describe all calls to that stage, not one call path.
The API deliberately reports inclusive time only. Aggregate across ranks in the
caller at a safe synchronization point if required; the timer scope destructor
never performs a collective.

```cpp
for (const auto& phase : solver.solver_phase_timings())
    if (phase.calls != 0)
        std::cout << phase.name << ' ' << phase.calls << ' ' << phase.seconds << '\n';
```
