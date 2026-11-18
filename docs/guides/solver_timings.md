# Solver Performance Timing Contract

The solver timing framework provides rank-local, inclusive wall-clock measurements for solver phases, facilitating performance profiling and regression analysis.

## Timing Architecture

- **Inclusive Scopes:** Timers record inclusive wall-clock seconds for each solver stage (e.g., transport, projection, ALE transaction). Nested stages are included in the cumulative time of their parent phases.
- **Independence:** Timers operate independently of the solver's physical state. Checkpoint restoration, trial failures, and rollbacks are explicitly tracked as attempted work, preserving the fidelity of the performance data.
- **Snapshot Atomicity:** Timings are reported as snapshots. Resetting or querying requires serialized access relative to the solver's execution state to ensure consistency across the communicator.

## Operational Requirements

- **Instrumented Stages:** Primary phases (e.g., flow, turbulence, ALE) are instrumented via native scope-based timers. New stages must adhere to the inclusive timing pattern, ensuring that diagnostic times are consistent with the established flow-chart hierarchy.
- **Non-Collective Traversal:** Timing records are inherently rank-local. Aggregation of global performance metrics must be performed by the caller at a safe synchronization point; timers never perform MPI reductions.
- **Diagnostic Isolation:** Diagnostic timings (e.g., checkpoint restoration) exist outside physical snapshot validity. This allows developers to measure the overhead of transactional rollbacks and replay attempts independently of physical time.

## Integration Principles

- **Performance Qualification:** Comparisons of performance variants must use identical configurations (meshes, MPI ranks, numerical gates, solver backends). The timing snapshot should be assessed over an identical physical horizon to validate efficiency gains.
- **Diagnostic Usage:** Phase timings are intended for performance characterization. They should not be utilized as control mechanisms for the solver's physical advancement or as a substitute for explicit convergence gates.

For usage examples, see the instrumentation headers in `src/solvers/`.
