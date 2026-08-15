# Benchmark regression baselines

The `debug-small` CTest gates load these checked-in CSV files. Each row
identifies one solver configuration and sets ceilings for nonlinear
iterations, linear solves, Krylov iterations, and total wall time.

Iteration ceilings are padded above the reproducible reference counts so
minor floating-point differences between supported compilers do not fail the
gate. The ten-second wall-time ceiling is intentionally loose: it catches a
runaway tiny case without treating timings from heterogeneous CI runners as
directly comparable.

Update a ceiling only with an intentional solver change and retain the before
and after benchmark CSVs in the pull-request evidence. Do not raise timing
ceilings to mask launcher or shared-runner failures.
