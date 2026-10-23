# Coupled operator measurement workflow

The current benchmark additionally accepts optional mesh, output and size
controls after its original arguments:

```sh
coupled_operator_memory assembled 21 cached_products native 1 output-native 24
coupled_operator_memory block_composite 21 streamed_products isoregion 2 output-region 24
```

These examples use 10,584 cells (21×21×24). Use `45` and `48` for 97,200 cells.
The optional final argument selects a common iteration budget (default 400),
while the true-residual tolerance remains `1e-9`. The full positional form is
`backend nx_and_ny [workspace] [native|isoregion] [regions] [output_dir] [nz] [max_iterations]`.
Increasing Z beyond X/Y and choosing a Z cell count divisible by the MPI rank
count makes native slab partitioning match canonical IsoRegion ownership.
Physical patch names remain unchanged across the mesh
representations.

Per-rank CSVs retain coordinates, solution iterates, RHS and the action of a
deterministic coordinate-based probe for each reached generation. Failed
iterates are retained for diagnosis; they are not accepted solutions.
Checkpoint JSON adds PSS, private-dirty memory and compact-mesh state. Sampled
PSS is not a continuous peak measurement, and CRS payload is only part of
process memory. The logged `block_size` is the requested Belos setting; the
installed adaptive-block behavior can reduce it to the actual RHS count.

## Measurement and acceptance contract

Run one backend/workspace choice per fresh process with identical physics,
precision, tolerances, iteration budgets and warmup. The fixture assembles,
warms application, solves and performs three numerical timestep-coefficient
updates from reset fields. It is an assembly/linear-solve benchmark, not a
time-accurate physical transient. Coupled MueLu uses `RP` numeric refresh;
stale `full`-reuse pilots are not a reference for the corrected implementation.

CRS payload counts unique graph/value view storage, excluding maps/importers,
preconditioner/Krylov internals and allocator metadata. Report composite scratch
separately. Maximum rank peak RSS, synchronized current RSS sums and sampled PSS
measure different things; summing independent rank peaks is not simultaneous
application memory. Separate preconditioner allocation/time instrumentation is
not supplied by this benchmark. Composite application may cost more even when
stored CRS payload is smaller.

Recorded bounded CPU runs completed initial solves and three updates for all
four choices at 512 and 1,728 cells, including two/four ranks at the larger size.
The [backend design](../../architecture/coupled_operator_backends.md#numerical-consistency-and-qualification)
owns numerical support and qualification limits. These observations establish
neither million-cell/device behavior nor a universal runtime or RSS benefit.

```sh
cmake --build --preset GCC-Debug --target coupled_operator_memory
mpiexec -n 1 build/gcc/bin/Debug/coupled_operator_memory assembled 8 cached_products
mpiexec -n 1 build/gcc/bin/Debug/coupled_operator_memory block_composite 8 streamed_products
```

Capture each invocation to a fresh local JSONL file. Use
`src/benchmarks/summarize_coupled_memory.py BASELINE.jsonl CANDIDATE.jsonl` to
check compatible metadata and reject incomplete runs before comparing results.
Retain source/runtime hashes, exact commands, residuals, iteration histories,
failed pilots and raw process measurements outside tracked design documents.
