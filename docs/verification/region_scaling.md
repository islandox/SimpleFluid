# Region Performance Measurement Contract

This contract defines the standard methodology for performance benchmarking of multi-region mesh operators, ensuring reproducible assessment of scaling and memory efficiency.

## Benchmarking Methodology

- **Isolated Baselines:** Performance comparisons must use isolated baselines (e.g., `native`, `composite`, `flattened`) to prevent memory contamination and crosstalk between independent operator implementations.
- **Strong Scaling Protocol:** Scaling tests are defined as strong-scaling sweeps (varying region counts $R$ and MPI rank counts $P$ for a fixed global problem size $N$), isolating the overhead of domain decomposition and halo exchange.
- **Standardized Assembly:** Benchmarks measure both the formation of linear operators (e.g., FVM diffusion assembly) and the solve phase (e.g., Belos Krylov solvers). Setup costs (descriptor validation, map creation) must be strictly separated from operational iteration time.

## Performance Requirements

- **Numerical Acceptance:** Any performance benchmark execution must satisfy strict physical conservation and numerical accuracy gates (e.g., true residual convergence $< 10^{-9}$, maximum solution error $< 10^{-7}$). Results that fail these gates are invalid.
- **Resource Tracking:** Benchmarks must report maximum-rank elapsed time, rank-local peak RSS, and structural storage components. Aggregate statistics (e.g., summed rank memory) are secondary to maximum-rank observations.
- **Memory Deduplication:** Storage metrics must explicitly account for shared graph payloads, device-resident fields, and allocator overhead. Deduplication of CRS data is required to provide genuine estimates of the memory footprint.

## Integration Principles

- **Configuration Integrity:** Benchmarks must be executed in a controlled environment with fixed compiler flags, library versions, MPI thread counts, and core bindings.
- **Durable Records:** Performance data (e.g., CSV outputs, JSON manifests) are treated as build artifacts. They are not to be embedded in architectural or modeling documentation.
- **Scope Limitation:** Scaling measurements describe the efficiency of the assembled system under specific fixtures; they are not an absolute guarantee of performance for arbitrary production workloads or configurations.

For workflow instructions, see the repository maintenance documentation.
