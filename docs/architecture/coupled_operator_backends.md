# Coupled operator backends

## Checkout audit

Implementation started on clean `develop`, commit
`57fdfe3370f3d7a18c7e97485437675d79ec19c4`. No applicable AGENTS.md was
present in the repository or ancestor directories. The GCC multi-config
preset uses C++23 and local Trilinos 17.2/OpenMPI. Assembled remains the default.

## Dependency and ownership inventory

The coupled map interleaves `(ux,uy,uz,q)` for each owned cell GID, using
`4*gid+component`; owned GIDs need not be contiguous. The scalar momentum
matrix is shared with the momentum equation cache and Ifpack2 preconditioner.
The assembled coupled matrix duplicates its values three times plus G, D,
and C. G and D are three separate scalar matrices. C contains the direct
pressure diffusion plus `dt * sum(D_i * H_i)`, where H is the boundary-aware
least-squares gradient. H, affine gradient constants, and gradient stencils
live in the static geometry cache; they are distinct from G.

Cached setup retains three D*H products, three inverse-momentum-diagonal
scaled G matrices, three D*scaled-G products, and one inverse-diagonal vector.
The final Schur matrix is `C - sum(D_i * diag(A_m)^-1 * G_i)`; each nongauge
row receives `1e-10 * max(1,max(abs(accumulated row entries)))` on its diagonal.
There is no per-component dropping in this checkout. Products use Tpetra's
distributed multiplication, including remote rows beyond a field's one-hop halo.

The block preconditioner shares momentum/G/Schur handles, owns Ifpack2 one-sweep Jacobi
and MueLu state, and retains pressure, velocity, gradient and packed-result
scratch. Belos owns its problem/Krylov state and can retain the previous
operator through a dormant iteration even after `setProblem(null)`. Solver
solution and residual vectors are additional storage. Matrix column maps and
importers belong to the individual distributed matrices; shared handles do
not imply duplicated payload. The assembled coupled column map is an alias
of its CRS column map.

## Algebra and boundaries

The true action is `A_m*u_i + G_i*q`, `sum(D_i*u_i)+C*q`, with positive G/D
signs as assembled. Pressure is normalized as `q=p/rho_ref`; conversion to
and from Pa stays at the solve boundary. Boundary pressure values/gradients,
fixed fluxes and the integrated continuity target are affine RHS terms.
The least-squares boundary constant contributes `-dt*sum(D_i*h_i)` to that
RHS. No boundary callback runs during operator application.

Pressure Dirichlet detection is collective. If none exists, the global
minimum owned cell GID selects the gauge. Its complete continuity row is
replaced by `q_gauge=0`, without eliminating the pressure column elsewhere.
Generalized-target entry points retain the all-Neumann global flux
compatibility check; historical zero-target overloads keep their existing
permissive behavior. Exact fixed boundary flux and ALE block assembly are
unchanged.

Geometry epochs invalidate numerical state. Static reconstruction additionally
checks pressure boundary types/values and reference density. Numeric assembly
refreshes timestep, material/flux and target data; graph reuse checks momentum
graph and pressure-boundary signature. Fixed-flux changes clear the cache.

## Configuration and application contract

```cpp
SimpleFluid::TimeStepperOptions options;
options.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::CoupledKrylov;
options.coupled_operator_backend = SimpleFluid::CoupledOperatorBackend::BlockComposite;
options.coupled_workspace_policy = SimpleFluid::CoupledWorkspacePolicy::StreamedProducts;
// Pass options to FluidSolver/BoussinesqSolver or direct coupled assemble().
```

Defaults are `Assembled` and `CachedProducts`. String adapters accept exactly
`assembled|block_composite` and `cached_products|streamed_products`, respectively.
The established time-stepping API uses typed options, not a file parser. A
Database caller can pass its string through these adapters. Unknown enum/string
values and rank-divergent selections are rejected before momentum mutation.
Rank zero logs requested/effective operator and workspace on selection changes.
No silent fallback occurs.

The returned `linear_operator` is always populated. For assembled systems it
aliases `matrix`; for composite systems both `matrix` and `overlap_map` are
null. An externally constructed historical system containing only `matrix`
still works with `solve()`. Composite communication uses the existing scalar
blocks and never constructs the coupled CRS column map, graph or values.
The four-entry packing layout is checked against actual scalar and coupled
GIDs. Noncontiguous IDs and arbitrary rank ownership are supported.

The composite implements arbitrary RHS column counts, alpha/beta, zero-alpha
short circuit and zero-beta overwrite without NaN propagation. It packs every
input before writing output, allowing identical and overlapping column views.
Tpetra single-vector views resolve nonconstant-stride columns. Momentum acts
on all velocity/RHS columns in one batch; component temporaries avoid the
linked Trilinos multivector-subview pitfalls. Domain/range and column counts
are validated. Transpose/conjugate-transpose explicitly throw and
`hasTransposeApply()` is false. Per-instance Kokkos scratch is reusable for a
fixed RHS count: ten scalar-cell vectors per RHS (80 bytes/cell/RHS for double).
There are no per-apply host mirrors or cell-sized allocation loops; small
Tpetra column-view wrappers are created per RHS. Applications are sequential,
non-reentrant, like the existing block preconditioner.

The true operator is independent of its existing assembled block
preconditioners. This change does **not** yet generalize `BelosLinearSolver`
to a diagonal provider or add scalar no-CRS Jacobi/transport paths. No new CG
eligibility is implied.

## Generation lifetime and streamed assembly

A returned system and composite operator hold a numerical lease on momentum.
The equation holds only a weak lease. Before normal assembly the coupled
solver checks ownership collectively, allowing its own aliases and Belos's
one dormant operator owner. If a caller retains the system/operator, momentum
assembly allocates a fresh generation and coupled blocks/RHS also remain
stable. Otherwise momentum and compatible blocks are updated normally. There
is no unconditional deep copy. Clearing the coupled cache leaves externally
held leases alive, including when the next assembly uses the equation directly.
The RHS is newly allocated each assembly, so a retained RHS alone is stable.
Public mutable block handles are retained for source compatibility; external
mutation of those handles during a solve is unsupported.

`StreamedProducts` forms one scaled G and one D*scaled-G matrix at a time,
accumulates directly into the final Schur matrix, then releases both. The
inverse diagonal remains one reusable vector. C assembly similarly releases
each D*H product after accumulation. Final Schur compression precedes the
unchanged diagonal regularization. Cached products remain available as the
reference/performance policy. No local-only sparse-product replacement was
introduced. Tpetra still allocates its distributed symbolic/import workspace;
streaming is not zero-workspace multiplication.

Storage statistics deduplicate local CRS value/index/offset views by their
allocation addresses, including shared graph payload. They inspect device
views without forcing new host mirrors. Weak observers count externally
retained generations and discard expired observations during assembly/query;
they do not own the data. Product counters count application-owned temporary
matrices, not allocations internal to Trilinos. Observed persistent matrix
count is 22 for assembled/cached, 21 for composite/cached, and 12 for
composite/streamed in the benchmark fixture. After cache clearing only the
momentum equation's one matrix remains. Maps/imports, static host stencils,
preconditioner internals and Krylov storage are additional allocations.

## Supported scope and remaining gates

The coupled backend shares existing block assembly for native MeshHandle and
legacy Mesh, with physical pressure normalization, material paths, generalized
continuity, exact fixed flux and constrained ALE semantics unchanged. Tests
exercise Cartesian, cylindrical, partitioned unstructured, synthetic
noncontiguous/reversed rank ownership, and remote two-hop products. Composite
ALE heating/rollback adds algebraic coverage to the existing supported model
matrix; it makes no new physical validation claim.

Only the configured Kokkos Serial CPU build has been exercised. Device behavior,
empty-rank contracts beyond existing mesh support, every boundary/material
revision combination, and a complete MPI physical-model matrix are not newly
qualified. Application shape/map errors require the ordinary collective MPI
calling contract. Scalar face-based diffusion/upwind/BE, independent diagonal
preconditioning and scalar unsupported-combination rejection are unfinished.

## Delivery and validation record

Implemented phases 1 and 2 as a coupled-only subset: true composite application,
conditional monolithic allocation, ownership-preserving numeric reuse,
streamed products/direct accumulation into final matrices, typed options,
regressions, and opt-in benchmark. Scalar phase 3 is unfinished. Detailed
allocation accounting and performance gates listed above remain unqualified;
this is not completion of the entire original request.

Changed implementation surfaces:

| Files | Change |
| --- | --- |
| `src/solvers/CoupledBlockOperator.hh` | New block application, map/view/alias contract and scratch accounting |
| `src/solvers/CoupledPressureVelocitySolver.hh`, `.tcc` | Optional matrix/true operator, allocation branch, collective generation ownership, streamed assembly and weak storage observers |
| `src/FVM/NumericAssemblyLease.hh`, `src/equations/IncompressibleMomentumEquation.hh`, `.tcc` | Protect retained momentum generations without normal-path deep copies |
| `src/equations/CoupledOperatorBackend.hh`, `TimeStepperOptions.hh` | Explicit typed operator/workspace selection and strict string adapters |
| `src/solvers/unitTests/testCoupledBlockOperator.cc`, `testBoussinesqSolver.cc`, `testBoussinesqPlanarALE.cc`, `CMakeLists.txt` | Operator, native/legacy, retained-generation, streamed, MPI and ALE tests |
| `src/benchmarks/coupled_operator_memory.cc`, `summarize_coupled_memory.py`, `CMakeLists.txt` | Opt-in separate-process JSONL benchmark and comparison tool |
| `README.md`, `TODO.md`, `MAINTENANCE.md`, this document and `docs/benchmarks/coupled_operator/` | Architecture, remaining scope, exact measurements and captured raw evidence |

The following commands were executed from the repository root. The existing
preset was verified against CMakePresets.json and the live cache; CMake's
source discovery reconfigured the build after new tests were added.

```sh
cmake --build --preset GCC-Debug --target \
  testCoupledBlockOperator testCoupledPressureVelocitySolver \
  testBoussinesqPlanarALE testBoussinesqSolver testTurbulentBoussinesqSolver \
  testFluidSolver testIncompressibleIsothermalSolver coupled_operator_memory -j 4

# After extending the existing legacy/ALE cases and benchmark metadata:
cmake --build --preset GCC-Debug --target testBoussinesqSolver -j 4
cmake --build --preset GCC-Debug --target testBoussinesqPlanarALE -j 4
cmake --build --preset GCC-Debug --target coupled_operator_memory -j 4

ctest --preset GCC-Debug -R '^(CoupledBlockOperatorTest\.|Native/CompositeMeshTest\.|CoupledPressureVelocitySolverTest\.|BoussinesqPlanarALETest.CompositeCoupledHeatingAndRollback|simplefluid_elf_export_boundary$)' --output-on-failure

ctest --preset GCC-Debug -R '^(FluidSolverTest\.|BoussinesqSolverTest\.|IncompressibleIsothermalSolverTest\.|TurbulentBoussinesqSolverTest\.|BoussinesqPlanarALETest\.|PlanarALECouplingModes/|CoupledBlockOperatorTest\.|Native/CompositeMeshTest\.|CoupledPressureVelocitySolverTest\.|simplefluid_elf_export_boundary$)' --output-on-failure

ctest --preset GCC-Debug -R '^(CoupledBlockOperator_[24]procs|BoussinesqPlanarALE_2procs|NativePhysicalBoussinesq_2procs|NativeUnstructuredPhysicalBoussinesq_2procs|NativeUnstructuredCoupledFlow_2procs|CoupledKrylovDirichletContinuity_2procs)$' --output-on-failure

git diff --check
```

Builds and whitespace check passed. The focused gate passed 16/16 (including
ELF exports); the final broader gate passed **109**, with **5 skipped** out of
114 registrations, in 20.34 s. Skips were the rank-required PVTU test, three
native distributed physical cases and the combined two-rank turbulence/gas
case. They are not counted as passes. The selected MPI gate passed **7/7
registrations** in 5.25 s with host networking; the MPI executables retain
applicable serial-only skips. New composite/streamed tests ran at both two
and four ranks. The full repository suite, all MPI registrations, accelerator
runs and large benchmarks were not run.

The first restricted-sandbox MPI launch exited 213 before tests started;
host-network reruns succeeded. An exploratory broad regex also selected a
stale turbulence executable after the options-layout change; rebuilding it
resolved its invalid-backend diagnostic, and its final tests passed. Initial
compilation errors were fixed before these final gates. None of these earlier
attempts is counted as successful validation.

All 12 final successful benchmark processes completed (four backend/policy
combinations at two serial sizes and one two-rank size). Exact measured bytes,
setup/apply/solve times, iteration sequences, RSS caveats, and the failed
512-cell pilot are in the [measurement report](../benchmarks/coupled_operator/README.md).
No commits or pushes were made; the checkout's initial worktree was clean.
