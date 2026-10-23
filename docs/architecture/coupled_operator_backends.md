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
pressure diffusion plus `dt * sum(D_i * H_i)`, where H is the configured boundary-aware
least-squares or Gauss-linear gradient. H, affine gradient constants, and gradient stencils
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
checks pressure boundary types/values, reference density and gradient scheme. Numeric assembly
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

## Numerical consistency and qualification

The coupled-only implementation covers all four combinations of assembled or
block-composite operators and cached or streamed products. Assembled/cached
remains the default. Scalar face-based operators, independent diagonal
preconditioning and broader device/performance work remain separate.

MueLu uses `RP` numeric refresh: retain transfers and rebuild coarse numerical
coefficients and smoothers when coefficients change. `full` reuse can retain
stale numeric state and is not equivalent. Live retained operator generations
remain immutable. `NumericUpdatesMatchFreshAssemblyAtScale` compares updated
operator, RHS, stabilization and Schur actions with fresh assembly across four
generations and every backend/workspace combination in serial/MPI.

Timesteppers pass explicit continuity targets, including zero, so coupled
assembly and reconstructed flux use the same distance interpolation. Historical
direct-call arithmetic overloads keep their contract. Stabilization uses the
selected boundary-aware least-squares or Gauss-linear gradient; changing that
scheme invalidates setup. Unknown or rank-divergent selections reject before
assembly. Independent pressure-row actions must agree with reconstructed flux
balances under nonzero boundary data and physical density.

`testCoupledSolverBackends` covers base, isothermal, kinematic and physical
Boussinesq paths over four timesteps, changing timestep and switching backend.
It compares fields, physical pressure, flux, temperature and independent cell
continuity. Native Cartesian/cylindrical, periodic, partitioned unstructured,
legacy and compact composite paths have serial/MPI coverage. Direct
semi-structured handles retain their serial-only limit. Composite cases include
coarse/fine, translated periodic, mixed HEX/prism and common affine axial ALE.
ALE gates retain GCL, mass/volume closure, exact normal mesh flux and rollback.
Existing turbulence/gas coupling tests retain their physical budget checks.

Composite production solves must build no monolithic matrix. Streamed setup
retains at most one application-owned product, then releases it; this counter
does not include Trilinos internals. Multivector checks cover aliasing,
alpha/beta, selected views, maps, noncontiguous/reversed ownership and transpose
rejection. These are CPU numerical regressions, not experimental physical,
accelerator, arbitrary-condition or million-cell qualification.

Use focused `testCoupledBlockOperator`, `testCoupledPressureVelocitySolver`,
`testCoupledSolverBackends` and affected solver/ALE registrations. The
[measurement workflow](../benchmarks/coupled_operator/README.md) compares
separate processes with residual and source/runtime gates. Reduced CRS payload
does not establish an equal RSS reduction or a universal apply-time advantage.
