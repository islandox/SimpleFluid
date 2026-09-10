# Steady non-orthogonal diffusion convergence

`FVM::solve_non_orthogonal_diffusion()` now checks the full corrected equation
for `Implicit` treatment on both `CellField` and mapped `FieldStored` overloads.
The existing matrix/RHS split is retained: owned-cell gradient stencils enter
the matrix, while the remote half of a partition-face gradient is reconstructed
from the current synchronized solution and placed on the RHS.

The optional final `NonOrthogonalConvergenceOptions` argument controls this
iteration. Existing call sites continue to compile with the defaults:

```cpp
FVM::NonOrthogonalConvergenceOptions corrections;
corrections.max_iterations = 40;       // includes the initial linear solve
corrections.update_tolerance = 1e-10;
corrections.residual_tolerance = 1e-10;
```

For a mesh with a nonzero tangential diffusion contribution across a partition,
success requires both:

- The global update `||phi_new - phi_old||_inf / max(1, ||phi_new||_inf)` is no
  greater than `update_tolerance`.
- After rebuilding the system from `phi_new` and synchronizing its gradients,
  `||A phi_new - rhs(phi_new)||_2 / ||b_physical||_2` is no greater than
  `residual_tolerance`. Here `b_physical` contains only the prescribed source
  and boundary load, excluding lagged partition-gradient contributions. A zero
  physical RHS uses the absolute residual norm. Any nonzero physical RHS,
  however small, retains relative normalization.

For a problem with partition-gradient coupling, the helper obtains this fixed
physical load by evaluating the existing implicit assembly with a zero
correction field once. This avoids reproducing source/boundary formulas and
keeps the denominator independent of partition corrections and the initial
field. Without partition corrections, the first assembled RHS already supplies
the physical load. Normalizing by the shrinking corrected RHS would prevent a
homogeneous equation from reporting convergence even as its solution and full
absolute residual approach zero.

A serial problem, or one without nonzero tangential partition contributions,
can pass after one implicit linear solve; it still checks the freshly rebuilt
equation. The initial field is synchronized before its gradients are used.
Linear convergence alone is insufficient for a successful implicit return.
Choosing a linear tolerance below the correction residual tolerance leaves room
for the remaining correction error.

A linear-solver failure or exhausted correction limit returns `false`. The
solution contains the last iterate and its synchronized overlap values; the
helper does not restore the input field. Invalid mesh/solver inputs, non-finite
or non-positive tolerances, non-positive correction limits, negative legacy
corrector counts, and rank-divergent controls are rejected collectively before
assembly. Every rank must use the same diffusivity and solver controls.

`Explicit` and `Hybrid` preserve the fixed-sweep contract. Their existing
`nNonOrthogonalCorrectors` argument specifies the number of correction sweeps
following the initial orthogonal or half-implicit solve; `true` reports that
these linear solves converged. The new convergence controls do not turn those
fixed-sweep calls into converged full-equation solves. Under `Implicit`, the
legacy corrector count remains unused and the new maximum controls iteration.

`testSteadyDiffusionConvergence` calls the public helper itself. It compares
non-exact initial fields against an independently solved serial communicator
reference on a sheared grid, verifies a fresh explicitly assembled residual,
and covers coarse/fine composite faces, empty ranks, rejected divergent
controls, correction exhaustion, separate update/residual gates, unchanged
explicit/hybrid semantics, the native implicit overload, homogeneous-load
convergence, and preservation of relative residual checks for tiny nonzero
loads. CTest registers its distributed cases at two and four ranks in addition to serial discovery.
These are solver regression checks, not a physical validation claim.

On September 11, 2026, LLVM Debug and GCC Release each passed all nine tests in
serial and at two and four MPI ranks. Every participating rank reported zero
failures. The runs used the default OpenMPI transports with host networking.
The [LLVM manifest](qualification/region_production/llvm_debug/testSteadyDiffusionConvergence_physical_rhs_manifest.json)
and [GCC manifest](qualification/region_production/gcc_release/testSteadyDiffusionConvergence_physical_rhs_manifest.json)
record the exact commands and binary/library hashes; adjacent logs and
per-rank GoogleTest XML retain the results. The
[native symbol audit](qualification/region_production/llvm_debug/steady_diffusion_native_symbols.txt)
confirms that the FVM static archive defines both original native helper
signatures and both overloads accepting explicit convergence controls.
