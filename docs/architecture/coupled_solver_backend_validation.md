# Coupled solver backend qualification

This follows commit `5708927` and numeric preconditioner correction `6033af4`.
It qualifies the existing coupled operator/workspace choices for the current
`FluidSolver`, `IncompressibleIsothermalSolver`, and `BoussinesqSolver` paths.
Boussinesq coverage includes its kinematic and physical-material modes.
Assembled/cached operation remains the default. The broader scalar face-based
FVM and general operator/diagonal/preconditioner separation project is still
separate, unfinished work.

## Defects reproduced and corrected

1. **Stale numeric preconditioning.** At 512 cells the third cached solve
   exhausted 400 iterations while a fresh solve converged in 56. True operator,
   RHS, C and Schur actions matched fresh assembly. The linked MueLu `full`
   reuse keeps old smoothers and coarse matrices; `RP` retains transfers while
   refreshing numeric state. The regression runs four generations for every
   backend/policy on one, two and four ranks without relaxing tolerances.
2. **Timestepper interpolation mismatch.** On cylindrical/skewed meshes,
   converged algebraic solves left reconstructed continuity errors around
   `1e-5`–`1e-4`. Timesteppers had selected the historical arithmetic-interpolation
   overload, while reconstructed face flux used distance weights. All three
   timestepper classes now pass an explicit continuity target, including zero,
   to use matching weights. Historical direct-call arithmetic behavior remains
   covered and unchanged.
3. **Pressure-gradient mismatch.** Selecting `GaussLinear` changed reconstructed
   face flux but left stabilization on a least-squares stencil. C now uses the
   selected boundary-aware affine gradient. Scheme changes invalidate operator
   setup; unknown/rank-divergent schemes fail collectively before assembly.
   Nonzero Dirichlet/Neumann data and density 13 are checked by comparing
   `A*x-rhs` pressure rows directly with independently reconstructed face-flux
   balances, including retained old operators across scheme changes.

These are numerical consistency/reuse corrections. No physical model,
conservation tolerance, or constrained-ALE support restriction was broadened.

## Qualification coverage

All four combinations are exercised explicitly:

| Operator | Workspace |
| --- | --- |
| assembled | cached_products |
| assembled | streamed_products |
| block_composite | cached_products |
| block_composite | streamed_products |

`testCoupledSolverBackends` advances four timesteps for each solver family and
choice, changes the timestep, compares velocity, physical pressure, corrected
face flux and temperature with assembled/cached reference runs, and switches
backend/policy within a running solver. It independently sums cell face fluxes
with a `1e-8` absolute continuity bound. Composite production solves assert zero
monolithic matrix builds; streamed solves assert one peak application-owned
product and none live after setup. Tests inspect the actual Problem registry;
legacy physical assembly and compatibility solve state have distinct owners.

| Path | Serial | Two/four ranks |
| --- | --- | --- |
| Native Cartesian and cylindrical wedge | All solver families and choices | Same |
| Native periodic cylindrical ring | Both gradients, all families/choices | Same |
| Native unstructured / partitioned unstructured | Pressure outlet at 17 Pa | Explicit partitioned mesh |
| Legacy Cartesian | Pressure outlet, density 7, both gradients | Same |
| Legacy generated cylinder and sphere | All families/choices | Same |
| Skewed semi-structured extrusion | Explicit/implicit/hybrid diffusion | Rejected by existing serial-only mesh contract |
| Gradient-scheme affine action and cache changes | LS/Gauss/LS, nonzero boundaries, density 13 | Same, including remote product rows |
| Coupled multivector contract | Alpha/beta, NaNs, aliasing, selected views, maps, transpose rejection | Noncontiguous IDs and reversed rank ownership |

The existing turbulence acceptance tests run each coupled choice for standard
and realizable k-epsilon and SST, their supported wall treatments, effective
viscosity/conductivity boundaries and buoyancy production. The existing
combined sheared RANS/Sheng/fission/density-feedback fixture now also runs all
four coupled choices for two steps in serial and on two ranks, retaining its
original physical budget checks. Isothermal RANS retains its no-temperature
contract.

Constrained ALE tests cover all choices with both gradient schemes for
Cartesian X/Y/Z motion, cylindrical axial motion, and serial semi-structured
axial motion. They retain GCL, liquid mass, volume/source closure, physical
continuity, exact mesh-normal flux and zero relative top flux checks.
Heating/forced-rejection tests preserve rollback checks. Unsupported ALE model
and geometry combinations retain their existing rejection tests.

## Commands and results

All solver test targets were rebuilt in GCC Debug against the live local
Trilinos 17.2 / Kokkos Serial installation. The target set is exactly the 23
`test*.cc` files in `src/solvers/unitTests`; no source-disabled million-cell
case was enabled.

```sh
python - <<'PY'
from pathlib import Path
import subprocess
names = sorted(p.stem for p in Path('src/solvers/unitTests').glob('test*.cc'))
subprocess.run(['cmake', '--build', '--preset', 'GCC-Debug', '--target',
                *names, 'coupled_operator_memory', '-j', '4'], check=True)
PY

# Rebuild the added periodic/axis tests after extending coverage:
cmake --build --preset GCC-Debug --target testCoupledSolverBackends -j 4
cmake --build --preset GCC-Debug --target testBoussinesqPlanarALE -j 4

ctest --test-dir build/gcc/src/solvers/unitTests -C Debug --output-on-failure
ctest --preset GCC-Debug -R '^simplefluid_elf_export_boundary$' --output-on-failure
git diff --check
```

MPI runs use host networking because the restricted sandbox does not support
the launcher. The complete solver-directory run includes all registered solver
MPI tests; skips are counted separately from passes. Additional topology-specific
skips inside multi-test MPI executables remain intentional (for example,
semi-structured meshes run only in serial).

The earlier complete gate passed 412 registrations with 19 skips out of 431.
The final expanded gate passed **418**, with **19 skipped**, out of **437**
registrations in **278.77 s**, including **30 MPI registrations**. It includes
the periodic-cylinder and all-axis ALE cases. ELF export validation passed
1/1 in 4.54 s; `git diff --check` passed. Skips are not counted as passes.

## Boundaries of the evidence

This qualifies CPU solver behavior and coupled representation equivalence;
it is not new experimental physical validation. It does not establish
accelerator execution, arbitrary ill-conditioned cases, every possible
parameter combination, or million-cell performance. The original small
`full`-reuse benchmark report is historical; current benchmarks identify
MueLu `RP` reuse explicitly. Detailed preconditioner/Krylov allocation tracing
and separate preconditioner setup timing remain unavailable in the existing
benchmark, rather than inferred from CRS payload counts.
