# Agent guidance for SimpleFluid

This file applies throughout the repository. Follow explicit user instructions
and any more specific directory guidance. Read [MAINTENANCE.md](MAINTENANCE.md)
for detailed engineering contracts, [README.md](README.md) for capabilities and
dependencies, and [TODO.md](TODO.md) for planned work. Roadmap checkmarks are not
validation evidence.

## Working scope

- Start with `git status --short`. Preserve unrelated changes, untracked files,
  and existing build or verification artifacts.
- Treat review, audit, explanation, and status requests as read-only unless the
  user also requests implementation. For implementation, complete the scoped
  change and focused verification without asking about routine choices.
- Read the relevant public header, implementation fragments, tests, and CMake
  registration together before editing. Prefer `rg` for repository searches.
- Keep changes focused. Do not commit, push, publish, or run broad benchmark
  campaigns unless requested. Explain the need and obtain direction before
  expanding focused validation to a full suite or broad toolchain matrix.

## Repository map

SimpleFluid is a C++23 finite-volume CFD library built on Trilinos, Kokkos,
Tpetra, and MPI. Google Test supplies unit tests.

| Location | Responsibility |
| --- | --- |
| `src/dataclass`, `src/utils` | Configuration and common types |
| `src/geometry`, `src/parallel`, `src/io` | Meshes, partitioning, distributed execution, and output |
| `src/fields` | Mesh-aware field storage and transfer |
| `src/FVM` | Reconstruction, fluxes, and discrete operators |
| `src/equations`, `src/solvers` | Physical equations, time integration, and coupling |
| `src/materials` | Material interfaces and optional water properties |
| `src/*/unitTests` | Component tests and their CMake registration |
| `src/examples`, `src/benchmarks` | Applications, verification workloads, and benchmarks |
| `cmake` | Shared build, dependency, testing, and ABI policy |
| `verification/openfoam` | External comparison cases and launchers |
| `docs/architecture`, `docs/modeling` | API/algorithm contracts and physical assumptions |

Keep dependencies directed from applications toward lower-level components.
Use existing `simplefluid_` CMake helpers and component targets; see
[cmake/README.md](cmake/README.md). Register new sources and explicit
instantiations in the owning target.

## Build and focused tests

Use the existing compatible build tree when available. Inspect
`CMakePresets.json` and the tree's `CMakeCache.txt` before changing configuration.
The checked-in presets use Ninja Multi-Config: GCC builds live in `build/gcc`
and LLVM builds in `build/llvm`. They reference local dependency installations;
override `Trilinos_DIR` and matching Google Test paths when necessary. Keep the
compiler, C++ standard library, Trilinos, and MPI ABI compatible.

Example focused field-storage workflow, run from the repository root:

```sh
# Configure only when needed; supply dependency overrides for this machine.
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug --target testFields --parallel 4
ctest --preset GCC-Debug -N -R '^FieldStoredTest\.'
ctest --preset GCC-Debug -R '^FieldStoredTest\.' --output-on-failure
```

Choose the target and test expression from the affected component's
`unitTests/CMakeLists.txt`; executable names and discovered CTest names differ.
For LLVM, use `LLVM-ninja-multi` and `LLVM-Debug`, then
`ctest --test-dir build/llvm -C Debug -R '<test-regex>' --output-on-failure`.
Multi-config executable paths include the configuration, such as
`build/gcc/bin/Debug/testFields`.

- Run focused Debug serial tests first, then affected MPI/integration tests.
  Add a meaningful regression for changed behavior, especially numerical bugs.
- Use registered MPI tests through `simplefluid_add_mpi_test`, which preserves
  launcher flags and rank counts. `SIMPLEFLUID_MAX_TEST_PROCS` can omit tests
  during configuration; an omitted or skipped test is not a pass. Verify that
  the intended tests actually ran with the required number of ranks.
- Shared-library, visibility, explicit-instantiation, or toolchain changes need
  the relevant ABI/export tests, including `simplefluid_elf_export_boundary`
  on supported ELF builds. See [the Trilinos backend guide](docs/guides/trilinos_backend.md).
- SimpleFluid-side OpenFOAM launchers must source
  `verification/environments.sh`, call `export_build_env`, build through
  `simplefluid_build_target`, and resolve binaries with `simplefluid_executable`.
  Keep launcher documentation aligned with these helpers and active presets.

## Implementation contracts

- Follow `.clang-format` and nearby conventions: four-space indentation,
  120-column limit, and the `SimpleFluid` namespace. Avoid unrelated formatting.
- Preserve established template boundaries: `.hh` declarations and short
  queries, `.ipp` inline/caller-dependent definitions, and compiled `.tcc`/`.cc`
  implementations with supported explicit instantiations. Check the owning
  module before moving definitions; custom template consumers may need their
  own implementation provider. See [the mesh traversal guide](docs/guides/mesh_traverse.md).
- Keep native `MeshHandle`/`FieldStored` execution native. Preserve legacy
  `Mesh` compatibility and caller object identity where that path is supported.
- Preserve owned/ghost maps, local/global indexing, canonical face identity,
  and owner-oriented flux signs. Keep collective call order and failure handling
  consistent across ranks, including ranks with no local cells or faces.
- Respect geometry epochs, execution leases, and numeric-cache invalidation.
  Reused allocation does not imply unchanged geometry or matrix coefficients.
- Preserve conservation, units, boundedness, and acceptance gates. Do not hide
  defects with clipping or relaxed tolerances. Trace storage volumes, source
  units, boundary contributions, and flux orientation when changing numerics.
- ALE transport uses mesh-relative flux and conservative old/new-volume
  storage; pressure and continuity retain their absolute-flux contract. Keep
  physical inventories distinct from numerical volume corrections.
- `MultiphysicsSolverCoupler` owns optional physics and coupled advancement and
  retains shared ownership of its solver. The solver owns core equations and
  fields. Preserve this ownership direction and atomic acceptance/rollback of
  geometry, fields, time, histories, and model ledgers. Publish accepted states
  only. See [the coupling contract](docs/architecture/multiphysics_solver_coupler.md).
- Honor documented support/rejection matrices. Component coverage does not
  establish support for every mesh, physics combination, or MPI rank count.

## VoF, AMR, and loop circuit-breakers

When working on `src/geometry/mesh/AMR`, `src/geometry/PLIC`, `src/FVM/*VoF*`, or `src/solvers/*VoF*`, read the active detailed contracts alongside `MAINTENANCE.md`:
- [VoF solver verification contracts](docs/verification/vof_solver_contracts.md) (Stage 7 Q0–Q7 gates, negative controls, and triage rules)
- [General VoF implementation boundaries](docs/architecture/general_vof_implementation.md) (`Ordinary`, `BoundedCertification`, and `Exact` modes, narrow solver components, and allocation ledgers)
- [Spatial adaptive closure](docs/architecture/spatial_adaptive_closure.md) (persistent `OrderedPages`, `DirectoryFactPages`, and cross-rank contact/subface ownership)

To prevent multi-hour review-debug-verification loops, strictly enforce these stopping and triage rules:

- **Two-attempt cap on budget and symbol gates:** If a change passes functional and numerical correctness but fails a fixed byte/cell cap (such as `700 B/cell` or `64 MiB`), cumulative non-refunded scratch admission, an `exact / one-byte-short` probe, or an ELF export ceiling (`CheckElfExports.cmake` / `CheckFvmExports.cmake`) after **at most two focused optimization attempts**, **STOP** rewriting data structures or ownership graphs. Report the exact measured vs. allowed telemetry (`used`, `request`, `limit`, `cumulative_spent`, or symbol count) to the user and ask whether to adjust the gate or pursue a separate structural refactor.
- **Model lower bound vs. optimizer failure:** Do not loop tuning Gauss-Newton/dogleg iteration limits, step bounds, or initial seeds when an `Ordinary` PLIC reconstruction fails the strict single-plane neighborhood RMS gate (`RequireModelAgreement`) on a curved or coarse-corner interface where the single-plane model residual has a positive mathematical lower bound. Use the documented `ConservativeApproximation` / `PolyhedralGradientSelection` path where applicable, or stop and report the model-gate collision.
- **Preserve intentional negative-control oracles:** Several regressions and qualification fixtures intentionally assert that lower-order or strict uncleaned paths fail a specific gate (for example, `DonorCell` missing full displacement in Q3, or strict `relative_cutoff=0` microscopic offset rejection at step 4 of `full50` MPI). Before altering shared operators to make a failing test pass, verify whether the test is an intentional negative control.
- **Fast inner-loop verification first:** During iterative debugging, build only the single affected test target and run the narrowest `--gtest_filter='<Suite.Case>'` in serial or 2-rank MPI with `--timeout 60`. Do not launch 300s–1800s `N32`/`N64`/`full50`/`Q3` campaigns or 4/9/33-rank oversubscribed MPI suites until the targeted fast check passes.
- **MPI sandbox socket failures:** If `mpiexec` fails before `main()` due to PMIx or local socket creation restrictions in a sandboxed runner, classify it immediately as an environment/launcher restriction (rerun with host socket permissions or report it), never as a solver deadlock or code regression.

## Commit messages

Follow the [commit-message format and workflow](MAINTENANCE.md#commit-messages)
when drafting messages or preparing an authorized commit.

## Documentation and handoff

Update public contracts with behavior changes. Keep architecture in
`docs/architecture`, physical assumptions in `docs/modeling`, and outstanding
work in `TODO.md`. Keep dated run logs, profiles, hashes, and timing samples in
local build artifacts rather than adding per-iteration review diaries.

For numerical or performance work, compare matched cases and record provenance,
conservation/accuracy results, and measurement limits. Distinguish implemented,
tested, externally compared, and physically validated behavior.

Before handing off, inspect the diff and run `git diff --check` and
`git status --short`. Report what changed, focused commands and results, and
any remaining limitations. For MPI work include actual ranks and skipped tests.
Whitespace checks are not runtime validation, historical logs are not fresh
tests, and local results do not establish hosted CI success.
