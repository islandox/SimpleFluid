# SimpleFluid Maintainer Guide

This guide is for code review, routine maintenance, and feature development.
It describes the repository as it exists today; `README.md` is the user-facing
capability overview and `TODO.md` is the roadmap. A checked roadmap item is not
a substitute for a passing test or a review of the implementation path.

## Start here

Before changing code:

1. Read the relevant public header, its implementation (`.cc`, `.tcc`, or
   `.ipp`), and its unit tests together.
2. Check `git status --short` and preserve unrelated worktree changes.
3. Build the narrowest affected target in Debug mode.
4. Run focused serial tests, then the affected MPI or integration tests.
5. For numerical or performance changes, compare results before and after the
   change rather than relying only on test exit status.

The code requires C++23 and a Trilinos build containing Kokkos, Teuchos,
Tpetra, STK, Belos, Ifpack2, MueLu, and Zoltan2. MPI and Google Test are used
for distributed execution and testing.

## Repository map and dependency direction

Production code is layered from low-level data types toward applications:

```text
dataclass / utils
        |
geometry + parallel + io
        |
fields
        |
FVM operators
        |
equations
        |
solvers
        |
examples / benchmarks
```

The CMake targets make the same boundary explicit:

| Target | Responsibility |
| --- | --- |
| `SimpleFluid::Core` | Typed configuration data and common low-level types |
| `SimpleFluid::Mesh` | Mesh implementations, partitioning, and VTU support |
| `SimpleFluid::Fields` | Distributed cell-, face-, and boundary-field storage |
| `SimpleFluid::FVM` | Finite-volume reconstruction and assembly operators |
| `SimpleFluid::Equations` | Momentum, pressure, temperature, turbulence, and source models |
| `SimpleFluid::Solvers` | Time stepping and pressure-velocity/physics orchestration |
| `SimpleFluid::SimpleFluid` | Umbrella interface for downstream applications |

Keep dependencies pointing down this table. If a low-level layer needs a
solver type, the abstraction is probably in the wrong layer.

Important entry points for a manual review are:

- `src/geometry/mesh/MeshBase.hh` for the CRTP mesh contract.
- `src/geometry/MeshHandle.hh` for runtime mesh type erasure.
- `src/fields/FieldStored.hh` for mesh-aware distributed field storage.
- `src/FVM/Operators.hh` and `src/FVM/TransportSystem.hh` for the public FVM
  operator surface.
- `src/equations/Equation.hh` and `src/problems/Problem.hh` for the framework
  ownership model.
- `src/solvers/FluidSolver.hh` for incompressible flow orchestration.
- `src/solvers/BoussinesqSolver.hh` for thermal and multiphysics integration.

Template declarations and definitions are intentionally split among `.hh`,
`.tcc`, and `.ipp` files in several modules. Review all included implementation
fragments before changing a template interface.

## Configure and build

The local preset points at the maintainer's Trilinos installation. Override
`Trilinos_DIR` when using another installation:

```bash
cmake --preset Local
cmake --build --preset Debug
```

An explicit configuration is useful on a new machine or in CI:

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
  -DTrilinos_DIR=/path/to/trilinos/lib/cmake/Trilinos
cmake --build build --config Debug --parallel 4
```

Useful configuration switches are:

| Option | Default | Purpose |
| --- | --- | --- |
| `SIMPLEFLUID_BUILD_BENCHMARKS` | `ON` | Build benchmark and performance-regression targets |
| `SIMPLEFLUID_ENABLE_COVERAGE` | `OFF` | Add GCC/Clang coverage instrumentation |
| `SIMPLEFLUID_MAX_TEST_PROCS` | `0` | Limit generated MPI tests; CI commonly uses `2` |
| `SIMPLEFLUID_GTEST_DISCOVERY_TIMEOUT` | `30` | Bound Google Test discovery time in seconds |
| `SIMPLEFLUID_BUILD_DOCS` | `OFF` | Require Doxygen and enable the `docs` target |

The project uses one shared precompiled header. Add only stable, frequently
included third-party or standard-library headers to
`cmake/PrecompiledHeaders.hh`; volatile project headers increase rebuild cost
and can create dependency cycles.

## Change discipline and code conventions

There is no repository-wide formatting configuration. Follow the surrounding
file, avoid drive-by reformatting, and keep comment-only, mechanical, and
behavioral changes separable during review.

- Use the configured `Pack` aliases for scalar, local ordinal, global ordinal,
  map, matrix, and vector types in templated numerical code.
- Keep mesh-local IDs, Tpetra local rows, and global IDs distinct. Convert at
  the existing mesh/map boundaries instead of relying on coincident values.
- Make ownership explicit in names and comments: owned storage is authoritative;
  overlap storage may require `sync_ghosts()` before a stencil read.
- State physical units in option, source, and output-field documentation.
- Parse `Database` options with defaults and validate them before allocating
  distributed state.
- Preserve established public names, including historical spellings, unless a
  compatibility plan is part of the change.
- Prefer a focused exception with the invalid value or invariant in its message
  over an unchecked assertion on user-provided configuration.

## Test strategy

Run CTest from the repository root so the multi-configuration build type is
unambiguous:

```bash
ctest --test-dir build -C Debug --output-on-failure -j4
```

For faster feedback, build and run a single test executable or use CTest name
and label filters:

```bash
cmake --build build --config Debug --target testFields
ctest --test-dir build -C Debug -R 'CellField' --output-on-failure
ctest --test-dir build -C Debug -L mpi --output-on-failure
ctest --test-dir build -C Debug -L integration --output-on-failure
ctest --test-dir build -C Debug -L benchmark --output-on-failure
```

Use `ctest --test-dir build -C Debug -N` to inspect the tests registered in the
current build tree. New `test*.cc` files are discovered through CMake globs in
several directories, so adding a file can trigger or require reconfiguration.

Serial success is not sufficient for code that touches ownership, ghost
values, global reductions, boundary partitioning, or collective validation.
Run the matching multi-rank test and exercise the largest process count
relevant to the change. An MPI launcher failure before a test body starts can
be an environment/interface problem; reproduce it in a normal shell before
classifying it as a solver regression.

### Numerical change checklist

For discretization, solver, or physics work, verify the applicable invariants:

- owned and overlap maps use the same global-ID convention as the mesh;
- ghost data is synchronized before stencil reads;
- boundary faces are assembled once and with the intended owner convention;
- source terms have documented units and are integrated with cell volume when
  converting density to totals;
- global acceptance checks use communicator collectives;
- disabled optional models reproduce the baseline path;
- bounded state variables remain within their configured limits;
- residuals, conservation errors, and Krylov iteration counts do not regress;
- results remain valid in the supported non-orthogonal treatment modes.

Tests should state the behavior they protect. Prefer analytical checks,
conservation balances, convergence rates, and cross-rank invariants over smoke
tests alone.

## Common extension paths

### Add a mesh implementation

1. Implement the `MeshBase` contract and its index types.
2. Add owned/local/global indexing and boundary batches explicitly.
3. If runtime selection is required, extend `MeshHandle` and its visitors.
4. Add geometry, connectivity, ownership, and multi-rank tests.
5. Register compiled sources in `SimpleFluidMesh`.

Preserve mesh object identity when refining or grading an existing mesh; fields
and solver components may retain shared pointers to it.

### Add an FVM operator or equation

Reuse the shared operator helpers and boundary caches before adding another
assembly path. Keep mesh traversal and coefficient construction in FVM, the
physical residual/source definition in equations, and time-step orchestration
in solvers. Test the algebra on a minimal mesh, then add an analytical or
conservation case.

### Add coupled physics

`BoussinesqSolver` owns the current source/model registries and configure,
find, and remove APIs. Extend those seams so optional models remain lazily
allocated and disabled cases retain existing behavior. Publish diagnostics
through existing output-field maps when possible instead of adding one getter
per field.

### Add an executable

Put runnable cases in `src/examples` and reusable timing cases in
`src/benchmarks`. Link through the narrowest suitable library target, add a
bounded smoke test, and document environment knobs that reduce runtime for CI.
Do not make an example the only verification of a numerical feature.

## Performance work

Use `RelWithDebInfo` for profiling; the project preserves frame pointers in
that configuration:

```bash
cmake --build --preset RelWithDebInfo --target simplefluid_benchmark
build/bin/RelWithDebInfo/simplefluid_benchmark \
  --preset release-profile \
  --output build/benchmarks/release-profile.csv
```

Record the command, mesh size, MPI size, solver configuration, compiler/build
mode, wall time, iteration counts, and numerical residuals. Treat profiler
instruction counts as a lead. Accept an optimization only after normal timing
and numerical-equivalence checks.

## Documentation policy

Public interfaces should be understandable without opening their
implementation. C++ files use Doxygen comments:

- each non-empty `.cc`, `.hh`, `.hpp`, `.ipp`, and `.tcc` file has a top-level
  `@file`, `@author`, `@brief`, `@version`, `@date`, and `@copyright` block;
- classes, structs, concepts, and enums explain their role and template
  parameters;
- non-inline out-of-class functions document parameters, return values, and
  only exceptions that can actually be thrown;
- units, ownership, collective behavior, and valid ranges are stated when they
  affect correct use;
- test comments explain behavior and regression intent, not line-by-line setup.

Do not restate an identifier in prose or add speculative `@throws` tags.
Update comments in the same change as an interface or invariant.

Generate the API reference with Doxygen installed:

```bash
cmake -S . -B build -DSIMPLEFLUID_BUILD_DOCS=ON \
  -DTrilinos_DIR=/path/to/trilinos/lib/cmake/Trilinos
cmake --build build --target docs
```

The HTML entry point is `build/docs/doxygen/html/index.html`. Review Doxygen
warnings, especially missing parameter documentation and unresolved links,
before merging interface changes.

## Review and handoff checklist

Before handing a change to another maintainer:

```bash
git diff --check
git status --short
```

Then confirm:

- the diff contains no unrelated cleanup or generated build products;
- every new source is registered in the correct CMake target;
- focused Debug tests and required MPI/integration tests pass;
- numerical and performance comparisons are attached when applicable;
- public comments and this guide remain consistent with the code;
- `README.md` and `TODO.md` claims are updated only after verification.

When a full suite cannot run, record exactly what was built and tested, the
reason for omissions, and the command another maintainer should run.
