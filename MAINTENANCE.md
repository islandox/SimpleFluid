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

The solver has two deliberate backend paths. Supported mapped meshes execute
natively through `MeshHandle` and `FieldStored`; callers that provide the
legacy `Mesh` use synchronized compatibility fields. Preserve that boundary
instead of converting a supplied native mesh into a legacy mesh.

Within the native path, Cartesian and cylindrical meshes construct their
distributed slab views directly. An `UnstructuredMesh` must first be
partitioned with `MeshPartitioner` and adapted through `PartitionedMesh` on a
multi-rank communicator. `SemiStructuredXY_Z` is serial-only; its
`MeshHandle` constructor deliberately rejects multi-rank use until an owned,
overlap, and ghost-indexing design is implemented for that topology.
Focused native regressions cover unstructured geometry, operators,
`FluidSolver`, and `BoussinesqSolver` directly in serial and across partition
faces in MPI. Do not infer multi-rank `SemiStructuredXY_Z` support from that
partitioned-unstructured coverage.

### Liquid-mass inventory transactions

`LiquidMassInventory` defaults to the global `globalConstantMass` approximation.
On a fixed grid, the opt-in `cellMassInventory` mode owns
`liquidMassInventory` in kg/m3 of fixed reference volume and advances it with
the accepted projected single-continuum face-volume flux through
`previewCellwiseAdvance()`. Under `planarALE`, the same named field is a
current-control-volume mass density: its conserved product uses accepted-old
and trial-new volumes and every advective term uses the trial
absolute-minus-mesh relative flux. This is not a separate phase-weighted
liquid flux. Keep that trial state private until
`commitPhaseChange()` follows a successful planar closure; do not publish or
consume the trial field directly. Refreshing `rhoLiquid` invalidates every
older global or cellwise preview because its derived volume is stale. Cellwise
mode currently requires `error` depletion and zero physical-boundary liquid
flux because inlet composition and post-boiling local clipping have no
conservative contract. Always derive liquid volume from this inventory and
pure `rhoLiquid`, never void-reduced mixture density. Its physical mass-closure
gate must remain independent of the Krylov tolerance; a loose linear solve must
fail conservation rather than relax it. Separate solvent, solute, and fissile
inventories remain future work.

When free-surface coupling has enabled boiling phase-inventory tracking,
`remove_free_surface_model()` must reject removal while submerged steam remains.
Do not weaken that guard or silently reset the steam ledger; add a conservative
transfer/disposition policy first.

### Geometry epochs and motion transactions

`PlanarALEMeshMotion` is the sole geometry mutator for the constrained
solver-integrated `planarALE` path. It supports fixed-topology Cartesian X/Y/Z
and cylindrical axial motion in serial/MPI plus serial `SemiStructuredXY_Z`
axial motion. Use its `begin_trial()` and exactly one of `accept_trial()` or
`rollback_trial()`; never update structured edge arrays directly after fields
or caches exist. The exclusive controller lease and geometry epoch belong to
the shared concrete mesh, so every alias `MeshHandle` observes the same
revision. The solver's mutable constructor is an opt-in ownership seam for the
same native handle; it does not authorize arbitrary geometry mutation or
weaken const access for equations and fields.

The geometry epoch is the authoritative numeric-cache revision. After a trial
changes coordinates, refresh or invalidate all of these before assembly or
output:

- cell-gradient and transport-geometry caches;
- Rhie--Chow face-flux geometry and boundary workspaces;
- momentum and temperature transport coefficients and their reusable linear
  solver/preconditioner state;
- pressure-projection matrices, boundary data, predictor state, and linear
  solver/preconditioner state;
- coupled pressure--velocity numeric blocks, Schur products, and block
  preconditioner state;
- model-owned transport geometry used by the liquid inventory, radiolytic gas,
  or another enabled extensive equation; and
- geometry-derived output state, including VTU point coordinates.

Topology-stable graphs, maps, entity IDs, and boundary membership may survive
motion; geometry-dependent numeric values may not. A surviving cache must
capture the concrete geometry identity and epoch, reject stale access, and
provide an explicit refresh/invalidation path. Raw mutation through
`MeshHandle::visit_mutable()` does not publish an epoch and is unsupported
after field/cache construction.

Treat one ALE step as an accepted-state transaction:

1. Snapshot every mutable field, model ledger, diagnostic/history cursor, and
   time state that the trial can touch.
2. Start one geometry trial and keep its accepted-old volumes, trial-new
   volumes, exact owner-oriented mesh flux, identity, and epochs alive through
   all assemblies.
3. Refresh all geometry consumers, derive a separate mesh-relative flux, and
   rerun each nonlinear corrector from the same accepted snapshot. Preview
   liquid, gas, level/headspace, and volume-source changes; do not publish them.
   `free_surface_ale_maximum_correctors` independently caps the outer Picard
   trials and the strict pressure-only continuity refinements within each
   trial.
4. Accept only after mesh quality, GCL, outer level/source plus material/gas
   state convergence, actual mesh/pool equality, generalized continuity,
   liquid/gas inventory, energy, and volume closure gates pass collectively.
5. Commit geometry, fields, ledgers, diagnostics, time, and one history record
   exactly once. On any exception, roll back the active geometry trial, restore
   all snapshots, refresh the accepted-epoch caches, and leave no output or
   history record for the rejected attempt.

For ALE temperature transport, the extensive sensible energy is
$V_c m_{l,c}^*c_{p,c}T_c$. Use accepted-old and trial-new
`liquidMassInventory` densities in the corresponding old/new control volumes.
Do not substitute `rhoLiquid` or void-reduced mixture density over the
bubble-displaced pool volume; `rhoLiquid` converts liquid mass to material
volume and has a different ownership role.

The volume-source ledger removes the exact implicit material face transport:
trial-new liquid mass divided by pure density uses the carrier upwind, while
each post-transport/pre-kinetics bubble fraction uses the combined
carrier-plus-slip upwind from its population equation. Do not reconstruct this
term from accepted-old aggregate occupancy or from post-kinetics bubbles.

`PlanarALEBoundary` marks the moving patch's trial velocity cache as `Slip`, so
face reconstruction retains the owner-cell tangential velocity while momentum
diffusion adds no boundary constraint. The fixed-flux pressure boundary
independently enforces
$\phi_{abs,f}=\phi_{m,f}$, leaving zero normal carrier flux relative to the
mesh. Accepted `PlanarALEStepDiagnostics` retain the
per-outer level, target, continuity, material-state, and gas-state histories, and
each accepted free-surface history record copies that diagnostic snapshot.
Rejected trials do not publish a new history record.

`FluidSolver::maximum_courant_number()` uses the accepted transport flux:
absolute projected flux on a fixed mesh and mesh-relative flux for planar ALE.
ALE initialization copies the accepted stationary projected flux into the
relative field before this selector changes.
Do not count swept mesh volume as transported fluid when adding another ALE
time-step or stability diagnostic.

When extending ALE support to another equation or model, require all of the
following before relaxing its setup rejection: conservative old/new-volume
storage, mesh-relative convection, explicit boundary/outflow ownership,
geometry-cache refresh, complete snapshot/restore coverage, one contribution
to the shared `VolumeContinuityTarget` when it changes material volume, and
serial/MPI conservation plus forced-rejection tests. A fixed-volume operator
followed by an inventory or level correction is not an ALE implementation.

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

`SimpleFluid::Equations` remains a logical link target, but compiled equation
and solver specializations share the `SimpleFluidSolvers` DSO. On ELF builds,
`cmake/SimpleFluidLinux.map` and `cmake/CheckSimpleFluidElfExports.cmake`
enforce the public API and the narrowly reviewed Trilinos/Kokkos runtime
bridges. Changes to explicit instantiations, visibility, or the shared PCH must
retain that audit.

Important entry points for a manual review are:

- `src/geometry/mesh/MeshBase.hh` for the CRTP mesh contract.
- `src/geometry/MeshHandle.hh` for runtime mesh type erasure.
- `src/geometry/MeshMotionModel.hh` and `src/geometry/PlanarALEMeshMotion.hh`
  for the fixed-topology motion transaction and GCL authority.
- `src/FVM/ALEControlVolumeState.hh` for the validated non-owning old/new
  control-volume and owner-oriented mesh-flux contract.
- `src/equations/VolumeContinuityTarget.hh`,
  `src/solvers/VolumeContinuityModel.hh`, and
  `src/solvers/PlanarALEBoundary.hh` for the generalized continuity,
  conservative material-volume ledger, and moving-top boundary contracts.
- `src/geometry/MeshReorderingFactory.hh` for collective selected-first local
  cell ordering and its owned/ghost range certificate.
- `src/geometry/SolidSubdomain.hh` for compact selected-cell mesh views and
  synthetic solid-interface boundaries.
- `src/fields/FieldStored.hh` for mesh-aware distributed field storage.
- `src/fields/MeshFieldTraits.hh` for native-versus-legacy field selection.
- `src/FVM/Operators.hh` and `src/FVM/TransportSystem.hh` for the public FVM
  operator surface.
- `src/solvers/PlanarFreeSurfaceModel.hh` for vessel/headspace closure and the
  global or cellwise liquid-mass inventory contracts.
- `src/FVM/details` for mapped-mesh implementations and reusable geometry,
  boundary, gradient, and flux caches.
- `src/equations/Equation.hh` and `src/problems/Problem.hh` for the framework
  ownership model.
- `src/solvers/FluidSolver.hh` for incompressible flow orchestration.
- `src/solvers/BoussinesqSolver.hh` for thermal and multiphysics integration.
- `src/equations/SolidHeatConductionEquation.hh` for zero-advection physical
  heat conduction in a solid region.

Template declarations and definitions are intentionally split among `.hh`,
`.tcc`, `.ipp`, and `FVM/details` files in several modules. Public FVM headers
should remain thin forwarding surfaces where a mapped implementation already
exists. Review all included implementation fragments before changing a
template interface.

## Configure and build

The checked-in presets point at the maintainer's matching GCC/libstdc++ and
LLVM/libc++ Trilinos installations. Override `Trilinos_DIR` and, for LLVM, the
matching Google Test location when using another installation:

```bash
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Debug
ctest --preset GCC-Debug
```

The GCC and LLVM presets use `build/gcc` and `build/llvm`. Keep the selected C++
standard library consistent across SimpleFluid, Trilinos, Google Test, and the
final link.

An explicit configuration is useful on a new machine or in CI:

```bash
cmake -S . -B build/manual -G "Ninja Multi-Config" \
  -DTrilinos_DIR=/path/to/trilinos/lib/cmake/Trilinos
cmake --build build/manual --config Debug --parallel 4
```

Useful configuration switches are:

| Option | Default | Purpose |
| --- | --- | --- |
| `SIMPLEFLUID_BUILD_BENCHMARKS` | `ON` | Build benchmark and performance-regression targets |
| `SIMPLEFLUID_ENABLE_COVERAGE` | `OFF` | Add GCC/Clang coverage instrumentation |
| `SIMPLEFLUID_ENABLE_LTO` | `ON` | Enable IPO in Release and RelWithDebInfo |
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
- Keep fields, caches, equations, and solvers bound to the same mesh object;
  reject mismatches instead of relying on equivalent-looking topology.
- Apply `MeshReorderingFactory` before constructing fields or caches. Preserve
  its separate owned-selected and ghost-selected prefixes: combining them into
  one local interval would violate the owned-first distributed-map contract.
- Transfer a uniquely owned mutable `MeshHandle` into a required reordering;
  do not restore the former full-handle copy, which duplicates parent-sized
  indexers and connectivity.
- Keep `SolidSubdomain` range-backed. Do not reintroduce parent-sized selection
  masks or dense parent-to-subdomain cell/face reverse arrays.
- Select native and legacy field families through `MeshFieldTraits` rather
  than adding another parallel equation implementation.
- Keep mapped FVM implementation bodies under `src/FVM/details` and preserve
  the stable public forwarding surface under `src/FVM`.
- Validate rank-local inputs collectively before a rank can throw while peers
  enter assembly or a Trilinos collective.
- State physical units in option, source, and output-field documentation.
- Parse `Database` options with defaults and validate them before allocating
  distributed state.
- Preserve established public names, including historical spellings, unless a
  compatibility plan is part of the change.
- Prefer a focused exception with the invalid value or invariant in its message
  over an unchecked assertion on user-provided configuration.

## Test strategy

Use the GCC test preset for the normal local gate:

```bash
ctest --preset GCC-Debug --parallel 4
```

For faster feedback, build and run a single test executable or use CTest name
and label filters:

```bash
cmake --build --preset GCC-Debug --target testFields
ctest --test-dir build/gcc -C Debug -R 'CellField' --output-on-failure
ctest --test-dir build/gcc -C Debug -L mpi --output-on-failure
ctest --test-dir build/gcc -C Debug -L integration --output-on-failure
ctest --test-dir build/gcc -C Debug -L benchmark --output-on-failure
```

Use `ctest --preset GCC-Debug -N` to inspect the tests registered in the GCC
tree. There is no checked-in LLVM test preset; use
`ctest --test-dir build/llvm -C Debug --output-on-failure` after an LLVM build.
New `test*.cc` files are discovered through CMake globs in several directories,
so adding a file can trigger or require reconfiguration.

Serial success is not sufficient for code that touches ownership, ghost
values, global reductions, boundary partitioning, or collective validation.
Run the matching multi-rank test and exercise the largest process count
relevant to the change. An MPI launcher failure before a test body starts can
be an environment/interface problem; reproduce it in a normal shell before
classifying it as a solver regression.

Changes to a shared operator, equation, or solver should exercise both the
native `MeshHandle`/`FieldStored` path and the legacy `Mesh` compatibility path
when both are applicable. Include Cartesian, cylindrical, semi-structured, or
partitioned coverage according to the claimed mesh support; do not infer
cross-backend parity from one smoke test.

### Numerical change checklist

For discretization, solver, or physics work, verify the applicable invariants:

- owned and overlap maps use the same global-ID convention as the mesh;
- fields and reusable caches retain the exact mesh identity expected by the
  assembly path;
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
2. Add owned/local/global cell and face indexing, boundary batches, and the
   maps required by `FieldStored` explicitly.
3. Use `PartitionedMesh` for a compatible distributed unstructured geometry;
   if a new direct runtime alternative is required, extend `MeshHandle` and
   its visitors.
4. Add geometry, connectivity, ownership, stored-field, operator, and
   multi-rank tests.
5. Register compiled sources in `SimpleFluidMesh` and explicit template
   instantiations where required.

Preserve mesh object identity when refining or grading an existing mesh; fields
and solver components may retain shared pointers to it.

### Add an FVM operator or equation

Reuse the shared operator helpers and boundary/geometry caches before adding
another assembly path. Put mapped implementations under `src/FVM/details` and
forward through the public header; use `MeshFieldTraits` when an equation must
support both stored and legacy fields. Keep mesh traversal and coefficient
construction in FVM, the physical residual/source definition in equations,
and time-step orchestration in solvers. Test the algebra on a minimal mesh,
then add an analytical or conservation case and the relevant native/legacy
parity test.

Legacy/native mapped weighted-scalar transport and native mapped
physical-temperature transport preserve Backward Euler and first-order upwind
as the default public behavior. The opt-in `ScalarTransportDiscretization`
path supports constant-step BDF2, which requires an older field on the same
mesh, and bounded linear-upwind convection, which retains the implicit upwind
matrix and applies a conservative limited deferred correction. Keep new scalar
policies in the shared legacy/mapped implementation seam, retain
default-overload compatibility, validate policy and optional-field selections
collectively, and add weighted-scalar parity tests across legacy `CellField`
and native `FieldStored` paths. A higher-order policy for scalar transport does
not by itself extend vector or momentum transport.

### Add coupled physics

`BoussinesqSolver` owns the current source/model registries and configure,
find, and remove APIs. Extend those seams so optional models remain lazily
allocated and disabled cases retain existing behavior. Publish diagnostics
through existing output-field maps when possible instead of adding one getter
per field. Template mesh-dependent models on their mesh type, select storage
through `MeshFieldTraits`, and retain explicit native and legacy
specializations at the shared-library boundary.

### Add an executable

Put runnable cases in `src/examples` and reusable timing cases in
`src/benchmarks`. Link through the narrowest suitable library target, add a
bounded smoke test, and document environment knobs that reduce runtime for CI.
Do not make an example the only verification of a numerical feature.

## Performance work

Use `RelWithDebInfo` for profiling; the project preserves frame pointers in
that configuration:

```bash
cmake --build --preset GCC-RelWithDebInfo --target simplefluid_benchmark
build/gcc/bin/RelWithDebInfo/simplefluid_benchmark \
  --preset release-profile \
  --output build/gcc/benchmarks/release-profile.csv
```

Record the command, mesh size, MPI size, solver configuration, compiler/build
mode, wall time, iteration counts, and numerical residuals. Treat profiler
instruction counts as a lead. Accept an optimization only after normal timing
and numerical-equivalence checks.

Release and RelWithDebInfo enable LTO by default. Compiler-specific PGO presets
designate `natural_convection_shiri` as the training workload. A generate
preset builds the instrumented target; run it to collect profiles and use the
LLVM merge preset where applicable before configuring the matching use preset.
Keep compiler, configuration, source fingerprint, and workload provenance
matched. A PGO-use build that lacks complete current profiles should fail
rather than silently optimize only part of a production target.

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

Leave self-explanatory variables, aliases, trivial inline accessors, and simple
forwarders uncommented. Do not restate an identifier in prose or add
speculative `@throws` tags.
Update comments in the same change as an interface or invariant.

Generate the API reference with Doxygen installed:

```bash
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_BUILD_DOCS=ON
cmake --build --preset GCC-Debug --target docs
```

The HTML entry point is `build/gcc/docs/doxygen/html/index.html`; warnings are
written to `build/gcc/docs/doxygen/warnings.log`. Review them for malformed
commands, stale parameter tags, and unresolved links. Missing documentation
alone is not a reason to narrate an otherwise obvious API.
The documentation target filters GitHub-style inline and display math into
Doxygen formulas; keep Markdown math in GitHub-compatible syntax.

The ELF export-boundary test checks symbol presence and visibility; it is not
a class-layout ABI checker. Public solver/equation templates and by-value
option/result structs expose their layout, and the shared library currently
has no versioned ABI promise. Rebuild downstream consumers whenever one of
those layouts changes, even when all legacy symbol names remain available.

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
- native and legacy backend behavior is covered where a shared path changed;
- `simplefluid_elf_export_boundary` passes after shared-library, template
  instantiation, visibility, or toolchain changes;
- numerical and performance comparisons are attached when applicable;
- public comments and this guide remain consistent with the code;
- `README.md` and `TODO.md` claims are updated only after verification.

When a full suite cannot run, record exactly what was built and tested, the
reason for omissions, and the command another maintainer should run.
