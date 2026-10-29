# CMake helpers

Include the main entry files from project or component CMake lists:

| Entry file | Purpose | Include timing |
| --- | --- | --- |
| [Options.cmake](Options.cmake) | Cache defaults and feature switches | Before CTest and dependency discovery |
| [Profiling.cmake](Profiling.cmake) | PGO and modular debug-symbol helpers | After options, before adding targets |
| [Build.cmake](Build.cmake) | Library, example, dependency, PCH and linkage helpers | After Trilinos/MPI discovery and component include paths |
| [Testing.cmake](Testing.cmake) | GoogleTest/MPI registration, smoke tests and ABI audits | Before registering tests |

Implementation modules live in `options/`, `targets/`, `dependencies/`,
`linkage/`, `profiling/`, and `testing/`. Linker maps are in `linkage/maps/`;
standalone ABI scripts are in `testing/abi/`. Component CMake files use the
entry files and helper commands instead of including these details directly.

Helper commands retain the `simplefluid_` namespace, and target names, cache
options, and ELF symbol-version names retain their established names.

`SIMPLEFLUID_REUSE_TRILINOS_BACKEND=/absolute/path/libSimpleFluidTrilinosBackend.so`
imports an existing shared runtime when building against static Trilinos. It
requires `SIMPLEFLUID_WRAP_STATIC_TRILINOS=ON` and a backend compatible with the
selected Trilinos installation, compiler and MPI. This lets a multiphysics host
rebuild SimpleFluid components while retaining the same runtime used by another
solver. The backend ownership/export tests still run; no consumer may acquire
static Trilinos archives. Leaving the option empty builds the backend normally.

## Focused debug symbols

`SIMPLEFLUID_DEBUG_SYMBOLS` and `SIMPLEFLUID_DEBUG_SYMBOLS_<MODULE>` accept
`INHERIT`, `ON`, or `OFF`; all default to `INHERIT`. They affect **Debug and
RelWithDebInfo**; Release is unchanged. `ON` appends `-g`, `OFF` appends `-g0`,
and a fully inherited setting leaves the compiler's existing flags unchanged.
GCC, Clang and AppleClang support
overrides; other compilers reject requested overrides during configuration.
Module selection preserves optimization, frame pointers, `NDEBUG`, and strict
floating-point source options.

`SIMPLEFLUID_SPLIT_DEBUG_INFO` defaults to `ON` on Linux with GCC or Clang and
`OFF` elsewhere. Configuration checks `-gsplit-dwarf` and GNU ld.bfd/LLD
linker-script support; use `OFF` with unsupported linkers such as gold.
It adds that compile-only flag privately to eligible project targets and shared
PCH providers in Debug and RelWithDebInfo; `-gsplit-dwarf` is not passed at link time.
For eligible targets, debug information stays in `.dwo` files beside the object
files; static-library consumers reuse those files, while executables and shared
libraries retain small skeleton references. This avoids copying each linked
archive object's full debug information into every unit-test binary. It does not
deduplicate header/template information emitted by separate translation units
or rebuild imported dependencies. Native executable/shared-library links omit
the optional `.debug_gnu_pubnames` and `.debug_gnu_pubtypes` accelerator tables,
which otherwise duplicate large PCH name indexes. Full names/types remain in
the `.dwo` files; DWARF skeletons, line tables and addresses remain in the binary.
This uses a supplemental GNU ld/LLD linker script and can make initial debugger
name lookup slower. IPO links and split-mode opt-out do not apply that script.

Split storage and symbol selection are independent: `-gsplit-dwarf` does not
enable debug information, and `OFF` selections still append `-g0` (the compiler
may emit an empty `.dwo`). Keep the `.dwo` files and their build-tree paths
available to the debugger; copying a binary alone is insufficient. Set
`SIMPLEFLUID_SPLIT_DEBUG_INFO=OFF` for conventional embedded debug information.

Split mode preserves optimization and IPO/LTO policy. A target is eligible only
when IPO is disabled for the active configuration: its
`INTERPROCEDURAL_OPTIMIZATION_<CONFIG>` property takes precedence over the
general `INTERPROCEDURAL_OPTIMIZATION` property. IPO targets retain conventional
embedded DWARF because GCC LTO cannot provide reusable split DWARF and Clang LTO
creates per-link debug bundles. Storage follows this target-level decision even
when a source's numerical options include `-fno-lto`.

With the defaults, Debug uses reusable split information, while RelWithDebInfo
targets with LTO retain embedded information. Set `SIMPLEFLUID_ENABLE_LTO=OFF`
to obtain reusable split information in both configurations for project targets
governed by that option. PGO remains available; split mode is part of profile
provenance, so existing profiles require fresh training.

| Module | Compiled scope and fallback order before the global setting |
| --- | --- |
| `CORE` | Database and project runtime/ABI bridge objects |
| `MESH` | Mesh archive sources |
| `IO`, `PARALLEL` | Their sources inside the mesh archive; then `MESH` |
| `PLIC`, `SWEPT_VOLUME` | Their geometry source paths, across Mesh and FVM; then the owning `MESH` or `FVM` |
| `FIELDS`, `FVM` | Remaining sources in those components |
| `EQUATIONS` | Equation sources inside the solver DSO; then `SOLVERS` |
| `SOLVERS` | Remaining solver sources |
| `NOX` | The `NoxNonlinearSolver.cc` adapter, including its disabled-feature fallback; then `SOLVERS` |
| `EXPERIMENTAL_VOF` | The optional VoF solver source groups; then `SOLVERS` |
| `IF97` | Optional `IF97Water.cc`; then `MATERIALS` |
| `MATERIALS` | Material implementations (currently IF97); header-only material interfaces emit symbols in their consumers |
| `TESTS` | Translation units compiled by unit-test targets |
| `BENCHMARKS`, `EXAMPLES` | Benchmark and example targets, including the optional main application |

The first non-`INHERIT` setting wins: explicit source module, its fallback
component from the table, target module, then global. Tests that compile a
registered production source directly retain its explicit source-module chain
before `TESTS`. Header and template code follows the translation unit that emits
it; these controls cannot isolate symbols originating in included headers.
They neither enable optional features nor rebuild imported dependencies.

For example, configure the existing compatible GCC tree with debug information
only in the swept-volume translation units, and build the general VoF fixture:

```sh
cmake -S . -B build/gcc \
  -DSIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF=ON \
  -DSIMPLEFLUID_DEBUG_SYMBOLS=OFF \
  -DSIMPLEFLUID_DEBUG_SYMBOLS_SWEPT_VOLUME=ON
cmake --build build/gcc --config RelWithDebInfo \
  --target testGeneralVoFSolver --parallel 4
```

Other module overrides in an existing cache still take precedence; set those
back to `INHERIT` for this focused selection. Reset the global and module entries
to `INHERIT` to restore the original symbol policy.

The global flag applies to project targets and shared PCH providers together.
Sources whose module setting differs from the global setting append their flag
after existing numerical source options and skip shared PCH reuse. CMake's
`SKIP_PRECOMPILE_HEADERS` source property is not configuration-specific, so such
sources bypass PCH in every configuration, while symbol flags remain limited to
Debug and RelWithDebInfo. Global-only overrides preserve PCH reuse.

Component lists register mixed-owner sources with
`simplefluid_debug_symbols_sources("CHILD;PARENT" ...)` and target defaults with
`simplefluid_debug_symbols_target(target MODULE)`. Keep those registrations with
the owning source lists when moving or adding sources. The top-level finalizer
runs after all source numerical properties, appending options in each target's
directory scope. New optional components should extend the module list in
`profiling/DebugSymbols.cmake` and register their compiled ownership here.
