# CMake helpers

Include the main entry files from project or component CMake lists:

| Entry file | Purpose | Include timing |
| --- | --- | --- |
| [Options.cmake](Options.cmake) | Cache defaults and feature switches | Before CTest and dependency discovery |
| [Profiling.cmake](Profiling.cmake) | PGO setup and source/target coverage helpers | After options, before adding targets |
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
