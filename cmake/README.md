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
