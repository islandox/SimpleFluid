# Component target, dependency, PCH, and shared-library build helpers.
# Include after Trilinos/MPI discovery and the component include directories.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/targets/Helpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dependencies/Runtime.cmake")
set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_LIST_DIR}/targets/PrecompiledHeaders.hh")
include("${CMAKE_CURRENT_LIST_DIR}/targets/PrecompiledHeaders.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linkage/SharedLibraries.cmake")
