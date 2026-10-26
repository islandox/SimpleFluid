# User-facing cache defaults; include before CTest and dependency discovery.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/options/Defaults.cmake")

# VoF solvers remain unqualified and are excluded from the supported default build.
option(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF
       "Build experimental VoF solvers, excluding the unmigrated reacting coupler" OFF)
