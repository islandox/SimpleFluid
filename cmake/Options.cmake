# User-facing cache defaults; include before CTest and dependency discovery.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/options/Defaults.cmake")

# VoF solvers remain unqualified and are excluded from the supported default build.
option(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF
       "Reserved: VoF is disabled pending migration and qualification" OFF)
if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
    message(FATAL_ERROR
        "VoF is disabled during the general reacting-flow refactor. "
        "The retained experimental sources require a separate configuration/API migration "
        "and qualification on complete test cases before they can be enabled. "
        "Leave SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF=OFF; shared geometry and FVM utilities remain available.")
endif()
