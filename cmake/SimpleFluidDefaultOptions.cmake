# User-facing configuration defaults. Cache entries and parent-scope options
# supplied by callers take precedence; no defaults are forced into the cache.
include_guard(GLOBAL)

# Features and instrumentation.
option(SIMPLEFLUID_ENABLE_COVERAGE
       "Build with compiler coverage instrumentation" OFF)
option(SIMPLEFLUID_BUILD_BENCHMARKS
       "Build performance and solver-regression benchmarks" ON)
option(SIMPLEFLUID_BUILD_DOCS
       "Enable the Doxygen API-documentation target" OFF)
option(SIMPLEFLUID_ENABLE_IF97
       "Build the optional IAPWS-IF97 water material library" OFF)
option(SIMPLEFLUID_ENABLE_NOX
       "Build the optional NOX/Thyra coupled nonlinear solver" OFF)
option(SIMPLEFLUID_ENABLE_LTO
       "Enable link-time optimization in Release and RelWithDebInfo" ON)
set(_simplefluid_wrap_static_trilinos_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_simplefluid_wrap_static_trilinos_default ON)
endif()
option(SIMPLEFLUID_WRAP_STATIC_TRILINOS
       "Export static Trilinos through one SimpleFluid shared backend (Linux)"
       ${_simplefluid_wrap_static_trilinos_default})
unset(_simplefluid_wrap_static_trilinos_default)

# Testing. CTest consumes BUILD_TESTING after these defaults are established.
option(BUILD_TESTING "Build the testing tree." ON)
set(SIMPLEFLUID_MAX_TEST_PROCS "0" CACHE STRING
    "Maximum MPI processes for tests (0 = unlimited). Set to 2 for free-tier runners.")
if(BUILD_TESTING)
    set(SIMPLEFLUID_GTEST_DISCOVERY_TIMEOUT "30" CACHE STRING
        "Maximum seconds allowed for GoogleTest test discovery")
endif()

# Profile-guided optimization.
set(SIMPLEFLUID_PGO_PHASE "OFF" CACHE STRING
    "Profile-guided optimization phase: OFF, GENERATE, or USE")
set_property(CACHE SIMPLEFLUID_PGO_PHASE
             PROPERTY STRINGS OFF GENERATE USE)
set(SIMPLEFLUID_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH
    "Compiler-specific profile data directory")
set(SIMPLEFLUID_PGO_CONFIG "RelWithDebInfo" CACHE STRING
    "Single optimized configuration trained by PGO")
set_property(CACHE SIMPLEFLUID_PGO_CONFIG
             PROPERTY STRINGS Release RelWithDebInfo)
set(SIMPLEFLUID_PGO_WORKLOAD "natural_convection_shiri" CACHE STRING
    "Production workload represented by the profile data")
set_property(CACHE SIMPLEFLUID_PGO_WORKLOAD
             PROPERTY STRINGS natural_convection_shiri)
