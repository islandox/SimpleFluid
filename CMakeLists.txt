cmake_minimum_required(VERSION 3.16)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_host_system_information(RESULT NCPU QUERY NUMBER_OF_LOGICAL_CORES)

# parallel job pool settings, only available for ninja build.
math(EXPR SIMPLEFLUID_COMPILE_JOBS "${NCPU} / 2")
math(EXPR SIMPLEFLUID_LINK_JOBS "${NCPU} / 5")
if(SIMPLEFLUID_COMPILE_JOBS LESS 1)
    set(SIMPLEFLUID_COMPILE_JOBS 1)
endif()
if(SIMPLEFLUID_LINK_JOBS LESS 1)
    set(SIMPLEFLUID_LINK_JOBS 1)
endif()
set_property(GLOBAL PROPERTY JOB_POOLS
    compile_pool=${SIMPLEFLUID_COMPILE_JOBS}
    link_pool=${SIMPLEFLUID_LINK_JOBS})
set(CMAKE_JOB_POOL_COMPILE compile_pool)
set(CMAKE_JOB_POOL_LINK link_pool)

option(SIMPLEFLUID_ENABLE_COVERAGE
       "Build with compiler coverage instrumentation" OFF)
option(SIMPLEFLUID_BUILD_BENCHMARKS
       "Build performance and solver-regression benchmarks" ON)

set(SIMPLEFLUID_MAX_TEST_PROCS "0" CACHE STRING
    "Maximum MPI processes for tests (0 = unlimited). Set to 2 for free-tier runners.")
if(SIMPLEFLUID_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR
            "SIMPLEFLUID_ENABLE_COVERAGE requires GCC or Clang")
    endif()
endif()

# Profiling flags for RelWithDebInfo: preserve frame pointers for accurate
# stack unwinding in performance profilers (Instruments, perf, etc.)
# Also define NDEBUG to disable expensive debug-only bounds checks while
# keeping debug symbols for the profiler.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")

find_package(Trilinos REQUIRED COMPONENTS
    Kokkos
    Teuchos
    Tpetra
    STKIO
    STKMesh
    STKTopology
    STKUtil
    Belos
    Ifpack2
    MueLu
    Zoltan2
)
find_package(MPI REQUIRED COMPONENTS CXX)

set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrecompiledHeaders.hh")

find_package(GTest)

enable_testing()
include(GoogleTest)

set(SIMPLEFLUID_GTEST_DISCOVERY_TIMEOUT "30" CACHE STRING
    "Maximum seconds allowed for GoogleTest test discovery")

function(simplefluid_discover_tests target)
    gtest_discover_tests(
        ${target}
        DISCOVERY_TIMEOUT ${SIMPLEFLUID_GTEST_DISCOVERY_TIMEOUT}
        ${ARGN})
endfunction()

add_subdirectory(src)
