cmake_minimum_required(VERSION 3.28)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(
    "-fno-semantic-interposition"
    SIMPLEFLUID_CXX_HAS_FNO_SEMANTIC_INTERPOSITION)

option(SIMPLEFLUID_ENABLE_COVERAGE
       "Build with compiler coverage instrumentation" OFF)
option(SIMPLEFLUID_BUILD_BENCHMARKS
       "Build performance and solver-regression benchmarks" ON)
option(SIMPLEFLUID_BUILD_DOCS
       "Enable the Doxygen API-documentation target" OFF)
option(SIMPLEFLUID_ENABLE_LTO
       "Enable link-time optimization in Release and RelWithDebInfo" ON)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/profileGuidedOptimization.cmake")

if(SIMPLEFLUID_ENABLE_LTO AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
    if(POLICY CMP0138)
        cmake_policy(SET CMP0138 NEW)
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT SIMPLEFLUID_CXX_IPO_SUPPORTED
        OUTPUT SIMPLEFLUID_CXX_IPO_ERROR
        LANGUAGES CXX)
    if(NOT SIMPLEFLUID_CXX_IPO_SUPPORTED)
        message(FATAL_ERROR
            "SIMPLEFLUID_ENABLE_LTO requires compiler IPO support: "
            "${SIMPLEFLUID_CXX_IPO_ERROR}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

    # GNU ld cannot safely resolve every hidden archive definition when
    # Clang ThinLTO is combined with the GCC-built static Trilinos libraries.
    # LLD is required together with the non-LTO compatibility object below.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        check_cxx_compiler_flag(
            "-fuse-ld=lld" SIMPLEFLUID_CXX_HAS_LLD_LINKER)
        if(NOT SIMPLEFLUID_CXX_HAS_LLD_LINKER)
            message(FATAL_ERROR
                "Clang LTO with the static Trilinos build requires LLD")
        endif()
        add_link_options(
            "$<$<CONFIG:Release,RelWithDebInfo>:-fuse-ld=lld>")
    endif()
elseif(SIMPLEFLUID_ENABLE_LTO AND SIMPLEFLUID_ENABLE_COVERAGE)
    message(STATUS "LTO is disabled for the coverage-instrumented build")
endif()

simplefluid_configure_profile_guided_optimization()

set(SIMPLEFLUID_ENABLE_CXX_MODULES_DEFAULT ON)
option(SIMPLEFLUID_ENABLE_CXX_MODULES
       "Build C++ module interfaces for module-capable toolchains"
       ${SIMPLEFLUID_ENABLE_CXX_MODULES_DEFAULT})
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Trilinos 17 has fallback template bodies that Clang diagnoses eagerly
    # against libstdc++'s C++23 string(nullptr_t) deletion.
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fdelayed-template-parsing>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-delayed-template-parsing-in-cxx20>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>"
    )
endif()

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

if(SIMPLEFLUID_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    configure_file(
        "${PROJECT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_BINARY_DIR}/Doxyfile"
        @ONLY
    )
    add_custom_target(docs
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_BINARY_DIR}/docs/doxygen"
        COMMAND Doxygen::doxygen "${CMAKE_BINARY_DIR}/Doxyfile"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating SimpleFluid API documentation"
        VERBATIM
    )
endif()

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
