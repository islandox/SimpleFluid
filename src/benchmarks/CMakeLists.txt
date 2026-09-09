add_library(SimpleFluidBenchmarkSupport STATIC
    BenchmarkSupport.cc
)
add_library(SimpleFluid::BenchmarkSupport ALIAS SimpleFluidBenchmarkSupport)
simplefluid_configure_library(SimpleFluidBenchmarkSupport)
simplefluid_enable_native_archive_fallback(SimpleFluidBenchmarkSupport)
target_link_libraries(SimpleFluidBenchmarkSupport PUBLIC SimpleFluid::Solvers)
set_target_properties(SimpleFluidBenchmarkSupport PROPERTIES POSITION_INDEPENDENT_CODE ON)
simplefluid_lib_reuse_precompiled_header(SimpleFluidBenchmarkSupport)
simplefluid_require_complete_pgo(SimpleFluidBenchmarkSupport)

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE SIMPLEFLUID_GIT_COMMIT_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT SIMPLEFLUID_GIT_COMMIT_VALUE)
    set(SIMPLEFLUID_GIT_COMMIT_VALUE "unknown")
endif()
execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE SIMPLEFLUID_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(SIMPLEFLUID_GIT_STATUS)
    set(SIMPLEFLUID_GIT_DIRTY_VALUE "true")
else()
    set(SIMPLEFLUID_GIT_DIRTY_VALUE "false")
endif()

add_executable(simplefluid_benchmark simplefluid_benchmark.cc)
target_link_libraries(
    simplefluid_benchmark PRIVATE SimpleFluid::BenchmarkSupport)
simplefluid_exe_reuse_precompiled_header(simplefluid_benchmark)
target_compile_definitions(simplefluid_benchmark PRIVATE
    SIMPLEFLUID_GIT_COMMIT="${SIMPLEFLUID_GIT_COMMIT_VALUE}"
    SIMPLEFLUID_GIT_DIRTY="${SIMPLEFLUID_GIT_DIRTY_VALUE}"
    SIMPLEFLUID_BUILD_TYPE="$<CONFIG>"
    SIMPLEFLUID_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
)
simplefluid_require_complete_pgo(simplefluid_benchmark)

if(BUILD_TESTING)
    add_subdirectory(unitTests)

    add_test(
        NAME benchmark_debug_regression
        COMMAND simplefluid_benchmark
            --preset debug-small
            --warmups 1
            --repetitions 3
            --baseline "${CMAKE_CURRENT_SOURCE_DIR}/baselines/debug-small-serial.csv"
            --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_debug_regression_$<CONFIG>.csv"
    )
    set_tests_properties(benchmark_debug_regression PROPERTIES
        LABELS "benchmark;performance"
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    set(_benchmark_mpi_preflags)
    if(MPIEXEC_EXECUTABLE)
        execute_process(
            COMMAND "${MPIEXEC_EXECUTABLE}" --version
            OUTPUT_VARIABLE _benchmark_mpi_version
            ERROR_VARIABLE _benchmark_mpi_version
        )
        if(_benchmark_mpi_version MATCHES "Open MPI")
            list(APPEND _benchmark_mpi_preflags
                -x OMPI_MCA_pml=ob1
                -x OMPI_MCA_btl=self,sm
            )
        endif()
    endif()

    simplefluid_add_mpi_test(
        NAME benchmark_mpi_regression_2procs
        TARGET simplefluid_benchmark
        PROCS 2
        PREFLAGS ${_benchmark_mpi_preflags}
        ARGS
            --preset debug-small
            --case lid_driven_cavity
            --warmups 1
            --repetitions 3
            --baseline "${CMAKE_CURRENT_SOURCE_DIR}/baselines/debug-small-mpi-2.csv"
            --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_mpi_regression_$<CONFIG>.csv"
        LABELS "benchmark;performance;mpi"
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# Deliberately opt-in: no timing or RSS threshold is a correctness test.
add_executable(region_mesh_scaling region_mesh_scaling.cc)
target_link_libraries(region_mesh_scaling PRIVATE SimpleFluid::Solvers)
simplefluid_exe_reuse_precompiled_header(region_mesh_scaling)
simplefluid_exclude_from_pgo_sources(region_mesh_scaling.cc)

# Opt-in memory benchmark: built only by explicitly requesting this target.
# Deliberately not registered in default CTest or run with two backends in-process.
add_executable(coupled_operator_memory EXCLUDE_FROM_ALL coupled_operator_memory.cc)
target_link_libraries(coupled_operator_memory PRIVATE SimpleFluid::Solvers)
simplefluid_exe_reuse_precompiled_header(coupled_operator_memory)
target_compile_definitions(coupled_operator_memory PRIVATE
    SIMPLEFLUID_GIT_COMMIT="${SIMPLEFLUID_GIT_COMMIT_VALUE}"
    SIMPLEFLUID_GIT_DIRTY="${SIMPLEFLUID_GIT_DIRTY_VALUE}"
    SIMPLEFLUID_BUILD_TYPE="$<CONFIG>"
    SIMPLEFLUID_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
