add_library(SimpleFluidBenchmarkSupport STATIC
    BenchmarkSupport.cc
)
add_library(SimpleFluid::BenchmarkSupport ALIAS SimpleFluidBenchmarkSupport)
simplefluid_configure_library(SimpleFluidBenchmarkSupport)
target_link_libraries(SimpleFluidBenchmarkSupport PUBLIC SimpleFluid::Solvers)
simplefluid_reuse_precompiled_header(SimpleFluidBenchmarkSupport)

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
simplefluid_reuse_precompiled_header(simplefluid_benchmark)
target_compile_definitions(simplefluid_benchmark PRIVATE
    SIMPLEFLUID_GIT_COMMIT="${SIMPLEFLUID_GIT_COMMIT_VALUE}"
    SIMPLEFLUID_GIT_DIRTY="${SIMPLEFLUID_GIT_DIRTY_VALUE}"
    SIMPLEFLUID_BUILD_TYPE="$<CONFIG>"
    SIMPLEFLUID_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
)

add_subdirectory(unitTests)

add_test(
    NAME benchmark_debug_small
    COMMAND simplefluid_benchmark
        --preset debug-small
        --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_debug_small_$<CONFIG>.csv"
)
set_tests_properties(benchmark_debug_small PROPERTIES
    LABELS "benchmark;performance"
    TIMEOUT 300
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(MPIEXEC_EXECUTABLE)
    set(_benchmark_mpi_preflags ${MPIEXEC_PREFLAGS})
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

    add_test(
        NAME benchmark_mpi_smoke_2procs
        COMMAND "${MPIEXEC_EXECUTABLE}" ${_benchmark_mpi_preflags}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:simplefluid_benchmark>"
            --preset debug-small
            --case pressure_velocity
            --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_mpi_smoke_$<CONFIG>.csv"
    )
    set_tests_properties(benchmark_mpi_smoke_2procs PROPERTIES
        LABELS "benchmark;performance;mpi"
        PROCESSORS 2
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
