set(SIMPLEFLUID_PUBLIC_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Trilinos_TPL_INCLUDE_DIRS}
)
if(SIMPLEFLUID_ENABLE_NOX)
    # Keep shared/executable PCH preprocessing consistent with their consumers.
    add_compile_definitions(SIMPLEFLUID_ENABLE_NOX=1)
endif()

# Shared build policy is defined once, before component directories consume it.
include("${PROJECT_SOURCE_DIR}/cmake/Build.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/Testing.cmake")
simplefluid_add_precompiled_headers()

add_subdirectory(dataclass)
add_subdirectory(io)
add_subdirectory(parallel)
add_subdirectory(geometry)
add_subdirectory(fields)
add_subdirectory(FVM)
add_subdirectory(equations)
add_subdirectory(solvers)

simplefluid_add_lto_compatibility()
simplefluid_configure_shared_linkage(SimpleFluidFVM SimpleFluidSolvers)
simplefluid_add_abi_tests()

# Preserve the original target as an umbrella for downstream users.
add_library(SimpleFluid INTERFACE)
add_library(SimpleFluid::SimpleFluid ALIAS SimpleFluid)
target_link_libraries(SimpleFluid INTERFACE SimpleFluidSolvers)

if(SIMPLEFLUID_ENABLE_IF97)
    add_subdirectory(materials)
    target_link_libraries(SimpleFluid INTERFACE SimpleFluidIF97)
endif()

# The Shiri workload covers these complete production targets. Mesh-specific
# coverage remains with the geometry component's source list.
foreach(component IN ITEMS SimpleFluidCore SimpleFluidFVM SimpleFluidSolvers)
    simplefluid_require_complete_pgo(${component})
endforeach()

# Only define the optional application when main.cc contains an entry point.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cc")
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/main.cc" SIMPLEFLUID_MAIN_SIZE)
    if(SIMPLEFLUID_MAIN_SIZE GREATER 0)
        add_executable(SimpleFluid_app main.cc)
        target_link_libraries(SimpleFluid_app PRIVATE SimpleFluid)
        target_precompile_headers(
            SimpleFluid_app REUSE_FROM SimpleFluidExecutablePch)
        simplefluid_require_complete_pgo(SimpleFluid_app)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(dataclass/unitTests)
    add_subdirectory(geometry/unitTests)
    add_subdirectory(parallel/unitTests)
    add_subdirectory(fields/unitTests)
    add_subdirectory(FVM/unitTests)
    add_subdirectory(equations/unitTests)
    add_subdirectory(solvers/unitTests)
    add_subdirectory(io/unitTests)
    if(SIMPLEFLUID_ENABLE_IF97)
        add_subdirectory(materials/unitTests)
    endif()
endif()

if(SIMPLEFLUID_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Retain the established verification output paths after colocation.
set(SIMPLEFLUID_EXAMPLE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
add_subdirectory(examples)
