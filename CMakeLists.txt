cmake_minimum_required(VERSION 3.16)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Trilinos REQUIRED COMPONENTS Kokkos Teuchos Tpetra STKIO STKMesh STKTopology STKUtil Belos Zoltan2)
find_package(MPI REQUIRED COMPONENTS CXX)

# Add source files
set(SOURCES "geometry/Mesh.cc"
            "geometry/STKMesh.cc"
            "geometry/MeshFactory.cc"
            "io/VTUWriter.cc"
            "parallel/MeshPartitioner.cc")

# Create library or executable
add_library(SimpleFluid STATIC ${SOURCES})

target_include_directories(SimpleFluid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SimpleFluid PUBLIC Trilinos::all_selected_libs MPI::MPI_CXX Zoltan2Core::all_libs)

set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrecompiledHeaders.hh")
target_precompile_headers(SimpleFluid PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)

function(simplefluid_reuse_precompiled_header target_name)
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluid)
endfunction()

# Optional: Add executable if main.cpp exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
    add_executable(SimpleFluid_app ${SOURCES})
    target_include_directories(SimpleFluid_app PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(SimpleFluid_app PRIVATE Trilinos::all_selected_libs MPI::MPI_CXX)
    simplefluid_reuse_precompiled_header(SimpleFluid_app)
endif()

find_package(GTest)

enable_testing()
include(GoogleTest)

add_subdirectory(dataclass/unitTests)
add_subdirectory(geometry/unitTests)
add_subdirectory(parallel/unitTests)
add_subdirectory(fields/unitTests)
add_subdirectory(operators/unitTests)
add_subdirectory(equations/unitTests)
add_subdirectory(solvers/unitTests)
add_subdirectory(io/unitTests)

add_executable(natural_convection_box examples/natural_convection_box.cc)
target_link_libraries(natural_convection_box PRIVATE SimpleFluid)
simplefluid_reuse_precompiled_header(natural_convection_box)

add_executable(natural_convection_cylinder examples/natural_convection_cylinder.cc)
target_link_libraries(natural_convection_cylinder PRIVATE SimpleFluid)
simplefluid_reuse_precompiled_header(natural_convection_cylinder)

add_executable(natural_convection_sphere examples/natural_convection_sphere.cc)
target_link_libraries(natural_convection_sphere PRIVATE SimpleFluid)
simplefluid_reuse_precompiled_header(natural_convection_sphere)

add_executable(natural_convection_boundary_layer_box examples/natural_convection_boundary_layer_box.cc)
target_link_libraries(natural_convection_boundary_layer_box PRIVATE SimpleFluid)
simplefluid_reuse_precompiled_header(natural_convection_boundary_layer_box)
