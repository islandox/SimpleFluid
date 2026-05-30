cmake_minimum_required(VERSION 3.16)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Trilinos REQUIRED COMPONENTS Kokkos Teuchos Tpetra STKIO STKMesh STKTopology STKUtil Belos Zoltan2)
find_package(MPI REQUIRED COMPONENTS CXX)

# Add source files
set(SOURCES "src/geometry/Mesh.cc"
            "src/geometry/STKMesh.cc"
            "src/geometry/MeshFactory.cc"
            "src/io/VTUWriter.cc"
            "src/parallel/MeshPartitioner.cc")

# Create library or executable
add_library(SimpleFluid STATIC ${SOURCES})

target_include_directories(SimpleFluid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(SimpleFluid PUBLIC Trilinos::all_selected_libs MPI::MPI_CXX Zoltan2Core::all_libs)

set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrecompiledHeaders.hh")
target_precompile_headers(SimpleFluid PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)

function(simplefluid_reuse_precompiled_header target_name)
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluid)
endfunction()


find_package(GTest)

enable_testing()
include(GoogleTest)

add_subdirectory(src)
