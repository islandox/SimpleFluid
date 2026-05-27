cmake_minimum_required(VERSION 3.10)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Trilinos REQUIRED COMPONENTS Kokkos Teuchos Tpetra STKIO STKMesh STKTopology STKUtil Belos)
find_package(MPI REQUIRED COMPONENTS CXX)

# Add source files
set(SOURCES "geometry/Mesh.cc"
            "geometry/STKMesh.cc"
            "geometry/MeshFactory.cc")

# Create library or executable
add_library(SimpleFluid STATIC ${SOURCES})

target_include_directories(SimpleFluid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SimpleFluid PUBLIC Trilinos::all_selected_libs MPI::MPI_CXX)

# Optional: Add executable if main.cpp exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
    add_executable(SimpleFluid_app ${SOURCES})
    target_include_directories(SimpleFluid_app PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

find_package(GTest)

enable_testing()
include(GoogleTest)

add_subdirectory(dataclass/unitTests)
add_subdirectory(geometry/unitTests)
add_subdirectory(fields/unitTests)
add_subdirectory(operators/unitTests)
add_subdirectory(equations/unitTests)
add_subdirectory(solvers/unitTests)

add_executable(natural_convection_box examples/natural_convection_box.cc)
target_link_libraries(natural_convection_box PRIVATE SimpleFluid)

add_executable(natural_convection_cylinder examples/natural_convection_cylinder.cc)
target_link_libraries(natural_convection_cylinder PRIVATE SimpleFluid)

add_executable(natural_convection_sphere examples/natural_convection_sphere.cc)
target_link_libraries(natural_convection_sphere PRIVATE SimpleFluid)
