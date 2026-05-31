cmake_minimum_required(VERSION 3.16)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Trilinos REQUIRED COMPONENTS Kokkos Teuchos Tpetra STKIO STKMesh STKTopology STKUtil Belos Zoltan2)
find_package(MPI REQUIRED COMPONENTS CXX)

set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrecompiledHeaders.hh")

find_package(GTest)

enable_testing()
include(GoogleTest)

add_subdirectory(src)
