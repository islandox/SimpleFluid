
# Optional: Add executable if main.cpp exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
    add_executable(SimpleFluid_app ${SOURCES})
    target_include_directories(SimpleFluid_app PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
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
add_subdirectory(FVM/unitTests)
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