set(SIMPLEFLUID_PUBLIC_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Trilinos_TPL_INCLUDE_DIRS}
)

set(SIMPLEFLUID_EXTERNAL_LIBRARIES
    Trilinos::all_selected_libs
    MPI::MPI_CXX
    Zoltan2Core::all_libs
    ${Trilinos_TPL_LIBRARIES}
)

function(simplefluid_configure_library target_name)
    target_include_directories(${target_name} PUBLIC
        ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
    )
    target_link_libraries(${target_name} PUBLIC
        ${SIMPLEFLUID_EXTERNAL_LIBRARIES}
    )
endfunction()

# Keep PCH ownership independent from the module dependency graph.
add_library(SimpleFluidPch STATIC
    ${PROJECT_SOURCE_DIR}/cmake/PrecompiledHeaders.cc
)
simplefluid_configure_library(SimpleFluidPch)
target_precompile_headers(SimpleFluidPch PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)
set_target_properties(SimpleFluidPch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Executables are compiled without `-fPIC`, so GCC cannot reuse the
# position-independent library PCH above.  Build one matching executable PCH
# instead of regenerating the same header for every test and example target.
add_library(SimpleFluidExecutablePch STATIC
    ${PROJECT_SOURCE_DIR}/cmake/PrecompiledHeaders.cc
)
simplefluid_configure_library(SimpleFluidExecutablePch)
target_precompile_headers(SimpleFluidExecutablePch PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)
set_target_properties(
    SimpleFluidExecutablePch
    PROPERTIES POSITION_INDEPENDENT_CODE OFF)

# Low-level data structures and the runtime database.
add_library(SimpleFluidCore STATIC
    dataclass/Database.cc
)
add_library(SimpleFluid::Core ALIAS SimpleFluidCore)
simplefluid_configure_library(SimpleFluidCore)
set_target_properties(SimpleFluidCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Mesh owns geometry, partitioning, and mesh output. These implementations are
# kept together because STKMesh and MeshHandle directly use both facilities.
add_library(SimpleFluidMesh STATIC
    geometry/BoundaryLayerMeshFactory.cc
    geometry/Mesh.cc
    geometry/MeshHandle.cc
    geometry/STKMesh.cc
    geometry/MeshFactory.cc
    geometry/mesh/LocalGlobalIndexer.cc
    geometry/mesh/OrthogonalLocalGlobalIndexer.cc
    geometry/mesh/PartitionedMeshBase.cc
    geometry/mesh/OrthoMeshPartitioner.cc
    geometry/mesh/OrthoMeshTopo.cc
    geometry/mesh/OrthogonalCartesian3D.cc
    geometry/mesh/OrthogonalCylindrial3D.cc
    geometry/mesh/FrontalDelaunay2D.cc
    geometry/mesh/SemiStructMeshTopo.cc
    geometry/mesh/SemiStructuredXY_Z.cc
    geometry/mesh/UnstructuredMesh.cc
    io/VTUWriter.cc
    parallel/MeshPartitioner.cc
)
add_library(SimpleFluid::Mesh ALIAS SimpleFluidMesh)
simplefluid_configure_library(SimpleFluidMesh)
set_target_properties(SimpleFluidMesh PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SimpleFluidMesh PUBLIC SimpleFluidCore)

# Fields and FVM are currently header-only. Interface targets still isolate
# their dependency surfaces and let consumers avoid linking higher layers.
add_library(SimpleFluidFields INTERFACE)
add_library(SimpleFluid::Fields ALIAS SimpleFluidFields)
target_include_directories(SimpleFluidFields INTERFACE
    ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
)
target_link_libraries(SimpleFluidFields INTERFACE SimpleFluidMesh)

add_library(SimpleFluidFVM INTERFACE)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
target_include_directories(SimpleFluidFVM INTERFACE
    ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
)
target_link_libraries(SimpleFluidFVM INTERFACE SimpleFluidFields)

# Trilinos installations used by SimpleFluid commonly provide static archives
# that are not position-independent (notably MueLu).  Keep the compiled upper
# layers static so a clean build never embeds those archives in a shared object.
# The public SimpleFluid::* aliases below preserve the downstream link API.
add_library(SimpleFluidEquations STATIC
    equations/BoussinesqMomentumEquation.cc
    equations/IncompressibleMomentumEquation.cc
    equations/PressureProjectionEquation.cc
    equations/RadiolyticGasProperties.cc
    equations/TemperatureDiffusionEquation.cc
    equations/turbulence/TurbulenceModel.cc
    equations/turbulence/TurbulenceScalarTransportEquation.cc
    equations/turbulence/TurbulenceWallTreatment.cc
    equations/turbulence/WallDistanceEquation.cc
)
add_library(SimpleFluid::Equations ALIAS SimpleFluidEquations)
simplefluid_configure_library(SimpleFluidEquations)
set_target_properties(SimpleFluidEquations PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SimpleFluidEquations PUBLIC SimpleFluidFVM)

add_library(SimpleFluidSolvers STATIC
    solvers/BoussinesqSolver.cc
    solvers/FluidSolver.cc
)
add_library(SimpleFluid::Solvers ALIAS SimpleFluidSolvers)
simplefluid_configure_library(SimpleFluidSolvers)
set_target_properties(SimpleFluidSolvers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SimpleFluidSolvers PUBLIC SimpleFluidEquations)

# Preserve the original target as an umbrella for downstream users.
add_library(SimpleFluid INTERFACE)
add_library(SimpleFluid::SimpleFluid ALIAS SimpleFluid)
target_link_libraries(SimpleFluid INTERFACE SimpleFluidSolvers)

# Build one PCH and reuse it for all compiled modules and test executables.
target_precompile_headers(SimpleFluidCore REUSE_FROM SimpleFluidPch)
target_precompile_headers(SimpleFluidMesh REUSE_FROM SimpleFluidPch)
target_precompile_headers(SimpleFluidEquations REUSE_FROM SimpleFluidPch)
target_precompile_headers(SimpleFluidSolvers REUSE_FROM SimpleFluidPch)

# Only define the optional application when main.cc contains an entry point.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cc")
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/main.cc" SIMPLEFLUID_MAIN_SIZE)
    if(SIMPLEFLUID_MAIN_SIZE GREATER 0)
        add_executable(SimpleFluid_app main.cc)
        target_link_libraries(SimpleFluid_app PRIVATE SimpleFluid)
        target_precompile_headers(
            SimpleFluid_app REUSE_FROM SimpleFluidExecutablePch)
    endif()
endif()


function(simplefluid_lib_reuse_precompiled_header target_name)
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluidPch)
endfunction()

function(simplefluid_exe_reuse_precompiled_header target_name)
    target_precompile_headers(
        ${target_name} REUSE_FROM SimpleFluidExecutablePch)
endfunction()

add_subdirectory(dataclass/unitTests)
add_subdirectory(geometry/unitTests)
add_subdirectory(parallel/unitTests)
add_subdirectory(fields/unitTests)
add_subdirectory(FVM/unitTests)
add_subdirectory(equations/unitTests)
add_subdirectory(solvers/unitTests)
add_subdirectory(io/unitTests)

if(SIMPLEFLUID_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_executable(natural_convection_box examples/natural_convection_box.cc)
target_link_libraries(natural_convection_box PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(natural_convection_box)

add_executable(natural_convection_cylinder examples/natural_convection_cylinder.cc)
target_link_libraries(natural_convection_cylinder PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(natural_convection_cylinder)

add_executable(natural_convection_sphere examples/natural_convection_sphere.cc)
target_link_libraries(natural_convection_sphere PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(natural_convection_sphere)

add_executable(natural_convection_boundary_layer_box examples/natural_convection_boundary_layer_box.cc)
target_link_libraries(natural_convection_boundary_layer_box PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(natural_convection_boundary_layer_box)

add_executable(natural_convection_shiri examples/natural_convection_shiri.cc)
target_link_libraries(natural_convection_shiri PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(natural_convection_shiri)

add_executable(pitz_daily examples/pitz_daily.cc)
target_link_libraries(pitz_daily PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(pitz_daily)

add_executable(fissile_solution_tank_demo examples/fissile_solution_tank_demo.cc)
target_link_libraries(fissile_solution_tank_demo PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(fissile_solution_tank_demo)

add_executable(
    constant_power_cylinder_vessel
    examples/constant_power_cylinder_vessel.cc)
target_link_libraries(constant_power_cylinder_vessel PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(constant_power_cylinder_vessel)

foreach(EXAMPLE_TARGET IN ITEMS
    natural_convection_box
    natural_convection_cylinder
    natural_convection_sphere
    natural_convection_boundary_layer_box
    natural_convection_shiri
    pitz_daily
    fissile_solution_tank_demo
    constant_power_cylinder_vessel)
  add_test(NAME ${EXAMPLE_TARGET}_smoke COMMAND ${EXAMPLE_TARGET})
  set_tests_properties(${EXAMPLE_TARGET}_smoke PROPERTIES
      LABELS integration
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

set_tests_properties(natural_convection_shiri_smoke PROPERTIES
    ENVIRONMENT
        "SIMPLEFLUID_SHIRI_NR=4;SIMPLEFLUID_SHIRI_NTHETA=4;SIMPLEFLUID_SHIRI_NZ=6;SIMPLEFLUID_SHIRI_STEPS=1")

set_tests_properties(pitz_daily_smoke PROPERTIES
    ENVIRONMENT
        "SIMPLEFLUID_PITZ_MESH_DIVISOR=40;SIMPLEFLUID_PITZ_STEPS=1;SIMPLEFLUID_PITZ_DT=1e-6")
