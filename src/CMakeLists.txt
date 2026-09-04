set(SIMPLEFLUID_PUBLIC_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Trilinos_TPL_INCLUDE_DIRS}
)

set(SIMPLEFLUID_EXTERNAL_LIBRARIES
    Trilinos::all_selected_libs
    Zoltan2Core::all_libs
    ${Trilinos_TPL_LIBRARIES}
    MPI::MPI_CXX
)

function(simplefluid_configure_library target_name)
    target_include_directories(${target_name} PUBLIC
        ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
    )
    target_link_libraries(${target_name} PUBLIC
        ${SIMPLEFLUID_EXTERNAL_LIBRARIES}
    )
endfunction()

# GCC's linker plugin can discard weak definitions from the non-LTO static
# Trilinos archives when a unit test also contributes LTO objects.  Preserve a
# native copy in SimpleFluid's static archives so GNU test executables can use a
# normal non-plugin link while production libraries and applications retain LTO.
function(simplefluid_enable_native_archive_fallback target_name)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND SIMPLEFLUID_ENABLE_LTO
       AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
        target_compile_options(${target_name} PRIVATE
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:-ffat-lto-objects>")
    endif()
endfunction()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[#include <version>
#ifndef _LIBCPP_VERSION
#error "not libc++"
#endif
int main() { return 0; }
]] SIMPLEFLUID_CXX_USES_LIBCXX)

# Apple libc++ requires default visibility for the complete image.  ELF libc++
# keeps SimpleFluid hidden here and gives only the Trilinos header region in the
# shared PCH default visibility; the Linux version script remains the final DSO
# export boundary.
set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET hidden)
set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN YES)
if(APPLE)
    set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET default)
    set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN NO)
endif()

# Keep PCH ownership independent from the module dependency graph.
add_library(SimpleFluidPch STATIC
    ${PROJECT_SOURCE_DIR}/cmake/PrecompiledHeaders.cc
)
simplefluid_configure_library(SimpleFluidPch)
target_precompile_headers(SimpleFluidPch PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)
set_target_properties(SimpleFluidPch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared-library consumers require a PCH compiled with the same visibility
# policy as the owning translation units.
add_library(SimpleFluidSharedPch STATIC
    ${PROJECT_SOURCE_DIR}/cmake/PrecompiledHeaders.cc
)
simplefluid_configure_library(SimpleFluidSharedPch)
target_precompile_headers(SimpleFluidSharedPch PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>"
)
set_target_properties(SimpleFluidSharedPch PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
    VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})

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
simplefluid_enable_native_archive_fallback(SimpleFluidCore)
set_target_properties(SimpleFluidCore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
    VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})

set(SIMPLEFLUID_MESH_SOURCES
    geometry/BoundaryLayerMeshFactory.cc
    geometry/Mesh.cc
    geometry/MeshHandle.cc
    geometry/PlanarALEMeshMotion.cc
    geometry/MeshReorderingFactory.cc
    geometry/SolidSubdomain.cc
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

# Mesh owns geometry, partitioning, and mesh output. These implementations are
# kept together because STKMesh and MeshHandle directly use both facilities.
add_library(SimpleFluidMesh STATIC ${SIMPLEFLUID_MESH_SOURCES})
add_library(SimpleFluid::Mesh ALIAS SimpleFluidMesh)
simplefluid_configure_library(SimpleFluidMesh)
simplefluid_enable_native_archive_fallback(SimpleFluidMesh)
set_target_properties(SimpleFluidMesh PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
    VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})
target_link_libraries(SimpleFluidMesh PUBLIC SimpleFluidCore)

# Fields remain header-only. FVM compiles the deliberately supported template
# pack so consumers do not repeatedly instantiate the large assembly kernels.
add_library(SimpleFluidFields INTERFACE)
add_library(SimpleFluid::Fields ALIAS SimpleFluidFields)
target_include_directories(SimpleFluidFields INTERFACE
    ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
)
target_link_libraries(SimpleFluidFields INTERFACE SimpleFluidMesh)

add_library(SimpleFluidFVM STATIC
    FVM/NonOrthogonalCorrection.cc
    FVM/TransportSystem.cc
)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
simplefluid_configure_library(SimpleFluidFVM)
simplefluid_enable_native_archive_fallback(SimpleFluidFVM)
set_target_properties(SimpleFluidFVM PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
    VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)

set(SIMPLEFLUID_EQUATION_SOURCES
    equations/BoussinesqMomentumEquation.cc
    equations/IncompressibleMomentumEquation.cc
    equations/PressureProjectionEquation.cc
    equations/RadiolyticGasPhysics.cc
    equations/RadiolyticGasProperties.cc
    equations/SolidHeatConductionEquation.cc
    equations/TemperatureDiffusionEquation.cc
    equations/turbulence/TurbulenceModel.cc
    equations/turbulence/TurbulenceScalarTransportEquation.cc
    equations/turbulence/TurbulenceWallTreatment.cc
    equations/turbulence/WallDistanceEquation.cc
)
set(SIMPLEFLUID_SOLVER_SOURCES
    solvers/BoussinesqSolver.cc
    solvers/CoupledPressureVelocitySolver.cc
    solvers/FluidSolver.cc
    solvers/IncompressibleIsothermalSolver.cc
    solvers/PlanarFreeSurfaceModel.cc
    solvers/SteadyStateSearch.cc
)

# Clang ThinLTO can select an incomplete consumer-emitted MueLu vtable COMDAT
# before it sees the complete GCC-built archive definition.  Put a non-LTO
# explicit instantiation ahead of every ThinLTO input on the DSO link line.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
   AND SIMPLEFLUID_ENABLE_LTO
   AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
    add_library(SimpleFluidClangLtoCompatibility OBJECT
        ClangLtoTrilinosCompatibility.cc)
    simplefluid_configure_library(SimpleFluidClangLtoCompatibility)
    set_target_properties(SimpleFluidClangLtoCompatibility PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    simplefluid_exclude_from_pgo_sources(
        ClangLtoTrilinosCompatibility.cc)
endif()

# Static Trilinos installations put vendor implementations, RTTI, and mutable
# process state into every shared object that consumes them.  Keep equations
# and solvers in one physical DSO so the ELF export boundary can localize those
# archive symbols without duplicating them or splitting cross-layer type
# identity.  The logical equation and solver link targets remain separate.
add_library(SimpleFluidSolvers SHARED
    ${SIMPLEFLUID_EQUATION_SOURCES}
    ${SIMPLEFLUID_SOLVER_SOURCES}
)
add_library(SimpleFluid::Solvers ALIAS SimpleFluidSolvers)
simplefluid_configure_library(SimpleFluidSolvers)
set_target_properties(SimpleFluidSolvers PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
    VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})
target_compile_definitions(SimpleFluidSolvers PRIVATE
    SimpleFluidEquations_EXPORTS)
target_link_libraries(SimpleFluidSolvers PUBLIC SimpleFluidFVM)
if(TARGET SimpleFluidClangLtoCompatibility)
    add_dependencies(
        SimpleFluidSolvers SimpleFluidClangLtoCompatibility)
    target_link_options(SimpleFluidSolvers PRIVATE
        "$<$<CONFIG:Release,RelWithDebInfo>:$<TARGET_OBJECTS:SimpleFluidClangLtoCompatibility>>")
endif()

add_library(SimpleFluidEquations INTERFACE)
add_library(SimpleFluid::Equations ALIAS SimpleFluidEquations)
target_link_libraries(SimpleFluidEquations INTERFACE SimpleFluidSolvers)

# Source visibility attributes select the public SimpleFluid API.  The ELF
# version script localizes embedded vendor archives except for a separately
# versioned Kokkos runtime bridge: executables and the DSO must share Kokkos
# initialization, allocation, and callback state.  Bind archive functions
# locally while leaving runtime data canonical across the process.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(SIMPLEFLUID_ELF_VERSION_SCRIPT
        "${PROJECT_SOURCE_DIR}/cmake/SimpleFluidLinux.map")
    target_link_options(SimpleFluidSolvers PRIVATE
        "LINKER:--version-script=${SIMPLEFLUID_ELF_VERSION_SCRIPT}"
        "LINKER:-Bsymbolic-functions")
    set_property(TARGET SimpleFluidSolvers APPEND PROPERTY
        LINK_DEPENDS "${SIMPLEFLUID_ELF_VERSION_SCRIPT}")
endif()

# Preserve the former linker names for source-level consumers.  New binaries
# record libSimpleFluidSolvers as the single implementation SONAME.
if(UNIX)
    set(SIMPLEFLUID_EQUATIONS_LINKER_ALIAS
        "$<TARGET_FILE_DIR:SimpleFluidSolvers>/${CMAKE_SHARED_LIBRARY_PREFIX}SimpleFluidEquations${CMAKE_SHARED_LIBRARY_SUFFIX}")
    add_custom_command(TARGET SimpleFluidSolvers POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E rm -f
                "${SIMPLEFLUID_EQUATIONS_LINKER_ALIAS}"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "$<TARGET_FILE_NAME:SimpleFluidSolvers>"
                "${SIMPLEFLUID_EQUATIONS_LINKER_ALIAS}"
        VERBATIM)
endif()

# Keep the export map enforceable as the static Trilinos dependency set evolves.
if(BUILD_TESTING
   AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Clang-built Trilinos emits substantially more weak Kokkos template
    # definitions in Debug than GCC-built archives.  Both sets are part of
    # the intentional, versioned runtime bridge; retain a tighter bound for
    # GCC while allowing the validated Clang archive shape.
    set(SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING 7000)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING 20000)
    endif()
    set(SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING 64)
    find_program(SIMPLEFLUID_CXXFILT
        NAMES c++filt llvm-cxxfilt
        REQUIRED)
    add_test(NAME simplefluid_elf_export_boundary
        COMMAND "${CMAKE_COMMAND}"
            "-DSIMPLEFLUID_LIBRARY=$<TARGET_FILE:SimpleFluidSolvers>"
            "-DSIMPLEFLUID_NM=${CMAKE_NM}"
            "-DSIMPLEFLUID_READELF=${CMAKE_READELF}"
            "-DSIMPLEFLUID_CXXFILT=${SIMPLEFLUID_CXXFILT}"
            "-DSIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING=${SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING}"
            "-DSIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING=${SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING}"
            "-DSIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES=$<BOOL:${SIMPLEFLUID_CXX_USES_LIBCXX}>"
            -P "${PROJECT_SOURCE_DIR}/cmake/CheckSimpleFluidElfExports.cmake")
    set_tests_properties(simplefluid_elf_export_boundary PROPERTIES
        LABELS abi)
endif()

# GCC's LTO plugin can select, then discard, weak libstdc++ COMDAT definitions
# that are also referenced by the non-LTO STK/Ioss archives.  Supply those
# definitions from one deliberately non-LTO object instead of forcing large
# archive members into each shared library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND SIMPLEFLUID_ENABLE_LTO
   AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
    add_library(SimpleFluidGccLtoCompatibility OBJECT
        GccLtoStdlibCompatibility.cc)
    set_target_properties(SimpleFluidGccLtoCompatibility PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    # Propagate the object from the root static-library interface so Core-only
    # tests as well as DSO consumers receive it before their LTO inputs.
    target_sources(SimpleFluidCore PUBLIC
        "$<$<CONFIG:Release,RelWithDebInfo>:$<TARGET_OBJECTS:SimpleFluidGccLtoCompatibility>>")
    simplefluid_require_complete_pgo(SimpleFluidGccLtoCompatibility)
endif()

# SimpleFluid does not expose ELF symbol interposition as an extension API.
# Let optimized shared-library builds bind their own definitions locally when
# the compiler supports that model.
if(SIMPLEFLUID_CXX_HAS_FNO_SEMANTIC_INTERPOSITION)
    target_compile_options(SimpleFluidSolvers PRIVATE
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-fno-semantic-interposition>")
endif()

# Preserve the original target as an umbrella for downstream users.
add_library(SimpleFluid INTERFACE)
add_library(SimpleFluid::SimpleFluid ALIAS SimpleFluid)
target_link_libraries(SimpleFluid INTERFACE SimpleFluidSolvers)

if(SIMPLEFLUID_ENABLE_IF97)
    add_subdirectory(materials)
    target_link_libraries(SimpleFluid INTERFACE SimpleFluidIF97)
endif()

# Reuse a visibility-compatible PCH for every object embedded in the DSO.
target_precompile_headers(SimpleFluidCore REUSE_FROM SimpleFluidSharedPch)
target_precompile_headers(SimpleFluidMesh REUSE_FROM SimpleFluidSharedPch)
target_precompile_headers(SimpleFluidFVM REUSE_FROM SimpleFluidSharedPch)
target_precompile_headers(SimpleFluidSolvers REUSE_FROM SimpleFluidSharedPch)

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


function(simplefluid_lib_reuse_precompiled_header target_name)
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluidPch)
endfunction()

function(simplefluid_exe_reuse_precompiled_header target_name)
    target_precompile_headers(
        ${target_name} REUSE_FROM SimpleFluidExecutablePch)
endfunction()

if(BUILD_TESTING)
    function(simplefluid_configure_test_executable target_name)
        simplefluid_exe_reuse_precompiled_header(${target_name})

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
           AND SIMPLEFLUID_ENABLE_LTO
           AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
            set_target_properties(${target_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION FALSE
                INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE)
            target_compile_options(${target_name} PRIVATE
                "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:-fno-lto>")
            target_link_options(${target_name} PRIVATE
                "$<$<CONFIG:Release,RelWithDebInfo>:-fno-lto>"
                "$<$<CONFIG:Release,RelWithDebInfo>:-fno-use-linker-plugin>")
        endif()
    endfunction()

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

add_executable(
    planar_free_surface_verification
    examples/planar_free_surface_verification.cc)
target_link_libraries(planar_free_surface_verification PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(planar_free_surface_verification)

# This analytic verification driver is not part of the production PGO training
# workload, so optimize it normally in profile-use builds.
simplefluid_exclude_from_pgo_sources(
    examples/planar_free_surface_verification.cc)

add_executable(
    planar_ale_verification
    examples/planar_ale_verification.cc)
target_link_libraries(planar_ale_verification PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(planar_ale_verification)
simplefluid_exclude_from_pgo_sources(
    examples/planar_ale_verification.cc)

foreach(VERIFICATION_TARGET IN ITEMS dispersed_bubble_verification planar_ale_comparison)
  if(SIMPLEFLUID_ENABLE_IF97)
    add_executable(${VERIFICATION_TARGET} examples/${VERIFICATION_TARGET}.cc)
    target_link_libraries(${VERIFICATION_TARGET} PRIVATE SimpleFluid SimpleFluid::IF97)
    simplefluid_exe_reuse_precompiled_header(${VERIFICATION_TARGET})
    simplefluid_exclude_from_pgo_sources(examples/${VERIFICATION_TARGET}.cc)
  else()
    # Select a separate source instead of changing compile definitions on a
    # target that reuses the shared executable PCH. Default builds retain a
    # useful launcher diagnostic without linking or discovering IF97.
    add_executable(${VERIFICATION_TARGET} examples/if97_verification_unavailable.cc)
  endif()
  if(BUILD_TESTING AND SIMPLEFLUID_ENABLE_IF97)
    foreach(VERIFICATION_MODE IN ITEMS steady transient)
      add_test(
        NAME ${VERIFICATION_TARGET}_${VERIFICATION_MODE}
        COMMAND ${VERIFICATION_TARGET} --mode ${VERIFICATION_MODE}
          --output "${CMAKE_CURRENT_BINARY_DIR}/verification/${VERIFICATION_TARGET}/${VERIFICATION_MODE}")
      set_tests_properties(${VERIFICATION_TARGET}_${VERIFICATION_MODE} PROPERTIES
        LABELS "verification;conservation"
        TIMEOUT 120
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endforeach()
  endif()
endforeach()

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

# The Shiri training workload covers these complete production targets.
simplefluid_require_complete_pgo(SimpleFluidCore)
simplefluid_require_complete_pgo(SimpleFluidFVM)
simplefluid_require_complete_pgo(SimpleFluidSolvers)

# Require profiles for every Mesh archive member linked into Shiri. The six
# alternate-path objects are compiled with ordinary LTO optimization in USE
# builds instead of silently applying profile-use without matching data.
set(SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES
    geometry/BoundaryLayerMeshFactory.cc
    geometry/MeshReorderingFactory.cc
    geometry/PlanarALEMeshMotion.cc
    geometry/SolidSubdomain.cc
    geometry/mesh/OrthogonalLocalGlobalIndexer.cc
    geometry/mesh/PartitionedMeshBase.cc
    geometry/mesh/SemiStructuredXY_Z.cc)
set(SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES ${SIMPLEFLUID_MESH_SOURCES})
list(REMOVE_ITEM SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES
    ${SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES})
simplefluid_require_complete_pgo_sources(
    ${SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES})
simplefluid_exclude_from_pgo_sources(
    ${SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES})

add_executable(pitz_daily examples/pitz_daily.cc)
target_link_libraries(pitz_daily PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(pitz_daily)

if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME pitz_daily_profile_comparator_fixture
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/pitzDaily/test_compare_profiles.py")
    set_tests_properties(pitz_daily_profile_comparator_fixture PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
    add_test(
      NAME openfoam_verification_comparator
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/test_compare_verification.py")
    set_tests_properties(openfoam_verification_comparator PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
  else()
    message(WARNING
      "Python 3 was not found; the OpenFOAM comparator fixture tests are disabled")
  endif()
endif()

add_executable(fissile_solution_tank_demo examples/fissile_solution_tank_demo.cc)
target_link_libraries(fissile_solution_tank_demo PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(fissile_solution_tank_demo)

add_executable(
    fissile_solution_tank_sst
    examples/fissile_solution_tank_sst.cc)
target_link_libraries(fissile_solution_tank_sst PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(fissile_solution_tank_sst)

add_executable(
    constant_power_cylinder_vessel
    examples/constant_power_cylinder_vessel.cc)
target_link_libraries(constant_power_cylinder_vessel PRIVATE SimpleFluid)
simplefluid_exe_reuse_precompiled_header(constant_power_cylinder_vessel)

# Any production executable requested from a USE tree must have been present in
# the training run. The presets build Shiri only; building another application
# later therefore fails loudly until its workload is added to training.
set(SIMPLEFLUID_EXAMPLE_TARGETS
    planar_free_surface_verification
    planar_ale_verification
    natural_convection_box
    natural_convection_cylinder
    natural_convection_sphere
    natural_convection_boundary_layer_box
    natural_convection_shiri
    pitz_daily
    fissile_solution_tank_demo
    fissile_solution_tank_sst
    constant_power_cylinder_vessel)
foreach(EXAMPLE_TARGET IN LISTS SIMPLEFLUID_EXAMPLE_TARGETS)
  simplefluid_require_complete_pgo(${EXAMPLE_TARGET})
endforeach()

if(BUILD_TESTING)
  foreach(EXAMPLE_TARGET IN LISTS SIMPLEFLUID_EXAMPLE_TARGETS)
    add_test(NAME ${EXAMPLE_TARGET}_smoke COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_TARGET}_smoke PROPERTIES
        LABELS integration
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()

  set(SIMPLEFLUID_PLANAR_FREE_SURFACE_VERIFICATION_SUBCASES
      uniformHeating
      gasGeneration
      completeEscape
      closedHeadspace
      boilingMassLoss)
  foreach(SUBCASE IN LISTS SIMPLEFLUID_PLANAR_FREE_SURFACE_VERIFICATION_SUBCASES)
    add_test(
        NAME planar_free_surface_${SUBCASE}
        COMMAND planar_free_surface_verification ${SUBCASE})
    set_tests_properties(planar_free_surface_${SUBCASE} PROPERTIES
        LABELS verification
        TIMEOUT 10
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()

  set(SIMPLEFLUID_PLANAR_ALE_VERIFICATION_SUBCASES
      aleUniformHeating
      aleGasGeneration
      aleCompleteEscape
      aleFailureRollback)
  foreach(SUBCASE IN LISTS SIMPLEFLUID_PLANAR_ALE_VERIFICATION_SUBCASES)
    add_test(
        NAME planar_ale_${SUBCASE}
        COMMAND planar_ale_verification ${SUBCASE})
    set_tests_properties(planar_ale_${SUBCASE} PROPERTIES
        LABELS "verification;ale;conservation"
        TIMEOUT 30
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()

  set_tests_properties(natural_convection_shiri_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_SHIRI_NR=4;SIMPLEFLUID_SHIRI_NTHETA=4;SIMPLEFLUID_SHIRI_NZ=6;SIMPLEFLUID_SHIRI_STEPS=1")

  set_tests_properties(fissile_solution_tank_sst_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_TANK_RZ_SPACING=0.05;SIMPLEFLUID_TANK_CIRCUMFERENTIAL_SPACING=0.1;SIMPLEFLUID_TANK_BOUNDARY_LAYER_COUNT=1;SIMPLEFLUID_TANK_BOUNDARY_LAYER_FIRST_HEIGHT=0.01;SIMPLEFLUID_TANK_STEPS=1;SIMPLEFLUID_TANK_DT=0.001")

  set_tests_properties(pitz_daily_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_PITZ_MESH_DIVISOR=40;SIMPLEFLUID_PITZ_STEPS=1;SIMPLEFLUID_PITZ_DT=1e-6")

  add_test(
      NAME pitz_daily_steady_smoke
      COMMAND ${CMAKE_COMMAND}
          "-DSIMPLEFLUID_PITZ_DAILY_EXECUTABLE=$<TARGET_FILE:pitz_daily>"
          "-DSIMPLEFLUID_PITZ_WORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/pitz_daily_steady_smoke"
          -P "${PROJECT_SOURCE_DIR}/cmake/RunPitzDailySteadySmoke.cmake")
  set_tests_properties(pitz_daily_steady_smoke PROPERTIES
      LABELS integration
      TIMEOUT 60)
endif()
