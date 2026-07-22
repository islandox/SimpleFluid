
# Unit tests for geometry

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testOrthogonalDiffusionProfile\\.cc$")
if(NOT SIMPLEFLUID_ENABLE_CXX_MODULES)
  list(REMOVE_ITEM TEST_SOURCES
       "${CMAKE_CURRENT_SOURCE_DIR}/testMeshModule.cc")
endif()
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Mesh GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  if(SIMPLEFLUID_ENABLE_CXX_MODULES AND TEST_NAME STREQUAL "testMeshModule")
    simplefluid_configure_module_test(${TEST_NAME} ${TEST_SRC})
  endif()
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME WallYPlusStatistics_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testYPlusBoundaryLayerController>"
            "--gtest_filter=WallYPlusStatisticsTest.ReducesAcrossRanksWhenSomeRanksOwnNoWallFaces")
  set_tests_properties(WallYPlusStatistics_2procs PROPERTIES
    LABELS "mpi;turbulence;mesh_adaptation"
    PROCESSORS 2
    TIMEOUT 120)
endif()

# The diffusion profiling test also needs the solvers library
add_executable(testOrthogonalDiffusionProfile testOrthogonalDiffusionProfile.cc)
target_link_libraries(testOrthogonalDiffusionProfile
      SimpleFluid::Mesh
      SimpleFluid::Solvers
      GTest::gtest_main)
simplefluid_configure_test_executable(testOrthogonalDiffusionProfile)
# This million-cell profiling executable is run manually with
# --gtest_also_run_disabled_tests and is intentionally not registered with CTest.
