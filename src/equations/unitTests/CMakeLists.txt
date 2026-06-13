# Unit tests for equation classes

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testPrescribedHeatSourceMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_reuse_precompiled_header(${TEST_NAME})
  if(TEST_NAME STREQUAL "testAnalyticalConvergence")
    gtest_discover_tests(${TEST_NAME} PROPERTIES LABELS "slow\\;analytical")
  else()
    gtest_discover_tests(${TEST_NAME})
  endif()
endforeach()

add_executable(
  testPrescribedHeatSourceMultiRank
  testPrescribedHeatSourceMultiRank.cc)
target_link_libraries(
  testPrescribedHeatSourceMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_reuse_precompiled_header(testPrescribedHeatSourceMultiRank)

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME PrescribedHeatSource_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPrescribedHeatSourceMultiRank>")
  set_tests_properties(PrescribedHeatSource_2procs PROPERTIES
    LABELS "mpi"
    PROCESSORS 2
    TIMEOUT 120)
endif()
