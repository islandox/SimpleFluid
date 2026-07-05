# Unit tests for equation classes

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFissionPowerSourceMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_exe_reuse_precompiled_header(${TEST_NAME})
  if(TEST_NAME STREQUAL "testAnalyticalConvergence")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "slow\\;analytical")
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

add_executable(
  testFissionPowerSourceMultiRank
  testFissionPowerSourceMultiRank.cc)
target_link_libraries(
  testFissionPowerSourceMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testFissionPowerSourceMultiRank)

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME FissionPowerSource_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testFissionPowerSourceMultiRank>")
  set_tests_properties(FissionPowerSource_2procs PROPERTIES
    LABELS "mpi"
    PROCESSORS 2
    TIMEOUT 120)
endif()
