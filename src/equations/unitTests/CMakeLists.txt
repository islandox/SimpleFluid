# Unit tests for equation classes

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testBoussinesqModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFissionPowerSourceMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testPressureProjectionMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testTurbulenceModelMultiRank\\.cc$")
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
  testBoussinesqModelMultiRank
  testBoussinesqModelMultiRank.cc)
target_link_libraries(
  testBoussinesqModelMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testBoussinesqModelMultiRank)

add_executable(
  testFissionPowerSourceMultiRank
  testFissionPowerSourceMultiRank.cc)
target_link_libraries(
  testFissionPowerSourceMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testFissionPowerSourceMultiRank)

add_executable(
  testPressureProjectionMultiRank
  testPressureProjectionMultiRank.cc)
target_link_libraries(
  testPressureProjectionMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testPressureProjectionMultiRank)

add_executable(
  testTurbulenceModelMultiRank
  testTurbulenceModelMultiRank.cc)
target_link_libraries(
  testTurbulenceModelMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testTurbulenceModelMultiRank)

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME BoussinesqModel_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testBoussinesqModelMultiRank>")
  set_tests_properties(BoussinesqModel_2procs PROPERTIES
    LABELS "mpi;boussinesq"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME FissionPowerSource_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testFissionPowerSourceMultiRank>")
  set_tests_properties(FissionPowerSource_2procs PROPERTIES
    LABELS "mpi"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME PressureProjection_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPressureProjectionMultiRank>")
  set_tests_properties(PressureProjection_2procs PROPERTIES
    LABELS "mpi;pressure_projection"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME TurbulenceModel_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testTurbulenceModelMultiRank>")
  set_tests_properties(TurbulenceModel_2procs PROPERTIES
    LABELS "mpi;turbulence"
    PROCESSORS 2
    TIMEOUT 120)
endif()
