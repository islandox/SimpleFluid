# Unit tests for solvers

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testRadiolyticGasModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFluidSolverMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Solvers GTest::gtest_main)
  simplefluid_exe_reuse_precompiled_header(${TEST_NAME})
  if(TEST_NAME STREQUAL "testVerificationCases")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "verification\\;integration"
    )
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

add_executable(
  testRadiolyticGasModelMultiRank
  testRadiolyticGasModelMultiRank.cc)
target_link_libraries(
  testRadiolyticGasModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testRadiolyticGasModelMultiRank)

add_executable(
  testFluidSolverMultiRank
  testFluidSolverMultiRank.cc)
target_link_libraries(
  testFluidSolverMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_exe_reuse_precompiled_header(testFluidSolverMultiRank)

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME RadiolyticGasModel_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testRadiolyticGasModelMultiRank>")
  set_tests_properties(RadiolyticGasModel_2procs PROPERTIES
    LABELS "mpi;radiolysis"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME FluidSolver_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testFluidSolverMultiRank>")
  set_tests_properties(FluidSolver_2procs PROPERTIES
    LABELS "mpi;pressure_velocity"
    PROCESSORS 2
    TIMEOUT 120)
endif()
