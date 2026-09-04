# Unit tests for equation classes

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testBoussinesqModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFissionPowerSourceMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testPressureProjectionMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testSolidHeatConductionMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testTurbulenceModelMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
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
simplefluid_configure_test_executable(testBoussinesqModelMultiRank)

add_executable(
  testFissionPowerSourceMultiRank
  testFissionPowerSourceMultiRank.cc)
target_link_libraries(
  testFissionPowerSourceMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_configure_test_executable(testFissionPowerSourceMultiRank)

add_executable(
  testPressureProjectionMultiRank
  testPressureProjectionMultiRank.cc)
target_link_libraries(
  testPressureProjectionMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_configure_test_executable(testPressureProjectionMultiRank)

add_executable(
  testSolidHeatConductionMultiRank
  testSolidHeatConductionMultiRank.cc)
target_link_libraries(
  testSolidHeatConductionMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_configure_test_executable(testSolidHeatConductionMultiRank)

add_executable(
  testTurbulenceModelMultiRank
  testTurbulenceModelMultiRank.cc)
target_link_libraries(
  testTurbulenceModelMultiRank
  SimpleFluid::Equations
  GTest::gtest_main)
simplefluid_configure_test_executable(testTurbulenceModelMultiRank)

simplefluid_add_mpi_test(
    NAME BoussinesqModel_2procs
    TARGET testBoussinesqModelMultiRank
    PROCS 2
    LABELS "mpi;boussinesq"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FissionPowerSource_2procs
    TARGET testFissionPowerSourceMultiRank
    PROCS 2
    LABELS "mpi"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ALEEquations_2procs
    TARGET testALEEquations
    PROCS 2
    LABELS "mpi;ale;equations;conservation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PressureProjection_2procs
    TARGET testPressureProjectionMultiRank
    PROCS 2
    LABELS "mpi;pressure_projection"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME SolidHeatConduction_2procs
    TARGET testSolidHeatConductionMultiRank
    PROCS 2
    LABELS "mpi;solid_conduction"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME TurbulenceModel_2procs
    TARGET testTurbulenceModelMultiRank
    PROCS 2
    LABELS "mpi;turbulence"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME WallDistanceEquation_2procs
    TARGET testWallDistanceEquation
    PROCS 2
    LABELS "mpi;turbulence;wall_distance"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME TurbulenceBuoyancy_2procs
    TARGET testTurbulenceBuoyancy
    PROCS 2
    LABELS "mpi;turbulence;buoyancy"
    TIMEOUT 120)
