# Unit tests for solvers

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testRadiolyticGasModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFluidSolverMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFeedbackMapMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Solvers GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  if(TEST_NAME STREQUAL "testVerificationCases")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "verification\\;integration"
    )
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

target_sources(testBoussinesqSolver PRIVATE
  PressureGradientStencilTestSupport.cc)

add_executable(
  testRadiolyticGasModelMultiRank
  testRadiolyticGasModelMultiRank.cc)
target_link_libraries(
  testRadiolyticGasModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testRadiolyticGasModelMultiRank)

add_executable(
  testFluidSolverMultiRank
  testFluidSolverMultiRank.cc)
target_link_libraries(
  testFluidSolverMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testFluidSolverMultiRank)

add_executable(
  testFeedbackMapMultiRank
  testFeedbackMapMultiRank.cc)
target_link_libraries(
  testFeedbackMapMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testFeedbackMapMultiRank)

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

  add_test(
    NAME CoupledKrylovDirichletContinuity_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testBoussinesqSolver>"
            "--gtest_filter=BoussinesqSolverTest.CoupledKrylovDirichletContinuityMatchesBoundaryFaceFlux")
  set_tests_properties(CoupledKrylovDirichletContinuity_2procs PROPERTIES
    LABELS "mpi;pressure_velocity"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME NativePhysicalBoussinesq_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testBoussinesqSolver>"
            "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnDistributedCartesianMesh")
  set_tests_properties(NativePhysicalBoussinesq_2procs PROPERTIES
    LABELS "mpi;boussinesq;physical_models;pressure_velocity"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME NativeUnstructuredPhysicalBoussinesq_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testBoussinesqSolver>"
            "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnPartitionedUnstructuredMesh")
  set_tests_properties(NativeUnstructuredPhysicalBoussinesq_2procs PROPERTIES
    LABELS "mpi;boussinesq;physical_models;pressure_velocity;unstructured"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME TurbulentBoussinesq_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testTurbulentBoussinesqSolver>"
            "--gtest_filter=TurbulentBoussinesqSolverTest.AdvancesTurbulenceThroughSegregatedAndCoupledPressureVelocitySolves:TurbulentBoussinesqSolverTest.PassesDirectBuoyancyContextThroughSegregatedAndCoupledSolvers")
  set_tests_properties(TurbulentBoussinesq_2procs PROPERTIES
    LABELS "mpi;turbulence;pressure_velocity"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME IncompressibleIsothermalSolver_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testIncompressibleIsothermalSolver>"
            "--gtest_filter=IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonThroughPisoAndCoupledKrylov:IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonOnNativeMeshHandle")
  set_tests_properties(IncompressibleIsothermalSolver_2procs PROPERTIES
    LABELS "mpi;turbulence;pressure_velocity;isothermal"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME CombinedTurbulentRadiolysis_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testTurbulentBoussinesqSolver>"
            "--gtest_filter=TurbulentBoussinesqSolverTest.CouplesShearedRansShengTransportFissionAndDensityFeedbackOnTwoRanks")
  set_tests_properties(CombinedTurbulentRadiolysis_2procs PROPERTIES
    LABELS "mpi;turbulence;radiolysis;physical_models;pressure_velocity;acceptance"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME PrecursorNonOrthogonalConservation_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPhase13PlusModels>"
            "--gtest_filter=DelayedNeutronPrecursorModelTest.EffectiveDiffusivityIncludesExplicitNonOrthogonalCorrection")
  set_tests_properties(PrecursorNonOrthogonalConservation_2procs PROPERTIES
    LABELS "mpi;precursors"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME PrecursorAdvectionConservation_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPhase13PlusModels>"
            "--gtest_filter=DelayedNeutronPrecursorModelTest.LiquidAdvectionConservesDistributedInventory:DelayedNeutronPrecursorModelTest.CollectivelyRejectsRankDivergentAdvanceSelection")
  set_tests_properties(PrecursorAdvectionConservation_2procs PROPERTIES
    LABELS "mpi;precursors;transport"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME PrecursorCollectiveConfiguration_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPhase13PlusModels>"
            "--gtest_filter=DelayedNeutronPrecursorModelTest.CollectivelyRejectsRankDivergentConfigurationAndPower:DelayedNeutronPrecursorModelTest.SolverRejectsRankDivergentPrecursorPresenceBeforeStep")
  set_tests_properties(PrecursorCollectiveConfiguration_2procs PROPERTIES
    LABELS "mpi;precursors;validation"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME ScalarVoidDiffusionConservation_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testPhase13PlusModels>"
            "--gtest_filter=ScalarVoidFractionModelTest.DiffusionSmoothsAndConservesGlobalVoid")
  set_tests_properties(ScalarVoidDiffusionConservation_2procs PROPERTIES
    LABELS "mpi;void_fraction"
    PROCESSORS 2
    TIMEOUT 120)

  add_test(
    NAME FeedbackMap_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testFeedbackMapMultiRank>")
  set_tests_properties(FeedbackMap_2procs PROPERTIES
    LABELS "mpi;feedback_mapping"
    PROCESSORS 2
    TIMEOUT 120)
endif()
