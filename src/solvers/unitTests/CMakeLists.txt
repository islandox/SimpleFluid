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

simplefluid_add_mpi_test(
    NAME RadiolyticGasModel_2procs
    TARGET testRadiolyticGasModelMultiRank
    PROCS 2
    LABELS "mpi;radiolysis"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FluidSolver_2procs
    TARGET testFluidSolverMultiRank
    PROCS 2
    LABELS "mpi;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME CoupledKrylovDirichletContinuity_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.CoupledKrylovDirichletContinuityMatchesBoundaryFaceFlux"
    LABELS "mpi;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativePhysicalBoussinesq_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnDistributedCartesianMesh"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativeUnstructuredPhysicalBoussinesq_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnPartitionedUnstructuredMesh"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity;unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativeUnstructuredCoupledFlow_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePartitionedUnstructuredCoupledKrylovAdvancesForcedFlow"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity;unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME TurbulentBoussinesq_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.AdvancesTurbulenceThroughSegregatedAndCoupledPressureVelocitySolves:TurbulentBoussinesqSolverTest.PassesDirectBuoyancyContextThroughSegregatedAndCoupledSolvers"
    LABELS "mpi;turbulence;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME IncompressibleIsothermalSolver_2procs
    TARGET testIncompressibleIsothermalSolver
    PROCS 2
    ARGS "--gtest_filter=IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonThroughPisoAndCoupledKrylov:IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonOnNativeMeshHandle"
    LABELS "mpi;turbulence;pressure_velocity;isothermal"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME CombinedTurbulentRadiolysis_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CouplesShearedRansShengTransportFissionAndDensityFeedbackOnTwoRanks"
    LABELS "mpi;turbulence;radiolysis;physical_models;pressure_velocity;acceptance"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PrecursorNonOrthogonalConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=DelayedNeutronPrecursorModelTest.EffectiveDiffusivityIncludesExplicitNonOrthogonalCorrection"
    LABELS "mpi;precursors"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PrecursorAdvectionConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=DelayedNeutronPrecursorModelTest.LiquidAdvectionConservesDistributedInventory:DelayedNeutronPrecursorModelTest.CollectivelyRejectsRankDivergentAdvanceSelection"
    LABELS "mpi;precursors;transport"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PrecursorCollectiveConfiguration_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=DelayedNeutronPrecursorModelTest.CollectivelyRejectsRankDivergentConfigurationAndPower:DelayedNeutronPrecursorModelTest.SolverRejectsRankDivergentPrecursorPresenceBeforeStep"
    LABELS "mpi;precursors;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarVoidDiffusionConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarVoidFractionModelTest.DiffusionSmoothsAndConservesGlobalVoid"
    LABELS "mpi;void_fraction"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FeedbackMap_2procs
    TARGET testFeedbackMapMultiRank
    PROCS 2
    LABELS "mpi;feedback_mapping"
    TIMEOUT 120)
