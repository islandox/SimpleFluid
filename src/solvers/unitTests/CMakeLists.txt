# Unit tests for solvers

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testRadiolyticGasModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFluidSolverMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFeedbackMapMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testPlanarFreeSurfaceModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testDICPreconditionerMultiRank\\.cc$")
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
  elseif(TEST_NAME STREQUAL "testBoussinesqPlanarALE")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "ale\\;free_surface\\;boussinesq\\;conservation\\;transaction"
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

add_executable(
  testPlanarFreeSurfaceModelMultiRank
  testPlanarFreeSurfaceModelMultiRank.cc)
target_link_libraries(
  testPlanarFreeSurfaceModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testPlanarFreeSurfaceModelMultiRank)

add_executable(
  testDICPreconditionerMultiRank
  testDICPreconditionerMultiRank.cc)
target_link_libraries(
  testDICPreconditionerMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testDICPreconditionerMultiRank)

foreach(_dic_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME DICPreconditioner_${_dic_procs}procs
      TARGET testDICPreconditionerMultiRank
      PROCS ${_dic_procs}
      LABELS "mpi;linear_solver;dic"
      TIMEOUT 120)
endforeach()

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
    NAME FluidSolverGeometryRefresh_2procs
    TARGET testFluidSolver
    PROCS 2
    ARGS "--gtest_filter=FluidSolverTest.CentralGeometryRefreshRestoresCoreCachesAndVtuPoints"
    LABELS "mpi;ale;geometry_epoch;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqMutableGeometryRefresh_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.MutableNativeConstructionPreservesFieldIdentityAndConstOverload"
    LABELS "mpi;ale;geometry_epoch;boussinesq"
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
    NAME BoilingCollectiveValidation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=BoilingSourceModelTest.CollectivelyRejectsRankAsymmetricInvalidCellState"
    LABELS "mpi;boiling;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FeedbackMap_2procs
    TARGET testFeedbackMapMultiRank
    PROCS 2
    LABELS "mpi;feedback_mapping"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PlanarFreeSurfaceModel_2procs
    TARGET testPlanarFreeSurfaceModelMultiRank
    PROCS 2
    LABELS "mpi;free_surface"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqFreeSurfaceConfig_2procs
    TARGET testBoussinesqFreeSurface
    PROCS 2
    ARGS "--gtest_filter=BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentOptions"
    LABELS "mpi;free_surface;boussinesq;configuration"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqFreeSurfaceStepPreflight_2procs
    TARGET testBoussinesqFreeSurface
    PROCS 2
    ARGS "--gtest_filter=BoussinesqFreeSurfaceTest.CellMassInventoryStepIsPartitionIndependent:BoussinesqFreeSurfaceTest.CollectivelyRejectsRemovalWhileBoilingOwnsSubmergedSteam:BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentExistingStateDuringConfiguration:BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentPresenceBeforeStep:BoussinesqFreeSurfaceTest.CollectivelyRejectsPositiveRankDivergentTimestepBeforeStep"
    LABELS "mpi;free_surface;boussinesq;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqPlanarALE_2procs
    TARGET testBoussinesqPlanarALE
    PROCS 2
    LABELS "mpi;ale;free_surface;boussinesq;conservation;transaction"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PlanarALEBoundaryValidation_2procs
    TARGET testPlanarALEBoundary
    PROCS 2
    ARGS "--gtest_filter=PlanarALEBoundaryTest.ValidatesGeometryPreservesTangentialSlipAndEnforcesZeroRelativeTopFlux:PlanarALEBoundaryTest.CollectivelyRejectsRankDivergentControlsAndForeignFields"
    LABELS "mpi;ale;free_surface;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME VolumeContinuityModelValidation_2procs
    TARGET testPlanarALEBoundary
    PROCS 2
    ARGS "--gtest_filter=VolumeContinuityModelTest.CollectivelyRejectsRankDivergentGlobalAndOptionalInputs"
    LABELS "mpi;ale;continuity;validation"
    TIMEOUT 120)

foreach(_composite_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME CoupledBlockOperator_${_composite_procs}procs
      TARGET testCoupledBlockOperator
      PROCS ${_composite_procs}
      LABELS "mpi;pressure_velocity;composite"
      TIMEOUT 180)
endforeach()

foreach(_backend_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME CoupledSolverBackends_${_backend_procs}procs
      TARGET testCoupledSolverBackends
      PROCS ${_backend_procs}
      ARGS "--gtest_filter=-*SerialSemiStructured*"
      LABELS "mpi;pressure_velocity;composite;solver_backends"
      TIMEOUT 300)
endforeach()

simplefluid_add_mpi_test(
    NAME CoupledBackendsCombinedPhysics_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CoupledBackendsAdvanceCombinedRansRadiolysisAndFeedback:TurbulentBoussinesqSolverTest.AdvancesWallTreatmentClosurePairingsThroughSegregatedAndCoupledSolvers"
    LABELS "mpi;pressure_velocity;composite;turbulence;radiolysis"
    TIMEOUT 180)

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME MultiRegionFlow_${_region_procs}procs TARGET testCoupledSolverBackends
        PROCS ${_region_procs} ARGS "--gtest_filter=*MultiRegion*" LABELS "mpi;multi_region;pressure_velocity" TIMEOUT 300)
    simplefluid_add_mpi_test(NAME MultiRegionSolverALE_${_region_procs}procs TARGET testBoussinesqPlanarALE
        PROCS ${_region_procs} ARGS "--gtest_filter=BoussinesqPlanarALESupportMatrixTest.AcceptsCompositeAxialMotion"
        LABELS "mpi;multi_region;ale;pressure_velocity" TIMEOUT 180)
endforeach()

simplefluid_add_mpi_test(
    NAME SASSupportedPaths_2procs
    TARGET testSASSupportedPaths
    PROCS 2
    LABELS "mpi;turbulence;sas;small_fixture"
    TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASMaterialFeedback_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.MaterialFeedbackPublishesAndRollsBackWithSAS"
    LABELS "mpi;turbulence;sas;feedback;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASScalarVoid_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.ScalarVoidDiffusionCollapseAndFeedbackRollBackWithSAS"
    LABELS "mpi;turbulence;sas;void;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASPrecursors_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.PrecursorInventoryAndDiagnosticsRollBackWithSAS"
    LABELS "mpi;turbulence;sas;precursors;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASRadiolysis_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.RadiolysisAndHydrogenLedgersRollBackWithSAS"
    LABELS "mpi;turbulence;sas;radiolysis;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASBoiling_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.BoilingSourcesAndPendingStateRollBackWithSAS"
    LABELS "mpi;turbulence;sas;boiling;small_fixture" TIMEOUT 120)
