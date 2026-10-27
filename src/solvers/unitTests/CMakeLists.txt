# Unit tests for solvers

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
if(NOT SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  # Solver/private-VoF fixtures are omitted, not registered as skipped passes.
  # Lower-level FVM and geometry tests retain their independent registrations.
  list(FILTER TEST_SOURCES EXCLUDE REGEX
       ".*/(test[^/]*VoF[^/]*|testBubbleRepresentation)\\.cc$")
endif()
# Reacting coupling and its representation fixtures need species-schema migration.
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/(testReactingVoFCoupler|testBubbleRepresentation)\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testReactingGasModelMultiRank\\.cc$")
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
  if(TEST_NAME STREQUAL "testLowMachFaceTransportAdapter")
    # Exercise the private adapter directly without exporting it from the DSO.
    target_sources(${TEST_NAME} PRIVATE ../details/LowMachFaceTransportAdapter.cc)
  endif()
  if(TEST_NAME STREQUAL "testConservativeVoFTransport" OR TEST_NAME STREQUAL "testExactConservativeVoFTransport")
    # Compile the private numerical boundary directly; it is not a DSO export.
    target_sources(${TEST_NAME} PRIVATE ../details/ConservativeVoFTransport.cc ../details/ConservativeVoFExactTransport.cc ../details/GeneralVoFExactStorage.cc ../details/ConservativeVoFExactArithmetic.cc ../details/GeneralVoFTransferBinding.cc)
  endif()
  if(TEST_NAME STREQUAL "testGeneralVoFCurvature" OR TEST_NAME STREQUAL "testExactGeneralVoFCurvature")
    target_sources(${TEST_NAME} PRIVATE ../details/GeneralVoFCurvature.cc)
  endif()
  if(TEST_NAME STREQUAL "testGeneralVoFFlowWorkspace" OR TEST_NAME STREQUAL "testExactGeneralVoFFlowWorkspace")
    target_sources(${TEST_NAME} PRIVATE ../details/GeneralVoFFlowWorkspace.cc ../details/GeneralVoFExactProjection.cc ../details/VoFProjectionWorkspace.cc
      ../details/GeneralVoFPhiClosure.cc ../details/GeneralVoFExactPhiClosure.cc ../details/GeneralVoFPhiClosureSupport.cc ../details/GeneralVoFProjectionArithmetic.cc ../details/GeneralVoFExactPressure.cc ../details/GeneralVoFExactStorage.cc ../details/ConservativeVoFExactArithmetic.cc)
  endif()
  if(TEST_NAME STREQUAL "testGeneralVoFExactPressure")
    target_sources(${TEST_NAME} PRIVATE ../details/GeneralVoFExactPressure.cc
      ../details/GeneralVoFExactStorage.cc ../details/ConservativeVoFExactArithmetic.cc)
  endif()
  if(TEST_NAME STREQUAL "testGeneralVoFPhiClosure" OR TEST_NAME STREQUAL "testExactGeneralVoFPhiClosure")
    target_sources(${TEST_NAME} PRIVATE ../details/GeneralVoFPhiClosure.cc ../details/GeneralVoFExactPhiClosure.cc ../details/GeneralVoFPhiClosureSupport.cc
      ../details/GeneralVoFProjectionArithmetic.cc ../details/GeneralVoFExactStorage.cc ../details/ConservativeVoFExactArithmetic.cc)
  endif()
  simplefluid_configure_test_executable(${TEST_NAME})
  if(TEST_NAME STREQUAL "testExactGeneralVoFSolver")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;exact\\;general_mesh\\;integration\\;experimental\\;policy" TIMEOUT 1800)
  elseif(TEST_NAME MATCHES "^testExact(GeneralVoFCurvature|GeneralVoFFlowWorkspace|GeneralVoFPhiClosure|ConservativeVoFTransport)$")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;exact\\;conservation\\;collective")
  elseif(TEST_NAME STREQUAL "testGeneralVoFSolver")
    simplefluid_discover_tests(${TEST_NAME} PROPERTIES LABELS "vof\\;general_mesh\\;integration" TIMEOUT 1800)
  elseif(TEST_NAME STREQUAL "testGeneralVoFInterfaceFlow")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;interface\\;conservation\\;transaction" TIMEOUT 300)
  elseif(TEST_NAME STREQUAL "testGeneralVoFHomogeneousFlow")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;general_mesh\\;homogeneous\\;conservation\\;transaction" TIMEOUT 180)
  elseif(TEST_NAME STREQUAL "testVerificationCases")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "verification\\;integration"
    )
  elseif(TEST_NAME STREQUAL "testBoussinesqPlanarALE")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "ale\\;free_surface\\;boussinesq\\;conservation\\;transaction"
    )
  elseif(TEST_NAME STREQUAL "testAdaptiveVoFLowMach" OR TEST_NAME STREQUAL "testVoFLowMach"
         OR TEST_NAME STREQUAL "testReactingVoFCoupler" OR TEST_NAME STREQUAL "testBubbleRepresentation")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;reacting_gas\\;thermodynamics\\;conservation\\;transaction\\;collective")
  elseif(TEST_NAME STREQUAL "testAdaptiveVoFSolver")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;amr\\;conservation\\;capillary\\;transaction\\;collective")
  elseif(TEST_NAME STREQUAL "testConservativeVoFTransport")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;conservation\\;collective")
  elseif(TEST_NAME STREQUAL "testVoFSolver")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;conservation\\;capillary\\;transaction")
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

# Finite-solvent fixed-reference Bouss bulk/wall sources and rank-coherent setup.
foreach(_bouss_vapor_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME BoussFiniteVapor_${_bouss_vapor_ranks}procs
        TARGET testBoussFiniteVaporCoupling PROCS ${_bouss_vapor_ranks}
        LABELS "mpi;species;boussinesq;boiling;wall_heat;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

# Clean schema configuration and explicit gas-only reference scheduling.
# Serial discovery uses TEST_SOURCES above; these are actual 2/4-rank groups.
foreach(_schema_api_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME ReactingLegacyCleanup_${_schema_api_ranks}procs
        TARGET testReactingLegacyCleanup PROCS ${_schema_api_ranks}
        LABELS "mpi;species;configuration;conservation;transaction;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME ReactingReferenceSchedule_${_schema_api_ranks}procs
        TARGET testReactingReferenceSchedule PROCS ${_schema_api_ranks}
        LABELS "mpi;species;reference;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

# Typed species surface coupling and exact transported liquid-volume packets.
foreach(_species_surface_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME BoussinesqSpeciesFreeSurface_${_species_surface_ranks}procs
        TARGET testBoussinesqSpeciesFreeSurface PROCS ${_species_surface_ranks}
        LABELS "mpi;species;free_surface;boussinesq;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME SpeciesLiquidVolumePacket_${_species_surface_ranks}procs
        TARGET testSpeciesLiquidVolumePacket PROCS ${_species_surface_ranks}
        LABELS "mpi;species;ale;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME BoussinesqSpeciesPlanarALE_${_species_surface_ranks}procs
        TARGET testBoussinesqPlanarALE PROCS ${_species_surface_ranks}
        ARGS "--gtest_filter=BoussinesqSpeciesPlanarALETest.*"
        LABELS "mpi;species;ale;boussinesq;conservation;transaction;empty_rank" TIMEOUT 120)
endforeach()

# Raw accepted population events preserve tiny material/count endpoints and replay.
foreach(_raw_population_ranks IN ITEMS 2 4)
    foreach(_raw_population_case IN ITEMS PopulationRawCountReplay PopulationRawMaterialReplay PopulationChemicalDonors)
        simplefluid_add_mpi_test(NAME ${_raw_population_case}_${_raw_population_ranks}procs
            TARGET test${_raw_population_case} PROCS ${_raw_population_ranks}
            LABELS "mpi;species;population;conservation;transaction;empty_rank" TIMEOUT 180)
    endforeach()
endforeach()

# Prescribed native ALE transactions and actual source geometry frames.
foreach(_native_ale_ranks IN ITEMS 2 4)
    foreach(_native_ale_case IN ITEMS LowMachPlanarALE LowMachALEConservation ReactingMixtureALE ReactingSourceLocation)
        simplefluid_add_mpi_test(NAME ${_native_ale_case}_${_native_ale_ranks}procs
            TARGET test${_native_ale_case} PROCS ${_native_ale_ranks}
            LABELS "mpi;reacting;low_mach;ale;conservation;transaction;empty_rank" TIMEOUT 180)
    endforeach()
endforeach()

# Configured local interphase thermal events, including the prescribed ALE transaction.
foreach(_interphase_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME ReactingMixtureInterphaseTransfer_${_interphase_ranks}procs
        TARGET testReactingMixtureInterphaseTransfer PROCS ${_interphase_ranks}
        LABELS "mpi;reacting;low_mach;thermal;ale;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

# Conservative native mixture flow and its external reaction controller.
foreach(_mixture_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME NativeSourceDonors_${_mixture_ranks}procs
        TARGET testNativeSourceDonors PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachMixture_${_mixture_ranks}procs
        TARGET testLowMachMixtureSolver PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixtureCoupler_${_mixture_ranks}procs
        TARGET testReactingMixtureCoupler PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachSourceContracts_${_mixture_ranks}procs
        TARGET testLowMachSourceContracts PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachPopulationNumbers_${_mixture_ranks}procs
        TARGET testLowMachPopulationNumbers PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;population;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixturePopulationEvents_${_mixture_ranks}procs
        TARGET testReactingMixturePopulationEvents PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;population;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachFaceTransport_${_mixture_ranks}procs
        TARGET testLowMachFaceTransportAdapter PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixtureFaces_${_mixture_ranks}procs
        TARGET testReactingMixtureFaceCoupling PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachCaloricFeasibility_${_mixture_ranks}procs
        TARGET testLowMachCaloricFeasibility PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;thermodynamics;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachPressureContracts_${_mixture_ranks}procs
        TARGET testLowMachPressureContracts PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;pressure;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingSpeciesHistory_${_mixture_ranks}procs
        TARGET testReactingSpeciesHistory PROCS ${_mixture_ranks}
        LABELS "mpi;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
endforeach()

# Schema-indexed reacting fields and the liquid inventory follower.
simplefluid_add_mpi_test(NAME ReactingSpeciesModel_2procs TARGET testReactingSpeciesModel PROCS 2
    ARGS "--gtest_filter=-*LegacyMesh*"
    LABELS "mpi;reacting_gas;species;conservation;transaction" TIMEOUT 180)
simplefluid_add_mpi_test(NAME ReactingSpeciesModel_EmptyRank_4procs TARGET testReactingSpeciesModel PROCS 4
    ARGS "--gtest_filter=*CollectiveInitialization*:*RankAsymmetric*"
    LABELS "mpi;reacting_gas;species;collective;empty_rank" TIMEOUT 120)
simplefluid_add_mpi_test(NAME ReactingBoussinesqCoupling_2procs TARGET testReactingBoussinesqCoupling PROCS 2
    LABELS "mpi;reacting_gas;boussinesq;conservation;transaction" TIMEOUT 180)
simplefluid_add_mpi_test(NAME SpeciesLiquidMassInventory_2procs TARGET testSpeciesLiquidMassInventory PROCS 2
    LABELS "mpi;reacting_gas;free_surface;ale;conservation;transaction" TIMEOUT 180)

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  simplefluid_add_mpi_test(
      NAME AdaptiveVoFLowMach_2procs
      TARGET testAdaptiveVoFLowMach
      PROCS 2
      LABELS "vof;reacting_gas;thermodynamics;conservation;transaction;collective"
      TIMEOUT 240)

  simplefluid_add_mpi_test(
      NAME AdaptiveVoFLowMach_EmptyRank_9procs
      TARGET testAdaptiveVoFLowMach
      PROCS 9
      ARGS "--gtest_filter=AdaptiveVoFLowMachTest.EmptyOwnedRankParticipatesInInitializationPopulationSourcesAndHintGuards"
      LABELS "vof;reacting_gas;thermodynamics;conservation;transaction;collective;empty_rank"
      TIMEOUT 120)

  simplefluid_add_mpi_test(
      NAME AdaptiveVoFSolver_2procs
      TARGET testAdaptiveVoFSolver
      PROCS 2
      LABELS "vof;amr;conservation;capillary;transaction;collective"
      TIMEOUT 1200)

  simplefluid_add_mpi_test(
      NAME VoFRejectsDistributedConstruction_2procs
      TARGET testVoFSolver
      PROCS 2
      ARGS "--gtest_filter=VoFSolverParallelTest.RejectsDistributedConstruction"
      LABELS "vof;collective"
      TIMEOUT 60)

endif()

simplefluid_add_mpi_test(
    NAME CoupledPreconditionerValidation_2procs
    TARGET testCoupledPressureVelocitySolver
    PROCS 2
    ARGS "--gtest_filter=*RejectsRankLocalInvalidPreconditionerStateCollectively*"
    LABELS "mpi;coupled;collective"
    TIMEOUT 60)

simplefluid_add_mpi_test(
    NAME CoupledPressureSmoothing_2procs
    TARGET testCoupledPressureVelocitySolver
    PROCS 2
    ARGS "--gtest_filter=CoupledPressureVelocitySolverTest.PressureMultigridContractsCheckerboardModeAndRefreshesNumerics"
    LABELS "mpi;pressure_velocity;preconditioner;accuracy;reuse"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BelosAccurateResidual_2procs
    TARGET testBelosLinearSolver
    PROCS 2
    ARGS "--gtest_filter=BelosLinearSolverTest.AccurateStoredCrsResidual*:BelosLinearSolverTest.RefinesGmresLossOfAccuracyAgainstOriginalTrueResidualGate:BelosLinearSolverTest.GmresLossOfAccuracyRefinementKeepsIterationAndRestartBounds:BelosLinearSolverTest.GmresDoesNotRetryOrdinaryFailureLargeGapOrNonfiniteResidual:BelosLinearSolverTest.GmresDefectCorrectionRestoresOriginalVectorsAfterException"
    LABELS "mpi;linear_solver;accuracy"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BelosSymmetricAMG_2procs
    TARGET testBelosLinearSolver
    PROCS 2
    ARGS "--gtest_filter=BelosLinearSolverTest.SymmetricSmoothedAmg*"
    LABELS "mpi;linear_solver;preconditioner"
    TIMEOUT 120)

target_sources(testBoussinesqSolver PRIVATE
  PressureGradientStencilTestSupport.cc)

if(SIMPLEFLUID_ENABLE_NOX)
  simplefluid_add_mpi_test(
      NAME CavityNoxVerification_2procs
      TARGET testVerificationCases
      PROCS 2
      ARGS "--gtest_filter=VerificationCasesTest.LidDrivenCavityRe100Nox:VerificationCasesTest.LidDrivenCavityRe1000Nox"
      LABELS "verification;integration;mpi;nonlinear_solver"
      TIMEOUT 180)
  if(TEST CavityNoxVerification_2procs)
    set_tests_properties(CavityNoxVerification_2procs PROPERTIES
        ENVIRONMENT "SIMPLEFLUID_PROFILE_OUTPUT_DIR=")
  endif()
endif()

simplefluid_add_mpi_test(
    NAME CoupledNonlinearSolver_2procs
    TARGET testCoupledNonlinearSolver
    PROCS 2
    LABELS "mpi;nonlinear_solver;pressure_velocity"
    TIMEOUT 180)

add_executable(
  testReactingGasModelMultiRank
  testReactingGasModelMultiRank.cc)
target_link_libraries(
  testReactingGasModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testReactingGasModelMultiRank)

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
    NAME ReactingGasModel_2procs
    TARGET testReactingGasModelMultiRank
    PROCS 2
    LABELS "mpi;reacting_gas"
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
    NAME CombinedTurbulentReactingGas_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CouplesShearedRansSpeciesTransportPrescribedHeatAndDensityFeedbackOnTwoRanks"
    LABELS "mpi;turbulence;reacting_gas;physical_models;pressure_velocity;acceptance"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarSourceDecayConfiguration_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.CollectivelyRejectsRankDivergentConfigurationAndSource"
    LABELS "mpi;scalar_source_decay;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarGroupNonOrthogonalConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.EffectiveDiffusivityIncludesExplicitNonOrthogonalCorrection"
    LABELS "mpi;scalar_source_decay"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarGroupAdvectionConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.LiquidAdvectionConservesDistributedInventory:ScalarSourceDecayTransportTest.CollectivelyRejectsRankDivergentAdvanceSelection"
    LABELS "mpi;scalar_source_decay;transport"
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
    ARGS "--gtest_filter=VolumeContinuityModelTest.CollectivelyRejectsRankDivergentGlobalAndOptionalInputs:VolumeContinuityModelTest.MeshDefectCorrectionPreservesPhysicalSourceAndUsesOwnedVolumeWeights"
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
      LABELS "mpi;pressure_velocity;composite;solver_backends"
      TIMEOUT 300)
endforeach()

simplefluid_add_mpi_test(
    NAME CoupledBackendsCombinedPhysics_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CoupledBackendsAdvanceCombinedRansReactingGasAndFeedback:TurbulentBoussinesqSolverTest.AdvancesWallTreatmentClosurePairingsThroughSegregatedAndCoupledSolvers"
    LABELS "mpi;pressure_velocity;composite;turbulence;reacting_gas"
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

simplefluid_add_mpi_test(NAME SASNonlinear_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.Nox*"
    LABELS "mpi;turbulence;sas;nonlinear_solver;transaction;small_fixture" TIMEOUT 180)

simplefluid_add_mpi_test(NAME SASScalarVoid_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.ScalarVoidDiffusionCollapseAndFeedbackRollBackWithSAS"
    LABELS "mpi;turbulence;sas;void;small_fixture" TIMEOUT 120)


simplefluid_add_mpi_test(NAME SASReactingGas_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.ReactingGasAndSpeciesLedgersRollBackWithSAS"
    LABELS "mpi;turbulence;sas;reacting_gas;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASBoiling_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.BoilingSourcesAndPendingStateRollBackWithSAS"
    LABELS "mpi;turbulence;sas;boiling;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASFreeSurface_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.FixedFreeSurface*:SASSupportedPathsTest.LazyFreeSurface*"
    LABELS "mpi;turbulence;sas;free_surface;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME VolumeContinuityAccess_2procs
    TARGET testVolumeContinuityGeometryAccess
    PROCS 2
    LABELS "mpi;ale;continuity;validation"
    TIMEOUT 60)

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)

  # Direct transport update, before pressure or accepted-state publication.
  simplefluid_add_mpi_test(
      NAME ConservativeVoFTransport_2procs
      TARGET testConservativeVoFTransport
      PROCS 2
      LABELS "vof;conservation;collective"
  )

  # The largest direct fixture has 32 cells; 33 ranks guarantee empty owners.
  simplefluid_add_mpi_test(
      NAME ConservativeVoFTransport_EmptyRank_33procs
      TARGET testConservativeVoFTransport
      PROCS 33
      LABELS "vof;conservation;collective;empty_rank"
      TIMEOUT 120)


  simplefluid_add_mpi_test(
      NAME GeneralVoFFlowWorkspace_2procs
      TARGET testGeneralVoFFlowWorkspace
      PROCS 2
      LABELS "vof;amr;flow;collective"
      TIMEOUT 180)

  simplefluid_add_mpi_test(
      NAME GeneralVoFSolver_2procs
      TARGET testGeneralVoFSolver
      PROCS 2
      ARGS "--gtest_filter=GeneralVoFSolverTest.*"
      LABELS "mpi;vof;general_mesh;integration"
      TIMEOUT 3600
  )

  simplefluid_add_mpi_test(
      NAME GeneralVoFFlowWorkspace_EmptyRank_4procs
      TARGET testGeneralVoFFlowWorkspace
      PROCS 4
      LABELS "mpi;vof;flow;collective;empty_rank"
      TIMEOUT 180)

  simplefluid_add_mpi_test(
      NAME GeneralVoFSolver_SingleLayer_4procs
      TARGET testGeneralVoFSolver
      PROCS 4
      ARGS "--gtest_filter=GeneralVoFSolverTest.ClosedTwoByTwoSingleLayerConstantCarrierAdvancesAndRejectsAtomically"
      LABELS "mpi;vof;general_mesh;integration;empty_rank"
      TIMEOUT 180)

  simplefluid_add_mpi_test(
      NAME GeneralVoFSolver_EmptyRank_4procs
      TARGET testGeneralVoFSolver
      PROCS 4
      ARGS "--gtest_filter=GeneralVoFSolverTest.NativeSourceFreeStationaryAdvanceAndRejectedStepRetainAcceptedPublication:GeneralVoFSolverTest.IndependentMixedPlaneRetainsBothPhaseMassesInSourceFreeStationaryFlow:GeneralVoFSolverTest.MicroscopicDeclaredTraceAggregatesPhaseMassBeforeDiagnosticNarrowing:GeneralVoFSolverTest.SymmetricNativeRegridRetainsAcceptedPhysicalMomentumAtFixedClock:GeneralVoFSolverTest.TinyCellRepartitionRetainsIndependentZeroRatesAndSubnormalAggregateMomentum"
      LABELS "mpi;vof;general_mesh;integration;empty_rank"
      TIMEOUT 3600
  )

  # Explicit optional-authority contract checks, not experimental physical qualification.
  set(general_vof_experimental_inventory_filter
      "--gtest_filter=ExactGeneralVoFSolverTest.ExperimentalInventoryIsExplicitAndRollbackPreservesItsSelectedAuthority:ExactGeneralVoFSolverTest.ExperimentalInventorySamePolicyRegridRetainsClockAndPhaseAuthority:ExactGeneralVoFSolverTest.ExperimentalInventoryRejectsForeignPolicyTrialWithoutChangingAcceptedSolver")
  simplefluid_add_mpi_test(
      NAME ExactGeneralVoFSolver_2procs
      TARGET testExactGeneralVoFSolver
      PROCS 2
      ARGS "--gtest_filter=ExactGeneralVoFSolverTest.*"
      LABELS "mpi;vof;exact;general_mesh;experimental;policy"
      TIMEOUT 3600)
  simplefluid_add_mpi_test(
      NAME ExactGeneralVoFSolver_4procs
      TARGET testExactGeneralVoFSolver
      PROCS 4
      ARGS "${general_vof_experimental_inventory_filter}"
      LABELS "mpi;vof;exact;general_mesh;experimental;policy;empty_rank"
      TIMEOUT 180)

  simplefluid_add_mpi_test(
      NAME GeneralVoFExactPressure_2procs
      TARGET testGeneralVoFExactPressure
      PROCS 2
      LABELS "vof;pressure;exact;collective;empty_rank"
      TIMEOUT 180)
  simplefluid_add_mpi_test(
      NAME GeneralVoFExactPressure_4procs
      TARGET testGeneralVoFExactPressure
      PROCS 4
      LABELS "vof;pressure;exact;collective;empty_rank"
      TIMEOUT 180)

  # Fixed-width finite arithmetic has independent capacity/empty-owner coverage.
  foreach(_finite_projection_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(
    NAME GeneralVoFWallPressure_${_finite_projection_procs}procs
    TARGET testGeneralVoFSolver
    PROCS ${_finite_projection_procs}
    ARGS "--gtest_filter=GeneralVoFSolverTest.SingleLayerNormalGravityHasWallPressureReactionAndZeroAcceptedMomentum:GeneralVoFSolverTest.SingleLayerInitialNormalMotionUsesPressureWithoutApplyingGravity"
    LABELS "mpi;vof;finite;pressure;collective;empty_rank"
    TIMEOUT 180)
  simplefluid_add_mpi_test(
    NAME GeneralVoFWallPressureWorkspace_${_finite_projection_procs}procs
    TARGET testGeneralVoFFlowWorkspace
    PROCS ${_finite_projection_procs}
    ARGS "--gtest_filter=GeneralVoFFlowWorkspaceTest.OrdinaryMissingAxisWallPressureBalancesPhysicalMassAndInitialNormalMotion:GeneralVoFFlowWorkspaceTest.CertifiedMissingAxisWallPressureBalancesPhysicalMassAndInitialNormalMotion:GeneralVoFFlowWorkspaceTest.SingleLayerPressureReactionPreservesResolvedTangentialProjection:GeneralVoFFlowWorkspaceTest.SampledThreeDimensionalProjectionHasNoAffineWallLift:GeneralVoFFlowWorkspaceTest.UnsupportedUnsampledNormalMotionRejectsWithoutChangingCandidates"
    LABELS "mpi;vof;finite;pressure;collective;empty_rank"
    TIMEOUT 180)
  simplefluid_add_mpi_test(
    NAME GeneralVoFFixedGeometry_${_finite_projection_procs}procs
    TARGET testGeneralVoFSolver
    PROCS ${_finite_projection_procs}
    ARGS "--gtest_filter=GeneralVoFSolverTest.FiniteSolverKeepsSelectedGeometryThroughStepAndRegrid"
    LABELS "mpi;vof;finite;geometry;collective"
    TIMEOUT 180)
  simplefluid_add_mpi_test(
      NAME GeneralVoFOrdinaryMomentum_${_finite_projection_procs}procs
      TARGET testConservativeVoFTransport
      PROCS ${_finite_projection_procs}
      ARGS "--gtest_filter=ConservativeVoFTransportTest.OrdinaryWideMomentumPreservesPhysicalStorageWithoutIntervalAdmission"
      LABELS "mpi;vof;ordinary;momentum;collective"
      TIMEOUT 180)
    simplefluid_add_mpi_test(
      NAME GeneralVoFFinitePassage_${_finite_projection_procs}procs
      TARGET testGeneralVoFSolver
      PROCS ${_finite_projection_procs}
      ARGS "--gtest_filter=GeneralVoFSolverTest.NonzeroSkewCoarseFinePhasePassagePreservesIndependentInventories"
      LABELS "mpi;vof;finite;general_mesh;integration"
      TIMEOUT 1800)
    simplefluid_add_mpi_test(
      NAME GeneralVoFFiniteProjection_${_finite_projection_procs}procs
      TARGET testGeneralVoFFlowWorkspace
      PROCS ${_finite_projection_procs}
      ARGS "--gtest_filter=GeneralVoFFlowWorkspaceTest.OrdinaryProjectionKeepsPhysicalThresholdWithoutIntervalAdmission:GeneralVoFFlowWorkspaceTest.ArithmeticModeMismatchRejectsIncludingEmptyOwners:GeneralVoFFlowWorkspaceTest.FiniteProjectionKeepsOriginalPhysicalThresholdWithObservedEmptyOwners:GeneralVoFFlowWorkspaceTest.FiniteExpressionBoundsRetainCanceledOperandsAndCachedCoefficientError:GeneralVoFFlowWorkspaceTest.FiniteConstantCurvatureRetainsPressureWallForceAndMassError"
      LABELS "mpi;vof;finite;projection;collective;empty_rank"
      TIMEOUT 180)
    simplefluid_add_mpi_test(
      NAME GeneralVoFFinitePhiClosure_${_finite_projection_procs}procs
      TARGET testGeneralVoFPhiClosure
      PROCS ${_finite_projection_procs}
      ARGS "--gtest_filter=GeneralVoFPhiClosureTest.CertifiedClosureRequiresExplicitBoundedMode:GeneralVoFPhiClosureTest.FiniteSparseForestPreservesCirculationAndRejectsIndependentCapacityExhaustion:GeneralVoFPhiClosureTest.FiniteCapacityRoutingPreservesFixedEdgesAndAvoidsNarrowShortcuts:GeneralVoFPhiClosureTest.FiniteWidestParentDepthPropagatesThroughLowerCapacityLeaf"
      LABELS "mpi;vof;finite;conservation;collective"
      TIMEOUT 180)
  endforeach()

  # Private numerical sources are compiled directly in these test executables.
  # Source properties are directory-scoped, so repeat the production policy here.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      set_source_files_properties(
          testGeneralVoFFlowWorkspace.cc
          testExactGeneralVoFFlowWorkspace.cc
          ../GeneralVoFSolver.cc
        ../details/GeneralVoF/SolverImpl.cc
          ../details/ConservativeVoFTransport.cc
          ../details/ConservativeVoFExactTransport.cc
          ../details/GeneralVoFFlowWorkspace.cc
          ../details/GeneralVoFExactProjection.cc
          ../details/GeneralVoFPhiClosure.cc
          ../details/GeneralVoFExactPhiClosure.cc
          ../details/GeneralVoFPhiClosureSupport.cc
          ../details/GeneralVoFCurvature.cc
          PROPERTIES
          COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
          SKIP_UNITY_BUILD_INCLUSION TRUE
          SKIP_PRECOMPILE_HEADERS TRUE)
  endif()

  # Preserve the rank coverage of cases extracted from the mixed executables.
  simplefluid_add_mpi_test(
      NAME ExactConservativeVoFTransport_2procs
      TARGET testExactConservativeVoFTransport PROCS 2
      LABELS "vof;exact;conservation;collective")
  simplefluid_add_mpi_test(
      NAME ExactConservativeVoFNorm_4procs
      TARGET testExactConservativeVoFTransport PROCS 4
      ARGS "--gtest_filter=ExactConservativeVoFTransportTest.LazyZeroNormSupportsEmptyOwnersAndBudgetRecovery:ExactConservativeVoFTransportTest.LeasedPhysicalNormSeparatesAggregateOperationAndCommunicationStorage"
      LABELS "vof;exact;conservation;collective;empty_rank" TIMEOUT 120)
  simplefluid_add_mpi_test(
      NAME ExactConservativeVoFTransport_EmptyRank_33procs
      TARGET testExactConservativeVoFTransport PROCS 33
      LABELS "vof;exact;conservation;collective;empty_rank" TIMEOUT 120)
  simplefluid_add_mpi_test(
      NAME ExactGeneralVoFFlowWorkspace_2procs
      TARGET testExactGeneralVoFFlowWorkspace PROCS 2
      LABELS "vof;exact;amr;flow;collective" TIMEOUT 180)
  simplefluid_add_mpi_test(
      NAME ExactGeneralVoFFlowWorkspace_EmptyRank_4procs
      TARGET testExactGeneralVoFFlowWorkspace PROCS 4
      LABELS "mpi;vof;exact;flow;collective;empty_rank" TIMEOUT 180)
endif()

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredFlow_${_semi_ranks}procs
        TARGET testCoupledSolverBackends PROCS ${_semi_ranks}
        ARGS "--gtest_filter=*SemiStructuredNonOrthogonal*"
        LABELS "mpi;semi_structured;pressure_velocity;non_orthogonal" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME SemiStructuredSolverALE_${_semi_ranks}procs
        TARGET testBoussinesqPlanarALE PROCS ${_semi_ranks}
        ARGS "--gtest_filter=BoussinesqPlanarALESupportMatrixTest.*SemiStructured*"
        LABELS "mpi;semi_structured;ale;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredVolumeContinuity_${_semi_ranks}procs
        TARGET testVolumeContinuityModel PROCS ${_semi_ranks}
        LABELS "mpi;semi_structured;ale;continuity;transaction;empty_rank" TIMEOUT 120)
endforeach()

foreach(PROCS 2 4)
  simplefluid_add_mpi_test(NAME GeneralVoFHomogeneousFlow_${PROCS}procs
    TARGET testGeneralVoFHomogeneousFlow PROCS ${PROCS}
    LABELS "mpi;vof;homogeneous;conservation;transaction;empty_rank" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME GeneralVoFHomogeneousMomentum_${PROCS}procs
    TARGET testConservativeVoFTransport PROCS ${PROCS}
    ARGS "--gtest_filter=ConservativeVoFTransportTest.HomogeneousSweptLedgerMatchesIndependentImplicitCirculation"
    LABELS "mpi;vof;homogeneous;momentum;conservation;empty_rank" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME GeneralVoFHomogeneousHydrostatic_${PROCS}procs
    TARGET testGeneralVoFSolver PROCS ${PROCS}
    ARGS "--gtest_filter=GeneralVoFSolverTest.PhysicalHydrostaticPressureBalancesSkewNativeCentersOverFourSteps"
    LABELS "mpi;vof;homogeneous;hydrostatic;empty_rank" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME GeneralVoFFaceFlow_${PROCS}procs
    TARGET testGeneralVoFFaceFlow PROCS ${PROCS} ARGS "--gtest_filter=GeneralVoFFaceFlow.*")
  simplefluid_add_mpi_test(NAME GeneralVoFFaceSurface_${PROCS}procs
    TARGET testGeneralVoFFaceSurface PROCS ${PROCS} ARGS "--gtest_filter=GeneralVoFFaceSurface.*-GeneralVoFFaceSurface.*TangentStripes*")
endforeach()

simplefluid_add_mpi_test(NAME GeneralVoFFaceSurface_Stripes_3procs
  TARGET testGeneralVoFFaceSurface PROCS 3
  ARGS "--gtest_filter=GeneralVoFFaceSurface.ThreeOrMoreTangentStripesUseAuthenticatedHaloFaceSupport")
simplefluid_add_mpi_test(NAME GeneralVoFFaceSurface_StripesEmpty_4procs
  TARGET testGeneralVoFFaceSurface PROCS 4
  ARGS "--gtest_filter=GeneralVoFFaceSurface.ThreeOrMoreTangentStripesRetainCollectivesWithAnEmptyOwner")

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  foreach(PROCS 2 4)
    simplefluid_add_mpi_test(NAME GeneralVoFFailures_${PROCS}procs
      TARGET testGeneralVoFFailures PROCS ${PROCS}
      LABELS "mpi;vof;diagnostics;collective;empty_rank" TIMEOUT 90)
  endforeach()
endif()

foreach(PROCS 2 4)
  simplefluid_add_mpi_test(NAME GeneralVoFSignedMomentum_${PROCS}procs
    TARGET testConservativeVoFTransport PROCS ${PROCS}
    ARGS "--gtest_filter=ConservativeVoFTransportTest.SignedNonincidentCounterflowUsesPhysicalOperatorAndCollectiveMode"
    LABELS "mpi;vof;signed;momentum;conservation;empty_rank" TIMEOUT 180)
endforeach()

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  foreach(PROCS 2 4)
    simplefluid_add_mpi_test(NAME GeneralVoFInterfaceFlow_${PROCS}procs
      TARGET testGeneralVoFInterfaceFlow PROCS ${PROCS}
      LABELS "mpi;vof;interface;conservation;transaction;empty_rank" TIMEOUT 300)
  endforeach()
endif()

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  foreach(PROCS 2 4)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryCatalog_${PROCS}procs
      TARGET testGeneralVoFCapillarySurface PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFCapillarySurface.DefaultFilterPreservesCubeVolumeAndRejectsUnderresolvedCurvature:GeneralVoFCapillarySurface.FilteringPreservesEachOfTwoDistinctEnclosedComponentVolumes:GeneralVoFCapillarySurface.FilteredEnclosedComponentCoexistsWithUnrelatedMixedPlane:GeneralVoFCapillarySurface.UnequalCellVolumesRetainVolumeWeightedFilteredConservation"
      LABELS "mpi;vof;capillary;surface;conservation;empty_rank" TIMEOUT 600)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryTopology_${PROCS}procs
      TARGET testGeneralVoFCapillarySurface PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFCapillarySurface.SingleCellPlateauRequestsResolutionWithoutChangingInventories:GeneralVoFCapillarySurface.FilterRejectsEdgeAndVertexContactsAsAmbiguousTopology:GeneralVoFCapillarySurface.FilterRejectsMergingCavitiesAndIncompleteSupport:GeneralVoFCapillarySurface.SpatialPartitionIncompleteHaloRejectsWithResolutionRecordOnEveryRank:GeneralVoFCapillarySurface.DivergentRankModelControlsRejectBeforeAnyConditionalCollective"
      LABELS "mpi;vof;capillary;topology;collective;empty_rank" TIMEOUT 600)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryFlow_${PROCS}procs
      TARGET testGeneralVoFCapillarySurface PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFCapillarySurface.ResolvedFilteredBinaryDropHasInspectibleCurvature:GeneralVoFCapillarySurface.MixedAndAlignedPlanarSectionsShareOneImmutableSourceCatalog:GeneralVoFCapillarySurface.RejectedInvalidInputsRetainAcceptedCapillaryDiagnostics"
      LABELS "mpi;vof;capillary;flow;transaction;empty_rank" TIMEOUT 600)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryRollback_${PROCS}procs
      TARGET testGeneralVoFCapillarySurface PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFCapillarySurface.MpiCapillaryFailureDuringRepartitionRetainsCompleteAcceptedPublication"
      LABELS "mpi;vof;capillary;transaction;empty_rank" TIMEOUT 600)
  endforeach()
endif()

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
  foreach(PROCS 2 4)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryDependencySupport_${PROCS}procs
      TARGET testGeneralVoFFlowWorkspace PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFFlowWorkspaceTest.CapillaryExtensionMatchesOrthogonalAndSkewSupportIncludingEmptyOwners:GeneralVoFFlowWorkspaceTest.CapillaryExtensionRetainsTracesAndRejectsInvalidInputsCollectively"
      LABELS "mpi;vof;capillary;operator;collective;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME GeneralVoFCapillaryDependencyProjection_${PROCS}procs
      TARGET testGeneralVoFFlowWorkspace PROCS ${PROCS}
      ARGS "--gtest_filter=GeneralVoFFlowWorkspaceTest.CapillaryExtensionInactiveCurvatureLeavesPhysicalProjectionIdentical"
      LABELS "mpi;vof;capillary;operator;collective;empty_rank" TIMEOUT 180)
  endforeach()
endif()
