# Unit tests for FVM

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  if(TEST_NAME STREQUAL "testExactGeneralVoF")
    target_sources(${TEST_NAME} PRIVATE exact/GeneralVoFRoundtrip.cc)
  endif()
  if(TEST_NAME STREQUAL "testGradientTraverse")
    target_sources(${TEST_NAME} PRIVATE
        gradient_traverse/ReferenceCases.cc
        gradient_traverse/RefreshCases.cc
        gradient_traverse/StructuredCases.cc
        gradient_traverse/LayeredCases.cc
        gradient_traverse/OnePassCases.cc)
  endif()
  if(TEST_NAME STREQUAL "testGeneralVoFPreparedTransport" OR TEST_NAME STREQUAL "testGeneralVoFFaceTransport" OR TEST_NAME STREQUAL "testExactGeneralVoF" OR TEST_NAME STREQUAL "testExactPolyhedralReconstruction" OR TEST_NAME STREQUAL "testFiniteVoFRemap" OR TEST_NAME STREQUAL "testBoundedPhaseArithmetic" OR TEST_NAME STREQUAL "testGeneralVoFFinite" OR TEST_NAME STREQUAL "testOrdinaryVoFSweep" OR TEST_NAME STREQUAL "testExactFaceTransfer" OR TEST_NAME STREQUAL "testExactPhaseTransfer" OR TEST_NAME STREQUAL "testPolyhedralReconstruction" OR TEST_NAME STREQUAL "testGeneralVoF" OR TEST_NAME STREQUAL "testFvmLibrary" OR TEST_NAME STREQUAL "testGeometricVoF"
      OR TEST_NAME STREQUAL "testAdaptiveVoF" OR TEST_NAME STREQUAL "testAdaptiveVoFThreeDimensional"
      OR TEST_NAME STREQUAL "testVoFPhysicalSources" OR TEST_NAME STREQUAL "testConservativeAMRRemap" OR TEST_NAME STREQUAL "testProductAMRRemap" OR TEST_NAME STREQUAL "testPolyhedralAMRRemap" OR TEST_NAME STREQUAL "testMixedAMRRemap" OR TEST_NAME STREQUAL "testDistributedNativeOperators" OR TEST_NAME STREQUAL "testDistributedAdaptiveState")
    # Catch dependencies from FVM back into the higher-level solver DSO.
    target_link_libraries(${TEST_NAME} SimpleFluid::FVM GTest::gtest_main)
  else()
    target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  endif()
  simplefluid_configure_test_executable(${TEST_NAME})
  if(TEST_NAME STREQUAL "testExactGeneralVoF" OR TEST_NAME STREQUAL "testExactPolyhedralReconstruction")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;exact\\;geometry\\;conservation\\;collective")
  elseif(TEST_NAME STREQUAL "testExactFaceTransfer" OR TEST_NAME STREQUAL "testExactPhaseTransfer" OR TEST_NAME STREQUAL "testAdaptiveVoF" OR TEST_NAME STREQUAL "testAdaptiveVoFThreeDimensional"
      OR TEST_NAME STREQUAL "testVoFPhysicalSources" OR TEST_NAME STREQUAL "testConservativeAMRRemap" OR TEST_NAME STREQUAL "testProductAMRRemap" OR TEST_NAME STREQUAL "testPolyhedralAMRRemap" OR TEST_NAME STREQUAL "testMixedAMRRemap" OR TEST_NAME STREQUAL "testDistributedNativeOperators" OR TEST_NAME STREQUAL "testDistributedAdaptiveState")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;amr\\;geometry\\;conservation\\;collective")
  elseif(TEST_NAME STREQUAL "testGeneralVoF" OR TEST_NAME STREQUAL "testPolyhedralReconstruction")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;geometry\\;conservation\\;collective")
  elseif(TEST_NAME STREQUAL "testGeometricVoF")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;geometry\\;conservation")
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

simplefluid_add_mpi_test(
    NAME PolyhedralAMRRemap_2procs
    TARGET testPolyhedralAMRRemap
    PROCS 2
    LABELS "amr;polyhedral;conservation;collective"
    TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME ProductAMRRemap_2procs
    TARGET testProductAMRRemap
    PROCS 2
    LABELS "amr;geometry;conservation;collective;constructed_product"
    TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME ConservativeAMRRemap_2procs
    TARGET testConservativeAMRRemap
    PROCS 2
    LABELS "vof;amr;geometry;conservation;collective"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME VoFPhysicalSources_2procs
    TARGET testVoFPhysicalSources
    PROCS 2
    LABELS "vof;amr;geometry;source;conservation;collective"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME AdaptiveVoF_2procs
    TARGET testAdaptiveVoF
    PROCS 2
    LABELS "vof;amr;geometry;conservation;collective"
    TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME AdaptiveVoFThreeDimensional_2procs
    TARGET testAdaptiveVoFThreeDimensional
    PROCS 2
    LABELS "vof;amr;geometry;conservation;collective"
    TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME CompositePartitionOperators_2procs
    TARGET testCompositePartitionOperators
    PROCS 2
    LABELS "mpi;multi_region;partition;transport"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ALETransport_2procs
    TARGET testALETransport
    PROCS 2
    LABELS "mpi;ale;transport;conservation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NonOrthogonalPartitionFaces_2procs
    TARGET testFvmOperators
    PROCS 2
    ARGS "--gtest_filter=FvmOperatorsTest.ImplicitNonOrthogonalMatrixMatchesFullResidual:FvmOperatorsTest.ImplicitNonOrthogonalRejectsInconsistentPartitionCorrectionInputs:FvmOperatorsTest.TransportAssemblersRejectInconsistentPartitionGradientInputs:FvmOperatorsTest.HybridNonOrthogonalSolveSupportsPartitionFaces:FvmOperatorsTest.VectorTransportImplicitNonOrthogonalDiffusionExpandsGradientGraph:FvmOperatorsTest.ExplicitVariableDiffusionUsesBoundaryValuesAndSparseCoefficients:FvmOperatorsTest.PhysicalTemperatureNonOrthogonalTreatmentsMatchExplicitResidual:FvmOperatorsTest.MappedUpwindConvectionUsesPartitionFaceGhostFlux"
    LABELS "mpi;non_orthogonal"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FieldStoredTransportValidation_2procs
    TARGET testFieldStoredOperators
    PROCS 2
    ARGS "--gtest_filter=FieldStoredOperatorsTest.MappedTransportValidationAndCachedGraphFailuresAreCollective:FieldStoredOperatorsTest.MappedExternalEquivalentMapsReuseCollectively:FieldStoredOperatorsTest.WeightedTransportCachedTopologyMatchesFreshAssembly:FieldStoredOperatorsTest.ZeroDiffusionNeumannTransportRetainsValidationAndReusableGraph:FieldStoredOperatorsTest.WeightedScalarZeroDiffusionRetainsBoundaryCorrections"
    LABELS "mpi;non_orthogonal;field_stored"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME LegacyTransportValidation_2procs
    TARGET testFvmOperators
    PROCS 2
    ARGS "--gtest_filter=FvmOperatorsTest.LegacyTransportCacheAndScalarFailuresAreCollective"
    LABELS "mpi;transport;legacy"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarTransportDiscretizationValidation_2procs
    TARGET testScalarTransportDiscretization
    PROCS 2
    ARGS "--gtest_filter=ScalarTransportDiscretizationTest.AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively:ScalarTransportDiscretizationTest.LegacyAcceptedOldStorageAvoidsEndpointCancellation:ScalarTransportDiscretizationTest.LegacyAcceptedStorageRejectsUnsupportedAndDivergentInputsCollectively"
    LABELS "mpi;scalar_transport"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME StoredTransportReuse_2procs
    TARGET testStoredTransportReuse
    PROCS 2
    LABELS "mpi;transport;field_stored"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PressureGeometryAccess_2procs
    TARGET testPressureGeometryAccess
    PROCS 2
    LABELS "mpi;fvm;pressure;geometry;collective"
    TIMEOUT 30)

foreach(_cached_kernel IN ITEMS CellFluxBalanceCache StoredPressureFaceFluxCache)
  simplefluid_add_mpi_test(
      NAME ${_cached_kernel}_2procs
      TARGET test${_cached_kernel}
      PROCS 2
      LABELS "mpi;fvm;cache"
      TIMEOUT 120)
endforeach()

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(
        NAME MultiRegionTransport_${_region_procs}procs
        TARGET testMultiRegionTransportMultiRank
        PROCS ${_region_procs}
        LABELS "mpi;multi_region;transport"
        TIMEOUT 120)
endforeach()

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME MultiRegionALE_${_region_procs}procs TARGET testMultiRegionALE
        PROCS ${_region_procs} LABELS "mpi;multi_region;ale;conservation" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(
    NAME MultiRegionALE_StaleChild_2procs
    TARGET testMultiRegionALE
    PROCS 2
    ARGS "--gtest_filter=MultiRegionALETest.RankLocalStaleChildFailsValidationCollectively"
    LABELS "mpi;multi_region;ale;collective"
    TIMEOUT 20)

simplefluid_add_mpi_test(
    NAME MultiRegionALE_RankLocalTraversal_2procs
    TARGET testMultiRegionALE
    PROCS 2
    ARGS "--gtest_filter=MultiRegionALETest.RankLocalTraversalFailureReleasesLeaseAndFailsCollectively"
    LABELS "mpi;multi_region;ale;collective"
    TIMEOUT 20)

simplefluid_add_mpi_test(
    NAME MultiRegionALE_MultiFaultPriority_2procs
    TARGET testMultiRegionALE
    PROCS 2
    ARGS "--gtest_filter=MultiRegionALETest.RankDivergentFaultsRetainCollectiveErrorPriority"
    LABELS "mpi;multi_region;ale;collective"
    TIMEOUT 20)

simplefluid_add_mpi_test(
    NAME MultiRegionALE_EmptyRankFailure_2procs
    TARGET testMultiRegionALE
    PROCS 2
    ARGS "--gtest_filter=MultiRegionALETest.EmptyRanksReceiveCollectiveGclFailure"
    LABELS "mpi;multi_region;ale;collective"
    TIMEOUT 20)

simplefluid_add_mpi_test(
    NAME ALETrialReplay_2procs
    TARGET testALETrialReplay
    PROCS 2
    ARGS "--gtest_filter=ALETrialReplayTest.SameTargetAndTimeStepRejectRetainedViewOnTwoRanks"
    LABELS "mpi;ale;replay;collective"
    TIMEOUT 20)

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME RegionQualification_${_region_procs}procs TARGET testRegionQualification
        PROCS ${_region_procs} LABELS "mpi;multi_region;accuracy;qualification" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME RegionDiffusionAssembly_${_region_procs}procs TARGET testRegionDiffusionAssembly
        PROCS ${_region_procs} LABELS "mpi;multi_region;diffusion;execution" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME SteadyDiffusionConvergence_${_region_procs}procs TARGET testSteadyDiffusionConvergence
        PROCS ${_region_procs} LABELS "mpi;fvm;non_orthogonal;multi_region" TIMEOUT 120)
endforeach()

foreach(_gradient_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME StoredGradientViews_${_gradient_procs}procs TARGET testStoredGradientViews
        PROCS ${_gradient_procs} LABELS "mpi;fvm;gradient;multi_region" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME RegionTransportAssembly_${_gradient_procs}procs TARGET testRegionTransportAssembly
        PROCS ${_gradient_procs} LABELS "mpi;fvm;transport;multi_region" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(
    NAME GradientTraverse_2procs
    TARGET testGradientTraverse
    PROCS 2
    LABELS "mpi;fvm;gradient;traverse"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME TransportTraverse_2procs
    TARGET testTransportTraverse
    PROCS 2
    LABELS "mpi;fvm;transport;traverse"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME AdaptiveVoFPhaseBudget_EmptyRank_9procs
    TARGET testAdaptiveVoF
    PROCS 9
    ARGS "--gtest_filter=AdaptiveVoFTest.MicroscopicPhaseOutflowCannotOverdrawCanonicalSubfaceDonors"
    LABELS "vof;amr;geometry;conservation;collective;empty_rank"
    TIMEOUT 120)

simplefluid_add_mpi_test(NAME MixedAMRRemap_2procs TARGET testMixedAMRRemap PROCS 2 LABELS "mpi;amr;mixed;conservation" TIMEOUT 180)

foreach(_distributed_native_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME DistributedNativeOperators_${_distributed_native_procs}procs
    TARGET testDistributedNativeOperators PROCS ${_distributed_native_procs} LABELS "mpi;amr;native;fvm" TIMEOUT 180)
endforeach()

foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(
    NAME DistributedAdaptiveState_${PROCS}procs
    TARGET testDistributedAdaptiveState
    PROCS ${PROCS}
    LABELS "amr;distributed;state;history;conservation;collective"
    TIMEOUT 240)
endforeach()

# General unsplit transport is separate from the optimized Cartesian backend.
simplefluid_add_mpi_test(
    NAME GeneralVoF_2procs
    TARGET testGeneralVoF
    PROCS 2
    LABELS "vof;geometry;conservation;collective"
    TIMEOUT 300)
simplefluid_add_mpi_test(
    NAME GeneralVoF_4procs
    TARGET testGeneralVoF
    PROCS 4
    LABELS "vof;geometry;conservation;collective"
    TIMEOUT 300)

foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(
    NAME ExactPhaseTransfer_${PROCS}procs
    TARGET testExactPhaseTransfer PROCS ${PROCS}
    LABELS "vof;amr;exact;history;conservation;collective"
    TIMEOUT 300)
endforeach()

foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(
    NAME ExactFaceTransfer_${PROCS}procs
    TARGET testExactFaceTransfer PROCS ${PROCS}
    LABELS "vof;amr;exact;face_flux;conservation;collective"
    TIMEOUT 300)
endforeach()

foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(
    NAME PolyhedralReconstruction_${PROCS}procs
    TARGET testPolyhedralReconstruction PROCS ${PROCS}
    LABELS "vof;geometry;resolution;finite;collective"
    TIMEOUT 180)
endforeach()

foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME GeneralVoFFinite_${PROCS}procs TARGET testGeneralVoFFinite PROCS ${PROCS}
    LABELS "vof;geometry;conservation;collective;finite" TIMEOUT 300)
endforeach()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(testGeneralVoFPreparedTransport.cc testFiniteVoFRemap.cc testBoundedPhaseArithmetic.cc testGeneralVoF.cc testGeneralVoFFinite.cc testOrdinaryVoFSweep.cc
        testExactGeneralVoF.cc exact/GeneralVoFRoundtrip.cc PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()

# Independent finite AMR remap oracles include explicitly empty owners.
foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME FiniteVoFRemap_${PROCS}procs TARGET testFiniteVoFRemap PROCS ${PROCS}
    LABELS "vof;amr;finite;history;conservation;collective;empty_rank" TIMEOUT 300)
endforeach()

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredDistributedOperators_${_semi_ranks}procs
        TARGET testSemiStructuredDistributedOperators PROCS ${_semi_ranks}
        LABELS "mpi;semi_structured;fields;transport;diffusion;conservation;empty_rank" TIMEOUT 120)
endforeach()

foreach(_ortho_ranks IN ITEMS 2 4 8)
    simplefluid_add_mpi_test(NAME OrthogonalDistributedOperators_${_ortho_ranks}procs
        TARGET testOrthogonalDistributedOperators PROCS ${_ortho_ranks}
        LABELS "mpi;orthogonal;fields;transport;conservation;empty_rank" TIMEOUT 120)
endforeach()

# Rational transport and explicit exact/finite handoffs are independently selectable.
foreach(PROCS IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME ExactGeneralVoF_${PROCS}procs TARGET testExactGeneralVoF PROCS ${PROCS}
    LABELS "vof;exact;geometry;conservation;collective" TIMEOUT 300)
endforeach()

foreach(PROCS 2 4)
  simplefluid_add_mpi_test(NAME GeneralVoFFaceTransport_${PROCS}procs
    TARGET testGeneralVoFFaceTransport PROCS ${PROCS}
    ARGS "--gtest_filter=GeneralVoFFaceTransport.*")
endforeach()

foreach(PROCS 2 4)
  simplefluid_add_mpi_test(NAME GeneralVoFPreparedTransport_${PROCS}procs
    TARGET testGeneralVoFPreparedTransport PROCS ${PROCS}
    ARGS "--gtest_filter=GeneralVoFPreparedTransport.*"
    LABELS "vof;transport;conservation;collective;empty_rank" TIMEOUT 180)
endforeach()
