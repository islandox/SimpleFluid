# Unit tests for FVM

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  if(TEST_NAME STREQUAL "testGradientTraverse")
    target_sources(${TEST_NAME} PRIVATE
        gradient_traverse/ReferenceCases.cc
        gradient_traverse/RefreshCases.cc
        gradient_traverse/StructuredCases.cc
        gradient_traverse/LayeredCases.cc
        gradient_traverse/OnePassCases.cc)
  endif()
  if(TEST_NAME STREQUAL "testFvmLibrary" OR TEST_NAME STREQUAL "testGeometricVoF"
      OR TEST_NAME STREQUAL "testAdaptiveVoF" OR TEST_NAME STREQUAL "testAdaptiveVoFThreeDimensional"
      OR TEST_NAME STREQUAL "testVoFPhysicalSources" OR TEST_NAME STREQUAL "testConservativeAMRRemap" OR TEST_NAME STREQUAL "testProductAMRRemap" OR TEST_NAME STREQUAL "testPolyhedralAMRRemap" OR TEST_NAME STREQUAL "testMixedAMRRemap")
    # Catch dependencies from FVM back into the higher-level solver DSO.
    target_link_libraries(${TEST_NAME} SimpleFluid::FVM GTest::gtest_main)
  else()
    target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  endif()
  simplefluid_configure_test_executable(${TEST_NAME})
  if(TEST_NAME STREQUAL "testAdaptiveVoF" OR TEST_NAME STREQUAL "testAdaptiveVoFThreeDimensional"
      OR TEST_NAME STREQUAL "testVoFPhysicalSources" OR TEST_NAME STREQUAL "testConservativeAMRRemap" OR TEST_NAME STREQUAL "testProductAMRRemap" OR TEST_NAME STREQUAL "testPolyhedralAMRRemap" OR TEST_NAME STREQUAL "testMixedAMRRemap")
    simplefluid_discover_tests(${TEST_NAME}
      PROPERTIES LABELS "vof\\;amr\\;geometry\\;conservation\\;collective")
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
    ARGS "--gtest_filter=ScalarTransportDiscretizationTest.AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively"
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
