# Unit tests for FVM

# These consumers must compile against declarations without the shared PCH.
set_source_files_properties(
    testFvmLibrary.cc
    testStoredPressureFaceFluxCache.cc
    testMultiRegionALE.cc
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

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
  if(TEST_NAME STREQUAL "testFvmLibrary")
    # Link the FVM DSO alone. GNU ld reports undefined shared-library
    # symbols when creating an executable, so this fails on Linux too if FVM
    # regresses to depending on the solver DSO (macOS ld64 fails first).
    target_link_libraries(${TEST_NAME} SimpleFluid::FVM GTest::gtest_main)
  else()
    # Tests calling the solver-owned steady-diffusion entry points link the
    # consumer layer; FVM-side export coverage stays in
    # simplefluid_fvm_elf_export_boundary.
    target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  endif()
  simplefluid_configure_test_executable(${TEST_NAME})
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

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
    NAME QbmmPlanarAleGcl_4procs
    TARGET testALETransport
    PROCS 4
    ARGS "--gtest_filter=ALETransportTest.GclPreservesUniformPopulationChannelsWithCompatibleTraces"
    LABELS "mpi;qbmm;ale;transport;conservation;empty_rank"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NonOrthogonalPartitionFaces_2procs
    TARGET testFvmOperators
    PROCS 2
    ARGS "--gtest_filter=FvmOperatorsTest.ImplicitNonOrthogonalMatrixMatchesFullResidual:FvmOperatorsTest.ImplicitNonOrthogonalRejectsInconsistentPartitionCorrectionInputs:FvmOperatorsTest.TransportAssemblersRejectInconsistentPartitionGradientInputs:FvmOperatorsTest.HybridNonOrthogonalSolveSupportsPartitionFaces:FvmOperatorsTest.VectorTransportImplicitNonOrthogonalDiffusionExpandsGradientGraph:FvmOperatorsTest.ExplicitVariableDiffusionUsesBoundaryValuesAndSparseCoefficients:FvmOperatorsTest.PhysicalTemperatureNonOrthogonalTreatmentsMatchExplicitResidual:FvmOperatorsTest.MappedUpwindConvectionUsesPartitionFaceGhostFlux"
    LABELS "mpi;non_orthogonal"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativePeriodicTopology_2procs
    TARGET testFvmOperators
    PROCS 2
    ARGS "--gtest_filter=FvmOperatorsTest.PeriodicTopologyUsesCanonicalNativeNeighbors"
    LABELS "mpi;periodic;topology"
    TIMEOUT 60)

simplefluid_add_mpi_test(
    NAME FieldStoredTransportValidation_2procs
    TARGET testFieldStoredOperators
    PROCS 2
    ARGS "--gtest_filter=FieldStoredOperatorsTest.MappedTransportValidationAndCachedGraphFailuresAreCollective:FieldStoredOperatorsTest.MappedExternalEquivalentMapsReuseCollectively:FieldStoredOperatorsTest.WeightedTransportCachedTopologyMatchesFreshAssembly:FieldStoredOperatorsTest.ZeroDiffusionNeumannTransportRetainsValidationAndReusableGraph:FieldStoredOperatorsTest.WeightedScalarZeroDiffusionRetainsBoundaryCorrections"
    LABELS "mpi;non_orthogonal;field_stored"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativeTransportValidation_2procs
    TARGET testFvmOperators
    PROCS 2
    ARGS "--gtest_filter=FvmOperatorsTest.NativeTransportCacheAndScalarFailuresAreCollective"
    LABELS "mpi;transport;native"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarTransportDiscretizationValidation_2procs
    TARGET testScalarTransportDiscretization
    PROCS 2
    ARGS "--gtest_filter=ScalarTransportDiscretizationTest.AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively:ScalarTransportDiscretizationTest.CartesianAcceptedOldStorageAvoidsEndpointCancellation:ScalarTransportDiscretizationTest.CartesianAcceptedStorageRejectsUnsupportedAndDivergentInputsCollectively"
    LABELS "mpi;scalar_transport"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME StoredTransportReuse_2procs
    TARGET testStoredTransportReuse
    PROCS 2
    LABELS "mpi;transport;field_stored"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME MultiColumnTransport_2procs
    TARGET testMultiColumnTransport
    PROCS 2
    LABELS "mpi;transport;multi_rhs;ale;empty_rank"
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

foreach(_steady_ranks IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME SteadyDiagnostics_${_steady_ranks}procs
    TARGET testSteadyDiagnostics PROCS ${_steady_ranks}
    LABELS "mpi;fvm;steady;diagnostics;empty_rank" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(
    NAME BasicTransportAssembly_2procs
    TARGET testBasicTransportAssembly
    PROCS 2
    LABELS "mpi;fvm;transport;native"
    TIMEOUT 120)

# Custom meshes own their implementation providers outside public headers.
target_sources(testStoredTransportReuse PRIVATE transport_providers/EpochMatrix.cc)
target_sources(testStoredPressureFaceFluxCache PRIVATE transport_providers/PressureFlux.cc)
target_sources(testMultiRegionALE PRIVATE transport_providers/ALEValidation.cc)
target_sources(testRegionTransportAssembly PRIVATE
    transport_providers/ReferenceGeometry.cc
    transport_providers/ReferenceWeightedScalar.cc
    transport_providers/ReferenceMomentum.cc
    transport_providers/ReferencePressureGeometry.cc)

# Test-owned generic providers have the same private visibility boundary as
# library implementation units. Bypass the executable PCH with different flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_stored_transport_test_providers
        transport_providers/EpochMatrix.cc
        transport_providers/PressureFlux.cc
        transport_providers/ALEValidation.cc
        transport_providers/ReferenceGeometry.cc
        transport_providers/ReferenceWeightedScalar.cc
        transport_providers/ReferenceMomentum.cc
        transport_providers/ReferencePressureGeometry.cc)
    set_source_files_properties(${_stored_transport_test_providers}
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    set_property(SOURCE ${_stored_transport_test_providers} APPEND PROPERTY
        COMPILE_OPTIONS -fvisibility=hidden -fvisibility-inlines-hidden)
    unset(_stored_transport_test_providers)
endif()

foreach(_closure_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(
        NAME PressurePoissonClosure_${_closure_procs}procs
        TARGET testPressurePoissonClosure
        PROCS ${_closure_procs}
        ARGS "--gtest_filter=PressurePoissonClosureTest.MultiRank*"
        LABELS "mpi;pressure_closure;multi_region"
        TIMEOUT 120)
endforeach()
