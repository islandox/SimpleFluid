# Unit tests for FVM

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

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
    ARGS "--gtest_filter=FieldStoredOperatorsTest.MappedTransportValidationAndCachedGraphFailuresAreCollective:FieldStoredOperatorsTest.WeightedTransportCachedTopologyMatchesFreshAssembly:FieldStoredOperatorsTest.ZeroDiffusionNeumannTransportRetainsValidationAndReusableGraph:FieldStoredOperatorsTest.WeightedScalarZeroDiffusionRetainsBoundaryCorrections"
    LABELS "mpi;non_orthogonal;field_stored"
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
