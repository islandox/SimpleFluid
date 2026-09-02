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
    ARGS "--gtest_filter=FieldStoredOperatorsTest.MappedTransportValidationAndCachedGraphFailuresAreCollective"
    LABELS "mpi;non_orthogonal;field_stored"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarTransportDiscretizationValidation_2procs
    TARGET testScalarTransportDiscretization
    PROCS 2
    ARGS "--gtest_filter=ScalarTransportDiscretizationTest.AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively"
    LABELS "mpi;scalar_transport"
    TIMEOUT 120)
