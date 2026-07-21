# Unit tests for FVM

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
if(NOT SIMPLEFLUID_ENABLE_CXX_MODULES)
  list(REMOVE_ITEM TEST_SOURCES
       "${CMAKE_CURRENT_SOURCE_DIR}/testFvmModule.cc")
endif()
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  if(SIMPLEFLUID_ENABLE_CXX_MODULES AND TEST_NAME STREQUAL "testFvmModule")
    set_target_properties(${TEST_NAME} PROPERTIES CXX_SCAN_FOR_MODULES ON)
    set_source_files_properties(${TEST_SRC}
        PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_OPTIONS
        "$<$<CXX_COMPILER_ID:Clang>:-fno-delayed-template-parsing>")
  endif()
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

if(MPIEXEC_EXECUTABLE
   AND (SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
        OR SIMPLEFLUID_MAX_TEST_PROCS GREATER_EQUAL 2))
  add_test(
    NAME NonOrthogonalPartitionFaces_2procs
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS}
            "${MPIEXEC_NUMPROC_FLAG}" "2"
            "$<TARGET_FILE:testFvmOperators>"
            "--gtest_filter=FvmOperatorsTest.ImplicitNonOrthogonalMatrixMatchesFullResidual:FvmOperatorsTest.ImplicitNonOrthogonalRejectsInconsistentPartitionCorrectionInputs:FvmOperatorsTest.TransportAssemblersRejectInconsistentPartitionGradientInputs:FvmOperatorsTest.HybridNonOrthogonalSolveSupportsPartitionFaces:FvmOperatorsTest.VectorTransportImplicitNonOrthogonalDiffusionExpandsGradientGraph:FvmOperatorsTest.ExplicitVariableDiffusionUsesBoundaryValuesAndSparseCoefficients:FvmOperatorsTest.PhysicalTemperatureNonOrthogonalTreatmentsMatchExplicitResidual")
  set_tests_properties(NonOrthogonalPartitionFaces_2procs PROPERTIES
    LABELS "mpi;non_orthogonal"
    PROCESSORS 2
    TIMEOUT 120)
endif()
