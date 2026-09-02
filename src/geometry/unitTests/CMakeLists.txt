
# Unit tests for geometry

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testOrthogonalDiffusionProfile\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Mesh GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

simplefluid_add_mpi_test(
    NAME WallYPlusStatistics_2procs
    TARGET testYPlusBoundaryLayerController
    PROCS 2
    ARGS "--gtest_filter=WallYPlusStatisticsTest.ReducesAcrossRanksWhenSomeRanksOwnNoWallFaces"
    LABELS "mpi;turbulence;mesh_adaptation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME SemiStructuredMeshHandleSerialOnly_2procs
    TARGET testMeshHandle
    PROCS 2
    ARGS "--gtest_filter=MeshHandleTest.SemiStructuredConstructionIsSerialOnly"
    LABELS "mpi;mesh_handle;semi_structured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PartitionedMeshHandleCommunicator_2procs
    TARGET testMeshHandle
    PROCS 2
    ARGS "--gtest_filter=MeshHandleTest.PreservesPartitionedMeshCommunicator"
    LABELS "mpi;mesh_handle;partitioned_unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME MeshReorderingFactory_2procs
    TARGET testMeshReorderingFactory
    PROCS 2
    ARGS "--gtest_filter=MeshReorderingFactoryMultiRankTest.*"
    LABELS "mpi;mesh_reordering;solid_subdomain"
    TIMEOUT 120)

# The diffusion profiling test also needs the solvers library
add_executable(testOrthogonalDiffusionProfile testOrthogonalDiffusionProfile.cc)
target_link_libraries(testOrthogonalDiffusionProfile
      SimpleFluid::Mesh
      SimpleFluid::Solvers
      GTest::gtest_main)
simplefluid_configure_test_executable(testOrthogonalDiffusionProfile)
# This million-cell profiling executable is run manually with
# --gtest_also_run_disabled_tests and is intentionally not registered with CTest.
