
# Unit tests for fields

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testGhostSyncMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testMeshToMeshTransferMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testPartitionedFieldSync\\.cc$")

add_executable(testFields ${TEST_SOURCES})
target_link_libraries(testFields SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testFields)
simplefluid_discover_tests(testFields)

simplefluid_add_mpi_test(
  NAME MeshToMeshTransferRZ_2procs
  TARGET testFields
  PROCS 2
  ARGS "--gtest_filter=MeshToMeshTransferRZTest.*"
  TIMEOUT 120
  LABELS "mpi;field_transfer;ale")

add_executable(testMeshToMeshTransferMultiRank testMeshToMeshTransferMultiRank.cc)
target_link_libraries(testMeshToMeshTransferMultiRank SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testMeshToMeshTransferMultiRank)
simplefluid_add_mpi_test(
  NAME MeshToMeshTransferMultiRank_2procs
  TARGET testMeshToMeshTransferMultiRank
  PROCS 2
  TIMEOUT 60
  LABELS mpi)
simplefluid_add_mpi_test(
  NAME MeshToMeshReverseCoverage_4procs
  TARGET testMeshToMeshTransferMultiRank
  PROCS 4
  ARGS "--gtest_filter=*ReverseOverlapPreservesRankAndLocalDonorOrder:*ReverseCoverageBoundedReductionMatchesDenseOwnerResults"
  TIMEOUT 60
  LABELS "mpi;field_transfer")

add_executable(testGhostSyncMultiRank testGhostSyncMultiRank.cc)
target_link_libraries(testGhostSyncMultiRank SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testGhostSyncMultiRank)
simplefluid_add_mpi_test(
  NAME GhostSyncMultiRank_2procs
  TARGET testGhostSyncMultiRank
  PROCS 2
  LABELS mpi)
simplefluid_add_mpi_test(
  NAME GhostSyncMultiRank_4procs
  TARGET testGhostSyncMultiRank
  PROCS 4
  ARGS "--gtest_filter=*LargeMesh*"
  LABELS mpi)

# Field storage is tested above the mesh layer, including custom mesh providers.
add_executable(testPartitionedFieldSync
  testPartitionedFieldSync.cc
  PartitionedUnstructuredFieldProvider.cc)
target_link_libraries(testPartitionedFieldSync SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testPartitionedFieldSync)
simplefluid_discover_tests(testPartitionedFieldSync)
foreach(_field_partition_ranks IN ITEMS 2 4)
  simplefluid_add_mpi_test(
    NAME PartitionedFieldSync_${_field_partition_ranks}procs
    TARGET testPartitionedFieldSync
    PROCS ${_field_partition_ranks}
    LABELS "mpi;fields;partition;mesh_reordering;empty_rank"
    TIMEOUT 120)
endforeach()
