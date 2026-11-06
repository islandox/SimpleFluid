
# Multi-rank MetisPartitioner tests
add_executable(testMeshPartitioner testMeshPartitioner.cc)
target_link_libraries(testMeshPartitioner SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testMeshPartitioner)
simplefluid_add_mpi_test(
  NAME MeshPartitioner_2procs
  TARGET testMeshPartitioner
  PROCS 2
  LABELS mpi
  TIMEOUT 60)
simplefluid_add_mpi_test(
  NAME MeshPartitioner_4procs
  TARGET testMeshPartitioner
  PROCS 4
  LABELS mpi
  TIMEOUT 60)

add_executable(testCompositePartition testCompositePartition.cc)
target_link_libraries(testCompositePartition SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testCompositePartition)
simplefluid_discover_tests(testCompositePartition)
simplefluid_add_mpi_test(
  NAME CompositePartition_2procs
  TARGET testCompositePartition
  PROCS 2
  LABELS mpi
  TIMEOUT 120)
simplefluid_add_mpi_test(
  NAME CompositePartition_4procs
  TARGET testCompositePartition
  PROCS 4
  LABELS mpi
  TIMEOUT 120)

add_executable(testMPIInterface testMPIInterface.cc)
target_link_libraries(testMPIInterface SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testMPIInterface)
simplefluid_add_mpi_test(
  NAME MPIInterface_2procs
  TARGET testMPIInterface
  PROCS 2
  LABELS mpi)

foreach(_centroid_ranks IN ITEMS 1 2 3 4)
  simplefluid_add_mpi_test(NAME PartitionedGradedQuadCentroids_${_centroid_ranks}procs
    TARGET testMeshPartitioner PROCS ${_centroid_ranks}
    ARGS "--gtest_filter=MeshPartitionerGeometryTest.GradedQuadCentroidsMatchSerialRule"
    LABELS "mpi;geometry;partition;centroid" TIMEOUT 60)
endforeach()
