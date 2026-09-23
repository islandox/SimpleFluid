
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
