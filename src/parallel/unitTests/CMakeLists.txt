
# Multi-rank MetisPartitioner tests
add_executable(testMeshPartitioner testMeshPartitioner.cc)
target_link_libraries(testMeshPartitioner SimpleFluid::Mesh GTest::gtest_main)
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

add_executable(testNativeMeshDistribution testNativeMeshDistribution.cc)
target_link_libraries(testNativeMeshDistribution SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testNativeMeshDistribution)
simplefluid_discover_tests(testNativeMeshDistribution)
foreach(_native_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME NativeMeshDistribution_${_native_ranks}procs
        TARGET testNativeMeshDistribution PROCS ${_native_ranks}
        LABELS "mpi;geometry;unstructured;import;empty_rank" TIMEOUT 120)
endforeach()

add_executable(testSparseOwnerExchange testSparseOwnerExchange.cc)
target_link_libraries(testSparseOwnerExchange SimpleFluid::Mesh GTest::gtest_main)
simplefluid_configure_test_executable(testSparseOwnerExchange)
simplefluid_discover_tests(testSparseOwnerExchange)
foreach(_owner_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME SparseOwnerExchange_${_owner_procs}procs
    TARGET testSparseOwnerExchange PROCS ${_owner_procs} LABELS "mpi;amr;exchange" TIMEOUT 60)
endforeach()

add_executable(testSparseNativeMesh testSparseNativeMesh.cc)
target_link_libraries(testSparseNativeMesh SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testSparseNativeMesh)
simplefluid_discover_tests(testSparseNativeMesh)
foreach(_native_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME SparseNativeMesh_${_native_procs}procs
    TARGET testSparseNativeMesh PROCS ${_native_procs} LABELS "mpi;amr;native" TIMEOUT 90)
endforeach()

add_executable(testSparseNativePeriodic testSparseNativePeriodic.cc)
target_link_libraries(testSparseNativePeriodic SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testSparseNativePeriodic)
simplefluid_discover_tests(testSparseNativePeriodic)
foreach(_periodic_native_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME SparseNativePeriodic_${_periodic_native_procs}procs
    TARGET testSparseNativePeriodic PROCS ${_periodic_native_procs} LABELS "mpi;amr;native;periodic" TIMEOUT 120)
endforeach()

add_executable(testFrozenNative testFrozenNative.cc)
target_link_libraries(testFrozenNative SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testFrozenNative)
simplefluid_discover_tests(testFrozenNative)
foreach(_frozen_native_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME FrozenNative_${_frozen_native_procs}procs
    TARGET testFrozenNative PROCS ${_frozen_native_procs} LABELS "mpi;amr;native;frozen" TIMEOUT 180)
endforeach()

add_executable(testNativeOwnerPacketAdmission testNativeOwnerPacketAdmission.cc)
target_link_libraries(testNativeOwnerPacketAdmission SimpleFluid::Mesh GTest::gtest_main)
simplefluid_configure_test_executable(testNativeOwnerPacketAdmission)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(testNativeOwnerPacketAdmission PRIVATE
        -fno-fast-math -ffp-contract=off -frounding-math -fno-lto)
    set_target_properties(testNativeOwnerPacketAdmission PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF DISABLE_PRECOMPILE_HEADERS ON)
endif()
simplefluid_discover_tests(testNativeOwnerPacketAdmission)
foreach(_native_packet_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME NativeOwnerPacketAdmission_${_native_packet_ranks}procs
        TARGET testNativeOwnerPacketAdmission PROCS ${_native_packet_ranks}
        LABELS "mpi;amr;native;admission" TIMEOUT 120)
endforeach()
