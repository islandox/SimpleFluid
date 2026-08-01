
# Multi-rank MetisPartitioner tests
add_executable(testMeshPartitioner testMeshPartitioner.cc)
target_link_libraries(testMeshPartitioner SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testMeshPartitioner)
if(CMAKE_CONFIGURATION_TYPES)
  set(METIS_PART_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\${_metis_config}/testMeshPartitioner")
else()
  set(METIS_PART_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testMeshPartitioner")
endif()
set(METIS_PART_CTEST_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/MeshPartitionerCTest.cmake")
if(NOT SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0 AND SIMPLEFLUID_MAX_TEST_PROCS LESS 4)
  set(_metis_4proc_test "")
else()
  set(_metis_4proc_test "add_test(MeshPartitioner_4procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"4\" \"${METIS_PART_EXECUTABLE}\")\nset_tests_properties(MeshPartitioner_4procs PROPERTIES PROCESSORS 4 TIMEOUT 60 LABELS mpi)\n")
endif()
if(CMAKE_CONFIGURATION_TYPES)
  file(WRITE "${METIS_PART_CTEST_FILE}"
       "if(CTEST_CONFIGURATION_TYPE)\n"
       "  set(_metis_config \"\${CTEST_CONFIGURATION_TYPE}\")\n"
       "else()\n"
       "  set(_metis_config \"Debug\")\n"
       "endif()\n"
       "add_test(MeshPartitioner_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_2procs PROPERTIES PROCESSORS 2 TIMEOUT 60 LABELS mpi)\n"
       "${_metis_4proc_test}")
else()
  file(WRITE "${METIS_PART_CTEST_FILE}"
       "add_test(MeshPartitioner_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_2procs PROPERTIES PROCESSORS 2 TIMEOUT 60 LABELS mpi)\n"
       "${_metis_4proc_test}")
endif()
set_property(DIRECTORY APPEND PROPERTY
             TEST_INCLUDE_FILES "${METIS_PART_CTEST_FILE}")

add_executable(testMPIInterface testMPIInterface.cc)
target_link_libraries(testMPIInterface SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testMPIInterface)
if(CMAKE_CONFIGURATION_TYPES)
  set(MPI_INTERFACE_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\${_mpi_interface_config}/testMPIInterface")
else()
  set(MPI_INTERFACE_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testMPIInterface")
endif()
set(MPI_INTERFACE_CTEST_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/MPIInterfaceCTest.cmake")
if(CMAKE_CONFIGURATION_TYPES)
  file(WRITE "${MPI_INTERFACE_CTEST_FILE}"
       "if(CTEST_CONFIGURATION_TYPE)\n"
       "  set(_mpi_interface_config \"\${CTEST_CONFIGURATION_TYPE}\")\n"
       "else()\n"
       "  set(_mpi_interface_config \"Debug\")\n"
       "endif()\n"
       "add_test(MPIInterface_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${MPI_INTERFACE_EXECUTABLE}\")\n"
       "set_tests_properties(MPIInterface_2procs PROPERTIES PROCESSORS 2 LABELS mpi)\n")
else()
  file(WRITE "${MPI_INTERFACE_CTEST_FILE}"
       "add_test(MPIInterface_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${MPI_INTERFACE_EXECUTABLE}\")\n"
       "set_tests_properties(MPIInterface_2procs PROPERTIES PROCESSORS 2 LABELS mpi)\n")
endif()
set_property(DIRECTORY APPEND PROPERTY
             TEST_INCLUDE_FILES "${MPI_INTERFACE_CTEST_FILE}")
