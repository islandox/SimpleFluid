
# Multi-rank MetisPartitioner tests
add_executable(testMeshPartitioner testMeshPartitioner.cc)
target_link_libraries(testMeshPartitioner SimpleFluid GTest::gtest_main)
simplefluid_reuse_precompiled_header(testMeshPartitioner)
if(CMAKE_CONFIGURATION_TYPES)
  set(METIS_PART_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\${_metis_config}/testMeshPartitioner")
else()
  set(METIS_PART_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testMeshPartitioner")
endif()
set(METIS_PART_CTEST_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/MeshPartitionerCTest.cmake")
if(CMAKE_CONFIGURATION_TYPES)
  file(WRITE "${METIS_PART_CTEST_FILE}"
       "if(CTEST_CONFIGURATION_TYPE)\n"
       "  set(_metis_config \"\${CTEST_CONFIGURATION_TYPE}\")\n"
       "else()\n"
       "  set(_metis_config \"Debug\")\n"
       "endif()\n"
       "add_test(MeshPartitioner_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_2procs PROPERTIES PROCESSORS 2)\n"
       "add_test(MeshPartitioner_4procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"4\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_4procs PROPERTIES PROCESSORS 4)\n")
else()
  file(WRITE "${METIS_PART_CTEST_FILE}"
       "add_test(MeshPartitioner_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_2procs PROPERTIES PROCESSORS 2)\n"
       "add_test(MeshPartitioner_4procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"4\" \"${METIS_PART_EXECUTABLE}\")\n"
       "set_tests_properties(MeshPartitioner_4procs PROPERTIES PROCESSORS 4)\n")
endif()
set_property(DIRECTORY APPEND PROPERTY
             TEST_INCLUDE_FILES "${METIS_PART_CTEST_FILE}")
