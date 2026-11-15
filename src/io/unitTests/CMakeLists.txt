# Unit tests for io

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Mesh GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  simplefluid_discover_tests(${TEST_NAME})
endforeach()

foreach(_input_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME STKFaceAuthority_${_input_ranks}procs
        TARGET testSTKFaceAuthority PROCS ${_input_ranks}
        LABELS "mpi;io;unstructured;source_authority" TIMEOUT 120)
endforeach()

foreach(_input_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME STKMeshInput_${_input_ranks}procs
        TARGET testSTKMeshInput PROCS ${_input_ranks}
        LABELS "mpi;io;unstructured;import;empty_rank" TIMEOUT 120)
endforeach()
