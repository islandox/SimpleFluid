
# Unit tests for fields

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testGhostSyncMultiRank\\.cc$")

add_executable(testFields ${TEST_SOURCES})
target_link_libraries(testFields SimpleFluid::Fields GTest::gtest_main)
simplefluid_reuse_precompiled_header(testFields)
simplefluid_discover_tests(testFields)

add_executable(testGhostSyncMultiRank testGhostSyncMultiRank.cc)
target_link_libraries(testGhostSyncMultiRank SimpleFluid::Fields GTest::gtest_main)
simplefluid_reuse_precompiled_header(testGhostSyncMultiRank)
if(CMAKE_CONFIGURATION_TYPES)
  set(GHOST_SYNC_MULTI_RANK_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\${_ghost_sync_config}/testGhostSyncMultiRank")
else()
  set(GHOST_SYNC_MULTI_RANK_EXECUTABLE
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testGhostSyncMultiRank")
endif()
set(GHOST_SYNC_MULTI_RANK_CTEST_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/GhostSyncMultiRankCTest.cmake")
if(NOT SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0 AND SIMPLEFLUID_MAX_TEST_PROCS LESS 4)
  set(_ghost_4proc_test "")
else()
  set(_ghost_4proc_test "add_test(GhostSyncMultiRank_4procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"4\" \"${GHOST_SYNC_MULTI_RANK_EXECUTABLE}\" --gtest_filter=\"*LargeMesh*\")\nset_tests_properties(GhostSyncMultiRank_4procs PROPERTIES PROCESSORS 4)\n")
endif()
if(CMAKE_CONFIGURATION_TYPES)
  file(WRITE "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}"
       "if(CTEST_CONFIGURATION_TYPE)\n"
       "  set(_ghost_sync_config \"\${CTEST_CONFIGURATION_TYPE}\")\n"
       "else()\n"
       "  set(_ghost_sync_config \"Debug\")\n"
       "endif()\n"
       "add_test(GhostSyncMultiRank_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${GHOST_SYNC_MULTI_RANK_EXECUTABLE}\")\n"
       "set_tests_properties(GhostSyncMultiRank_2procs PROPERTIES PROCESSORS 2)\n"
       "${_ghost_4proc_test}")
else()
  file(WRITE "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}"
       "add_test(GhostSyncMultiRank_2procs \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${GHOST_SYNC_MULTI_RANK_EXECUTABLE}\")\n"
       "set_tests_properties(GhostSyncMultiRank_2procs PROPERTIES PROCESSORS 2)\n"
       "${_ghost_4proc_test}")
endif()
set_property(DIRECTORY APPEND PROPERTY
             TEST_INCLUDE_FILES "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}")
