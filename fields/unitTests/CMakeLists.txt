
# Unit tests for fields

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testGhostSyncMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid GTest::gtest_main)
  simplefluid_reuse_precompiled_header(${TEST_NAME})
  gtest_discover_tests(${TEST_NAME})
endforeach()

add_executable(testGhostSyncMultiRank testGhostSyncMultiRank.cc)
target_link_libraries(testGhostSyncMultiRank SimpleFluid GTest::gtest_main)
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
if(CMAKE_CONFIGURATION_TYPES)
  file(WRITE "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}"
       "if(CTEST_CONFIGURATION_TYPE)\n"
       "  set(_ghost_sync_config \"\${CTEST_CONFIGURATION_TYPE}\")\n"
       "else()\n"
       "  set(_ghost_sync_config \"Debug\")\n"
       "endif()\n"
       "add_test(GhostSyncMultiRank \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${GHOST_SYNC_MULTI_RANK_EXECUTABLE}\")\n"
       "set_tests_properties(GhostSyncMultiRank PROPERTIES PROCESSORS 2)\n")
else()
  file(WRITE "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}"
       "add_test(GhostSyncMultiRank \"${MPIEXEC_EXECUTABLE}\" \"${MPIEXEC_NUMPROC_FLAG}\" \"2\" \"${GHOST_SYNC_MULTI_RANK_EXECUTABLE}\")\n"
       "set_tests_properties(GhostSyncMultiRank PROPERTIES PROCESSORS 2)\n")
endif()
set_property(DIRECTORY APPEND PROPERTY
             TEST_INCLUDE_FILES "${GHOST_SYNC_MULTI_RANK_CTEST_FILE}")
