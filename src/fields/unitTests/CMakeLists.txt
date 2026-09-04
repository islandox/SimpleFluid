
# Unit tests for fields

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testGhostSyncMultiRank\\.cc$")

add_executable(testFields ${TEST_SOURCES})
target_link_libraries(testFields SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testFields)
simplefluid_discover_tests(testFields)

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
