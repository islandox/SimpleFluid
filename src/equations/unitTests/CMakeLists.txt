# Unit tests for equation classes

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Equations GTest::gtest_main)
  simplefluid_reuse_precompiled_header(${TEST_NAME})
  if(TEST_NAME STREQUAL "testAnalyticalConvergence")
    gtest_discover_tests(${TEST_NAME} PROPERTIES LABELS "slow\\;analytical")
  else()
    gtest_discover_tests(${TEST_NAME})
  endif()
endforeach()
