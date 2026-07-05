# Unit tests for io

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Mesh GTest::gtest_main)
  simplefluid_exe_reuse_precompiled_header(${TEST_NAME})
  simplefluid_discover_tests(${TEST_NAME})
endforeach()
