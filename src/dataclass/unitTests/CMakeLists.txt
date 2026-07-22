# Unit tests for dataclass

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
if(NOT SIMPLEFLUID_ENABLE_CXX_MODULES)
  list(REMOVE_ITEM TEST_SOURCES
       "${CMAKE_CURRENT_SOURCE_DIR}/testCoreModule.cc")
endif()
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Core GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  if(SIMPLEFLUID_ENABLE_CXX_MODULES AND TEST_NAME STREQUAL "testCoreModule")
    simplefluid_configure_module_test(${TEST_NAME} ${TEST_SRC})
  endif()
  simplefluid_discover_tests(${TEST_NAME})
endforeach()
