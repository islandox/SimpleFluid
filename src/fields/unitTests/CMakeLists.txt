
# Unit tests for fields

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/testGhostSyncMultiRank\\.cc$")
set(TRILINOS_MODULE_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/testKokkosModule.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/testSTKModule.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/testTeuchosModule.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/testTpetraModule.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/testZoltan2Module.cc")
list(REMOVE_ITEM TEST_SOURCES ${TRILINOS_MODULE_TEST_SOURCES})
if(NOT SIMPLEFLUID_ENABLE_CXX_MODULES)
  list(REMOVE_ITEM TEST_SOURCES
       "${CMAKE_CURRENT_SOURCE_DIR}/testFieldsModule.cc")
endif()

add_executable(testFields ${TEST_SOURCES})
target_link_libraries(testFields SimpleFluid::Fields GTest::gtest_main)
simplefluid_configure_test_executable(testFields)
if(SIMPLEFLUID_ENABLE_CXX_MODULES)
  simplefluid_configure_module_test(testFields testFieldsModule.cc)
endif()
simplefluid_discover_tests(testFields)

if(SIMPLEFLUID_ENABLE_CXX_MODULES)
  foreach(MODULE_TEST_SRC IN LISTS TRILINOS_MODULE_TEST_SOURCES)
    get_filename_component(MODULE_TEST_NAME "${MODULE_TEST_SRC}" NAME_WE)
    add_executable(${MODULE_TEST_NAME} "${MODULE_TEST_SRC}")
    target_link_libraries(${MODULE_TEST_NAME}
        SimpleFluid::TrilinosModules)
    simplefluid_configure_test_executable(${MODULE_TEST_NAME})
    set_target_properties(${MODULE_TEST_NAME} PROPERTIES
        CXX_SCAN_FOR_MODULES ON)
    set_source_files_properties(${MODULE_TEST_SRC} PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON)
    add_test(NAME ${MODULE_TEST_NAME}.ImportsPackageView
        COMMAND ${MODULE_TEST_NAME})
  endforeach()
endif()

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
