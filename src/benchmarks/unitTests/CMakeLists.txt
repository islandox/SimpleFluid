add_executable(testBenchmarkSupport testBenchmarkSupport.cc)
target_link_libraries(
    testBenchmarkSupport
    SimpleFluid::BenchmarkSupport
    GTest::gtest_main
)
simplefluid_configure_test_executable(testBenchmarkSupport)
if(SIMPLEFLUID_USE_BENCHMARK_MODULES)
  simplefluid_configure_module_consumers(
    testBenchmarkSupport testBenchmarkSupport.cc)
endif()
simplefluid_discover_tests(testBenchmarkSupport)
