add_executable(testBenchmarkSupport testBenchmarkSupport.cc)
target_link_libraries(
    testBenchmarkSupport
    SimpleFluid::BenchmarkSupport
    GTest::gtest_main
)
simplefluid_configure_test_executable(testBenchmarkSupport)
simplefluid_discover_tests(testBenchmarkSupport)
