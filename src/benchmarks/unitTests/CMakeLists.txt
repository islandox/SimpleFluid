add_executable(testBenchmarkSupport testBenchmarkSupport.cc)
target_link_libraries(
    testBenchmarkSupport
    SimpleFluid::BenchmarkSupport
    GTest::gtest_main
)
simplefluid_exe_reuse_precompiled_header(testBenchmarkSupport)
simplefluid_discover_tests(testBenchmarkSupport)
