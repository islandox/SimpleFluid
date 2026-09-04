add_executable(testIF97Water testIF97Water.cc)
target_link_libraries(testIF97Water PRIVATE SimpleFluid::IF97 GTest::gtest_main)
simplefluid_discover_tests(testIF97Water PROPERTIES LABELS "material\\;if97")

add_executable(testIF97Material testIF97Material.cc)
target_link_libraries(testIF97Material PRIVATE
    SimpleFluid::IF97 SimpleFluid::Equations GTest::gtest_main)
simplefluid_configure_test_executable(testIF97Material)
simplefluid_discover_tests(testIF97Material PROPERTIES LABELS "material\\;if97")

simplefluid_add_mpi_test(
    NAME IF97Material_2procs
    TARGET testIF97Material
    PROCS 2
    ARGS --gtest_filter=IF97MaterialTest.DistributedUpdateSynchronizesAndRejectsCollectively
    LABELS "material;if97"
    TIMEOUT 60)
