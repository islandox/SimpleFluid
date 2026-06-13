#include <gtest/gtest.h>

#include "equations/FissionPowerSource.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cmath>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using SourceType = SimpleFluid::FissionPowerSource<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

TEST(FissionPowerSourceMultiRankTest, NormalizesAcrossDistributedMesh)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);
    source.initialize_gaussian(
        123.0,
        {0.5, 0.5, 0.5},
        {0.2, 0.3, 0.4});

    EXPECT_NEAR(source.integrated_power(), 123.0, 1.0e-10);
    for (size_t local = 0;
         local < mesh->num_local_cells();
         ++local)
    {
        const auto lid =
            static_cast<Pack::local_ordinal_type>(local);
        EXPECT_TRUE(std::isfinite(source.field().local_value(lid)));
        EXPECT_GE(source.field().local_value(lid), 0.0);
    }
}

} // namespace
