/**
 * @file testFissionPowerSourceMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Distributed normalization tests for prescribed fission power.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/FissionPowerSource.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cmath>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using SourceType = SimpleFluid::FissionPowerSource<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Verifies global fission-power normalization across a distributed mesh. */
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

/** @brief Verifies rank-local invalid input is reduced before a source import. */
TEST(FissionPowerSourceMultiRankTest, RankLocalInvalidPowerDensityThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
    ASSERT_GT(mesh->num_owned_cells(), 0U);

    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);
    FieldType power_density(mesh, 1.0, "supplied_power_density");
    if (comm->getRank() == 0)
    {
        power_density.set_owned_value(0, -1.0);
    }

    EXPECT_ANY_THROW(
        source.initialize_from_power_density(power_density));
}

/** @brief Verifies rank-local Gaussian validation fails before normalization. */
TEST(FissionPowerSourceMultiRankTest, RankLocalInvalidGaussianThrowsCoherently)
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
    const SimpleFluid::vec3<double> width =
        comm->getRank() == 0
            ? SimpleFluid::vec3<double>{-0.2, 0.3, 0.4}
            : SimpleFluid::vec3<double>{0.2, 0.3, 0.4};

    EXPECT_ANY_THROW(
        source.initialize_gaussian(
            123.0,
            {0.5, 0.5, 0.5},
            width));
}

/** @brief Verifies rank-inconsistent valid profiles fail before branching. */
TEST(FissionPowerSourceMultiRankTest, MixedProfilesThrowCoherently)
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
    SimpleFluid::FissionPowerSourceOptions options;
    options.profile = comm->getRank() == 0
        ? SimpleFluid::FissionPowerProfile::Constant
        : SimpleFluid::FissionPowerProfile::Gaussian;

    EXPECT_ANY_THROW(source.configure(options));
}

/** @brief Verifies a distributed constant is identical on every rank. */
TEST(FissionPowerSourceMultiRankTest, RankVaryingConstantThrowsCoherently)
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
    const auto local_power_density =
        static_cast<double>(comm->getRank() + 1);

    EXPECT_ANY_THROW(
        source.initialize_constant(local_power_density));
}

} // namespace
