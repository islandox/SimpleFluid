/**
 * @file testBoussinesqModelMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI regressions for rank-coherent Boussinesq callback validation.
 * @version 0.1
 * @date 2026-07-22
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoussinesqModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <limits>
#include <stdexcept>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_distributed_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
}

void require_multiple_ranks(const MeshType& mesh)
{
    if (mesh.owned_cell_map()->getComm()->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
}

SimpleFluid::MaterialPropertyFields<Pack> make_material(
    SimpleFluid::SP<const MeshType> mesh)
{
    const SimpleFluid::TimeStepperOptions time_options;
    const auto options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    return {std::move(mesh), options, time_options};
}

/** @brief Verifies a rank-local initializer failure is reduced before import. */
TEST(BoussinesqModelMultiRankTest, RankLocalInitializerFailureThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    auto& source = registry.add("heat");

    EXPECT_ANY_THROW(
        source.initialize(
            [rank](const auto& centroid)
            {
                if (rank == 0)
                {
                    throw std::runtime_error(
                        "rank-local initializer failure");
                }
                return centroid.x;
            }));
}

/** @brief Verifies a rank-local source callback failure is reduced before import. */
TEST(BoussinesqModelMultiRankTest, RankLocalSourceCallbackFailureThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    auto& source = registry.add("heat");
    source.set_updater(
        [rank](const auto&, auto& field)
        {
            if (rank == 0)
            {
                throw std::runtime_error(
                    "rank-local source callback failure");
            }
            field.set_owned_value(0, 1.0);
        });

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(source.update(context));
}

/** @brief Verifies rank-local invalid source output is reduced before import. */
TEST(BoussinesqModelMultiRankTest, RankLocalInvalidSourceOutputThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    auto& source = registry.add("heat");
    source.set_updater(
        [rank](const auto&, auto& field)
        {
            const auto value =
                rank == 0
                    ? std::numeric_limits<double>::quiet_NaN()
                    : 1.0;
            field.set_owned_value(0, value);
        });

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(source.update(context));
}

/** @brief Verifies multiple dynamic source failures share one coherent batch. */
TEST(BoussinesqModelMultiRankTest, RegistryBatchPropagatesRankLocalFailure)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    auto& first = registry.add("first");
    auto& second = registry.add("second");
    first.set_updater(
        [](const auto&, auto& field)
        {
            field.set_owned_value(0, 1.0);
        });
    second.set_updater(
        [rank](const auto&, auto& field)
        {
            if (rank == 0)
            {
                throw std::runtime_error(
                    "rank-local batched source failure");
            }
            field.set_owned_value(0, 2.0);
        });

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(registry.update(context));
}

/** @brief Verifies rank-varying activation fails before any source import. */
TEST(BoussinesqModelMultiRankTest, RegistryRejectsRankVaryingActivation)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    auto& source = registry.add("heat");
    source.set_updater(
        [](const auto&, auto&) {});
    source.set_enabled(rank != 0);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(registry.update(context));
}

/** @brief Verifies empty/nonempty registries fail before a sized reduction. */
TEST(BoussinesqModelMultiRankTest,
     RegistryRejectsRankVaryingEmptyStateCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    registry.add("heat");
    if (rank == 0)
    {
        ASSERT_TRUE(registry.remove("heat"));
    }

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_THROW(registry.update(context), std::invalid_argument);
}

/** @brief Verifies unequal nonzero source counts fail collectively. */
TEST(BoussinesqModelMultiRankTest,
     RegistryRejectsDifferentSourceCountsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    registry.add("first");
    registry.add("second");

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_NO_THROW(registry.update(context));
    if (rank == 0)
    {
        ASSERT_TRUE(registry.remove("second"));
    }
    EXPECT_THROW(registry.update(context), std::invalid_argument);
}

/** @brief Verifies equal-sized but differently named schemas fail collectively. */
TEST(BoussinesqModelMultiRankTest,
     RegistryRejectsDifferentSourceNamesCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    registry.add("first");
    registry.add("other");
    ASSERT_TRUE(registry.remove(rank == 0 ? "other" : "first"));

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_THROW(registry.update(context), std::invalid_argument);
}

/** @brief Verifies a rank-local material callback failure is reduced before imports. */
TEST(BoussinesqModelMultiRankTest, RankLocalMaterialCallbackFailureThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    auto material = make_material(mesh);
    material.set_updater(
        [rank](const auto&, auto& fields)
        {
            if (rank == 0)
            {
                throw std::runtime_error(
                    "rank-local material callback failure");
            }
            fields.density.set_owned_value(0, 2.0);
        });

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(material.update(context));
}

/** @brief Verifies rank-local invalid material output is reduced before imports. */
TEST(BoussinesqModelMultiRankTest, RankLocalInvalidMaterialOutputThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    auto material = make_material(mesh);
    material.set_updater(
        [rank](const auto&, auto& fields)
        {
            fields.density.set_owned_value(
                0, rank == 0 ? -1.0 : 2.0);
        });

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    EXPECT_ANY_THROW(material.update(context));
}

} // namespace
