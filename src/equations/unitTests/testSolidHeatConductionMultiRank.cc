/**
 * @file testSolidHeatConductionMultiRank.cc
 * @brief MPI regression for solid conduction across a partition face.
 */

#include <gtest/gtest.h>

#include "equations/SolidHeatConductionEquation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <memory>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using ParentMesh = SimpleFluid::MeshHandle<Pack>;
using SolidMesh = SimpleFluid::SolidSubdomain<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, SolidMesh>;
using Material = SimpleFluid::MaterialPropertyFields<Pack, SolidMesh>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<const SolidMesh> make_partitioned_solid()
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<ParentMesh>(std::move(geometry));
    return std::make_shared<SolidMesh>(std::move(parent), [](Pack::global_ordinal_type, const SolidMesh::Vec3& centroid)
        { return centroid.x >= 1.0 && centroid.x < 3.0; });
}

Material make_material(SimpleFluid::SP<const SolidMesh> mesh, double conductivity = 1.0)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1.0;
    options.density = 1.0;
    options.specific_heat_capacity = 1.0;
    options.thermal_conductivity = conductivity;
    return {std::move(mesh), options, time_options};
}

double global_energy(const Field& temperature)
{
    const auto& mesh = temperature.mesh();
    double local = 0.0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        local += mesh.cell_volume(cell_lid) * temperature.value(cell_lid);
    }
    double global = 0.0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

} // namespace

/** @brief Heat crosses the selected-cell partition face without losing energy. */
TEST(SolidHeatConductionMultiRankTest, ConservesEnergyAcrossPartitionedSolidCells)
{
    SKIP_SINGLE_RANK(ConservesEnergyAcrossPartitionedSolidCells);
    const auto mesh = make_partitioned_solid();
    const auto comm = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(comm->getSize(), 2);
    ASSERT_EQ(mesh->num_owned_cells(), 1U);

    Field temperature(mesh, "solid_temperature");
    const auto center = mesh->cell_centroid(0);
    temperature.set_owned_value(0, center.x < 2.0 ? 400.0 : 300.0);
    temperature.sync_ghosts();
    const auto energy_before = global_energy(temperature);

    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::SolidHeatConductionEquation<Pack> equation(mesh, boundaries);
    const auto statistics =
        equation.advance(temperature, 0.1, material, temperature, SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(global_energy(temperature), energy_before, 1.0e-10);

    const auto local_temperature = temperature.value(0);
    double minimum_temperature = 0.0;
    double maximum_temperature = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_temperature, &minimum_temperature);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_temperature, &maximum_temperature);
    EXPECT_GT(minimum_temperature, 300.0);
    EXPECT_LT(maximum_temperature, 400.0);
}

/** @brief Ranks with no selected cells still participate in the solid solve. */
TEST(SolidHeatConductionMultiRankTest, SupportsRanksWithoutSolidCells)
{
    SKIP_SINGLE_RANK(SupportsRanksWithoutSolidCells);
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<ParentMesh>(std::move(geometry));
    auto mesh = std::make_shared<SolidMesh>(
        std::move(parent), [](Pack::global_ordinal_type, const SolidMesh::Vec3& centroid) { return centroid.x < 1.0; });
    const auto comm = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(comm->getSize(), 2);
    EXPECT_EQ(mesh->num_owned_cells(), comm->getRank() == 0 ? 1U : 0U);

    Field temperature(mesh, 300.0, "solid_temperature");
    auto material = make_material(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::SolidHeatConductionEquation<Pack> equation(mesh, boundaries);
    const auto first_statistics =
        equation.advance(temperature, 0.5, material, temperature, [](Pack::local_ordinal_type) { return 16.0; });
    const auto second_statistics =
        equation.advance(temperature, 0.5, material, temperature, [](Pack::local_ordinal_type) { return 16.0; });
    EXPECT_TRUE(first_statistics.converged);
    EXPECT_TRUE(second_statistics.converged);

    const auto local_temperature = mesh->num_owned_cells() == 0 ? 0.0 : temperature.value(0);
    double global_temperature = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_temperature, &global_temperature);
    EXPECT_NEAR(global_temperature, 316.0, 1.0e-12);
}

/** @brief A rank-local missing source fails coherently before assembly. */
TEST(SolidHeatConductionMultiRankTest, RejectsRankLocalMissingSource)
{
    SKIP_SINGLE_RANK(RejectsRankLocalMissingSource);
    const auto mesh = make_partitioned_solid();
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    Field temperature(mesh, 300.0, "solid_temperature");
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::SolidHeatConductionEquation<Pack> equation(mesh, boundaries);
    SimpleFluid::SolidHeatConductionEquation<Pack>::source_type source;
    if (rank == 0)
    {
        source = [](Pack::local_ordinal_type) { return 0.0; };
    }

    EXPECT_ANY_THROW(equation.advance(temperature, 0.1, material, temperature, source));
}

/** @brief Conductivity-override selection must agree on every rank. */
TEST(SolidHeatConductionMultiRankTest, RejectsRankInconsistentConductivityOverride)
{
    SKIP_SINGLE_RANK(RejectsRankInconsistentConductivityOverride);
    const auto mesh = make_partitioned_solid();
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    Field temperature(mesh, 300.0, "solid_temperature");
    Field conductivity_override(mesh, 2.0, "conductivity_override");
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::SolidHeatConductionEquation<Pack> equation(mesh, boundaries);
    const Field* override_on_this_rank = rank == 0 ? &conductivity_override : nullptr;

    EXPECT_THROW(equation.advance(temperature, 0.1, material, temperature,
                     SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, {}, override_on_this_rank),
        std::invalid_argument);
}

/** @brief Boundary-conductivity cache selection must agree on every rank. */
TEST(SolidHeatConductionMultiRankTest, RejectsRankInconsistentBoundaryConductivityCache)
{
    SKIP_SINGLE_RANK(RejectsRankInconsistentBoundaryConductivityCache);
    const auto mesh = make_partitioned_solid();
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    Field temperature(mesh, 300.0, "solid_temperature");
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundaries;
    SimpleFluid::SolidHeatConductionEquation<Pack> equation(mesh, boundaries);
    SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, SolidMesh> cache{{}, mesh};
    const auto* cache_on_this_rank = rank == 0 ? &cache : nullptr;

    EXPECT_THROW(equation.advance(temperature, 0.1, material, temperature,
                     SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, {}, nullptr, cache_on_this_rank),
        std::invalid_argument);
}

/** @brief A subdomain rejects partition faces missing from parent overlap. */
TEST(SolidHeatConductionMultiRankTest, RejectsInsufficientParentOverlap)
{
    SKIP_SINGLE_RANK(RejectsInsufficientParentOverlap);
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<ParentMesh>(std::move(geometry), ParentMesh::DistributionOptions{.ghost_layers = 0});

    EXPECT_THROW(static_cast<void>(std::make_shared<SolidMesh>(parent)), std::invalid_argument);
}

/** @brief Synthetic interface names are part of the distributed schema. */
TEST(SolidHeatConductionMultiRankTest, RejectsRankInconsistentInterfaceName)
{
    SKIP_SINGLE_RANK(RejectsRankInconsistentInterfaceName);
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<ParentMesh>(std::move(geometry));
    const auto rank = parent->owned_cell_map()->getComm()->getRank();

    EXPECT_THROW(static_cast<void>(std::make_shared<SolidMesh>(
                     std::move(parent), [](Pack::global_ordinal_type, const SolidMesh::Vec3& centroid)
                     { return centroid.x >= 1.0 && centroid.x < 3.0; }, rank == 0 ? "interface_a" : "interface_b")),
        std::invalid_argument);
}
