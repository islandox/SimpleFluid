/**
 * @file testBoundaryCache.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Direct tests for finite-volume boundary-condition caching.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/BoundaryCache.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <type_traits>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;

static_assert(std::is_same_v<
              SimpleFluid::FVM::MeshBoundaryCache<
                  Pack, SimpleFluid::Mesh<Pack>>,
              SimpleFluid::FVM::BoundaryCache<Pack>>);
static_assert(std::is_same_v<
              SimpleFluid::FVM::MeshBoundaryCache<Pack, Handle>,
              SimpleFluid::FVM::FieldStoredBoundaryCache<Pack, Handle>>);

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build a runtime handle without materializing a legacy Mesh. */
SimpleFluid::SP<const Handle> make_cartesian_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

} // namespace

/** @brief Verifies the cache stores only configured Dirichlet boundary batches. */
TEST(BoundaryCacheTest, CachesOnlyConfiguredDirichletBatches)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    SimpleFluid::BoundaryConditionMap conditions;
    conditions["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 7.5};
    conditions["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 2.0};

    const auto cache =
        SimpleFluid::cache_boundary_conditions<Pack>(mesh, conditions);
    EXPECT_EQ(cache.mesh, mesh);

    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin")
        {
            ASSERT_TRUE(cache.value.contains(batch_id));
            EXPECT_EQ(cache.value.at(batch_id).size(),
                      batch.face_lids.size());
            for (const auto value : cache.value.at(batch_id))
            {
                EXPECT_DOUBLE_EQ(value, 7.5);
            }
        }
        else
        {
            EXPECT_FALSE(cache.value.contains(batch_id));
        }
    }
}

/** @brief Verifies boundary-condition caching rejects a null mesh. */
TEST(BoundaryCacheTest, RejectsNullMesh)
{
    EXPECT_THROW(
        SimpleFluid::cache_boundary_conditions<Pack>(
            {}, SimpleFluid::BoundaryConditionMap{}),
        std::invalid_argument);
}

/** @brief Cache values directly on a runtime mapped mesh and use stored-field fallbacks. */
TEST(BoundaryCacheTest, SupportsMeshHandleAndStoredCellCoefficients)
{
    auto mesh = make_cartesian_handle();
    ASSERT_FALSE(mesh->legacy_mesh());

    SimpleFluid::BoundaryConditionMap conditions;
    conditions["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 7.5};
    const auto cache =
        SimpleFluid::cache_boundary_conditions<Pack>(mesh, conditions);

    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(cache)>,
                  SimpleFluid::FieldStoredBoundaryCache<Pack, Handle>>);
    EXPECT_EQ(cache.mesh, mesh);
    EXPECT_NO_THROW(
        SimpleFluid::FVM::validate_boundary_coefficient_cache<Pack>(
            *mesh, &cache, "stored temperature transport"));

    SimpleFluid::ScalarCellFieldStored<Pack> conductivity(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("conductivity"),
        mesh,
        2.0);

    bool found_cached_batch = false;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto owner = mesh->owner_cell(batch.face_lids.front());
        const auto owner_value = conductivity.local_value(owner);
        const auto coefficient =
            SimpleFluid::FVM::boundary_coefficient<Pack>(
                &cache, batch_id, 0, owner_value);
        if (mesh->boundary_batch_name(batch_id) == "xmin")
        {
            found_cached_batch = true;
            EXPECT_DOUBLE_EQ(coefficient, 7.5);
        }
        else
        {
            EXPECT_DOUBLE_EQ(coefficient, owner_value);
        }
    }
    EXPECT_TRUE(found_cached_batch);
}

/** @brief Generic cache validation retains mesh identity and coefficient checks. */
TEST(BoundaryCacheTest, ValidatesMappedMeshCoefficientCache)
{
    auto mesh = make_cartesian_handle();
    SimpleFluid::FVM::MeshBoundaryCache<Pack, Handle> cache{{}, mesh};
    const auto& [batch_id, batch] = *mesh->boundary_batches().begin();
    cache.value[batch_id] =
        SimpleFluid::Arr<Pack::scalar_type>(batch.face_lids.size(), 1.0);

    EXPECT_NO_THROW(
        SimpleFluid::FVM::validate_boundary_coefficient_cache<Pack>(
            *mesh, &cache, "mapped transport"));

    cache.value[batch_id].front() = -1.0;
    EXPECT_THROW(
        SimpleFluid::FVM::validate_boundary_coefficient_cache<Pack>(
            *mesh, &cache, "mapped transport"),
        std::invalid_argument);

    auto other_mesh = make_cartesian_handle();
    cache.value[batch_id].front() = 1.0;
    cache.mesh = other_mesh;
    EXPECT_THROW(
        SimpleFluid::FVM::validate_boundary_coefficient_cache<Pack>(
            *mesh, &cache, "mapped transport"),
        std::invalid_argument);
}
