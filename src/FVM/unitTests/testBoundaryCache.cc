/**
 * @file testBoundaryCache.cc
 * @brief Direct tests for finite-volume boundary-condition caching.
 */

#include <gtest/gtest.h>

#include "FVM/BoundaryCache.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

TEST(BoundaryCacheTest, CachesOnlyConfiguredDirichletPatches)
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

TEST(BoundaryCacheTest, RejectsNullMesh)
{
    EXPECT_THROW(
        SimpleFluid::cache_boundary_conditions<Pack>(
            {}, SimpleFluid::BoundaryConditionMap{}),
        std::invalid_argument);
}
