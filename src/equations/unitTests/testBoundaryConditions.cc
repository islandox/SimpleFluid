/**
 * @file testBoundaryConditions.cc
 * @brief Unit tests for boundary-condition data types.
 */

#include <gtest/gtest.h>

#include "equations/BoundaryConditions.hh"

namespace
{

using namespace SimpleFluid;

TEST(BoundaryConditionTest, ScalarConditionStoresTypeAndValue)
{
    const BoundaryCondition dirichlet{BoundaryConditionType::Dirichlet, 0.75};
    EXPECT_EQ(dirichlet.type, BoundaryConditionType::Dirichlet);
    EXPECT_DOUBLE_EQ(dirichlet.value, 0.75);

    const BoundaryCondition neumann{BoundaryConditionType::Neumann, 2.5};
    EXPECT_EQ(neumann.type, BoundaryConditionType::Neumann);
    EXPECT_DOUBLE_EQ(neumann.value, 2.5);
}

TEST(BoundaryConditionTest, VectorConditionStoresTypeAndValue)
{
    const VectorBoundaryCondition no_slip{BoundaryConditionType::NoSlip};
    EXPECT_EQ(no_slip.type, BoundaryConditionType::NoSlip);
    EXPECT_EQ(no_slip.value, (vec3<real_t>{0.0, 0.0, 0.0}));

    const VectorBoundaryCondition moving{
        BoundaryConditionType::Dirichlet, vec3<real_t>{1.0, 2.0, 3.0}};
    EXPECT_EQ(moving.type, BoundaryConditionType::Dirichlet);
    EXPECT_EQ(moving.value, (vec3<real_t>{1.0, 2.0, 3.0}));
}

TEST(BoundaryConditionTest, SetStoresSeparatePhysicsMaps)
{
    BoundaryConditionSet bcs;
    EXPECT_TRUE(bcs.temperature.empty());
    EXPECT_TRUE(bcs.velocity.empty());
    EXPECT_TRUE(bcs.pressure.empty());

    bcs.temperature["hot_wall"] = {BoundaryConditionType::Dirichlet, 1.0};
    bcs.velocity["moving_wall"] = {
        BoundaryConditionType::Dirichlet, vec3<real_t>{0.0, 1.0, 0.0}};
    bcs.pressure["outlet"] = {BoundaryConditionType::Neumann, 0.0};

    EXPECT_DOUBLE_EQ(bcs.temperature.at("hot_wall").value, 1.0);
    EXPECT_DOUBLE_EQ(bcs.velocity.at("moving_wall").value.y, 1.0);
    EXPECT_EQ(bcs.pressure.at("outlet").type, BoundaryConditionType::Neumann);
}

} // namespace
