/**
 * @file testBoundaryConditions.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for boundary-condition data types.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
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

    const VectorBoundaryCondition slip{BoundaryConditionType::Slip};
    EXPECT_EQ(slip.type, BoundaryConditionType::Slip);
    EXPECT_EQ(slip.value, (vec3<real_t>{0.0, 0.0, 0.0}));

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
    EXPECT_TRUE(bcs.turbulence.turbulent_kinetic_energy.empty());
    EXPECT_TRUE(bcs.turbulence.dissipation_rate.empty());
    EXPECT_TRUE(bcs.turbulence.specific_dissipation_rate.empty());

    bcs.temperature["hot_wall"] = {BoundaryConditionType::Dirichlet, 1.0};
    bcs.velocity["moving_wall"] = {
        BoundaryConditionType::Dirichlet, vec3<real_t>{0.0, 1.0, 0.0}};
    bcs.pressure["outlet"] = {BoundaryConditionType::Neumann, 0.0};
    bcs.turbulence.turbulent_kinetic_energy["inlet"] = {
        BoundaryConditionType::Dirichlet, 0.25};
    bcs.turbulence.dissipation_rate["inlet"] = {
        BoundaryConditionType::Dirichlet, 0.5};
    bcs.turbulence.specific_dissipation_rate["wall"] = {
        BoundaryConditionType::Neumann, 0.0};

    EXPECT_DOUBLE_EQ(bcs.temperature.at("hot_wall").value, 1.0);
    EXPECT_DOUBLE_EQ(bcs.velocity.at("moving_wall").value.y, 1.0);
    EXPECT_EQ(bcs.pressure.at("outlet").type, BoundaryConditionType::Neumann);
    EXPECT_DOUBLE_EQ(
        bcs.turbulence.turbulent_kinetic_energy.at("inlet").value,
        0.25);
    EXPECT_DOUBLE_EQ(
        bcs.turbulence.dissipation_rate.at("inlet").value,
        0.5);
    EXPECT_EQ(
        bcs.turbulence.specific_dissipation_rate.at("wall").type,
        BoundaryConditionType::Neumann);
}

TEST(BoundaryConditionTest, TurbulenceScalarMapsUseScalarDefaults)
{
    TurbulenceBoundaryConditionSet turbulence;

    const auto& kinetic_energy =
        turbulence.turbulent_kinetic_energy["unspecified"];
    const auto& dissipation =
        turbulence.dissipation_rate["unspecified"];
    const auto& specific_dissipation =
        turbulence.specific_dissipation_rate["unspecified"];

    for (const auto* condition : {
             &kinetic_energy,
             &dissipation,
             &specific_dissipation})
    {
        EXPECT_EQ(condition->type, BoundaryConditionType::Neumann);
        EXPECT_DOUBLE_EQ(condition->value, 0.0);
        EXPECT_DOUBLE_EQ(condition->robin_coefficient, 0.0);
    }
}

} // namespace
