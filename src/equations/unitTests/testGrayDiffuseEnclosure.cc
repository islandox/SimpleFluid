#include "equations/GrayDiffuseEnclosure.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace SimpleFluid;

TEST(GrayDiffuseEnclosureTest, AnalyticalTwoGraySurfacesAndRepeatedRightHandSides)
{
    const GrayDiffuseEnclosure enclosure({2.0, 2.0}, {0.0, 2.0, 2.0, 0.0}, {0.4, 0.7});
    for (const auto emission : {100.0, 250.0, 1.0e-8})
    {
        const std::array<real_t, 2> delta{emission, 0.0};
        const auto result = enclosure.evaluate(delta, 300.0);
        const auto expected = 2.0 * emission / (1.0 / 0.4 + 1.0 / 0.7 - 1.0);
        EXPECT_NEAR(result.outward_heat_rate[0], expected, std::abs(expected) * 2.0e-14);
        EXPECT_DOUBLE_EQ(result.outward_heat_rate[0], -result.outward_heat_rate[1]);
        EXPECT_NEAR(result.irradiation_delta[0], result.radiosity_delta[1], 1.0e-14);
        EXPECT_LE(result.equation_residual, 1.0e-12);
    }
}

TEST(GrayDiffuseEnclosureTest, CommonTemperatureHasExactlyZeroHeat)
{
    const GrayDiffuseEnclosure enclosure({1.0, 2.0}, {0.0, 1.0, 1.0, 1.0}, {0.03, 0.8});
    const auto result = enclosure.evaluate(std::array<real_t, 2>{0.0, 0.0}, 459.3);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_DOUBLE_EQ(result.outward_heat_rate[i], 0.0);
        EXPECT_DOUBLE_EQ(result.radiosity[i], 459.3);
        EXPECT_DOUBLE_EQ(result.irradiation[i], 459.3);
    }
    EXPECT_DOUBLE_EQ(result.equation_residual, 0.0);
}

TEST(GrayDiffuseEnclosureTest, HeterogeneousPatchUsesAreaWeightedFourthMoment)
{
    const GrayDiffuseEnclosure enclosure({4.0, 4.0}, {0.0, 4.0, 4.0, 0.0}, {1.0, 1.0});
    const auto emission = (GrayDiffuseEnclosure::blackbody_delta(300.0, 350.0) +
        3.0 * GrayDiffuseEnclosure::blackbody_delta(400.0, 350.0)) / 4.0;
    const auto result = enclosure.evaluate(std::array<real_t, 2>{emission, 0.0}, 5.670374419e-8 * std::pow(350.0, 4));
    EXPECT_NEAR(result.outward_heat_rate[0], 4.0 * emission, 1.0e-11);
    EXPECT_GT(std::abs(emission - GrayDiffuseEnclosure::blackbody_delta(375.0, 350.0)), 10.0);
    const auto t = std::nextafter(350.0, 400.0);
    EXPECT_GT(GrayDiffuseEnclosure::blackbody_delta(t, 350.0), 0.0);
}

TEST(GrayDiffuseEnclosureTest, RejectsInvalidGeometryAndEmission)
{
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {0., 1., 0.9, 0.}, {1., 1.})), std::invalid_argument);
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {0., 0.9, 0.9, 0.}, {1., 1.})), std::invalid_argument);
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {1., 1.e-30, 0., 1.}, {1., 1.})), std::invalid_argument);
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {0., 1., 1., 0.}, {0., 1.})), std::invalid_argument);
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {0., 1., 1., 0.}, {1.1, 1.})), std::invalid_argument);
    const GrayDiffuseEnclosure enclosure({1.}, {1.}, {1.});
    EXPECT_THROW(enclosure.evaluate(std::array<real_t, 1>{-1.}), std::invalid_argument);
    EXPECT_THROW(enclosure.evaluate(std::array<real_t, 1>{std::numeric_limits<real_t>::quiet_NaN()}), std::invalid_argument);
}

TEST(GrayDiffuseEnclosureTest, ReflectiveConditionAndResidualAreQualified)
{
    const GrayDiffuseEnclosure enclosure({1., 1.}, {0., 1., 1., 0.}, {1.e-6, 1.e-6});
    EXPECT_GT(enclosure.reciprocal_condition_number(), 4.9e-7);
    EXPECT_LT(enclosure.reciprocal_condition_number(), 5.1e-7);
    const auto result = enclosure.evaluate(std::array<real_t, 2>{100., 0.});
    EXPECT_NEAR(result.outward_heat_rate[0], 100. / (2.e6 - 1.), 1.e-13);
    EXPECT_LE(result.equation_residual, 1.e-12);
    EXPECT_THROW((GrayDiffuseEnclosure({1., 1.}, {0., 1., 1., 0.}, {1.e-16, 1.e-16})), std::domain_error);
}
