#include "equations/GrayDiffuseEnclosure.hh"

#include <gtest/gtest.h>
#include <Teuchos_LAPACK.hpp>

#include <array>
#include <algorithm>
#include <vector>
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

namespace
{
// The previous constructor metric, evaluated independently for gate parity.
real_t inverse_column_condition_reference(const std::vector<real_t>& areas,
    const std::vector<real_t>& exchange, const std::vector<real_t>& emissivity)
{
    const auto count = areas.size();
    std::vector<real_t> lu(count * count);
    real_t norm = 0.0;
    for (size_t j = 0; j < count; ++j)
    {
        real_t column_sum = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const auto symmetric_exchange = 0.5 * exchange[i * count + j] + 0.5 * exchange[j * count + i];
            const auto value = (i == j ? 1.0 : 0.0) - (1.0 - emissivity[i]) * (symmetric_exchange / areas[i]);
            lu[j * count + i] = value;
            column_sum += std::abs(value);
        }
        norm = std::max(norm, column_sum);
    }
    const int n = static_cast<int>(count);
    Teuchos::LAPACK<int, real_t> lapack;
    std::vector<int> pivots(count);
    int info = 0;
    lapack.GETRF(n, n, lu.data(), n, pivots.data(), &info);
    if (info != 0) return 0.0;
    std::vector<real_t> column(count);
    real_t inverse_norm = 0.0;
    for (size_t j = 0; j < count; ++j)
    {
        std::fill(column.begin(), column.end(), 0.0);
        column[j] = 1.0;
        lapack.GETRS('N', n, 1, lu.data(), n, pivots.data(), column.data(), n, &info);
        if (info != 0) throw std::runtime_error("Condition reference solve failed.");
        real_t sum = 0.0;
        for (const auto value : column) sum += std::abs(value);
        inverse_norm = std::max(inverse_norm, sum);
    }
    return 1.0 / (norm * inverse_norm);
}
} // namespace

TEST(GrayDiffuseEnclosureTest, CachedLUConditionEstimateMatchesTwoSurfaceAnalyticCondition)
{
    constexpr real_t reflectivity_one = 0.2, reflectivity_two = 0.05;
    // A=[[1,-r1],[-r2,1]], ||A||1=1+max(r1,r2).
    // The inverse has the same largest column sum divided by its determinant.
    const auto exact = (1.0 - reflectivity_one * reflectivity_two) /
        std::pow(1.0 + std::max(reflectivity_one, reflectivity_two), 2);
    const GrayDiffuseEnclosure enclosure({1.0, 1.0}, {0.0, 1.0, 1.0, 0.0}, {0.8, 0.95});
    EXPECT_NEAR(enclosure.reciprocal_condition_number(), exact, 2e-14);
    const auto result = enclosure.evaluate(std::array<real_t, 2>{100.0, 0.0});
    EXPECT_NEAR(result.outward_heat_rate[0], 100.0 / (1.0 / 0.8 + 1.0 / 0.95 - 1.0), 1e-12);
    EXPECT_DOUBLE_EQ(result.outward_heat_rate[0], -result.outward_heat_rate[1]);
}

TEST(GrayDiffuseEnclosureTest, NearReflectingGateMatchesPreviousInverseColumnMetric)
{
    const auto threshold = 64.0 * std::numeric_limits<real_t>::epsilon();
    const std::vector<real_t> areas{1.0, 1.0}, exchange{0.0, 1.0, 1.0, 0.0};
    for (const auto emissivity : {1e-16, 2e-14, 4e-14, 1e-13, 1e-6, 0.8})
    {
        SCOPED_TRACE(emissivity);
        const std::vector<real_t> emissivities{emissivity, emissivity};
        const auto previous = inverse_column_condition_reference(areas, exchange, emissivities);
        if (!std::isfinite(previous) || previous <= threshold)
            EXPECT_THROW((GrayDiffuseEnclosure(areas, exchange, emissivities)), std::domain_error);
        else
        {
            const GrayDiffuseEnclosure enclosure(areas, exchange, emissivities);
            EXPECT_NEAR(enclosure.reciprocal_condition_number(), previous, previous * 1e-12);
        }
    }
}

TEST(GrayDiffuseEnclosureTest, RowClosurePerturbationAndReducibleCasesRetainExactGate)
{
    struct Fixture { std::vector<real_t> areas, exchange, emissivities; };
    const std::vector<Fixture> fixtures{
        {{1.0, 1.0}, {0.0, 1.0 + 5e-11, 1.0 + 5e-11, 0.0}, {1e-12, 1e-12}},
        {{1.0, 1.0}, {0.0, 1.0 - 5e-11, 1.0 - 5e-11, 0.0}, {1e-14, 1e-14}},
        {{1.0, 1.0, 1.0, 1.0},
            {0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}, {1e-16, 1e-16, 0.8, 0.8}},
        {{1.0, 1.0, 1.0, 1.0},
            {0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}, {1e-13, 1e-13, 0.8, 0.8}}};
    const auto threshold = 64.0 * std::numeric_limits<real_t>::epsilon();
    for (size_t index = 0; index < fixtures.size(); ++index)
    {
        SCOPED_TRACE(index);
        const auto& fixture = fixtures[index];
        const auto previous = inverse_column_condition_reference(fixture.areas, fixture.exchange, fixture.emissivities);
        if (!std::isfinite(previous) || previous <= threshold)
            EXPECT_THROW((GrayDiffuseEnclosure(fixture.areas, fixture.exchange, fixture.emissivities)), std::domain_error);
        else
        {
            const GrayDiffuseEnclosure enclosure(fixture.areas, fixture.exchange, fixture.emissivities);
            EXPECT_NEAR(enclosure.reciprocal_condition_number(), previous, previous * 1e-12);
        }
    }
}
