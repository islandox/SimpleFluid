#include <gtest/gtest.h>
#include "equations/turbulence/SSTSASSource.hh"
#include <fstream>
#include <filesystem>
#include <sstream>

using namespace SimpleFluid;

TEST(SSTSASSourceTest, IndependentPointwiseReference)
{
    // Decimal reference generator is deliberately independent of the C++ policy.
    const auto path = std::filesystem::path(__FILE__).parent_path() / "sst_sas_reference.csv";
    std::ifstream stream(path);
    ASSERT_TRUE(stream.good());
    std::string line;
    std::getline(stream, line);
    size_t rows = 0;
    while (std::getline(stream, line))
    {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream data(line);
        double f1, k, omega, s2, h, gk, gw, delta, dt;
        std::array<double, 11> reference;
        ASSERT_TRUE(bool(data >> f1 >> k >> omega >> s2 >> h >> gk >> gw >> delta >> dt));
        for (auto& v : reference) ASSERT_TRUE(bool(data >> v));
        SSTKOmegaEquation parent;
        const auto blend = parent.blended_coefficients(f1);
        auto result = SSTSASSource{}.evaluate({{k, omega}, s2, h, gk, gw, delta, dt,
            parent.coefficients().beta_star, blend.beta, blend.gamma, parent.coefficients().kappa});
        const auto values = result.values();
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_NEAR(values[i], reference[i], 2e-13 * std::max(1.0, std::abs(reference[i]))) << rows << ':' << i;
        ++rows;
    }
    EXPECT_EQ(rows, 6U);
}

TEST(SSTSASSourceTest, ZeroCurvatureStrainAndSmallBoundedStates)
{
    SSTSASInputs in{.state = {0.2, 2}, .strain_squared = 4};
    const auto affine = SSTSASSource{}.evaluate(in);
    EXPECT_DOUBLE_EQ(affine.Q_applied, 0);
    EXPECT_DOUBLE_EQ(affine.Lvk_flow, std::numeric_limits<double>::max());
    in.velocity_laplacian_magnitude = 100;
    in.strain_squared = 0;
    EXPECT_DOUBLE_EQ(SSTSASSource{}.evaluate(in).Q_applied, 0);
    in.state = {1e-12, 1e-12}; in.strain_squared = 1e-20;
    EXPECT_TRUE(std::isfinite(SSTSASSource{}.evaluate(in).Q_raw));
    in.velocity_laplacian_magnitude = 1e-16;
    in.state = {0.2, 2}; in.strain_squared = 4;
    EXPECT_LT(SSTSASSource{}.evaluate(in).Q_raw, 1e-30);
}

TEST(SSTSASSourceTest, DisableAndCapControlsAreNonsingular)
{
    SSTSASInputs in{.state = {1, 1}, .strain_squared = 100,
        .velocity_laplacian_magnitude = 100, .delta = 0.1, .time_step = 1};
    const auto capped = SSTSASSource{}.evaluate(in);
    EXPECT_DOUBLE_EQ(capped.Q_applied, 10);
    EXPECT_GT(capped.Q_raw, 10);
    auto options = SSTSASOptions{}; options.time_limiter = false;
    EXPECT_DOUBLE_EQ(SSTSASSource(options).evaluate(in).Q_applied, capped.Q_raw);
    options.enabled = false;
    EXPECT_DOUBLE_EQ(SSTSASSource(options).evaluate(in).Q_applied, 0);
}

TEST(SSTSASSourceTest, InvalidDataIsRejected)
{
    for (const auto value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        SSTSASOptions options; options.cs = value;
        EXPECT_THROW(SSTSASSource{options}, std::invalid_argument);
        EXPECT_THROW(SSTSASSource{}.filter_width(value), std::invalid_argument);
        SSTSASInputs in{.state = {1, 1}}; in.time_step = value;
        EXPECT_THROW(SSTSASSource{}.evaluate(in), std::invalid_argument);
        in.time_step = 1; in.delta = value;
        EXPECT_THROW(SSTSASSource{}.evaluate(in), std::invalid_argument);
    }
    SSTSASInputs in{.state = {1, 1}}; in.gamma = 1;
    EXPECT_THROW(SSTSASSource{}.evaluate(in), std::invalid_argument);
    in.gamma = 0.55; in.state.k = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(SSTSASSource{}.evaluate(in), std::invalid_argument);
    in.state.k = 1; in.velocity_laplacian_magnitude = std::numeric_limits<double>::infinity();
    EXPECT_THROW(SSTSASSource{}.evaluate(in), std::invalid_argument);
}

TEST(SSTSASSourceTest, UsesCoordinatedParentCoefficients)
{
    auto coefficients = SSTKOmegaEquation::Coefficients{};
    coefficients.kappa = 0.39; coefficients.beta_1 = 0.08;
    const SSTKOmegaEquation parent(coefficients);
    for (const double f1 : {0., 0.25, 1.})
    {
        const auto blend = parent.blended_coefficients(f1);
        const double gamma = blend.beta / coefficients.beta_star
            - (f1 * 0.5 + (1-f1)*0.856) * 0.39*0.39 / std::sqrt(0.09);
        EXPECT_NEAR(blend.gamma, gamma, 1e-15);
        const auto result = SSTSASSource{}.evaluate({{0.2, 2}, 4, 100, 0, 0, 0.1, 0.001,
            0.09, blend.beta, blend.gamma, 0.39});
        const auto expected_grid = 0.11*0.1*std::sqrt(0.39*3.51/(blend.beta/0.09-gamma));
        EXPECT_NEAR(result.Lvk_grid, expected_grid, 1e-15);
        EXPECT_GT(result.Q_applied, 0);
    }
}
