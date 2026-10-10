#include "equations/ConvectionBoundary.hh"
#include <gtest/gtest.h>
#include <limits>

namespace
{
SimpleFluid::ConvectionBoundary natural_boundary()
{
    SimpleFluid::ConvectionBoundary boundary;
    boundary.correlation = SimpleFluid::ConvectionCorrelation::ChurchillChuVerticalPlate;
    boundary.characteristic_length = 1.0;
    boundary.fluid_thermal_conductivity = 0.03;
    boundary.kinematic_viscosity = 0.7e-5;
    boundary.thermal_diffusivity = 1e-5;
    boundary.thermal_expansion = 0.001;
    boundary.gravity = 7.0;
    return boundary;
}
} // namespace

TEST(ConvectionBoundaryTest, PublishedCorrelationReferencePoints)
{
    auto boundary = natural_boundary();
    const auto natural = SimpleFluid::evaluate_convection(boundary, 301.0);
    EXPECT_NEAR(natural.prandtl, 0.7, 1e-14);
    EXPECT_NEAR(natural.rayleigh, 1e8, 1e-7);
    EXPECT_NEAR(natural.nusselt, 60.94918389235828, 1e-10);
    EXPECT_NEAR(natural.heat_transfer_coefficient, 0.03 * natural.nusselt, 1e-14);
    EXPECT_DOUBLE_EQ(SimpleFluid::evaluate_convection(boundary, 299.0).nusselt, natural.nusselt);

    boundary.correlation = SimpleFluid::ConvectionCorrelation::ChurchillBernsteinCylinder;
    boundary.free_stream_speed = 0.07;
    const auto forced = SimpleFluid::evaluate_convection(boundary, 301.0);
    EXPECT_NEAR(forced.reynolds, 1e4, 1e-10);
    EXPECT_NEAR(forced.nusselt, 53.327788670209, 1e-10);
}

TEST(ConvectionBoundaryTest, ReconstructsWallThroughConductionResistance)
{
    const auto boundary = natural_boundary();
    const auto evaluated = SimpleFluid::evaluate_convection_at_cell(boundary, 400.0, 2.0);
    const auto h = evaluated.heat_transfer_coefficient;
    const auto surface = (2.0 * 400.0 + h * 300.0) / (2.0 + h);
    EXPECT_GT(surface, 300.0);
    EXPECT_LT(surface, 400.0);
    EXPECT_NEAR(h, SimpleFluid::evaluate_convection(boundary, surface).heat_transfer_coefficient, 1e-11);
    const auto heating = SimpleFluid::evaluate_convection_at_cell(boundary, 200.0, 2.0);
    EXPECT_NEAR(heating.heat_transfer_coefficient, h, 1e-11);
    EXPECT_DOUBLE_EQ(SimpleFluid::evaluate_convection_at_cell(boundary, 400.0, 0.0).rayleigh, 0.0);
}

TEST(ConvectionBoundaryTest, RejectsInvalidInputsAndUnsupportedRanges)
{
    auto boundary = natural_boundary();
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 20000.0), std::domain_error);
    boundary.kinematic_viscosity = 0.0;
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 310.0), std::invalid_argument);
    boundary = natural_boundary();
    boundary.correlation = SimpleFluid::ConvectionCorrelation::ChurchillBernsteinCylinder;
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 310.0), std::domain_error);
    boundary = {};
    boundary.heat_transfer_coefficient = -1.0;
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 310.0), std::invalid_argument);
    boundary.heat_transfer_coefficient = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 310.0), std::invalid_argument);
    boundary.heat_transfer_coefficient = 2.0;
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 0.0), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::evaluate_convection_at_cell(boundary, 310.0, -1.0), std::invalid_argument);
    boundary.correlation = static_cast<SimpleFluid::ConvectionCorrelation>(100);
    EXPECT_THROW(SimpleFluid::evaluate_convection(boundary, 310.0), std::invalid_argument);
}
