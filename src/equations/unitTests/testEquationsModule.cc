#include <gtest/gtest.h>

#include <type_traits>

import SimpleFluid.Equations;

namespace
{

TEST(EquationsModuleTest, ExportsEquationApisAndLowerLayers)
{
    EXPECT_EQ(
        SimpleFluid::pressure_velocity_coupling_from_string("PIMPLE"),
        SimpleFluid::PressureVelocityCoupling::PIMPLE);
    EXPECT_DOUBLE_EQ(
        SimpleFluid::RadiolyticGasPhysics::ideal_gas_alpha_source(
            1.0, 1.0, 1.0, 2.0, 4.0, 5.0, 10.0, 100.0),
        4.0);

    static_assert(std::is_class_v<SimpleFluid::Equation<
                  SimpleFluid::ScalarCellFieldStored<>>>);
    static_assert(std::is_same_v<
                  SimpleFluid::VectorBoundaryConditionMap::mapped_type,
                  SimpleFluid::VectorBoundaryCondition>);
}

} // namespace
