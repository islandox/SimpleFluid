#include <gtest/gtest.h>

#include <type_traits>

import SimpleFluid.Solvers;

namespace
{

TEST(SolversModuleTest, ExportsSolverApisAndProblem)
{
    static_assert(std::is_class_v<SimpleFluid::Problem<>>);
    static_assert(std::is_class_v<SimpleFluid::CoupledPressureVelocitySolver<>>);
    static_assert(std::is_class_v<SimpleFluid::FluidSolver<>>);
    static_assert(std::is_class_v<SimpleFluid::BoussinesqSolver<>>);
    static_assert(std::is_class_v<SimpleFluid::TemperatureDiffusionEquation<>>);
}

} // namespace
