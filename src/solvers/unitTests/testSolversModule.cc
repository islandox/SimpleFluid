#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#endif

import SimpleFluid.Solvers;

namespace
{

TEST(SolversModuleTest, ExportsSolverApisAndProblem)
{
    static_assert(std::is_class_v<SimpleFluid::Problem<>>);
    static_assert(std::is_class_v<SimpleFluid::CoupledPressureVelocitySolver<>>);
    static_assert(std::is_class_v<SimpleFluid::FluidSolver<>>);
    static_assert(std::is_class_v<SimpleFluid::BoussinesqSolver<>>);
    static_assert(std::is_class_v<
                  SimpleFluid::AdaptiveSteadyStateController>);
    static_assert(std::is_class_v<SimpleFluid::SteadyStateFieldMonitor<>>);
    static_assert(SimpleFluid::CoupledRebuildPolicy::Always
                  != SimpleFluid::CoupledRebuildPolicy::OnOperatorGraphChange);
    static_assert(std::is_class_v<SimpleFluid::TemperatureDiffusionEquation<>>);
}

} // namespace
