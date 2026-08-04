#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#endif

import SimpleFluid.Problems;

namespace
{

TEST(ProblemsModuleTest, ExportsProblemAndItsConfigurationTypes)
{
    static_assert(std::is_class_v<SimpleFluid::Problem<>>);
    static_assert(std::is_class_v<SimpleFluid::BoundaryConditionSet>);
    static_assert(std::is_class_v<SimpleFluid::TimeStepperOptions>);
    static_assert(std::is_class_v<SimpleFluid::LinearSolverOptions>);
}

} // namespace
