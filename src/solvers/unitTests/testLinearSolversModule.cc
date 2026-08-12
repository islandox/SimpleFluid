#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#endif

import SimpleFluid.LinearSolvers;

namespace
{

TEST(LinearSolversModuleTest, ExportsBelosWrapperAndFieldTypes)
{
    static_assert(std::is_class_v<SimpleFluid::LinearSolverOptions>);
    static_assert(std::is_class_v<SimpleFluid::LinearResidualScaling>);
    static_assert(std::is_class_v<SimpleFluid::BelosLinearSolver<>>);
    static_assert(SimpleFluid::LinearSolverBackend::Gmres
                  != SimpleFluid::LinearSolverBackend::BiCGStab);
    static_assert(std::is_class_v<SimpleFluid::CellField<>>);
}

} // namespace
