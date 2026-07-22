#include <gtest/gtest.h>

#include <type_traits>

import SimpleFluid.LinearSolvers;

namespace
{

TEST(LinearSolversModuleTest, ExportsBelosWrapperAndFieldTypes)
{
    static_assert(std::is_class_v<SimpleFluid::LinearSolverOptions>);
    static_assert(std::is_class_v<SimpleFluid::BelosLinearSolver<>>);
    static_assert(std::is_class_v<SimpleFluid::CellField<>>);
}

} // namespace
