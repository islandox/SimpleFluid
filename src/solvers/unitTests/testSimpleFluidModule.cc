#include <gtest/gtest.h>

#include <type_traits>

import SimpleFluid;

namespace
{

TEST(SimpleFluidModuleTest, ReexportsTheCompleteProjectApi)
{
    static_assert(std::is_class_v<SimpleFluid::Database>);
    static_assert(std::is_class_v<SimpleFluid::MeshHandle<>>);
    static_assert(std::is_class_v<SimpleFluid::CellField<>>);
    static_assert(std::is_class_v<
                  SimpleFluid::FVM::TransportSystem<
                      SimpleFluid::DefaultTpetraTypes>>);
    static_assert(std::is_class_v<SimpleFluid::TimeStepperOptions>);
    static_assert(std::is_class_v<SimpleFluid::BoussinesqSolver<>>);
}

} // namespace
