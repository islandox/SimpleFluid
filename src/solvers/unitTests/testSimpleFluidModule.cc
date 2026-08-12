#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#endif

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
    static_assert(std::is_class_v<SimpleFluid::MeshQualityGate>);
    static_assert(std::is_class_v<
                  SimpleFluid::PoissonWallDistanceEquation<>>);
    static_assert(std::is_class_v<
                  SimpleFluid::AdaptiveSteadyStateController>);
    static_assert(std::is_class_v<SimpleFluid::BoussinesqSolver<>>);
}

} // namespace
