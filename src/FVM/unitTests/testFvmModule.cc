#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#include <utility>
#endif

import SimpleFluid.FVM;

namespace
{

TEST(FvmModuleTest, ExportsFiniteVolumeTypes)
{
    using Pack = SimpleFluid::DefaultTpetraTypes;

    static_assert(std::is_same_v<
                  typename SimpleFluid::BoundaryCache<Pack>::value_type,
                  typename Pack::scalar_type>);
    static_assert(std::is_same_v<
                  typename SimpleFluid::FVM::VelocityBoundaryCache<Pack>::vec_type,
                  SimpleFluid::vec3<typename Pack::scalar_type>>);
    static_assert(std::is_class_v<
                  SimpleFluid::FVM::CellGradientCache<Pack>>);
    static_assert(std::is_class_v<
                  SimpleFluid::FVM::TransportGeometryCache<
                      SimpleFluid::Mesh<Pack>>>);
    static_assert(SimpleFluid::FVM::CellGradientScheme::LeastSquares
                  != SimpleFluid::FVM::CellGradientScheme::GaussLinear);
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(
                      *std::declval<SimpleFluid::FVM::TransportSystem<Pack>&>()
                           .matrix)>,
                  typename Pack::matrix_type>);
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(
                      *std::declval<SimpleFluid::FVM::TransportSystem<Pack>&>()
                           .rhs)>,
                  typename Pack::vector_type>);
}

} // namespace
