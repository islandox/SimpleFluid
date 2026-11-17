/** Private direct Ordinary cell-certificate encoding in one native domain. */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/details/NativeConstructorStorage.hh"
namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativeCellCodecAccess final
{
public:
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Ordinary> decode(
        std::span<const char>, std::size_t, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static std::size_t size(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const geometry_detail::NativeConstructionAdmission&);
    static void append(std::vector<char>&, const ConvexCellGeometry<ArithmeticMode::Ordinary>&,
        const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static std::vector<char> encode(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const geometry_detail::NativeConstructorStorageOwner&);
};
} // namespace SimpleFluid::PLIC::details
