#include "geometry/PLIC/details/GeometryKernels.hh"

namespace SimpleFluid::PLIC
{
const void* geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::PLIC

#include "geometry/PLIC/details/bounded/ConvexPackets.tcc"
#include "geometry/PLIC/details/ordinary/ConvexPackets.tcc"

namespace SimpleFluid::PLIC
{
std::vector<char> DynamicConvexGeometry::encode(const GeometryPacketContext& context, std::size_t byte_budget) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).encode(context, byte_budget);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).encode(
        context, byte_budget);
}

DynamicConvexGeometryPtr DynamicConvexGeometry::decode(std::span<const char> packet,
    const GeometryPacketContext& expected, std::span<const DynamicConvexGeometryPtr> authenticated_sources,
    GeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::decode(
            packet, expected, authenticated_sources, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::decode(
        packet, expected, authenticated_sources, limits);
}
} // namespace SimpleFluid::PLIC
