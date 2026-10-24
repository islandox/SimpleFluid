#include "geometry/details/BoundedGeometryKernels.hh"

namespace SimpleFluid::Meshes
{
const void* bounded_geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::Meshes

#include "geometry/details/ordinary/BoundedConvexPackets.tcc"
#include "geometry/details/BoundedConvexPackets.tcc"

namespace SimpleFluid::Meshes
{
std::vector<char> BoundedConvexGeometry::encode(
    const BoundedGeometryPacketContext& context, std::size_t byte_budget) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).encode(context, byte_budget);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).encode(
        context, byte_budget);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::decode(std::span<const char> packet,
    const BoundedGeometryPacketContext& expected, std::span<const BoundedConvexGeometryPtr> authenticated_sources,
    BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::decode(
            packet, expected, authenticated_sources, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::decode(
        packet, expected, authenticated_sources, limits);
}
} // namespace SimpleFluid::Meshes
