#include "geometry/details/BoundedGeometryKernels.hh"

namespace SimpleFluid::Meshes
{
const void* bounded_geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::Meshes

#define SIMPLEFLUID_GEOMETRY_KERNEL ordinary_kernel
#include "geometry/details/BoundedConvexPackets.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL
#define SIMPLEFLUID_GEOMETRY_KERNEL certification_kernel
#include "geometry/details/BoundedConvexPackets.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL

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
