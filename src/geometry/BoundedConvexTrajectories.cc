#include "geometry/details/BoundedGeometryKernels.hh"

#define SIMPLEFLUID_GEOMETRY_KERNEL ordinary_kernel
#include "geometry/details/BoundedConvexTrajectories.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL
#define SIMPLEFLUID_GEOMETRY_KERNEL certification_kernel
#include "geometry/details/BoundedConvexTrajectories.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL

namespace SimpleFluid::Meshes
{
BoundedPoint bounded_source_point(const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode)
{
    if ((mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_source_point<ArithmeticMode::Ordinary>(source, identity, mode);
    return certification_kernel::bounded_source_point<ArithmeticMode::BoundedCertification>(source, identity, mode);
}

BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode)
{
    if ((mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_source_point<ArithmeticMode::Ordinary>(sources, identity, mode);
    return certification_kernel::bounded_source_point<ArithmeticMode::BoundedCertification>(sources, identity, mode);
}

BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement)
{
    if ((source.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_point_displaced<ArithmeticMode::Ordinary>(source, displacement);
    return certification_kernel::bounded_point_displaced<ArithmeticMode::BoundedCertification>(source, displacement);
}

BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits)
{
    if ((limits.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::backtrace(
            source, velocity, timestep, walls, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::backtrace(
        source, velocity, timestep, walls, limits);
}
} // namespace SimpleFluid::Meshes
