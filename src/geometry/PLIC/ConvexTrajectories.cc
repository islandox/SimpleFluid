#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexTrajectories.tcc"
#include "geometry/PLIC/details/ordinary/ConvexTrajectories.tcc"

namespace SimpleFluid::PLIC
{
DynamicPoint dynamic_source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, std::span<const char> identity, ArithmeticMode mode)
{
    if ((mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::dynamic_source_point<ArithmeticMode::Ordinary>(source, identity, mode);
    return certification_kernel::dynamic_source_point<ArithmeticMode::BoundedCertification>(source, identity, mode);
}

DynamicPoint dynamic_source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources,
    std::span<const char> identity, ArithmeticMode mode)
{
    if ((mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::dynamic_source_point<ArithmeticMode::Ordinary>(sources, identity, mode);
    return certification_kernel::dynamic_source_point<ArithmeticMode::BoundedCertification>(sources, identity, mode);
}

DynamicPoint dynamic_point_displaced(const DynamicPoint& source, const DynamicPoint& displacement)
{
    if ((source.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::dynamic_point_displaced<ArithmeticMode::Ordinary>(source, displacement);
    return certification_kernel::dynamic_point_displaced<ArithmeticMode::BoundedCertification>(source, displacement);
}

DynamicPoint dynamic_backtrace_constrained(const DynamicPoint& source, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint> walls, GeometryLimits limits)
{
    if ((limits.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::backtrace(
            source, velocity, timestep, walls, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::backtrace(
        source, velocity, timestep, walls, limits);
}
} // namespace SimpleFluid::PLIC
