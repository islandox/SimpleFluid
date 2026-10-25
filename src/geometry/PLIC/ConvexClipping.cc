#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexClipping.tcc"
#include "geometry/PLIC/details/ordinary/ConvexClipping.tcc"

namespace SimpleFluid::PLIC
{
DynamicClipResult DynamicConvexGeometry::clip(const GeometryHalfspace& input, bool measure_section) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip(input, measure_section);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip(
        input, measure_section);
}

DynamicClipResult DynamicConvexGeometry::clip_by_face(
    const DynamicConvexGeometry& source, std::size_t face, bool measure_section) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip_by_face(
            source, face, measure_section);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip_by_face(
        source, face, measure_section);
}

DynamicClipResult DynamicConvexGeometry::clip_impl(const std::shared_ptr<const Impl>& source, std::size_t plane_id,
    bool measure_section, bool construct_outside, bool partition_proof) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip_impl(
            source, plane_id, measure_section, construct_outside, partition_proof);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip_impl(
        source, plane_id, measure_section, construct_outside, partition_proof);
}

DynamicPhaseMeasures DynamicConvexGeometry::phases(const GeometryHalfspace& gas) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).phases(gas);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).phases(gas);
}

bool DynamicConvexGeometry::bounds_prove_disjoint_interiors(const DynamicConvexGeometry& other) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).bounds_prove_disjoint_interiors(other);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_)
        .bounds_prove_disjoint_interiors(other);
}

DynamicConvexGeometryPtr DynamicConvexGeometry::intersect(const DynamicConvexGeometry& other) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).intersect(other);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).intersect(other);
}

DynamicConvexGeometryPtr DynamicConvexGeometry::translated(GeometryPoint displacement) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).translated(displacement);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).translated(displacement);
}
} // namespace SimpleFluid::PLIC
