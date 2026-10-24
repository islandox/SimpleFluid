#include "geometry/details/BoundedGeometryKernels.hh"

#define SIMPLEFLUID_GEOMETRY_KERNEL ordinary_kernel
#include "geometry/details/BoundedConvexClipping.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL
#define SIMPLEFLUID_GEOMETRY_KERNEL certification_kernel
#include "geometry/details/BoundedConvexClipping.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL

namespace SimpleFluid::Meshes
{
BoundedClipResult BoundedConvexGeometry::clip(const GeometryHalfspace& input, bool measure_section) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip(input, measure_section);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip(
        input, measure_section);
}

BoundedClipResult BoundedConvexGeometry::clip_by_face(
    const BoundedConvexGeometry& source, std::size_t face, bool measure_section) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip_by_face(
            source, face, measure_section);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip_by_face(
        source, face, measure_section);
}

BoundedClipResult BoundedConvexGeometry::clip_impl(const std::shared_ptr<const Impl>& source, std::size_t plane_id,
    bool measure_section, bool construct_outside, bool partition_proof) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).clip_impl(
            source, plane_id, measure_section, construct_outside, partition_proof);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).clip_impl(
        source, plane_id, measure_section, construct_outside, partition_proof);
}

BoundedPhaseMeasures BoundedConvexGeometry::phases(const GeometryHalfspace& gas) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).phases(gas);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).phases(gas);
}

bool BoundedConvexGeometry::bounds_prove_disjoint_interiors(const BoundedConvexGeometry& other) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).bounds_prove_disjoint_interiors(other);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_)
        .bounds_prove_disjoint_interiors(other);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::intersect(const BoundedConvexGeometry& other) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).intersect(other);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).intersect(other);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::translated(GeometryPoint displacement) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).translated(displacement);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).translated(displacement);
}
} // namespace SimpleFluid::Meshes
