#include "geometry/details/BoundedGeometryKernels.hh"

#define SIMPLEFLUID_GEOMETRY_KERNEL ordinary_kernel
#include "geometry/details/BoundedPlanarFace.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL
#define SIMPLEFLUID_GEOMETRY_KERNEL certification_kernel
#include "geometry/details/BoundedPlanarFace.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL

namespace SimpleFluid::Meshes
{
BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits)
{
    if ((limits.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_planar_face_overlap<ArithmeticMode::Ordinary>(first, second, support, limits);
    return certification_kernel::bounded_planar_face_overlap<ArithmeticMode::BoundedCertification>(
        first, second, support, limits);
}
} // namespace SimpleFluid::Meshes
