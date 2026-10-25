#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexPartitions.tcc"
#include "geometry/PLIC/details/ordinary/ConvexPartitions.tcc"

namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> std::size_t DynamicConvexGeometry::storage_bytes() const
{
    return geometry_allocation_footprint<Mode>(impl_);
}
template std::size_t DynamicConvexGeometry::storage_bytes<ArithmeticMode::Ordinary>() const;
template std::size_t DynamicConvexGeometry::storage_bytes<ArithmeticMode::BoundedCertification>() const;
} // namespace SimpleFluid::PLIC
