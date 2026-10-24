#include "geometry/details/BoundedGeometryKernels.hh"

namespace SimpleFluid::Meshes
{
std::size_t BoundedConvexPartition::storage_bytes() const noexcept
{
    return sizeof(*this) + donors_.capacity() * sizeof(BoundedConvexGeometryPtr);
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
namespace
{
template<ArithmeticMode Mode>
std::size_t geometry_storage_bytes(const std::shared_ptr<const BoundedConvexGeometry::Impl>& base)
{
    using Impl = GeometryStorage<Mode>;
    using MomentCache = StoredMomentCache<Mode>;
    using Vertex = StoredVertex<Mode>;
    using Surface = StoredSurface<Mode>;
    using Plane = StoredPlane<Mode>;
    using PlanePtr = StoredPlanePtr<Mode>;
    const auto impl_ = checked_geometry_storage<Mode>(base);
    std::size_t bytes = sizeof(Impl) + sizeof(MomentCache) + impl_->vertices.capacity() * sizeof(Vertex) +
                        impl_->faces.capacity() * sizeof(Face) + impl_->surfaces.capacity() * sizeof(Surface) +
                        impl_->displayed.capacity() * sizeof(BoundedPoint);
    if (impl_->partition_certificate)
        bytes += sizeof(PartitionCertificate) +
                 impl_->partition_certificate->members.capacity() * sizeof(std::weak_ptr<const Ancestry>);
    std::set<PlanePtr> planes;
    for (const auto& vertex : impl_->vertices)
    {
        bytes += vertex.incident.capacity() * sizeof(PlanePtr) +
                 vertex.authorities.capacity() * sizeof(BoundedPlaneAuthority);
        bytes += BoundedGeometryAccess::identity_bytes(vertex.source_identity);
        for (const auto& key : vertex.authorities)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
        planes.insert(vertex.incident.begin(), vertex.incident.end());
    }
    for (const auto& face : impl_->faces)
        bytes += face.capacity() * sizeof(std::size_t);
    for (const auto& p : impl_->displayed)
        bytes += bounded_point_storage_bytes(p) - sizeof(BoundedPoint);
    for (const auto& plane : planes)
    {
        bytes += sizeof(Plane) + plane->authorities.capacity() * sizeof(BoundedPlaneAuthority);
        bytes += plane->source_vertices.capacity() * sizeof(std::weak_ptr<const BoundedPointProvenance>);
        for (const auto& key : plane->authorities)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
    }
    std::set<const Split*> splits;
    for (auto node = impl_->ancestry; node; node = node->parent)
    {
        bytes += sizeof(Ancestry);
        if (node->split && splits.insert(node->split.get()).second)
            bytes += sizeof(Split);
    }
    return bytes;
}
} // namespace

std::size_t BoundedConvexGeometry::storage_bytes() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return geometry_storage_bytes<ArithmeticMode::Ordinary>(impl_);
    return geometry_storage_bytes<ArithmeticMode::BoundedCertification>(impl_);
}

} // namespace SimpleFluid::Meshes

#include "geometry/details/ordinary/BoundedConvexPartitions.tcc"
#include "geometry/details/BoundedConvexPartitions.tcc"

namespace SimpleFluid::Meshes
{
void BoundedConvexGeometry::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).certify_partition(pieces, relative_budget);
    else
        certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).certify_partition(
            pieces, relative_budget);
}

BoundedConvexPartition::BoundedConvexPartition(
    std::span<const BoundedConvexGeometryPtr> donors, BoundedGeometryLimits limits)
    : limits_(limits)
{
    donors_ = limits.arithmetic_mode == ArithmeticMode::Ordinary
                  ? ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::validate_donors(donors, limits)
                  : certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::validate_donors(
                        donors, limits);
}

std::vector<BoundedPartitionPiece> BoundedConvexPartition::partition(
    const BoundedConvexGeometry& parent, std::size_t maximum_pieces) const
{
    if (limits_.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::PartitionKernel<ArithmeticMode::Ordinary>(*this).partition(parent, maximum_pieces);
    return certification_kernel::PartitionKernel<ArithmeticMode::BoundedCertification>(*this).partition(
        parent, maximum_pieces);
}
} // namespace SimpleFluid::Meshes
