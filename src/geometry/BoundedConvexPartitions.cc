#include "geometry/details/BoundedConvexGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
void BoundedConvexGeometry::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const
{
    Operation scope(impl_->limits);
    ++statistics.partition_checks;
    if (!std::isfinite(relative_budget) || relative_budget < 0)
        reject("invalid partition physical budget.");
    std::vector<std::shared_ptr<const Ancestry>> leaves;
    Interval total;
    for (const auto& piece : pieces)
    {
        if (!piece)
            reject("null partition piece.");
        leaves.push_back(piece->impl_->ancestry);
        total = total + piece->impl_->volume;
    }
    const bool structural = structural_cover(impl_->ancestry, leaves);
    if (structural)
        ++statistics.structural_partition_reuses;
    else
    {
        for (const auto& piece : pieces)
            for (const auto& surface : impl_->surfaces)
                for (const auto& vertex : piece->impl_->vertices)
                {
                    active_work->pair();
                    const auto d = distance(vertex, piece->impl_->frame, surface.plane) * Interval(surface.orientation);
                    if (sign(d) > 0)
                        reject("partition piece escapes its parent.");
                }
        for (std::size_t i = 0; i < pieces.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                active_work->pair();
                if (pieces[i]->intersect(*pieces[j]))
                    reject("partition pieces have overlapping interiors.");
            }
    }
    const auto residual = total - impl_->volume;
    const auto allowance = Interval(relative_budget) * Interval(impl_->volume.lo);
    if (std::max(std::abs(residual.lo), std::abs(residual.hi)) > allowance.lo)
        reject("partition closure exceeds its physical budget.");
}

std::size_t BoundedConvexGeometry::storage_bytes() const
{
    std::size_t bytes = sizeof(Impl) + sizeof(MomentCache) + impl_->vertices.capacity() * sizeof(Vertex) +
                        impl_->faces.capacity() * sizeof(Face) + impl_->surfaces.capacity() * sizeof(Surface) +
                        impl_->displayed.capacity() * sizeof(BoundedPoint);
    std::set<PlanePtr> planes;
    for (const auto& vertex : impl_->vertices)
    {
        bytes += vertex.incident.capacity() * sizeof(PlanePtr) +
                 vertex.authorities.capacity() * sizeof(BoundedPlaneAuthority);
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
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
BoundedPoint source_point_fan(std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity)
{
    Operation scope({});
    ++statistics.source_imports;
    if (sources.empty() || identity.size() > active_work->limits.byte_budget)
        reject("source point export requires a capped authenticated source fan.");
    std::size_t faces = 0;
    for (const auto* source : sources)
    {
        if (!source || source->storage_bytes() > active_work->limits.byte_budget ||
            source->num_faces() > active_work->limits.maximum_faces - faces)
            reject("source point authority exceeds bootstrap storage/fan cap.");
        faces += source->num_faces();
    }
    const auto coordinates = numerical_point(identity);
    BoundedPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.value[axis] = coordinates[axis].value;
        result.absolute_error[axis] = coordinates[axis].absolute_error;
    }
    std::vector<BoundedPlaneAuthority> supports;
    for (const auto* source : sources)
        for (std::size_t f = 0; f < source->num_faces(); ++f)
        {
            const auto face = source->face_geometry(f);
            if (!face->contains_exact_point(identity))
                continue;
            const auto bytes = exact_planar_support(*face);
            BoundedPlaneAuthority authority;
            authority.key.push_back(bytes.size());
            authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
            for (std::size_t i = 0; i < bytes.size(); ++i)
                authority.key[1 + i / 8] |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                                            << (8 * (i % 8));
            active_work->charge(authority.key.size(), sizeof(std::uint64_t));
            supports.push_back(std::move(authority));
        }
    BoundedGeometryAccess::seal(result, std::move(supports));
    return result;
}
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
