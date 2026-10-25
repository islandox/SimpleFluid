/** @file ConstructionRegistry.cc
 * @brief Capped construction metadata and finite registry instantiations.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/ConstructionRegistry.tcc"
#include "geometry/PLIC/details/ConstructionIncidence.tcc"

namespace SimpleFluid::PLIC::details
{
template<ArithmeticMode Mode>
std::shared_ptr<const GeometryConstruction> construction_source(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& source)
{
    require_mode(source->limits.arithmetic_mode, Mode);
    if (source->construction)
        return source->construction;
    active_work->charge(1, sizeof(GeometryConstruction));
    auto result = std::make_shared<GeometryConstruction>();
    result->source = source;
    // This root has no construction record of its own, so retaining it cannot
    // create a geometry-self cycle. It pins all original point definitions.
    return result;
}

std::size_t construction_storage_bytes(const std::shared_ptr<const GeometryConstruction>& root)
{
    if (!root)
        return 0;
    std::size_t bytes = 0;
    std::set<const GeometryConstruction*> nodes;
    std::set<const GeometryClipConstruction*> clips;
    std::set<const DynamicConvexGeometry::Impl*> sources;
    std::set<const DynamicPointProvenance*> points;
    std::vector<std::shared_ptr<const GeometryConstruction>> pending{root};
    const auto retained = [&](const auto& values)
    {
        bytes += values.capacity() * sizeof(std::shared_ptr<const DynamicPointProvenance>);
        for (const auto& point : values)
            if (point && points.insert(point.get()).second)
                bytes += DynamicGeometryAccess::identity_bytes(point);
    };
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        const auto node = pending[i];
        if (!node || !nodes.insert(node.get()).second)
            continue;
        bytes += sizeof(GeometryConstruction);
        retained(node->retained_points);
        if (node->source && sources.insert(node->source.get()).second)
        {
            const auto source = DynamicGeometryAccess::geometry(node->source);
            bytes += node->source->limits.arithmetic_mode == ArithmeticMode::Ordinary
                         ? source->storage_bytes<ArithmeticMode::Ordinary>()
                         : source->storage_bytes<ArithmeticMode::BoundedCertification>();
        }
        if (node->parent)
            pending.push_back(node->parent);
        if (node->clip && clips.insert(node->clip.get()).second)
        {
            bytes += sizeof(GeometryClipConstruction);
            retained(node->clip->retained_points);
            pending.push_back(node->clip->parent);
        }
    }
    return bytes;
}

template std::shared_ptr<const GeometryConstruction> construction_source<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
template std::shared_ptr<const GeometryConstruction> construction_source<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
template class ConstructionRegistry<ArithmeticMode::Ordinary>;
template class ConstructionRegistry<ArithmeticMode::BoundedCertification>;
template std::vector<char> ConvexCellGeometry<ArithmeticMode::Ordinary>::encode(
    const ConstructionRegistry<ArithmeticMode::Ordinary>&, std::size_t) const;
template std::vector<char> ConvexCellGeometry<ArithmeticMode::BoundedCertification>::encode(
    const ConstructionRegistry<ArithmeticMode::BoundedCertification>&, std::size_t) const;
template ConvexGeometryPtr<ArithmeticMode::Ordinary> ConvexCellGeometry<ArithmeticMode::Ordinary>::decode(
    std::span<const char>, const ConstructionRegistry<ArithmeticMode::Ordinary>&, GeometryLimits);
template ConvexGeometryPtr<ArithmeticMode::BoundedCertification>
ConvexCellGeometry<ArithmeticMode::BoundedCertification>::decode(
    std::span<const char>, const ConstructionRegistry<ArithmeticMode::BoundedCertification>&, GeometryLimits);
} // namespace SimpleFluid::PLIC
