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
std::shared_ptr<const GeometryConstruction> construction_node(GeometryConstruction::Source value)
{
    active_work->charge(1, sizeof(GeometryConstruction));
    ordinary_kernel::NativeSplitCppProfile::combined<const GeometryConstruction>();
    return std::make_shared<const GeometryConstruction>(std::move(value));
}
std::shared_ptr<const GeometryConstruction> construction_node(GeometryConstruction::Clip value)
{
    active_work->charge(1, sizeof(GeometryConstruction));
    ordinary_kernel::NativeSplitCppProfile::combined<const GeometryConstruction>();
    return std::make_shared<const GeometryConstruction>(std::move(value));
}
std::shared_ptr<const GeometryConstruction> construction_native(NativeGeometryConstruction value)
{
    active_work->charge(1, sizeof(GeometryConstruction) + sizeof(NativeGeometryConstruction));
    return std::make_shared<const GeometryConstruction>(GeometryConstruction::Source{
        std::make_unique<const NativeGeometryConstruction>(std::move(value))});
}
std::shared_ptr<const GeometryConstruction> retained_native_plane_source(
    std::shared_ptr<const GeometryConstruction> node, const void* plane)
{
    const auto owns_plane = [plane](const std::shared_ptr<const GeometryConstruction>& root)
    {
        if (!root || root->kind() != GeometryConstruction::Kind::Source)
            return false;
        const auto* native = root->source().native();
        return native && native->body && std::any_of(native->body->topology->planes.begin(),
            native->body->topology->planes.end(), [&](const auto& original) { return original.get() == plane; });
    };
    // Factory-created construction lineages are immutable and acyclic. Each
    // retention edge terminates at a Source, so this walk needs no work arrays
    // or numerical predicates and does not change the clip operation order.
    while (node)
    {
        if (node->kind() == GeometryConstruction::Kind::Source)
            return owns_plane(node) ? node : nullptr;
        if (node->kind() == GeometryConstruction::Kind::Rebase)
            node = node->rebase().parent;
        else
        {
            const auto& clip = node->clip().clip;
            if (!clip)
                return {};
            if (owns_plane(clip->native_plane_source))
                return clip->native_plane_source;
            node = clip->parent;
        }
    }
    return {};
}
bool same_native_construction(const GeometryConstruction& first, const GeometryConstruction& second)
{
    if (first.kind() != GeometryConstruction::Kind::Source || second.kind() != GeometryConstruction::Kind::Source)
        return false;
    const auto* a = first.source().native();
    const auto* b = second.source().native();
    return a && b && (a == b || (a->primed && b->primed && a->source == b->source && a->cell_slot == b->cell_slot));
}
void retain_rebase_export(const std::shared_ptr<const DynamicConvexGeometry::Impl>& source,
    DynamicConvexGeometry::Impl& target)
{
    if (source->native_export_source)
        target.native_export_source = source->native_export_source;
    else if (!source->export_source && source->construction &&
             source->construction->kind() == GeometryConstruction::Kind::Source &&
             source->construction->source().native())
        target.native_export_source = source->construction;
    else
        target.export_source = source->export_source ? source->export_source : source;
}
std::shared_ptr<const GeometryConstruction> construction_rebase(
    std::shared_ptr<const GeometryConstruction> parent, const GeometryHalfspace& input)
{
    // Charge the node and its exclusive rare payload before either allocation.
    active_work->charge(1, sizeof(GeometryConstruction) + sizeof(GeometryConstruction::Rebase));
    ordinary_kernel::NativeSplitCppProfile::request(1, sizeof(GeometryConstruction::Rebase));
    ordinary_kernel::NativeSplitCppProfile::combined<const GeometryConstruction>();
    auto payload = std::make_unique<const GeometryConstruction::Rebase>(
        GeometryConstruction::Rebase{std::move(parent), input});
    return std::make_shared<const GeometryConstruction>(std::move(payload));
}

template<ArithmeticMode Mode>
std::shared_ptr<const GeometryConstruction> construction_source(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& source)
{
    require_mode(source->limits.arithmetic_mode, Mode);
    if (source->construction)
        return source->construction;
    // This root has no construction record of its own, so retaining it cannot
    // create a geometry-self cycle. It pins all original point definitions.
    return construction_node(GeometryConstruction::Source{source});
}

std::size_t construction_storage_bytes(const std::shared_ptr<const GeometryConstruction>& root)
{
    if (!root)
        return 0;
    std::size_t recipe_bytes = 0, ledger_used = 0;
    const auto recipe = [&](std::size_t count, std::size_t size)
    {
        if (size && count > (std::numeric_limits<std::size_t>::max() - recipe_bytes) / size)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Construction recipe payload size overflows.");
        recipe_bytes += count * size;
    };
    std::set<const GeometryConstruction*> nodes;
    std::set<const GeometryClipConstruction*> clips;
    std::set<const DynamicConvexGeometry::Impl*> sources;
    std::set<const void*> moments;
    GeometryAllocationLedger retained_allocations;
    std::vector<std::shared_ptr<const GeometryConstruction>> pending{root};
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        const auto node = pending[i];
        if (!node || !nodes.insert(node.get()).second)
            continue;
        recipe(1, sizeof(GeometryConstruction));
        if (node->kind() == GeometryConstruction::Kind::Source)
        {
            if (const auto* native = node->source().native())
            {
                recipe(1, sizeof(NativeGeometryConstruction));
                retained_allocations.retain(native->source, ledger_used, std::numeric_limits<std::size_t>::max());
                DynamicGeometryAccess::retain_native_body(native->body, retained_allocations,
                    ledger_used, std::numeric_limits<std::size_t>::max());
                if (native->moments && moments.insert(native->moments.get()).second)
                    recipe(1, sizeof(StoredMomentCache<ArithmeticMode::Ordinary>));
                continue;
            }
            const auto owner = node->source().geometry();
            if (owner && sources.insert(owner.get()).second)
            {
                if (owner->limits.arithmetic_mode == ArithmeticMode::Ordinary)
                    DynamicGeometryAccess::retain_geometry_storage<ArithmeticMode::Ordinary>(
                        owner, retained_allocations, ledger_used, std::numeric_limits<std::size_t>::max());
                else
                    DynamicGeometryAccess::retain_geometry_storage<ArithmeticMode::BoundedCertification>(
                        owner, retained_allocations, ledger_used, std::numeric_limits<std::size_t>::max());
            }
        }
        else if (node->kind() == GeometryConstruction::Kind::Rebase)
        {
            recipe(1, sizeof(GeometryConstruction::Rebase));
            pending.push_back(node->rebase().parent);
        }
        else
        {
            const auto& clip = node->clip().clip;
            if (clip && clips.insert(clip.get()).second)
            {
                recipe(1, sizeof(GeometryClipConstruction));
                recipe(clip->retained_points.capacity(), sizeof(DynamicStoredPointRef));
                if (clip->point_context)
                {
                    if (clip->mode == ArithmeticMode::Ordinary)
                        retained_allocations.retain_snapshot<ArithmeticMode::Ordinary>(
                            clip->point_context, ledger_used, std::numeric_limits<std::size_t>::max());
                    else
                        retained_allocations.retain_snapshot<ArithmeticMode::BoundedCertification>(
                            clip->point_context, ledger_used, std::numeric_limits<std::size_t>::max());
                }
                pending.push_back(clip->parent);
                pending.push_back(clip->native_plane_source);
            }
        }
    }
    const auto payload = retained_allocations.payload_bytes();
    if (payload > std::numeric_limits<std::size_t>::max() - recipe_bytes)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Construction recipe payload size overflows.");
    return recipe_bytes + payload;
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
