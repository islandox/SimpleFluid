#include "geometry/mesh/AMR/NativePublicationProof.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/NativeAdaptivePayloadAccess.hh"
#include "geometry/mesh/NativeConstructionAdmission.hh"
#include <bit>
#include <limits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
template<class A, class B> bool same_control(const std::shared_ptr<A>& a, const std::shared_ptr<B>& b) noexcept
{
    return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
template<class T> void require_owned_control(const std::shared_ptr<T>& value)
{
    const std::shared_ptr<T> empty;
    const bool owns = value.owner_before(empty) || empty.owner_before(value);
    if (bool(value) != owns)
        throw std::invalid_argument("Native source selector has an empty or hidden alias control.");
}
bool same_fact(double a, double b) noexcept
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
bool same_surface(const AdaptiveSurface& a, const AdaptiveSurface& b) noexcept
{
    return a.key == b.key && same_control(a.geometry, b.geometry) && a.orientation == b.orientation &&
           a.exterior == b.exterior && a.boundary == b.boundary && a.native_face == b.native_face &&
           a.authority_image == b.authority_image && a.require_native_whole_tile == b.require_native_whole_tile &&
           same_control(a.native_whole_geometry, b.native_whole_geometry);
}
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Native source binding admission overflows.");
    return a + b;
}
} // namespace

NativeSourceBinding::NativeSourceBinding(ConstructionKey, geometry_detail::NativeConstructionAdmission admission)
    : admission_(std::move(admission)), metadata_(geometry_detail::native_constructor_storage(admission_)) {}
std::shared_ptr<const NativeSourceBinding> NativeSourceBinding::capture(
    const AdaptiveLeaf& source, std::shared_ptr<Budget> arena)
{
    return capture(source, geometry_detail::NativeConstructionAdmission(std::move(arena)));
}
std::shared_ptr<const NativeSourceBinding> NativeSourceBinding::capture(
    const AdaptiveLeaf& source, const geometry_detail::NativeConstructionAdmission& admission)
{
    require_owned_control(source.native_recipe);
    require_owned_control(source.geometry);
    require_owned_control(source.join_recipe);
    require_owned_control(source.factor_recipe);
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceBinding>(
        admission, ConstructionKey{}, admission);
    auto& result = *tracked.owner;
    result.allocation_ = tracked.envelope;
    auto& storage = *result.metadata_.owner;
    auto& copy = result.source_;
    copy.key.region = source.key.region;
    std::visit([&](const auto& ancestry)
    {
        using A = std::decay_t<decltype(ancestry)>;
        A value{};
        if constexpr (std::is_same_v<A, ConvexCellAncestry>)
        {
            value.root = ancestry.root;
            storage.copy(value.path, std::span<const std::uint64_t>(ancestry.path));
        }
        else if constexpr (std::is_same_v<A, ProductCellAncestry>)
        {
            value.base.root = ancestry.base.root;
            value.base.root_template = ancestry.base.root_template;
            value.axial.root = ancestry.axial.root;
            storage.copy(value.base.path, std::span<const std::uint64_t>(ancestry.base.path));
            storage.copy(value.axial.path, std::span<const std::uint64_t>(ancestry.axial.path));
        }
        else value = ancestry;
        copy.key.ancestry = std::move(value);
    }, source.key.ancestry);
    storage.copy(copy.domain, std::span<const char>(source.domain));
    storage.copy(copy.parent_family, std::span<const char>(source.parent_family));
    copy.material = source.material;
    copy.revision = source.revision;
    copy.provider = source.provider;
    copy.geometry = source.geometry;
    copy.native_recipe = source.native_recipe;
    copy.native_cell = source.native_cell;
    copy.join_recipe = source.join_recipe;
    copy.factor_recipe = source.factor_recipe;
    copy.work = source.work;
    storage.reserve(copy.surfaces, source.surfaces.size());
    for (const auto& surface : source.surfaces)
    {
        require_owned_control(surface.geometry);
        require_owned_control(surface.native_whole_geometry);
        AdaptiveSurface value;
        storage.copy(value.key, std::span<const char>(surface.key));
        storage.string(value.boundary, surface.boundary);
        value.geometry = surface.geometry;
        value.orientation = surface.orientation;
        value.exterior = surface.exterior;
        value.native_face = surface.native_face;
        value.authority_image = surface.authority_image;
        value.require_native_whole_tile = surface.require_native_whole_tile;
        value.native_whole_geometry = surface.native_whole_geometry;
        copy.surfaces.push_back(std::move(value));
    }
    if (source.native_recipe)
    {
        result.recipe_kind_ = source.native_recipe->kind;
        storage.copy(result.recipe_view_, std::span<const char>(source.native_recipe->view));
        storage.copy(result.recipe_parts_,
            std::span<const std::shared_ptr<const NativeFragmentBytes>>(source.native_recipe->parts));
        for (const auto& part : result.recipe_parts_) require_owned_control(part);
    }
    if (source.factor_recipe)
        result.factor_parts_ = {source.factor_recipe->root, source.factor_recipe->base, source.factor_recipe->axis};
    for (const auto& part : result.factor_parts_) require_owned_control(part);
    result.payload_ = result.metadata_.owner->exclusive_bytes();
    result.control_allocation_ = add(tracked.envelope->fixed_allocation_bytes(),
        result.metadata_.envelope->fixed_allocation_bytes());
    return std::move(tracked.owner);
}
NativeSourceBinding::~NativeSourceBinding() = default;
bool NativeSourceBinding::matches(const AdaptiveLeaf& value) const noexcept
{
    const auto& a = source_;
    if (a.key != value.key || a.domain != value.domain || a.material != value.material ||
        a.revision != value.revision || a.provider != value.provider || a.native_cell != value.native_cell ||
        a.parent_family != value.parent_family || !same_control(a.native_recipe, value.native_recipe) ||
        !same_control(a.geometry, value.geometry) || !same_control(a.join_recipe, value.join_recipe) ||
        !same_control(a.factor_recipe, value.factor_recipe) ||
        !same_fact(a.work.reconstruction, value.work.reconstruction) ||
        !same_fact(a.work.interface, value.work.interface) || !same_fact(a.work.population, value.work.population) ||
        !same_fact(a.work.migration, value.work.migration) || a.surfaces.size() != value.surfaces.size())
        return false;
    if (value.native_recipe)
    {
        const auto& recipe = *value.native_recipe;
        if (recipe.kind != recipe_kind_ || recipe.view != recipe_view_ || recipe.parts.size() != recipe_parts_.size())
            return false;
        for (std::size_t index = 0; index < recipe_parts_.size(); ++index)
            if (!same_control(recipe.parts[index], recipe_parts_[index]))
                return false;
    }
    if (value.factor_recipe && (!same_control(factor_parts_[0], value.factor_recipe->root) ||
                                   !same_control(factor_parts_[1], value.factor_recipe->base) ||
                                   !same_control(factor_parts_[2], value.factor_recipe->axis)))
        return false;
    for (std::size_t index = 0; index < a.surfaces.size(); ++index)
        if (!same_surface(a.surfaces[index], value.surfaces[index]))
            return false;
    return true;
}
void NativeSourceBinding::graph(const std::shared_ptr<const NativeSourceBinding>& binding,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid,
    PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!binding) return;
    NativePublicationOwnerGraph::Context context(graph, retain, prepaid, accounting);
    if (!context.sink.semantic(binding, geometry_detail::NativeConstructorOwnerKind::NativeMesh, binding->allocation_))
        return;
    context.sink.fixed(binding->allocation_);
    context.sink.storage(binding->metadata_);
    context.sink.state_edge(context.sink.context, binding->admission_.state());
    binding->graph_children(graph, retain, accounting);
}
void NativeSourceBinding::graph_children(PLIC::details::GeometryOwnerGraphEditor& graph, bool retain,
    PLIC::details::NativeGeometryAccounting* accounting) const
{
    NativeSourceOwnerStore::leaf(source_, graph, retain, bool(accounting), accounting);
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    const auto bytes = [&](const std::shared_ptr<const AdaptiveBytes>& value)
    {
        if (value && !NativeAdaptivePayloadAccess::bytes(value, context.sink))
            graph.external(value, sizeof(*value) + value->capacity(), retain, bool(accounting));
    };
    for (const auto& part : recipe_parts_) bytes(part);
    for (const auto& part : factor_parts_) bytes(part);
}

void NativeGeometryRootStore::certify(const PLIC::details::GeometryOwnerUnion& union_owners,
    PLIC::GeometryAllocationLedger& ordinary, PLIC::GeometryAllocationLedger& bounded, std::size_t& used,
    std::size_t limit) const
{
    if (root_)
        union_owners.import_subset(binding_, ordinary, bounded, used, limit);
}
PLIC::details::GeometryOwnerUnion NativeGeometryRootStore::stage(std::shared_ptr<const NativePublicationRoot> root,
    const PLIC::details::GeometryOwnerUnion& union_owners, PLIC::GeometryAllocationLedger& ordinary,
    PLIC::GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit, NativeGeometryRootStore& successor,
    PLIC::details::GeometryOwnerUnionWork* work) const
{
    if (&successor == this)
        throw std::invalid_argument("Native proof staging may not mutate its accepted root store.");
    NativeGeometryRootStore candidate;
    candidate.root_ = std::move(root);
    if (candidate.root_)
        candidate.binding_ = PLIC::details::GeometryOwnerRootBinding(
            candidate.root_, 0, candidate.root_->owners(), &candidate.root_->paid_receipts());
    auto staged_union = union_owners.staged(binding_ ? &binding_ : nullptr,
        candidate.binding_ ? &candidate.binding_ : nullptr, ordinary, bounded, used, limit, work);
    if (candidate.root_)
        staged_union.import_subset(candidate.binding_, ordinary, bounded, used, limit);
    successor.swap(candidate);
    return staged_union;
}
void NativeGeometryRootStore::swap(NativeGeometryRootStore& other) noexcept
{
    using std::swap;
    swap(root_, other.root_);
    swap(binding_, other.binding_);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
