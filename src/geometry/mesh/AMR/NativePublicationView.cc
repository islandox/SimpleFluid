#include "geometry/details/NativeSchemaOwner.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/NativePublicationViewData.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
const AdaptiveCellKey& NativeRemoteKeyCatalog::at_ordinal(std::size_t ordinal) const
{
    if (ordinal >= size())
        throw std::out_of_range("Native remote key ordinal is outside current membership.");
    auto node = root_.get();
    while (!node->leaf())
    {
        bool selected = false;
        for (const auto& child : node->children)
        {
            if (ordinal < child->size)
            {
                node = child.get();
                selected = true;
                break;
            }
            ordinal -= child->size;
        }
        if (!selected)
            throw std::logic_error("Native remote key subtree extent is inconsistent.");
    }
    return node->rows.at(ordinal).key();
}
std::optional<std::size_t> NativeRemoteKeyCatalog::ordinal_of(const AdaptiveCellKey& key) const
{
    auto node = root_.get();
    std::size_t prefix = 0;
    while (node && !node->leaf())
    {
        bool selected = false;
        for (const auto& child : node->children)
        {
            if (!(child->maximum->key() < key))
            {
                node = child.get();
                selected = true;
                break;
            }
            prefix += child->size;
        }
        if (!selected)
            return {};
    }
    if (!node)
        return {};
    const auto found = std::lower_bound(node->rows.begin(), node->rows.end(), key,
        [](const Row& row, const AdaptiveCellKey& value) { return row.key() < value; });
    return found == node->rows.end() || found->key() != key ? std::optional<std::size_t>{}
                                                            : prefix + std::size_t(found - node->rows.begin());
}
NativeRemoteKeyCatalog NativeRemoteKeyCatalog::edited(const AdaptiveCellKey& key,
    std::shared_ptr<const NativeRemoteKey> binding, std::shared_ptr<Budget> arena,
    persistent_ordered_detail::OrderedPageWork* work,
    std::shared_ptr<const geometry_detail::NativeAllocationState> native) const
{
    const auto* previous = Pages::lower_bound(root_, [&](const Row& row) { return row.key() < key; });
    const bool exists = previous && previous->key() == key;
    if (!binding && !exists)
        throw std::invalid_argument("Native remote key retirement lacks current membership.");
    if (binding && binding->key() != key)
        throw std::invalid_argument("Native remote key insertion differs from its detached source.");
    if (exists && binding == previous->binding)
        return *this;
    Row replacement{std::move(binding)};
    const auto& selector = replacement.binding ? replacement : *previous;
    NativeRemoteKeyCatalog result;
    result.root_ = Pages(std::move(arena), work, {}, std::move(native))
                       .edit(root_, selector, replacement.binding ? &replacement : nullptr, exists);
    return result;
}
std::shared_ptr<const NativeRemoteKey> NativeRemoteKeyCatalog::capture(
    const AdaptiveCellKey& source, const geometry_detail::NativeConstructionAdmission& admission)
{
    auto value = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeRemoteKey>(admission);
    value.owner->metadata_ = geometry_detail::native_constructor_storage(admission);
    value.owner->allocation_ = value.envelope;
    value.owner->key_.region = source.region;
    std::visit([&](const auto& ancestry)
    {
        using T = std::decay_t<decltype(ancestry)>;
        if constexpr (std::is_same_v<T, ConvexCellAncestry>)
        {
            ConvexCellAncestry copy;
            copy.root = ancestry.root;
            value.owner->metadata_.owner->copy(copy.path, std::span<const std::uint64_t>(ancestry.path));
            value.owner->key_.ancestry = std::move(copy);
        }
        else if constexpr (std::is_same_v<T, ProductCellAncestry>)
        {
            ProductCellAncestry copy;
            copy.base.root = ancestry.base.root;
            copy.base.root_template = ancestry.base.root_template;
            copy.axial.root = ancestry.axial.root;
            value.owner->metadata_.owner->copy(copy.base.path, std::span<const std::uint64_t>(ancestry.base.path));
            value.owner->metadata_.owner->copy(copy.axial.path, std::span<const std::uint64_t>(ancestry.axial.path));
            value.owner->key_.ancestry = std::move(copy);
        }
        else value.owner->key_.ancestry = ancestry;
    }, source.ancestry);
    return std::move(value.owner);
}
NativeRemoteKeyCatalog NativeRemoteKeyCatalog::with_key(
    const AdaptiveCellKey& key, const geometry_detail::NativeConstructionAdmission& admission) const
{
    if (ordinal_of(key)) return *this;
    return edited(key, capture(key, admission),
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission.state()->counter_owner)),
        nullptr, admission.state());
}
NativeRemoteKeyCatalog NativeRemoteKeyCatalog::without_key(
    const AdaptiveCellKey& key, const geometry_detail::NativeConstructionAdmission& admission) const
{
    return edited(key, {},
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission.state()->counter_owner)),
        nullptr, admission.state());
}
std::size_t NativePublicationStorageSummary::retained_bytes() const
{
    auto total = persistent_ordered_detail::checked_sum(metadata, physical_geometry);
    total = persistent_ordered_detail::checked_sum(total, scalar_numeric);
    total = persistent_ordered_detail::checked_sum(total, library_upper);
    return persistent_ordered_detail::checked_sum(total, fixed);
}
MeshStorageReport NativePublicationStorageSummary::report() const noexcept
{
    MeshStorageReport result;
    result.geometry = physical_geometry;
    result.indexing = metadata;
    result.objects = fixed;
    result.output_only = scalar_numeric;
    result.required_maps_estimate = library_upper;
    return result;
}
const AdaptiveCellKey& NativePublicationPayload::key(const NativePublicationKeyRoute& route) const
{
    return route.remote ? remote.at_ordinal(route.ordinal) : owned.at_ordinal(route.ordinal);
}
const AdaptiveCellKey& NativePublicationViewData::key(const NativePublicationKeyRoute& route) const
{
    return compatibility ? compatibility->local_keys.at(route.gid) : payload->key(route);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR
{
std::span<const char> AdaptiveNativeSchemaView::span() const noexcept
{
    return owner_ ? owner_->span() : std::span<const char>{};
}
AdaptiveNativeSchemaView native_publication_schema(const DistributedAdaptiveHierarchy& hierarchy)
{
    std::shared_ptr<const geometry_detail::NativeSchemaOwner> owner;
    std::weak_ptr<const adaptive_detail::NativeKernelAuthority> authority;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    adaptive_detail::collective(*hierarchy.communicator(),
        [&]
        {
            if (!hierarchy.geometry_unchanged())
                throw std::invalid_argument("Native schema source geometry observation is stale.");
            const auto& kernel = hierarchy.native_publication_kernel();
            owner = adaptive_detail::NativeKernelAuthority::schema_observation(kernel);
            if (owner)
            {
                authority = adaptive_detail::NativeKernelAuthority::observation(kernel);
                allocation = adaptive_detail::NativeKernelAuthority::allocation_observation(kernel);
                return;
            }
            auto bytes = kernel->descriptor();
            if (bytes.size() > hierarchy.options().byte_budget)
                throw std::invalid_argument("Full fallback native schema exceeds its byte cap.");
            auto counter = geometry_detail::NativeConstructionCounterAccess::create<adaptive_detail::Budget>(
                hierarchy.options().byte_budget);
            geometry_detail::NativeConstructionAdmission admission(counter);
            owner = geometry_detail::NativeSchemaOwnerAccess::adopt(std::move(bytes), admission);
        });
    return AdaptiveNativeSchemaView(std::move(owner), std::move(authority), std::move(allocation));
}
OwnedNativeGidsView::const_iterator::const_iterator(
    std::shared_ptr<const adaptive_detail::NativePublicationViewData> data, std::size_t ordinal)
    : data_(std::move(data)), position_(ordinal)
{
}
OwnedNativeGidsView::const_iterator::reference OwnedNativeGidsView::const_iterator::operator*() const
{
    if (const auto* at = std::get_if<CompatibilityIterator>(&position_))
    {
        if (!data_ || *at == data_->compatibility->owned_gids.end())
            throw std::out_of_range("Native owning key iterator is outside current membership.");
        const auto& entry = **at;
        return {entry.first, entry.second};
    }
    const auto ordinal = std::get<std::size_t>(position_);
    if (!data_ || !data_->payload || ordinal >= data_->payload->owned.size())
        throw std::out_of_range("Native owned key iterator is outside current membership.");
    return {data_->payload->owned.at_ordinal(ordinal), data_->payload->cell_begin + ordinal};
}
std::size_t OwnedNativeGidsView::size() const noexcept
{
    return data_ && data_->compatibility ? data_->compatibility->owned_gids.size() :
        (data_ && data_->payload ? data_->payload->owned.size() : 0);
}
std::uint64_t OwnedNativeGidsView::at(const AdaptiveCellKey& key) const
{
    const auto found = find(key);
    if (found == end())
        throw std::out_of_range("Native owned key is not in current membership.");
    return (*found).second;
}
bool OwnedNativeGidsView::contains(const AdaptiveCellKey& key) const
{
    return find(key) != end();
}
OwnedNativeGidsView::const_iterator OwnedNativeGidsView::find(const AdaptiveCellKey& key) const
{
    if (data_ && data_->compatibility)
        return {data_, data_->compatibility->owned_gids.find(key)};
    const auto ordinal = data_ && data_->payload ? data_->payload->owned.ordinal_of(key) : std::optional<std::size_t>{};
    return {data_, ordinal ? *ordinal : size()};
}
OwnedNativeGidsView::const_iterator OwnedNativeGidsView::begin() const
{
    if (data_ && data_->compatibility)
        return {data_, data_->compatibility->owned_gids.begin()};
    return {data_, 0};
}
OwnedNativeGidsView::const_iterator OwnedNativeGidsView::end() const
{
    if (data_ && data_->compatibility)
        return {data_, data_->compatibility->owned_gids.end()};
    return {data_, size()};
}

LocalNativeKeysView::const_iterator::const_iterator(
    std::shared_ptr<const adaptive_detail::NativePublicationViewData> data, std::size_t ordinal)
    : data_(std::move(data)), ordinal_(ordinal)
{
}
LocalNativeKeysView::const_iterator::reference LocalNativeKeysView::const_iterator::operator*() const
{
    if (!data_ || ordinal_ >= data_->payload->keys.size())
        throw std::out_of_range("Native local key iterator is outside current routes.");
    const auto& route = data_->payload->keys[ordinal_];
    return {route.gid, data_->key(route)};
}
std::size_t LocalNativeKeysView::size() const noexcept
{
    return data_ ? data_->payload->keys.size() : 0;
}
const AdaptiveCellKey& LocalNativeKeysView::at(std::uint64_t gid) const
{
    const auto found = find(gid);
    if (found == end())
        throw std::out_of_range("Native local GID is not in current routes.");
    return (*found).second;
}
bool LocalNativeKeysView::contains(std::uint64_t gid) const
{
    return find(gid) != end();
}
LocalNativeKeysView::const_iterator LocalNativeKeysView::find(std::uint64_t gid) const
{
    if (!data_)
        return end();
    const auto found = std::lower_bound(data_->payload->keys.begin(), data_->payload->keys.end(), gid,
        [](const adaptive_detail::NativePublicationKeyRoute& row, std::uint64_t value) { return row.gid < value; });
    return found == data_->payload->keys.end() || found->gid != gid
               ? end()
               : const_iterator(data_, std::size_t(found - data_->payload->keys.begin()));
}
LocalNativeKeysView::const_iterator LocalNativeKeysView::begin() const
{
    return {data_, 0};
}
LocalNativeKeysView::const_iterator LocalNativeKeysView::end() const
{
    return {data_, size()};
}

const std::shared_ptr<const CompositePartition<>>& AdaptiveNativePublicationView::partition() const
{
    if (!data_)
        throw std::logic_error("Empty native publication has no partition.");
    return data_->payload->partition;
}
const std::shared_ptr<const MeshHandle<>>& AdaptiveNativePublicationView::mesh() const
{
    if (!data_)
        throw std::logic_error("Empty native publication has no mesh.");
    return data_->payload->mesh;
}
MeshStorageReport AdaptiveNativePublicationView::storage() const
{
    return data_ && data_->compatibility ? data_->compatibility->storage :
        (data_ && data_->payload ? data_->payload->summary.report() : MeshStorageReport{});
}
AdaptiveNativePublicationView::KeyRegistryStorage AdaptiveNativePublicationView::key_registry_storage() const noexcept
{
    if (!data_ || !data_->payload)
        return {};
    return {data_->payload->keys.capacity() * sizeof(adaptive_detail::NativePublicationKeyRoute),
        bool(data_->compatibility)};
}

AdaptiveNativePublication export_adaptive_native_publication(
    const AdaptiveNativePublicationView& view, std::size_t byte_budget)
{
    if (!view.data_ || !byte_budget)
        throw std::invalid_argument("Native owning export requires an accepted publication and positive budget.");
    adaptive_detail::Budget budget{byte_budget};
    budget.add(view.data_->resident_bytes());
    AdaptiveNativePublication result;
    result.partition = view.partition();
    result.mesh = view.mesh();
    result.storage = view.storage();
    const auto key_bytes = [](const AdaptiveCellKey& key)
    {
        return std::visit(
            [](const auto& ancestry) -> std::size_t
            {
                if constexpr (requires { ancestry.path; })
                    return ancestry.path.size() * sizeof(std::uint64_t);
                else if constexpr (requires { ancestry.base.path; })
                    return (ancestry.base.path.size() + ancestry.axial.path.size()) * sizeof(std::uint64_t);
                return 0;
            },
            key.ancestry);
    };
    for (const auto [key, gid] : view.owned_gids())
    {
        budget.add(sizeof(decltype(result.owned_gids)::value_type) + 4 * sizeof(void*) + key_bytes(key));
        if (!result.owned_gids.emplace(key, gid).second)
            throw std::logic_error("Native owning export repeats an owned key.");
    }
    for (const auto [gid, key] : view.local_keys())
    {
        budget.add(sizeof(decltype(result.local_keys)::value_type) + 4 * sizeof(void*) + key_bytes(key));
        if (!result.local_keys.emplace(gid, key).second)
            throw std::logic_error("Native owning export repeats a local GID.");
    }
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
