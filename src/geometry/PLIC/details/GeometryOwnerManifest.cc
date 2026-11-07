/** @file GeometryOwnerManifest.cc
 * @brief Compiled persistent exact owner bindings and first/last reference accounting.
 */
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include <limits>

namespace SimpleFluid::PLIC::details::geometry_demand
{
namespace
{
struct OwnerLess
{
    bool operator()(const OwnerRecord& a, const OwnerRecord& b) const noexcept
    {
        if (a.key.channel != b.key.channel)
            return a.key.channel < b.key.channel;
        if (a.key.kind != b.key.kind)
            return a.key.kind < b.key.kind;
        return std::less<const void*>{}(a.key.address, b.key.address);
    }
};
using Pages = OrderedPages<OwnerRecord, OwnerLess>;
OwnerRecord lookup(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner)
{
    const std::shared_ptr<const void> empty;
    if (!owner || (!owner.owner_before(empty) && !empty.owner_before(owner)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry manifest requires an actual owner control.");
    return {{channel, kind, owner.get()}, owner, 0, 0};
}
bool same_owner(const std::shared_ptr<const void>& first, const std::shared_ptr<const void>& second) noexcept
{
    return first.get() == second.get() && !first.owner_before(second) && !second.owner_before(first);
}
const OwnerRecord* find(const Pages::Ptr& root, const OwnerRecord& key, PageWork* work = nullptr)
{
    const auto* row = Pages::lower_bound(root, [&](const auto& value) { return OwnerLess{}(value, key); }, work);
    if (!row || OwnerLess{}(key, *row))
        return nullptr;
    if (!same_owner(row->owner, key.owner))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
    return row;
}
} // namespace
struct GeometryOwnerManifest::State
{
    Pages::Ptr root;
    explicit State(Pages::Ptr value) : root(std::move(value)) {}
};
GeometryOwnerManifest GeometryOwnerManifest::retained(std::uint8_t channel, std::uint8_t kind,
    std::shared_ptr<const void> owner, std::size_t bytes, Admission admission, PageWork* work) const
{
    auto row = lookup(channel, kind, owner);
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (accepted)
    {
        if (accepted->bytes != bytes || accepted->references == std::numeric_limits<std::size_t>::max())
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner admission changes its immutable size.");
        row = *accepted;
    }
    row.bytes = bytes;
    ++row.references;
    auto root = Pages(admission, work).edit(state_ ? state_->root : nullptr, row, &row, accepted != nullptr);
    admission.charge(1, sizeof(State) + control_bytes);
    GeometryOwnerManifest result;
    result.state_ = std::make_shared<const State>(std::move(root));
    return result;
}
GeometryOwnerManifest GeometryOwnerManifest::retired(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner, Admission admission, PageWork* work) const
{
    auto row = lookup(channel, kind, owner);
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (!accepted || !accepted->references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
    row = *accepted;
    --row.references;
    auto root = Pages(admission, work).edit(state_->root, row, row.references ? &row : nullptr, true);
    GeometryOwnerManifest result;
    if (root)
    {
        admission.charge(1, sizeof(State) + control_bytes);
        result.state_ = std::make_shared<const State>(std::move(root));
    }
    return result;
}
bool GeometryOwnerManifest::admits(
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    const auto* row = state_ ? find(state_->root, lookup(channel, kind, owner)) : nullptr;
    return row && row->bytes == bytes;
}
std::size_t GeometryOwnerManifest::references(
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner) const
{
    const auto* row = state_ ? find(state_->root, lookup(channel, kind, owner)) : nullptr;
    return row ? row->references : 0;
}
std::size_t GeometryOwnerManifest::payload_bytes() const noexcept
{
    return state_ ? state_->root->subtree_owner_bytes : 0;
}
std::size_t GeometryOwnerManifest::metadata_admission_bytes() const noexcept
{
    return state_ ? sizeof(State) + control_bytes + state_->root->subtree_bytes : 0;
}
std::size_t GeometryOwnerManifest::metadata_payload_bytes() const noexcept
{
    return state_ ? sizeof(State) + state_->root->subtree_payload_bytes : 0;
}
std::size_t GeometryOwnerManifest::owner_count() const noexcept
{
    return state_ ? state_->root->size : 0;
}
void GeometryOwnerManifest::visit(const std::function<void(const OwnerRecord&)>& visitor) const
{
    if (state_)
        Pages::visit_from(
            state_->root, [](const auto&) { return false; },
            [&](const auto& row)
            {
                visitor(row);
                return true;
            });
}
void GeometryOwnerManifest::visit_metadata(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (!state_)
        return;
    visitor(state_.get(), sizeof(State));
    Pages::visit_pages(state_->root,
        [&](const auto& page)
        {
            visitor(page.get(), sizeof(*page) + page->rows.capacity() * sizeof(OwnerRecord) +
                                    page->children.capacity() * sizeof(Pages::Ptr));
        });
}
} // namespace SimpleFluid::PLIC::details::geometry_demand
