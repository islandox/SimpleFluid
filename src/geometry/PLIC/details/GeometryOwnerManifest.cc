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
    /** The single admitted row of a one-owner set, held inline. A singleton set
     * needs no index page; the row and its multiplicity live with the state. */
    OwnerRecord single;
    explicit State(Pages::Ptr value) : root(std::move(value)) {}
    explicit State(OwnerRecord value) : single(std::move(value)) {}
    [[nodiscard]] bool inline_row() const noexcept { return !root; }
};
void GeometryOwnerManifest::visit_dependencies(MetadataSink metadata, RecordSink records) const
{
    if (!state_ || !metadata(state_, sizeof(State), sizeof(State) + control_bytes))
        return;
    if (state_->inline_row())
        records(state_->single);
    else
        Pages::visit_dependencies(state_->root, metadata, records);
}
void GeometryOwnerManifest::visit_borrowed_dependencies(
    BorrowedMetadataSink metadata, BorrowedRecordSink records, const void* parent) const
{
    if (!state_ || !metadata(state_.get(), parent, sizeof(State)))
        return;
    if (state_->inline_row())
        records(state_->single, state_.get());
    else
        Pages::visit_borrowed_dependencies(state_->root, metadata, records, state_.get());
}
GeometryOwnerManifest GeometryOwnerManifest::built(
    std::span<const OwnerRecord> rows, Admission admission, PageWork* work)
{
    GeometryOwnerManifest result;
    if (rows.empty())
        return result;
    for (const auto& row : rows)
    {
        (void) lookup(row.key.channel, row.key.kind, row.owner);
        if (!row.references || row.bytes == std::numeric_limits<std::size_t>::max())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner set has an unusable row.");
        if (row.key.address != row.owner.get())
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner row address differs from its control.");
    }
    for (std::size_t index = 1; index < rows.size(); ++index)
        if (!OwnerLess{}(rows[index - 1], rows[index]))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner set rows are not in strict canonical order.");
    admission.charge(1, sizeof(State) + control_bytes);
    if (rows.size() == 1)
        result.state_ = std::make_shared<const State>(rows[0]);
    else
        result.state_ = std::make_shared<const State>(Pages(admission, work).build(rows));
    return result;
}
GeometryOwnerManifest GeometryOwnerManifest::retained(std::uint8_t channel, std::uint8_t kind,
    std::shared_ptr<const void> owner, std::size_t bytes, Admission admission, PageWork* work) const
{
    return retained(channel, kind, std::move(owner), bytes, 1, admission, work);
}
GeometryOwnerManifest GeometryOwnerManifest::retained(std::uint8_t channel, std::uint8_t kind,
    std::shared_ptr<const void> owner, std::size_t bytes, std::size_t references, Admission admission,
    PageWork* work) const
{
    if (!references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner admission lacks a reference.");
    auto row = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        const auto& accepted = state_->single;
        const bool same = !OwnerLess{}(accepted, row) && !OwnerLess{}(row, accepted);
        GeometryOwnerManifest result;
        if (same)
        {
            if (!same_owner(accepted.owner, row.owner))
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
            if (accepted.bytes != bytes || accepted.references > std::numeric_limits<std::size_t>::max() - references)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Geometry owner admission changes its immutable size.");
            row = accepted;
            row.bytes = bytes;
            row.references += references;
            admission.charge(1, sizeof(State) + control_bytes);
            result.state_ = std::make_shared<const State>(std::move(row));
            return result;
        }
        // A second distinct owner needs the indexed representation.
        row.bytes = bytes;
        row.references = references;
        std::array<OwnerRecord, 2> rows{accepted, row};
        if (OwnerLess{}(rows[1], rows[0]))
            std::swap(rows[0], rows[1]);
        admission.charge(1, sizeof(State) + control_bytes);
        result.state_ = std::make_shared<const State>(Pages(admission, work).build(rows));
        return result;
    }
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (accepted)
    {
        if (accepted->bytes != bytes || accepted->references > std::numeric_limits<std::size_t>::max() - references)
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner admission changes its immutable size.");
        row = *accepted;
    }
    row.bytes = bytes;
    row.references += references;
    auto root = Pages(admission, work).edit(state_ ? state_->root : nullptr, row, &row, accepted != nullptr);
    admission.charge(1, sizeof(State) + control_bytes);
    GeometryOwnerManifest result;
    result.state_ = std::make_shared<const State>(std::move(root));
    return result;
}
GeometryOwnerManifest GeometryOwnerManifest::retired(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner, Admission admission, PageWork* work) const
{
    return retired(channel, kind, owner, 1, admission, work);
}
GeometryOwnerManifest GeometryOwnerManifest::retired(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t references, Admission admission, PageWork* work) const
{
    if (!references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner lacks a reference.");
    auto row = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        const auto& accepted = state_->single;
        if (OwnerLess{}(accepted, row) || OwnerLess{}(row, accepted))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
        if (!same_owner(accepted.owner, row.owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner has foreign controls.");
        if (accepted.references < references)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
        row = accepted;
        row.references -= references;
        GeometryOwnerManifest result;
        if (row.references)
        {
            admission.charge(1, sizeof(State) + control_bytes);
            result.state_ = std::make_shared<const State>(std::move(row));
        }
        return result;
    }
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (!accepted || accepted->references < references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
    row = *accepted;
    row.references -= references;
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
    const auto wanted = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        if (OwnerLess{}(state_->single, wanted) || OwnerLess{}(wanted, state_->single))
            return false;
        if (!same_owner(state_->single.owner, wanted.owner))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
        return state_->single.bytes == bytes;
    }
    const auto* row = state_ ? find(state_->root, wanted) : nullptr;
    return row && row->bytes == bytes;
}
std::size_t GeometryOwnerManifest::references(
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner) const
{
    const auto wanted = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        if (OwnerLess{}(state_->single, wanted) || OwnerLess{}(wanted, state_->single))
            return 0;
        if (!same_owner(state_->single.owner, wanted.owner))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
        return state_->single.references;
    }
    const auto* row = state_ ? find(state_->root, wanted) : nullptr;
    return row ? row->references : 0;
}
std::optional<std::uint8_t> GeometryOwnerManifest::uniform_channel() const
{
    if (!state_)
        return std::nullopt;
    if (state_->inline_row())
        return state_->single.key.channel;
    const auto first = Pages::lower_bound(state_->root, [](const auto&) { return false; });
    if (first->key.channel != state_->root->maximum->key.channel)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner manifest mixes arithmetic channels.");
    return first->key.channel;
}
std::size_t GeometryOwnerManifest::payload_bytes() const noexcept
{
    if (!state_)
        return 0;
    return state_->inline_row() ? state_->single.bytes : state_->root->subtree_owner_bytes;
}
std::size_t GeometryOwnerManifest::metadata_admission_bytes() const noexcept
{
    if (!state_)
        return 0;
    return sizeof(State) + control_bytes + (state_->inline_row() ? 0 : state_->root->subtree_bytes);
}
std::size_t GeometryOwnerManifest::metadata_payload_bytes() const noexcept
{
    if (!state_)
        return 0;
    return sizeof(State) + (state_->inline_row() ? 0 : state_->root->subtree_payload_bytes);
}
std::size_t GeometryOwnerManifest::owner_count() const noexcept
{
    if (!state_)
        return 0;
    return state_->inline_row() ? 1 : state_->root->size;
}
void GeometryOwnerManifest::visit(const std::function<void(const OwnerRecord&)>& visitor) const
{
    if (state_ && state_->inline_row())
    {
        visitor(state_->single);
        return;
    }
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
    if (state_->inline_row())
        return;
    Pages::visit_pages(state_->root,
        [&](const auto& page)
        {
            visitor(page.get(), sizeof(*page) + page->rows.capacity() * sizeof(OwnerRecord) +
                                    page->children.capacity() * sizeof(Pages::Ptr));
        });
}
} // namespace SimpleFluid::PLIC::details::geometry_demand
