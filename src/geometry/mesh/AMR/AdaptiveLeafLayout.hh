/** @file AdaptiveLeafLayout.hh
 * @brief Immutable owner-local active leaves of a distributed AMR generation.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"

namespace SimpleFluid::Meshes::AMR
{
class DistributedAdaptiveHierarchy;
namespace adaptive_detail
{
class ContactExternalResidencyLease;
}

/** Authoritative typed active-leaf records in local key order.
 * A shared owner pins the exact layout, including ancestry, revisions and
 * provider recipes. The layout lifetime is independent of hierarchy and halo
 * lifetimes. Borrowed leaves remain valid while that owner is held. */
class AdaptiveLeafLayout
{
public:
    using const_iterator = adaptive_detail::AdaptiveLeafPages::const_iterator;

    AdaptiveLeafLayout(const AdaptiveLeafLayout&) = delete;
    AdaptiveLeafLayout& operator=(const AdaptiveLeafLayout&) = delete;

    [[nodiscard]] const_iterator begin() const noexcept { return pages_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return pages_.end(); }
    [[nodiscard]] std::size_t size() const noexcept { return pages_.size(); }
    [[nodiscard]] bool empty() const noexcept { return pages_.empty(); }
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const { return pages_.contains(key); }
    [[nodiscard]] const AdaptiveLeaf* find(const AdaptiveCellKey& key) const { return pages_.find(key); }
    [[nodiscard]] const AdaptiveLeaf& at(const AdaptiveCellKey& key) const { return pages_.at(key); }
    [[nodiscard]] std::optional<std::size_t> ordinal_of(const AdaptiveCellKey& key) const { return pages_.ordinal_of(key); }
    [[nodiscard]] const AdaptiveLeaf& at_ordinal(std::size_t ordinal) const { return pages_.at_ordinal(ordinal); }

private:
    friend class DistributedAdaptiveHierarchy;

    AdaptiveLeafLayout(adaptive_detail::AdaptiveLeafPages pages,
        std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission)
        : admission_(std::move(admission)), pages_(std::move(pages))
    {
    }

    // Destroy page owners before releasing their hierarchy residency charge.
    const std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission_;
    const adaptive_detail::AdaptiveLeafPages pages_;
};
} // namespace SimpleFluid::Meshes::AMR
