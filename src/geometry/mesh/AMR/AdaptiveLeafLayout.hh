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

class AdaptiveLeafLayout;
/** Opaque complete page transition issued only by the hierarchy after a
 * successful edit and exact selection validation. No caller row list can
 * omit an unchanged record on a copied page. The witness is operation-local. */
class AdaptiveLeafTransition
{
public:
    [[nodiscard]] const std::shared_ptr<const AdaptiveLeafLayout>& source() const noexcept { return source_; }
    [[nodiscard]] const std::shared_ptr<const AdaptiveLeafLayout>& target() const noexcept { return target_; }
    void visit_rebound_leaves(const std::function<void(const AdaptiveLeaf&)>&) const;

private:
    friend class DistributedAdaptiveHierarchy;
    AdaptiveLeafTransition(std::shared_ptr<const AdaptiveLeafLayout> source,
        std::shared_ptr<const AdaptiveLeafLayout> target, std::shared_ptr<adaptive_detail::Budget> arena)
        : source_(std::move(source)), target_(std::move(target)), arena_(std::move(arena)) {}
    std::shared_ptr<const AdaptiveLeafLayout> source_, target_;
    std::shared_ptr<adaptive_detail::Budget> arena_;
};
/** Authoritative typed active-leaf records in local key order.
 * A shared owner pins the exact layout, including ancestry, revisions and
 * provider recipes. The layout lifetime is independent of hierarchy and halo
 * lifetimes. Borrowed leaves remain valid while that owner is held. */
class AdaptiveLeafLayout
{
    class NativePreparedKey
    {
        friend class DistributedAdaptiveHierarchy;
        NativePreparedKey() = default;
    public:
        NativePreparedKey(const NativePreparedKey&) = default;
    };
public:
    using const_iterator = adaptive_detail::AdaptiveLeafPages::const_iterator;

    AdaptiveLeafLayout(NativePreparedKey, adaptive_detail::AdaptiveLeafPages pages,
        std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission)
        : AdaptiveLeafLayout(std::move(pages), std::move(admission))
    {
    }
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
    friend class AdaptiveLeafTransition;

    AdaptiveLeafLayout(adaptive_detail::AdaptiveLeafPages pages,
        std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission)
        : admission_(std::move(admission)), pages_(std::move(pages))
    {
    }

    // Destroy page owners before releasing their hierarchy residency charge.
    const std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission_;
    const adaptive_detail::AdaptiveLeafPages pages_;
};
inline void AdaptiveLeafTransition::visit_rebound_leaves(
    const std::function<void(const AdaptiveLeaf&)>& visitor) const
{
    target_->pages_.visit_created_leaf_pages(*arena_, [&](std::span<const AdaptiveLeaf> rows)
        {
            for (const auto& leaf : rows)
                visitor(leaf);
        });
}
} // namespace SimpleFluid::Meshes::AMR
