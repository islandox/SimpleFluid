/** @file AdaptiveLeafLayout.hh
 * @brief Immutable owner-local active leaves of a distributed AMR generation.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"

#include <map>
#include <memory>
#include <utility>

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
    using LeafMap = std::map<AdaptiveCellKey, AdaptiveLeaf>;

    AdaptiveLeafLayout(const AdaptiveLeafLayout&) = delete;
    AdaptiveLeafLayout& operator=(const AdaptiveLeafLayout&) = delete;

    [[nodiscard]] const LeafMap& owned() const noexcept { return owned_; }
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const { return owned_.contains(key); }
    [[nodiscard]] const AdaptiveLeaf* find(const AdaptiveCellKey& key) const
    {
        const auto leaf = owned_.find(key);
        return leaf == owned_.end() ? nullptr : &leaf->second;
    }

private:
    friend class DistributedAdaptiveHierarchy;

    AdaptiveLeafLayout(LeafMap&& owned, std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission)
        : admission_(std::move(admission)), owned_(std::move(owned))
    {
    }

    // Destroy the map before releasing its charge.
    const std::shared_ptr<const adaptive_detail::ContactExternalResidencyLease> admission_;
    const LeafMap owned_;
};
} // namespace SimpleFluid::Meshes::AMR
