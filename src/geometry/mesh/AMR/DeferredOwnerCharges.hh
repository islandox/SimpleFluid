#pragma once
#include "ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Owns already-admitted credits for retired immutable bodies that another
 * source or local alias still retains. Weak ownership observes physical expiry
 * without keeping geometry or an old session alive. Reap at operation entry
 * and after retiring staged owners, not once per streamed record. Until reap,
 * live admission is conservative. Cumulative admissions are never refunded.
 *
 * adopt transfers a credit, not a strong owner: on its first true return the
 * caller removes bytes from its own component counter without changing
 * Budget::used. Bodies with a same-arena self-releasing deleter must not be
 * transferred. Every alias must consult this table before immediate release. */
class DeferredOwnerCharges
{
    using Owner = std::weak_ptr<const void>;
    using Entries = std::map<Owner, std::size_t, std::owner_less<>>;
    static constexpr std::size_t node_bytes = sizeof(Entries::value_type) + 4 * sizeof(void*);

public:
    explicit DeferredOwnerCharges(Budget& budget) noexcept : budget_(&budget) {}
    ~DeferredOwnerCharges()
    {
        // The entire admission domain is retiring. Its surviving owners, if
        // any, have been independently admitted in their other domains.
        entries_.clear();
        budget_->used -= body_bytes_ + metadata_bytes_;
    }
    DeferredOwnerCharges(const DeferredOwnerCharges&) = delete;
    DeferredOwnerCharges& operator=(const DeferredOwnerCharges&) = delete;

    template<class T> [[nodiscard]] bool contains(const std::shared_ptr<T>& owner) const noexcept
    {
        return entries_.find(Owner(owner)) != entries_.end();
    }
    template<class T> bool adopt(const std::shared_ptr<T>& owner, std::size_t bytes)
    {
        if (!owner || !bytes || bytes > budget_->used)
            throw std::invalid_argument("Invalid deferred immutable-owner credit.");
        const Owner weak(owner);
        if (const auto found = entries_.find(weak); found != entries_.end())
        {
            if (found->second != bytes)
                throw std::invalid_argument("Conflicting deferred immutable-owner credit.");
            return false;
        }
        if (bytes > std::numeric_limits<std::size_t>::max() - body_bytes_ ||
            node_bytes > std::numeric_limits<std::size_t>::max() - metadata_bytes_)
            throw std::invalid_argument("Deferred immutable-owner credit overflows.");
        budget_->add(node_bytes);
        try
        {
            entries_.emplace(weak, bytes);
        }
        catch (...)
        {
            budget_->used -= node_bytes;
            throw;
        }
        body_bytes_ += bytes;
        metadata_bytes_ += node_bytes;
        return true;
    }
    void reap() noexcept
    {
        for (auto position = entries_.begin(); position != entries_.end();)
        {
            if (!position->first.expired())
            {
                ++position;
                continue;
            }
            const auto bytes = position->second;
            // Destroy the final weak reference before returning its credit:
            // allocate_shared may keep inline storage in the control block.
            position = entries_.erase(position);
            body_bytes_ -= bytes;
            metadata_bytes_ -= node_bytes;
            budget_->used -= bytes + node_bytes;
        }
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return metadata_bytes_; }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return body_bytes_ + metadata_bytes_; }
    [[nodiscard]] std::size_t body_bytes() const noexcept { return body_bytes_; }
    [[nodiscard]] std::size_t owners() const noexcept { return entries_.size(); }

    /** Weak entries are value payload; their targets are never promoted. */
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, ContactPayloadRole::DeferredOwnerChargesObject, sizeof(*this));
        for (const auto& value : entries_)
            visitor(ContactPayloadAllocation{{&value, ContactPayloadRole::DeferredOwnerChargeNodes}, this,
                ContactPayloadKind::NodeValue, sizeof(value), sizeof(value), 1, 1, sizeof(value), true, true, true});
    }

private:
    Budget* budget_;
    Entries entries_;
    std::size_t body_bytes_ = 0, metadata_bytes_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
