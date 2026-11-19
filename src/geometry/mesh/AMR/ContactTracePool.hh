#pragma once
#include "ContactDomainStorage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Exact ordered, point-independent trace inventories. Each arena admits its
 * immutable owners independently; records retain owners without a pool link. */
class ContactTracePool
{
public:
    explicit ContactTracePool(Budget&);
    ContactTracePool(const ContactTracePool&, Budget&);
    ~ContactTracePool();
    ContactTracePool(const ContactTracePool&) = delete;
    ContactTracePool& operator=(const ContactTracePool&) = delete;
    /** Transfer the already-admitted owned trace buffers out of a record body.
     * Returns the charge to subtract from that body's credit. Restricted face
     * inventories remain owned, and the record's point-independent flag is
     * unchanged. A record must not have been adopted previously. */
    [[nodiscard]] std::size_t adopt(ContactBody&);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return charged_; }

private:
    using Inventory = std::vector<AdaptiveTraceFace>;
    using Owner = std::shared_ptr<const Inventory>;
    struct Less
    {
        using is_transparent = void;
        bool operator()(const Owner&, const Owner&) const noexcept;
        bool operator()(const Owner&, std::span<const AdaptiveTraceFace>) const noexcept;
        bool operator()(std::span<const AdaptiveTraceFace>, const Owner&) const noexcept;
    };
    using Lookup = ContactMap<Owner, std::size_t, Less>;
    [[nodiscard]] static constexpr std::size_t node_bytes() noexcept
    {
        return sizeof(Lookup::value_type) + 4 * sizeof(void*);
    }
    void admit(std::size_t);
    void clear() noexcept;
    Budget* budget_;
    std::size_t charged_ = 0;
    Lookup inventories_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
