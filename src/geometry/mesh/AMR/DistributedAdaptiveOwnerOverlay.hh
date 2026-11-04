/** Candidate-private sparse owner authority over the immutable accepted directory. */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveOwnerOverlay
{
public:
    // The result owns its budget charge until destruction or move. The session
    // and its Budget must outlive every lookup lease.
    class Lookup
    {
    public:
        Lookup() = default;
        Lookup(const Lookup&) = delete;
        Lookup& operator=(const Lookup&) = delete;
        Lookup(Lookup&& other) noexcept;
        Lookup& operator=(Lookup&&) = delete;
        ~Lookup();
        [[nodiscard]] const std::vector<AdaptiveOwner>& values() const noexcept { return values_; }

    private:
        friend class DistributedAdaptiveOwnerOverlay;
        Budget* budget_ = nullptr;
        std::size_t charge_ = 0;
        std::vector<AdaptiveOwner> values_;
    };
    DistributedAdaptiveOwnerOverlay(
        const DistributedAncestryDirectory& base, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& shared_budget);
    ~DistributedAdaptiveOwnerOverlay();
    DistributedAdaptiveOwnerOverlay(const DistributedAdaptiveOwnerOverlay&) = delete;
    DistributedAdaptiveOwnerOverlay& operator=(const DistributedAdaptiveOwnerOverlay&) = delete;
    // Retired and inserted objects are locally owned. Retirement revisions must
    // match; insertion requires absence after the delta's retirements. Entire
    // delta is staged, voted, and then published without allocation.
    void apply(std::span<const AdaptiveOwner> retired, std::span<const AdaptiveOwner> inserted);
    [[nodiscard]] Lookup lookup(std::span<const AdaptiveBytes> keys, bool require_all = true) const;

private:
    struct Entry
    {
        bool active;
        AdaptiveOwner value;
    };
    const DistributedAncestryDirectory& base_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    Budget& budget_;
    std::map<AdaptiveBytes, Entry> shard_;
    std::size_t retained_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
