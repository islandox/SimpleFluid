/** @file DistributedAncestryDirectory.hh
 * @brief Exact sharded owner authority for adaptive cells and saved families.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <map>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveOwner
{
    AdaptiveBytes key;
    int owner = -1;
    std::uint64_t revision = 0;
    std::uint64_t address = std::numeric_limits<std::uint64_t>::max(); ///< Observed GID, never ancestry.
};
class DistributedAncestryDirectory
{
public:
    /** Every owned object registers once. No rank receives the global key set.
     * Collective duplicate registration and budget failures reject all ranks. */
    DistributedAncestryDirectory(std::span<const AdaptiveOwner> owned,
        Teuchos::RCP<const Teuchos::Comm<int>> communicator, std::size_t byte_budget);
    /** Collective sparse requests. Output preserves local request order. */
    [[nodiscard]] std::vector<AdaptiveOwner> lookup(std::span<const AdaptiveBytes> keys, bool require_all = true) const;
    [[nodiscard]] std::size_t local_entries() const noexcept { return shard_.size(); }
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] const OwnerExchangeStats& registration_stats() const noexcept { return registration_; }

private:
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    std::size_t budget_;
    std::map<AdaptiveBytes, AdaptiveOwner> shard_;
    OwnerExchangeStats registration_;
};
} // namespace SimpleFluid::Meshes::AMR
