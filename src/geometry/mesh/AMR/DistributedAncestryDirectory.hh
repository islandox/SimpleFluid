/** @file DistributedAncestryDirectory.hh
 * @brief Exact sharded owner authority for adaptive cells and saved families.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <map>
#include <optional>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveOwner
{
    AdaptiveBytes key;
    int owner = -1;
    std::uint64_t revision = 0;
    std::uint64_t address = std::numeric_limits<std::uint64_t>::max(); ///< Observed GID, never ancestry.
};
class DistributedAdaptiveHierarchy;
namespace adaptive_detail
{
class DirectoryRegistrationAccess;
class DirectorySourceCredit;
class FamilyDirectoryProjection;
} // namespace adaptive_detail
class DistributedAncestryDirectory
{
public:
    /** Every owned object registers once. No rank receives the global key set.
     * Collective duplicate registration and budget failures reject all ranks. */
    DistributedAncestryDirectory(std::span<const AdaptiveOwner> owned,
        Teuchos::RCP<const Teuchos::Comm<int>> communicator, std::size_t byte_budget);
    /** Explicit collective transient lease; the configured lookup budget remains unchanged. */
    DistributedAncestryDirectory(std::span<const AdaptiveOwner> owned,
        Teuchos::RCP<const Teuchos::Comm<int>> communicator, std::size_t byte_budget,
        std::optional<std::size_t> registration_transient_cap);
    DistributedAncestryDirectory(const DistributedAncestryDirectory&);
    DistributedAncestryDirectory& operator=(const DistributedAncestryDirectory&);
    DistributedAncestryDirectory(DistributedAncestryDirectory&&);
    DistributedAncestryDirectory& operator=(DistributedAncestryDirectory&&);
    ~DistributedAncestryDirectory();
    /** Collective sparse requests. Output preserves local request order.
     * An explicit transient cap partitions live packet/result arenas, cannot
     * exceed the configured budget and rejects collectively on exhaustion.
     * Omitting it retains the legacy service behavior. */
    [[nodiscard]] std::vector<AdaptiveOwner> lookup(std::span<const AdaptiveBytes> keys, bool require_all = true,
        std::optional<std::size_t> transient_cap = {}) const;
    [[nodiscard]] std::size_t local_entries() const noexcept { return entries_; }
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] const OwnerExchangeStats& registration_stats() const noexcept { return registration_; }

private:
    friend class adaptive_detail::DirectoryRegistrationAccess;
    friend class adaptive_detail::DirectorySourceCredit;
    struct Storage;
    struct SmallShard;
    struct Value
    {
        int owner;
        std::uint64_t revision, address;
    };
    DistributedAncestryDirectory(std::span<const AdaptiveOwner>, Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t,
        std::optional<std::size_t>, const adaptive_detail::DirectorySourceCredit*,
        const adaptive_detail::FamilyDirectoryProjection*, bool persistent_registration);
    [[nodiscard]] std::optional<Value> find(std::span<const char>) const noexcept;

    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    std::size_t budget_, entries_ = 0, resident_bytes_ = 0;
    std::shared_ptr<Storage> storage_;
    std::unique_ptr<SmallShard> small_;
    OwnerExchangeStats registration_;
};
} // namespace SimpleFluid::Meshes::AMR
