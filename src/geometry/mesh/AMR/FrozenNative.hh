/** @file FrozenNative.hh
 * @brief Detached accepted-native copy contracts for ownership-only migration.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include <map>
namespace SimpleFluid::Meshes::AMR
{
class FrozenNativeModel;
class FrozenNativeAccess;
struct FrozenNativeBootstrap;
struct FrozenNativeOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024, maximum_owned = 65536, maximum_imported = 262144;
};
/** An exporter-created witness of the accepted native source. Its implementation
 * retains compact metadata and weak original observations, never an old
 * hierarchy, global mesh, provider catalog or source Data pointer. */
class FrozenNativeRoot
{
public:
    struct Data;
    struct SourceObservation
    {
        std::weak_ptr<const void> identity, topology_identity, partition_identity;
        std::uint64_t topology = 0, partition = 0, geometry = 0;
        const void* native_geometry = nullptr;
        const void* communicator = nullptr;
    };
    [[nodiscard]] const SourceObservation& source_observation() const noexcept;
    [[nodiscard]] std::span<const char> schema() const noexcept;

private:
    friend class FrozenNativeAccess;
    friend class FrozenNativeModel;
    explicit FrozenNativeRoot(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
/** Opaque native records copied under the root's accepted-source witness.
 * Procedural recipes retain their original family; explicit records retain
 * actual loops and metrics. This is not a convex or exact-planar certificate.
 * There is deliberately no public constructor or raw packet decoder. */
class FrozenNativeBundle
{
public:
    struct Data;
    [[nodiscard]] const AdaptiveCellKey& key() const noexcept;
    [[nodiscard]] std::uint64_t revision() const noexcept;
    [[nodiscard]] std::uint64_t material() const noexcept;
    [[nodiscard]] std::uint64_t source_gid() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const;
    /** Actual standalone record packet size. Shared peer-batch dictionaries and
     * framing can change migration cost; this is a per-record work sample. */
    [[nodiscard]] std::size_t encoded_payload_bytes() const;

private:
    friend class FrozenNativeAccess;
    friend class FrozenNativeModel;
    explicit FrozenNativeBundle(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
struct FrozenNativeBootstrap
{
    std::shared_ptr<const FrozenNativeRoot> root;
    std::vector<FrozenNativeBundle> owned;
    std::vector<AdaptiveSourceCell> source_cells;
};
struct FrozenNativePublication
{
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const MeshHandle<>> mesh;
    std::map<AdaptiveCellKey, std::uint64_t> owned_gids;
    std::map<std::uint64_t, AdaptiveCellKey> local_keys;
    MeshStorageReport storage;
};
/** Collectively snapshot only selected owned records from an accepted hierarchy.
 * The exporter observes the actual source before and after copying and creates
 * the witness itself. A caller-supplied schema is not an acceptance proof.
 * A nonidentity accepted axial frame is explicitly unsupported by this path.
 * Existing periodic native incidence and image vectors remain part of the copy.
 */
[[nodiscard]] FrozenNativeBootstrap detach_frozen_native(const AdaptiveHierarchy&,
    std::span<const std::uint64_t> owned_source_gids, std::span<const char> root_schema,
    Teuchos::RCP<const Teuchos::Comm<int>>, FrozenNativeOptions = {});
/** Entire-model ownership-only boundary when any participant lacks the exact
 * adaptation certificate contract. No topology-changing subset exception is
 * permitted. Named state/history ownership copies are owned by the state layer.
 */
class FrozenNativeModel
{
public:
    struct Data;
    struct Storage
    {
        std::size_t owned_cells = 0, imported_cells = 0, metadata_bytes = 0, procedural_bytes = 0, explicit_bytes = 0;
        std::size_t procedural_fragments = 0, shared_native_parts = 0, canonical_routes = 0, routing_bytes = 0,
                    explicit_payload_bytes = 0;
        // Capacities estimate retained allocations; explicit_payload_bytes is
        // serialized payload, separately from routes, maps and process RSS.
    };
    FrozenNativeModel(FrozenNativeBootstrap, Teuchos::RCP<const Teuchos::Comm<int>>, FrozenNativeOptions = {});
    [[nodiscard]] const std::map<AdaptiveCellKey, FrozenNativeBundle>& owned() const noexcept;
    [[nodiscard]] const std::shared_ptr<const void>& identity() const noexcept;
    [[nodiscard]] std::uint64_t topology_generation() const noexcept;
    [[nodiscard]] std::uint64_t partition_generation() const noexcept;
    [[nodiscard]] Storage storage() const;
    [[nodiscard]] Teuchos::RCP<const Teuchos::Comm<int>> communicator() const noexcept;
    [[nodiscard]] FrozenNativeOptions options() const noexcept;
    [[nodiscard]] std::span<const char> configuration_descriptor() const noexcept;
    [[nodiscard]] std::uint64_t global_cells() const noexcept;
    [[nodiscard]] std::vector<AdaptiveOwner> lookup_cells(std::span<const AdaptiveCellKey>) const;
    /** Already encoded immutable local packets, prepared without ownership
     * exchange. Byte getters describe a proposal, not paid transfer traffic. */
    class PreparedMigration
    {
    public:
        PreparedMigration(PreparedMigration&&) noexcept = default;
        PreparedMigration(const PreparedMigration&) = delete;
        [[nodiscard]] std::uint64_t local_application_bytes() const noexcept { return local_bytes_; }
        [[nodiscard]] std::uint64_t application_bytes() const noexcept { return global_bytes_; }

    private:
        friend class FrozenNativeModel;
        PreparedMigration() = default;
        std::shared_ptr<const void> source_;
        std::uint64_t topology_ = 0, partition_ = 0, transaction_ = 0, local_bytes_ = 0, global_bytes_ = 0,
                      global_changed_ = 0;
        std::vector<OwnerPacket> packets_;
    };
    [[nodiscard]] PreparedMigration prepare_migration(std::span<const int> destinations) const;
    class Plan
    {
    public:
        [[nodiscard]] const std::shared_ptr<const FrozenNativeModel>& candidate() const noexcept { return candidate_; }
        /** Global sum of owner-migration application packets and framing. */
        [[nodiscard]] std::uint64_t application_bytes() const noexcept { return application_bytes_; }
        /** This rank's contribution; combine with local state bytes before a global sum. */
        [[nodiscard]] std::uint64_t local_application_bytes() const noexcept { return local_application_bytes_; }
        /** Owner-exchange packet scratch only; excludes source/candidate/halos and allocator RSS. */
        [[nodiscard]] std::optional<std::size_t> native_packet_phase_peak_bytes() const noexcept
        {
            return packet_phase_peak_;
        }

    private:
        friend class FrozenNativeModel;
        std::shared_ptr<const void> source_;
        std::uint64_t topology_ = 0, partition_ = 0, transaction_ = 0, application_bytes_ = 0,
                      local_application_bytes_ = 0;
        std::shared_ptr<const FrozenNativeModel> candidate_;
        std::optional<std::size_t> packet_phase_peak_;
    };
    /** Sorted local owned-key order. Owner records and packets are authenticated
     * against this immutable source before a candidate is published. */
    [[nodiscard]] Plan plan_migration(std::span<const int> destinations) const;
    [[nodiscard]] Plan plan_migration(const PreparedMigration&) const;
    [[nodiscard]] Plan plan_migration(const AdaptiveBalancePlan&) const;
    /** Empty requests are a no-op; any global nonempty topology request rejects
     * collectively, including requests involving only certified participants. */
    [[nodiscard]] Plan plan_adaptation(
        std::span<const AdaptiveMark>, std::span<const AdaptiveBytes> coarsen = {}) const;
    [[nodiscard]] std::shared_ptr<const FrozenNativeModel> apply(const Plan&) const;
    /** Fetch authenticated full first-ring and requested reconstruction ghosts,
     * then outer metric neighbors. No global cell/assignment array is created. */
    [[nodiscard]] FrozenNativePublication publication(std::span<const AdaptiveCellKey> extra_ghosts = {}) const;

private:
    explicit FrozenNativeModel(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes::AMR
