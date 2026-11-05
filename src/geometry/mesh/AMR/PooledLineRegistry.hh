#pragma once
#include "CartesianSourceLineRanges.hh"
#include "DeferredOwnerCharges.hh"
#include "PooledLineQuery.hh"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct PooledLineEdit
{
    std::size_t line = 0;
    PooledLineSubscriber subscriber;
    bool remove = false;
};
/** A changed query is identified by its ordinal in the complete line. The
 * packed result can be borrowed after recomputation, without retaining the
 * invocation-local legacy subscriber array. */
using PooledChangedLineQuery = std::function<void(std::size_t line, std::size_t subscriber)>;
struct PooledLineResultView
{
    // Metadata only: explicit result storage is borrowed separately below.
    PooledLineSubscriber subscriber;
    std::span<const std::size_t> explicit_result;
    const CartesianSourceLineRangeRecipe* source_result = nullptr;
    const IncidencePools* source_pools = nullptr;
    std::size_t source_ordinal = 0;
    [[nodiscard]] std::size_t result_size() const
    {
        if (source_result)
            return source_result->result_size(source_ordinal, *source_pools);
        return subscriber.result_state == PooledLineSubscriber::ResultState::ForwardEndpoints ? 2
                                                                                              : explicit_result.size();
    }
    [[nodiscard]] std::size_t result_at(std::size_t index) const
    {
        if (source_result)
            return source_result->result_at(source_ordinal, index, *source_pools);
        if (subscriber.result_state == PooledLineSubscriber::ResultState::ForwardEndpoints)
            return subscriber.result_at(index);
        if (index >= explicit_result.size())
            throw std::out_of_range("Invalid borrowed line result index.");
        return explicit_result[index];
    }
};
class PooledLineRegistry
{
public:
    struct ActiveStorage
    {
        std::size_t lines = 0, explicit_records = 0, run_records = 0, runs = 0;
        std::size_t explicit_bytes = 0, run_bytes = 0;
        std::size_t source_records = 0, source_families = 0, source_bytes = 0;
        std::size_t source_catalog_bytes = 0, source_maps = 0, source_mapped_components = 0;
        std::size_t source_nonforward_records = 0, source_result_selectors = 0;
        std::size_t retained_explicit_records = 0, retained_run_records = 0, retained_source_records = 0;
    };
    PooledLineRegistry(Budget&, IncidencePools&);
    PooledLineRegistry(const PooledLineRegistry&, Budget&, IncidencePools&);
    ~PooledLineRegistry();
    PooledLineRegistry(const PooledLineRegistry&) = delete;
    PooledLineRegistry& operator=(const PooledLineRegistry&) = delete;
    /** Layer zero is intrinsic geometry. A replay preserves every old later
     * layer before editing an earlier one, preventing deletion cycles. */
    void begin_replay(bool bootstrap);
    void set_layer(unsigned);
    void finish_replay(unsigned last_layer);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return lines_.size() * index_node_bytes + (dirty_.size() + touched_.size()) * set_node_bytes +
               deferred_.metadata_bytes() + source_lease_bytes_ + source_owners_.capacity() * sizeof(SourceOwner);
    }
    [[nodiscard]] std::size_t processed_lines() const noexcept { return processed_lines_; }
    [[nodiscard]] bool imports_source_owner(const void*) const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    /** Effective current-layer representation; saved future/older snapshots
     * remain separately included in retained_bytes(). This census allocates nothing. */
    [[nodiscard]] ActiveStorage active_storage() const;
    [[nodiscard]] std::size_t retained_bytes() const noexcept;
    void reap_deferred() noexcept
    {
        deferred_.reap();
        reap_source_owners();
    }
    void apply(std::span<const PooledLineEdit>);
    void recompute(const PooledChangedLineQuery&);
    [[nodiscard]] std::optional<std::size_t> next_dirty() const;
    void recompute_one(std::size_t, const PooledChangedLineQuery&);
    /** Borrowed until the next registry mutation. */
    [[nodiscard]] PooledLineResultView query_result(std::size_t line, std::size_t subscriber) const;
    /** Read-only attestation of an exact ALREADY completed private member. The
     * result is borrowed until registry mutation; no query is recomputed here. */
    struct CompletedMember
    {
        PooledLineResultView result;
        unsigned kernel_witness_layer = 0;
    };
    [[nodiscard]] std::optional<CompletedMember> find_completed_member(std::size_t line, unsigned effective_layer,
        const PooledLineSubscriber& original, const IncidencePools& request_namespace) const;
    struct SavedSourceQuery
    {
        std::shared_ptr<const CartesianSourceLineRangeRecipe> recipe;
        std::size_t ordinal = 0;
    };
    /** Captures a query only from its exact kernel-layer witness and complete
     * member key, including the original supplied physical endpoints. */
    [[nodiscard]] std::optional<SavedSourceQuery> capture_saved_source_query(
        std::size_t line, unsigned effective_layer, unsigned kernel_witness_layer, const PooledLineSubscriber&) const;
    /** A closed actual native pass and published point snapshot authorize one
     * shared source-axis proof pass. Every ordered output is replayed before
     * any existing immutable bucket is replaced. */
    void certify_source_ranges(const CartesianLineRangeAdmission&, std::shared_ptr<CartesianAuthorityArena>);

private:
    struct SubscriberRun
    {
        /** support/first/last/emission/edge/actual support generation. */
        std::array<std::uint64_t, 6> origin{};
        std::array<std::int64_t, 6> step{};
        std::size_t first = 0, count = 0;
        int resident = -1, support_authority = -1;
        std::uint8_t image = 0;
        std::uint8_t period = 1;
        std::uint32_t pattern_offset = 0;
        [[nodiscard]] PooledLineSubscriber at(std::size_t, std::span<const std::array<std::uint64_t, 6>>) const;
    };
    struct SubscriberSpill
    {
        std::size_t support = 0, first = 0, last = 0;
        std::uint64_t emission = 0, edge = 0;
        std::uint64_t source_generation = 0;
        std::vector<std::size_t> result;
    };
    struct StoredSubscriber
    {
        std::uint32_t support = 0, first = 0, last = 0, emission = 0, edge = 0;
        int resident = -1, support_authority = -1;
        std::uint8_t image = 0;
        PooledLineSubscriber::ResultState result_state = PooledLineSubscriber::ResultState::Unset;
        std::uint16_t source_generation = 0;
        /** High image bits carry private codec flags; exposed image stays0/1.
         * Reusing existing padding preserves the ordinary40-byte record. */
        [[nodiscard]] bool wide() const noexcept { return image & 0x80; }
        [[nodiscard]] bool native() const noexcept { return image & 0x40; }
        // Ordinary forward-endpoint answers need neither a vector header nor
        // a payload allocation. Wide metadata and explicit answers share this
        // uncommon owned spill, copied only with their complete line bucket.
        std::unique_ptr<SubscriberSpill> spill;
        [[nodiscard]] PooledLineSubscriber metadata() const;
        [[nodiscard]] StoredSubscriber clone() const;
        [[nodiscard]] static bool needs_wide(const PooledLineSubscriber&) noexcept;
        [[nodiscard]] static StoredSubscriber pack(PooledLineSubscriber&&);
        [[nodiscard]] std::size_t spill_bytes() const;
    };
    struct Bucket
    {
        std::vector<StoredSubscriber> subscribers;
        std::vector<SubscriberRun> runs;
        std::vector<std::array<std::uint64_t, 6>> patterns;
        std::shared_ptr<const CartesianSourceLineRangeRecipe> source_ranges;
        std::size_t subscription_count = 0;
        unsigned certified_layer = 0;
        std::size_t charge = 0;
        [[nodiscard]] std::size_t size() const noexcept
        {
            return source_ranges ? source_ranges->size() : (runs.empty() ? subscribers.size() : subscription_count);
        }
        [[nodiscard]] PooledLineSubscriber metadata(std::size_t, const IncidencePools&) const;
        [[nodiscard]] std::span<const std::size_t> explicit_result(std::size_t) const;
        [[nodiscard]] std::size_t result_size(std::size_t, const IncidencePools&) const;
        [[nodiscard]] std::size_t result_at(std::size_t, std::size_t, const IncidencePools&) const;
        [[nodiscard]] std::size_t spill_bytes(std::size_t, const IncidencePools&) const;
        [[nodiscard]] StoredSubscriber clone(std::size_t, const IncidencePools&) const;
    };
    using Key = std::pair<std::size_t, unsigned>;
    using Index = std::map<Key, std::shared_ptr<const Bucket>>;
    static constexpr std::size_t index_node_bytes = sizeof(Index::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t set_node_bytes = sizeof(std::size_t) + 4 * sizeof(void*);
    [[nodiscard]] Index::const_iterator effective(std::size_t, unsigned) const;
    void preserve_future(std::size_t);
    void replace(std::size_t, std::shared_ptr<const Bucket>);
    static std::size_t body(const Bucket&);
    /** Build from real complete-line output only. Every ordered full identity,
     * metadata field and source generation is compared before array retirement. */
    [[nodiscard]] bool certify_runs(Bucket&, std::span<const PooledLineSubscriber>, std::size_t& scratch);
    void admit(std::size_t);
    void release(std::size_t);
    void reap_source_owners() noexcept;
    Budget* budget_;
    DeferredOwnerCharges deferred_;
    IncidencePools* pools_;
    std::size_t charged_ = 0;
    Index lines_;
    std::set<std::size_t> dirty_;
    std::set<std::size_t> touched_;
    unsigned layer_ = 0;
    bool replay_ = false;
    std::size_t processed_lines_ = 0;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> source_leases_;
    std::size_t source_lease_bytes_ = 0;
    struct SourceOwner
    {
        std::weak_ptr<const CartesianSourceLineRangeRecipe> owner;
        const void* control = nullptr;
        std::size_t control_bytes = 0;
        SourceOwner(const std::shared_ptr<const CartesianSourceLineRangeRecipe>& value) : owner(value)
        {
            const auto observation = value->borrowed_control_payload();
            control = observation.first;
            control_bytes = observation.second;
        }
        [[nodiscard]] bool expired() const noexcept { return owner.expired(); }
        [[nodiscard]] auto lock() const noexcept { return owner.lock(); }
    };
    std::vector<SourceOwner> source_owners_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
