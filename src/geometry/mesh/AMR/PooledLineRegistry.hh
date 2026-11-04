#pragma once
#include "DeferredOwnerCharges.hh"
#include "PooledLineQuery.hh"
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
    [[nodiscard]] std::size_t result_size() const noexcept
    {
        return subscriber.result_state == PooledLineSubscriber::ResultState::ForwardEndpoints ? 2
                                                                                              : explicit_result.size();
    }
    [[nodiscard]] std::size_t result_at(std::size_t index) const
    {
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
               deferred_.metadata_bytes();
    }
    [[nodiscard]] std::size_t processed_lines() const noexcept { return processed_lines_; }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return charged_ + deferred_.retained_bytes(); }
    void reap_deferred() noexcept { deferred_.reap(); }
    void apply(std::span<const PooledLineEdit>);
    void recompute(const PooledChangedLineQuery&);
    [[nodiscard]] std::optional<std::size_t> next_dirty() const;
    void recompute_one(std::size_t, const PooledChangedLineQuery&);
    /** Borrowed until the next registry mutation. */
    [[nodiscard]] PooledLineResultView query_result(std::size_t line, std::size_t subscriber) const;

private:
    struct SubscriberSpill
    {
        std::size_t support = 0, first = 0, last = 0;
        std::uint64_t emission = 0, edge = 0;
        std::vector<std::size_t> result;
    };
    struct StoredSubscriber
    {
        std::uint32_t support = 0, first = 0, last = 0, emission = 0, edge = 0;
        int resident = -1, support_authority = -1;
        std::uint8_t image = 0;
        PooledLineSubscriber::ResultState result_state = PooledLineSubscriber::ResultState::Unset;
        bool wide = false;
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
        std::size_t charge = 0;
    };
    using Key = std::pair<std::size_t, unsigned>;
    using Index = std::map<Key, std::shared_ptr<const Bucket>>;
    static constexpr std::size_t index_node_bytes = sizeof(Index::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t set_node_bytes = sizeof(std::size_t) + 4 * sizeof(void*);
    [[nodiscard]] Index::const_iterator effective(std::size_t, unsigned) const;
    void preserve_future(std::size_t);
    void replace(std::size_t, std::shared_ptr<const Bucket>);
    static std::size_t body(const Bucket&);
    void admit(std::size_t);
    void release(std::size_t);
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
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
