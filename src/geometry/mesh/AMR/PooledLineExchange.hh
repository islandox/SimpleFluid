#pragma once
#include "CartesianAuthorityCredits.hh"
#include "ContactSourcePoints.hh"
#include "PooledLineRegistry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using NextPooledLineEdit = std::function<bool(PooledLineEdit&)>;
class PooledLineCompletion;
using PooledLineCompletionView = std::shared_ptr<const PooledLineCompletion>;
using PooledLineReply = std::function<void(std::size_t line, const PooledLineSubscriber&)>;
/** Private fault-injection seam, applied after changed-query counts are minted.
 * The callback may alter only this page; the candidate is discarded on failure. */
using PooledLineBeforeSend = std::function<void(unsigned layer, std::uint64_t wave, std::vector<OwnerPacket>&)>;
using PooledLineDataObserve = std::function<void(
    bool reply, unsigned layer, std::uint64_t wave, std::uint64_t page, std::span<const OwnerPacket>)>;
/** Private bounded fault seam; production defaults to None. It never hands a
 * callback an unadmitted mutable output container. */
struct PooledNativeLineFault
{
    enum class Kind
    {
        None,
        MissingCompletion,
        DuplicateCompletion,
        StaleContext,
        ReorderedReference,
        TruncatedCompletion
    };
    Kind kind = Kind::None;
    bool reply = true;
    int rank = 0;
    std::uint64_t wave = 0, page = 0;
};
/** A read-only request for an already completed exact native member. */
struct PooledLineCompletionRequest
{
    std::size_t line = 0;
    PooledLineSubscriber original;
};
using NextPooledLineCompletionRequest = std::function<bool(PooledLineCompletionRequest&)>;
/** Distinct cached-read control fault seam. Mutations are bounded and never
 * expose an unadmitted mutable output to a callback. */
struct PooledCachedLineFault
{
    enum class Kind
    {
        None,
        MissingEnvelope,
        DuplicateEnvelope,
        StaleContext,
        TruncatedEnvelope,
        InvalidEmptyCount,
        MissingCompletion,
        DuplicateChunk,
        StaleGeneration
    };
    Kind kind = Kind::None;
    bool reply = true;
    int rank = 0;
    int peer = -1;
    std::uint64_t wave = 0, page = 0;
};
/** Exchange-authenticated synchronous borrow after all terminal/count gates.
 * No receipt, result allocation or retained endpoint/key copy is created by this view.
 * References and spans are valid only during the consume callback; consumers
 * may neither retain the view nor treat copied values as an owning proof. */
class PooledLineBorrowedCompletion
{
public:
    PooledLineBorrowedCompletion(const PooledLineBorrowedCompletion&) = delete;
    PooledLineBorrowedCompletion& operator=(const PooledLineBorrowedCompletion&) = delete;
    PooledLineBorrowedCompletion(PooledLineBorrowedCompletion&&) = delete;
    PooledLineBorrowedCompletion& operator=(PooledLineBorrowedCompletion&&) = delete;
    [[nodiscard]] const PooledLineSubscriber& original_request() const noexcept { return original_; }
    [[nodiscard]] ContactHaloContext context() const noexcept { return context_; }
    [[nodiscard]] unsigned layer() const noexcept { return layer_; }
    [[nodiscard]] unsigned kernel_witness_layer() const noexcept { return layer_; }
    [[nodiscard]] std::span<const char> line_key() const;
    [[nodiscard]] std::span<const char> support_key() const;
    [[nodiscard]] ContactSourcePoint endpoint(unsigned) const;
    [[nodiscard]] const void* point_pool_identity() const noexcept;
    [[nodiscard]] std::span<const std::size_t> result() const noexcept { return result_; }

private:
    PooledLineBorrowedCompletion(std::size_t, const PooledLineSubscriber&, std::span<const std::size_t>,
        ContactHaloContext, unsigned, const IncidencePools&);
    std::size_t line_;
    const PooledLineSubscriber& original_;
    std::span<const std::size_t> result_;
    ContactHaloContext context_;
    unsigned layer_;
    const IncidencePools& pools_;
    friend void exchange_pooled_line_edits(PooledLineRegistry&, IncidencePools&, const NextPooledLineEdit&, unsigned,
        ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::size_t, const PooledLineReply&,
        const PooledLineBeforeSend&, bool, PooledNativeLineFault, const PooledLineDataObserve&,
        const std::function<std::optional<PooledLineSubscriber>(std::size_t, const PooledLineSubscriber&)>&,
        const std::function<void(std::size_t, PooledLineCompletion&&)>&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>,
        const std::function<bool(std::size_t, const PooledLineBorrowedCompletion&)>&);
};
/** True consumes this reply without constructing an owning receipt. False
 * preserves the normal seal/completed callback path. Never retain the borrow. */
using PooledLineCompletionConsume = std::function<bool(std::size_t, const PooledLineBorrowedCompletion&)>;
/** A receiver-issued immutable proof of an actual completed native reply.
 * Result bytes remain in the move-only completion until exact value retirement.
 * Neither arbitrary descriptors nor a source eligibility flag can issue one. */
class PooledLineCompletion
{
public:
    class Proof
    {
    public:
        ~Proof();
        Proof(const Proof&) = delete;
        Proof& operator=(const Proof&) = delete;
        Proof(Proof&&) = delete;
        Proof& operator=(Proof&&) = delete;
        [[nodiscard]] const PooledLineSubscriber& original_request() const noexcept { return original_; }
        /** Cached reads bind the current request context/effective layer. The
         * actual earlier kernel witness is preserved separately below. */
        [[nodiscard]] ContactHaloContext context() const noexcept { return context_; }
        [[nodiscard]] unsigned layer() const noexcept { return layer_; }
        [[nodiscard]] unsigned kernel_witness_layer() const noexcept { return kernel_layer_; }
        [[nodiscard]] bool is_cached_read() const noexcept { return cached_read_; }
        [[nodiscard]] std::span<const char> line_key() const noexcept { return line_; }
        [[nodiscard]] std::span<const char> support_key() const noexcept { return support_; }
        [[nodiscard]] const ContactSourcePoint& endpoint(unsigned i) const { return endpoints_.at(i); }
        [[nodiscard]] const void* point_pool_identity() const noexcept { return point_pool_; }
        [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
        [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
        [[nodiscard]] std::size_t metadata_bytes() const noexcept;
        [[nodiscard]] std::size_t known_payload_bytes() const noexcept;
        [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept;
        void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

    private:
        Proof(PooledLineSubscriber, ContactHaloContext, unsigned, const IncidencePools&);
        PooledLineSubscriber original_;
        ContactHaloContext context_;
        unsigned layer_, kernel_layer_;
        bool cached_read_ = false;
        AdaptiveBytes line_, support_;
        std::array<ContactSourcePoint, 2> endpoints_;
        const void* point_pool_;
        Budget* budget_ = nullptr;
        std::shared_ptr<Budget> keeper_;
        std::size_t charge_ = 0;
        const void* control_ = nullptr;
        friend class PooledLineCompletion;
    };
    PooledLineCompletion(PooledLineCompletion&&) noexcept;
    PooledLineCompletion& operator=(PooledLineCompletion&&) noexcept;
    PooledLineCompletion(const PooledLineCompletion&) = delete;
    PooledLineCompletion& operator=(const PooledLineCompletion&) = delete;
    ~PooledLineCompletion();
    /** Move the one validated output allocation into a copyable immutable owner.
     * Layer clone/fork then shares data without another expanded result array. */
    [[nodiscard]] PooledLineCompletionView share_immutable() &&;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    /** Actual headers, proof keys and control token; excludes result capacity
     * and unknown shared-control/allocator overhead. */
    [[nodiscard]] std::size_t known_payload_bytes() const noexcept;
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept;
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    [[nodiscard]] bool imports_owner(const void*) const noexcept;
    [[nodiscard]] const std::shared_ptr<const Proof>& proof() const noexcept { return proof_; }
    [[nodiscard]] std::span<const std::size_t> result() const noexcept { return result_; }
    [[nodiscard]] std::size_t result_capacity() const noexcept { return result_.capacity(); }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return output_budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(output_keeper_); }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    PooledLineCompletion(std::shared_ptr<const Proof>, std::vector<std::size_t>&&, Budget&, std::shared_ptr<Budget>);
    [[nodiscard]] static PooledLineCompletion seal(std::size_t, PooledLineSubscriber, std::vector<std::size_t>&&,
        ContactHaloContext, unsigned, const IncidencePools&, Budget&, std::shared_ptr<Budget>,
        const std::shared_ptr<CartesianAuthorityArena>&, unsigned kernel_witness_layer, bool cached_read);
    std::shared_ptr<const Proof> proof_;
    std::vector<std::size_t> result_;
    Budget* output_budget_ = nullptr;
    std::shared_ptr<Budget> output_keeper_;
    std::size_t output_charge_ = 0, wrapper_charge_ = 0;
    const void* wrapper_control_ = nullptr;
    friend void fetch_pooled_line_completions(PooledLineRegistry&, IncidencePools&,
        const NextPooledLineCompletionRequest&, unsigned, ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>,
        Budget&, std::size_t, const std::function<void(std::size_t, PooledLineCompletion&&)>&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>, PooledCachedLineFault);
    friend void exchange_pooled_line_edits(PooledLineRegistry&, IncidencePools&, const NextPooledLineEdit&, unsigned,
        ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::size_t, const PooledLineReply&,
        const PooledLineBeforeSend&, bool, PooledNativeLineFault, const PooledLineDataObserve&,
        const std::function<std::optional<PooledLineSubscriber>(std::size_t, const PooledLineSubscriber&)>&,
        const std::function<void(std::size_t, PooledLineCompletion&&)>&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>, const PooledLineCompletionConsume&);
};
/** Resolve only already captured original local requests. The decoded reply's
 * first/last are deliberately never consulted: version two does not carry them. */
using PooledLineOriginalRequest =
    std::function<std::optional<PooledLineSubscriber>(std::size_t, const PooledLineSubscriber&)>;
using PooledLineCompletionReceive = std::function<void(std::size_t, PooledLineCompletion&&)>;
/** Stream authenticated per-layer subscriptions, then recompute complete
 * dirty line queries. All ranks participate even with no local edits. */
void exchange_pooled_line_edits(PooledLineRegistry&, IncidencePools&, const NextPooledLineEdit&, unsigned layer,
    ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::size_t page_bytes, const PooledLineReply&,
    const PooledLineBeforeSend& = {}, bool source_protocol = false, PooledNativeLineFault = {},
    const PooledLineDataObserve& = {}, const PooledLineOriginalRequest& = {}, const PooledLineCompletionReceive& = {},
    std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {},
    const PooledLineCompletionConsume& consume_completion = {});
/** Distinct bounded cached-read controls, including all empty peers and ranks.
 * Authority matches the complete stored input and returns its actual immutable
 * cached output; it never reruns a numerical query or changes version-two data.
 * The callback escapes only after complete per-wave receipt validation. */
void fetch_pooled_line_completions(PooledLineRegistry&, IncidencePools&, const NextPooledLineCompletionRequest&,
    unsigned effective_layer, ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, Budget&,
    std::size_t page_bytes, const PooledLineCompletionReceive&, std::shared_ptr<Budget>,
    std::shared_ptr<CartesianAuthorityArena>, PooledCachedLineFault = {});
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
