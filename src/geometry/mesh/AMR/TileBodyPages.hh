#pragma once
#include "ContactSourcePoints.hh"
#include "PooledTile.hh"
#include "utils/export_helper.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class IncidencePhysicalOwners;
struct TileBodyPageCredit;
struct TileSourceCredit;
class TileBodyView;
class TileBodyLoopView;
/** A page owner and a record ordinal, without an individual body allocation. */
class SIMPLEFLUID_LOCAL TileBodyHandle
{
public:
    TileBodyHandle() = default;
    TileBodyHandle(const TileBodyHandle&);
    TileBodyHandle(TileBodyHandle&&) noexcept;
    TileBodyHandle& operator=(const TileBodyHandle&);
    TileBodyHandle& operator=(TileBodyHandle&&) noexcept;
    ~TileBodyHandle();
    explicit operator bool() const noexcept { return bool(credit_); }
    [[nodiscard]] TileBodyView view() const;
    [[nodiscard]] TileBodyScalars scalars() const;
    [[nodiscard]] TileBodyLoopView intrinsic_loop() const;
    [[nodiscard]] std::shared_ptr<const void> credit_owner() const noexcept;
    [[nodiscard]] std::size_t credit_bytes() const noexcept;
    void observe_source_owners(IncidencePhysicalOwners&) const;
    [[nodiscard]] const Budget* credit_origin() const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    [[nodiscard]] const void* page_identity() const noexcept;
    /** Weak physical-body observation does not pin a page or an arena credit. */
    [[nodiscard]] std::weak_ptr<const void> body_owner() const noexcept;
    /** Physical payload pin for exact owner accounting across arena credits. */
    [[nodiscard]] std::shared_ptr<const void> body_storage_owner() const noexcept;
    [[nodiscard]] std::size_t body_bytes() const noexcept;

private:
    friend class TileBodyPages;
    friend class TileBodyView;
    friend class TileBodyLoopView;
    TileBodyHandle(std::shared_ptr<TileBodyPageCredit>, std::uint8_t);
    void retain();
    void release() noexcept;
    std::shared_ptr<TileBodyPageCredit> credit_;
    std::uint8_t slot_ = 0;
};
/** Allocation-free range over native slots or immutable compact/wide rings.
 * Iterators borrow this range, just as vector iterators borrow their owner. */
class SIMPLEFLUID_LOCAL TileBodyLoopView
{
public:
    class const_iterator
    {
    public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        const_iterator() = default;
        reference operator*() const { return (*owner_)[index_]; }
        const_iterator& operator++()
        {
            ++index_;
            return *this;
        }
        const_iterator operator++(int)
        {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const const_iterator&) const = default;

    private:
        friend class TileBodyLoopView;
        const_iterator(const TileBodyLoopView* owner, std::size_t index) : owner_(owner), index_(index) {}
        const TileBodyLoopView* owner_ = nullptr;
        std::size_t index_ = 0;
    };
    TileBodyLoopView() = default;
    TileBodyLoopView(std::span<const std::size_t> values) : direct_(values) {}
    TileBodyLoopView(const std::vector<std::size_t>& values) : direct_(values) {}
    /** Borrowed indexed saved storage; caller retains its immutable owner. */
    TileBodyLoopView(const void* owner, std::size_t count, std::size_t (*at)(const void*, std::size_t))
        : indexed_owner_(owner), indexed_count_(count), indexed_at_(at)
    {
    }
    friend bool operator==(const TileBodyLoopView&, const TileBodyLoopView&);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] std::size_t operator[](std::size_t) const;
    [[nodiscard]] std::size_t at(std::size_t) const;
    [[nodiscard]] const_iterator begin() const { return {this, 0}; }
    [[nodiscard]] const_iterator end() const { return {this, size()}; }

private:
    friend class TileBodyView;
    friend class TileBodyHandle;
    explicit TileBodyLoopView(TileBodyHandle handle) : handle_(std::move(handle)) {}
    const void* indexed_owner_ = nullptr;
    std::size_t indexed_count_ = 0;
    std::size_t (*indexed_at_)(const void*, std::size_t) = nullptr;
    std::span<const std::size_t> direct_;
    TileBodyHandle handle_;
};
class SIMPLEFLUID_LOCAL TileBodyView
{
public:
    [[nodiscard]] TileBodyScalars scalars() const;
    [[nodiscard]] TileBodyLoopView intrinsic_loop() const { return TileBodyLoopView(handle_); }

private:
    friend class TileBodyHandle;
    explicit TileBodyView(TileBodyHandle handle) : handle_(std::move(handle)) {}
    TileBodyHandle handle_;
};
struct TileBodyPageStatistics
{
    std::size_t pages = 0, wide_pages = 0;
    std::size_t occupied_records = 0, live_records = 0, record_capacity = 0;
    std::size_t ring_size = 0, ring_capacity = 0, ring_bytes = 0;
    std::size_t metadata_bytes = 0, page_credit_bytes = 0;
    std::size_t page_body_bytes = 0, page_credit_overhead_bytes = 0;
    std::size_t deferred_pages = 0, deferred_body_bytes = 0, deferred_metadata_bytes = 0;
    std::size_t source_pages = 0, source_records = 0;
};
struct TileBodyReapStatistics
{
    std::size_t pages = 0, occupied_records = 0, released_bytes = 0;
    std::size_t deferred_pages = 0, deferred_body_bytes = 0;
};
/** Freeze before publishing partial pages or forking. Forked pages get a new
 * arena credit; adopt rebinds handles to that credit. Reap at mutation boundaries
 * releases sealed pages retained only by this builder. Empty builders allocate
 * nothing. All callers of a raw-budget builder must keep its arena alive. */
class SIMPLEFLUID_LOCAL TileBodyPages
{
public:
    explicit TileBodyPages(Budget&, std::shared_ptr<Budget> keeper = {});
    TileBodyPages(const TileBodyPages&, Budget&, std::shared_ptr<Budget> keeper = {},
        std::shared_ptr<CartesianAuthorityArena> source_arena = {});
    ~TileBodyPages();
    TileBodyPages(const TileBodyPages&) = delete;
    TileBodyPages& operator=(const TileBodyPages&) = delete;
    [[nodiscard]] TileBodyHandle append(const PooledTile&);
    [[nodiscard]] TileBodyHandle adopt(const TileBodyHandle&);
    void freeze() noexcept;
    /** Shared physical bodies retain their existing body admission in a weak
     * retirement table. A new table node is admitted before dropping a credit;
     * a failed admission leaves that page unchanged. Re-adoption reuses it. */
    [[nodiscard]] TileBodyReapStatistics reap();
    [[nodiscard]] TileBodyPageStatistics statistics() const noexcept;
    /** Only unexposed initial-session pages. Whole loops and FULL identities
     * match the real support/line outputs before their ring arrays retire. */
    std::size_t certify_private_source_loops(
        std::shared_ptr<const ContactSourcePointSnapshot>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    /** Admission metadata private to the builder, apart from the exact
     * current-origin contexts carried in the shared subface owner union. */
    [[nodiscard]] std::size_t source_only_metadata_bytes(std::size_t exported_context_bytes) const noexcept;
    [[nodiscard]] std::size_t page_credit_metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t retained_bytes() const noexcept;
    [[nodiscard]] bool imports_source_owner(const void*) const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    template<class Visitor> void visit_credit_owners(Visitor&& visitor) const
    {
        for (const auto& credit : pages_)
            if (credit)
                visitor(std::static_pointer_cast<const void>(credit), credit_bytes(credit));
    }

private:
    using WeakBodyOwner = std::weak_ptr<const void>;
    using DeferredBodies = std::map<WeakBodyOwner, std::size_t, std::owner_less<>>;
    static constexpr std::size_t deferred_node_bytes = sizeof(DeferredBodies::value_type) + 4 * sizeof(void*);
    struct SourceContextUse
    {
        std::weak_ptr<TileSourceCredit> owner;
        std::size_t references = 0;
    };
    using SourceContexts = std::map<const ContactSourcePointSnapshot*, SourceContextUse>;
    static constexpr std::size_t source_context_node_bytes = sizeof(SourceContexts::value_type) + 4 * sizeof(void*);
    void retain_source_context(const std::shared_ptr<TileSourceCredit>&);
    void release_source_context(const std::shared_ptr<TileSourceCredit>&) noexcept;
    void release_source_context_index() noexcept;
    void defer_body(TileBodyPageCredit&);
    void release_deferred() noexcept;
    static std::size_t credit_bytes(const std::shared_ptr<TileBodyPageCredit>&) noexcept;
    std::shared_ptr<TileSourceCredit> source_context(std::shared_ptr<const ContactSourcePointSnapshot>);
    void grow_directory();
    [[nodiscard]] std::size_t install(std::shared_ptr<TileBodyPageCredit>);
    Budget* budget_;
    std::shared_ptr<Budget> keeper_;
    std::shared_ptr<CartesianAuthorityArena> source_arena_;
    std::shared_ptr<TileSourceCredit> source_credit_;
    SourceContexts source_contexts_;
    std::size_t source_context_bytes_ = 0, active_page_credits_ = 0;
    std::size_t charged_ = 0;
    std::size_t active_ = std::numeric_limits<std::size_t>::max(), first_free_ = 0;
    std::vector<std::shared_ptr<TileBodyPageCredit>> pages_;
    DeferredBodies deferred_;
    std::size_t deferred_body_bytes_ = 0, deferred_metadata_bytes_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
