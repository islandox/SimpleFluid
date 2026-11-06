#pragma once
#include "ContactStorageBreakdown.hh"
#include "IdentityPoolBorrowedPayload.hh"
#include "IdentityPoolHash.hh"
#include "IdentityPoolPayload.hh"
#include "PersistentAppendSlots.hh"
#include "PersistentPlaneOwnerIndex.hh"
#include "PhysicalOwnerIndex.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <atomic>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
enum class IdentityPoolStorage
{
    Flat,
    Persistent
};

namespace identity_pool_detail
{
inline std::uint64_t next_scope()
{
    // Shared across point and typed metadata pools. The maximum value is
    // reserved for the incidence tile-body credit family.
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;)
    {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Identity pool scope exhausted.");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}
} // namespace identity_pool_detail
/** Full identities reside in immutable shared pages. Persistent forks share
 * the ordinal directory and lookup roots, path-copying only appended records.
 * Flat storage supports standalone pools with a borrowed allocation arena.
 * Slots never participate in numerical ordering. */
template<class T, class Hasher = IdentityPoolHasher<T>, class Payload = IdentityPoolPayload<T>> class IdentityPool
{
public:
    class PageView;

private:
    static constexpr std::size_t records_per_page = 16;
    static constexpr bool owns_plane_backing =
        std::is_same_v<T, PersistentPlaneOwnerIndex::Plane> && std::is_same_v<Payload, IdentityPoolPayload<T>>;
    struct Empty
    {
    };
    using PlaneOwner = std::shared_ptr<const PersistentPlaneOwnerIndex::Owner>;
    struct Page
    {
        [[no_unique_address]] std::conditional_t<owns_plane_backing, std::vector<PlaneOwner>, Empty> plane_owners;
        std::vector<T> values;
        const Budget* body_origin = nullptr;
        std::uint64_t body_scope = 0;
        std::atomic<bool> frozen{false};
    };
    struct PageCredit
    {
        Budget* budget;
        std::shared_ptr<Budget> keeper;

    private:
        friend class IdentityPool;
        friend class PageView;
        std::shared_ptr<Page> page;

    public:
        std::size_t bytes = 0;
        std::uint64_t pool_scope;
        bool persistent_owner;
        PageCredit(Budget& b, std::shared_ptr<Budget> k, std::shared_ptr<Page> p, std::uint64_t scope = 0,
            bool persistent = false)
            : budget(&b), keeper(std::move(k)), page(std::move(p)), pool_scope(scope), persistent_owner(persistent)
        {
        }
        ~PageCredit()
        {
            page.reset();
            budget->used -= bytes;
        }
        void admit(std::size_t n)
        {
            if (n > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::invalid_argument("Identity page credit overflows.");
            budget->add(n);
            bytes += n;
        }
        [[nodiscard]] std::shared_ptr<const void> body_owner() const noexcept { return page; }
        [[nodiscard]] const Budget* body_origin() const noexcept { return page->body_origin; }
        [[nodiscard]] std::uint64_t body_scope() const noexcept { return page->body_scope; }
        [[nodiscard]] std::size_t body_bytes() const noexcept { return bytes - sizeof(PageCredit) - 128; }
        void release(std::size_t n) noexcept
        {
            bytes -= n;
            budget->used -= n;
        }
    };
    struct PageEntry
    {
        std::shared_ptr<PageCredit> credit;
        std::size_t first_slot;
    };

    struct ValueSelector
    {
        std::uint64_t hash;
        const T* value;
        std::size_t slot;
    };
    struct ValueLess
    {
        bool operator()(const ValueSelector& a, const ValueSelector& b) const
        {
            return a.hash != b.hash ? a.hash < b.hash : *a.value < *b.value;
        }
    };
    struct PageSelector
    {
        const void* address;
        bool credit;
        std::size_t ordinal;
    };
    struct PageLess
    {
        bool operator()(const PageSelector& a, const PageSelector& b) const noexcept
        {
            return a.credit != b.credit ? a.credit < b.credit : std::less<const void*>{}(a.address, b.address);
        }
    };
    using PageDirectory = PersistentAppendSlots<PageEntry>;
    using ValueIndex = persistent_ordered_detail::OrderedPages<ValueSelector, ValueLess>;
    using PageIndex = persistent_ordered_detail::OrderedPages<PageSelector, PageLess>;
    struct PlaneStorage
    {
        PersistentPlaneOwnerIndex::Root index;
        std::size_t bytes = 0;
    };
    struct PersistentStorage
    {
        typename PageDirectory::Root pages;
        typename ValueIndex::Ptr values;
        typename PageIndex::Ptr owners;
        [[no_unique_address]] std::conditional_t<owns_plane_backing, PlaneStorage, Empty> plane;
        std::size_t records = 0, body_bytes = 0;
        std::size_t foreign_page_bytes = 0, foreign_index_bytes = 0, retired_overlap = 0;
        bool overlapping = false;
    };
    static constexpr auto persistent_storage_bytes = sizeof(PersistentStorage) + 128;
    static constexpr auto credit_overhead = sizeof(PageCredit) + 128;

    template<class Visitor> static void visit_credit_payload(const PageCredit& credit, Visitor& visitor)
    {
        contact_payload_object(
            visitor, &credit, ContactPayloadRole::IdentityPageCreditObject, sizeof(PageCredit), true);
        const auto& page = *credit.page;
        contact_payload_object(visitor, &page, ContactPayloadRole::IdentityPageObject, sizeof(Page), true);
        contact_payload_buffer(visitor, &page, ContactPayloadRole::IdentityPageValues, page.values, page.values.size(),
            !IdentityPoolBorrowedPayload<T>::complete);
        if constexpr (owns_plane_backing)
            contact_payload_buffer(visitor, &page, ContactPayloadRole::IdentityPersistentIndex, page.plane_owners);
        for (const auto& value : page.values)
            IdentityPoolBorrowedPayload<T>::visit(value, visitor);
    }

public:
    using Slot = std::size_t;
    using PayloadPolicy = Payload;
    using HashPolicy = Hasher;
    static constexpr bool uses_hashing = Hasher::enabled;
    struct Less
    {
        using is_transparent = void;
        bool operator()(const T* a, const T* b) const { return *a < *b; }
        bool operator()(const T* a, const T& b) const { return *a < b; }
        bool operator()(const T& a, const T* b) const { return a < *b; }
    };
    IdentityPool(Budget& budget, std::shared_ptr<Budget> keeper = {}, Hasher hasher = {}, Payload payload = {},
        IdentityPoolStorage storage = IdentityPoolStorage::Flat)
        : budget_(&budget), keeper_(std::move(keeper)), payload_(std::move(payload)), hasher_(std::move(hasher)),
          page_index_(budget)
    {
        if (keeper_ && keeper_.get() != &budget)
            throw std::invalid_argument("Identity pool budget keeper mismatch.");
        if (storage == IdentityPoolStorage::Persistent && keeper_)
        {
            admit(persistent_storage_bytes);
            try
            {
                persistent_ = std::make_unique<PersistentStorage>();
            }
            catch (...)
            {
                release(persistent_storage_bytes);
                throw;
            }
        }
    }
    IdentityPool(const IdentityPool& source, Budget& budget, std::shared_ptr<Budget> keeper = {})
        : IdentityPool(budget, std::move(keeper), source.hasher_, source.payload_,
              source.persistent_ && source.budget_ != &budget ? IdentityPoolStorage::Persistent
                                                              : IdentityPoolStorage::Flat)
    {
        try
        {
            if (persistent_)
            {
                auto next = *source.persistent_;
                next.foreign_page_bytes = checked_sum(source.shared_body_bytes(), source.page_credit_overhead_bytes());
                next.foreign_index_bytes = source.persistent_index_bytes();
                next.retired_overlap = 0;
                next.overlapping = true;
                scope_id_ = source.scope_id_;
                admit(checked_sum(next.foreign_page_bytes, next.foreign_index_bytes));
                *persistent_ = std::move(next);
                if (source.owner_pages())
                    source.page_at(source.owner_pages() - 1).credit->page->frozen = true;
                return;
            }
            if (source.persistent_)
            {
                copy_flat_metadata(source);
                return;
            }
            admit(product(source.by_value_.size(), node_bytes()));
            reserve_exact(by_slot_, source.by_slot_.size());
            reserve_exact(pages_, source.pages_.size());
            reserve_exact(buckets_, source.buckets_.size());
            buckets_.assign(source.buckets_.begin(), source.buckets_.end());
            for (const auto& entry : source.pages_)
            {
                auto credit = make_credit(entry.credit->page, entry.credit->bytes - sizeof(PageCredit) - 128);
                pages_.push_back({std::move(credit), entry.first_slot});
                page_index_.insert(pages_.back().credit->page.get(), pages_.size() - 1);
            }
            by_value_ = source.by_value_;
            by_slot_.assign(source.by_slot_.begin(), source.by_slot_.end());
        }
        catch (...)
        {
            clear();
            throw;
        }
    }
    ~IdentityPool() { clear(); }
    IdentityPool(const IdentityPool&) = delete;
    IdentityPool& operator=(const IdentityPool&) = delete;

    [[nodiscard]] Slot intern(const T& value) { return intern_value(value); }
    [[nodiscard]] Slot intern(T&& value) { return intern_value(std::move(value)); }
    [[nodiscard]] const T& get(Slot slot) const
    {
        if (slot >= size())
            throw std::invalid_argument("Invalid identity pool slot.");
        if (persistent_)
        {
            const auto& entry = page_for(slot);
            return entry.credit->page->values[slot - entry.first_slot];
        }
        return *by_slot_[slot];
    }
    [[nodiscard]] std::optional<Slot> find(const T& value) const
    {
        if (persistent_)
        {
            const ValueSelector key{hash_value(value), &value, 0};
            const auto* found =
                ValueIndex::lower_bound(persistent_->values, [&](const auto& row) { return ValueLess{}(row, key); });
            return found && !ValueLess{}(key, *found) ? std::optional(found->slot) : std::nullopt;
        }
        if constexpr (Hasher::enabled)
        {
            if (buckets_.empty())
                return {};
            const auto bucket = find_bucket(value, hasher_(value));
            return buckets_[bucket] == empty_slot ? std::optional<Slot>{} : buckets_[bucket];
        }
        const auto found = by_value_.find(value);
        return found == by_value_.end() ? std::optional<Slot>{} : found->second;
    }
    /** Borrowed canonical byte ranges avoid constructing a temporary vector.
     * A compatible typed policy hashes the identical stream; custom policies
     * lacking that overload use full-value search rather than a different hash.
     * No route/hash decides equality. Existing pool/payload policies are intact. */
    template<class View>
        requires(std::is_same_v<T, AdaptiveBytes>)
    [[nodiscard]] std::optional<Slot> find_borrowed(const View& value) const
    {
        if (persistent_)
        {
            if constexpr (!Hasher::enabled || requires(const Hasher& hash) { hash(value); })
            {
                std::uint64_t hash = 0;
                if constexpr (Hasher::enabled)
                    hash = hasher_(value);
                const auto compare = [&](const auto& row)
                {
                    if (row.hash != hash)
                        return row.hash < hash ? std::strong_ordering::less : std::strong_ordering::greater;
                    const auto& stored = *row.value;
                    for (std::size_t i = 0; i < std::min(stored.size(), value.size()); ++i)
                        if (const auto ordering = stored[i] <=> static_cast<char>(value[i]); ordering != 0)
                            return ordering;
                    return stored.size() <=> value.size();
                };
                const auto* found =
                    ValueIndex::lower_bound(persistent_->values, [&](const auto& row) { return compare(row) < 0; });
                return found && compare(*found) == 0 ? std::optional(found->slot) : std::nullopt;
            }
        }
        const auto equal = [&](Slot slot)
        {
            const auto& stored = get(slot);
            if (stored.size() != value.size())
                return false;
            for (std::size_t i = 0; i < stored.size(); ++i)
                if (static_cast<unsigned char>(stored[i]) != static_cast<unsigned char>(value[i]))
                    return false;
            return true;
        };
        if constexpr (Hasher::enabled && requires(const Hasher& hash) { hash(value); })
        {
            if (buckets_.empty())
                return {};
            const auto mask = buckets_.size() - 1;
            auto bucket = static_cast<Slot>(mixed_hash(hasher_(value))) & mask;
            for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
            {
                const auto slot = buckets_[bucket];
                if (slot == empty_slot)
                    return {};
                if (equal(slot))
                    return slot;
                bucket = (bucket + 1) & mask;
            }
            throw std::invalid_argument("Borrowed identity lookup lacks an empty slot.");
        }
        for (Slot slot = 0; slot < size(); ++slot)
            if (equal(slot))
                return slot;
        return {};
    }
    [[nodiscard]] std::size_t size() const noexcept { return persistent_ ? persistent_->records : by_slot_.size(); }
    [[nodiscard]] bool persistent() const noexcept { return bool(persistent_); }
    [[nodiscard]] std::size_t retained_bytes() const noexcept
    {
        if (persistent_)
            return metadata_bytes() + page_credit_overhead_bytes() + shared_body_bytes();
        auto total = charged_ + page_index_.metadata_bytes();
        for (const auto& entry : pages_)
            total += entry.credit->bytes;
        return total;
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        if (persistent_)
            return persistent_storage_bytes + persistent_index_bytes();
        return by_value_.size() * node_bytes() + by_slot_.capacity() * sizeof(const T*) +
               pages_.capacity() * sizeof(PageEntry) + buckets_.capacity() * sizeof(Slot) +
               page_index_.metadata_bytes();
    }
    [[nodiscard]] std::uint64_t scope_id() const noexcept { return scope_id_; }
    [[nodiscard]] std::size_t page_credit_overhead_bytes() const noexcept { return owner_pages() * credit_overhead; }
    [[nodiscard]] bool owns_page_credit(const void* identity) const noexcept
    {
        if (persistent_)
            return find_page(identity, true);
        return std::any_of(
            pages_.begin(), pages_.end(), [identity](const auto& entry) { return entry.credit.get() == identity; });
    }
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept
    {
        return metadata_bytes() + page_credit_overhead_bytes();
    }
    [[nodiscard]] std::size_t shared_body_bytes() const noexcept
    {
        if (persistent_)
        {
            if constexpr (owns_plane_backing)
                return persistent_->body_bytes + persistent_->plane.bytes;
            return persistent_->body_bytes;
        }
        std::size_t bytes = 0;
        for (const auto& entry : pages_)
            bytes += entry.credit->bytes - sizeof(PageCredit) - 128;
        return bytes;
    }
    [[nodiscard]] std::size_t page_owner_lookup_probes() const noexcept { return page_index_.probes(); }
    [[nodiscard]] const void* metadata_identity() const noexcept { return &charged_; }
    [[nodiscard]] bool owns_page_admission(const void* body, const void* metadata) const noexcept
    {
        const auto* found = find_page(body);
        return found && &found->credit->bytes == metadata;
    }
    template<class Visitor> void visit_page_admissions(Visitor&& visitor) const
    {
        for_each_page(
            [&](const auto& entry)
            {
                visitor(&entry.credit->bytes, entry.credit->page.get(), credit_overhead,
                    entry.credit->bytes - credit_overhead);
            });
    }
    template<class Visitor> void visit_borrowed_pages(Visitor&& visitor) const
    {
        for_each_page(
            [&](const auto& entry)
            {
                visitor(ContactBorrowedPage<T>{entry.credit.get(), entry.credit->page.get(), sizeof(PageCredit),
                    sizeof(Page), entry.credit->page->values, entry.credit->page->values.capacity()});
            });
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, ContactPayloadRole::IdentityPoolObject, sizeof(*this));
        contact_payload_buffer(visitor, this, ContactPayloadRole::IdentityBySlot, by_slot_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::IdentityPageEntries, pages_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::IdentityHashBuckets, buckets_);
        for (const auto& value : by_value_)
            contact_payload_node_value(visitor, this, ContactPayloadRole::IdentityOrderedValues, value);
        page_index_.visit_borrowed_payload(visitor, ContactPayloadRole::IdentityPageOwnerBuckets);
        if (persistent_)
        {
            contact_payload_object(
                visitor, persistent_.get(), ContactPayloadRole::IdentityPersistentRoot, sizeof(PersistentStorage));
            const auto visit_index = [&](const auto& node)
            {
                contact_payload_object(visitor, node.get(), ContactPayloadRole::IdentityPersistentIndex,
                    node->charge.bytes - persistent_ordered_detail::control_bytes, true);
            };
            PageDirectory::visit_pages(persistent_->pages,
                [&](const auto* address, std::size_t bytes)
                {
                    contact_payload_object(visitor, address, ContactPayloadRole::IdentityPersistentIndex,
                        bytes - persistent_ordered_detail::control_bytes, true);
                });
            ValueIndex::visit_pages(persistent_->values, visit_index);
            PageIndex::visit_pages(persistent_->owners, visit_index);
            if constexpr (owns_plane_backing)
            {
                PersistentPlaneOwnerIndex::visit_pages(persistent_->plane.index, visit_index);
                PersistentPlaneOwnerIndex::visit_pages(persistent_->plane.index,
                    [&](const auto& node)
                    {
                        for (const auto& row : node->rows)
                            contact_payload_object(visitor, row.owner.get(), ContactPayloadRole::IdentityPersistentRoot,
                                sizeof(PersistentPlaneOwnerIndex::Owner), true);
                    });
            }
        }
        for_each_page([&](const auto& entry) { visit_credit_payload(*entry.credit, visitor); });
    }
    [[nodiscard]] std::size_t lookup_bucket_capacity() const noexcept { return buckets_.capacity(); }
    [[nodiscard]] std::size_t owner_pages() const noexcept
    {
        return persistent_ ? PageDirectory::size(persistent_->pages) : pages_.size();
    }
    [[nodiscard]] std::size_t record_capacity() const noexcept { return owner_pages() * records_per_page; }
    [[nodiscard]] std::size_t source_only_metadata_bytes() const noexcept
    {
        return persistent_ ? persistent_storage_bytes : private_metadata_bytes();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return persistent_ ? persistent_->retired_overlap : 0; }
    [[nodiscard]] std::size_t inherited_page_bytes() const noexcept
    {
        return persistent_ ? persistent_->foreign_page_bytes : 0;
    }
    /** Transfer imported backing ownership without releasing live arena bytes. */
    void transfer_inherited_page_bytes(std::size_t bytes)
    {
        if (!bytes)
            return;
        if (!persistent_ || bytes > persistent_->foreign_page_bytes)
            throw std::logic_error("Identity backing transfer exceeds the imported page credit.");
        persistent_->foreign_page_bytes -= bytes;
        charged_ -= bytes;
    }
    /** A whole directory replacement still overlaps its accepted source. */
    [[nodiscard]] std::size_t take_replaced_overlap() noexcept
    {
        if (!persistent_ || !persistent_->overlapping)
            return 0;
        const auto bytes = std::exchange(persistent_->foreign_page_bytes, 0) +
                           std::exchange(persistent_->foreign_index_bytes, 0) +
                           std::exchange(persistent_->retired_overlap, 0);
        charged_ -= bytes;
        return bytes;
    }
    void finish_overlap() noexcept
    {
        if (persistent_)
        {
            release(std::exchange(persistent_->retired_overlap, 0));
            persistent_->overlapping = false;
        }
    }

    /** A captured reference addresses a directory page and a record within it;
     * it is deliberately a separate type from a logical insertion slot. */
    struct CapturedReference
    {
        std::uint64_t value;
        bool operator==(const CapturedReference&) const = default;
    };
    class PageView
    {
    public:
        ~PageView()
        {
            decltype(directory_)().swap(directory_);
            decltype(index_)().swap(index_);
            budget_->used -= charged_;
        }
        PageView(const PageView&) = delete;
        PageView& operator=(const PageView&) = delete;
        [[nodiscard]] CapturedReference capture(const IdentityPool& pool, Slot slot)
        {
            if (frozen_ || pool.budget_ != budget_)
                throw std::invalid_argument("Invalid frozen identity page capture.");
            (void) pool.get(slot);
            const auto* entry = &pool.page_for(slot);
            const auto offset = slot - entry->first_slot;
            if (offset >= entry->credit->page->values.size())
                throw std::invalid_argument("Invalid identity page locator.");
            auto found = index_.find(entry->credit.get());
            std::size_t page_index;
            if (found == index_.end())
            {
                page_index = directory_.size();
                if (page_index > (std::numeric_limits<std::uint64_t>::max() - offset) / records_per_page)
                    throw std::invalid_argument("Captured identity reference overflows.");
                if (directory_.size() == directory_.capacity())
                {
                    const auto previous = directory_.capacity();
                    if (previous > std::numeric_limits<std::size_t>::max() / 2)
                        throw std::invalid_argument("Frozen page directory capacity overflows.");
                    const auto next = previous ? previous * 2 : 1;
                    admit(product(next, sizeof(typename decltype(directory_)::value_type)));
                    directory_.reserve(next);
                    if (directory_.capacity() != next)
                        throw std::invalid_argument("Frozen page directory capacity differs from admission.");
                    release(product(previous, sizeof(typename decltype(directory_)::value_type)));
                }
                admit(index_node_bytes);
                index_.emplace(entry->credit.get(), page_index);
                directory_.push_back(entry->credit);
            }
            else
                page_index = found->second;
            return {static_cast<std::uint64_t>(page_index) * records_per_page + offset};
        }
        struct CapturedPage
        {
            Slot first_slot;
            std::size_t count;
            CapturedReference first_reference;
        };
        /** The captured physical page keeps these logical source slots
         * contiguous. Store one range per page when a whole loop uses slots. */
        [[nodiscard]] CapturedPage capture_page(const IdentityPool& pool, Slot slot)
        {
            const auto ref = capture(pool, slot);
            const auto offset = ref.value % records_per_page;
            const auto& page = directory_[ref.value / records_per_page]->page;
            return {slot - static_cast<Slot>(offset), page->values.size(), {ref.value - offset}};
        }
        void freeze()
        {
            if (frozen_)
                return;
            for (const auto& credit : directory_)
                credit->page->frozen = true;
            const auto bytes = product(index_.size(), index_node_bytes);
            index_.clear();
            release(bytes);
            frozen_ = true;
        }
        [[nodiscard]] const T& get(CapturedReference reference) const
        {
            if (!frozen_)
                throw std::invalid_argument("Identity page batch is not frozen.");
            const auto page = reference.value / records_per_page;
            const auto offset = reference.value % records_per_page;
            if (page >= directory_.size() || offset >= directory_[page]->page->values.size())
                throw std::out_of_range("Invalid captured identity reference.");
            return directory_[page]->page->values[offset];
        }
        /** Import one immutable physical page, never the originating directory.
         * The optional pool is consulted synchronously for an existing credit
         * in this arena and is not retained by the frozen result. */
        [[nodiscard]] CapturedReference import_reference(const PageView& source, CapturedReference reference,
            const IdentityPool* target_pool = nullptr, bool shared_physical_admission = false)
        {
            if (frozen_ || !source.frozen_ || (target_pool && target_pool->budget_ != budget_))
                throw std::invalid_argument("Invalid selective identity-page import.");
            (void) source.get(reference);
            const auto& source_credit = source.directory_[reference.value / records_per_page];
            const auto offset = reference.value % records_per_page;
            for (std::size_t i = 0; i < directory_.size(); ++i)
                if (directory_[i]->page.get() == source_credit->page.get())
                    return {static_cast<std::uint64_t>(i) * records_per_page + offset};
            std::shared_ptr<PageCredit> credit;
            if (source_credit->budget == budget_ || (shared_physical_admission && source_credit->persistent_owner))
                credit = source_credit;
            else if (target_pool)
            {
                if (const auto* found = target_pool->find_page(source_credit->page.get()))
                    credit = found->credit;
            }
            if (!credit)
            {
                constexpr auto overhead = sizeof(PageCredit) + 128;
                budget_->add(overhead);
                try
                {
                    credit = std::make_shared<PageCredit>(*budget_, keeper_, source_credit->page);
                }
                catch (...)
                {
                    budget_->used -= overhead;
                    throw;
                }
                credit->bytes = overhead;
                credit->admit(source_credit->bytes - overhead);
            }
            const auto i = directory_.size();
            if (i > (std::numeric_limits<std::uint64_t>::max() - offset) / records_per_page)
                throw std::invalid_argument("Selective identity-page reference overflows.");
            if (directory_.size() == directory_.capacity())
            {
                const auto old = directory_.capacity();
                if (old > std::numeric_limits<std::size_t>::max() / 2)
                    throw std::invalid_argument("Selective identity-page directory overflows.");
                const auto next = old ? 2 * old : 1;
                const auto bytes = product(next, sizeof(typename decltype(directory_)::value_type));
                admit(bytes);
                try
                {
                    directory_.reserve(next);
                    if (directory_.capacity() != next)
                        throw std::invalid_argument("Selective page capacity differs from admission.");
                }
                catch (...)
                {
                    release(bytes);
                    throw;
                }
                release(product(old, sizeof(typename decltype(directory_)::value_type)));
            }
            directory_.push_back(std::move(credit));
            return {static_cast<std::uint64_t>(i) * records_per_page + offset};
        }
        [[nodiscard]] const void* physical_page_identity(CapturedReference reference) const
        {
            (void) get(reference);
            return directory_[reference.value / records_per_page]->page.get();
        }
        [[nodiscard]] const void* page_credit_identity(CapturedReference reference) const
        {
            (void) get(reference);
            return directory_[reference.value / records_per_page].get();
        }
        [[nodiscard]] std::size_t imported_credit_overhead_bytes() const noexcept { return 0; }
        template<class Visitor> void visit_page_admissions(Visitor&& visitor) const
        {
            for (const auto& credit : directory_)
                visitor(&credit->bytes, credit->page.get(), sizeof(PageCredit) + 128,
                    credit->bytes - sizeof(PageCredit) - 128);
        }
        template<class Visitor> void visit_borrowed_pages(Visitor&& visitor) const
        {
            for (const auto& credit : directory_)
                visitor(ContactBorrowedPage<T>{credit.get(), credit->page.get(), sizeof(PageCredit), sizeof(Page),
                    credit->page->values, credit->page->values.capacity()});
        }
        [[nodiscard]] const void* metadata_identity() const noexcept { return &charged_; }
        template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
        {
            contact_payload_object(visitor, this, ContactPayloadRole::PageViewObject, sizeof(*this), true);
            contact_payload_buffer(visitor, this, ContactPayloadRole::PageViewDirectory, directory_);
            for (const auto& value : index_)
                contact_payload_node_value(visitor, this, ContactPayloadRole::PageViewCaptureValues, value);
            for (const auto& credit : directory_)
                visit_credit_payload(*credit, visitor);
        }
        [[nodiscard]] std::size_t page_count() const noexcept { return directory_.size(); }
        [[nodiscard]] std::size_t page_capacity() const noexcept { return directory_.capacity(); }
        [[nodiscard]] std::size_t capture_index_nodes() const noexcept { return index_.size(); }
        [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charged_; }
        [[nodiscard]] bool frozen() const noexcept { return frozen_; }
        [[nodiscard]] const Budget* origin_arena() const noexcept { return budget_; }
        /** A borrowed directory also retains its originating credit objects.
         * A different arena accounts these objects separately from the page
         * bodies already admitted by that arena's independently forked pool. */
        template<class Visitor> void visit_credit_owners(Visitor&& visitor) const
        {
            for (const auto& credit : directory_)
                visitor(std::shared_ptr<const PageCredit>(credit), sizeof(PageCredit) + 128);
        }

    private:
        friend class IdentityPool;
        static constexpr auto index_node_bytes =
            sizeof(std::pair<const PageCredit* const, std::size_t>) + 4 * sizeof(void*);
        PageView(Budget& budget, std::shared_ptr<Budget> keeper) : budget_(&budget), keeper_(std::move(keeper)) {}
        void admit(std::size_t bytes)
        {
            if (bytes > std::numeric_limits<std::size_t>::max() - charged_)
                throw std::invalid_argument("Frozen identity page metadata overflows.");
            budget_->add(bytes);
            charged_ += bytes;
        }
        void release(std::size_t bytes)
        {
            charged_ -= bytes;
            budget_->used -= bytes;
        }
        Budget* budget_;
        std::shared_ptr<Budget> keeper_;
        std::size_t charged_ = 0;
        bool frozen_ = false;
        std::vector<std::shared_ptr<PageCredit>> directory_;
        std::map<const PageCredit*, std::size_t> index_;
    };
    [[nodiscard]] std::shared_ptr<PageView> page_view()
    {
        if (!keeper_)
            throw std::invalid_argument("Escaping identity pages require a shared budget keeper.");
        constexpr auto bytes = sizeof(PageView) + 128;
        budget_->add(bytes);
        try
        {
            auto result = std::shared_ptr<PageView>(new PageView(*budget_, keeper_));
            result->charged_ = bytes;
            return result;
        }
        catch (...)
        {
            budget_->used -= bytes;
            throw;
        }
    }

private:
    static constexpr Slot empty_slot = std::numeric_limits<Slot>::max();
    std::uint64_t scope_id_ = identity_pool_detail::next_scope();
    static std::size_t checked_sum(std::size_t a, std::size_t b)
    {
        return persistent_ordered_detail::checked_sum(a, b);
    }
    [[nodiscard]] std::uint64_t hash_value(const T& value) const
    {
        if constexpr (Hasher::enabled)
            return hasher_(value);
        return 0;
    }
    [[nodiscard]] const PageEntry& page_at(std::size_t ordinal) const
    {
        return persistent_ ? PageDirectory::at(persistent_->pages, ordinal) : pages_[ordinal];
    }
    [[nodiscard]] const PageEntry& page_for(Slot slot) const
    {
        if (persistent_)
        {
            const auto* found = PageDirectory::lower_bound(persistent_->pages, [&](const PageEntry& page)
                { return slot >= page.first_slot && slot - page.first_slot >= page.credit->page->values.size(); });
            if (!found || slot < found->first_slot)
                throw std::logic_error("Identity slot directory is inconsistent.");
            return *found;
        }
        const auto found = std::upper_bound(pages_.begin(), pages_.end(), slot,
            [](Slot value, const PageEntry& page) { return value < page.first_slot; });
        return *std::prev(found);
    }
    [[nodiscard]] const PageEntry* find_page(const void* address, bool credit = false) const noexcept
    {
        if (persistent_)
        {
            const PageSelector key{address, credit, 0};
            const auto* found =
                PageIndex::lower_bound(persistent_->owners, [&](const auto& row) { return PageLess{}(row, key); });
            return found && !PageLess{}(key, *found) ? &page_at(found->ordinal) : nullptr;
        }
        const auto found = page_index_.find(address);
        return found ? &pages_[*found] : nullptr;
    }
    template<class Visitor> void for_each_page(const Visitor& visit) const
    {
        if (persistent_)
        {
            for (std::size_t ordinal = 0; ordinal < owner_pages(); ++ordinal)
                visit(page_at(ordinal));
        }
        else
        {
            for (const auto& entry : pages_)
                visit(entry);
        }
    }
    [[nodiscard]] std::size_t persistent_index_bytes() const noexcept
    {
        auto bytes = PageDirectory::storage_bytes(persistent_->pages) +
                     (persistent_->values ? persistent_->values->subtree_bytes : 0) +
                     (persistent_->owners ? persistent_->owners->subtree_bytes : 0);
        if constexpr (owns_plane_backing)
            if (persistent_->plane.index)
                bytes += persistent_->plane.index->subtree_bytes;
        return bytes;
    }
    template<class Node> [[nodiscard]] std::size_t foreign_ordered_bytes(const std::shared_ptr<const Node>& node) const
    {
        return node ? node->subtree_bytes - (node->charge.arena == keeper_ ? node->same_arena_bytes : 0) : 0;
    }
    template<class U> [[nodiscard]] Slot intern_persistent(U&& value)
    {
        auto& current = *persistent_;
        const auto slot = current.records;
        if (slot == empty_slot)
            throw std::invalid_argument("Identity pool slot capacity exhausted.");
        const auto payload = payload_(value);
        const auto hash = hash_value(value);
        const auto count = owner_pages();
        const auto* last = count ? &page_at(count - 1) : nullptr;
        const bool new_page = !last || last->credit->budget != budget_ || last->credit->page.use_count() != 1 ||
                              last->credit->page->frozen || last->credit->page->values.size() == records_per_page;
        auto next = current;
        auto credit = last ? last->credit : nullptr;
        if (new_page)
        {
            auto body = sizeof(Page) + 128 + product(records_per_page, sizeof(T));
            if constexpr (owns_plane_backing)
                body = checked_sum(body, product(records_per_page, sizeof(PlaneOwner)));
            next.body_bytes = checked_sum(next.body_bytes, body);
            credit = make_credit({}, body);
            auto page = std::make_shared<Page>();
            page->body_origin = budget_;
            page->body_scope = scope_id_;
            if constexpr (owns_plane_backing)
            {
                page->plane_owners.reserve(records_per_page);
                if (page->plane_owners.capacity() != records_per_page)
                    throw std::invalid_argument("Identity page owner capacity differs from admission.");
            }
            page->values.reserve(records_per_page);
            if (page->values.capacity() != records_per_page)
                throw std::invalid_argument("Identity page capacity differs from admission.");
            credit->page = std::move(page);
            next.pages = PageDirectory(keeper_).append(next.pages, PageEntry{credit, slot});
            const PageIndex index(keeper_);
            const PageSelector body_key{credit->page.get(), false, count};
            const PageSelector credit_key{credit.get(), true, count};
            auto with_body = index.edit(next.owners, body_key, &body_key, false);
            next.owners = index.edit(with_body, credit_key, &credit_key, false);
        }
        next.body_bytes = checked_sum(next.body_bytes, payload);
        credit->admit(payload);
        auto& values = credit->page->values;
        const auto old_size = values.size();
        try
        {
            values.push_back(std::forward<U>(value));
            if (payload_(values.back()) > payload)
                throw std::invalid_argument("Identity owner payload exceeds its admission.");
            PlaneOwner plane_owner;
            if constexpr (owns_plane_backing)
            {
                auto retained = PersistentPlaneOwnerIndex(keeper_).retain(next.plane.index, values.back());
                next.plane.index = std::move(retained.index);
                next.plane.bytes = checked_sum(next.plane.bytes, retained.new_payload);
                plane_owner = std::move(retained.owner);
            }
            const ValueSelector selector{hash, &values.back(), slot};
            next.values = ValueIndex(keeper_).edit(next.values, selector, &selector, false);
            ++next.records;
            next.foreign_index_bytes = checked_sum(PageDirectory(keeper_).inherited_bytes(next.pages),
                checked_sum(foreign_ordered_bytes(next.values), foreign_ordered_bytes(next.owners)));
            if constexpr (owns_plane_backing)
                next.foreign_index_bytes =
                    checked_sum(next.foreign_index_bytes, foreign_ordered_bytes(next.plane.index));
            if (next.foreign_index_bytes > current.foreign_index_bytes)
                throw std::logic_error("Identity pool append introduced foreign metadata.");
            const auto retired = current.foreign_index_bytes - next.foreign_index_bytes;
            if (current.overlapping)
                next.retired_overlap = checked_sum(next.retired_overlap, retired);
            else
                release(retired);
            if constexpr (owns_plane_backing)
                credit->page->plane_owners.push_back(std::move(plane_owner));
            current = std::move(next);
        }
        catch (...)
        {
            if (values.size() != old_size)
                values.pop_back();
            credit->release(payload);
            throw;
        }
        return slot;
    }
    void copy_flat_metadata(const IdentityPool& source)
    {
        reserve_exact(by_slot_, source.size());
        reserve_exact(pages_, source.owner_pages());
        page_index_.reserve(source.owner_pages());
        if constexpr (!Hasher::enabled)
            admit(product(source.size(), node_bytes()));
        source.for_each_page(
            [&](const PageEntry& entry)
            {
                auto credit = make_credit(entry.credit->page, entry.credit->bytes - credit_overhead);
                const auto ordinal = pages_.size();
                pages_.push_back({std::move(credit), entry.first_slot});
                page_index_.insert(entry.credit->page.get(), ordinal);
                for (const auto& value : entry.credit->page->values)
                {
                    const auto slot = by_slot_.size();
                    if constexpr (Hasher::enabled)
                    {
                        ensure_lookup_capacity();
                        buckets_[find_bucket(value, hasher_(value))] = slot;
                    }
                    else
                        by_value_.emplace(&value, slot);
                    by_slot_.push_back(&value);
                }
            });
    }
    [[nodiscard]] std::shared_ptr<PageCredit> make_credit(std::shared_ptr<Page> page, std::size_t bytes)
    {
        // Admit the control object before constructing it. Its destructor owns
        // exactly this arena's credit, independently of shared page bodies.
        constexpr auto overhead = sizeof(PageCredit) + 128;
        budget_->add(overhead);
        std::shared_ptr<PageCredit> result;
        try
        {
            result = std::make_shared<PageCredit>(*budget_, keeper_, std::move(page), scope_id_, bool(persistent_));
        }
        catch (...)
        {
            budget_->used -= overhead;
            throw;
        }
        result->bytes = overhead;
        result->admit(bytes);
        return result;
    }
    template<class U> [[nodiscard]] Slot intern_value(U&& value)
    {
        if (const auto found = find(value))
            return *found;
        if (persistent_)
            return intern_persistent(std::forward<U>(value));
        if (by_slot_.size() == std::numeric_limits<Slot>::max())
            throw std::invalid_argument("Identity pool slot capacity exhausted.");
        grow(by_slot_);
        std::uint64_t hash = 0;
        if constexpr (Hasher::enabled)
        {
            hash = hasher_(value);
            ensure_lookup_capacity();
        }
        if (pages_.empty() || pages_.back().credit->page.use_count() != 1 || pages_.back().credit->page->frozen ||
            pages_.back().credit->page->values.size() == records_per_page)
        {
            grow(pages_);
            page_index_.reserve(pages_.size() + 1);
            // Keep the existing conservative shared-control allowance, once
            // per page. Values have no individual control blocks or owners.
            const auto bytes = sizeof(Page) + 128 + product(records_per_page, sizeof(T));
            auto credit = make_credit({}, bytes);
            auto page = std::make_shared<Page>();
            page->body_origin = budget_;
            page->body_scope = scope_id_;
            page->values.reserve(records_per_page);
            if (page->values.capacity() != records_per_page)
                throw std::invalid_argument("Identity page capacity differs from admission.");
            credit->page = std::move(page);
            pages_.push_back({std::move(credit), by_slot_.size()});
            page_index_.insert(pages_.back().credit->page.get(), pages_.size() - 1);
        }
        const auto payload = payload_(value);
        pages_.back().credit->admit(payload);
        if constexpr (!Hasher::enabled)
            admit(node_bytes());
        auto& values = pages_.back().credit->page->values;
        values.push_back(std::forward<U>(value));
        if (payload_(values.back()) > payload)
            throw std::invalid_argument("Identity owner payload exceeds its admission.");
        const auto* owner = &values.back();
        const auto slot = by_slot_.size();
        if constexpr (Hasher::enabled)
        {
            const auto bucket = find_bucket(*owner, hash);
            if (buckets_[bucket] != empty_slot)
                throw std::invalid_argument("Duplicate identity pool key.");
            by_slot_.push_back(owner);
            buckets_[bucket] = slot;
        }
        else
        {
            if (!by_value_.emplace(owner, slot).second)
                throw std::invalid_argument("Duplicate identity pool key.");
            by_slot_.push_back(owner);
        }
        return slot;
    }
    static std::uint64_t mixed_hash(std::uint64_t value) noexcept
    {
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
    [[nodiscard]] Slot find_bucket(const T& value, std::uint64_t hash) const
    {
        const auto mask = buckets_.size() - 1;
        auto bucket = static_cast<Slot>(mixed_hash(hash)) & mask;
        for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
        {
            const auto slot = buckets_[bucket];
            if (slot == empty_slot || (!(*by_slot_[slot] < value) && !(value < *by_slot_[slot])))
                return bucket;
            bucket = (bucket + 1) & mask;
        }
        throw std::invalid_argument("Identity lookup table has no empty slot.");
    }
    void ensure_lookup_capacity()
    {
        // Lookup only: full values decide identity, and creation slots preserve
        // all ordering. Seven-eighths occupancy leaves a terminating empty slot.
        if (!buckets_.empty() && size() + 1 <= buckets_.size() - buckets_.size() / 8)
            return;
        const auto old_capacity = buckets_.capacity();
        if (old_capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Identity lookup capacity overflows.");
        const auto capacity = old_capacity ? 2 * old_capacity : 8;
        const auto bytes = product(capacity, sizeof(Slot));
        admit(bytes);
        std::vector<Slot> next;
        try
        {
            next.reserve(capacity);
            if (next.capacity() != capacity)
                throw std::invalid_argument("Identity lookup capacity differs from admission.");
            next.assign(capacity, empty_slot);
            for (Slot slot = 0; slot < by_slot_.size(); ++slot)
            {
                auto bucket = static_cast<Slot>(mixed_hash(hasher_(*by_slot_[slot]))) & (capacity - 1);
                while (next[bucket] != empty_slot)
                    bucket = (bucket + 1) & (capacity - 1);
                next[bucket] = slot;
            }
        }
        catch (...)
        {
            std::vector<Slot>().swap(next);
            release(bytes);
            throw;
        }
        buckets_.swap(next);
        std::vector<Slot>().swap(next);
        release(product(old_capacity, sizeof(Slot)));
    }
    static constexpr std::size_t node_bytes() { return sizeof(std::pair<const T* const, Slot>) + 4 * sizeof(void*); }
    static std::size_t product(std::size_t a, std::size_t b)
    {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::invalid_argument("Identity pool size overflow.");
        return a * b;
    }
    template<class U> void reserve_exact(std::vector<U>& values, std::size_t n)
    {
        admit(product(n, sizeof(U)));
        values.reserve(n);
        if (values.capacity() != n)
            throw std::invalid_argument("Identity pool capacity differs from admission.");
    }
    template<class U> void grow(std::vector<U>& values)
    {
        if (values.size() != values.capacity())
            return;
        const auto old = values.capacity();
        if (old > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Identity pool capacity overflow.");
        reserve_exact(values, old ? 2 * old : 1);
        release(product(old, sizeof(U)));
    }
    void admit(std::size_t bytes)
    {
        if (bytes > std::numeric_limits<std::size_t>::max() - charged_)
            throw std::invalid_argument("Identity pool charge overflow.");
        budget_->add(bytes);
        charged_ += bytes;
    }
    void release(std::size_t bytes)
    {
        budget_->used -= bytes;
        charged_ -= bytes;
    }
    void clear() noexcept
    {
        persistent_.reset();
        decltype(by_value_)().swap(by_value_);
        decltype(by_slot_)().swap(by_slot_);
        decltype(pages_)().swap(pages_);
        decltype(buckets_)().swap(buckets_);
        budget_->used -= charged_;
        charged_ = 0;
    }
    Budget* budget_;
    std::shared_ptr<Budget> keeper_;
    [[no_unique_address]] Payload payload_;
    [[no_unique_address]] Hasher hasher_;
    std::size_t charged_ = 0;
    std::map<const T*, Slot, Less> by_value_;
    std::vector<const T*> by_slot_;
    std::vector<Slot> buckets_;
    std::vector<PageEntry> pages_;
    PhysicalOwnerIndex<Slot> page_index_;
    std::unique_ptr<PersistentStorage> persistent_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
