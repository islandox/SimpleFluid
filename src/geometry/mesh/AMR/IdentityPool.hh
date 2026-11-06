#pragma once
#include "ContactStorageBreakdown.hh"
#include "IdentityPoolBorrowedPayload.hh"
#include "IdentityPoolHash.hh"
#include "IdentityPoolPayload.hh"
#include "PhysicalOwnerIndex.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Full identities reside in immutable shared pages. A fork copies only its
 * lookup metadata; neither branch appends into a shared page. Slots never
 * participate in numerical ordering and retired identities remain admitted
 * until session destruction. This storage is independent of PLIC snapshots. */
template<class T, class Hasher = IdentityPoolHasher<T>, class Payload = IdentityPoolPayload<T>> class IdentityPool
{
    static constexpr std::size_t records_per_page = 16;
    struct Page
    {
        std::vector<T> values;
        bool frozen = false;
    };
    struct PageCredit
    {
        Budget* budget;
        std::shared_ptr<Budget> keeper;
        std::shared_ptr<Page> page;
        std::size_t bytes = 0;
        PageCredit(Budget& b, std::shared_ptr<Budget> k, std::shared_ptr<Page> p)
            : budget(&b), keeper(std::move(k)), page(std::move(p))
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
    };
    struct PageEntry
    {
        std::shared_ptr<PageCredit> credit;
        std::size_t first_slot;
    };

    template<class Visitor> static void visit_credit_payload(const PageCredit& credit, Visitor& visitor)
    {
        contact_payload_object(
            visitor, &credit, ContactPayloadRole::IdentityPageCreditObject, sizeof(PageCredit), true);
        const auto& page = *credit.page;
        contact_payload_object(visitor, &page, ContactPayloadRole::IdentityPageObject, sizeof(Page), true);
        contact_payload_buffer(visitor, &page, ContactPayloadRole::IdentityPageValues, page.values, page.values.size(),
            !IdentityPoolBorrowedPayload<T>::complete);
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
    IdentityPool(Budget& budget, std::shared_ptr<Budget> keeper = {}, Hasher hasher = {}, Payload payload = {})
        : budget_(&budget), keeper_(std::move(keeper)), payload_(std::move(payload)), hasher_(std::move(hasher)),
          page_index_(budget)
    {
        if (keeper_ && keeper_.get() != &budget)
            throw std::invalid_argument("Identity pool budget keeper mismatch.");
    }
    IdentityPool(const IdentityPool& source, Budget& budget, std::shared_ptr<Budget> keeper = {})
        : IdentityPool(budget, std::move(keeper), source.hasher_, source.payload_)
    {
        try
        {
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
        if (slot >= by_slot_.size())
            throw std::invalid_argument("Invalid identity pool slot.");
        return *by_slot_[slot];
    }
    [[nodiscard]] std::optional<Slot> find(const T& value) const
    {
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
        const auto equal = [&](Slot slot)
        {
            const auto& stored = *by_slot_[slot];
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
        for (Slot slot = 0; slot < by_slot_.size(); ++slot)
            if (equal(slot))
                return slot;
        return {};
    }
    [[nodiscard]] std::size_t size() const noexcept { return by_slot_.size(); }
    [[nodiscard]] std::size_t retained_bytes() const noexcept
    {
        auto total = charged_ + page_index_.metadata_bytes();
        for (const auto& entry : pages_)
            total += entry.credit->bytes;
        return total;
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return by_value_.size() * node_bytes() + by_slot_.capacity() * sizeof(const T*) +
               pages_.capacity() * sizeof(PageEntry) + buckets_.capacity() * sizeof(Slot) +
               page_index_.metadata_bytes();
    }
    [[nodiscard]] std::size_t page_credit_overhead_bytes() const noexcept
    {
        return pages_.size() * (sizeof(PageCredit) + 128);
    }
    [[nodiscard]] bool owns_page_credit(const void* identity) const noexcept
    {
        return std::any_of(
            pages_.begin(), pages_.end(), [identity](const auto& entry) { return entry.credit.get() == identity; });
    }
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept
    {
        return metadata_bytes() + page_credit_overhead_bytes();
    }
    [[nodiscard]] std::size_t shared_body_bytes() const noexcept
    {
        std::size_t bytes = 0;
        for (const auto& entry : pages_)
            bytes += entry.credit->bytes - sizeof(PageCredit) - 128;
        return bytes;
    }
    [[nodiscard]] std::size_t page_owner_lookup_probes() const noexcept { return page_index_.probes(); }
    [[nodiscard]] const void* metadata_identity() const noexcept { return &charged_; }
    [[nodiscard]] bool owns_page_admission(const void* body, const void* metadata) const noexcept
    {
        const auto found = page_index_.find(body);
        return found && &pages_[*found].credit->bytes == metadata;
    }
    template<class Visitor> void visit_page_admissions(Visitor&& visitor) const
    {
        for (const auto& entry : pages_)
            visitor(&entry.credit->bytes, entry.credit->page.get(), sizeof(PageCredit) + 128,
                entry.credit->bytes - sizeof(PageCredit) - 128);
    }
    template<class Visitor> void visit_borrowed_pages(Visitor&& visitor) const
    {
        for (const auto& entry : pages_)
            visitor(ContactBorrowedPage<T>{entry.credit.get(), entry.credit->page.get(), sizeof(PageCredit),
                sizeof(Page), entry.credit->page->values, entry.credit->page->values.capacity()});
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
        for (const auto& entry : pages_)
            visit_credit_payload(*entry.credit, visitor);
    }
    [[nodiscard]] std::size_t lookup_bucket_capacity() const noexcept { return buckets_.capacity(); }
    [[nodiscard]] std::size_t owner_pages() const noexcept { return pages_.size(); }
    [[nodiscard]] std::size_t record_capacity() const noexcept { return pages_.size() * records_per_page; }

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
            auto entry = std::upper_bound(pool.pages_.begin(), pool.pages_.end(), slot,
                [](Slot value, const PageEntry& page) { return value < page.first_slot; });
            --entry;
            const auto offset = static_cast<std::size_t>(pool.by_slot_[slot] - entry->credit->page->values.data());
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
        [[nodiscard]] CapturedReference import_reference(
            const PageView& source, CapturedReference reference, const IdentityPool* target_pool = nullptr)
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
            if (source_credit->budget == budget_)
                credit = source_credit;
            else if (target_pool)
            {
                if (const auto found = target_pool->page_index_.find(source_credit->page.get()))
                    credit = target_pool->pages_[*found].credit;
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
                visitor(credit, sizeof(PageCredit) + 128);
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
    [[nodiscard]] std::shared_ptr<PageCredit> make_credit(std::shared_ptr<Page> page, std::size_t bytes)
    {
        // Admit the control object before constructing it. Its destructor owns
        // exactly this arena's credit, independently of shared page bodies.
        constexpr auto overhead = sizeof(PageCredit) + 128;
        budget_->add(overhead);
        std::shared_ptr<PageCredit> result;
        try
        {
            result = std::make_shared<PageCredit>(*budget_, keeper_, std::move(page));
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
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
