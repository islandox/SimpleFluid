#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
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
template<class T> class IdentityPool
{
public:
    using Slot = std::size_t;
    using PayloadBytes = std::size_t (*)(const T&);
    using Hash = std::uint64_t (*)(const T&);
    struct Less
    {
        using is_transparent = void;
        bool operator()(const T* a, const T* b) const { return *a < *b; }
        bool operator()(const T* a, const T& b) const { return *a < b; }
        bool operator()(const T& a, const T* b) const { return a < *b; }
    };
    IdentityPool(Budget& budget, PayloadBytes payload, Hash hash = nullptr)
        : budget_(&budget), payload_(payload), hash_(hash)
    {
    }
    IdentityPool(const IdentityPool& source, Budget& budget)
        : budget_(&budget), payload_(source.payload_), hash_(source.hash_)
    {
        try
        {
            admit(source.owner_bytes_);
            owner_bytes_ = source.owner_bytes_;
            admit(product(source.by_value_.size(), node_bytes()));
            reserve_exact(by_slot_, source.by_slot_.size());
            reserve_exact(pages_, source.pages_.size());
            reserve_exact(buckets_, source.buckets_.size());
            buckets_.assign(source.buckets_.begin(), source.buckets_.end());
            pages_.assign(source.pages_.begin(), source.pages_.end());
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
        if (hash_)
        {
            if (buckets_.empty())
                return {};
            const auto bucket = find_bucket(value, hash_(value));
            return buckets_[bucket] == empty_slot ? std::optional<Slot>{} : buckets_[bucket];
        }
        const auto found = by_value_.find(value);
        return found == by_value_.end() ? std::optional<Slot>{} : found->second;
    }
    [[nodiscard]] std::size_t size() const noexcept { return by_slot_.size(); }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return charged_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return by_value_.size() * node_bytes() + by_slot_.capacity() * sizeof(const T*) +
               pages_.capacity() * sizeof(std::shared_ptr<Page>) + buckets_.capacity() * sizeof(Slot);
    }
    [[nodiscard]] std::size_t lookup_bucket_capacity() const noexcept { return buckets_.capacity(); }
    [[nodiscard]] std::size_t owner_pages() const noexcept { return pages_.size(); }
    [[nodiscard]] std::size_t record_capacity() const noexcept { return pages_.size() * records_per_page; }

private:
    static constexpr std::size_t records_per_page = 16;
    static constexpr Slot empty_slot = std::numeric_limits<Slot>::max();
    struct Page
    {
        std::vector<T> values;
    };
    template<class U> [[nodiscard]] Slot intern_value(U&& value)
    {
        if (const auto found = find(value))
            return *found;
        if (by_slot_.size() == std::numeric_limits<Slot>::max())
            throw std::invalid_argument("Identity pool slot capacity exhausted.");
        grow(by_slot_);
        const auto hash = hash_ ? hash_(value) : 0;
        if (hash_)
            ensure_lookup_capacity();
        if (pages_.empty() || pages_.back().use_count() != 1 || pages_.back()->values.size() == records_per_page)
        {
            grow(pages_);
            // Keep the existing conservative shared-control allowance, once
            // per page. Values have no individual control blocks or owners.
            const auto bytes = sizeof(Page) + 128 + product(records_per_page, sizeof(T));
            admit(bytes);
            auto page = std::make_shared<Page>();
            page->values.reserve(records_per_page);
            if (page->values.capacity() != records_per_page)
                throw std::invalid_argument("Identity page capacity differs from admission.");
            pages_.push_back(std::move(page));
            owner_bytes_ += bytes;
        }
        const auto payload = payload_(value);
        admit(payload);
        if (!hash_)
            admit(node_bytes());
        auto& values = pages_.back()->values;
        values.push_back(std::forward<U>(value));
        if (payload_(values.back()) > payload)
            throw std::invalid_argument("Identity owner payload exceeds its admission.");
        const auto* owner = &values.back();
        const auto slot = by_slot_.size();
        if (hash_)
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
        owner_bytes_ += payload;
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
                auto bucket = static_cast<Slot>(mixed_hash(hash_(*by_slot_[slot]))) & (capacity - 1);
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
        charged_ = owner_bytes_ = 0;
    }
    Budget* budget_;
    PayloadBytes payload_;
    Hash hash_;
    std::size_t charged_ = 0, owner_bytes_ = 0;
    std::map<const T*, Slot, Less> by_value_;
    std::vector<const T*> by_slot_;
    std::vector<Slot> buckets_;
    std::vector<std::shared_ptr<Page>> pages_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
