#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "ContactDomainStorage.hh"
#include "ContactPayloadCensus.hh"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct PhysicalOwnerHasher
{
    std::uint64_t operator()(const void* owner) const noexcept
    {
        auto value = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(owner));
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
};
/** Admitted owner lookup. Pointer equality, never a hash or an arena address,
 * establishes physical owner identity. Hash policies are compile-time types. */
template<class Value, class Hasher = PhysicalOwnerHasher> class PhysicalOwnerIndex
{
    struct Bucket { const void* owner = nullptr; Value value{}; bool retired = false; };
public:
    explicit PhysicalOwnerIndex(Budget& budget, Hasher hasher = {}) : budget_(&budget), hasher_(std::move(hasher)) {}
    PhysicalOwnerIndex(const PhysicalOwnerIndex& source, Budget& budget) : PhysicalOwnerIndex(budget, source.hasher_)
    {
        allocate_copy(source.buckets_.size());
        try { buckets_.assign(source.buckets_.begin(), source.buckets_.end()); }
        catch (...) { budget_->used -= charged_; charged_ = 0; throw; }
        size_ = source.size_;
        retired_ = source.retired_;
    }
    ~PhysicalOwnerIndex() { decltype(buckets_)().swap(buckets_); budget_->used -= charged_; }
    PhysicalOwnerIndex(const PhysicalOwnerIndex&) = delete;
    PhysicalOwnerIndex& operator=(const PhysicalOwnerIndex&) = delete;
    [[nodiscard]] const Value* find(const void* owner) const
    {
        if (!owner || buckets_.empty()) return nullptr;
        auto slot = static_cast<std::size_t>(hasher_(owner)) & (buckets_.size() - 1);
        for (std::size_t n = 0; n < buckets_.size(); ++n)
        {
            ++probes_;
            const auto& bucket = buckets_[slot];
            if (!bucket.owner && !bucket.retired) return nullptr;
            if (bucket.owner == owner) return &bucket.value;
            slot = (slot + 1) & (buckets_.size() - 1);
        }
        return nullptr;
    }
    void reserve(std::size_t count)
    {
        std::size_t capacity = 8;
        while (count > capacity - capacity / 4)
        {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                throw std::invalid_argument("Physical owner index capacity overflows.");
            capacity *= 2;
        }
        if (capacity <= buckets_.size() && size_ + retired_ + 1 <= buckets_.size() - buckets_.size() / 4)
            return;
        capacity = std::max(capacity, buckets_.size());
        const auto bytes = product(capacity), old = charged_;
        budget_->add(bytes);
        ContactVector<Bucket> next;
        try
        {
            next.reserve(capacity);
            if (next.capacity() != capacity)
                throw std::invalid_argument("Physical owner index capacity differs from admission.");
            next.resize(capacity);
            for (const auto& bucket : buckets_)
                if (bucket.owner)
                {
                    auto slot = static_cast<std::size_t>(hasher_(bucket.owner)) & (capacity - 1);
                    while (next[slot].owner) { ++probes_; slot = (slot + 1) & (capacity - 1); }
                    next[slot] = bucket;
                }
        }
        catch (...) { decltype(next)().swap(next); budget_->used -= bytes; throw; }
        buckets_.swap(next);
        decltype(next)().swap(next);
        budget_->used -= old;
        charged_ = bytes;
        retired_ = 0;
    }
    void insert(const void* owner, Value value)
    {
        if (!owner) throw std::invalid_argument("Null physical owner identity.");
        if (find(owner)) throw std::invalid_argument("Duplicate physical owner identity.");
        reserve(size_ + 1);
        auto slot = static_cast<std::size_t>(hasher_(owner)) & (buckets_.size() - 1);
        while (buckets_[slot].owner) { ++probes_; slot = (slot + 1) & (buckets_.size() - 1); }
        if (buckets_[slot].retired) --retired_;
        buckets_[slot] = {owner, std::move(value), false};
        ++size_;
    }
    bool erase(const void* owner)
    {
        if (!find(owner)) return false;
        auto slot = static_cast<std::size_t>(hasher_(owner)) & (buckets_.size() - 1);
        while (buckets_[slot].owner != owner) slot = (slot + 1) & (buckets_.size() - 1);
        buckets_[slot].value = {};
        buckets_[slot].owner = nullptr;
        buckets_[slot].retired = true;
        --size_;
        ++retired_;
        return true;
    }
    template<class Visitor> void visit(Visitor&& visitor) const
    {
        for (const auto& bucket : buckets_) if (bucket.owner) visitor(bucket.owner, bucket.value);
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charged_; }
    [[nodiscard]] const void* metadata_identity() const noexcept { return &charged_; }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor,
        ContactPayloadRole role = ContactPayloadRole::PhysicalOwnerBuckets, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, ContactPayloadRole::PhysicalOwnerIndexObject, sizeof(*this));
        contact_payload_buffer(visitor, this, role, buckets_, size_);
    }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return buckets_.size(); }
    [[nodiscard]] std::size_t probes() const noexcept { return probes_; }
private:
    static std::size_t product(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(Bucket))
            throw std::invalid_argument("Physical owner index admission overflows.");
        return n * sizeof(Bucket);
    }
    void allocate_copy(std::size_t n)
    {
        const auto bytes = product(n);
        budget_->add(bytes);
        try
        {
            buckets_.reserve(n);
            if (buckets_.capacity() != n)
                throw std::invalid_argument("Physical owner copy capacity differs from admission.");
        }
        catch (...) { budget_->used -= bytes; throw; }
        charged_ = bytes;
    }
    Budget* budget_;
    [[no_unique_address]] Hasher hasher_;
    ContactVector<Bucket> buckets_;
    std::size_t size_ = 0, charged_ = 0, retired_ = 0;
    mutable std::size_t probes_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
