/** @file ImmutableSequence.hh
 * @brief Shared immutable, value-comparable tile metadata.
 */
#pragma once
#include <algorithm>
#include <compare>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Snapshot a vector builder into an independent immutable owner, including
 * rvalue builders: old mutable element aliases cannot modify the snapshot.
 * Copies share that owner; modifications require a separate vector builder and
 * replacement of the complete handle. This container neither issues geometric
 * authority nor proves incidence; storage identity denotes only an allocation. */
template<class T> class ImmutableSequence
{
    std::shared_ptr<const std::vector<T>> storage_;
    std::uint32_t offset_ = 0, count_ = 0;
    static std::uint32_t narrow_size(std::size_t count)
    {
        if (count > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("Immutable authority slice exceeds its index range.");
        return static_cast<std::uint32_t>(count);
    }

public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;
    using difference_type = typename std::vector<T>::difference_type;
    using const_iterator = typename std::span<const T>::iterator;
    ImmutableSequence() = default;
    ImmutableSequence(const ImmutableSequence&) = default;
    ImmutableSequence& operator=(const ImmutableSequence&) = default;
    ImmutableSequence(ImmutableSequence&& other) noexcept
        : storage_(std::move(other.storage_)), offset_(std::exchange(other.offset_, 0)),
          count_(std::exchange(other.count_, 0))
    {
    }
    ImmutableSequence& operator=(ImmutableSequence&& other) noexcept
    {
        if (this != &other)
        {
            storage_ = std::move(other.storage_);
            offset_ = std::exchange(other.offset_, 0);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    ImmutableSequence(const std::vector<T>& input)
    {
        count_ = narrow_size(input.size());
        if (!input.empty())
            storage_ = std::make_shared<const std::vector<T>>(input);
    }
    ImmutableSequence(std::vector<T>&& input) : ImmutableSequence(std::as_const(input)) {}
    ImmutableSequence(std::initializer_list<T> input) : ImmutableSequence(std::vector<T>(input)) {}
    /** Trusted construction boundary only: the fresh local builder must have
     * no escaped mutable aliases, and its actual capacity must already be
     * admitted. Only the new shared owner/control allocation is added here. */
    template<class Budget> static ImmutableSequence freeze_builder(std::vector<T>&& input, Budget& budget)
    {
        ImmutableSequence result;
        result.count_ = narrow_size(input.size());
        if (input.capacity())
        {
            budget.add(owner_bytes());
            result.storage_ = std::make_shared<const std::vector<T>>(std::move(input));
        }
        return result;
    }
    /** Copy directly from a bounded input span, admitting owner and payload
     * before either allocation. No intermediate mutable vector is allocated. */
    template<class Budget> static ImmutableSequence snapshot(std::span<const T> input, Budget& budget)
    {
        ImmutableSequence result;
        result.count_ = narrow_size(input.size());
        if (!input.empty())
        {
            budget.add(allocation_bytes(input.size()));
            result.storage_ = std::make_shared<const std::vector<T>>(input.begin(), input.end());
            if (result.storage_capacity() != input.size())
                throw std::length_error("Immutable authority snapshot capacity differs from admission.");
        }
        return result;
    }
    [[nodiscard]] std::span<const T> span() const noexcept
    {
        return storage_ ? std::span<const T>(*storage_).subspan(offset_, count_) : std::span<const T>{};
    }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    [[nodiscard]] size_type size() const noexcept { return count_; }
    /** Logical slice capacity; the entire backing allocation is reported only
     * by storage_capacity()/storage_bytes(), never by a first slice's length. */
    [[nodiscard]] size_type capacity() const noexcept { return count_; }
    [[nodiscard]] size_type storage_size() const noexcept { return storage_ ? storage_->size() : 0; }
    [[nodiscard]] size_type storage_capacity() const noexcept { return storage_ ? storage_->capacity() : 0; }
    [[nodiscard]] const T* data() const noexcept { return span().data(); }
    const T& operator[](size_type index) const { return span()[index]; }
    const T& at(size_type index) const
    {
        if (index >= size())
            throw std::out_of_range("Immutable authority slice index is outside its values.");
        return span()[index];
    }
    const T& front() const { return span().front(); }
    const T& back() const { return span().back(); }
    const_iterator begin() const noexcept { return span().begin(); }
    const_iterator end() const noexcept { return span().end(); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    [[nodiscard]] ImmutableSequence slice(size_type offset, size_type count) const
    {
        if (offset > size() || count > size() - offset)
            throw std::out_of_range("Immutable authority slice exceeds its source values.");
        ImmutableSequence result;
        result.offset_ = narrow_size(std::size_t(offset_) + offset);
        result.count_ = narrow_size(count);
        result.storage_ = storage_;
        return result;
    }
    [[nodiscard]] const void* storage_identity() const noexcept { return storage_.get(); }
    [[nodiscard]] std::shared_ptr<const void> owner_pin() const noexcept { return storage_; }
    [[nodiscard]] static constexpr std::size_t owner_bytes() noexcept
    {
        return sizeof(std::vector<T>) + 2 * sizeof(void*);
    }
    /** Actual vector capacity plus vector object and shared control allowance. */
    [[nodiscard]] static std::size_t allocation_bytes(std::size_t capacity)
    {
        if (capacity > (std::numeric_limits<std::size_t>::max() - owner_bytes()) / sizeof(T))
            throw std::length_error("Immutable authority allocation size overflows.");
        return owner_bytes() + capacity * sizeof(T);
    }
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return storage_ ? allocation_bytes(storage_capacity()) : 0;
    }
    friend bool operator==(const ImmutableSequence& a, const ImmutableSequence& b)
    {
        return std::ranges::equal(a.span(), b.span());
    }
    friend auto operator<=>(const ImmutableSequence& a, const ImmutableSequence& b)
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
    }
    friend bool operator==(const ImmutableSequence& a, const std::vector<T>& b)
    {
        return std::ranges::equal(a.span(), b);
    }
    friend auto operator<=>(const ImmutableSequence& a, const std::vector<T>& b)
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
    }
};

/** Traversal-local owner ledger. Tracking allocation participates in admission;
 * null empty sequences allocate neither a payload owner nor a tracking row. */
class ImmutableSequenceLedger
{
    std::set<std::shared_ptr<const void>, std::owner_less<std::shared_ptr<const void>>> owners_;

public:
    template<class T, class Budget> std::size_t retain(const ImmutableSequence<T>& values, Budget& budget)
    {
        auto owner = values.owner_pin();
        if (!owner || owners_.contains(owner))
            return 0;
        const auto bytes = values.storage_bytes();
        budget.add(bytes);
        budget.add(sizeof(owner) + 4 * sizeof(void*));
        owners_.insert(std::move(owner));
        return bytes;
    }
};

/** Short-lived exact-value interning. Pool entries only pin immutable handles;
 * the pool is destroyed after the retained catalog has received its copies. */
template<class T> class ImmutableSequencePool
{
    std::map<ImmutableSequence<T>, std::uint32_t> values_;
    ImmutableSequence<T> backing_;
    bool finalized_ = false;

public:
    template<class Budget> void intern(ImmutableSequence<T>& values, Budget& budget)
    {
        if (finalized_)
            throw std::logic_error("Immutable authority pool is already finalized.");
        if (!values.storage_identity())
            return;
        if (const auto found = values_.find(values); found != values_.end())
            values = found->first;
        else
        {
            budget.add(sizeof(typename decltype(values_)::value_type) + 4 * sizeof(void*));
            values_.emplace(values, 0);
        }
    }
    template<class Budget> void finalize(Budget& budget)
    {
        if (finalized_)
            throw std::logic_error("Immutable authority pool is already finalized.");
        std::size_t count = 0;
        for (const auto& [values, offset] : values_)
        {
            if (values.size() > std::numeric_limits<std::uint32_t>::max() - count)
                throw std::length_error("Immutable authority pool exceeds its slice index range.");
            count += values.size();
        }
        // Admit the complete new payload while all original keys remain pinned.
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::length_error("Immutable authority pool payload size overflows.");
        budget.add(count * sizeof(T));
        std::vector<T> builder;
        builder.reserve(count);
        if (builder.capacity() != count)
            throw std::length_error("Immutable authority pool capacity differs from admission.");
        for (auto& [values, offset] : values_)
        {
            offset = static_cast<std::uint32_t>(builder.size());
            builder.insert(builder.end(), values.begin(), values.end());
        }
        backing_ = ImmutableSequence<T>::freeze_builder(std::move(builder), budget);
        finalized_ = true;
    }
    void rebind(ImmutableSequence<T>& values) const
    {
        if (!finalized_)
            throw std::logic_error("Immutable authority pool has not been finalized.");
        if (!values.storage_identity())
            return;
        const auto found = values_.find(values);
        if (found == values_.end())
            throw std::invalid_argument("Immutable authority value is absent from its pool.");
        values = backing_.slice(found->second, values.size());
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
