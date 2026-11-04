#pragma once
#include "ContactPointPageView.hh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactPointIncidence
{
    ContactPointIdentity normalized, physical;
};
/** Immutable readers borrow the exact normalized and physical values. Ordinary
 * identities normalize signed zero at construction, so equal identities share
 * the physical array without changing their canonical representation. A second
 * array is allocated only when some normalization actually changes a point.
 * Mutators are construction-only; references survive until the next mutation. */
class ContactPointArray
{
    struct Owning
    {
        std::vector<ContactPointIdentity> physical, normalized;
    };
    template<class UInt> struct Pooled
    {
        std::vector<UInt> physical, normalized;
        std::shared_ptr<const ContactPointPageView> view;
    };

public:
    struct Reference
    {
        const ContactPointIdentity& normalized;
        const ContactPointIdentity& physical;
        operator ContactPointIncidence() const { return {normalized, physical}; }
    };
    class const_iterator
    {
    public:
        using value_type = ContactPointIncidence;
        using reference = Reference;
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
        friend class ContactPointArray;
        const_iterator(const ContactPointArray* owner, std::size_t index) : owner_(owner), index_(index) {}
        const ContactPointArray* owner_ = nullptr;
        std::size_t index_ = 0;
    };
    ContactPointArray() = default;
    ContactPointArray(std::initializer_list<ContactPointIncidence> points)
    {
        reserve(points.size());
        for (const auto& point : points)
            push_back(point);
    }
    /** Canonical rows retain references into their batch directory. Buffer
     * admissions transfer to the row owner; directory/page credits own theirs. */
    static ContactPointArray pooled(ContactPointBatch& batch, std::size_t capacity, Budget& budget)
    {
        ContactPointArray result;
        auto& pooled = result.storage_.emplace<Pooled<std::uint32_t>>();
        pooled.view = batch.view();
        reserve_references(pooled.physical, capacity, budget);
        return result;
    }
    [[nodiscard]] static constexpr bool reference_uses_wide(std::uint64_t value) noexcept
    {
        return value > std::numeric_limits<std::uint32_t>::max();
    }
    void push_reference(CapturedPointReference physical, CapturedPointReference normalized, Budget& budget)
    {
        const auto view = page_view();
        if (!view || view->frozen())
            throw std::invalid_argument("Contact point reference construction requires an active batch.");
        if ((reference_uses_wide(physical.value) || reference_uses_wide(normalized.value)) &&
            std::holds_alternative<Pooled<std::uint32_t>>(storage_))
            widen(budget);
        std::visit(
            [&](auto& values)
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Owning>)
                    throw std::invalid_argument("Contact point array does not use pooled references.");
                else
                {
                    if (values.normalized.empty() && normalized != physical)
                    {
                        reserve_references(values.normalized, values.physical.capacity(), budget);
                        values.normalized.assign(values.physical.begin(), values.physical.end());
                    }
                    if (!values.normalized.empty() || normalized != physical)
                        append_reference(values.normalized, normalized.value, budget);
                    append_reference(values.physical, physical.value, budget);
                }
            },
            storage_);
    }
    void reserve(std::size_t capacity)
    {
        auto* values = std::get_if<Owning>(&storage_);
        if (!values)
            throw std::invalid_argument("Cannot mutate a pooled contact point array.");
        values->physical.reserve(capacity);
        if (!values->normalized.empty())
            values->normalized.reserve(capacity);
    }
    void push_back(ContactPointIncidence point)
    {
        auto* values = std::get_if<Owning>(&storage_);
        if (!values)
            throw std::invalid_argument("Cannot append owning values to a pooled contact point array.");
        auto& physical = values->physical;
        auto& normalized = values->normalized;
        if (physical.size() == physical.capacity())
        {
            if (physical.capacity() > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("Contact point capacity overflows.");
            reserve(physical.empty() ? 1 : 2 * physical.capacity());
        }
        if (normalized.empty() && point.normalized != point.physical)
        {
            std::vector<ContactPointIdentity> next;
            next.reserve(physical.capacity());
            next.assign(physical.begin(), physical.end());
            next.push_back(std::move(point.normalized));
            physical.push_back(std::move(point.physical));
            normalized.swap(next);
        }
        else
        {
            if (!normalized.empty())
                normalized.push_back(std::move(point.normalized));
            physical.push_back(std::move(point.physical));
        }
    }
    [[nodiscard]] std::size_t size() const noexcept
    {
        return std::visit([](const auto& values) { return values.physical.size(); }, storage_);
    }
    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return std::visit([](const auto& values) { return values.physical.capacity(); }, storage_);
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool shares_normalized_storage() const noexcept
    {
        return std::visit([](const auto& values) { return values.normalized.empty(); }, storage_);
    }
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return std::visit(
            [](const auto& values)
            {
                const auto physical = buffer_bytes(values.physical), normalized = buffer_bytes(values.normalized);
                if (normalized > std::numeric_limits<std::size_t>::max() - physical)
                    throw std::length_error("Contact point storage size overflows.");
                return physical + normalized;
            },
            storage_);
    }
    [[nodiscard]] Reference operator[](std::size_t i) const
    {
        return std::visit(
            [i](const auto& values) -> Reference
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Owning>)
                    return {values.normalized.empty() ? values.physical[i] : values.normalized[i], values.physical[i]};
                else
                {
                    const auto physical = CapturedPointReference{values.physical[i]};
                    const auto normalized =
                        values.normalized.empty() ? physical : CapturedPointReference{values.normalized[i]};
                    return {values.view->get(normalized), values.view->get(physical)};
                }
            },
            storage_);
    }
    [[nodiscard]] Reference at(std::size_t i) const
    {
        if (i >= size())
            throw std::out_of_range("Contact point index is out of range.");
        return (*this)[i];
    }
    [[nodiscard]] const_iterator begin() const noexcept { return {this, 0}; }
    [[nodiscard]] const_iterator end() const noexcept { return {this, size()}; }
    [[nodiscard]] bool pooled() const noexcept { return !std::holds_alternative<Owning>(storage_); }
    [[nodiscard]] std::shared_ptr<const ContactPointPageView> page_view() const noexcept
    {
        return std::visit(
            [](const auto& values) -> std::shared_ptr<const ContactPointPageView>
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Owning>)
                    return {};
                else
                    return values.view;
            },
            storage_);
    }

private:
    template<class Value> static std::size_t buffer_bytes(const std::vector<Value>& array)
    {
        if (array.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Value))
            throw std::length_error("Contact point capacity overflows.");
        return array.capacity() * sizeof(Value);
    }
    template<class UInt> static void reserve_references(std::vector<UInt>& array, std::size_t capacity, Budget& budget)
    {
        if (capacity <= array.capacity())
            return;
        if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(UInt))
            throw std::length_error("Contact reference capacity overflows.");
        const auto previous = buffer_bytes(array), next_bytes = capacity * sizeof(UInt);
        budget.add(next_bytes);
        {
            std::vector<UInt> next;
            try
            {
                next.reserve(capacity);
                if (next.capacity() != capacity)
                    throw std::invalid_argument("Contact reference capacity differs from admission.");
                next.assign(array.begin(), array.end());
            }
            catch (...)
            {
                std::vector<UInt>().swap(next);
                budget.used -= next_bytes;
                throw;
            }
            array.swap(next);
        }
        budget.used -= previous;
    }
    template<class UInt> static void append_reference(std::vector<UInt>& array, std::uint64_t value, Budget& budget)
    {
        if (value > std::numeric_limits<UInt>::max())
            throw std::invalid_argument("Contact reference narrowing would lose identity.");
        if (array.size() == array.capacity())
        {
            const auto old = array.capacity();
            if (old > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("Contact reference capacity overflows.");
            reserve_references(array, old ? 2 * old : 1, budget);
        }
        array.push_back(static_cast<UInt>(value));
    }
    void widen(Budget& budget)
    {
        const auto& narrow = std::get<Pooled<std::uint32_t>>(storage_);
        Pooled<std::uint64_t> wide;
        std::size_t admitted = 0;
        try
        {
            reserve_references(wide.physical, narrow.physical.capacity(), budget);
            admitted += buffer_bytes(wide.physical);
            reserve_references(wide.normalized, narrow.normalized.capacity(), budget);
            admitted += buffer_bytes(wide.normalized);
            wide.physical.assign(narrow.physical.begin(), narrow.physical.end());
            wide.normalized.assign(narrow.normalized.begin(), narrow.normalized.end());
            wide.view = narrow.view;
        }
        catch (...)
        {
            std::vector<std::uint64_t>().swap(wide.physical);
            std::vector<std::uint64_t>().swap(wide.normalized);
            budget.used -= admitted;
            throw;
        }
        const auto previous = buffer_bytes(narrow.physical) + buffer_bytes(narrow.normalized);
        // Vector/shared_ptr moves are nonthrowing. Destroy both old buffers
        // before returning their admission; failure above leaves them intact.
        storage_.emplace<Pooled<std::uint64_t>>(std::move(wide));
        budget.used -= previous;
    }
    std::variant<Owning, Pooled<std::uint32_t>, Pooled<std::uint64_t>> storage_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
