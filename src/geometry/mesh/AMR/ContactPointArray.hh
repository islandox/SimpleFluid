#pragma once
#include "ContactDomainStorage.hh"
#include "ContactPayloadCensus.hh"
#include "ContactPointPageView.hh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
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
/** Immutable readers return owned normalized/physical identity values. Stored
 * pages and certified source components preserve the exact original identities;
 * reads neither normalize nor keep references to assembled temporaries. Equal
 * identities share the physical array; a second array is allocated only when
 * publication normalization actually changes a point. Mutators are construction-only. */
class ContactPointArray
{
    struct Owning
    {
        ContactFixedVector<ContactPointIdentity> physical, normalized;
    };
    template<class UInt> struct Pooled
    {
        using Reference = CapturedPointReference;
        ContactFixedVector<UInt> physical, normalized;
        std::shared_ptr<const ContactPointPageView> view;
    };
    template<class UInt> struct Sourced
    {
        using Reference = CapturedSourcePointReference;
        ContactFixedVector<UInt> physical, normalized;
        std::shared_ptr<const ContactSourcePointView> view;
    };

public:
    struct Locator
    {
        const ContactPointPageView* view;
        CapturedPointReference reference;
        const ContactSourcePointView* source_view = nullptr;
        CapturedSourcePointReference source_reference{};
    };
    struct Reference
    {
        ContactPointIdentity normalized;
        ContactPointIdentity physical;
        operator ContactPointIncidence() const { return {normalized, physical}; }
    };

private:
    struct Generated
    {
        std::shared_ptr<const void> snapshot;
        std::size_t row = 0, count = 0;
        Reference (*read)(const void*, std::size_t, std::size_t) = nullptr;
    };

public:
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
    /** Allocation-free immutable reader. The pin must retain every object read
     * by the callback, and those objects must remain immutable for its lifetime.
     * The callback supplies normalized and physical identities independently in
     * their existing row order; this adapter does not normalize or reorder. */
    [[nodiscard]] static ContactPointArray generated(std::shared_ptr<const void> snapshot, std::size_t row,
        std::size_t count, Reference (*read)(const void*, std::size_t, std::size_t))
    {
        if (!snapshot || !read)
            throw std::invalid_argument("Generated contact points require an immutable snapshot and reader.");
        ContactPointArray result;
        result.storage_.emplace<Generated>(std::move(snapshot), row, count, read);
        return result;
    }
    /** Canonical rows retain references into their batch directory. Buffer
     * admissions transfer to the row owner; directory/page credits own theirs. */
    static ContactPointArray pooled(ContactPointBatch& batch, std::size_t capacity, Budget& budget)
    {
        ContactPointArray result;
        if (batch.source_backed())
        {
            auto& source = result.storage_.emplace<Sourced<std::uint32_t>>();
            source.view = batch.source_view();
            reserve_references(source.physical, capacity, budget);
            return result;
        }
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
                if constexpr (std::is_same_v<Values, Owning> || std::is_same_v<Values, Generated> ||
                              std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
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
    void push_reference(CapturedSourcePointReference physical, CapturedSourcePointReference normalized, Budget& budget)
    {
        const auto view = source_page_view();
        if (!view || view->frozen())
            throw std::invalid_argument("Source point reference requires an active source batch.");
        if ((reference_uses_wide(physical.value) || reference_uses_wide(normalized.value)) &&
            std::holds_alternative<Sourced<std::uint32_t>>(storage_))
            widen_source(budget);
        std::visit(
            [&](auto& values)
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
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
                else
                    throw std::invalid_argument("Source reference used for a physical point array.");
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
            ContactFixedVector<ContactPointIdentity> next;
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
        return std::visit(
            [](const auto& values) -> std::size_t
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Generated>)
                    return values.count;
                else
                    return values.physical.size();
            },
            storage_);
    }
    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return std::visit(
            [](const auto& values) -> std::size_t
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Generated>)
                    return 0;
                else
                    return values.physical.capacity();
            },
            storage_);
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool shares_normalized_storage() const noexcept
    {
        return std::visit(
            [](const auto& values)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Generated>)
                    return false;
                else
                    return values.normalized.empty();
            },
            storage_);
    }
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return std::visit(
            [](const auto& values)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Generated>)
                    return std::size_t{0};
                else
                {
                    const auto physical = buffer_bytes(values.physical), normalized = buffer_bytes(values.normalized);
                    if (normalized > std::numeric_limits<std::size_t>::max() - physical)
                        throw std::length_error("Contact point storage size overflows.");
                    return physical + normalized;
                }
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
                else if constexpr (std::is_same_v<Values, Generated>)
                    return values.read(values.snapshot.get(), values.row, i);
                else
                {
                    const auto physical = typename Values::Reference{values.physical[i]};
                    const auto normalized =
                        values.normalized.empty() ? physical : typename Values::Reference{values.normalized[i]};
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
    [[nodiscard]] bool pooled() const noexcept
    {
        return !std::holds_alternative<Owning>(storage_) && !std::holds_alternative<Generated>(storage_);
    }
    [[nodiscard]] bool generated() const noexcept { return std::holds_alternative<Generated>(storage_); }
    /** Construction directories are borrowed only during this call. A consumer
     * that escapes must selectively import the returned physical page. */
    [[nodiscard]] std::optional<Locator> normalized_locator(std::size_t i) const { return locator(i, true); }
    [[nodiscard]] std::optional<Locator> physical_locator(std::size_t i) const { return locator(i, false); }
    [[nodiscard]] const ContactPointPageView* borrowed_page_view() const noexcept
    {
        return std::visit(
            [](const auto& values) -> const ContactPointPageView*
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Owning> || std::is_same_v<Values, Generated> ||
                              std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
                    return nullptr;
                else
                    return values.view.get();
            },
            storage_);
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, BasicContactPayloadRole::PointArrayObject, sizeof(*this));
        std::visit(
            [&](const auto& values)
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Owning>)
                {
                    contact_payload_buffer(visitor, this, BasicContactPayloadRole::PointArrayPhysical, values.physical);
                    contact_payload_buffer(
                        visitor, this, BasicContactPayloadRole::PointArrayNormalized, values.normalized);
                }
                else if constexpr (!std::is_same_v<Values, Generated>)
                {
                    contact_payload_buffer(
                        visitor, this, BasicContactPayloadRole::PointArrayPhysicalRefs, values.physical);
                    contact_payload_buffer(
                        visitor, this, BasicContactPayloadRole::PointArrayNormalizedRefs, values.normalized);
                    if (values.view)
                        values.view->visit_borrowed_payload(visitor);
                }
            },
            storage_);
    }
    [[nodiscard]] std::shared_ptr<const ContactPointPageView> page_view() const noexcept
    {
        return std::visit(
            [](const auto& values) -> std::shared_ptr<const ContactPointPageView>
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Owning> || std::is_same_v<Values, Generated> ||
                              std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
                    return {};
                else
                    return values.view;
            },
            storage_);
    }
    [[nodiscard]] std::shared_ptr<const ContactSourcePointView> source_page_view() const noexcept
    {
        return std::visit(
            [](const auto& values) -> std::shared_ptr<const ContactSourcePointView>
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
                    return values.view;
                else
                    return {};
            },
            storage_);
    }

    /** Construction-only transfer for a new row that has NEVER escaped
     * its session. Buffers keep exact widths/capacities; published row/page
     * generations are never redirected. */
    bool bind_private_initial_source(
        IncidencePointDirectory<>& pool, const std::shared_ptr<ContactSourcePointView>& target)
    {
        if (generated())
            throw std::invalid_argument("Cannot rebind generated contact points.");
        if (!target || target->frozen() || target->source() != pool.source())
            throw std::invalid_argument("Invalid private source-row construction binding.");
        return std::visit(
            [&](auto& values) -> bool
            {
                using Values = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<Values, Pooled<std::uint32_t>> ||
                              std::is_same_v<Values, Pooled<std::uint64_t>> ||
                              std::is_same_v<Values, Sourced<std::uint32_t>> ||
                              std::is_same_v<Values, Sourced<std::uint64_t>>)
                {
                    using UInt = typename std::remove_reference_t<decltype(values.physical)>::value_type;
                    if (pool.source_size() && pool.source_size() - 1 > std::numeric_limits<UInt>::max())
                        return false;
                    const auto map = [&](UInt reference)
                    {
                        const auto point = [&]
                        {
                            if constexpr (std::is_same_v<Values, Pooled<UInt>>)
                                return values.view->get(CapturedPointReference{reference});
                            else
                                return values.view->get(CapturedSourcePointReference{reference});
                        }();
                        const auto slot = pool.find(point);
                        if (!slot || *slot >= pool.source_size())
                            throw std::invalid_argument("Private source row lacks its full original point identity.");
                        return static_cast<UInt>(target->capture(pool, *slot).value);
                    };
                    for (const auto reference : values.physical)
                        (void) map(reference);
                    for (const auto reference : values.normalized)
                        (void) map(reference);
                    for (auto& reference : values.physical)
                        reference = map(reference);
                    for (auto& reference : values.normalized)
                        reference = map(reference);
                    Sourced<UInt> source{std::move(values.physical), std::move(values.normalized), target};
                    storage_.template emplace<Sourced<UInt>>(std::move(source));
                    return true;
                }
                else
                    return false;
            },
            storage_);
    }

private:
    [[nodiscard]] std::optional<Locator> locator(std::size_t i, bool normalized) const
    {
        if (i >= size())
            throw std::out_of_range("Contact point locator index is out of range.");
        return std::visit(
            [&](const auto& values) -> std::optional<Locator>
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Owning> ||
                              std::is_same_v<std::decay_t<decltype(values)>, Generated>)
                    return {};
                else if constexpr (std::is_same_v<std::decay_t<decltype(values)>, Sourced<std::uint32_t>> ||
                                   std::is_same_v<std::decay_t<decltype(values)>, Sourced<std::uint64_t>>)
                {
                    const CapturedSourcePointReference reference{
                        normalized && !values.normalized.empty() ? values.normalized[i] : values.physical[i]};
                    if (const auto stored = values.view->stored_location(reference))
                        return Locator{stored->view, stored->reference};
                    return Locator{nullptr, {0}, values.view.get(), reference};
                }
                else
                    return Locator{values.view.get(),
                        CapturedPointReference{
                            normalized && !values.normalized.empty() ? values.normalized[i] : values.physical[i]}};
            },
            storage_);
    }
    template<class Value> static std::size_t buffer_bytes(const ContactFixedVector<Value>& array)
    {
        if (array.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Value))
            throw std::length_error("Contact point capacity overflows.");
        return array.capacity() * sizeof(Value);
    }
    template<class UInt>
    static void reserve_references(ContactFixedVector<UInt>& array, std::size_t capacity, Budget& budget)
    {
        if (capacity <= array.capacity())
            return;
        if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(UInt))
            throw std::length_error("Contact reference capacity overflows.");
        const auto previous = buffer_bytes(array), next_bytes = capacity * sizeof(UInt);
        budget.add(next_bytes);
        {
            ContactFixedVector<UInt> next;
            try
            {
                next.reserve(capacity);
                if (next.capacity() != capacity)
                    throw std::invalid_argument("Contact reference capacity differs from admission.");
                next.assign(array.begin(), array.end());
            }
            catch (...)
            {
                ContactFixedVector<UInt>().swap(next);
                budget.used -= next_bytes;
                throw;
            }
            array.swap(next);
        }
        budget.used -= previous;
    }
    template<class UInt>
    static void append_reference(ContactFixedVector<UInt>& array, std::uint64_t value, Budget& budget)
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
            ContactFixedVector<std::uint64_t>().swap(wide.physical);
            ContactFixedVector<std::uint64_t>().swap(wide.normalized);
            budget.used -= admitted;
            throw;
        }
        const auto previous = buffer_bytes(narrow.physical) + buffer_bytes(narrow.normalized);
        // Vector/shared_ptr moves are nonthrowing. Destroy both old buffers
        // before returning their admission; failure above leaves them intact.
        storage_.emplace<Pooled<std::uint64_t>>(std::move(wide));
        budget.used -= previous;
    }
    void widen_source(Budget& budget)
    {
        const auto& narrow = std::get<Sourced<std::uint32_t>>(storage_);
        Sourced<std::uint64_t> wide;
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
            ContactFixedVector<std::uint64_t>().swap(wide.physical);
            ContactFixedVector<std::uint64_t>().swap(wide.normalized);
            budget.used -= admitted;
            throw;
        }
        const auto previous = buffer_bytes(narrow.physical) + buffer_bytes(narrow.normalized);
        storage_.emplace<Sourced<std::uint64_t>>(std::move(wide));
        budget.used -= previous;
    }
    std::variant<Owning, Pooled<std::uint32_t>, Pooled<std::uint64_t>, Sourced<std::uint32_t>, Sourced<std::uint64_t>,
        Generated>
        storage_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
