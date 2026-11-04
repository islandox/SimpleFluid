#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using ContactPointIdentity = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
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
    void reserve(std::size_t capacity)
    {
        physical_.reserve(capacity);
        if (!normalized_.empty())
            normalized_.reserve(capacity);
    }
    void push_back(ContactPointIncidence point)
    {
        if (physical_.size() == physical_.capacity())
        {
            if (physical_.capacity() > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("Contact point capacity overflows.");
            reserve(physical_.empty() ? 1 : 2 * physical_.capacity());
        }
        if (normalized_.empty() && point.normalized != point.physical)
        {
            std::vector<ContactPointIdentity> normalized;
            normalized.reserve(physical_.capacity());
            normalized.assign(physical_.begin(), physical_.end());
            normalized.push_back(std::move(point.normalized));
            physical_.push_back(std::move(point.physical));
            normalized_.swap(normalized);
        }
        else
        {
            if (!normalized_.empty())
                normalized_.push_back(std::move(point.normalized));
            physical_.push_back(std::move(point.physical));
        }
    }
    [[nodiscard]] std::size_t size() const noexcept { return physical_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return physical_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return physical_.empty(); }
    [[nodiscard]] bool shares_normalized_storage() const noexcept { return normalized_.empty(); }
    [[nodiscard]] std::size_t storage_bytes() const
    {
        constexpr auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(ContactPointIdentity);
        if (physical_.capacity() > maximum || normalized_.capacity() > maximum - physical_.capacity())
            throw std::length_error("Contact point storage size overflows.");
        return (physical_.capacity() + normalized_.capacity()) * sizeof(ContactPointIdentity);
    }
    [[nodiscard]] Reference operator[](std::size_t i) const
    {
        return {normalized_.empty() ? physical_[i] : normalized_[i], physical_[i]};
    }
    [[nodiscard]] Reference at(std::size_t i) const
    {
        if (i >= size())
            throw std::out_of_range("Contact point index is out of range.");
        return (*this)[i];
    }
    [[nodiscard]] const_iterator begin() const noexcept { return {this, 0}; }
    [[nodiscard]] const_iterator end() const noexcept { return {this, size()}; }

private:
    std::vector<ContactPointIdentity> physical_, normalized_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
