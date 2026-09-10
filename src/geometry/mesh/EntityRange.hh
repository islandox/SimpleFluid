/**
 * @file EntityRange.hh
 * @author islandox
 * @brief Allocation-free indexed entity observation.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include <cstddef>
#include <compare>
#include <iterator>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
/**
 * @brief Sized, random-access range of computed entity IDs returned by value.
 *
 * The source must outlive the range. Each range carries its own immutable
 * query state; nested queries need neither allocation nor shared scratch.
 * Iterators own the query state and may outlive the range object itself.
 * @tparam ID Entity identifier type.
 */
template<class ID>
class EntityRange
{
public:
    using Getter = ID (*)(const void*, size_t, size_t);

    EntityRange() = default;
    EntityRange(const void* source, size_t key, size_t count, Getter getter)
        : d_source(source), d_key(key), d_count(count), d_getter(getter) {}

    size_t size() const noexcept { return d_count; }
    bool empty() const noexcept { return d_count == 0; }
    ID operator[](size_t i) const { return d_getter(d_source, d_key, i); }
    ID at(size_t i) const
    {
        if (i >= size()) throw std::out_of_range("Entity range index out of bounds.");
        return (*this)[i];
    }
    ID front() const { return at(0); }
    ID back() const { return at(size() - 1); }
    /** @brief Iterator carrying the source and query without a view backpointer. */
    struct Iterator
    {
        using value_type = ID;
        using reference = ID;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
        using iterator_concept = std::random_access_iterator_tag;
        const void* source = nullptr;
        size_t key = 0;
        Getter getter = nullptr;
        difference_type position = 0;
        ID operator*() const { return getter(source, key, static_cast<size_t>(position)); }
        ID operator[](difference_type n) const { return *(*this + n); }
        Iterator& operator++() { ++position; return *this; }
        Iterator operator++(int) { auto old = *this; ++*this; return old; }
        Iterator& operator--() { --position; return *this; }
        Iterator operator--(int) { auto old = *this; --*this; return old; }
        Iterator& operator+=(difference_type n) { position += n; return *this; }
        Iterator& operator-=(difference_type n) { position -= n; return *this; }
        friend Iterator operator+(Iterator i, difference_type n) { return i += n; }
        friend Iterator operator+(difference_type n, Iterator i) { return i += n; }
        friend Iterator operator-(Iterator i, difference_type n) { return i -= n; }
        friend difference_type operator-(Iterator a, Iterator b) { return a.position - b.position; }
        bool operator==(const Iterator&) const = default;
        auto operator<=>(const Iterator& other) const { return position <=> other.position; }
    };
    Iterator begin() const { return {d_source, d_key, d_getter, 0}; }
    Iterator end() const { return {d_source, d_key, d_getter, static_cast<std::ptrdiff_t>(d_count)}; }

private:
    const void* d_source = nullptr;
    size_t d_key = 0;
    size_t d_count = 0;
    Getter d_getter = nullptr;
};
} // namespace SimpleFluid::Meshes
