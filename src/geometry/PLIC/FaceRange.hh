/** @file FaceRange.hh
 * @brief Allocation-free borrowed immutable face connectivity.
 */
#pragma once
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
namespace SimpleFluid::PLIC
{
namespace details
{
template<class Range> class ValueRangeIterator
{
    Range range_{};
    std::ptrdiff_t index_ = 0;

public:
    using value_type = typename Range::value_type;
    using reference = decltype(std::declval<Range>()[0]);
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;
    ValueRangeIterator() = default;
    ValueRangeIterator(Range range, difference_type index) : range_(range), index_(index) {}
    reference operator*() const { return range_[index_]; }
    reference operator[](difference_type n) const { return range_[index_ + n]; }
    ValueRangeIterator& operator++()
    {
        ++index_;
        return *this;
    }
    ValueRangeIterator operator++(int)
    {
        auto old = *this;
        ++*this;
        return old;
    }
    ValueRangeIterator& operator--()
    {
        --index_;
        return *this;
    }
    ValueRangeIterator operator--(int)
    {
        auto old = *this;
        --*this;
        return old;
    }
    ValueRangeIterator& operator+=(difference_type n)
    {
        index_ += n;
        return *this;
    }
    ValueRangeIterator& operator-=(difference_type n)
    {
        index_ -= n;
        return *this;
    }
    friend ValueRangeIterator operator+(ValueRangeIterator it, difference_type n) { return it += n; }
    friend ValueRangeIterator operator+(difference_type n, ValueRangeIterator it) { return it += n; }
    friend ValueRangeIterator operator-(ValueRangeIterator it, difference_type n) { return it -= n; }
    friend difference_type operator-(const ValueRangeIterator& a, const ValueRangeIterator& b)
    {
        return a.index_ - b.index_;
    }
    friend bool operator==(const ValueRangeIterator& a, const ValueRangeIterator& b) { return a.index_ == b.index_; }
    friend auto operator<=>(const ValueRangeIterator& a, const ValueRangeIterator& b) { return a.index_ <=> b.index_; }
};
} // namespace details
class FaceVertexRange
{
    const void* values_ = nullptr;
    std::size_t first_ = 0, size_ = 0;
    unsigned width_ = sizeof(std::size_t);

public:
    using value_type = std::size_t;
    using iterator = details::ValueRangeIterator<FaceVertexRange>;
    FaceVertexRange() = default;
    FaceVertexRange(const void* values, std::size_t first, std::size_t size, unsigned width)
        : values_(values), first_(first), size_(size), width_(width)
    {
    }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t operator[](std::size_t index) const noexcept
    {
        return width_ == 4 ? static_cast<const std::uint32_t*>(values_)[first_ + index]
                           : static_cast<const std::uint64_t*>(values_)[first_ + index];
    }
    std::size_t at(std::size_t index) const
    {
        if (index >= size_)
            throw std::out_of_range("face vertex index");
        return (*this)[index];
    }
    std::size_t front() const { return (*this)[0]; }
    std::size_t back() const { return (*this)[size_ - 1]; }
    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, static_cast<std::ptrdiff_t>(size_)}; }
    friend bool operator==(FaceVertexRange a, const std::vector<std::size_t>& b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
    friend bool operator==(FaceVertexRange a, FaceVertexRange b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
};
class FaceRange
{
    const void* offsets_ = nullptr;
    const void* values_ = nullptr;
    std::size_t size_ = 0;
    unsigned width_ = sizeof(std::size_t);

public:
    using value_type = FaceVertexRange;
    using iterator = details::ValueRangeIterator<FaceRange>;
    FaceRange() = default;
    FaceRange(const void* offsets, const void* values, std::size_t size, unsigned width)
        : offsets_(offsets), values_(values), size_(size), width_(width)
    {
    }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    FaceVertexRange operator[](std::size_t index) const noexcept
    {
        const FaceVertexRange offsets(offsets_, 0, size_ + 1, width_);
        return {values_, offsets[index], offsets[index + 1] - offsets[index], width_};
    }
    FaceVertexRange at(std::size_t index) const
    {
        if (index >= size_)
            throw std::out_of_range("face index");
        return (*this)[index];
    }
    FaceVertexRange front() const { return (*this)[0]; }
    FaceVertexRange back() const { return (*this)[size_ - 1]; }
    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, static_cast<std::ptrdiff_t>(size_)}; }
    friend bool operator==(FaceRange a, const std::vector<std::vector<std::size_t>>& b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
    friend bool operator==(FaceRange a, FaceRange b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
};
} // namespace SimpleFluid::PLIC
