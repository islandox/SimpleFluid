#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Stable mutable metadata slots. Geometry bodies keep their own immutable
 * ownership; this container owns only record objects and its page index. */
template<class T, std::size_t PageSize = 16> class PagedRecords
{
    static_assert(PageSize > 0);
    struct Page
    {
        std::array<T, PageSize> values{};
    };
    template<bool Const> class Iterator
    {
        using Owner = std::conditional_t<Const, const PagedRecords, PagedRecords>;

    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using reference = std::conditional_t<Const, const T&, T&>;
        using pointer = std::conditional_t<Const, const T*, T*>;
        using iterator_category = std::forward_iterator_tag;
        Iterator() = default;
        reference operator*() const { return (*owner_)[index_]; }
        pointer operator->() const { return &**this; }
        Iterator& operator++()
        {
            ++index_;
            return *this;
        }
        Iterator operator++(int)
        {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator&) const = default;

    private:
        friend class PagedRecords;
        Iterator(Owner* owner, std::size_t index) : owner_(owner), index_(index) {}
        Owner* owner_ = nullptr;
        std::size_t index_ = 0;
    };

public:
    explicit PagedRecords(Budget& budget) noexcept : budget_(&budget) {}
    ~PagedRecords()
    {
        decltype(pages_)().swap(pages_);
        budget_->used -= charge_;
    }
    PagedRecords(const PagedRecords&) = delete;
    PagedRecords& operator=(const PagedRecords&) = delete;
    void reserve(std::size_t count)
    {
        const auto needed = count / PageSize + (count % PageSize != 0);
        if (needed > budget_->limit / sizeof(Page))
            throw std::invalid_argument("Record page capacity exceeds budget.");
        if (needed > pages_.capacity())
        {
            const auto before = pages_.capacity() * sizeof(std::unique_ptr<Page>);
            const auto bytes = needed * sizeof(std::unique_ptr<Page>);
            admit(bytes);
            pages_.reserve(needed);
            if (pages_.capacity() != needed)
                throw std::invalid_argument("Record page index differs from admission.");
            budget_->used -= before;
            charge_ -= before;
        }
        while (pages_.size() < needed)
        {
            admit(sizeof(Page));
            pages_.push_back(std::make_unique<Page>());
        }
    }
    T& emplace_back()
    {
        if (size_ == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Record slot count overflows.");
        if (size_ == capacity())
            grow();
        auto& result = pages_[size_ / PageSize]->values[size_ % PageSize];
        ++size_;
        return result;
    }
    void push_back(const T& value)
    {
        if (size_ == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Record slot count overflows.");
        if (size_ == capacity())
            grow();
        pages_[size_ / PageSize]->values[size_ % PageSize] = value;
        ++size_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return pages_.size() * PageSize; }
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return charge_; }
    T& operator[](std::size_t i) noexcept { return pages_[i / PageSize]->values[i % PageSize]; }
    const T& operator[](std::size_t i) const noexcept { return pages_[i / PageSize]->values[i % PageSize]; }
    T& back() noexcept { return (*this)[size_ - 1]; }
    const T& back() const noexcept { return (*this)[size_ - 1]; }
    auto begin() noexcept { return Iterator<false>(this, 0); }
    auto end() noexcept { return Iterator<false>(this, size_); }
    auto begin() const noexcept { return Iterator<true>(this, 0); }
    auto end() const noexcept { return Iterator<true>(this, size_); }

private:
    void admit(std::size_t bytes)
    {
        budget_->add(bytes);
        charge_ += bytes;
    }
    void grow()
    {
        if (pages_.size() == pages_.capacity())
        {
            const auto old = pages_.capacity();
            if (old > std::numeric_limits<std::size_t>::max() / 2 / sizeof(std::unique_ptr<Page>))
                throw std::invalid_argument("Record page index overflows.");
            const auto next = old ? 2 * old : 1;
            admit(next * sizeof(std::unique_ptr<Page>));
            pages_.reserve(next);
            if (pages_.capacity() != next)
                throw std::invalid_argument("Record page index differs from admission.");
            budget_->used -= old * sizeof(std::unique_ptr<Page>);
            charge_ -= old * sizeof(std::unique_ptr<Page>);
        }
        if (pages_.size() >= budget_->limit / sizeof(Page))
            throw std::invalid_argument("Record pages exceed byte budget.");
        admit(sizeof(Page));
        pages_.push_back(std::make_unique<Page>());
    }
    Budget* budget_;
    std::vector<std::unique_ptr<Page>> pages_;
    std::size_t size_ = 0, charge_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
