/**
 * @file RandomAccessView.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief random accessible view of a contiguous data array, with random access iterators
 * @version 0.1
 * @date 2026-05-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "typedefs.hh"

#include <compare>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace SimpleFluid
{

/**
 * @brief A random accessible view from a pointer and size, with random access iterators.
 *
 * The view is non-owning. The caller must keep the viewed storage alive and
 * must not invalidate its address while this object or any iterator obtained
 * from it is in use.
 * Element access is unchecked; indices must be less than size().
 * 
 * @tparam T data type
 */
template <typename T>
class RandomAccessView
{
public:
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

    /**
     * @brief Random-access iterator over the viewed contiguous storage.
     *
     * @tparam IsConst Whether dereferencing yields a const reference.
     */
    template<bool IsConst>
    class IteratorImpl
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = std::conditional_t<IsConst, const T*, T*>;
        using reference = std::conditional_t<IsConst, const T&, T&>;

        IteratorImpl() = default;

        IteratorImpl& operator++() { ++d_ptr; return *this; }
        IteratorImpl operator++(int)
        {
            IteratorImpl tmp = *this;
            ++d_ptr;
            return tmp;
        }
        IteratorImpl& operator--() { --d_ptr; return *this; }
        IteratorImpl operator--(int)
        {
            IteratorImpl tmp = *this;
            --d_ptr;
            return tmp;
        }
        
        reference operator*() const { return *d_ptr; }
        pointer operator->() const { return d_ptr; }
        reference operator[](difference_type n) const { return d_ptr[n]; }
        
        IteratorImpl& operator+=(difference_type n)
        {
            d_ptr += n;
            return *this;
        }
        IteratorImpl& operator-=(difference_type n)
        {
            d_ptr -= n;
            return *this;
        }
        IteratorImpl operator+(difference_type n) const
        {
            return IteratorImpl(d_ptr + n);
        }
        IteratorImpl operator-(difference_type n) const
        {
            return IteratorImpl(d_ptr - n);
        }
        friend IteratorImpl operator+(
            difference_type n, IteratorImpl iter)
        {
            return iter + n;
        }
        
        difference_type operator-(const IteratorImpl& other) const
        {
            return d_ptr - other.d_ptr;
        }
        
        auto operator<=>(const IteratorImpl&) const = default;

    private:
        friend class RandomAccessView;

        explicit IteratorImpl(pointer ptr) : d_ptr(ptr) {}
        pointer d_ptr = nullptr;
    };

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;


    RandomAccessView()
        : d_data(nullptr), d_size(0)
    {
    }

    /**
     * @pre @p data addresses at least @p size live elements (or is null when
     *      @p size is zero).
     */
    RandomAccessView(T* data, size_type size)
        : d_data(data), d_size(size)
    {
    }

    /**
     * @warning Destruction or any operation that reallocates @p data
     *          invalidates the view and its iterators.
     */
    RandomAccessView(std::vector<T>& data)
        : d_data(data.data()), d_size(data.size())
    {
    }

    iterator begin() { return iterator(d_data); }
    iterator end() { return iterator(d_data + d_size); }
    
    const_iterator begin() const { return const_iterator(d_data); }
    const_iterator end() const { return const_iterator(d_data + d_size); }
    
    reference operator[](size_t index) { return d_data[index]; }
    const_reference operator[](size_t index) const { return d_data[index]; }
    size_type size() const { return d_size; }
    bool empty() const { return d_size == 0; }

private:
    T* d_data;
    size_t d_size;
};

}
