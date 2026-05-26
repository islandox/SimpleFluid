/**
 * @file View.hh
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

namespace SimpleFluid
{

/**
 * @brief A random accessible view from a pointer and size, with random access iterators.
 * 
 * @tparam T data type
 */
template <typename T>
class RandomAccessView
{
public:
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    class iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator& operator++() { ++d_ptr; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++d_ptr; return tmp; }
        iterator& operator--() { --d_ptr; return *this; }
        iterator operator--(int) { iterator tmp = *this; --d_ptr; return tmp; }
        
        reference operator*() { return *d_ptr; }
        pointer operator->() { return d_ptr; }
        reference operator[](difference_type n) { return d_ptr[n]; }
        
        iterator& operator+=(difference_type n) { d_ptr += n; return *this; }
        iterator& operator-=(difference_type n) { d_ptr -= n; return *this; }
        iterator operator+(difference_type n) const { return iterator(d_ptr + n); }
        iterator operator-(difference_type n) const { return iterator(d_ptr - n); }
        
        difference_type operator-(const iterator& other) const { return d_ptr - other.d_ptr; }
        
        bool operator==(const iterator& other) const { return d_ptr == other.d_ptr; }
        bool operator!=(const iterator& other) const { return d_ptr != other.d_ptr; }
        bool operator<(const iterator& other) const { return d_ptr < other.d_ptr; }
        bool operator>(const iterator& other) const { return d_ptr > other.d_ptr; }
        bool operator<=(const iterator& other) const { return d_ptr <= other.d_ptr; }
        bool operator>=(const iterator& other) const { return d_ptr >= other.d_ptr; }

    private:
        friend class RandomAccessView;

        iterator(T* ptr) : d_ptr(ptr) {}
        T* d_ptr;
    };

    class const_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator& operator++() { ++d_ptr; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++d_ptr; return tmp; }
        const_iterator& operator--() { --d_ptr; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --d_ptr; return tmp; }
        
        reference operator*() const { return *d_ptr; }
        pointer operator->() const { return d_ptr; }
        reference operator[](difference_type n) const { return d_ptr[n]; }
        
        const_iterator& operator+=(difference_type n) { d_ptr += n; return *this; }
        const_iterator& operator-=(difference_type n) { d_ptr -= n; return *this; }
        const_iterator operator+(difference_type n) const { return const_iterator(d_ptr + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(d_ptr - n); }
        
        difference_type operator-(const const_iterator& other) const { return d_ptr - other.d_ptr; }
        
        bool operator==(const const_iterator& other) const { return d_ptr == other.d_ptr; }
        bool operator!=(const const_iterator& other) const { return d_ptr != other.d_ptr; }
        bool operator<(const const_iterator& other) const { return d_ptr < other.d_ptr; }
        bool operator>(const const_iterator& other) const { return d_ptr > other.d_ptr; }
        bool operator<=(const const_iterator& other) const { return d_ptr <= other.d_ptr; }
        bool operator>=(const const_iterator& other) const { return d_ptr >= other.d_ptr; }

    private:
        friend class RandomAccessView;

        const_iterator(const T* ptr) : d_ptr(ptr) {}
        const T* d_ptr;
    };


    RandomAccessView(const std::vector<T>& data)
        : d_data(data.data()), d_size(data.size())
    {
    }

    iterator begin() { return iterator(d_data); }
    iterator end() { return iterator(d_data + d_size); }
    
    const_iterator begin() const { return const_iterator(d_data); }
    const_iterator end() const { return const_iterator(d_data + d_size); }
    
    reference operator[](std::size_t index) { return d_data[index]; }
    const_reference operator[](std::size_t index) const { return d_data[index]; }
    size_type size() const { return d_size; }
    bool empty() const { return d_size == 0; }

private:
    T* const d_data;
    const std::size_t d_size;
};

}