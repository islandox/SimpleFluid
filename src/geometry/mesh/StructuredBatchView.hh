/**
 * @file StructuredBatchView.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cartesian-product views for structured mesh batches.
 * @version 0.1
 * @date 2026-07-21
 *
 * Delegates to `std::views::cartesian_product` when the standard library
 * provides it; falls back to a minimal forward-range implementation on
 * toolchains that lack the C++23 feature (e.g. Apple Clang / LLVM libc++).
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <ranges>
#include <tuple>

namespace SimpleFluid::Meshes
{

// ---------------------------------------------------------------------------
// Standard-library path
// ---------------------------------------------------------------------------
#if __cpp_lib_ranges_cartesian_product >= 202207L

/**
 * @brief Create a cartesian-product view over two iota ranges.
 * @tparam T Coordinate value type.
 * @param a_beg Inclusive beginning of the first range.
 * @param a_end Exclusive end of the first range.
 * @param b_beg Inclusive beginning of the second range.
 * @param b_end Exclusive end of the second range.
 * @return Lazy view of `(a, b)` tuples in lexicographic order.
 */
template <class T>
auto cartesian_product_2d(T a_beg, T a_end, T b_beg, T b_end)
{
    return std::views::cartesian_product(
        std::views::iota(a_beg, a_end),
        std::views::iota(b_beg, b_end));
}

/**
 * @brief Create a cartesian-product view over three iota ranges.
 * @tparam T Coordinate value type.
 * @param a_beg Inclusive beginning of the first range.
 * @param a_end Exclusive end of the first range.
 * @param b_beg Inclusive beginning of the second range.
 * @param b_end Exclusive end of the second range.
 * @param c_beg Inclusive beginning of the third range.
 * @param c_end Exclusive end of the third range.
 * @return Lazy view of `(a, b, c)` tuples in lexicographic order.
 */
template <class T>
auto cartesian_product_3d(T a_beg, T a_end,
                          T b_beg, T b_end,
                          T c_beg, T c_end)
{
    return std::views::cartesian_product(
        std::views::iota(a_beg, a_end),
        std::views::iota(b_beg, b_end),
        std::views::iota(c_beg, c_end));
}

// ---------------------------------------------------------------------------
// Workaround for toolchains without std::views::cartesian_product
// ---------------------------------------------------------------------------
#else

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace detail
{

/**
 * @brief Iterator over the cartesian product of 2 iota ranges.
 * @tparam T Coordinate value type.
 */
template <class T>
class CartesianProduct2DIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<T, T>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    CartesianProduct2DIterator() = default;

    CartesianProduct2DIterator(T a_beg, T a_end, T b_beg, T b_end,
                               bool at_end = false)
        : d_a(a_beg), d_b(b_beg),
          d_a_beg(a_beg), d_a_end(a_end),
          d_b_beg(b_beg), d_b_end(b_end),
          d_val{d_a, d_b}
    {
        if (at_end)
        {
            d_a = d_a_end;
            d_b = d_b_end;
        }
        else if (d_a == d_a_end || d_b == d_b_end)
        {
            d_a = d_a_end;
            d_b = d_b_end;
        }
    }

    reference operator*() const noexcept { return d_val; }
    pointer operator->() const noexcept { return &d_val; }

    CartesianProduct2DIterator& operator++() noexcept
    {
        ++d_b;
        if (d_b == d_b_end)
        {
            d_b = d_b_beg;
            ++d_a;
            if (d_a == d_a_end)
            {
                d_b = d_b_end;
            }
        }
        d_val = {d_a, d_b};
        return *this;
    }

    CartesianProduct2DIterator operator++(int) noexcept
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const CartesianProduct2DIterator& other) const noexcept
    {
        return d_a == other.d_a && d_b == other.d_b;
    }

    bool operator!=(const CartesianProduct2DIterator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    T d_a{};
    T d_b{};
    T d_a_beg{};
    T d_a_end{};
    T d_b_beg{};
    T d_b_end{};
    value_type d_val{};
};

/**
 * @brief Iterator over the cartesian product of 3 iota ranges.
 * @tparam T Coordinate value type.
 */
template <class T>
class CartesianProduct3DIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<T, T, T>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    CartesianProduct3DIterator() = default;

    CartesianProduct3DIterator(T a_beg, T a_end,
                               T b_beg, T b_end,
                               T c_beg, T c_end,
                               bool at_end = false)
        : d_a(a_beg), d_b(b_beg), d_c(c_beg),
          d_a_beg(a_beg), d_a_end(a_end),
          d_b_beg(b_beg), d_b_end(b_end),
          d_c_beg(c_beg), d_c_end(c_end),
          d_val{d_a, d_b, d_c}
    {
        if (at_end)
        {
            d_a = d_a_end;
            d_b = d_b_end;
            d_c = d_c_end;
        }
        else if (d_a == d_a_end || d_b == d_b_end || d_c == d_c_end)
        {
            d_a = d_a_end;
            d_b = d_b_end;
            d_c = d_c_end;
        }
    }

    reference operator*() const noexcept { return d_val; }
    pointer operator->() const noexcept { return &d_val; }

    CartesianProduct3DIterator& operator++() noexcept
    {
        ++d_c;
        if (d_c == d_c_end)
        {
            d_c = d_c_beg;
            ++d_b;
            if (d_b == d_b_end)
            {
                d_b = d_b_beg;
                ++d_a;
                if (d_a == d_a_end)
                {
                    d_b = d_b_end;
                    d_c = d_c_end;
                }
            }
        }
        d_val = {d_a, d_b, d_c};
        return *this;
    }

    CartesianProduct3DIterator operator++(int) noexcept
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const CartesianProduct3DIterator& other) const noexcept
    {
        return d_a == other.d_a
            && d_b == other.d_b
            && d_c == other.d_c;
    }

    bool operator!=(const CartesianProduct3DIterator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    T d_a{}, d_b{}, d_c{};
    T d_a_beg{}, d_a_end{};
    T d_b_beg{}, d_b_end{};
    T d_c_beg{}, d_c_end{};
    value_type d_val{};
};

/**
 * @brief Sized view over the cartesian product of 2 iota ranges.
 * @tparam T Coordinate value type.
 */
template <class T>
class CartesianProductView2D : public std::ranges::view_interface<CartesianProductView2D<T>>
{
public:
    using iterator = CartesianProduct2DIterator<T>;
    using value_type = typename iterator::value_type;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    CartesianProductView2D() = default;
    CartesianProductView2D(T a_beg, T a_end, T b_beg, T b_end)
        : d_a_beg(a_beg), d_a_end(a_end),
          d_b_beg(b_beg), d_b_end(b_end) {}

    iterator begin() const noexcept
    {
        return iterator(d_a_beg, d_a_end, d_b_beg, d_b_end);
    }

    iterator end() const noexcept
    {
        return iterator(d_a_beg, d_a_end, d_b_beg, d_b_end, true);
    }

    size_type size() const noexcept
    {
        if (d_a_end <= d_a_beg || d_b_end <= d_b_beg) return 0;
        const auto na = static_cast<size_type>(d_a_end - d_a_beg);
        const auto nb = static_cast<size_type>(d_b_end - d_b_beg);
        return na * nb;
    }

private:
    T d_a_beg{}, d_a_end{};
    T d_b_beg{}, d_b_end{};
};

/**
 * @brief Sized view over the cartesian product of 3 iota ranges.
 * @tparam T Coordinate value type.
 */
template <class T>
class CartesianProductView3D : public std::ranges::view_interface<CartesianProductView3D<T>>
{
public:
    using iterator = CartesianProduct3DIterator<T>;
    using value_type = typename iterator::value_type;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    CartesianProductView3D() = default;
    CartesianProductView3D(T a_beg, T a_end,
                           T b_beg, T b_end,
                           T c_beg, T c_end)
        : d_a_beg(a_beg), d_a_end(a_end),
          d_b_beg(b_beg), d_b_end(b_end),
          d_c_beg(c_beg), d_c_end(c_end) {}

    iterator begin() const noexcept
    {
        return iterator(d_a_beg, d_a_end,
                        d_b_beg, d_b_end,
                        d_c_beg, d_c_end);
    }

    iterator end() const noexcept
    {
        return iterator(d_a_beg, d_a_end,
                        d_b_beg, d_b_end,
                        d_c_beg, d_c_end, true);
    }

    size_type size() const noexcept
    {
        if (d_a_end <= d_a_beg
            || d_b_end <= d_b_beg
            || d_c_end <= d_c_beg) return 0;
        const auto na = static_cast<size_type>(d_a_end - d_a_beg);
        const auto nb = static_cast<size_type>(d_b_end - d_b_beg);
        const auto nc = static_cast<size_type>(d_c_end - d_c_beg);
        return na * nb * nc;
    }

private:
    T d_a_beg{}, d_a_end{};
    T d_b_beg{}, d_b_end{};
    T d_c_beg{}, d_c_end{};
};

} // namespace detail

/**
 * @brief Create a cartesian-product view over two iota ranges.
 * @tparam T Coordinate value type.
 * @param a_beg Inclusive beginning of the first range.
 * @param a_end Exclusive end of the first range.
 * @param b_beg Inclusive beginning of the second range.
 * @param b_end Exclusive end of the second range.
 * @return Portable lazy view of `(a, b)` tuples.
 */
template <class T>
auto cartesian_product_2d(T a_beg, T a_end, T b_beg, T b_end)
{
    return detail::CartesianProductView2D<T>(a_beg, a_end, b_beg, b_end);
}

/**
 * @brief Create a cartesian-product view over three iota ranges.
 * @tparam T Coordinate value type.
 * @param a_beg Inclusive beginning of the first range.
 * @param a_end Exclusive end of the first range.
 * @param b_beg Inclusive beginning of the second range.
 * @param b_end Exclusive end of the second range.
 * @param c_beg Inclusive beginning of the third range.
 * @param c_end Exclusive end of the third range.
 * @return Portable lazy view of `(a, b, c)` tuples.
 */
template <class T>
auto cartesian_product_3d(T a_beg, T a_end,
                          T b_beg, T b_end,
                          T c_beg, T c_end)
{
    return detail::CartesianProductView3D<T>(a_beg, a_end,
                                             b_beg, b_end,
                                             c_beg, c_end);
}

#endif // __cpp_lib_ranges_cartesian_product

} // namespace SimpleFluid::Meshes
