/**
 * @file debug_check.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Debug-only runtime assertion macros with configurable exception types.
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "DEBUG_MACROS.h"
#include "TMP_helpers.hh"

#include <concepts>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace utils
{

namespace detail
{

/**
 * @brief Determine whether adding two same-type integers would overflow.
 *
 * @tparam T Integral operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return `true` if `lhs + rhs` is not representable by @p T.
 */
template <std::integral T>
constexpr bool addition_overflows(T lhs, T rhs) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
    {
        return lhs > std::numeric_limits<T>::max() - rhs;
    }
    else
    {
        return (rhs > 0 && lhs > std::numeric_limits<T>::max() - rhs)
            || (rhs < 0 && lhs < std::numeric_limits<T>::min() - rhs);
    }
}

/**
 * @brief Determine whether multiplying two same-type integers would overflow.
 *
 * @tparam T Integral operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return `true` if `lhs * rhs` is not representable by @p T.
 */
template <std::integral T>
constexpr bool multiplication_overflows(T lhs, T rhs) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
    {
        return rhs != 0 && lhs > std::numeric_limits<T>::max() / rhs;
    }
    else
    {
        if (lhs == 0 || rhs == 0) return false;

        if (lhs == -1) return rhs == std::numeric_limits<T>::min();
        if (rhs == -1) return lhs == std::numeric_limits<T>::min();
        if (lhs > 0)
        {
            return rhs > 0
                ? lhs > std::numeric_limits<T>::max() / rhs
                : rhs < std::numeric_limits<T>::min() / lhs;
        }
        return rhs > 0
            ? lhs < std::numeric_limits<T>::min() / rhs
            : lhs < std::numeric_limits<T>::max() / rhs;
    }
}

} // namespace detail

/**
 * @brief Perform a debug-only runtime check and throw on failure.
 *
 * @param condition Condition that must be true.
 * @param message Error message used when the check fails.
 * @tparam Error Type of exception to throw on failure (default: std::runtime_error).
 */
template <typename Error = std::runtime_error>
inline void check(bool condition, std::string message = "Debug check failed")
{
    if (!condition)
    {
        throw Error(message);
    }
}

/**
 * @brief Check whether applying an arithmetic operation would overflow.
 *
 * @tparam T1 Left operand integral type.
 * @tparam T2 Right operand integral type.
 * @tparam Op Supported arithmetic functor type.
 * @param v1 Left operand.
 * @param v2 Right operand.
 * @param op Binary operation to check for overflow (e.g., std::multiplies<>()).
 * @param message Error message used when the check fails.
 * @throws std::overflow_error If the requested operation would overflow its
 *         result type.
 */
template <std::integral T1, std::integral T2, class Op>
requires std::invocable<Op, T1, T2>
      && std::integral<std::remove_cvref_t<std::invoke_result_t<Op, T1, T2>>>
inline void check_overflow(
    T1 v1,
    T2 v2,
    Op&&,
    std::string message = "Debug check failed: overflow detected.")
{
    using operation_type = std::remove_cvref_t<Op>;
    using result_type =
        std::remove_cvref_t<std::invoke_result_t<Op, T1, T2>>;

    const auto lhs = static_cast<result_type>(v1);
    const auto rhs = static_cast<result_type>(v2);

    bool overflow = false;
    if constexpr (TMP::is_multiplies<operation_type>::value)
    {
        overflow = detail::multiplication_overflows(lhs, rhs);
    }
    else if constexpr (TMP::is_plus<operation_type>::value)
    {
        overflow = detail::addition_overflows(lhs, rhs);
    }
    else
    {
        static_assert(TMP::always_false_v<operation_type>,
                      "check_overflow only supports std::plus and "
                      "std::multiplies.");
    }

    if (overflow)
    {
        throw std::overflow_error(message);
    }
}

/**
 * @brief Check a left-associative operation involving three values.
 *
 * @tparam T1 First operand integral type.
 * @tparam T2 Second operand integral type.
 * @tparam T3 Third operand integral type.
 * @tparam Op Supported arithmetic functor type.
 * @param v1 First operand.
 * @param v2 Second operand.
 * @param v3 Third operand.
 * @param op Binary operation applied from left to right.
 * @param message Error message used when either operation overflows.
 * @throws std::overflow_error If either left-associative operation would
 *         overflow its result type.
 */
template <std::integral T1, std::integral T2, std::integral T3, class Op>
requires std::invocable<Op&, T1, T2>
inline void check_overflow(
    T1 v1,
    T2 v2,
    T3 v3,
    Op&& op,
    std::string message = "Debug check failed: overflow detected.")
{
    check_overflow(v1, v2, op, message);
    check_overflow(std::invoke(op, v1, v2), v3, op, std::move(message));
}

} // namespace utils
