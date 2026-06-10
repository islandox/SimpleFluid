/**
 * @file TMP_helpers.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief template meta programming helper functions
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <type_traits>
#include <concepts>
#include <functional>

namespace utils::TMP
{

/**
 * @brief Always-false type trait used for static_assert in constexpr-if branches.
 *
 * @tparam T Any type.
 */
template <typename T>
struct always_false
{
    static constexpr bool value = false;
};

/**
 * @brief Returns an invalid sentinel value of type T (cast from -1).
 *
 * @tparam T Any integral or enum type.
 * @return constexpr T The value static_cast<T>(-1).
 */
template <typename T>
requires (std::is_enum_v<T> || std::integral<T>)
constexpr T invalid_value() noexcept { return static_cast<T>(-1); }

/**
 * @brief Convenience variable template for always_false.
 *
 * @tparam T Any type.
 */
template <typename T>
inline constexpr bool always_false_v = always_false<T>::value;


/**
 * @brief Type trait to detect std::plus<T>.
 *
 * @tparam The type to check.
 */
template <class>
struct is_plus : std::false_type
{
};

template <class T>
struct is_plus<std::plus<T>> : std::true_type
{
};

/**
 * @brief Type trait to detect std::multiplies<T>.
 *
 * @tparam The type to check.
 */
template <class>
struct is_multiplies : std::false_type
{
};

template <class T>
struct is_multiplies<std::multiplies<T>> : std::true_type
{
};


} // namespace utils::TMP