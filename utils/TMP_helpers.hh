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

namespace utils
{

template <typename T>
struct always_false
{
    static constexpr bool value = false;
};

template <typename T>
inline constexpr bool always_false_v = always_false<T>::value;

} // namespace utils