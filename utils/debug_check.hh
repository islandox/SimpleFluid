/**
 * @file debug_check.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "USEFUL_MARCOS.h"

#include <stdexcept>

#ifndef NDEBUG
    #define DEBUG_CHECK_ENABLED
#endif

#ifdef DEBUG_CHECK_ENABLED

#define CHECK1(condition) utils::check<>((condition), \
    std::string("Debug check failed: ") + #condition \
    + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))

#define CHECK2(condition, message) utils::check<>((condition), \
    std::string("Debug check failed: ") + (message) \
    + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))

#define CHECK3(condition, message, error) utils::check<error>((condition), \
    std::string("Debug check failed: ") + (message) \
    + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))

#else
#define CHECK1(condition) // do nothing
#define CHECK2(condition, message) // do nothing
#define CHECK3(condition, message, error) // do nothing
#endif

#define CHECK(...) DISPATCH(CHECK, __VA_ARGS__)

namespace utils
{

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

} // namespace utils