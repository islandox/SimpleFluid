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

#include <stdexcept>

#ifndef NDEBUG
    #define DEBUG_CHECK_ENABLED
#endif

#ifdef DEBUG_CHECK_ENABLED
#define CHECK(condition) utils::check(condition, \
    std::string("Debug check failed: ") + #condition \
    + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))
#else
#define CHECK(condition) // do nothing
#endif

namespace utils
{

/**
 * @brief Perform a debug-only runtime check and throw on failure.
 *
 * @param condition Condition that must be true.
 * @param message Error message used when the check fails.
 */
inline void check(bool condition, std::string message = "Debug check failed")
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}
} // namespace utils