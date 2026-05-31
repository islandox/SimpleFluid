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

#include <stdexcept>


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