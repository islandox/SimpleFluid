/**
 * @file DEBUG_MACROS.h
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Preprocessor macro utilities for argument counting and dispatch for debug checks.
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#define CAT(a, b) CAT_I(a, b)
#define CAT_I(a, b) a##b

#define NARGS_I(_0, _1, _2, _3, N, ...) N
#define NARGS(...) NARGS_I(_0 __VA_OPT__(,) __VA_ARGS__, 3, 2, 1, 0)

#define DISPATCH(name, ...) CAT(name, NARGS(__VA_ARGS__))(__VA_ARGS__)


#ifndef NDEBUG
    #define DEBUG_CHECK_ENABLED
    #define CHECK_BOUNDS_ENABLED
#endif

#ifdef SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS
    #ifndef CHECK_BOUNDS_ENABLED
        #define CHECK_BOUNDS_ENABLED
    #endif
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

#ifdef CHECK_BOUNDS_ENABLED
    #define CHECK_BOUNDS(value, lower, upper) \
        utils::check<std::out_of_range>(((value) >= (lower)) && ((value) < (upper)))
#else
    #define CHECK_BOUNDS(value, lower, upper) // do nothing
#endif

#define CHECK(...) DISPATCH(CHECK, __VA_ARGS__)