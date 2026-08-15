/**
 * @file DEBUG_MACROS.h
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Preprocessor macro utilities for argument counting and dispatch for debug checks.
 *
 * `CHECK` and overflow checks compile out when `NDEBUG` is defined.
 * `CHECK_BOUNDS` does likewise unless
 * `SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS` is enabled explicitly.
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#define CAT(a, b) CAT_I(a, b)
#define CAT_I(a, b) a##b

#define NARGS_I(_0, _1, _2, _3, _4, N, ...) N
#define NARGS(...) NARGS_I(_0 __VA_OPT__(,) __VA_ARGS__, 4, 3, 2, 1, 0)

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

#ifdef DEBUG_CHECK_ENABLED
    #define CHECK_OVERFLOW3(op, v1, v2) utils::check_overflow((v1), (v2), (op), \
        std::string("Debug check failed: overflow detected for ") + #v1 + " and " + #v2 \
        + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))
    #define CHECK_OVERFLOW4(op, v1, v2, v3) \
        utils::check_overflow((v1), (v2), (v3), (op), \
        std::string("Debug check failed: overflow detected for ") + #v1 + ", " + #v2 + ", and " + #v3 \
        + "\n\tAt " + __FILE__ + ":" + std::to_string(__LINE__))

    #define CHECK_SUM_OVERFLOW(...) CHECK_OVERFLOW(std::plus<>(), __VA_ARGS__)
    #define CHECK_PRODUCT_OVERFLOW(...) CHECK_OVERFLOW(std::multiplies<>(), __VA_ARGS__)
#else
    #define CHECK_OVERFLOW3(v1, v2, op) // do nothing
    #define CHECK_OVERFLOW4(v1, v2, v3, op) // do nothing
    #define CHECK_SUM_OVERFLOW(...) // do nothing
    #define CHECK_PRODUCT_OVERFLOW(...) // do nothing
#endif

#ifdef CHECK_BOUNDS_ENABLED
    #define CHECK_BOUNDS(value, lower, upper) \
        utils::check<std::out_of_range>(((value) >= (lower)) && ((value) < (upper)))
#else
    #define CHECK_BOUNDS(value, lower, upper) // do nothing
#endif

#define CHECK(...) DISPATCH(CHECK, __VA_ARGS__)
#define CHECK_OVERFLOW(...) DISPATCH(CHECK_OVERFLOW, __VA_ARGS__)
