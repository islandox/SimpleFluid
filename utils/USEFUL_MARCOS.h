/**
 * @file USEFUL_MARCOS.h
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief 
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