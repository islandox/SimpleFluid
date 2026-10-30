/** @file plic_algebraic_native_cell37_fixture.hh
 * @brief Same-candidate native target37 with selected first ring and complete halo.
 *
 * Source: algebraic_vof_continuation_20261030/native-cell37, captured 2026-10-30.
 * All target+20 rows match actual kernel input-043.json in every retained field.
 * The same live Data physical minority and finite
 * p64 bulk corroborate the captured fitting labels and binary64 publication.
 * No labels from earlier calls or accepted traces are combined here.
 * Native support packets retain represented source identity, not old pointers.
 */
#pragma once
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace captured_native_cell37
{
using namespace SimpleFluid;
using WidePoint = std::array<long double, 3>;
using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using Sample = PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>;
struct Row
{
    std::uint64_t id;
    PLIC::GeometryPoint origin;
    int frame_exponent;
    std::array<WidePoint, 8> stored_vertices;
    double gas, carrier;
    bool observed_mixed;
    long double stored_volume;
    std::array<unsigned, 6> support_indices;
};
inline const std::vector<std::vector<std::size_t>> faces{
    {3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
inline constexpr std::size_t first_ring_size = 6;
inline constexpr std::array<std::uint64_t, 20> stencil_ids{
    13, 33, 36, 38, 41, 48, 9, 12, 14, 17, 26, 30, 32, 34, 45, 40, 39, 42, 49, 24};
inline constexpr std::array<Row, 21> rows{{
    {37, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.25b6e6c954330p-20, 0x1.ffeda491936acp-8, true, 0x1p-7L, {0, 1, 2, 3, 4, 5}},
    {13, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x1.146a04e42c92ep-10, 0x1.bae57ec6f4db4p-8, true, 0x1p-7L, {0, 1, 5, 6, 7, 8}},
    {33, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.a380a15f3ca43p-12, 0x1.e5c7f5ea0c35cp-8, true, 0x1p-7L, {0, 1, 5, 4, 3, 9}},
    {36, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.6b1787c8d0c89p-15, 0x1.fd29d0f06e5e7p-8, true, 0x1p-7L, {10, 0, 2, 3, 4, 5}},
    {38, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {1, 11, 2, 3, 4, 5}},
    {41, {0x1.e666666666666p-1, 0x1.0000000000000p+0, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x1.0b61d8279bdfcp-28, 0x1.ffffef49e27d8p-8, true, 0x1p-7L, {0, 1, 12, 13, 3, 2}},
    {48, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x8000000000001p-53L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x8000000000001p-53L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.fdcbe701bd4f4p-29, 0x1.fffffe0234192p-5, true, 0x10000000000001p-56L, {10, 1, 5, 14, 12, 2}},
    {9, {0x1.8000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.b6a27d716025bp-9, 0x1.24aec1474fed2p-8, true, 0x1p-7L, {0, 1, 8, 7, 15, 16}},
    {12, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x1.1f97f8a38465ep-9, 0x1.703403ae3dcd1p-8, true, 0x1p-7L, {10, 0, 5, 6, 7, 8}},
    {14, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x1.97166db8d5618p-15, 0x1.fcd1d3248e554p-8, true, 0x1p-7L, {1, 11, 5, 6, 7, 8}},
    {17, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x4cccccccccccdp-53L, -0x1p-1L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {0x4cccccccccccdp-53L, -0x1p-1L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.d6d34e522c60ep-8, 0x1.49658d6e9cf94p-11, true, 0x1p-7L, {0, 1, 17, 7, 6, 5}},
    {26, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x3333333333333p-54L, -0x1p-2L, 0x0p-0L},
            {0x999999999999bp-55L, -0x1p-2L, 0x0p-0L},
            {0x8000000000001p-54L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x3333333333333p-54L, -0x1p-2L, 0x1p-1L},
            {0x999999999999bp-55L, -0x1p-2L, 0x1p-1L},
            {0x8000000000001p-54L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.0cb9d7b3ae4b0p-28, 0x1.fffff79a31429p-7, true, 0x20000000000003p-59L, {10, 1, 8, 7, 18, 5}},
    {30, {0x1.6666666666666p-2, 0x1.0000000000000p-1, 0x0.0p+0}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x6666666666667p-54L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x6666666666667p-54L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.f9de9410ffaf9p-9, 0x1.e06216bef004fp-5, true, 0x1fffffffffffffp-57L, {10, 1, 5, 9, 12, 19}},
    {32, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.c0d79a5c3ef4cp-11, 0x1.c7e50cb478216p-8, true, 0x1p-7L, {10, 0, 5, 4, 3, 9}},
    {34, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.7f6018dd8e600p-14, 0x1.fa027f9c89c68p-8, true, 0x1p-7L, {1, 11, 5, 4, 3, 9}},
    {45, {0x1.6666666666666p-1, 0x1.0000000000000p+0, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x4cccccccccccdp-53L, -0x1p-1L, 0x0p-0L},
            {0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {0x4cccccccccccdp-53L, -0x1p-1L, 0x1p-1L},
            {0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.09abd52c7cbcep-26, 0x1.ffffbd950ab4ep-8, true, 0x1p-7L, {0, 1, 9, 3, 13, 12}},
    {40, {0x1.e666666666666p-1, 0x1.0000000000000p+0, 0x0.0p+0}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {10, 0, 12, 13, 3, 2}},
    {39, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.8000000000000p-1}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
        }},
        0x1.4251da93c31e6p-92, 0x1.0000000000000p-7, true, 0x1p-7L, {11, 20, 2, 3, 4, 5}},
    {42, {0x1.e666666666666p-1, 0x1.0000000000000p+0, 0x1.0000000000000p-1}, -1,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x1p-2L, 0x0p-0L, 0x0p-0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x1p-2L, 0x0p-0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x1.62958f802d844p-38, 0x1.fffffffa75a9cp-8, true, 0x1p-7L, {1, 11, 12, 13, 3, 2}},
    {49, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x8000000000001p-53L, 0x0p-0L, 0x0p-0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x8000000000001p-53L, 0x0p-0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.16542c7b99f53p-29, 0x1.fffffee9abd3ap-5, true, 0x10000000000001p-56L, {1, 20, 5, 14, 12, 2}},
    {24, {0x1.199999999999ap+0, 0x1.0000000000000p-1, 0x0.0p+0}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {-0x8000000000001p-54L, 0x0p-0L, 0x0p-0L},
            {-0x16666666666669p-55L, -0x1p-2L, 0x0p-0L},
            {-0xcccccccccccdp-52L, -0x1p-2L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {-0x8000000000001p-54L, 0x0p-0L, 0x1p-1L},
            {-0x16666666666669p-55L, -0x1p-2L, 0x1p-1L},
            {-0xcccccccccccdp-52L, -0x1p-2L, 0x1p-1L},
        }},
        0x1.ae2963dea8c9fp-39, 0x1.fffffffe51d6dp-7, true, 0x20000000000003p-59L, {10, 1, 5, 18, 7, 21}},
}};
// Byte count followed by little-endian words; shared face packets are deduplicated.
inline const std::vector<std::vector<std::uint64_t>> support_keys{
    {0x29ULL, 0x210ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xf3333333000000ULL,
        0x33333000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0x1cfULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL,
        0x0ULL, 0x4000000000ULL, 0xb333333300000000ULL, 0x3333300000000000ULL, 0x200000000ULL, 0x4000000000ULL,
        0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x5100000000ULL, 0x0ULL, 0x400000000000ULL,
        0x3333330000000000ULL, 0x33300000000000f3ULL, 0x20000000033ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL,
        0x0ULL, 0x190000000000ULL, 0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL, 0x0ULL,
        0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xb3333333000000ULL,
        0x33333000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x1000000ULL, 0x0ULL},
    {0x19ULL, 0x210ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1cfULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL,
        0x0ULL, 0x4000000000ULL, 0xb333333300000000ULL, 0x3333300000000000ULL, 0x200000000ULL, 0x4000000000ULL,
        0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x5100000000ULL, 0x0ULL, 0x400000000000ULL,
        0x3333330000000000ULL, 0x33300000000000f3ULL, 0x20000000033ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL,
        0x0ULL, 0x190000000000ULL, 0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x20000ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x20000ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x2000000ULL, 0x40000000ULL,
        0x8ccccccc000000ULL, 0xccccd000000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL,
        0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x19ULL, 0x110ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL, 0x200000000ULL, 0x4000000000ULL, 0x8ccccccc00000000ULL,
        0xccccd00000000000ULL, 0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x4100000000ULL, 0x20000000000ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x190000000000ULL, 0x2010000000000ULL, 0x0ULL, 0x2000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x1fffffffe0000ULL, 0x400000ULL, 0xb33333330000ULL, 0x333330000000ULL, 0x0ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffe000000ULL, 0x40000000ULL,
        0xe6666666000000ULL, 0x66666800000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x20000ULL, 0x400000ULL, 0x8ccccccc0000ULL, 0xccccd0000000ULL, 0x0ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x2000000ULL, 0x40000000ULL,
        0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
};
// The ordinary certificate uses portable numeric limbs, never raw ABI padding.
struct CertificateWriter
{
    std::vector<char> bytes;
    void word(std::uint64_t value)
    {
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
    }
    void support(const std::vector<std::uint64_t>& key)
    {
        const auto count = key.front();
        word(count);
        for (std::uint64_t i = 0; i < count; ++i)
            bytes.push_back(static_cast<char>((key[1 + i / 8] >> (8 * (i % 8))) & 255));
    }
    std::vector<char> certificate() const
    {
        using Header = PLIC::GeometryPacketHeader;
        std::vector<char> packet(sizeof(Header), 0);
        packet[offsetof(Header, family)] = static_cast<char>(PLIC::GeometryPacketFamily::Geometry);
        packet[offsetof(Header, version)] = static_cast<char>(PLIC::geometry_packet_version);
        packet[offsetof(Header, kind)] = static_cast<char>(PLIC::GeometryPacketKind::CellCertificate);
        packet[offsetof(Header, mode)] = static_cast<char>(ArithmeticMode::Ordinary);
        const auto profile = PLIC::geometry_packet_profile(ArithmeticMode::Ordinary);
        if (profile == PLIC::GeometryNumericProfile::Unsupported)
            throw std::invalid_argument("Unsupported captured cell numeric profile.");
        packet[offsetof(Header, profile)] = static_cast<char>(profile);
        const auto length = static_cast<std::uint64_t>(bytes.size());
        for (unsigned i = 0; i < 8; ++i)
            packet[offsetof(Header, body_length) + i] = static_cast<char>((length >> (8 * i)) & 255);
        packet.insert(packet.end(), bytes.begin(), bytes.end());
        return packet;
    }
    void real(long double value)
    {
        if (value == 0)
        {
            word(0);
            word(0);
            return;
        }
        int exponent = 0;
        auto fraction = std::frexp(std::abs(value), &exponent);
        word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
        word(std::numeric_limits<long double>::digits);
        for (int i = 0; i < std::numeric_limits<long double>::digits; i += 32)
        {
            fraction = std::ldexp(fraction, 32);
            const auto limb = static_cast<std::uint32_t>(fraction);
            word(limb);
            fraction -= limb;
        }
    }
    void point(const WidePoint& point)
    {
        for (const auto coordinate : point)
            real(coordinate);
    }
};
inline std::vector<char> certificate(const Row& row)
{
    CertificateWriter writer;
    writer.bytes.push_back(static_cast<char>(PLIC::GeometryTag::OrdinaryCell));
    writer.point({row.origin[0], row.origin[1], row.origin[2]});
    writer.word(row.stored_vertices.size());
    for (auto point : row.stored_vertices)
    {
        for (auto& coordinate : point)
            coordinate = std::ldexp(coordinate, row.frame_exponent);
        writer.point(point);
    }
    writer.word(faces.size());
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
        writer.word(faces[f].size());
        for (const auto vertex : faces[f])
            writer.word(vertex);
        writer.support(support_keys[row.support_indices[f]]);
    }
    return writer.certificate();
}
inline Sample sample(const Row& row)
{
    return {Cell::decode_certificate(certificate(row)), row.gas, row.carrier, row.id, row.observed_mixed};
}
inline PLIC::PolyhedralReconstructionOptions options()
{
    PLIC::PolyhedralReconstructionOptions result;
    result.measure_only = true;
    result.maximum_offset_iterations = 192;
    result.maximum_fit_iterations = 24;
    result.maximum_stencil_cells = 128;
    result.maximum_clip_evaluations = 32768;
    result.volume_tolerance = 8e-15;
    result.fit_tolerance = 2e-9;
    result.maximum_condition = 1e8;
    result.orientation_tolerance = 2e-8;
    result.maximum_model_residual = .2;
    result.minimum_phase_support_fraction = .5;
    result.minimum_optimized_phase_length_ratio = .05;
    return result;
}

} // namespace captured_native_cell37
