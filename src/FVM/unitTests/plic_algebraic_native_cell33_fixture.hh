/** @file plic_algebraic_native_cell33_fixture.hh
 * @brief Same-candidate native target33 with selected first ring and complete halo.
 *
 * Source: algebraic_vof_continuation_20261030/native-cell33, captured 2026-10-30.
 * All target+20 rows match actual kernel input-015.json in every retained field.
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
namespace captured_native_cell33
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
    17, 30, 32, 34, 37, 45, 2, 5, 13, 16, 18, 28, 31, 44, 36, 35, 38, 46, 41, 48};
inline constexpr std::array<Row, 21> rows{{
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
        0x1.154286c3f2be3p-14, 0x1.fbaaf5e4f0350p-8, true, 0x1p-7L, {0, 1, 2, 3, 4, 5}},
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
        0x1.279bd6642ab8ap-8, 0x1.b0c85337aa8ebp-9, true, 0x1p-7L, {0, 1, 6, 7, 8, 2}},
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
        0x1.0860e5741760ap-7, 0x1.bde7c6a2fa27cp-5, true, 0x1fffffffffffffp-57L, {9, 1, 2, 5, 10, 11}},
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
        0x1.36e46a2b87c1fp-13, 0x1.f648dcaea3c1fp-8, true, 0x1p-7L, {9, 0, 2, 3, 4, 5}},
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
        0x1.082ab0a575060p-21, 0x1.fff7beaa7ad45p-8, true, 0x1p-7L, {1, 12, 2, 3, 4, 5}},
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
        0x1.b6c844ad2c55ap-141, 0x1.0000000000000p-7, true, 0x1p-7L, {0, 1, 13, 4, 3, 2}},
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
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {0, 1, 5, 4, 14, 10}},
    {2, {0x1.0000000000000p-2, 0x0.0p+0, 0x0.0p+0}, 0,
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
        0x1.ffa19ccddfbf3p-5, 0x1.798cc881035e4p-15, true, 0x1p-4L, {9, 1, 15, 6, 2, 16}},
    {5, {0x1.0000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-2}, -1,
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
        0x1.f0ea61da0848fp-8, 0x1.e2b3c4bef6e26p-13, true, 0x1p-7L, {0, 1, 15, 17, 7, 6}},
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
        0x1.b80ee1a3b003ap-18, 0x1.ff91fc4797140p-8, true, 0x1p-7L, {0, 1, 2, 8, 7, 18}},
    {16, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
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
        0x1.94a0fd403774fp-8, 0x1.ad7c0aff222c4p-10, true, 0x1p-7L, {9, 0, 6, 7, 8, 2}},
    {18, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, -1,
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
        0x1.1be0bd2bbf9e0p-9, 0x1.720fa16a20310p-8, true, 0x1p-7L, {1, 12, 6, 7, 8, 2}},
    {28, {0x1.999999999999ap-4, 0x1.0000000000000p-1, 0x0.0p+0}, 0,
        {{
            {0x0p-0L, 0x0p-0L, 0x0p-0L},
            {0x1fffffffffffffp-55L, 0x0p-0L, 0x0p-0L},
            {0x2ccccccccccccdp-55L, 0x1p-1L, 0x0p-0L},
            {0xccccccccccccdp-55L, 0x1p-1L, 0x0p-0L},
            {0x0p-0L, 0x0p-0L, 0x1p-1L},
            {0x1fffffffffffffp-55L, 0x0p-0L, 0x1p-1L},
            {0x2ccccccccccccdp-55L, 0x1p-1L, 0x1p-1L},
            {0xccccccccccccdp-55L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.c9af7c35f821ep-5, 0x1.b2841e503ef0fp-8, true, 0x3fffffffffffffp-58L, {9, 1, 2, 11, 10, 19}},
    {31, {0x1.6666666666666p-2, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, 0,
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
        0x1.ce957d6208ebcp-10, 0x1.f18b5414efb89p-5, true, 0x1fffffffffffffp-57L, {1, 20, 2, 5, 10, 11}},
    {44, {0x1.6666666666666p-1, 0x1.0000000000000p+0, 0x0.0p+0}, -1,
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
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {9, 0, 5, 4, 14, 10}},
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
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {9, 0, 13, 4, 3, 2}},
    {35, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.8000000000000p-1}, -1,
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
        0x1.768f947ff35a2p-27, 0x1.ffffd12e0d700p-8, true, 0x1p-7L, {12, 20, 2, 3, 4, 5}},
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
        0x1.239785257daa3p-72, 0x1.0000000000000p-7, true, 0x1p-7L, {1, 12, 13, 4, 3, 2}},
    {46, {0x1.6666666666666p-1, 0x1.0000000000000p+0, 0x1.0000000000000p-1}, -1,
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
        0x1.746aae23d6ed7p-40, 0x1.fffffffe8b955p-8, true, 0x1p-7L, {1, 12, 5, 4, 14, 10}},
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
        0x1.7526de147fbd8p-27, 0x1.ffffd15b243d7p-8, true, 0x1p-7L, {0, 1, 10, 14, 4, 13}},
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
        0x1.18559d3a3224dp-29, 0x1.fffffee7aa62ep-5, true, 0x10000000000001p-56L, {9, 1, 2, 21, 10, 13}},
}};
// Byte count followed by little-endian words; shared face packets are deduplicated.
inline const std::vector<std::vector<std::uint64_t>> support_keys{
    {0x29ULL, 0x210ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1cfULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL,
        0x0ULL, 0x4000000000ULL, 0xb333333300000000ULL, 0x3333300000000000ULL, 0x200000000ULL, 0x4000000000ULL,
        0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x5100000000ULL, 0x0ULL, 0x400000000000ULL,
        0x3333330000000000ULL, 0x33300000000000f3ULL, 0x20000000033ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL,
        0x0ULL, 0x190000000000ULL, 0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xb3333333000000ULL,
        0x33333000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x1000000ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL, 0x0ULL,
        0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x19ULL, 0x210ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x1fffffffe0000ULL, 0x400000ULL, 0xb33333330000ULL, 0x333330000000ULL, 0x0ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffe000000ULL, 0x40000000ULL,
        0xe6666666000000ULL, 0x66666800000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xf3333333000000ULL,
        0x33333000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x1000000ULL, 0x0ULL},
    {0x1cfULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL,
        0x0ULL, 0x4000000000ULL, 0xb333333300000000ULL, 0x3333300000000000ULL, 0x200000000ULL, 0x4000000000ULL,
        0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x5100000000ULL, 0x0ULL, 0x400000000000ULL,
        0x3333330000000000ULL, 0x33300000000000f3ULL, 0x20000000033ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL,
        0x0ULL, 0x190000000000ULL, 0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0x19ULL, 0x110ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x4100ULL, 0x1fffffffe0000ULL, 0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffe000000ULL, 0x40000000ULL, 0xb3333333000000ULL, 0x33333000000000ULL,
        0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL,
        0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x1fffffffa0000ULL, 0x400000ULL, 0xcccccccc0000ULL, 0xccccd0000000ULL, 0x0ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffc000000ULL, 0x40000000ULL,
        0xcccccccc000000ULL, 0xccccd000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x20000ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x20000ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x2000000ULL, 0x40000000ULL,
        0x8ccccccc000000ULL, 0xccccd000000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
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

} // namespace captured_native_cell33
