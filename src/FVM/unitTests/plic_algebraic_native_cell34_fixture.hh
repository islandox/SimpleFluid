/** @file plic_algebraic_native_cell34_fixture.hh
 * @brief Same-candidate native target34 with selected first ring and complete halo.
 *
 * Source: algebraic_vof_continuation_20261030/native-cell34, captured 2026-10-30.
 * Target+first6 rows match actual kernel input-015.json in every retained field.
 * Remaining labels derive from the SAME live Data physical minority and finite
 * p64 bulk, using the documented production complement and binary64 publication.
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
namespace captured_native_cell34
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
    18, 31, 33, 35, 38, 46, 3, 6, 14, 17, 19, 29, 30, 47, 32, 37, 45, 39, 42, 49};
inline constexpr std::array<Row, 21> rows{{
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
        0x1.7c3c0b5fb20abp-20, 0x1.ffe83c3f4a04ep-8, true, 0x1p-7L, {0, 1, 2, 3, 4, 5}},
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
        0x1.31534dd12c990p-9, 0x1.6756591769b38p-8, true, 0x1p-7L, {0, 1, 6, 7, 8, 2}},
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
        0x1.a1f81a30d9d0bp-10, 0x1.f2f03f2e79317p-5, true, 0x1fffffffffffffp-57L, {0, 9, 2, 5, 10, 11}},
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
        0x1.26a67f0cbd56ep-14, 0x1.fb656603cd0aap-8, true, 0x1p-7L, {12, 0, 2, 3, 4, 5}},
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
        0x1.ae5bd6060bed5p-27, 0x1.ffffca34853f4p-8, true, 0x1p-7L, {1, 9, 2, 3, 4, 5}},
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
        0x1.213210477271ep-80, 0x1.0000000000000p-7, true, 0x1p-7L, {0, 1, 13, 4, 3, 2}},
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
        0x1.342468b962cadp-42, 0x1.ffffffffb2f6ep-8, true, 0x1p-7L, {0, 1, 5, 4, 14, 10}},
    {3, {0x1.0000000000000p-2, 0x0.0p+0, 0x1.0000000000000p-1}, 0,
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
        0x1.f52ac515d3b62p-5, 0x1.5aa75d45893b2p-10, true, 0x1p-4L, {0, 9, 15, 6, 2, 16}},
    {6, {0x1.0000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-1}, -1,
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
        0x1.c70e41fae3010p-8, 0x1.c78df028e7f81p-11, true, 0x1p-7L, {0, 1, 15, 17, 7, 6}},
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
        0x1.ba6b76570ea5ap-33, 0x1.ffffff22ca44dp-8, true, 0x1p-7L, {0, 1, 2, 8, 7, 18}},
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
        0x1.354ad058323e1p-8, 0x1.956a5f4f9b83ep-9, true, 0x1p-7L, {12, 0, 6, 7, 8, 2}},
    {19, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.8000000000000p-1}, -1,
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
        0x1.1b150e5f62e85p-10, 0x1.b93abc682745fp-8, true, 0x1p-7L, {1, 9, 6, 7, 8, 2}},
    {29, {0x1.999999999999ap-4, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, 0,
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
        0x1.8fd4bcc1f666bp-5, 0x1.c0ad0cf826653p-7, true, 0x3fffffffffffffp-58L, {0, 9, 2, 11, 10, 19}},
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
        0x1.fa695d4e6888bp-8, 0x1.c0b2d45632eeep-5, true, 0x1fffffffffffffp-57L, {20, 0, 2, 5, 10, 11}},
    {47, {0x1.6666666666666p-1, 0x1.0000000000000p+0, 0x1.8000000000000p-1}, -1,
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
        0x1.466760c9cc684p-46, 0x1.fffffffffae66p-8, true, 0x1p-7L, {1, 9, 5, 4, 14, 10}},
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
        0x1.67bc478bbe20ap-13, 0x1.f4c21dc3a20f0p-8, true, 0x1p-7L, {20, 12, 2, 3, 4, 5}},
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
        0x1.cfe3184ac40e1p-141, 0x1.0000000000000p-7, true, 0x1p-7L, {12, 0, 13, 4, 3, 2}},
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
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {12, 0, 5, 4, 14, 10}},
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
        0x1.e3e061a8602e2p-25, 0x1.ffff0e0fcf2bdp-8, true, 0x1p-7L, {1, 9, 13, 4, 3, 2}},
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
        0x1.8f96ae8d1984ep-30, 0x1.fffff9c1a545dp-8, true, 0x1p-7L, {0, 1, 10, 14, 4, 13}},
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
        0x1.4a80ff6f2b5a5p-33, 0x1.ffffffeb57f03p-5, true, 0x10000000000001p-56L, {0, 9, 2, 21, 10, 13}},
}};
// Byte count followed by little-endian words; shared face packets are deduplicated.
inline const std::vector<std::vector<std::uint64_t>> support_keys{
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
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
    {0x29ULL, 0x210ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x200ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x1fffffffe0000ULL, 0x400000ULL, 0xb33333330000ULL, 0x333330000000ULL, 0x0ULL,
        0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffe000000ULL, 0x40000000ULL,
        0xe6666666000000ULL, 0x66666800000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
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
    {0x19ULL, 0x210ULL, 0x0ULL, 0x0ULL, 0x0ULL},
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

} // namespace captured_native_cell34
