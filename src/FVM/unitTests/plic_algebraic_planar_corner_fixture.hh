/** @file plic_algebraic_planar_corner_fixture.hh
 * @brief Ordinary accepted-source stencil captured before the first GeometricFace step.
 *
 * Captured on 2026-10-28 from PlanarPatch_0_2_spatial_N16_N16_C50, gid 106,
 * owner 0, complete two-ring halo (target plus 17 neighbors). The rows retain
 * production persistent-key BFS order and independent binary64 inventories.
 * Dyadic .0625 x .0625 x .5 boxes reproduce the represented source geometry;
 * lossless native certificates remain in build verification artifacts.
 *
 * This coarse polygon-corner stencil is a negative plane-model qualification
 * case. It does not establish a resolved-interface positive support claim.
 */
#pragma once
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include <array>
#include <cstdint>

namespace captured_algebraic_corner
{
using namespace SimpleFluid;
struct Row
{
    std::uint64_t gid;
    unsigned ring;
    PLIC::GeometryPoint lower, upper;
    double physical_gas, physical_liquid;
};
// Source: algebraic_vof_20261028/q2-geometric-planar-cartesian-c50/failing-states/
// PlanarPatch_0_2_spatial_N16_N16_C50_rank0.state.txt and replay-gid106-rank0.txt.
inline constexpr std::array<Row, 18> rows{{
    {106, 0, {0x1.8000000000000p-3, 0x1.4000000000000p-2, 0x0.0p+0},
        {0x1.0000000000000p-2, 0x1.8000000000000p-2, 0x1.0000000000000p-1}, 0x1.d065a6e428661p-11,
        0x1.17cd2c8debccfp-10},
    {74, 1, {0x1.0000000000000p-3, 0x1.4000000000000p-2, 0x0.0p+0},
        {0x1.8000000000000p-3, 0x1.8000000000000p-2, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {104, 1, {0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x0.0p+0},
        {0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p-1}, 0x1.67af68e57d27cp-14,
        0x1.e9850971a82d8p-10},
    {107, 1, {0x1.8000000000000p-3, 0x1.4000000000000p-2, 0x1.0000000000000p-1},
        {0x1.0000000000000p-2, 0x1.8000000000000p-2, 0x1.0000000000000p+0}, 0x1.d065a6e428661p-11,
        0x1.17cd2c8debccfp-10},
    {108, 1, {0x1.8000000000000p-3, 0x1.8000000000000p-2, 0x0.0p+0},
        {0x1.0000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-1}, 0x1.b2c59da164d5ep-11,
        0x1.269d312f4d951p-10},
    {138, 1, {0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x0.0p+0},
        {0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.0000000000000p-1}, 0x1.df0eb45136c8bp-10,
        0x1.078a5d7649babp-13},
    {42, 2, {0x1.0000000000000p-4, 0x1.4000000000000p-2, 0x0.0p+0},
        {0x1.0000000000000p-3, 0x1.8000000000000p-2, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {72, 2, {0x1.0000000000000p-3, 0x1.0000000000000p-2, 0x0.0p+0},
        {0x1.8000000000000p-3, 0x1.4000000000000p-2, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {75, 2, {0x1.0000000000000p-3, 0x1.4000000000000p-2, 0x1.0000000000000p-1},
        {0x1.8000000000000p-3, 0x1.8000000000000p-2, 0x1.0000000000000p+0}, 0x0.0p+0, 0x1.0000000000000p-9},
    {76, 2, {0x1.0000000000000p-3, 0x1.8000000000000p-2, 0x0.0p+0},
        {0x1.8000000000000p-3, 0x1.c000000000000p-2, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {102, 2, {0x1.8000000000000p-3, 0x1.8000000000000p-3, 0x0.0p+0},
        {0x1.0000000000000p-2, 0x1.0000000000000p-2, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {105, 2, {0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.0000000000000p-1},
        {0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p+0}, 0x1.67af68e57d27cp-14,
        0x1.e9850971a82d8p-10},
    {136, 2, {0x1.0000000000000p-2, 0x1.0000000000000p-2, 0x0.0p+0},
        {0x1.4000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p-1}, 0x1.1d9d03e2d6b1bp-14,
        0x1.ee262fc1d294ep-10},
    {109, 2, {0x1.8000000000000p-3, 0x1.8000000000000p-2, 0x1.0000000000000p-1},
        {0x1.0000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p+0}, 0x1.b2c59da164d5ep-11,
        0x1.269d312f4d951p-10},
    {139, 2, {0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p-1},
        {0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.0000000000000p+0}, 0x1.df0eb45136c8bp-10,
        0x1.078a5d7649babp-13},
    {110, 2, {0x1.8000000000000p-3, 0x1.c000000000000p-2, 0x0.0p+0},
        {0x1.0000000000000p-2, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, 0x0.0p+0, 0x1.0000000000000p-9},
    {140, 2, {0x1.0000000000000p-2, 0x1.8000000000000p-2, 0x0.0p+0},
        {0x1.4000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-1}, 0x1.cd76cc871cd45p-10,
        0x1.94499bc7195d4p-13},
    {170, 2, {0x1.4000000000000p-2, 0x1.4000000000000p-2, 0x0.0p+0},
        {0x1.8000000000000p-2, 0x1.8000000000000p-2, 0x1.0000000000000p-1}, 0x1.2a52847a1f10cp-10,
        0x1.ab5af70bc1de8p-11},
}};
// Independently assembled long-double weighted-LS direction in the diagnostic
// replay, before any production angular search. This is an offset-feasibility
// witness, not an optimized orientation or an independent plane-model oracle.
inline constexpr PLIC::GeometryPoint diagnostic_normal{
    -0x1.cd6b86e9fd0c2p-1, -0x1.ba88a1d8c2ea4p-2, 0x1.0a11cb6f92672p-5};
inline PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary> fit_sample(const Row& row)
{
    const auto geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(row.lower, row.upper);
    // Match production reconstruction_sample: complement only the dominant
    // geometric fitting label, preserving both independently stored inventories.
    const bool gas = row.physical_gas <= row.physical_liquid;
    const double minority = gas ? row.physical_gas : row.physical_liquid;
    const double dominant = static_cast<double>(geometry->volume().value - minority);
    return {geometry, gas ? minority : dominant, gas ? dominant : minority};
}
} // namespace captured_algebraic_corner
