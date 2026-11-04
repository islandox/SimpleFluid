#pragma once
#include "IncidencePools.hh"
#include <limits>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Compact owner-side tile state. Slot order is never used as canonical order.
 * The intrinsic layer is the only seed on a deletion replay. Cell slots retain
 * only this resident's owned incidences; authenticated source memberships stay
 * in their complete registries. Normal/image values carry no authority. */
struct PooledTile
{
    using Slot = std::size_t;
    static constexpr Slot absent_cell = std::numeric_limits<Slot>::max();
    Slot support = 0, domain = 0, plane = 0, normal_slot = 0, image_slot = 0;
    Slot first_cell = absent_cell, second_cell = absent_cell;
    std::uint64_t emission = 0;
    std::vector<Slot> intrinsic_loop;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
