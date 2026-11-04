#pragma once
#include "PooledTile.hh"
#include <functional>
#include <span>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Visits actual physical images of an exact loop for each local incident
 * cell retained for the tile's resident. The second image uses the same sign
 * as the legacy authority owner reconstruction. Slot lookups compare full
 * typed identities in their pools. The rank must be that frozen resident. */
void visit_tile_physical_points(const PooledTile&, std::span<const std::size_t> loop, int rank, IncidencePools&,
    const std::function<void(std::size_t cell, std::size_t point)>&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
