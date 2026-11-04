#pragma once
#include "PooledTile.hh"
#include "TileBodyPages.hh"
#include "TileRecipeCodec.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** The caller admits the returned tile/loop owner and all scratch before
 * calling; this helper moves full identities into the session's pools. The
 * complete recipe is authenticated before entry; only resident-owned incident
 * cells are retained, while both physical images remain available. */
[[nodiscard]] PooledTile freeze_tile(const ReceivedTileRecipe&, int resident_rank, IncidencePools&);
[[nodiscard]] SupportFace restore_tile(const TileBodyScalars&, TileBodyLoopView loop, const IncidencePools&);
[[nodiscard]] inline SupportFace restore_tile(
    const PooledTile& tile, std::span<const std::size_t> loop, const IncidencePools& pools)
{
    return restore_tile(pooled_tile_scalars(tile), TileBodyLoopView(loop), pools);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
