#pragma once
#include "PooledTile.hh"
#include "TileRecipeCodec.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** The caller admits the returned tile/loop owner and all scratch before
 * calling; this helper moves full identities into the session's pools. The
 * complete recipe is authenticated before entry; only resident-owned incident
 * cells are retained, while both physical images remain available. */
[[nodiscard]] PooledTile freeze_tile(const ReceivedTileRecipe&, int resident_rank, IncidencePools&);
[[nodiscard]] SupportFace restore_tile(const PooledTile&, std::span<const std::size_t> loop, const IncidencePools&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
