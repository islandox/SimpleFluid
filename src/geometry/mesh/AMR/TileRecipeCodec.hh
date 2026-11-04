#pragma once
#include "PrivateTileRecipe.hh"
#include "SupportRegistry.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ReceivedTileRecipe
{
    AdaptiveBytes support, domain;
    RegistryMember first;
    std::optional<RegistryMember> second;
    SupportFace face;
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> second_image;
    std::uint64_t emission = 0;
};
/** Tile packets omit full canonical-face identity; duplicates were checked on
 * the complete support. They are temporary, until owner frame/point interning. */
[[nodiscard]] AdaptiveBytes encode_tile_recipe(std::span<const char> support, std::span<const char> domain,
    std::span<const RegistryMember>, const PrivateTileRecipe&, std::uint64_t emission);
[[nodiscard]] std::size_t tile_recipe_encoded_size(std::span<const char> support, std::span<const char> domain,
    std::span<const RegistryMember>, const PrivateTileRecipe&);
[[nodiscard]] ReceivedTileRecipe decode_tile_recipe(std::span<const char>, int source_authority, int resident,
    int ranks, std::size_t budget);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
