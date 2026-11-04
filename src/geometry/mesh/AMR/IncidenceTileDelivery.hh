/** @file IncidenceTileDelivery.hh
 * @brief Paged authenticated delivery of complete support recipes to owners.
 */
#pragma once
#include "DistributedAdaptiveIncidence.hh"
#include "TileRecipeCodec.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct IncidenceDeliverySource
{
    std::span<const char> support, domain;
    std::span<const RegistryMember> members;
    std::span<const PrivateTileRecipe> recipes;
    std::uint64_t generation = 0;
};
struct IncidenceDeliveryExpected
{
    std::span<const char> support;
    std::uint64_t minimum_generation = 0;
    std::span<const RegistryMember> local_members;
};
/** Each rank supplies zero or more complete source groups and an independent
 * manifest of every local group expected during this collective. Multiple
 * records share byte-bounded transport pages. All receipts and partitions are checked
 * collectively before callbacks. Callbacks construct candidate-only state;
 * they must admit and own every allocation retained beyond the callback. */
void deliver_incidence_tiles(std::span<const IncidenceDeliverySource> sources,
    std::span<const IncidenceDeliveryExpected> expected, ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>,
    Budget&, IdentityBudget&, const OwnedLeafLookup&,
    const std::function<void(std::span<const char>, std::uint64_t, std::span<const ReceivedTileRecipe>)>&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
