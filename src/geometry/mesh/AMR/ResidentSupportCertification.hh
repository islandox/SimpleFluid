#pragma once
#include "DistributedAdaptiveIncidence.hh"
#include "TileRecipeCodec.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** One complete group of received tiles, and all expected locally owned
 * memberships for that support. Caller pre-admits all scratch and geometry. */
void certify_resident_support(std::span<const char> support, std::span<const RegistryMember> local_members,
    std::span<const ReceivedTileRecipe>, int resident, const OwnedLeafLookup&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
