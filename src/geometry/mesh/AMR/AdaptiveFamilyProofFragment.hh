/** @file AdaptiveFamilyProofFragment.hh
 * @brief Bounded exact family records with borrowed dictionary selectors.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using AdaptiveFamilyProofPart = std::shared_ptr<const AdaptiveBytes>;
struct AdaptiveFamilyProofView
{
    std::span<const char> record;
    std::span<const AdaptiveFamilyProofPart> parts;
};
struct AdaptiveFamilyProofFragment
{
    AdaptiveBytes record;
    std::vector<AdaptiveFamilyProofPart> parts;
    [[nodiscard]] AdaptiveFamilyProofView view() const noexcept { return {record, parts}; }
};

/** The record uses the same exact fields as the batch codec, with part IDs
 * scoped to this borrowed selector array. No recipe bytes are copied here. */
[[nodiscard]] AdaptiveFamilyProofFragment encode_family_proof_fragment(const AdaptiveFamily&, std::size_t budget);
[[nodiscard]] AdaptiveFamily decode_family_proof_fragment(AdaptiveFamilyProofView, std::size_t budget);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
