#pragma once
#include "CompleteSupportOverlay.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Exact existing SurfaceRequest encoding, generated only after an authority
 * asks for a member of its current dirty complete support. */
[[nodiscard]] AdaptiveBytes encode_support_surface(const AdaptiveLeaf&, std::size_t surface_ordinal);
[[nodiscard]] std::size_t support_surface_encoded_size(const AdaptiveLeaf&, std::size_t surface_ordinal);
[[nodiscard]] SupportOccurrence decode_support_surface(std::span<const char>, std::span<const char> expected_support,
    int owner, std::size_t byte_budget);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
