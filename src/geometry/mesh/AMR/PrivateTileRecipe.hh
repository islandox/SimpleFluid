/** @file PrivateTileRecipe.hh
 * @brief Transient complete-support tile recipes before compact pool internment.
 */
#pragma once
#include "CompleteSupportOverlay.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct PrivateTileRecipe
{
    std::size_t first_member = 0;
    std::optional<std::size_t> second_member;
    SupportFace face; // first-cell physical image; caller interns frame and loop
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> second_image;
};
using RecipeEmit = BorrowedSupportCallback<PrivateTileRecipe>;

/** Run only on one complete support. Duplicate canonical identities are
 * checked here and then destroyed; output does not retain identity bytes. */
void make_support_recipes(std::span<const char> support, std::span<const SupportOccurrence> occurrences,
    std::size_t byte_budget, std::size_t tile_budget, const SupportCharge&, const SupportChargeFace&,
    const RecipeEmit&, const SupportConstructionAdmission* = nullptr);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
