/** @file CellCatalogInternal.hh
 * @brief Direct dependencies and borrowed native recipe decoder declarations.
 */
#pragma once
#include "FVM/details/GeneralVoF/CellCatalog.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
struct RecipeAxes
{
    std::array<std::uint64_t, 3> counts{};
    std::array<std::span<const char>, 3> values;
    explicit RecipeAxes(const Meshes::NativeFragmentDescriptor&);
    double value(unsigned, std::uint64_t) const;
    std::array<std::uint64_t, 3> location(const AMR::AdaptiveLeaf&) const;
    AMR::AdaptiveCellMetrics metrics(const AMR::AdaptiveLeaf&) const;
};
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
