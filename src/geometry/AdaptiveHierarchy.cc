/** @file AdaptiveHierarchy.cc
 * @brief Native hierarchy lifetime and checked Cartesian parent ancestry.
 */
#include "geometry/AdaptiveHierarchy.hh"
#include <stdexcept>

namespace SimpleFluid::Meshes
{
AdaptiveHierarchy::~AdaptiveHierarchy() = default;

CartesianCellAncestry CartesianCellAncestry::parent() const
{
    if (level == 0)
        throw std::invalid_argument("Root Cartesian cells have no adaptive parent.");
    auto result = *this;
    --result.level;
    for (auto& coordinate : result.index)
        coordinate /= 2;
    return result;
}
} // namespace SimpleFluid::Meshes
