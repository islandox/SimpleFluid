/** @file ExtrudedConstructionTypes.hh
 * @brief Mode-selected affine constructions for native extrusions.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
namespace SimpleFluid::PLIC
{
using Meshes::ExtrudedTopology;
/** Empty parents identify represented roots; otherwise the point is the exact
 * arithmetic mean of earlier recipe points. Display coordinates may round. */
struct PointRecipe
{
    GeometryPoint root{};
    std::vector<std::size_t> parents;
};
struct ScalarRecipe
{
    double root = 0;
    std::vector<std::size_t> parents;
};

} // namespace SimpleFluid::PLIC
