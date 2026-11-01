/** @file PolyhedralInterface.hh
 * @brief Represented physical PLIC interface geometry.
 */
#pragma once
#include "geometry/PLIC/ConvexCellTypes.hh"

namespace SimpleFluid::PLIC
{
/** Mode-neutral represented plane and pure-phase state. Exact reconstruction
 * also publishes this binary64 plane; arithmetic evidence belongs to its result. */
struct PolyhedralInterface
{
    GeometryHalfspace gas;
    bool pure_gas = false, pure_carrier = false;
};
} // namespace SimpleFluid::PLIC
