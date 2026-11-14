/**
 * @file FVM/details/FieldStoredVelocityBoundaryCache.hh
 * @brief Internal construction kernel for stored velocity boundary caches.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "dataclass/TpetraTypes.hh"
#include "equations/BoundaryConditions.hh"

namespace SimpleFluid::FVM::detail
{

/** @brief Construct a velocity boundary cache for a mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class CacheType>
SIMPLEFLUID_LOCAL CacheType cache_field_stored_velocity_boundary_conditions_impl(
    SP<const MeshType> mesh, const BoundaryConditionSet& boundary_conditions);

} // namespace SimpleFluid::FVM::detail
