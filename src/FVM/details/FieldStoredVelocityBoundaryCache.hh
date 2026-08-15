/**
 * @file FVM/details/FieldStoredVelocityBoundaryCache.hh
 * @brief Internal construction kernel for stored velocity boundary caches.
 */

#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "dataclass/TpetraTypes.hh"
#include "equations/BoundaryConditions.hh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace SimpleFluid::FVM::detail
{

/** @brief Construct a velocity boundary cache for a mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class CacheType>
CacheType cache_field_stored_velocity_boundary_conditions_impl(
    SP<const MeshType> mesh, const BoundaryConditionSet& boundary_conditions)
{
    if (!mesh)
    {
        throw std::invalid_argument("cache_velocity_boundary_conditions requires a non-null mesh.");
    }

    CacheType cache(mesh);
    for (const auto& [name, condition] : boundary_conditions.velocity)
    {
        cache.type_by_name[name] = condition.type;
    }

    std::unordered_map<int, size_t> local_batch_sizes;
    for (const auto& location : boundary_face_locations(*mesh))
    {
        if (location.active)
        {
            auto& size = local_batch_sizes[location.batch_id];
            size = std::max(size, location.in_batch_id + 1);
        }
    }

    for (const auto& [batch_id, batch_size] : local_batch_sizes)
    {
        typename CacheType::vec_type prescribed_value{};
        auto boundary_type = BoundaryConditionType::Neumann;
        const auto condition_iter = boundary_conditions.velocity.find(mesh->boundary_batch_name(batch_id));
        if (condition_iter != boundary_conditions.velocity.end())
        {
            boundary_type = condition_iter->second.type;
            if (boundary_type == BoundaryConditionType::NoSlip)
            {
                prescribed_value = {};
            }
            else if (boundary_type == BoundaryConditionType::Dirichlet)
            {
                prescribed_value = condition_iter->second.value;
            }
            else if (boundary_type == BoundaryConditionType::Neumann &&
                     (condition_iter->second.value.x != 0.0 || condition_iter->second.value.y != 0.0 ||
                         condition_iter->second.value.z != 0.0))
            {
                throw std::invalid_argument("Cache-based incompressible velocity transport supports "
                                            "homogeneous Neumann outlet conditions only.");
            }
        }

        cache.value[batch_id] = Arr<typename CacheType::vec_type>(batch_size, prescribed_value);
        cache.type[batch_id] = boundary_type;
    }

    return cache;
}

} // namespace SimpleFluid::FVM::detail
