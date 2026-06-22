/**
 * @file BoundaryCache.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Pre-computed boundary condition values cached per batch for FVM operator assembly.
 * @version 0.1
 * @date 2026-05-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"

#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Pre-computed boundary condition values cached per batch for FVM operator assembly.
 *
 * @tparam Pack Tpetra type pack providing scalar and mesh types.
 */
template<TpetraTypePack Pack>
struct BoundaryCache
{
    using value_type = typename Pack::scalar_type;

    std::unordered_map<int, Arr<value_type>> value;
    SP<const Mesh<Pack>> mesh;
};

/**
 * @brief Build a boundary-condition value cache from a shared mesh pointer
 *        and a boundary-condition map.
 *
 * Caches Dirichlet boundary values per batch; other condition types are
 * ignored.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param boundary_conditions Map from batch name to boundary condition.
 * @return BoundaryCache populated with the Dirichlet values of owned
 *         boundary faces.
 */
template<TpetraTypePack Pack>
BoundaryCache<Pack> cache_boundary_conditions(
    SP<const Mesh<Pack>> mesh,
    const BoundaryConditionMap& boundary_conditions)
{
    using value_type = typename Pack::scalar_type;
    if (!mesh)
    {
        throw std::invalid_argument(
            "cache_boundary_conditions requires a non-null mesh.");
    }
    BoundaryCache<Pack> cache{ {}, mesh };

    for (const auto& [batch_id, boundary_batch] : mesh->boundary_batches())
    {
        const auto& boundary_name = mesh->boundary_batch_name(batch_id);
        const auto& bc_it = boundary_conditions.find(boundary_name);
        if (bc_it == boundary_conditions.end())
        {
            continue;
        }

        const auto& bc = bc_it->second;
        if (bc.type == BoundaryConditionType::Dirichlet)
        {
            Arr<value_type> batch_values(boundary_batch.face_lids.size(),
                                        static_cast<value_type>(bc.value));
            cache.value[batch_id] = std::move(batch_values);
        }
    }

    return cache;
}

}
