/**
 * @file BoundaryCache.hh
 * @author your name (you@domain.com)
 * @brief 
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

namespace SimpleFluid
{

template<TpetraTypePack Pack>
struct BoundaryCache
{
    using value_type = typename Pack::scalar_type;

    std::unordered_map<int, Arr<value_type>> value;
    SP<const Mesh<Pack>> mesh;
};

template<TpetraTypePack Pack>
BoundaryCache<Pack> cache_boundary_conditions(
    SP<const Mesh<Pack>> mesh,
    const BoundaryConditionMap& boundary_conditions)
{
    using value_type = typename Pack::scalar_type;
    BoundaryCache<Pack> cache{ {}, mesh };

    for (const auto& [patch_id, boundary_patch] : mesh->boundary_patches())
    {
        const auto& boundary_name = mesh->boundary_patch_name(patch_id);
        const auto& bc_it = boundary_conditions.find(boundary_name);
        if (bc_it == boundary_conditions.end())
        {
            continue;
        }

        const auto& bc = bc_it->second;
        if (bc.type == BoundaryConditionType::Dirichlet)
        {
            Arr<value_type> patch_values(boundary_patch.face_lids.size(),
                                        static_cast<value_type>(bc.value));
            cache.value[patch_id] = std::move(patch_values);
        }
    }

    return cache;
}

}