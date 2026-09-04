/**
 * @file FVM/details/BoundaryCache.hh
 * @brief Internal validation and lookup kernels for boundary caches.
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "equations/BoundaryConditions.hh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::FVM::detail
{

/** @brief Construct a scalar boundary cache for a mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class CacheType>
CacheType cache_field_stored_boundary_conditions_impl(
    SP<const MeshType> mesh, const BoundaryConditionMap& boundary_conditions)
{
    using value_type = typename Pack::scalar_type;
    if (!mesh)
    {
        throw std::invalid_argument("cache_boundary_conditions requires a non-null mesh.");
    }

    CacheType cache{{}, mesh};
    for (const auto& [batch_id, boundary_batch] : mesh->boundary_batches())
    {
        const auto& boundary_name = mesh->boundary_batch_name(batch_id);
        const auto boundary_iter = boundary_conditions.find(boundary_name);
        if (boundary_iter == boundary_conditions.end() ||
            boundary_iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        Arr<value_type> batch_values(
            boundary_batch.face_lids.size(), static_cast<value_type>(boundary_iter->second.value));
        cache.value[batch_id] = std::move(batch_values);
    }

    return cache;
}

template<TpetraTypePack Pack, class MeshType, class CacheType>
void validate_boundary_coefficient_cache_impl(const MeshType& mesh, const CacheType* cache, const std::string& context)
{
    using scalar_type = typename Pack::scalar_type;

    if (cache == nullptr)
    {
        return;
    }
    if (!cache->mesh || cache->mesh.get() != &mesh)
    {
        throw std::invalid_argument(context + " received a boundary-coefficient cache on the wrong mesh.");
    }

    for (const auto& [batch_id, batch_values] : cache->value)
    {
        const auto batch_it = mesh.boundary_batches().find(batch_id);
        if (batch_it == mesh.boundary_batches().end() || batch_values.size() != batch_it->second.face_lids.size())
        {
            throw std::invalid_argument(context + " received an invalid boundary-coefficient batch.");
        }
        for (const auto value : batch_values)
        {
            if (!std::isfinite(value) || value < scalar_type{})
            {
                throw std::invalid_argument(context + " requires finite non-negative boundary "
                                                      "coefficients.");
            }
        }
    }
}

template<class CacheType, class Scalar>
Scalar boundary_coefficient_impl(const CacheType* cache, int batch_id, size_t in_batch_id, Scalar owner_cell_value)
{
    if (cache == nullptr)
    {
        return owner_cell_value;
    }
    const auto iter = cache->value.find(batch_id);
    return iter == cache->value.end() ? owner_cell_value : iter->second.at(in_batch_id);
}

} // namespace SimpleFluid::FVM::detail
