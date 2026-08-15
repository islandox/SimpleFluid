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

#include <cmath>
#include <stdexcept>
#include <string>

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

namespace FVM
{

/**
 * @brief Backward-compatible FVM-qualified name for a scalar boundary cache.
 *
 * BoundaryCache predates the FVM namespace used by the operator assembly
 * functions.  Keep the original SimpleFluid::BoundaryCache name available
 * while allowing transport APIs to spell the ownership of the cache clearly.
 */
template<TpetraTypePack Pack>
using BoundaryCache = ::SimpleFluid::BoundaryCache<Pack>;

/**
 * @brief Validate an optional sparse boundary transport-coefficient cache.
 *
 * A cache may omit boundary batches; omitted batches retain the owner-cell
 * coefficient.  Every supplied batch must belong to @p mesh, contain one
 * value per batch face, and contain only finite non-negative coefficients. A
 * null @p cache is accepted; @p context prefixes validation errors.
 *
 * @throws std::invalid_argument if the cache references another mesh or
 *         contains an invalid batch, size, or coefficient.
 */
template<TpetraTypePack Pack>
void validate_boundary_coefficient_cache(
    const Mesh<Pack>& mesh,
    const BoundaryCache<Pack>* cache,
    const std::string& context)
{
    using scalar_type = typename Pack::scalar_type;

    if (cache == nullptr)
    {
        return;
    }
    if (!cache->mesh || cache->mesh.get() != &mesh)
    {
        throw std::invalid_argument(
            context + " received a boundary-coefficient cache on the wrong mesh.");
    }

    for (const auto& [batch_id, batch_values] : cache->value)
    {
        const auto batch_it = mesh.boundary_batches().find(batch_id);
        if (batch_it == mesh.boundary_batches().end()
            || batch_values.size() != batch_it->second.face_lids.size())
        {
            throw std::invalid_argument(
                context + " received an invalid boundary-coefficient batch.");
        }
        for (const auto value : batch_values)
        {
            if (!std::isfinite(value) || value < scalar_type{})
            {
                throw std::invalid_argument(
                    context + " requires finite non-negative boundary coefficients.");
            }
        }
    }
}

/**
 * @brief Return a cached boundary coefficient or an owner-cell fallback.
 */
template<TpetraTypePack Pack>
typename Pack::scalar_type boundary_coefficient(
    const BoundaryCache<Pack>* cache,
    int batch_id,
    size_t in_batch_id,
    typename Pack::scalar_type owner_cell_value)
{
    if (cache == nullptr)
    {
        return owner_cell_value;
    }
    const auto iter = cache->value.find(batch_id);
    return iter == cache->value.end()
        ? owner_cell_value
        : iter->second.at(in_batch_id);
}

} // namespace FVM

}
