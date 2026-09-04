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
#include "FVM/details/BoundaryCache.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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
 * @brief Scalar boundary cache for a mapped mesh used by stored fields.
 *
 * This parallel type keeps the established BoundaryCache<Pack> layout and
 * symbol spelling unchanged while supporting MeshHandle and specialized mesh
 * implementations directly.
 */
template<TpetraTypePack Pack, class MeshType>
struct FieldStoredBoundaryCache
{
    using value_type = typename Pack::scalar_type;
    using mesh_type = MeshType;

    std::unordered_map<int, Arr<value_type>> value;
    SP<const mesh_type> mesh;
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

/**
 * @brief Build a boundary-condition value cache for a mapped mesh.
 *
 * The overload is constrained away from the legacy Mesh specialization so
 * existing calls keep returning the ABI-stable BoundaryCache<Pack> type.
 */
template<TpetraTypePack Pack, class MeshType>
requires (!std::is_same_v<std::remove_cv_t<MeshType>, Mesh<Pack>>)
FieldStoredBoundaryCache<Pack, MeshType> cache_boundary_conditions(
    SP<const MeshType> mesh,
    const BoundaryConditionMap& boundary_conditions)
{
    using cache_type = FieldStoredBoundaryCache<Pack, MeshType>;
    return FVM::detail::cache_field_stored_boundary_conditions_impl<
        Pack, MeshType, cache_type>(std::move(mesh), boundary_conditions);
}

/** @brief Forward a mutable mapped mesh pointer to the const cache builder. */
template<TpetraTypePack Pack, class MeshType>
requires (!std::is_const_v<MeshType>
          && !std::is_same_v<MeshType, Mesh<Pack>>)
FieldStoredBoundaryCache<Pack, MeshType> cache_boundary_conditions(
    SP<MeshType> mesh,
    const BoundaryConditionMap& boundary_conditions)
{
    return cache_boundary_conditions<Pack, MeshType>(
        SP<const MeshType>(std::move(mesh)), boundary_conditions);
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
 * @brief Explicit name for a scalar cache paired with mesh-aware stored fields.
 *
 * This mirrors FieldStoredVelocityBoundaryCache while BoundaryCache<Pack>
 * remains the concise legacy spelling.
 */
template<TpetraTypePack Pack, class MeshType>
using FieldStoredBoundaryCache =
    ::SimpleFluid::FieldStoredBoundaryCache<Pack, MeshType>;

/**
 * @brief Select the ABI-stable legacy cache or the mapped-mesh cache.
 *
 * Solver templates can use this alias without converting a MeshHandle to a
 * legacy Mesh.  The Mesh<Pack> specialization remains exactly
 * FVM::BoundaryCache<Pack>.
 */
template<TpetraTypePack Pack, class MeshType>
using MeshBoundaryCache = std::conditional_t<
    std::is_same_v<std::remove_cv_t<MeshType>, Mesh<Pack>>,
    BoundaryCache<Pack>,
    FieldStoredBoundaryCache<Pack, std::remove_cv_t<MeshType>>>;

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
    detail::validate_boundary_coefficient_cache_impl<Pack>(
        mesh, cache, context);
}

/** @brief Validate a mapped-mesh sparse boundary coefficient cache. */
template<TpetraTypePack Pack, class MeshType>
void validate_boundary_coefficient_cache(
    const MeshType& mesh,
    const FieldStoredBoundaryCache<Pack, MeshType>* cache,
    const std::string& context)
{
    detail::validate_boundary_coefficient_cache_impl<Pack>(
        mesh, cache, context);
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
    return detail::boundary_coefficient_impl(
        cache, batch_id, in_batch_id, owner_cell_value);
}

/** @brief Return a mapped-mesh cached coefficient or owner-cell fallback. */
template<TpetraTypePack Pack, class MeshType>
typename Pack::scalar_type boundary_coefficient(
    const FieldStoredBoundaryCache<Pack, MeshType>* cache,
    int batch_id,
    size_t in_batch_id,
    typename Pack::scalar_type owner_cell_value)
{
    return detail::boundary_coefficient_impl(
        cache, batch_id, in_batch_id, owner_cell_value);
}

} // namespace FVM

}
