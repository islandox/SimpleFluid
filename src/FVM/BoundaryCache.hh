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
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
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
 * @brief Scalar boundary cache for a mapped mesh used by stored fields.
 *
 * Stores sampled values against the native MeshHandle or specialized mesh.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
struct FieldStoredBoundaryCache
{
    using value_type = typename Pack::scalar_type;
    using mesh_type = MeshType;

    std::unordered_map<int, Arr<value_type>> value;
    SP<const mesh_type> mesh;
};

/** Scalar boundary-cache spelling for the default native mesh. */
template<TpetraTypePack Pack>
using BoundaryCache = FieldStoredBoundaryCache<Pack, MeshHandle<Pack>>;

/**
 * @brief Build a boundary-condition value cache for a mapped mesh.
 *
 * The cache retains the exact supplied native mesh identity.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
FieldStoredBoundaryCache<Pack, MeshType> cache_boundary_conditions(
    SP<const MeshType> mesh,
    const BoundaryConditionMap& boundary_conditions)
{
    using cache_type = FieldStoredBoundaryCache<Pack, MeshType>;
    return FVM::detail::cache_field_stored_boundary_conditions_impl<
        Pack, MeshType, cache_type>(std::move(mesh), boundary_conditions);
}

/** @brief Forward a mutable mapped mesh pointer to the const cache builder. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
requires (!std::is_const_v<MeshType>)
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
 * provides the concise default-mesh spelling.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
using FieldStoredBoundaryCache =
    ::SimpleFluid::FieldStoredBoundaryCache<Pack, MeshType>;

/** Scalar cache paired directly with its native mesh interface. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
using MeshBoundaryCache = FieldStoredBoundaryCache<Pack, std::remove_cv_t<MeshType>>;

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
    const MeshHandle<Pack>& mesh,
    const BoundaryCache<Pack>* cache,
    const std::string& context)
{
    detail::validate_boundary_coefficient_cache_impl<Pack>(
        mesh, cache, context);
}

/** @brief Validate a mapped-mesh sparse boundary coefficient cache. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
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
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
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
