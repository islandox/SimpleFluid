/**
 * @file FVM/details/FieldStoredPressureWeightedFaceFlux.hh
 * @brief Internal Rhie-Chow face-flux kernels for mesh-aware stored fields.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/CellGradientScheme.hh"
#include "equations/BoundaryConditions.hh"
#include "fields/FieldStored.hh"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** @brief Ordered owned-face topology and epoch-dependent Rhie--Chow metrics. */
template<TpetraTypePack Pack> struct StoredPressureFaceGeometry
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;

    local_ordinal_type face{}, owner{}, neighbor{};
    bool interior = false;
    bool boundary = false;
    bool owned_owner = false;
    vec_type normal{}, owner_normal{}, area_vector{}, center_delta{};
    scalar_type area{}, owner_weight{}, neighbor_weight{};
    // Store alpha*d2 so all consumers recover the same pressure coefficient.
    scalar_type center_projection{}, distance_squared{}, boundary_diffusion{};
};

/** @brief Rebuild numeric geometry without retaining field or boundary values. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL auto stored_pressure_face_geometry(const MeshType& mesh)
    -> std::vector<StoredPressureFaceGeometry<Pack>>;

/** @brief Validate pressure/velocity boundary compatibility for any cache. */
template<class VelocityBoundaryCacheType>
SIMPLEFLUID_LOCAL void validate_pressure_velocity_boundary_compatibility(
    const VelocityBoundaryCacheType& velocity_boundary_cache, const BoundaryConditionMap& pressure_boundary_conditions);

/**
 * @brief Compute Rhie-Chow face fluxes on a mapped mesh using stored fields.
 *
 * Interior faces and physical boundary batches are evaluated separately.
 * Output storage must not alias input storage across these independent passes.
 *
 * @tparam Pack Tpetra type pack used by the fields.
 * @tparam MeshType Runtime or statically dispatched mapped mesh.
 * @tparam VelocityBoundaryCacheType Stored-field velocity boundary cache.
 * @tparam Workspace Stored-field pressure-flux workspace.
 */
template<TpetraTypePack Pack, class MeshType, class VelocityBoundaryCacheType, class Workspace>
SIMPLEFLUID_LOCAL void pressure_weighted_stored_face_fluxes_impl(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCacheType& boundary_cache, const BoundaryConditionMap* pressure_boundary_conditions,
    Workspace& workspace, VectorCellFieldStored<Pack, MeshType>* precomputed_pressure_gradient,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes,
    CellGradientScheme gradient_scheme = CellGradientScheme::LeastSquares,
    ScalarFaceFieldStored<Pack, MeshType>* velocity_component = nullptr,
    ScalarFaceFieldStored<Pack, MeshType>* pressure_component = nullptr);

} // namespace SimpleFluid::FVM::detail

#include "FVM/details/FieldStoredPressureWeightedFaceFlux.ipp"
