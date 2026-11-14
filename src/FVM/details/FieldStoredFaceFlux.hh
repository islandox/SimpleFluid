/**
 * @file FVM/details/FieldStoredFaceFlux.hh
 * @brief Internal face interpolation and flux kernels for stored fields.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/StoredMeshIdentity.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::FVM::detail
{

/** @brief Distance-weighted interpolation coefficients for a packed face. */
template<class MeshType, class LocalOrdinal>
SIMPLEFLUID_LOCAL auto stored_interior_face_linear_weights(
    const MeshType& mesh, LocalOrdinal face_lid, LocalOrdinal owner_lid, LocalOrdinal neighbor_lid)
    -> std::pair<typename MeshType::scalar_type, typename MeshType::scalar_type>;

/** @brief Tangential owner-cell velocity at a slip boundary face. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL auto stored_slip_face_velocity(const VectorCellFieldStored<Pack, MeshType>& velocity,
    typename Pack::local_ordinal_type face_lid) -> typename VectorCellFieldStored<Pack, MeshType>::value_type;

/** @brief Assemble stored face velocities with an optional boundary cache. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCache>
SIMPLEFLUID_LOCAL void assemble_stored_face_velocities(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const BoundaryCache* boundary_cache, VectorFaceFieldStored<Pack, MeshType>& face_velocity);

/** @brief Project stored face velocities onto owner-oriented area vectors. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL void stored_normal_face_fluxes(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/** @brief Assemble normal fluxes directly from stored cell velocities. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCache>
SIMPLEFLUID_LOCAL void assemble_stored_normal_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const BoundaryCache* boundary_cache, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/** @brief Form and synchronize owner-oriented mesh-relative face fluxes. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL void stored_mesh_relative_face_fluxes(const ScalarFaceFieldStored<Pack, MeshType>& absolute_fluxes,
    const ALEControlVolumeState& ale, ScalarFaceFieldStored<Pack, MeshType>& relative_fluxes);

} // namespace SimpleFluid::FVM::detail

#include "FVM/details/FieldStoredFaceFlux.ipp"
