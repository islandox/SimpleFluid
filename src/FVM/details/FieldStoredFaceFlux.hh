/**
 * @file FVM/details/FieldStoredFaceFlux.hh
 * @brief Internal face interpolation and flux kernels for stored fields.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "fields/FieldStored.hh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::FVM::detail
{

/** @brief Distance-weighted interpolation coefficients for a packed face. */
template<class MeshType, class LocalOrdinal>
auto stored_interior_face_linear_weights(const MeshType& mesh, LocalOrdinal face_lid, LocalOrdinal owner_lid,
    LocalOrdinal neighbor_lid) -> std::pair<typename MeshType::scalar_type, typename MeshType::scalar_type>
{
    using scalar_type = typename MeshType::scalar_type;
    const auto face_id = query_face_id(mesh, face_lid);
    const auto owner_id = query_cell_id(mesh, owner_lid);
    const auto neighbor_id = query_cell_id(mesh, neighbor_lid);
    const auto owner_distance = static_cast<scalar_type>(mesh.cell_to_face_distance(face_id, owner_id));
    const auto neighbor_distance = static_cast<scalar_type>(mesh.cell_to_face_distance(face_id, neighbor_id));
    const auto total_distance = owner_distance + neighbor_distance;
    if (!std::isfinite(owner_distance) || !std::isfinite(neighbor_distance) || total_distance <= scalar_type{})
    {
        return {scalar_type{0.5}, scalar_type{0.5}};
    }
    return {neighbor_distance / total_distance, owner_distance / total_distance};
}

/** @brief Validate the mesh identity shared by stored face-flux fields. */
template<class InputField, class OutputField>
void require_same_face_flux_mesh(const InputField& input, const OutputField& output, const char* operation)
{
    if (input.mesh_ptr().get() != output.mesh_ptr().get())
    {
        throw std::invalid_argument(std::string(operation) + " requires input and output fields on one mesh.");
    }
}

/** @brief Tangential owner-cell velocity at a slip boundary face. */
template<TpetraTypePack Pack, class MeshType>
auto stored_slip_face_velocity(const VectorCellFieldStored<Pack, MeshType>& velocity,
    typename Pack::local_ordinal_type face_lid) -> typename VectorCellFieldStored<Pack, MeshType>::value_type
{
    const auto& mesh = velocity.mesh();
    const auto face_id = query_face_id(mesh, face_lid);
    const auto owner_id = mesh.owner_cell(face_id);
    const auto owner_lid = packed_cell_local_id(mesh, owner_id);
    const auto cell_velocity = velocity.local_value(owner_lid);
    const auto normal = mesh.face_normal_outward(face_id, owner_id);
    return cell_velocity - normal * cell_velocity.dot(normal);
}

/** @brief Assemble stored face velocities with an optional boundary cache. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCache>
void assemble_stored_face_velocities(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const BoundaryCache* boundary_cache, VectorFaceFieldStored<Pack, MeshType>& face_velocity)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellFieldStored<Pack, MeshType>::value_type;

    require_same_face_flux_mesh(velocity, face_velocity, "face_velocities");
    if (boundary_cache != nullptr && boundary_cache->mesh.get() != velocity.mesh_ptr().get())
    {
        throw std::invalid_argument("face_velocities received a boundary cache for another mesh.");
    }

    const auto& mesh = velocity.mesh();
    face_velocity.put_value(vec_type{});
    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!face_velocity.is_owned(face_lid))
        {
            continue;
        }
        const auto face_id = query_face_id(mesh, face_lid);
        const auto owner_id = mesh.owner_cell(face_id);
        const auto owner_lid = packed_cell_local_id(mesh, owner_id);
        if (mesh.is_interior_face(face_id))
        {
            const auto neighbor_id = mesh.opposite_or_periodic_neighbor_cell(face_id, owner_id);
            const auto neighbor_lid = packed_cell_local_id(mesh, neighbor_id);
            const auto [owner_weight, neighbor_weight] =
                stored_interior_face_linear_weights(mesh, face_lid, owner_lid, neighbor_lid);
            face_velocity.set_owned_value(face_lid,
                velocity.local_value(owner_lid) * owner_weight + velocity.local_value(neighbor_lid) * neighbor_weight);
            continue;
        }
        if (boundary_cache == nullptr || !mesh.is_boundary_face(face_id))
        {
            continue;
        }
        const auto location = boundary_locations.at(face);
        if (!location.active)
        {
            continue;
        }
        const auto value_iter = boundary_cache->value.find(location.batch_id);
        if (value_iter == boundary_cache->value.end() || location.in_batch_id >= value_iter->second.size())
        {
            continue;
        }
        const auto type_iter = boundary_cache->type.find(location.batch_id);
        const auto type = type_iter == boundary_cache->type.end() ? BoundaryConditionType::Neumann : type_iter->second;
        face_velocity.set_owned_value(face_lid, type == BoundaryConditionType::Slip
                                                    ? stored_slip_face_velocity(velocity, face_lid)
                                                    : value_iter->second[location.in_batch_id]);
    }
}

/** @brief Project stored face velocities onto owner-oriented area vectors. */
template<TpetraTypePack Pack, class MeshType>
void stored_normal_face_fluxes(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    require_same_face_flux_mesh(face_velocity, fluxes, "normal_face_fluxes");
    const auto& mesh = face_velocity.mesh();
    fluxes.put_value(scalar_type{});
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!fluxes.is_owned(face_lid))
        {
            continue;
        }
        const auto face_id = query_face_id(mesh, face_lid);
        fluxes.set_owned_value(face_lid, face_velocity.value(face_lid).dot(mesh.face_normal(face_id)) *
                                             static_cast<scalar_type>(mesh.face_area(face_id)));
    }
}

/** @brief Assemble normal fluxes directly from stored cell velocities. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCache>
void assemble_stored_normal_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const BoundaryCache* boundary_cache, ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using vec_type = typename VectorCellFieldStored<Pack, MeshType>::value_type;

    require_same_face_flux_mesh(velocity, fluxes, "face_fluxes");
    if (boundary_cache != nullptr && boundary_cache->mesh.get() != velocity.mesh_ptr().get())
    {
        throw std::invalid_argument("face_fluxes received a boundary cache for another mesh.");
    }

    const auto& mesh = velocity.mesh();
    fluxes.put_value(scalar_type{});
    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!fluxes.is_owned(face_lid))
        {
            continue;
        }
        const auto face_id = query_face_id(mesh, face_lid);
        const auto owner_id = mesh.owner_cell(face_id);
        const auto owner_lid = packed_cell_local_id(mesh, owner_id);
        vec_type value{};
        if (mesh.is_interior_face(face_id))
        {
            const auto neighbor_id = mesh.opposite_or_periodic_neighbor_cell(face_id, owner_id);
            const auto neighbor_lid = packed_cell_local_id(mesh, neighbor_id);
            const auto [owner_weight, neighbor_weight] =
                stored_interior_face_linear_weights(mesh, face_lid, owner_lid, neighbor_lid);
            value =
                velocity.local_value(owner_lid) * owner_weight + velocity.local_value(neighbor_lid) * neighbor_weight;
        }
        else
        {
            if (boundary_cache == nullptr || !mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(face);
            if (!location.active)
            {
                continue;
            }
            const auto value_iter = boundary_cache->value.find(location.batch_id);
            if (value_iter == boundary_cache->value.end() || location.in_batch_id >= value_iter->second.size())
            {
                continue;
            }
            const auto type_iter = boundary_cache->type.find(location.batch_id);
            const auto type =
                type_iter == boundary_cache->type.end() ? BoundaryConditionType::Neumann : type_iter->second;
            value = type == BoundaryConditionType::Slip ? stored_slip_face_velocity(velocity, face_lid)
                                                        : value_iter->second[location.in_batch_id];
        }
        fluxes.set_owned_value(
            face_lid, value.dot(mesh.face_normal(face_id)) * static_cast<scalar_type>(mesh.face_area(face_id)));
    }
}

} // namespace SimpleFluid::FVM::detail
