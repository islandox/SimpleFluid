/** @file FieldStoredFaceFlux.ipp
 * @brief Caller-dependent stored-field queries and kernels.
 */
#pragma once

namespace SimpleFluid::FVM::detail
{

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

template<TpetraTypePack Pack, class MeshType>
auto stored_slip_face_velocity(const VectorCellFieldStored<Pack, MeshType>& velocity,
    typename Pack::local_ordinal_type face_lid) -> typename VectorCellFieldStored<Pack, MeshType>::value_type
{
    const auto& mesh = velocity.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    const auto face_id = query_face_id(mesh, face_lid);
    const auto owner_id = mesh.owner_cell(face_id);
    const auto owner_lid = packed_cell_local_id(mesh, owner_id);
    const auto cell_velocity = velocity.local_value(owner_lid);
    const auto normal = mesh.face_normal_outward(face_id, owner_id);
    return cell_velocity - normal * cell_velocity.dot(normal);
}

} // namespace SimpleFluid::FVM::detail
