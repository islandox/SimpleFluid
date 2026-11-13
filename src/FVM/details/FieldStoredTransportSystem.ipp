/** @file FVM/details/FieldStoredTransportSystem.ipp
 * @brief Read-only transport queries for caller-defined mesh types.
 */
#pragma once

#include "geometry/GeometryEpoch.hh"

namespace SimpleFluid::FVM::detail
{

template<class MeshType> bool stored_transport_has_non_orthogonal_faces(const MeshType& mesh)
{
    const auto execution = acquire_mesh_execution(mesh);
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }
            const auto area = mesh.face_area_vector_outward(face_lid, cell_lid);
            if (direction.dot(direction) <= 0 || area.dot(area) <= 0 || area.x * direction.y != area.y * direction.x ||
                area.x * direction.z != area.z * direction.x || area.y * direction.z != area.z * direction.y)
            {
                return true;
            }
        }
    }
    return false;
}

} // namespace SimpleFluid::FVM::detail
