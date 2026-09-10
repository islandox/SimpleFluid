/**
 * @file FVM/details/TransportAssemblyGeometry.hh
 * @brief Ordered host topology and volumes for repeated transport assembly.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** Geometry-epoch-owned data; face entries retain the mesh's row traversal order. */
template<class MeshType> struct TransportAssemblyGeometry
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    struct Face
    {
        local_ordinal_type face_lid{};
        local_ordinal_type other{};
        bool interior = false;
        bool boundary = false;
        bool owned_orientation = false;
    };

    std::vector<typename MeshType::scalar_type> volumes;
    std::vector<size_t> face_offsets;
    std::vector<Face> faces;
    std::vector<bool> physical_boundary_faces;
    std::map<std::pair<int, size_t>, size_t> boundary_indices;
};

template<class MeshType>
TransportAssemblyGeometry<MeshType> transport_assembly_geometry(const MeshType& mesh)
{
    const auto execution = acquire_mesh_execution(mesh);
    using geometry_type = TransportAssemblyGeometry<MeshType>;
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    geometry_type result;
    result.volumes.reserve(mesh.num_owned_cells());
    result.face_offsets.reserve(mesh.num_owned_cells() + 1);
    result.face_offsets.push_back(0);
    result.faces.reserve(mesh.num_owned_cells() * 6);
    result.physical_boundary_faces = physical_boundary_face_mask(mesh);
    const auto locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < locations.size(); ++face)
    {
        const auto& location = locations[face];
        if (location.active && result.physical_boundary_faces[face])
        {
            result.boundary_indices.emplace(std::pair{location.batch_id, location.in_batch_id}, face);
        }
    }
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        result.volumes.push_back(mesh.cell_volume(cell_lid));
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename geometry_type::Face face;
            face.face_lid = face_lid;
            face.interior = mesh.is_interior_face(face_lid);
            face.boundary = mesh.is_boundary_face(face_lid);
            face.owned_orientation = mesh.owner_cell(face_lid) == cell_lid;
            if (face.interior)
            {
                face.other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            }
            result.faces.push_back(face);
        }
        result.face_offsets.push_back(result.faces.size());
    }
    return result;
}

} // namespace SimpleFluid::FVM::detail
