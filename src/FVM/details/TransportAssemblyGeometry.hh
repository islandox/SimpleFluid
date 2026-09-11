/**
 * @file FVM/details/TransportAssemblyGeometry.hh
 * @brief Ordered host topology and volumes for repeated transport assembly.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/ResolvedTransportGeometry.hh"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <type_traits>
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
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            mesh.visit_owned_region_cells([&](local_ordinal_type, auto canonical_cell,
                                             auto native_cell, const auto& region)
            {
                result.volumes.push_back(region.cell_volume(native_cell));
                region.visit_cell_faces(native_cell, [&](const auto& resolved)
                {
                    typename geometry_type::Face face;
                    face.face_lid = mesh.region_face_local_id(resolved.face);
                    face.interior = resolved.interior();
                    face.owned_orientation = resolved.owner == canonical_cell;
                    if (face.face_lid == mesh.invalid_local_id())
                        throw std::logic_error("Region transport requires the incident face in the existing halo.");
                    face.boundary = result.physical_boundary_faces[face.face_lid];
                    if (face.interior)
                    {
                        face.other = mesh.region_cell_local_id(resolved.opposite_cell(canonical_cell));
                        if (face.other == mesh.invalid_local_id())
                            throw std::logic_error("Region transport requires the adjacent cell in the existing halo.");
                    }
                    result.faces.push_back(face);
                });
                result.face_offsets.push_back(result.faces.size());
            });
            return result;
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

/**
 * @brief Visit the retained row topology with operation-local region metrics.
 *
 * Region traversal and the epoch-owned topology cache use the same native
 * incidence order, including periodic and coarse/fine faces. Generic meshes
 * keep their existing cached traversal. No face metrics survive the callback.
 */
template<class MeshType, class Visitor>
void visit_transport_assembly_rows(const MeshType& mesh,
    const TransportAssemblyGeometry<MeshType>& geometry, Visitor&& visitor)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            mesh.visit_owned_region_cells([&](local_ordinal_type cell, auto canonical_cell,
                                             auto native_cell, const auto& region)
            {
                visitor(cell, geometry.volumes[cell], [&](auto&& visit_face)
                {
                    auto index = geometry.face_offsets[cell];
                    const auto end = geometry.face_offsets[cell + 1];
                    region.visit_cell_faces(native_cell, [&](const auto& resolved)
                    {
                        if (index == end)
                            throw std::logic_error("Region transport geometry lost its cached incidence order.");
                        const auto& face = geometry.faces[index++];
                        const ResolvedTransportGeometry<MeshType, std::remove_cvref_t<decltype(resolved)>>
                            metrics(resolved, cell, canonical_cell);
                        visit_face(face, metrics);
                    });
                    if (index != end)
                        throw std::logic_error("Region transport geometry lost its cached incidence order.");
                });
            });
            return;
        }
    }
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        visitor(cell, geometry.volumes[owned], [&](auto&& visit_face)
        {
            for (size_t index = geometry.face_offsets[owned]; index < geometry.face_offsets[owned + 1]; ++index)
                visit_face(geometry.faces[index], mesh);
        });
    }
}

} // namespace SimpleFluid::FVM::detail
