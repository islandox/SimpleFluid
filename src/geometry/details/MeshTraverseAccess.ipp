/** @file MeshTraverseAccess.ipp
 * @brief Inline accessors and numerical loops for MeshTraverseAccess.
 */
#pragma once

#include "geometry/details/MeshTraverseAccess.hh"

#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{

template<class PublicMesh, class GeometryMesh>
inline TraverseAccess<PublicMesh, GeometryMesh>::TraverseAccess(const PublicMesh& mesh, const GeometryMesh& geometry)
    : d_mesh(mesh), d_geometry(geometry), d_num_cells(mesh.num_local_cells()), d_num_faces(mesh.num_faces())
{
}

template<class PublicMesh, class GeometryMesh>
template<class Visitor>
inline void TraverseAccess<PublicMesh, GeometryMesh>::face_ids(size_t cell, Visitor&& visitor) const
{
    for (const auto face : d_geometry.faces(cell_id(cell)))
    {
        const auto local_face = field_face(face);
        if (local_face != invalid_local_id())
            visitor(local_face);
    }
}

template<class PublicMesh, class GeometryMesh>
inline real_t TraverseAccess<PublicMesh, GeometryMesh>::cell_to_face_distance(size_t face, size_t cell) const
{
    if constexpr (traverse_is_handle<PublicMesh>)
    {
        // Match MeshHandle's public distance arithmetic, including the
        // wrapped legacy and Cartesian periodic-face distances.
        if constexpr (traverse_is_stk_adapter<GeometryMesh>)
            return d_geometry.mesh().cell_to_face_distance(face_id(face), cell_id(cell));
        else if constexpr (std::is_same_v<GeometryMesh, Meshes::OrthogonalCartesian3D>)
            return d_geometry.cell_to_face_distance(face_id(face), cell_id(cell));
        else
            return face_center_vector(face, cell).norm();
    }
    else if constexpr (requires { typename PublicMesh::partitioned_mesh_base_tag; })
        return (face_centroid(face) - cell_centroid(cell)).norm();
    else
        return d_geometry.cell_to_face_distance(face_id(face), cell_id(cell));
}

template<class PublicMesh, class GeometryMesh>
inline size_t TraverseAccess<PublicMesh, GeometryMesh>::geometry_cell(size_t cell) const
{
    if (cell >= d_num_cells)
        throw std::out_of_range("Mesh traverse cell ordinal is out of range.");
    if constexpr (traverse_is_handle<PublicMesh>)
    {
        if (!d_mesh.d_cell_geometry_lids.empty())
            return static_cast<size_t>(d_mesh.d_cell_geometry_lids[cell]);
        if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh> || traverse_is_stk_adapter<GeometryMesh>)
            return cell;
        if (d_mesh.d_serial_identity)
            return cell;
        if (d_mesh.d_semi_structured_topology)
            return static_cast<size_t>(d_mesh.geometry_cell_lid(local(cell)));
        return static_cast<size_t>(d_mesh.d_map_indexing ? d_mesh.d_overlap_cell_map->getGlobalElement(local(cell))
                                                         : d_mesh.d_indexer.cell_global_id(local(cell)));
    }
    else
        return cell;
}

template<class PublicMesh, class GeometryMesh>
inline size_t TraverseAccess<PublicMesh, GeometryMesh>::geometry_face(size_t face) const
{
    if (face >= d_num_faces)
        throw std::out_of_range("Mesh traverse face ordinal is out of range.");
    if constexpr (traverse_is_handle<PublicMesh>)
    {
        if constexpr (traverse_is_stk_adapter<GeometryMesh>)
            return d_mesh.d_legacy_face_geometry_lids.empty()
                       ? face
                       : static_cast<size_t>(d_mesh.d_legacy_face_geometry_lids[face]);
        if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>)
            return face;
        if (d_mesh.d_serial_identity)
            return face;
        if (d_mesh.d_semi_structured_topology)
            return static_cast<size_t>(d_mesh.geometry_face_lid(local(face)));
        return static_cast<size_t>(d_mesh.d_map_indexing ? d_mesh.d_overlap_face_map->getGlobalElement(local(face))
                                                         : d_mesh.d_indexer.face_global_id(local(face)));
    }
    else
        return face;
}

template<class PublicMesh, class GeometryMesh>
template<class Cell>
inline typename TraverseAccess<PublicMesh, GeometryMesh>::local_ordinal_type
TraverseAccess<PublicMesh, GeometryMesh>::field_cell(Cell cell) const
{
    if (invalid_cell(cell))
        return invalid_local_id();
    const auto ordinal = [&]
    {
        if constexpr (requires { d_geometry.cell_local_id(cell); })
            return static_cast<size_t>(d_geometry.cell_local_id(cell));
        else
            return static_cast<size_t>(cell);
    }();
    if constexpr (traverse_is_handle<PublicMesh>)
    {
        if (!d_mesh.d_cell_local_lids_by_geometry.empty())
            return ordinal < d_mesh.d_cell_local_lids_by_geometry.size() ? d_mesh.d_cell_local_lids_by_geometry[ordinal]
                                                                         : invalid_local_id();
        if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh> || traverse_is_stk_adapter<GeometryMesh>)
            return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
        if (d_mesh.d_serial_identity)
            return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
        using global_ordinal_type = typename PublicMesh::global_ordinal_type;
        if (d_mesh.d_semi_structured_topology)
            return d_mesh.geometry_to_local_cell(ordinal);
        return d_mesh.d_map_indexing
                   ? d_mesh.d_overlap_cell_map->getLocalElement(static_cast<global_ordinal_type>(ordinal))
                   : d_mesh.d_indexer.cell_local_id(static_cast<global_ordinal_type>(ordinal));
    }
    else
        return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
}

template<class PublicMesh, class GeometryMesh>
template<class Face>
inline typename TraverseAccess<PublicMesh, GeometryMesh>::local_ordinal_type
TraverseAccess<PublicMesh, GeometryMesh>::field_face(Face face) const
{
    const auto ordinal = [&]
    {
        if constexpr (requires { d_geometry.face_local_id(face); })
            return static_cast<size_t>(d_geometry.face_local_id(face));
        else
            return static_cast<size_t>(face);
    }();
    if constexpr (traverse_is_handle<PublicMesh>)
    {
        if constexpr (traverse_is_stk_adapter<GeometryMesh>)
        {
            if (ordinal >= num_faces())
                return invalid_local_id();
            return d_mesh.d_legacy_face_local_lids.empty() ? local(ordinal) : d_mesh.d_legacy_face_local_lids[ordinal];
        }
        if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>)
            return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
        if (d_mesh.d_serial_identity)
            return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
        using global_ordinal_type = typename PublicMesh::global_ordinal_type;
        if (d_mesh.d_semi_structured_topology)
            return d_mesh.geometry_to_local_face(ordinal);
        return d_mesh.d_map_indexing
                   ? d_mesh.d_overlap_face_map->getLocalElement(static_cast<global_ordinal_type>(ordinal))
                   : d_mesh.d_indexer.face_local_id(static_cast<global_ordinal_type>(ordinal));
    }
    else
        return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
}

template<class Access>
inline StructuredTraverseAccess<Access>::StructuredTraverseAccess(const Access& access) : Access(access)
{
}

template<class Access>
inline LayeredTraverseAccess<Access>::LayeredTraverseAccess(const Access& access) : Access(access)
{
}

template<class Mesh, class Visitor> decltype(auto) with_traverse_access(const Mesh& mesh, Visitor&& visitor)
{
    const auto invoke = [&](const auto& geometry) -> decltype(auto)
    {
        // Native value meshes also reject supported mutation during callbacks.
        // Composite execution additionally validates and pins each provider.
        [[maybe_unused]] const auto execution = [&]
        {
            if constexpr (requires { geometry.acquire_execution_view(); })
                return geometry.acquire_execution_view();
            else if constexpr (requires { geometry.acquire_geometry_read(); })
                return geometry.acquire_geometry_read();
            else
            {
                struct LegacyExecution
                {
                };
                return LegacyExecution{};
            }
        }();
        const TraverseAccess<Mesh, std::remove_cvref_t<decltype(geometry)>> access(mesh, geometry);
        return std::forward<Visitor>(visitor)(access);
    };
    if constexpr (traverse_is_handle<Mesh>)
        return mesh.visit(invoke);
    else if constexpr (requires { typename Mesh::partitioned_mesh_base_tag; })
        return invoke(mesh.mesh());
    else
        return invoke(mesh);
}

} // namespace SimpleFluid::detail
