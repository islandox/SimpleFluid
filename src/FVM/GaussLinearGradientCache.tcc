/** @file GaussLinearGradientCache.tcc @brief Template implementations for GaussLinearGradientCache. */
#pragma once

#include "FVM/GaussLinearGradientCache.hh"

#include <array>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
GaussLinearGradientCache<Pack, MeshType>::GaussLinearGradientCache(SP<const mesh_type> mesh)
    : d_mesh(std::move(mesh))
{
    if (!d_mesh)
        throw std::invalid_argument("GaussLinearGradientCache requires a non-null mesh.");
    refresh();
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::require_mesh(const mesh_type& mesh) const
{
    if (&mesh != d_mesh.get())
        throw std::invalid_argument("Gauss-linear gradient cache belongs to another mesh.");
    if (mesh_geometry_epoch(mesh) != d_geometry_epoch)
        throw std::invalid_argument("Gauss-linear gradient cache is stale for the mesh geometry epoch.");
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::refresh()
{
    const auto& mesh = *d_mesh;
    const auto execution = acquire_mesh_execution(mesh);
    const auto locations = detail::boundary_face_locations(mesh);
    Geometry geometry;
    const auto count = mesh.num_owned_cells();
    geometry.offsets.reserve(count + 1);
    geometry.volumes.reserve(count);
    geometry.faces.reserve(count * 6);
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = detail::query_cell_id(mesh, cell_lid);
        geometry.offsets.push_back(geometry.faces.size());
        geometry.volumes.push_back(mesh.cell_volume(cell_id));
        for (const auto face_id : mesh.faces(cell_id))
        {
            FaceGeometry face;
            face.face_lid = static_cast<local_ordinal_type>(detail::packed_face_local_id(mesh, face_id));
            face.area = mesh.face_area_vector_outward(face_id, cell_id);
            face.interior = mesh.is_interior_face(face_id);
            if (face.interior)
            {
                const auto other_id = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                face.other_lid = static_cast<local_ordinal_type>(detail::packed_cell_local_id(mesh, other_id));
                face.cell_distance = mesh.cell_to_face_distance(face_id, cell_id);
                face.other_distance = mesh.cell_to_face_distance(face_id, other_id);
            }
            else if (mesh.is_boundary_face(face_id))
            {
                face.boundary = locations.at(static_cast<size_t>(face.face_lid));
                if (face.boundary.active)
                    face.normal_distance = detail::boundary_normal_distance(mesh, face.face_lid, cell_lid);
            }
            geometry.faces.push_back(face);
        }
    }
    geometry.offsets.push_back(geometry.faces.size());
    d_geometry = std::move(geometry);
    d_geometry_epoch = mesh_geometry_epoch(mesh);
}

} // namespace SimpleFluid::FVM
