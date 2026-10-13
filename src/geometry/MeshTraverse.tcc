/** @file MeshTraverse.tcc
 * @brief Compiled template construction and storage accounting for mesh traverse plans.
 */
#pragma once

#include "geometry/MeshTraverse.hh"
#include "geometry/details/MeshTraverseAccess.tcc"

#include <limits>
#include <stdexcept>

namespace SimpleFluid
{

template<class Mesh>
MeshTraversePlan<Mesh>::MeshTraversePlan(const Mesh& mesh, MeshTraverseEntities entities)
    : d_mesh(&mesh), d_entities(entities)
{
    const auto lease = detail::acquire_traverse_geometry(mesh);
    d_counts = detail::mesh_traverse_counts(mesh);
    d_binding = detail::mesh_traverse_binding(mesh);
    if ((d_counts[0] && d_counts[0] - 1 > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max())) ||
        (d_counts[1] && d_counts[1] - 1 > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max())))
        throw std::overflow_error("Mesh traverse local ordinal overflow.");

    detail::with_traverse_access(mesh,
        [&](const auto& access)
        {
            d_structured = access.structured_layout();
            if (d_structured)
                return;
            d_layered = access.layered_layout();
            if (d_layered)
                return;
            if (entities != MeshTraverseEntities::Faces)
                for (size_t cell = 0; cell < d_counts[0]; ++cell)
                {
                    const bool interior = !access.is_exterior_cell(cell);
                    d_cells[1].append(cell, interior, -1, false);
                    if (access.is_owned_cell(cell))
                        d_cells[0].append(cell, interior, -1, false);
                }
            if (entities != MeshTraverseEntities::Cells)
                for (size_t face = 0; face < d_counts[1]; ++face)
                {
                    const bool interior = !access.is_exterior_face(face);
                    const int batch = interior ? -1 : access.boundary_id(face);
                    d_faces[1].append(face, interior, batch, true);
                    if (access.is_owned_face(face))
                        d_faces[0].append(face, interior, batch, true);
                }
            if (entities == MeshTraverseEntities::AllWithCellFaces)
            {
                d_cell_faces.resize(d_counts[0]);
                for (size_t cell = 0; cell < d_counts[0]; ++cell)
                    access.face_ids(cell,
                        [&](auto face)
                        {
                            const bool interior = !access.is_exterior_face(face);
                            const int batch = interior ? -1 : access.boundary_id(face);
                            const auto position = d_incidence.size();
                            d_incidence.push_back({static_cast<local_ordinal_type>(face),
                                access.owner_cell(face) == static_cast<local_ordinal_type>(cell) ? 1 : -1});
                            d_cell_faces[cell].append(position, interior, batch, true);
                        });
            }
        });
}

template<class Mesh> size_t MeshTraversePlan<Mesh>::storage_bytes() const noexcept
{
    size_t bytes = sizeof(*this) + d_incidence.capacity() * sizeof(detail::MeshTraverseIncidence<local_ordinal_type>) +
                   d_cell_faces.capacity() * sizeof(detail::MeshTraverseSchedule);
    for (const auto& schedule : d_cells)
        bytes += schedule.storage_bytes();
    for (const auto& schedule : d_faces)
        bytes += schedule.storage_bytes();
    for (const auto& schedule : d_cell_faces)
        bytes += schedule.storage_bytes();
    if (d_layered)
        bytes += d_layered->storage_bytes();
    return bytes;
}

} // namespace SimpleFluid
