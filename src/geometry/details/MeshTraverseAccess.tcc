/** @file MeshTraverseAccess.tcc
 * @brief Compiled canonical-layout detection for mesh traverse access.
 */
#pragma once

#include "geometry/details/MeshTraverseAccess.hh"

namespace SimpleFluid::detail
{

template<class PublicMesh, class GeometryMesh>
std::optional<StructuredTraverseLayout> TraverseAccess<PublicMesh, GeometryMesh>::structured_layout() const
{
    if constexpr (has_structured_traverse_layout<GeometryMesh>)
    {
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            // Reordering and partition maps need the generic field-local
            // schedules even when their geometry provider is structured.
            if (!d_mesh.d_serial_identity || !d_mesh.d_cell_geometry_lids.empty())
                return std::nullopt;
        }
        else if constexpr (!std::is_same_v<PublicMesh, GeometryMesh>)
            return std::nullopt;
        const auto& indexer = d_geometry.indexer();
        if (indexer.num_cells_per_dim[0] && indexer.num_cells_per_dim[1] && indexer.num_cells_per_dim[2] &&
            d_num_cells == indexer.total_cells() && d_num_faces == indexer.total_faces() &&
            d_mesh.num_owned_cells() == d_num_cells && d_mesh.num_owned_faces() == d_num_faces)
            return StructuredTraverseLayout(indexer);
    }
    return std::nullopt;
}

template<class PublicMesh, class GeometryMesh>
std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<PublicMesh, GeometryMesh>::layered_layout() const
{
    if constexpr (has_layered_traverse_layout<GeometryMesh>)
    {
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            if (!d_mesh.d_serial_identity || !d_mesh.d_cell_geometry_lids.empty())
                return {};
        }
        else if constexpr (!std::is_same_v<PublicMesh, GeometryMesh>)
            return {};
        const auto& indexer = d_geometry.indexer();
        if (indexer.num_cells_per_layer && indexer.num_layers && d_num_cells == indexer.total_cells() &&
            d_num_faces == indexer.total_faces() && d_mesh.num_owned_cells() == d_num_cells &&
            d_mesh.num_owned_faces() == d_num_faces)
            return std::make_shared<const LayeredTraverseLayout>(d_geometry.topology());
    }
    return {};
}

} // namespace SimpleFluid::detail
