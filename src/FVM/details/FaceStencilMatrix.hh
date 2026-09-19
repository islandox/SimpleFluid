/**
 * @file FaceStencilMatrix.hh
 * @author islandox
 * @brief Topology-sized allocation for diagonal-plus-face-neighbor matrices.
 * @version 0.1
 * @date 2026-09-18
 * @copyright Copyright (c) 2026
 */
#pragma once
#include "FVM/details/OperatorDetails.hh"
#include "dataclass/TpetraTypes.hh"
#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

namespace SimpleFluid::FVM::detail
{
/**
 * @brief Allocate one diagonal plus at most one entry per incident logical face.
 *
 * Expanded coarse/fine subfaces may exceed the usual hexahedral stencil.
 * Counting faces also bounds duplicate neighbor insertions before fillComplete.
 * This bound is for face-neighbor operators, not extended gradient stencils.
 * @param mesh Mesh with owned rows and a complete face-neighbor overlap map.
 * @return Empty fill-active matrix with a per-row capacity bound.
 */
template<TpetraTypePack Pack, class MeshType>
Teuchos::RCP<typename Pack::matrix_type> make_face_stencil_matrix(const MeshType& mesh)
{
    const auto execution = acquire_mesh_execution(mesh);
    Teuchos::Array<size_t> capacities(mesh.num_owned_cells());
    if constexpr (requires { mesh.supports_region_execution(); mesh.visit([](const auto&) {}); })
    {
        if (mesh.supports_region_execution())
        {
            // The native logical range already includes coarse/fine expansion.
            // Its size bounds all insertions without translating every face to
            // the overlap map just to count it.
            mesh.visit([&](const auto& native)
            {
                if constexpr (requires { native.native_cell(uint64_t{}); native.cell_faces(uint64_t{}); })
                    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
                        capacities[owned] = native.cell_faces(mesh.cell_geometry_global_id(owned)).size() + 1;
            });
            return Teuchos::rcp(new typename Pack::matrix_type(
                mesh.owned_cell_map(), mesh.overlap_cell_map(), capacities()));
        }
    }
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = query_cell_id(mesh, static_cast<typename Pack::local_ordinal_type>(owned));
        capacities[owned] = mesh.faces(cell).size() + 1;
    }
    return Teuchos::rcp(new typename Pack::matrix_type(
        mesh.owned_cell_map(), mesh.overlap_cell_map(), capacities()));
}
} // namespace SimpleFluid::FVM::detail
