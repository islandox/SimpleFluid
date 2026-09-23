/** @file FaceStencilMatrix.hh @brief Matrix allocation for variable cell incidence. */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "dataclass/TpetraTypes.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

namespace SimpleFluid::FVM::detail
{

/**
 * @brief Allocate the diagonal and every interior-face contribution per row.
 *
 * Count incidences rather than assuming a hexahedral stencil. Repeated neighbor
 * columns remain separate during insertion and are combined by fillComplete.
 */
template<TpetraTypePack Pack, class MeshType>
Teuchos::RCP<typename Pack::matrix_type> make_face_stencil_matrix(const MeshType& mesh)
{
    Teuchos::Array<size_t> entries_per_row;
    entries_per_row.reserve(mesh.num_owned_cells());
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = query_cell_id(mesh, static_cast<typename Pack::local_ordinal_type>(owned));
        size_t entries = 1;
        for (const auto face : mesh.faces(cell))
            if (mesh.is_interior_face(face)) ++entries;
        entries_per_row.push_back(entries);
    }
    return Teuchos::rcp(new typename Pack::matrix_type(
        mesh.owned_cell_map(), mesh.overlap_cell_map(), entries_per_row()));
}

} // namespace SimpleFluid::FVM::detail
