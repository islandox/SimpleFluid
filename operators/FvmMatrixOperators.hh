/**
 * @file FvmMatrixOperators.hh
 * @brief Finite-volume matrix assembly operators.
 */
#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FvmOperators
{

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
identity_matrix(const Teuchos::RCP<const typename Pack::map_type>& map,
                typename Pack::scalar_type diagonal = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(map, 1));
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(
            static_cast<typename Pack::local_ordinal_type>(row));
        Teuchos::Array<global_ordinal_type> cols{gid};
        Teuchos::Array<typename Pack::scalar_type> vals{diagonal};
        matrix->insertGlobalValues(gid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    Teuchos::Array<global_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        cols.clear();
        vals.clear();
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            const auto distance = mesh.face_cell_center_distance(face_lid);
            if (distance <= 0.0)
            {
                throw std::runtime_error("Cannot assemble diffusion across coincident cells.");
            }

            const auto coeff = diffusivity * mesh.face_area(face_lid) / distance;
            diagonal += coeff;
            cols.push_back(mesh.cell_global_id(other));
            vals.push_back(-coeff);
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
upwind_convection_matrix(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (face_fluxes.size() != mesh.num_faces())
    {
        throw std::invalid_argument("upwind_convection_matrix received the wrong face-flux size.");
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    Teuchos::Array<global_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        cols.clear();
        vals.clear();
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes[static_cast<std::size_t>(face_lid)];
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                ? owner_oriented_flux
                                : -owner_oriented_flux;

            if (out_flux >= 0.0)
            {
                diagonal += out_flux;
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_cell(face_lid, cell_lid);
                cols.push_back(mesh.cell_global_id(other));
                vals.push_back(out_flux);
            }
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    Teuchos::Array<global_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        cols.clear();
        vals.clear();

        if (row_gid == gauge_cell_gid)
        {
            cols.push_back(row_gid);
            vals.push_back(1.0);
            matrix->insertGlobalValues(row_gid, cols(), vals());
            continue;
        }

        scalar_type diagonal = 0.0;
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto distance = mesh.face_cell_center_distance(face_lid);
            if (distance <= 0.0)
            {
                throw std::runtime_error(
                    "Cannot assemble pressure Poisson matrix across coincident cells.");
            }
            const auto coeff = mesh.face_area(face_lid) / distance;
            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            diagonal += coeff;
            cols.push_back(mesh.cell_global_id(other));
            vals.push_back(-coeff);
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal > 0.0 ? diagonal : 1.0);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

} // namespace SimpleFluid::FvmOperators
