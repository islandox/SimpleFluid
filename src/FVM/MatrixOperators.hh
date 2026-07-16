/**
 * @file FVM/MatrixOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume matrix assembly operators (identity, diffusion, convection,
 *        pressure Poisson).
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "FVM/OperatorDetails.hh"
#include "fields/FaceField.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>

namespace SimpleFluid::FVM
{

/**
 * @brief Build an identity (or scaled-identity) square matrix over the
 *        given map.
 *
 * @tparam Pack The Tpetra type pack.
 * @param map Row/column map for the matrix.
 * @param diagonal Value to place on the diagonal (default 1.0).
 * @return Fill-complete sparse matrix with @p diagonal on the diagonal.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
identity_matrix(const Teuchos::RCP<const typename Pack::map_type>& map,
                typename Pack::scalar_type diagonal = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(map, map, 1));
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto lid = static_cast<local_ordinal_type>(row);
        Teuchos::Array<local_ordinal_type> cols{lid};
        Teuchos::Array<typename Pack::scalar_type> vals{diagonal};
        matrix->insertLocalValues(lid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble the finite-volume diffusion matrix (negative Laplacian).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @return Fill-complete sparse matrix representing the diffusion operator.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        cols.clear();
        vals.clear();
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto coeff =
                detail::interior_diffusion_coefficient(
                    mesh, face_lid, cell_lid, other, diffusivity);
            diagonal += coeff;
            cols.push_back(other);
            vals.push_back(-coeff);
        }

        cols.push_back(cell_lid);
        vals.push_back(diagonal);
        matrix->insertLocalValues(cell_lid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble the first-order upwind convection matrix from
 *        pre-computed face fluxes stored in a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar face fluxes.
 * @return Fill-complete sparse matrix representing the upwind convection
 *         operator.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
upwind_convection_matrix(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        cols.clear();
        vals.clear();
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid) ? face_fluxes.value(face_lid) : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                ? owner_oriented_flux
                                : -owner_oriented_flux;

            if (out_flux >= 0.0)
            {
                diagonal += out_flux;
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                cols.push_back(other);
                vals.push_back(out_flux);
            }
        }

        cols.push_back(cell_lid);
        vals.push_back(diagonal);
        matrix->insertLocalValues(cell_lid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble the pressure Poisson matrix with pressure-correction
 *        boundary conditions and an optional gauge constraint.
 *
 * Dirichlet pressure boundaries add the homogeneous correction coefficient
 * to the owner diagonal. Neumann pressure boundaries require no matrix term.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning a pressure boundary
 *         condition for a batch ID and in-batch face index.
 * @param mesh The computational mesh.
 * @param gauge_cell_gid Optional global row-map ID used to fix the pressure
 *        level for an all-Neumann system.
 * @param boundary_condition Pressure-boundary provider.
 * @return Fill-complete sparse matrix representing the Poisson operator.
 * @throws std::invalid_argument for unsupported pressure boundary types.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid,
    BoundaryConditionProvider boundary_condition)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid =
            mesh.owned_cell_map()->getGlobalElement(cell_lid);

        cols.clear();
        vals.clear();

        if (gauge_cell_gid && row_gid == *gauge_cell_gid)
        {
            cols.push_back(cell_lid);
            vals.push_back(1.0);
            matrix->insertLocalValues(cell_lid, cols(), vals());
            continue;
        }

        scalar_type diagonal = 0.0;
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto coeff =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other, scalar_type{1});
                diagonal += coeff;
                cols.push_back(other);
                vals.push_back(-coeff);
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                   >= boundary_locations.size())
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            if (!location.active)
            {
                continue;
            }
            const auto condition =
                boundary_condition(
                    location.batch_id, location.in_batch_id);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                diagonal +=
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, cell_lid, scalar_type{1});
            }
            else if (condition.type != BoundaryConditionType::Neumann)
            {
                throw std::invalid_argument(
                    "pressure_poisson_matrix supports only Dirichlet and "
                    "Neumann pressure boundary conditions.");
            }
        }

        cols.push_back(cell_lid);
        vals.push_back(diagonal > 0.0 ? diagonal : 1.0);
        matrix->insertLocalValues(cell_lid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble an all-Neumann pressure Poisson matrix with one gauge row.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param gauge_cell_gid Global row-map ID of the pressure gauge cell.
 * @return Fill-complete sparse matrix representing the Poisson operator.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
{
    auto homogeneous_neumann =
        [](int, size_t)
    {
        return BoundaryCondition{};
    };
    return pressure_poisson_matrix<Pack>(
        mesh, gauge_cell_gid, homogeneous_neumann);
}

} // namespace SimpleFluid::FVM
