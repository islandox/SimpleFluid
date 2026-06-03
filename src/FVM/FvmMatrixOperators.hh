/**
 * @file FvmMatrixOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume matrix assembly operators.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/FaceField.hh"
#include "FVM/FvmOperatorDetails.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <stdexcept>

namespace SimpleFluid::FvmOperators
{

/**
 * @brief Holds an assembled scalar diffusion matrix and right-hand side.
 *
 * The operator represents @f$-\nabla\cdot(\Gamma\nabla\phi)@f$ integrated
 * over each control volume.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
struct DiffusionSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::vector_type> rhs;
};

/**
 * @brief Holds a shared diffusion matrix and three-component right-hand side.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
struct VectorDiffusionSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::multi_vector_type> rhs;
};

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
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
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
 * @brief Assemble a scalar orthogonal diffusion system with source and
 *        boundary contributions.
 *
 * The source term is interpreted per unit volume. Dirichlet conditions
 * contribute a diagonal coefficient and RHS value; Neumann values are
 * interpreted as outward normal gradients and contribute
 * @f$\Gamma g_f |S_f|@f$ to the RHS.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
 * @tparam SourceProvider Callable returning scalar source for a cell LID.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @return Matrix/RHS pair for the scalar diffusion problem.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
diffusion_system(const Mesh<Pack>& mesh,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition,
                 SourceProvider right_hand_source)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument(
            "diffusion_system requires non-negative diffusivity.");
    }

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    auto rhs = Teuchos::rcp(
        new typename Pack::vector_type(mesh.owned_cell_map(), true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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
        rhs->replaceLocalValue(
            cell_lid, mesh.cell_volume(cell_lid) * right_hand_source(cell_lid));
    }

    for (const auto& [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        for (std::size_t in_patch_id = 0;
             in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
        {
            const auto face_lid = boundary_patch.face_lids[in_patch_id];
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto bc = boundary_condition(patch_id, in_patch_id);
            if (bc.type == BoundaryConditionType::Dirichlet)
            {
                const auto coeff =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, owner, diffusivity);
                if (coeff > scalar_type{0})
                {
                    local_ordinal_type col = owner;
                    matrix->sumIntoLocalValues(
                        owner,
                        Teuchos::arrayView(&col, 1),
                        Teuchos::arrayView(&coeff, 1));
                    rhs->sumIntoLocalValue(owner, coeff * bc.value);
                }
            }
            else if (bc.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(
                    owner, diffusivity * bc.value * mesh.face_area(face_lid));
            }
            else if (bc.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in diffusion_system.");
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble a scalar orthogonal diffusion system with zero source.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
DiffusionSystem<Pack>
diffusion_system(const Mesh<Pack>& mesh,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return diffusion_system<Pack>(mesh, diffusivity,
                                  boundary_condition, zero_source);
}

/**
 * @brief Assemble a vector orthogonal diffusion system with source and
 *        boundary contributions.
 *
 * The same scalar diffusion matrix is shared by all three vector
 * components; the RHS stores one column per component.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
VectorDiffusionSystem<Pack>
vector_diffusion_system(const Mesh<Pack>& mesh,
                        typename Pack::scalar_type diffusivity,
                        BoundaryConditionProvider boundary_condition,
                        SourceProvider right_hand_source)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr std::size_t num_components = 3;

    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument(
            "vector_diffusion_system requires non-negative diffusivity.");
    }

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    auto rhs = Teuchos::rcp(
        new typename Pack::multi_vector_type(
            mesh.owned_cell_map(), num_components, true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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

        const auto source_value = right_hand_source(cell_lid);
        for (std::size_t component = 0; component < num_components; ++component)
        {
            rhs->replaceLocalValue(
                cell_lid, component,
                mesh.cell_volume(cell_lid) * source_value.component(component));
        }
    }

    for (const auto& [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        for (std::size_t in_patch_id = 0;
             in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
        {
            const auto face_lid = boundary_patch.face_lids[in_patch_id];
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto bc = boundary_condition(patch_id, in_patch_id);
            if (bc.type == BoundaryConditionType::Dirichlet
                || bc.type == BoundaryConditionType::NoSlip)
            {
                const auto coeff =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, owner, diffusivity);
                if (coeff > scalar_type{0})
                {
                    local_ordinal_type col = owner;
                    matrix->sumIntoLocalValues(
                        owner,
                        Teuchos::arrayView(&col, 1),
                        Teuchos::arrayView(&coeff, 1));
                    for (std::size_t component = 0;
                         component < num_components;
                         ++component)
                    {
                        rhs->sumIntoLocalValue(
                            owner, component,
                            coeff * bc.value.component(component));
                    }
                }
            }
            else if (bc.type == BoundaryConditionType::Neumann)
            {
                for (std::size_t component = 0;
                     component < num_components;
                     ++component)
                {
                    rhs->sumIntoLocalValue(
                        owner, component,
                        diffusivity * bc.value.component(component)
                      * mesh.face_area(face_lid));
                }
            }
            else if (bc.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in vector_diffusion_system.");
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble a vector orthogonal diffusion system with zero source.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
VectorDiffusionSystem<Pack>
vector_diffusion_system(const Mesh<Pack>& mesh,
                        typename Pack::scalar_type diffusivity,
                        BoundaryConditionProvider boundary_condition)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;

    auto zero_source =
        [](local_ordinal_type) -> vec_type
    {
        return {};
    };

    return vector_diffusion_system<Pack>(
        mesh, diffusivity, boundary_condition, zero_source);
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
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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
 * @brief Assemble the pressure Poisson matrix with a gauge-cell
 *        constraint (sets the gauge row to identity).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param gauge_cell_gid Global ID of the cell used to fix the pressure
 *        level.
 * @return Fill-complete sparse matrix representing the Poisson operator.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
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
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        cols.clear();
        vals.clear();

        if (row_gid == gauge_cell_gid)
        {
            cols.push_back(cell_lid);
            vals.push_back(1.0);
            matrix->insertLocalValues(cell_lid, cols(), vals());
            continue;
        }

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
                    mesh, face_lid, cell_lid, other, scalar_type{1});
            diagonal += coeff;
            cols.push_back(other);
            vals.push_back(-coeff);
        }

        cols.push_back(cell_lid);
        vals.push_back(diagonal > 0.0 ? diagonal : 1.0);
        matrix->insertLocalValues(cell_lid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

} // namespace SimpleFluid::FvmOperators
