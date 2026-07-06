/**
 * @file DiffusionSystem.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Scalar and vector diffusion system assembly (orthogonal two-point flux).
 * @version 0.1
 * @date 2026-06-04
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "FVM/OperatorDetails.hh"
#include "fields/CellField.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <stdexcept>

namespace SimpleFluid::FVM
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
 *         for (batch id, in-batch face id).
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
        rhs->replaceLocalValue(
            cell_lid, mesh.cell_volume(cell_lid) * right_hand_source(cell_lid));
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0;
             in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
        {
            const auto face_lid = boundary_batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto bc = boundary_condition(batch_id, in_batch_id);
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
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (batch id, in-batch face id).
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @return Matrix/RHS pair for the scalar diffusion problem.
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
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (batch id, in-batch face id).
 * @tparam SourceProvider Callable returning vector source for a cell LID.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider returning a three-component
 *        source vector per cell.
 * @return VectorDiffusionSystem containing the shared matrix and
 *         three-column RHS.
 * @throws std::invalid_argument if @p diffusivity is negative.
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

    constexpr size_t num_components = 3;

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

        const auto source_value = right_hand_source(cell_lid);
        for (size_t component = 0; component < num_components; ++component)
        {
            rhs->replaceLocalValue(
                cell_lid, component,
                mesh.cell_volume(cell_lid) * source_value.component(component));
        }
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0;
             in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
        {
            const auto face_lid = boundary_batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto bc = boundary_condition(batch_id, in_batch_id);
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
                    for (size_t component = 0;
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
                for (size_t component = 0;
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

extern template struct DiffusionSystem<DefaultTpetraTypes>;
extern template struct VectorDiffusionSystem<DefaultTpetraTypes>;

} // namespace SimpleFluid::FVM
