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
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "FVM/FvmCellOperators.hh"
#include "FVM/FvmOperatorDetails.hh"
#include "geometry/Mesh.hh"
#include "solvers/BelosLinearSolver.hh"

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
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
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
 * @brief Add the explicit non-orthogonal diffusion correction to an RHS.
 *
 * For each face, the area vector is split into @f$S_f = E_f + T_f@f$.
 * The implicit system owns the @f$E_f@f$ contribution; this routine adds
 * @f$\Gamma_f \nabla\phi_f\cdot T_f@f$ to the RHS for the explicit
 * correction of @f$-\nabla\cdot(\Gamma\nabla\phi)@f$.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
 * @param correction_field Scalar field used to compute the gradient for
 *        the non-orthogonal correction.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param[in,out] rhs Owned-cell RHS vector; receives the correction.
 * @throws std::invalid_argument if @p rhs is not on the owned-cell map.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
void add_explicit_non_orthogonal_correction(
    const CellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    typename Pack::vector_type& rhs)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = correction_field.mesh();
    if (diffusivity <= scalar_type{0})
    {
        return;
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get())
    {
        throw std::invalid_argument(
            "add_explicit_non_orthogonal_correction requires an owned-cell RHS.");
    }

    std::vector<typename Mesh<Pack>::Vec3> gradients;
    cell_gradient(correction_field, gradients);

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid) -> typename Mesh<Pack>::Vec3
    {
        auto gradient = gradients[static_cast<std::size_t>(cell_lid)];
        if (mesh.is_owned_cell(other_lid)
            && static_cast<std::size_t>(other_lid) < gradients.size())
        {
            gradient = (gradient
                      + gradients[static_cast<std::size_t>(other_lid)])
                     / 2.0;
        }
        return gradient;
    };

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto area_vector =
                mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto d = mesh.cell_center_vector(face_lid, cell_lid);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(area_vector, d);
            const auto gradient = gradient_for_face(cell_lid, other);

            rhs.sumIntoLocalValue(
                cell_lid, diffusivity * gradient.dot(tangential_area));
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

            const auto bc = boundary_condition(patch_id, in_patch_id);
            if (bc.type != BoundaryConditionType::Dirichlet)
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto area_vector =
                mesh.face_area_vector_outward(face_lid, owner);
            const auto d =
                mesh.face_centroid(face_lid) - mesh.cell_centroid(owner);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(area_vector, d);
            const auto& gradient = gradients[static_cast<std::size_t>(owner)];

            rhs.sumIntoLocalValue(
                owner, diffusivity * gradient.dot(tangential_area));
        }
    }
}

/**
 * @brief Assemble a scalar diffusion system with an explicit
 *        non-orthogonal correction from a previous solution.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
 * @tparam SourceProvider Callable returning scalar source for a cell LID.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @param correction_field Scalar field whose gradient drives the
 *        non-orthogonal correction.
 * @return Matrix/RHS pair with the explicit correction included.
 * @throws std::invalid_argument if @p correction_field is not on the
 *         target mesh.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
explicit_non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    const CellField<Pack>& correction_field)
{
    if (&correction_field.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "explicit_non_orthogonal_diffusion_system requires correction field on the target mesh.");
    }

    auto system = diffusion_system<Pack>(
        mesh, diffusivity, boundary_condition, right_hand_source);
    add_explicit_non_orthogonal_correction<Pack>(
        correction_field, diffusivity, boundary_condition, *system.rhs);
    return system;
}

/**
 * @brief Solve a scalar diffusion equation with explicit non-orthogonal
 *        correction sweeps.
 *
 * A value of zero performs only the orthogonal solve. Positive values run
 * that solve followed by @p nNonOrthogonalCorrectors explicit correction
 * solves using the latest solution gradient.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
 * @tparam SourceProvider Callable returning scalar source for a cell LID.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @param[in,out] solution Solution field; zeroed before each solve and
 *        updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 * @throws std::invalid_argument if @p solution is not on the target mesh
 *         or @p nNonOrthogonalCorrectors is negative.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
bool solve_explicit_non_orthogonal_diffusion(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    CellField<Pack>& solution,
    int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {})
{
    if (&solution.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "solve_explicit_non_orthogonal_diffusion requires solution on the target mesh.");
    }
    if (nNonOrthogonalCorrectors < 0)
    {
        throw std::invalid_argument(
            "nNonOrthogonalCorrectors cannot be negative.");
    }

    bool converged = true;
    for (int corrector = 0;
         corrector <= nNonOrthogonalCorrectors;
         ++corrector)
    {
        auto system =
            corrector == 0
          ? diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source)
          : explicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source,
                solution);

        solution.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        converged = solve_linear_system<Pack>(
            matrix, *system.rhs, solution.owned_data(), linear_options);
        mesh.sync_periodic_boundaries(solution);
        if (!converged)
        {
            return false;
        }
    }

    return converged;
}

/**
 * @brief Solve a scalar diffusion equation with zero source and explicit
 *        non-orthogonal correction sweeps.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (patch id, in-patch face id).
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param[in,out] solution Solution field; updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
bool solve_explicit_non_orthogonal_diffusion(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    CellField<Pack>& solution,
    int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {})
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return solve_explicit_non_orthogonal_diffusion<Pack>(
        mesh, diffusivity, boundary_condition, zero_source, solution,
        nNonOrthogonalCorrectors, linear_options);
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
 *         for (patch id, in-patch face id).
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
