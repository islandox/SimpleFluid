/**
 * @file FVM/NonOrthogonalCorrection.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Non-orthogonal diffusion correction: explicit, implicit, hybrid treatments,
 *        residual evaluation, and solver wrappers.
 * @version 0.1
 * @date 2026-06-04
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "FVM/CellOperators.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/OperatorDetails.hh"
#include "solvers/BelosLinearSolver.hh"

#include <stdexcept>
#include <vector>

namespace SimpleFluid::FVM
{

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
 *         for (batch id, in-batch face id).
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
    typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0)
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
        auto gradient = gradients[static_cast<size_t>(cell_lid)];
        if (mesh.is_owned_cell(other_lid)
            && static_cast<size_t>(other_lid) < gradients.size())
        {
            gradient = (gradient
                      + gradients[static_cast<size_t>(other_lid)])
                     / 2.0;
        }
        return gradient;
    };

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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
                cell_lid,
                correction_weight * diffusivity
              * gradient.dot(tangential_area));
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

            const auto bc = boundary_condition(batch_id, in_batch_id);
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
            const auto& gradient = gradients[static_cast<size_t>(owner)];

            rhs.sumIntoLocalValue(
                owner,
                correction_weight * diffusivity
              * gradient.dot(tangential_area));
        }
    }
}

/**
 * @brief Add the explicit non-orthogonal diffusion correction for a
 *        vector field to a three-column RHS.
 *
 * The correction is applied component-wise using the vector least-squares
 * gradient reconstruction. Boundary faces are treated as prescribed-value
 * diffusion faces, matching vector transport-system assembly.
 */
template<TpetraTypePack Pack>
void add_explicit_non_orthogonal_correction(
    const VectorCellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity,
    typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = VectorCellField<Pack>::num_components;

    const auto& mesh = correction_field.mesh();
    if (diffusivity <= scalar_type{0}
        || correction_weight == scalar_type{0})
    {
        return;
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get())
    {
        throw std::invalid_argument(
            "add_explicit_non_orthogonal_correction requires an owned-cell RHS.");
    }
    if (rhs.getNumVectors() != num_components)
    {
        throw std::invalid_argument(
            "add_explicit_non_orthogonal_correction requires a three-component RHS.");
    }

    std::vector<VectorCellGradient<Pack>> gradients;
    cell_gradient(correction_field, gradients);

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid) -> VectorCellGradient<Pack>
    {
        auto gradient = gradients[static_cast<size_t>(cell_lid)];
        if (mesh.is_owned_cell(other_lid)
            && static_cast<size_t>(other_lid) < gradients.size())
        {
            const auto& other_gradient =
                gradients[static_cast<size_t>(other_lid)];
            for (size_t component = 0;
                 component < num_components;
                 ++component)
            {
                gradient[component] =
                    (gradient[component] + other_gradient[component]) / 2.0;
            }
        }
        return gradient;
    };

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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

            for (size_t component = 0;
                 component < num_components;
                 ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid, component,
                    correction_weight * diffusivity
                  * gradient[component].dot(tangential_area));
            }
        }
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        (void)batch_id;
        for (const auto face_lid : boundary_batch.face_lids)
        {
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
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
            const auto& gradient = gradients[static_cast<size_t>(owner)];

            for (size_t component = 0;
                 component < num_components;
                 ++component)
            {
                rhs.sumIntoLocalValue(
                    owner, component,
                    correction_weight * diffusivity
                  * gradient[component].dot(tangential_area));
            }
        }
    }
}

/**
 * @brief Add a scalar explicit non-orthogonal correction using a
 *        cell-centered variable diffusion coefficient.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
void add_variable_explicit_non_orthogonal_correction(
    const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    BoundaryConditionProvider boundary_condition,
    typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = correction_field.mesh();
    if (&coefficient_field.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "Variable non-orthogonal correction requires coefficient and "
            "solution fields on the same mesh.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get())
    {
        throw std::invalid_argument(
            "Variable non-orthogonal correction requires an owned-cell RHS.");
    }

    std::vector<typename Mesh<Pack>::Vec3> gradients;
    cell_gradient(correction_field, gradients);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(
                    face_lid, cell_lid);
            auto gradient = gradients[owned];
            if (mesh.is_owned_cell(other)
                && static_cast<size_t>(other) < gradients.size())
            {
                gradient =
                    (gradient + gradients[static_cast<size_t>(other)])
                  / scalar_type{2};
            }
            const auto face_coefficient =
                detail::harmonic_face_value(
                    mesh, face_lid, cell_lid, other,
                    coefficient_field);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid),
                    mesh.cell_center_vector(face_lid, cell_lid));
            rhs.sumIntoLocalValue(
                cell_lid,
                correction_weight * face_coefficient
              * gradient.dot(tangential_area));
        }
    }

    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0;
             in_batch_id < batch.face_lids.size();
             ++in_batch_id)
        {
            const auto face_lid = batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid)
                || !mesh.is_boundary_face(face_lid)
                || boundary_condition(batch_id, in_batch_id).type
                    != BoundaryConditionType::Dirichlet)
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, owner),
                    mesh.face_centroid(face_lid)
                        - mesh.cell_centroid(owner));
            rhs.sumIntoLocalValue(
                owner,
                correction_weight
              * coefficient_field.local_value(owner)
              * gradients[static_cast<size_t>(owner)].dot(
                    tangential_area));
        }
    }
}

/**
 * @brief Add a vector explicit non-orthogonal correction using a
 *        cell-centered variable diffusion coefficient.
 */
template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(
    const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = VectorCellField<Pack>::num_components;

    const auto& mesh = correction_field.mesh();
    if (&coefficient_field.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "Variable vector non-orthogonal correction requires coefficient "
            "and solution fields on the same mesh.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get()
        || rhs.getNumVectors() != components)
    {
        throw std::invalid_argument(
            "Variable vector non-orthogonal correction received an "
            "incompatible RHS.");
    }

    std::vector<VectorCellGradient<Pack>> gradients;
    cell_gradient(correction_field, gradients);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto tangential_area =
                detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid),
                    mesh.is_interior_face(face_lid)
                        ? mesh.cell_center_vector(face_lid, cell_lid)
                        : mesh.face_centroid(face_lid)
                            - mesh.cell_centroid(cell_lid));

            scalar_type face_coefficient =
                coefficient_field.local_value(cell_lid);
            auto gradient = gradients[owned];
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                face_coefficient =
                    detail::harmonic_face_value(
                        mesh, face_lid, cell_lid, other,
                        coefficient_field);
                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other) < gradients.size())
                {
                    const auto& other_gradient =
                        gradients[static_cast<size_t>(other)];
                    for (size_t component = 0;
                         component < components;
                         ++component)
                    {
                        gradient[component] =
                            (gradient[component]
                             + other_gradient[component])
                          / scalar_type{2};
                    }
                }
            }

            for (size_t component = 0;
                 component < components;
                 ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid, component,
                    correction_weight * face_coefficient
                  * gradient[component].dot(tangential_area));
            }
        }
    }
}

/**
 * @brief Assemble a scalar diffusion system with an explicit
 *        non-orthogonal correction from a previous solution.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (batch id, in-batch face id).
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
 * @brief Assemble scalar diffusion with an implicit least-squares
 *        non-orthogonal contribution.
 *
 * The orthogonal two-point contribution is assembled exactly as in
 * diffusion_system(). The non-orthogonal part linearizes the same
 * least-squares gradient reconstruction used by cell_gradient().
 *
 * @param non_orthogonal_implicit_weight Fraction of the tangential term to
 *        place in the matrix. Use 1.0 for a fully implicit operator and
 *        0.5 for the built-in hybrid treatment.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
implicit_non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    typename Pack::scalar_type non_orthogonal_implicit_weight = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument(
            "implicit_non_orthogonal_diffusion_system requires non-negative diffusivity.");
    }
    if (non_orthogonal_implicit_weight < scalar_type{0}
        || non_orthogonal_implicit_weight > scalar_type{1})
    {
        throw std::invalid_argument(
            "non-orthogonal implicit weight must be in [0, 1].");
    }

    const auto gradient_stencils =
        detail::least_squares_gradient_stencils(mesh);
    const auto boundary_locations = detail::boundary_face_locations(mesh);

    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 32));
    auto rhs = Teuchos::rcp(
        new typename Pack::vector_type(mesh.owned_cell_map(), true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(64);
    vals.reserve(64);

    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        row_values.clear();
        auto rhs_value =
            mesh.cell_volume(cell_lid) * right_hand_source(cell_lid);

        auto add_non_orthogonal_stencil =
            [&](local_ordinal_type gradient_cell_lid,
                scalar_type face_gradient_weight,
                const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (non_orthogonal_implicit_weight == scalar_type{0}
                || face_gradient_weight == scalar_type{0}
                || !mesh.is_owned_cell(gradient_cell_lid)
                || static_cast<size_t>(gradient_cell_lid)
                   >= gradient_stencils.size())
            {
                return;
            }

            const auto scale =
                -non_orthogonal_implicit_weight
              * diffusivity
              * face_gradient_weight;
            for (const auto& entry :
                 gradient_stencils[static_cast<size_t>(gradient_cell_lid)])
            {
                detail::add_matrix_entry(
                    row_values, entry.cell_lid,
                    scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto coeff =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other, diffusivity);
                detail::add_matrix_entry(row_values, cell_lid, coeff);
                detail::add_matrix_entry(row_values, other, -coeff);

                const auto area_vector =
                    mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.cell_center_vector(face_lid, cell_lid);
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(area_vector, d);

                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other)
                       < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(
                        other, scalar_type{0.5}, tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{1}, tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                   >= boundary_locations.size()
                || !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }

            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            const auto bc =
                boundary_condition(location.batch_id, location.in_batch_id);
            if (bc.type == BoundaryConditionType::Dirichlet)
            {
                const auto coeff =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, cell_lid, diffusivity);
                if (coeff > scalar_type{0})
                {
                    detail::add_matrix_entry(row_values, cell_lid, coeff);
                    rhs_value += coeff * bc.value;
                }

                const auto area_vector =
                    mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d =
                    mesh.face_centroid(face_lid)
                  - mesh.cell_centroid(cell_lid);
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(area_vector, d);
                add_non_orthogonal_stencil(
                    cell_lid, scalar_type{1}, tangential_area);
            }
            else if (bc.type == BoundaryConditionType::Neumann)
            {
                rhs_value +=
                    diffusivity * bc.value * mesh.face_area(face_lid);
            }
            else if (bc.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in implicit_non_orthogonal_diffusion_system.");
            }
        }

        if (row_values.find(cell_lid) == row_values.end())
        {
            row_values[cell_lid] = scalar_type{};
        }

        cols.clear();
        vals.clear();
        cols.reserve(row_values.size());
        vals.reserve(row_values.size());
        for (const auto& [column, value] : row_values)
        {
            cols.push_back(column);
            vals.push_back(value);
        }

        matrix->insertLocalValues(cell_lid, cols(), vals());
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble a fully implicit scalar non-orthogonal diffusion system.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
fully_implicit_non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source)
{
    return implicit_non_orthogonal_diffusion_system<Pack>(
        mesh, diffusivity, boundary_condition, right_hand_source, 1.0);
}

/**
 * @brief Assemble a diffusion system selected by the runtime
 *        non-orthogonal treatment switch.
 *
 * Explicit and hybrid systems use @p correction_field for the explicit RHS
 * fraction when it is supplied. If no correction field is provided,
 * `explicit` falls back to the orthogonal matrix and `hybrid` assembles only
 * its implicit half.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr)
{
    if (correction_field != nullptr
        && &correction_field->mesh() != &mesh)
    {
        throw std::invalid_argument(
            "non_orthogonal_diffusion_system requires correction field on the target mesh.");
    }

    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            if (correction_field == nullptr)
            {
                return diffusion_system<Pack>(
                    mesh, diffusivity, boundary_condition, right_hand_source);
            }
            return explicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source,
                *correction_field);

        case NonOrthogonalTreatment::Implicit:
            return fully_implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source);

        case NonOrthogonalTreatment::Hybrid:
        {
            auto system = implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source, 0.5);
            if (correction_field != nullptr)
            {
                add_explicit_non_orthogonal_correction<Pack>(
                    *correction_field, diffusivity, boundary_condition,
                    *system.rhs, 0.5);
            }
            return system;
        }
    }

    throw std::invalid_argument("Unknown NonOrthogonalTreatment value.");
}

/**
 * @brief Apply the full scalar non-orthogonal diffusion residual.
 *
 * The returned vector stores the integrated operator
 * @f$-\nabla\cdot(\Gamma\nabla\phi)@f$ using the orthogonal two-point
 * stencil plus the least-squares tangential correction. The input field's
 * overlap values must be synchronized before calling.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
Teuchos::RCP<typename Pack::vector_type>
full_diffusion_residual(
    const CellField<Pack>& field,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument(
            "full_diffusion_residual requires non-negative diffusivity.");
    }

    const auto& mesh = field.mesh();
    auto residual = Teuchos::rcp(
        new typename Pack::vector_type(mesh.owned_cell_map(), true));
    const auto boundary_locations = detail::boundary_face_locations(mesh);

    std::vector<typename Mesh<Pack>::Vec3> gradients;
    cell_gradient(field, gradients);

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid) -> typename Mesh<Pack>::Vec3
    {
        auto gradient = gradients[static_cast<size_t>(cell_lid)];
        if (mesh.is_owned_cell(other_lid)
            && static_cast<size_t>(other_lid) < gradients.size())
        {
            const auto& other_gradient =
                gradients[static_cast<size_t>(other_lid)];
            gradient = {(gradient.x + other_gradient.x) / 2.0,
                        (gradient.y + other_gradient.y) / 2.0,
                        (gradient.z + other_gradient.z) / 2.0};
        }
        return gradient;
    };

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = field.value(cell_lid);
        scalar_type value = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto coeff =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other, diffusivity);
                const auto area_vector =
                    mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.cell_center_vector(face_lid, cell_lid);
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(area_vector, d);
                const auto gradient = gradient_for_face(cell_lid, other);

                value += coeff * (phi_p - field.local_value(other))
                       - diffusivity * gradient.dot(tangential_area);
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                   >= boundary_locations.size()
                || !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }

            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            const auto bc =
                boundary_condition(location.batch_id, location.in_batch_id);
            if (bc.type == BoundaryConditionType::Dirichlet)
            {
                const auto coeff =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, cell_lid, diffusivity);
                const auto area_vector =
                    mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d =
                    mesh.face_centroid(face_lid)
                  - mesh.cell_centroid(cell_lid);
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(area_vector, d);
                const auto& gradient =
                    gradients[static_cast<size_t>(cell_lid)];

                value += coeff * (phi_p - bc.value)
                       - diffusivity * gradient.dot(tangential_area);
            }
            else if (bc.type == BoundaryConditionType::Neumann)
            {
                value -= diffusivity * bc.value * mesh.face_area(face_lid);
            }
            else if (bc.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in full_diffusion_residual.");
            }
        }

        residual->replaceLocalValue(cell_lid, value);
    }

    return residual;
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
 *         for (batch id, in-batch face id).
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
 * @brief Solve scalar diffusion using the selected non-orthogonal treatment.
 *
 * For `explicit`, @p nNonOrthogonalCorrectors has the same meaning as in
 * solve_explicit_non_orthogonal_diffusion(). For `implicit`, one fully
 * implicit solve is performed. For `hybrid`, an initial half-implicit solve
 * is followed by @p nNonOrthogonalCorrectors correction solves that put the
 * remaining half of the tangential term on the RHS.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
bool solve_non_orthogonal_diffusion(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    CellField<Pack>& solution,
    NonOrthogonalTreatment treatment,
    int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {})
{
    if (&solution.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "solve_non_orthogonal_diffusion requires solution on the target mesh.");
    }
    if (nNonOrthogonalCorrectors < 0)
    {
        throw std::invalid_argument(
            "nNonOrthogonalCorrectors cannot be negative.");
    }

    if (treatment == NonOrthogonalTreatment::Explicit)
    {
        return solve_explicit_non_orthogonal_diffusion<Pack>(
            mesh, diffusivity, boundary_condition, right_hand_source,
            solution, nNonOrthogonalCorrectors, linear_options);
    }

    if (treatment == NonOrthogonalTreatment::Implicit)
    {
        auto system =
            fully_implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source);
        solution.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto converged = solve_linear_system<Pack>(
            matrix, *system.rhs, solution.owned_data(), linear_options);
        mesh.sync_periodic_boundaries(solution);
        return converged;
    }

    if (treatment != NonOrthogonalTreatment::Hybrid)
    {
        throw std::invalid_argument("Unknown NonOrthogonalTreatment value.");
    }

    bool converged = true;
    for (int corrector = 0;
         corrector <= nNonOrthogonalCorrectors;
         ++corrector)
    {
        const auto* correction_field = corrector == 0 ? nullptr : &solution;
        auto system = non_orthogonal_diffusion_system<Pack>(
            mesh, diffusivity, boundary_condition, right_hand_source,
            NonOrthogonalTreatment::Hybrid, correction_field);

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
 *         for (batch id, in-batch face id).
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
 * @brief Solve a zero-source scalar diffusion equation using the selected
 *        non-orthogonal treatment.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
bool solve_non_orthogonal_diffusion(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    CellField<Pack>& solution,
    NonOrthogonalTreatment treatment,
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

    return solve_non_orthogonal_diffusion<Pack>(
        mesh, diffusivity, boundary_condition, zero_source, solution,
        treatment, nNonOrthogonalCorrectors, linear_options);
}

} // namespace SimpleFluid::FVM
