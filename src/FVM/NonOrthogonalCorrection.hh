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

#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::FVM
{

namespace detail
{

/** @brief Default boundary policy for a cell-centered coefficient field. */
struct OwnerCellBoundaryCoefficient
{
    template<class Scalar>
    Scalar operator()(int, size_t, Scalar owner_cell_value) const noexcept
    {
        return owner_cell_value;
    }
};

/** @brief Tag retaining the legacy interior-only gradient reconstruction. */
struct OmitBoundaryGradientSamples
{};

/** @brief Reconstruct one three-component value from a cached field view. */
template<TpetraTypePack Pack, class View>
inline auto vector_view_value(
    const View& values,
    typename Pack::local_ordinal_type cell_lid)
    -> typename VectorCellField<Pack>::vec_type
{
    return {values(cell_lid, 0),
            values(cell_lid, 1),
            values(cell_lid, 2)};
}

/** @brief Reconstruct one row-major 3x3 value from a cached field view. */
template<TpetraTypePack Pack, class View>
inline auto tensor_view_value(
    const View& values,
    typename Pack::local_ordinal_type cell_lid)
    -> typename TensorCellField<Pack>::tensor_type
{
    return {
        typename VectorCellField<Pack>::vec_type{
            values(cell_lid, 0),
            values(cell_lid, 1),
            values(cell_lid, 2)},
        typename VectorCellField<Pack>::vec_type{
            values(cell_lid, 3),
            values(cell_lid, 4),
            values(cell_lid, 5)},
        typename VectorCellField<Pack>::vec_type{
            values(cell_lid, 6),
            values(cell_lid, 7),
            values(cell_lid, 8)}};
}

/** @brief Evaluate cached scalar affine reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_scalar_affine_gradients(
    const CellField<Pack>& field,
    const std::vector<AffineLeastSquaresGradientStencil<Mesh<Pack>>>& stencils,
    VectorCellField<Pack>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument(
            "Scalar gradient stencil cache is incompatible with the mesh.");
    }
    const auto field_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        auto gradient = stencils[owned].constant;
        for (const auto& entry : stencils[owned].entries)
        {
            const auto value = field_values(entry.cell_lid, 0);
            gradient.x += entry.coefficient.x * value;
            gradient.y += entry.coefficient.y * value;
            gradient.z += entry.coefficient.z * value;
        }
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Evaluate cached vector affine reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_vector_affine_gradients(
    const VectorCellField<Pack>& field,
    const std::vector<VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>&
        stencils,
    TensorCellField<Pack>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument(
            "Vector gradient stencil cache is incompatible with the mesh.");
    }
    const auto field_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            auto gradient = stencils[owned].constants[component];
            for (const auto& entry : stencils[owned].entries)
            {
                const auto value =
                    field_values(entry.cell_lid, component);
                gradient.x += entry.coefficient.x * value;
                gradient.y += entry.coefficient.y * value;
                gradient.z += entry.coefficient.z * value;
            }
            gradient_values(cell_lid, component * 3) = gradient.x;
            gradient_values(cell_lid, component * 3 + 1) = gradient.y;
            gradient_values(cell_lid, component * 3 + 2) = gradient.z;
        }
    }
}

/** @brief Evaluate cached interior-only vector reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_vector_interior_gradients(
    const VectorCellField<Pack>& field,
    const std::vector<LeastSquaresGradientStencil<Mesh<Pack>>>& stencils,
    TensorCellField<Pack>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument(
            "Interior gradient stencil cache is incompatible with the mesh.");
    }
    const auto field_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (size_t component = 0;
             component < TensorCellField<Pack>::num_components;
             ++component)
        {
            gradient_values(cell_lid, component) = scalar_type{};
        }
        for (const auto& entry : stencils[owned])
        {
            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto component_value =
                    field_values(entry.cell_lid, component);
                gradient_values(cell_lid, component * 3) +=
                    entry.coefficient.x * component_value;
                gradient_values(cell_lid, component * 3 + 1) +=
                    entry.coefficient.y * component_value;
                gradient_values(cell_lid, component * 3 + 2) +=
                    entry.coefficient.z * component_value;
            }
        }
    }
}

} // namespace detail

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
 * @param correction_weight Fraction of the correction added to @p rhs.
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

    VectorCellField<Pack> gradients(
        correction_field.mesh_ptr(), "non_orthogonal_gradient");
    cell_gradient(correction_field, gradients);
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid) -> typename Mesh<Pack>::Vec3
    {
        return (detail::vector_view_value<Pack>(
                    gradient_values, cell_lid)
                + detail::vector_view_value<Pack>(
                    gradient_values, other_lid))
             / scalar_type{2};
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
            const auto gradient =
                detail::vector_view_value<Pack>(
                    gradient_values, owner);

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
 *
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param gradient_stencils Optional cached interior reconstruction stencil.
 * @throws std::invalid_argument if @p rhs is incompatible with the mesh or
 *         does not contain three component vectors.
 */
template<TpetraTypePack Pack,
         class BoundaryDiffusionProvider = detail::AlwaysDiffuseBoundary>
void add_explicit_non_orthogonal_correction(
    const VectorCellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity,
    typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryDiffusionProvider boundary_diffusion =
        BoundaryDiffusionProvider{},
    const std::vector<detail::LeastSquaresGradientStencil<Mesh<Pack>>>*
        gradient_stencils = nullptr)
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

    TensorCellField<Pack> gradients(
        correction_field.mesh_ptr(), "vector_non_orthogonal_gradient");
    if (gradient_stencils == nullptr)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        detail::evaluate_vector_interior_gradients(
            correction_field, *gradient_stencils, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid)
            -> typename TensorCellField<Pack>::tensor_type
    {
        auto gradient = detail::tensor_view_value<Pack>(
            gradient_values, cell_lid);
        const auto other_gradient =
            detail::tensor_view_value<Pack>(
                gradient_values, other_lid);
        for (size_t component = 0;
             component < num_components;
             ++component)
        {
            gradient[component] =
                (gradient[component] + other_gradient[component])
              / scalar_type{2};
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
        for (size_t in_batch_id = 0;
             in_batch_id < boundary_batch.face_lids.size();
             ++in_batch_id)
        {
            const auto face_lid = boundary_batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            if (!boundary_diffusion(batch_id, in_batch_id))
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
            const auto gradient =
                detail::tensor_view_value<Pack>(
                    gradient_values, owner);

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
 *        cell-centered variable diffusion coefficient and boundary samples.
 *
 * @param boundary_value Boundary-value provider used by gradient reconstruction.
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param boundary_coefficient Boundary-face coefficient provider receiving
 *        the owner-cell value as its fallback.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @throws std::invalid_argument if fields use different meshes, @p rhs uses
 *         an incompatible map, or a cell coefficient is negative.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider,
         class BoundaryCoefficientProvider =
             detail::OwnerCellBoundaryCoefficient>
void add_variable_explicit_non_orthogonal_correction(
    const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryCoefficientProvider boundary_coefficient =
        BoundaryCoefficientProvider{},
    const std::vector<
        detail::AffineLeastSquaresGradientStencil<Mesh<Pack>>>*
        gradient_stencils = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation =
        FaceCoefficientInterpolation::Harmonic)
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

    VectorCellField<Pack> gradients(
        correction_field.mesh_ptr(), "variable_non_orthogonal_gradient");
    if (gradient_stencils != nullptr)
    {
        detail::evaluate_scalar_affine_gradients(
            correction_field, *gradient_stencils, gradients);
    }
    else if constexpr (std::is_same_v<
                           std::remove_cvref_t<BoundaryValueProvider>,
                           detail::OmitBoundaryGradientSamples>)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        cell_gradient(
            correction_field, boundary_condition, boundary_value, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto coefficient_values =
        coefficient_field.local_read_view();

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
            const auto gradient =
                (detail::vector_view_value<Pack>(
                     gradient_values, cell_lid)
                 + detail::vector_view_value<Pack>(
                     gradient_values, other))
              / scalar_type{2};
            const auto face_coefficient =
                detail::face_coefficient_value(
                    mesh, face_lid, cell_lid, other,
                    coefficient_values(cell_lid, 0),
                    coefficient_values(other, 0),
                    coefficient_interpolation);
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
              * boundary_coefficient(
                    batch_id,
                    in_batch_id,
                    coefficient_values(owner, 0))
              * detail::vector_view_value<Pack>(
                    gradient_values, owner)
                    .dot(tangential_area));
        }
    }
}

/** @brief Backward-compatible scalar overload omitting boundary samples. */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryCoefficientProvider =
             detail::OwnerCellBoundaryCoefficient>
void add_variable_explicit_non_orthogonal_correction(
    const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    BoundaryConditionProvider boundary_condition,
    typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryCoefficientProvider boundary_coefficient =
        BoundaryCoefficientProvider{})
{
    add_variable_explicit_non_orthogonal_correction<Pack>(
        correction_field, coefficient_field, boundary_condition,
        detail::OmitBoundaryGradientSamples{}, rhs, correction_weight,
        boundary_coefficient);
}

/**
 * @brief Add a vector explicit non-orthogonal correction using a
 *        cell-centered variable diffusion coefficient and boundary samples.
 *
 * @param boundary_value Boundary-value provider used by gradient reconstruction.
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param boundary_coefficient Boundary-face coefficient provider receiving
 *        the owner-cell value as its fallback.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @param cached_boundary_locations Optional mesh-bound boundary lookup.
 * @throws std::invalid_argument if fields use different meshes, @p rhs is
 *         incompatible, or a cell coefficient is negative.
 */
template<TpetraTypePack Pack,
         class BoundaryValueProvider,
         class BoundaryDiffusionProvider = detail::AlwaysDiffuseBoundary,
         class BoundaryCoefficientProvider =
             detail::OwnerCellBoundaryCoefficient>
void add_variable_explicit_non_orthogonal_correction(
    const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    BoundaryValueProvider boundary_value,
    typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryDiffusionProvider boundary_diffusion =
        BoundaryDiffusionProvider{},
    BoundaryCoefficientProvider boundary_coefficient =
        BoundaryCoefficientProvider{},
    const std::vector<
        detail::VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>*
        gradient_stencils = nullptr,
    const std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>*
        cached_boundary_locations = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation =
        FaceCoefficientInterpolation::Harmonic)
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

    TensorCellField<Pack> gradients(
        correction_field.mesh_ptr(),
        "variable_vector_non_orthogonal_gradient");
    if (gradient_stencils != nullptr)
    {
        detail::evaluate_vector_affine_gradients(
            correction_field, *gradient_stencils, gradients);
    }
    else if constexpr (std::is_same_v<
                           std::remove_cvref_t<BoundaryValueProvider>,
                           detail::OmitBoundaryGradientSamples>)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        cell_gradient(correction_field, boundary_value, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto coefficient_values =
        coefficient_field.local_read_view();

    std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>
        local_boundary_locations;
    if (cached_boundary_locations == nullptr)
    {
        local_boundary_locations = detail::boundary_face_locations(mesh);
        cached_boundary_locations = &local_boundary_locations;
    }
    if (cached_boundary_locations->size() != mesh.num_faces())
    {
        throw std::invalid_argument(
            "Variable vector non-orthogonal correction received boundary "
            "locations for another mesh.");
    }
    const auto& boundary_locations = *cached_boundary_locations;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (mesh.is_boundary_face(face_lid))
            {
                const auto location =
                    boundary_locations.at(static_cast<size_t>(face_lid));
                if (!location.active
                    || !boundary_diffusion(
                        location.batch_id, location.in_batch_id))
                {
                    continue;
                }
            }
            const auto tangential_area =
                detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid),
                    mesh.is_interior_face(face_lid)
                        ? mesh.cell_center_vector(face_lid, cell_lid)
                        : mesh.face_centroid(face_lid)
                            - mesh.cell_centroid(cell_lid));

            scalar_type face_coefficient =
                coefficient_values(cell_lid, 0);
            auto gradient = detail::tensor_view_value<Pack>(
                gradient_values, cell_lid);
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                face_coefficient =
                    detail::face_coefficient_value(
                        mesh, face_lid, cell_lid, other,
                        coefficient_values(cell_lid, 0),
                        coefficient_values(other, 0),
                        coefficient_interpolation);
                const auto other_gradient =
                    detail::tensor_view_value<Pack>(
                        gradient_values, other);
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
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto location =
                    boundary_locations.at(static_cast<size_t>(face_lid));
                face_coefficient = boundary_coefficient(
                    location.batch_id,
                    location.in_batch_id,
                    face_coefficient);
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

/** @brief Backward-compatible vector overload omitting boundary samples. */
template<TpetraTypePack Pack,
         class BoundaryDiffusionProvider = detail::AlwaysDiffuseBoundary,
         class BoundaryCoefficientProvider =
             detail::OwnerCellBoundaryCoefficient>
void add_variable_explicit_non_orthogonal_correction(
    const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field,
    typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryDiffusionProvider boundary_diffusion =
        BoundaryDiffusionProvider{},
    BoundaryCoefficientProvider boundary_coefficient =
        BoundaryCoefficientProvider{})
{
    add_variable_explicit_non_orthogonal_correction<Pack>(
        correction_field, coefficient_field,
        detail::OmitBoundaryGradientSamples{}, rhs, correction_weight,
        boundary_diffusion, boundary_coefficient);
}

/**
 * @brief Add the explicit deviatoric transpose-gradient part of a symmetric
 *        viscous stress to a momentum RHS.
 *
 * The component Laplacian in physical_momentum_transport_system() supplies
 * @f$\nabla\cdot(\mu\nabla\mathbf{u})/\rho@f$ implicitly.  A Newtonian/RANS
 * stress is symmetric and deviatoric, so this routine adds the remaining
 * lagged term
 * @f$\nabla\cdot[\mu((\nabla\mathbf{u})^T
 * - 2/3\,\nabla\cdot\mathbf{u}\,I)]/\rho@f$ as face tractions. Cell gradients
 * are reconstructed from @p old_velocity and the supplied boundary values,
 * then interpolated to interior faces. The contribution is integrated over
 * each control volume because @p rhs stores integrated momentum balances.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryValueProvider Callable returning the velocity value at a
 *         boundary face.
 * @tparam BoundaryStressProvider Callable selecting boundary faces on which
 *         viscous stress is applied (for example, excluding slip faces).
 * @tparam BoundaryCoefficientProvider Callable providing boundary-face
 *         dynamic viscosity from its owner-cell fallback.
 * @param old_velocity Lagged velocity used by the explicit stress term.
 * @param dynamic_viscosity Molecular or effective dynamic-viscosity field.
 * @param reference_density Constant reference density.
 * @param boundary_value Boundary-face velocity provider.
 * @param[in,out] rhs Three-component owned-cell momentum RHS.
 * @param boundary_stress Boundary-face stress selector.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @param cached_boundary_locations Optional mesh-bound boundary lookup.
 */
template<TpetraTypePack Pack,
         class BoundaryValueProvider,
         class BoundaryStressProvider = detail::AlwaysDiffuseBoundary,
         class BoundaryCoefficientProvider =
             detail::OwnerCellBoundaryCoefficient>
void add_explicit_deviatoric_transpose_gradient_stress(
    const VectorCellField<Pack>& old_velocity,
    const CellField<Pack>& dynamic_viscosity,
    typename Pack::scalar_type reference_density,
    BoundaryValueProvider boundary_value,
    typename Pack::multi_vector_type& rhs,
    BoundaryStressProvider boundary_stress = BoundaryStressProvider{},
    BoundaryCoefficientProvider boundary_coefficient =
        BoundaryCoefficientProvider{},
    const std::vector<
        detail::VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>*
        gradient_stencils = nullptr,
    const std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>*
        cached_boundary_locations = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation =
        FaceCoefficientInterpolation::Harmonic)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = VectorCellField<Pack>::num_components;

    const auto& mesh = old_velocity.mesh();
    if (&dynamic_viscosity.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "Explicit transpose-gradient stress requires velocity and "
            "viscosity fields on the same mesh.");
    }
    if (!std::isfinite(reference_density)
        || reference_density <= scalar_type{})
    {
        throw std::invalid_argument(
            "Explicit transpose-gradient stress requires a finite positive "
            "reference density.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get()
        || rhs.getNumVectors() != components)
    {
        throw std::invalid_argument(
            "Explicit transpose-gradient stress received an incompatible "
            "momentum RHS.");
    }

    TensorCellField<Pack> gradients(
        old_velocity.mesh_ptr(), "transpose_gradient_stress_velocity_gradient");
    if (gradient_stencils == nullptr)
    {
        cell_gradient(old_velocity, boundary_value, gradients);
    }
    else
    {
        detail::evaluate_vector_affine_gradients(
            old_velocity, *gradient_stencils, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto viscosity_values =
        dynamic_viscosity.local_read_view();

    std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>
        local_boundary_locations;
    if (cached_boundary_locations == nullptr)
    {
        local_boundary_locations = detail::boundary_face_locations(mesh);
        cached_boundary_locations = &local_boundary_locations;
    }
    if (cached_boundary_locations->size() != mesh.num_faces())
    {
        throw std::invalid_argument(
            "Explicit transpose-gradient stress received boundary "
            "locations for another mesh.");
    }
    const auto& boundary_locations = *cached_boundary_locations;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto face_gradient =
                detail::tensor_view_value<Pack>(
                    gradient_values, cell_lid);
            auto face_viscosity =
                viscosity_values(cell_lid, 0);

            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto other_gradient =
                    detail::tensor_view_value<Pack>(
                        gradient_values, other);
                for (size_t component = 0;
                     component < components;
                     ++component)
                {
                    face_gradient[component] =
                        (face_gradient[component]
                         + other_gradient[component])
                      / scalar_type{2};
                }
                face_viscosity = detail::face_coefficient_value(
                    mesh, face_lid, cell_lid, other,
                    viscosity_values(cell_lid, 0),
                    viscosity_values(other, 0),
                    coefficient_interpolation);
            }
            else
            {
                if (!mesh.is_boundary_face(face_lid)
                    || static_cast<size_t>(face_lid)
                       >= boundary_locations.size())
                {
                    continue;
                }
                const auto location =
                    boundary_locations[static_cast<size_t>(face_lid)];
                if (!location.active
                    || !boundary_stress(
                        location.batch_id, location.in_batch_id))
                {
                    continue;
                }
                face_viscosity = boundary_coefficient(
                    location.batch_id,
                    location.in_batch_id,
                    face_viscosity);
            }

            if (!std::isfinite(face_viscosity)
                || face_viscosity < scalar_type{})
            {
                throw std::invalid_argument(
                    "Explicit transpose-gradient stress requires finite "
                    "non-negative dynamic viscosity.");
            }

            const auto area =
                mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto scale = face_viscosity / reference_density;

            // Tensor rows are grad(u_i).  Thus ((grad U)^T A)_i is the
            // area-weighted i-th column of the stored gradient tensor.
            const auto velocity_divergence =
                face_gradient[0].x
              + face_gradient[1].y
              + face_gradient[2].z;
            const auto isotropic_scale =
                scalar_type{2.0 / 3.0} * velocity_divergence;
            const typename VectorCellField<Pack>::vec_type traction{
                face_gradient[0].x * area.x
                    + face_gradient[1].x * area.y
                    + face_gradient[2].x * area.z
                    - isotropic_scale * area.x,
                face_gradient[0].y * area.x
                    + face_gradient[1].y * area.y
                    + face_gradient[2].y * area.z
                    - isotropic_scale * area.y,
                face_gradient[0].z * area.x
                    + face_gradient[1].z * area.y
                    + face_gradient[2].z * area.z
                    - isotropic_scale * area.z};
            for (size_t component = 0;
                 component < components;
                 ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid,
                    component,
                    scale * traction.component(component));
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
 * @param partition_correction_field Lagged field used to reconstruct the
 *        remote half of a partition-face gradient. The owned half remains in
 *        the matrix. This field is required when a nonzero implicit weight is
 *        assembled on a distributed partition face because the remote
 *        cell's extended least-squares stencil is not locally addressable.
 * @throws std::invalid_argument if diffusivity is not finite and
 *         non-negative, the implicit weight is not finite or outside
 *         `[0, 1]`, ranks disagree about whether an implicit correction or
 *         correction field is present, the correction field uses another
 *         mesh, or a distributed partition face needs a missing field.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
implicit_non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    typename Pack::scalar_type non_orthogonal_implicit_weight = 1.0,
    const CellField<Pack>* partition_correction_field = nullptr)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto communicator = mesh.owned_cell_map()->getComm();
    const int local_maximum_state[] = {
        !std::isfinite(diffusivity) || diffusivity < scalar_type{0},
        !std::isfinite(non_orthogonal_implicit_weight)
            || non_orthogonal_implicit_weight < scalar_type{0}
            || non_orthogonal_implicit_weight > scalar_type{1},
        non_orthogonal_implicit_weight > scalar_type{} ? 1 : 0,
        partition_correction_field == nullptr
            ? 0
            : (&partition_correction_field->mesh() == &mesh ? 1 : 2)};
    int global_maximum_state[4] = {};
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 4,
        local_maximum_state, global_maximum_state);

    const int local_minimum_state[] = {
        local_maximum_state[2], local_maximum_state[3]};
    int global_minimum_state[2] = {};
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 2,
        local_minimum_state, global_minimum_state);

    if (global_maximum_state[0] != 0)
    {
        throw std::invalid_argument(
            "implicit_non_orthogonal_diffusion_system requires finite "
            "non-negative diffusivity.");
    }
    if (global_maximum_state[1] != 0)
    {
        throw std::invalid_argument(
            "non-orthogonal implicit weight must be finite and in [0, 1].");
    }
    if (global_minimum_state[0] != global_maximum_state[2])
    {
        throw std::invalid_argument(
            "Ranks must agree whether implicit non-orthogonal correction "
            "is active.");
    }
    if (global_minimum_state[1] != global_maximum_state[3])
    {
        throw std::invalid_argument(
            "Ranks must supply a consistent partition correction field.");
    }
    if (global_maximum_state[3] == 2)
    {
        throw std::invalid_argument(
            "implicit_non_orthogonal_diffusion_system requires its "
            "partition correction field on the target mesh.");
    }
    if (non_orthogonal_implicit_weight > scalar_type{}
        && partition_correction_field == nullptr)
    {
        int local_partition_face = 0;
        for (size_t owned = 0;
             owned < mesh.num_owned_cells()
             && local_partition_face == 0;
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            for (const auto face_lid : mesh.faces(cell_lid))
            {
                if (!mesh.is_interior_face(face_lid))
                {
                    continue;
                }
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                if (!mesh.is_owned_cell(other))
                {
                    local_partition_face = 1;
                    break;
                }
            }
        }
        int global_partition_face = 0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_MAX, 1,
            &local_partition_face, &global_partition_face);
        if (global_partition_face != 0)
        {
            throw std::invalid_argument(
                "Distributed implicit non-orthogonal diffusion requires "
                "a synchronized partition correction field.");
        }
    }

    const auto gradient_stencils =
        detail::least_squares_gradient_stencils(mesh);
    const auto boundary_locations = detail::boundary_face_locations(mesh);
    std::unique_ptr<VectorCellField<Pack>> partition_gradients;
    if (non_orthogonal_implicit_weight > scalar_type{}
        && partition_correction_field != nullptr)
    {
        partition_gradients = std::make_unique<VectorCellField<Pack>>(
            partition_correction_field->mesh_ptr(),
            "partition_diffusion_non_orthogonal_gradient");
        cell_gradient(*partition_correction_field, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

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
                        cell_lid, scalar_type{0.5}, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error(
                            "Partition-face correction validation lost its "
                            "synchronized gradient field.");
                    }
                    rhs_value +=
                        non_orthogonal_implicit_weight
                      * diffusivity
                      * scalar_type{0.5}
                      * partition_gradients->local_value(other).dot(
                            tangential_area);
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
 *
 * On distributed meshes, @p partition_correction_field supplies the lagged
 * remote half of partition-face gradients.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack>
fully_implicit_non_orthogonal_diffusion_system(
    const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source,
    const CellField<Pack>* partition_correction_field = nullptr)
{
    return implicit_non_orthogonal_diffusion_system<Pack>(
        mesh, diffusivity, boundary_condition, right_hand_source, 1.0,
        partition_correction_field);
}

/**
 * @brief Assemble a diffusion system selected by the runtime
 *        non-orthogonal treatment switch.
 *
 * Explicit and hybrid systems use @p correction_field for the explicit RHS
 * fraction when it is supplied. If no correction field is provided,
 * `explicit` falls back to the orthogonal matrix. A nonzero implicit fraction
 * additionally requires the field when the mesh has distributed partition
 * faces so their remote gradient half can be synchronized.
 *
 * @throws std::invalid_argument if @p correction_field uses another mesh or
 *         @p treatment is invalid.
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
                mesh, diffusivity, boundary_condition, right_hand_source,
                correction_field);

        case NonOrthogonalTreatment::Hybrid:
        {
            auto system = implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source, 0.5,
                correction_field);
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
 *
 * @throws std::invalid_argument if @p diffusivity is negative.
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

    VectorCellField<Pack> gradients(
        field.mesh_ptr(), "full_diffusion_gradient");
    cell_gradient(field, gradients);
    gradients.sync_ghosts();
    const auto field_values = field.local_read_view();
    const auto gradient_values = gradients.local_read_view();

    auto gradient_for_face =
        [&](local_ordinal_type cell_lid,
            local_ordinal_type other_lid) -> typename Mesh<Pack>::Vec3
    {
        return (detail::vector_view_value<Pack>(
                    gradient_values, cell_lid)
                + detail::vector_view_value<Pack>(
                    gradient_values, other_lid))
             / scalar_type{2};
    };

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = field_values(cell_lid, 0);
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

                value += coeff * (phi_p - field_values(other, 0))
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
                const auto gradient =
                    detail::vector_view_value<Pack>(
                        gradient_values, cell_lid);

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
 *
 * @return `true` if every linear solve converged; otherwise `false`.
 * @throws std::invalid_argument if @p solution uses another mesh, the
 *         corrector count is negative, or @p treatment is invalid.
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
    using scalar_type = typename Pack::scalar_type;

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
                mesh, diffusivity, boundary_condition, right_hand_source,
                &solution);
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
        auto system =
            corrector == 0
          ? implicit_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source,
                scalar_type{0.5}, &solution)
          : non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source,
                NonOrthogonalTreatment::Hybrid, &solution);

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
