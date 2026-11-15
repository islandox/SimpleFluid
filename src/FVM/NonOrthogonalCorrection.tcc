/**
 * @file FVM/NonOrthogonalCorrection.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template implementations for compiled FVM assembly.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "utils/export_helper.hh"

#include "FVM/CellOperators.hh"
#include "FVM/FieldViewAccess.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/details/FieldStoredNonOrthogonalDiffusion.tcc"

#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace SimpleFluid::FVM
{
namespace detail
{

template<TpetraTypePack Pack>
void evaluate_scalar_affine_gradients(const CellField<Pack>& field,
    const std::vector<AffineLeastSquaresGradientStencil<MeshHandle<Pack>>>& stencils, VectorCellField<Pack>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Scalar gradient stencil cache is incompatible with the mesh.");
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

template<TpetraTypePack Pack>
void evaluate_vector_affine_gradients(const VectorCellField<Pack>& field,
    const std::vector<VectorAffineLeastSquaresGradientStencil<MeshHandle<Pack>>>& stencils, TensorCellField<Pack>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Vector gradient stencil cache is incompatible with the mesh.");
    }
    const auto field_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (size_t component = 0; component < VectorCellField<Pack>::num_components; ++component)
        {
            auto gradient = stencils[owned].constants[component];
            for (const auto& entry : stencils[owned].entries)
            {
                const auto value = field_values(entry.cell_lid, component);
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

template<TpetraTypePack Pack>
void evaluate_vector_interior_gradients(const VectorCellField<Pack>& field,
    const std::vector<LeastSquaresGradientStencil<MeshHandle<Pack>>>& stencils, TensorCellField<Pack>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Interior gradient stencil cache is incompatible with the mesh.");
    }
    const auto field_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (size_t component = 0; component < TensorCellField<Pack>::num_components; ++component)
        {
            gradient_values(cell_lid, component) = scalar_type{};
        }
        for (const auto& entry : stencils[owned])
        {
            for (size_t component = 0; component < VectorCellField<Pack>::num_components; ++component)
            {
                const auto component_value = field_values(entry.cell_lid, component);
                gradient_values(cell_lid, component * 3) += entry.coefficient.x * component_value;
                gradient_values(cell_lid, component * 3 + 1) += entry.coefficient.y * component_value;
                gradient_values(cell_lid, component * 3 + 2) += entry.coefficient.z * component_value;
            }
        }
    }
}

} // namespace detail

template<TpetraTypePack Pack>
void add_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    typename Pack::vector_type& rhs, typename Pack::scalar_type correction_weight)
{
    detail::add_stored_explicit_diffusion_correction<Pack>(
        correction_field, diffusivity, std::move(boundary_condition), rhs, correction_weight);
}

template<TpetraTypePack Pack>
void add_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryFaceSelector boundary_diffusion,
    const std::vector<detail::LeastSquaresGradientStencil<MeshHandle<Pack>>>* gradient_stencils)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = VectorCellField<Pack>::num_components;

    const auto& mesh = correction_field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    if (diffusivity <= scalar_type{0} || correction_weight == scalar_type{0})
    {
        return;
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get())
    {
        throw std::invalid_argument("add_explicit_non_orthogonal_correction requires an owned-cell RHS.");
    }
    if (rhs.getNumVectors() != num_components)
    {
        throw std::invalid_argument("add_explicit_non_orthogonal_correction requires a three-component RHS.");
    }

    TensorCellField<Pack> gradients(correction_field.mesh_ptr(), "vector_non_orthogonal_gradient");
    if (gradient_stencils == nullptr)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        detail::evaluate_vector_interior_gradients(correction_field, *gradient_stencils, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();

    auto gradient_for_face = [&](local_ordinal_type cell_lid, local_ordinal_type other_lid) ->
        typename TensorCellField<Pack>::tensor_type
    {
        auto gradient = detail::tensor_view_value<Pack>(gradient_values, cell_lid);
        const auto other_gradient = detail::tensor_view_value<Pack>(gradient_values, other_lid);
        for (size_t component = 0; component < num_components; ++component)
        {
            gradient[component] = (gradient[component] + other_gradient[component]) / scalar_type{2};
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

            const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto area_vector = mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto d = mesh.cell_center_vector(face_lid, cell_lid);
            const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);
            const auto gradient = gradient_for_face(cell_lid, other);

            for (size_t component = 0; component < num_components; ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid, component, correction_weight * diffusivity * gradient[component].dot(tangential_area));
            }
        }
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0; in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
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
            const auto area_vector = mesh.face_area_vector_outward(face_lid, owner);
            const auto d = mesh.face_centroid(face_lid) - mesh.cell_centroid(owner);
            const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);
            const auto gradient = detail::tensor_view_value<Pack>(gradient_values, owner);

            for (size_t component = 0; component < num_components; ++component)
            {
                rhs.sumIntoLocalValue(
                    owner, component, correction_weight * diffusivity * gradient[component].dot(tangential_area));
            }
        }
    }
}

template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryCoefficientProvider<Pack> boundary_coefficient,
    const std::vector<detail::AffineLeastSquaresGradientStencil<MeshHandle<Pack>>>* gradient_stencils,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (!boundary_coefficient)
    {
        boundary_coefficient = [](int, size_t, scalar_type owner_cell_value) { return owner_cell_value; };
    }

    const auto& mesh = correction_field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    if (&coefficient_field.mesh() != &mesh)
    {
        throw std::invalid_argument("Variable non-orthogonal correction requires coefficient and "
                                    "solution fields on the same mesh.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get())
    {
        throw std::invalid_argument("Variable non-orthogonal correction requires an owned-cell RHS.");
    }

    VectorCellField<Pack> gradients(correction_field.mesh_ptr(), "variable_non_orthogonal_gradient");
    if (gradient_stencils != nullptr)
    {
        detail::evaluate_scalar_affine_gradients(correction_field, *gradient_stencils, gradients);
    }
    else if (!boundary_value)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        cell_gradient(correction_field, boundary_condition, boundary_value, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto coefficient_values = coefficient_field.local_read_view();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto gradient = (detail::vector_view_value<Pack>(gradient_values, cell_lid) +
                                      detail::vector_view_value<Pack>(gradient_values, other)) /
                                  scalar_type{2};
            const auto face_coefficient = detail::face_coefficient_value(mesh, face_lid, cell_lid, other,
                coefficient_values(cell_lid, 0), coefficient_values(other, 0), coefficient_interpolation);
            const auto tangential_area = detail::non_orthogonal_area_vector(
                mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
            rhs.sumIntoLocalValue(cell_lid, correction_weight * face_coefficient * gradient.dot(tangential_area));
        }
    }

    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0; in_batch_id < batch.face_lids.size(); ++in_batch_id)
        {
            const auto face_lid = batch.face_lids[in_batch_id];
            const auto condition = boundary_condition(batch_id, in_batch_id);
            if (!mesh.is_owned_face(face_lid) || !mesh.is_boundary_face(face_lid) ||
                (condition.type != BoundaryConditionType::Dirichlet && condition.type != BoundaryConditionType::Robin))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, owner),
                    mesh.face_centroid(face_lid) - mesh.cell_centroid(owner));
            const auto weight = condition.type == BoundaryConditionType::Robin
                                    ? detail::robin_boundary_weight(mesh, face_lid, owner, condition.robin_coefficient)
                                    : scalar_type{1};
            rhs.sumIntoLocalValue(owner,
                weight * correction_weight * boundary_coefficient(batch_id, in_batch_id, coefficient_values(owner, 0)) *
                    detail::vector_view_value<Pack>(gradient_values, owner).dot(tangential_area));
        }
    }
}

template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    typename Pack::vector_type& rhs, typename Pack::scalar_type correction_weight,
    BoundaryCoefficientProvider<Pack> boundary_coefficient)
{
    add_variable_explicit_non_orthogonal_correction<Pack>(correction_field, coefficient_field, boundary_condition,
        ScalarBoundaryValueProvider<Pack>{}, rhs, correction_weight, boundary_coefficient);
}

template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, VectorBoundaryValueProvider<Pack> boundary_value,
    typename Pack::multi_vector_type& rhs, typename Pack::scalar_type correction_weight,
    BoundaryFaceSelector boundary_diffusion, BoundaryCoefficientProvider<Pack> boundary_coefficient,
    const std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshHandle<Pack>>>* gradient_stencils,
    const std::vector<detail::BoundaryFaceLocation<MeshHandle<Pack>>>* cached_boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = VectorCellField<Pack>::num_components;

    if (!boundary_coefficient)
    {
        boundary_coefficient = [](int, size_t, scalar_type owner_cell_value) { return owner_cell_value; };
    }

    const auto& mesh = correction_field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    if (&coefficient_field.mesh() != &mesh)
    {
        throw std::invalid_argument("Variable vector non-orthogonal correction requires coefficient "
                                    "and solution fields on the same mesh.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get() || rhs.getNumVectors() != components)
    {
        throw std::invalid_argument("Variable vector non-orthogonal correction received an "
                                    "incompatible RHS.");
    }

    TensorCellField<Pack> gradients(correction_field.mesh_ptr(), "variable_vector_non_orthogonal_gradient");
    if (gradient_stencils != nullptr)
    {
        detail::evaluate_vector_affine_gradients(correction_field, *gradient_stencils, gradients);
    }
    else if (!boundary_value)
    {
        cell_gradient(correction_field, gradients);
    }
    else
    {
        cell_gradient(correction_field, boundary_value, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto coefficient_values = coefficient_field.local_read_view();

    std::vector<detail::BoundaryFaceLocation<MeshHandle<Pack>>> local_boundary_locations;
    if (cached_boundary_locations == nullptr)
    {
        local_boundary_locations = detail::boundary_face_locations(mesh);
        cached_boundary_locations = &local_boundary_locations;
    }
    if (cached_boundary_locations->size() != mesh.num_faces())
    {
        throw std::invalid_argument("Variable vector non-orthogonal correction received boundary "
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
                const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                if (!location.active || !boundary_diffusion(location.batch_id, location.in_batch_id))
                {
                    continue;
                }
            }
            const auto tangential_area =
                detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                    mesh.is_interior_face(face_lid) ? mesh.cell_center_vector(face_lid, cell_lid)
                                                    : mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));

            scalar_type face_coefficient = coefficient_values(cell_lid, 0);
            auto gradient = detail::tensor_view_value<Pack>(gradient_values, cell_lid);
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                face_coefficient = detail::face_coefficient_value(mesh, face_lid, cell_lid, other,
                    coefficient_values(cell_lid, 0), coefficient_values(other, 0), coefficient_interpolation);
                const auto other_gradient = detail::tensor_view_value<Pack>(gradient_values, other);
                for (size_t component = 0; component < components; ++component)
                {
                    gradient[component] = (gradient[component] + other_gradient[component]) / scalar_type{2};
                }
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                face_coefficient = boundary_coefficient(location.batch_id, location.in_batch_id, face_coefficient);
            }

            for (size_t component = 0; component < components; ++component)
            {
                rhs.sumIntoLocalValue(cell_lid, component,
                    correction_weight * face_coefficient * gradient[component].dot(tangential_area));
            }
        }
    }
}

template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryFaceSelector boundary_diffusion,
    BoundaryCoefficientProvider<Pack> boundary_coefficient)
{
    add_variable_explicit_non_orthogonal_correction<Pack>(correction_field, coefficient_field,
        VectorBoundaryValueProvider<Pack>{}, rhs, correction_weight, boundary_diffusion, boundary_coefficient);
}

template<TpetraTypePack Pack>
void add_explicit_deviatoric_transpose_gradient_stress(const VectorCellField<Pack>& old_velocity,
    const CellField<Pack>& dynamic_viscosity, typename Pack::scalar_type reference_density,
    VectorBoundaryValueProvider<Pack> boundary_value, typename Pack::multi_vector_type& rhs,
    BoundaryFaceSelector boundary_stress, BoundaryCoefficientProvider<Pack> boundary_coefficient,
    const std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshHandle<Pack>>>* gradient_stencils,
    const std::vector<detail::BoundaryFaceLocation<MeshHandle<Pack>>>* cached_boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = VectorCellField<Pack>::num_components;

    if (!boundary_coefficient)
    {
        boundary_coefficient = [](int, size_t, scalar_type owner_cell_value) { return owner_cell_value; };
    }

    const auto& mesh = old_velocity.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    if (&dynamic_viscosity.mesh() != &mesh)
    {
        throw std::invalid_argument("Explicit transpose-gradient stress requires velocity and "
                                    "viscosity fields on the same mesh.");
    }
    if (!std::isfinite(reference_density) || reference_density <= scalar_type{})
    {
        throw std::invalid_argument("Explicit transpose-gradient stress requires a finite positive "
                                    "reference density.");
    }
    if (rhs.getMap().get() != mesh.owned_cell_map().get() || rhs.getNumVectors() != components)
    {
        throw std::invalid_argument("Explicit transpose-gradient stress received an incompatible "
                                    "momentum RHS.");
    }

    TensorCellField<Pack> gradients(old_velocity.mesh_ptr(), "transpose_gradient_stress_velocity_gradient");
    if (gradient_stencils == nullptr)
    {
        cell_gradient(old_velocity, boundary_value, gradients);
    }
    else
    {
        detail::evaluate_vector_affine_gradients(old_velocity, *gradient_stencils, gradients);
    }
    gradients.sync_ghosts();
    const auto gradient_values = gradients.local_read_view();
    const auto viscosity_values = dynamic_viscosity.local_read_view();

    std::vector<detail::BoundaryFaceLocation<MeshHandle<Pack>>> local_boundary_locations;
    if (cached_boundary_locations == nullptr)
    {
        local_boundary_locations = detail::boundary_face_locations(mesh);
        cached_boundary_locations = &local_boundary_locations;
    }
    if (cached_boundary_locations->size() != mesh.num_faces())
    {
        throw std::invalid_argument("Explicit transpose-gradient stress received boundary "
                                    "locations for another mesh.");
    }
    const auto& boundary_locations = *cached_boundary_locations;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto face_gradient = detail::tensor_view_value<Pack>(gradient_values, cell_lid);
            auto face_viscosity = viscosity_values(cell_lid, 0);

            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_gradient = detail::tensor_view_value<Pack>(gradient_values, other);
                for (size_t component = 0; component < components; ++component)
                {
                    face_gradient[component] = (face_gradient[component] + other_gradient[component]) / scalar_type{2};
                }
                face_viscosity = detail::face_coefficient_value(mesh, face_lid, cell_lid, other,
                    viscosity_values(cell_lid, 0), viscosity_values(other, 0), coefficient_interpolation);
            }
            else
            {
                if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size())
                {
                    continue;
                }
                const auto location = boundary_locations[static_cast<size_t>(face_lid)];
                if (!location.active || !boundary_stress(location.batch_id, location.in_batch_id))
                {
                    continue;
                }
                face_viscosity = boundary_coefficient(location.batch_id, location.in_batch_id, face_viscosity);
            }

            if (!std::isfinite(face_viscosity) || face_viscosity < scalar_type{})
            {
                throw std::invalid_argument("Explicit transpose-gradient stress requires finite "
                                            "non-negative dynamic viscosity.");
            }

            const auto area = mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto scale = face_viscosity / reference_density;

            // Tensor rows are grad(u_i).  Thus ((grad U)^T A)_i is the
            // area-weighted i-th column of the stored gradient tensor.
            const auto velocity_divergence = face_gradient[0].x + face_gradient[1].y + face_gradient[2].z;
            const auto isotropic_scale = scalar_type{2.0 / 3.0} * velocity_divergence;
            const typename VectorCellField<Pack>::vec_type traction{
                face_gradient[0].x * area.x + face_gradient[1].x * area.y + face_gradient[2].x * area.z -
                    isotropic_scale * area.x,
                face_gradient[0].y * area.x + face_gradient[1].y * area.y + face_gradient[2].y * area.z -
                    isotropic_scale * area.y,
                face_gradient[0].z * area.x + face_gradient[1].z * area.y + face_gradient[2].z * area.z -
                    isotropic_scale * area.z};
            for (size_t component = 0; component < components; ++component)
            {
                rhs.sumIntoLocalValue(cell_lid, component, scale * traction.component(component));
            }
        }
    }
}

template<TpetraTypePack Pack>
DiffusionSystem<Pack> explicit_non_orthogonal_diffusion_system(const MeshHandle<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, const CellField<Pack>& correction_field)
{
    return non_orthogonal_diffusion_system<Pack, MeshHandle<Pack>>(mesh, diffusivity,
        std::move(boundary_condition), std::move(right_hand_source), NonOrthogonalTreatment::Explicit, &correction_field);
}

template<TpetraTypePack Pack>
DiffusionSystem<Pack> fully_implicit_non_orthogonal_diffusion_system(const MeshHandle<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, const CellField<Pack>* partition_correction_field)
{
    return implicit_non_orthogonal_diffusion_system<Pack, MeshHandle<Pack>>(
        mesh, diffusivity, boundary_condition, right_hand_source, 1.0, partition_correction_field);
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::vector_type> full_diffusion_residual(const CellField<Pack>& field,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument("full_diffusion_residual requires non-negative diffusivity.");
    }

    const auto& mesh = field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    auto residual = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    const auto boundary_locations = detail::boundary_face_locations(mesh);

    VectorCellField<Pack> gradients(field.mesh_ptr(), "full_diffusion_gradient");
    cell_gradient(field, gradients);
    gradients.sync_ghosts();
    const auto field_values = field.local_read_view();
    const auto gradient_values = gradients.local_read_view();

    auto gradient_for_face = [&](local_ordinal_type cell_lid, local_ordinal_type other_lid) -> typename MeshHandle<Pack>::Vec3
    {
        return (detail::vector_view_value<Pack>(gradient_values, cell_lid) +
                   detail::vector_view_value<Pack>(gradient_values, other_lid)) /
               scalar_type{2};
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
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto coeff = detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                const auto area_vector = mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.cell_center_vector(face_lid, cell_lid);
                const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);
                const auto gradient = gradient_for_face(cell_lid, other);

                value += coeff * (phi_p - field_values(other, 0)) - diffusivity * gradient.dot(tangential_area);
                continue;
            }

            if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size() ||
                !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }

            const auto location = boundary_locations[static_cast<size_t>(face_lid)];
            const auto bc = boundary_condition(location.batch_id, location.in_batch_id);
            if (bc.type == BoundaryConditionType::Dirichlet)
            {
                const auto coeff = detail::boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
                const auto area_vector = mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
                const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);
                const auto gradient = detail::vector_view_value<Pack>(gradient_values, cell_lid);

                value += coeff * (phi_p - bc.value) - diffusivity * gradient.dot(tangential_area);
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

} // namespace SimpleFluid::FVM
