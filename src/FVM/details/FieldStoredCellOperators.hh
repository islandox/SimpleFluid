/**
 * @file FVM/details/FieldStoredCellOperators.hh
 * @brief Internal cell-operator kernels for mesh-aware stored fields.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "fields/FieldStored.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** @brief Require two stored fields to reference the same mesh object. */
template<class InputField, class OutputField>
void require_same_stored_mesh(const InputField& input, const OutputField& output, const char* operation)
{
    if (input.mesh_ptr().get() != output.mesh_ptr().get())
    {
        throw std::invalid_argument(std::string(operation) + " requires input and output fields on one mesh.");
    }
}

/** @brief Evaluate interior least-squares scalar-gradient stencils. */
template<TpetraTypePack Pack, class MeshType>
void stored_scalar_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = least_squares_gradient_stencils(field.mesh());
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        vec_type gradient{};
        for (const auto& entry : stencils[owned])
        {
            gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
        }
        gradients.set_owned_value(static_cast<local_ordinal_type>(owned), gradient);
    }
}

/** @brief Evaluate boundary-aware affine scalar-gradient stencils. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = scalar_affine_gradient_stencils(field.mesh(), boundary_condition, boundary_value);
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        auto gradient = stencils[owned].constant;
        for (const auto& entry : stencils[owned].entries)
        {
            gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
        }
        gradients.set_owned_value(static_cast<local_ordinal_type>(owned), gradient);
    }
}

/** @brief Evaluate cached interior scalar-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache>
void stored_cached_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& geometry = cache.interior_geometry();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = field.value(cell_lid);
        vec_type gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            gradient = gradient + sample.weight * (field.local_value(sample.other_lid) - value_p);
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/** @brief Evaluate cached boundary-aware scalar-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_cached_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using vec_type = vec3<scalar_type>;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& interior_geometry = cache.interior_geometry();
    const auto& boundary_geometry = cache.boundary_geometry();
    for (size_t owned = 0; owned < boundary_geometry.size(); ++owned)
    {
        const auto& geometry =
            boundary_geometry[owned].boundary_samples.empty() ? interior_geometry[owned] : boundary_geometry[owned];
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = field.value(cell_lid);
        vec_type gradient{};
        for (const auto& sample : geometry.interior_samples)
        {
            gradient = gradient + sample.weight * (field.local_value(sample.other_lid) - value_p);
        }
        for (const auto& sample : geometry.boundary_samples)
        {
            const auto condition = boundary_condition(sample.location.batch_id, sample.location.in_batch_id);
            scalar_type increment{};
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                increment =
                    static_cast<scalar_type>(boundary_value(sample.location.batch_id, sample.location.in_batch_id)) -
                    value_p;
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                increment =
                    static_cast<scalar_type>(condition.value) * static_cast<scalar_type>(sample.normal_distance);
            }
            else
            {
                throw std::invalid_argument("cell_gradient supports only Dirichlet and Neumann "
                                            "scalar boundary conditions.");
            }
            gradient = gradient + sample.weight * increment;
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/** @brief Evaluate interior vector-gradient stencils into tensor storage. */
template<TpetraTypePack Pack, class MeshType>
void stored_vector_cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = least_squares_gradient_stencils(field.mesh());
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        tensor_type gradient{};
        for (const auto& entry : stencils[owned])
        {
            const auto value = field.local_value(entry.cell_lid);
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + entry.coefficient * value.component(component);
            }
        }
        gradients.set_owned_value(static_cast<local_ordinal_type>(owned), gradient);
    }
}

/** @brief Evaluate boundary-aware vector-gradient stencils. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = vector_affine_gradient_stencils(field.mesh(), boundary_value);
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        tensor_type gradient = stencils[owned].constants;
        for (const auto& entry : stencils[owned].entries)
        {
            const auto value = field.local_value(entry.cell_lid);
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + entry.coefficient * value.component(component);
            }
        }
        gradients.set_owned_value(static_cast<local_ordinal_type>(owned), gradient);
    }
}

/** @brief Evaluate cached interior vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache>
void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& geometry = cache.interior_geometry();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = field.value(cell_lid);
        tensor_type gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            const auto delta = field.local_value(sample.other_lid) - value_p;
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + sample.weight * delta.component(component);
            }
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/** @brief Evaluate cached boundary-aware vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache, class BoundaryValueProvider>
void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& interior_geometry = cache.interior_geometry();
    const auto& boundary_geometry = cache.boundary_geometry();
    for (size_t owned = 0; owned < boundary_geometry.size(); ++owned)
    {
        const auto& geometry =
            boundary_geometry[owned].boundary_samples.empty() ? interior_geometry[owned] : boundary_geometry[owned];
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = field.value(cell_lid);
        tensor_type gradient{};
        for (const auto& sample : geometry.interior_samples)
        {
            const auto delta = field.local_value(sample.other_lid) - value_p;
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + sample.weight * delta.component(component);
            }
        }
        for (const auto& sample : geometry.boundary_samples)
        {
            const auto delta = boundary_value(sample.location.batch_id, sample.location.in_batch_id) - value_p;
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + sample.weight * delta.component(component);
            }
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/** @brief Gauss-linear scalar gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        const auto cell_value = field.local_value(cell_lid);
        vec_type gradient{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            scalar_type face_value = cell_value;
            if (mesh.is_interior_face(face_id))
            {
                const auto other_id = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid = packed_cell_local_id(mesh, other_id);
                const auto cell_distance = mesh.cell_to_face_distance(face_id, cell_id);
                const auto other_distance = mesh.cell_to_face_distance(face_id, other_id);
                const auto total_distance = cell_distance + other_distance;
                face_value =
                    total_distance > scalar_type{}
                        ? (other_distance * cell_value + cell_distance * field.local_value(other_lid)) / total_distance
                        : scalar_type{0.5} * (cell_value + field.local_value(other_lid));
            }
            else if (mesh.is_boundary_face(face_id))
            {
                const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                if (location.active)
                {
                    const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                    if (condition.type == BoundaryConditionType::Dirichlet)
                    {
                        face_value = boundary_value(location.batch_id, location.in_batch_id);
                    }
                    else if (condition.type == BoundaryConditionType::Neumann)
                    {
                        face_value = cell_value +
                                     static_cast<scalar_type>(condition.value) *
                                         static_cast<scalar_type>(boundary_normal_distance(mesh, face_lid, cell_lid));
                    }
                    else
                    {
                        throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                    "Dirichlet and Neumann boundaries.");
                    }
                }
            }
            gradient = gradient + mesh.face_area_vector_outward(face_id, cell_id) * face_value;
        }
        gradients.set_owned_value(cell_lid, gradient / static_cast<scalar_type>(mesh.cell_volume(cell_id)));
    }
}

/** @brief Gauss-linear vector gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        const auto cell_value = field.local_value(cell_lid);
        tensor_type gradient{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            auto face_value = cell_value;
            if (mesh.is_interior_face(face_id))
            {
                const auto other_id = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid = packed_cell_local_id(mesh, other_id);
                const auto other_value = field.local_value(other_lid);
                const auto cell_distance = mesh.cell_to_face_distance(face_id, cell_id);
                const auto other_distance = mesh.cell_to_face_distance(face_id, other_id);
                const auto total_distance = cell_distance + other_distance;
                face_value = total_distance > scalar_type{}
                                 ? (cell_value * other_distance + other_value * cell_distance) / total_distance
                                 : (cell_value + other_value) / scalar_type{2};
            }
            else if (mesh.is_boundary_face(face_id))
            {
                const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                if (location.active)
                {
                    face_value = boundary_value(location.batch_id, location.in_batch_id);
                }
            }
            const auto area = mesh.face_area_vector_outward(face_id, cell_id);
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + area * face_value.component(component);
            }
        }
        const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(mesh.cell_volume(cell_id));
        for (auto& component_gradient : gradient)
        {
            component_gradient = component_gradient * inverse_volume;
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/** @brief Net owner-oriented stored face flux around one cell. */
template<TpetraTypePack Pack, class MeshType>
auto stored_cell_flux_balance(const MeshType& mesh, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid) -> typename Pack::scalar_type
{
    using scalar_type = typename Pack::scalar_type;

    scalar_type balance{};
    const auto cell_id = query_cell_id(mesh, cell_lid);
    for (const auto face_id : mesh.faces(cell_id))
    {
        const auto face_lid = static_cast<typename Pack::local_ordinal_type>(packed_face_local_id(mesh, face_id));
        const auto owner_lid = packed_cell_local_id(mesh, mesh.owner_cell(face_id));
        const auto sign = owner_lid == cell_lid ? scalar_type{1} : scalar_type{-1};
        balance += sign * face_fluxes.local_value(face_lid);
    }
    return balance;
}

} // namespace SimpleFluid::FVM::detail
