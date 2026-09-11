/**
 * @file FVM/details/FieldStoredCellOperators.hh
 * @brief Internal cell-operator kernels for mesh-aware stored fields.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/ResolvedDiffusionGeometry.hh"
#include "fields/FieldStored.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        vec_type gradient{};
        for (const auto& entry : stencils[owned])
        {
            gradient = gradient + entry.coefficient * local_values(entry.cell_lid, 0);
        }
        gradient_values(static_cast<local_ordinal_type>(owned), 0) = gradient.x;
        gradient_values(static_cast<local_ordinal_type>(owned), 1) = gradient.y;
        gradient_values(static_cast<local_ordinal_type>(owned), 2) = gradient.z;
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
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        auto gradient = stencils[owned].constant;
        for (const auto& entry : stencils[owned].entries)
        {
            gradient = gradient + entry.coefficient * local_values(entry.cell_lid, 0);
        }
        gradient_values(static_cast<local_ordinal_type>(owned), 0) = gradient.x;
        gradient_values(static_cast<local_ordinal_type>(owned), 1) = gradient.y;
        gradient_values(static_cast<local_ordinal_type>(owned), 2) = gradient.z;
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
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = owned_values(cell_lid, 0);
        vec_type gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            gradient = gradient + sample.weight * (local_values(sample.other_lid, 0) - value_p);
        }
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
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
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < boundary_geometry.size(); ++owned)
    {
        const auto& geometry =
            boundary_geometry[owned].boundary_samples.empty() ? interior_geometry[owned] : boundary_geometry[owned];
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = owned_values(cell_lid, 0);
        vec_type gradient{};
        for (const auto& sample : geometry.interior_samples)
        {
            gradient = gradient + sample.weight * (local_values(sample.other_lid, 0) - value_p);
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
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Evaluate interior vector-gradient stencils into tensor storage. */
template<TpetraTypePack Pack, class MeshType>
void stored_vector_cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = least_squares_gradient_stencils(field.mesh());
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        tensor_type gradient{};
        for (const auto& entry : stencils[owned])
        {
            const auto value = vec_type{
                local_values(entry.cell_lid, 0), local_values(entry.cell_lid, 1), local_values(entry.cell_lid, 2)};
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + entry.coefficient * value.component(component);
            }
        }
        for (size_t component = 0; component < 3; ++component)
        {
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 0) = gradient[component].x;
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 1) = gradient[component].y;
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 2) = gradient[component].z;
        }
    }
}

/** @brief Evaluate boundary-aware vector-gradient stencils. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    const auto stencils = vector_affine_gradient_stencils(field.mesh(), boundary_value);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        tensor_type gradient = stencils[owned].constants;
        for (const auto& entry : stencils[owned].entries)
        {
            const auto value = vec_type{
                local_values(entry.cell_lid, 0), local_values(entry.cell_lid, 1), local_values(entry.cell_lid, 2)};
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + entry.coefficient * value.component(component);
            }
        }
        for (size_t component = 0; component < 3; ++component)
        {
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 0) = gradient[component].x;
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 1) = gradient[component].y;
            gradient_values(static_cast<local_ordinal_type>(owned), component * 3 + 2) = gradient[component].z;
        }
    }
}

/** @brief Evaluate cached interior vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache>
void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& geometry = cache.interior_geometry();
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = vec_type{owned_values(cell_lid, 0), owned_values(cell_lid, 1), owned_values(cell_lid, 2)};
        tensor_type gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            const auto delta = vec_type{local_values(sample.other_lid, 0), local_values(sample.other_lid, 1),
                                   local_values(sample.other_lid, 2)} -
                               value_p;
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] + sample.weight * delta.component(component);
            }
        }
        for (size_t component = 0; component < 3; ++component)
        {
            gradient_values(cell_lid, component * 3 + 0) = gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
        }
    }
}

/** @brief Evaluate cached boundary-aware vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache, class BoundaryValueProvider>
void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto& interior_geometry = cache.interior_geometry();
    const auto& boundary_geometry = cache.boundary_geometry();
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < boundary_geometry.size(); ++owned)
    {
        const auto& geometry =
            boundary_geometry[owned].boundary_samples.empty() ? interior_geometry[owned] : boundary_geometry[owned];
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = vec_type{owned_values(cell_lid, 0), owned_values(cell_lid, 1), owned_values(cell_lid, 2)};
        tensor_type gradient{};
        for (const auto& sample : geometry.interior_samples)
        {
            const auto delta = vec_type{local_values(sample.other_lid, 0), local_values(sample.other_lid, 1),
                                   local_values(sample.other_lid, 2)} -
                               value_p;
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
        for (size_t component = 0; component < 3; ++component)
        {
            gradient_values(cell_lid, component * 3 + 0) = gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
        }
    }
}

/** @brief Gauss-linear scalar gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient_reference(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    const auto boundary_locations = boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        const auto cell_value = local_values(cell_lid, 0);
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
                        ? (other_distance * cell_value + cell_distance * local_values(other_lid, 0)) / total_distance
                        : scalar_type{0.5} * (cell_value + local_values(other_lid, 0));
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
        gradient = gradient / static_cast<scalar_type>(mesh.cell_volume(cell_id));
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Gauss-linear vector gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient_reference(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    const auto boundary_locations = boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        const auto cell_value =
            vec_type{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
        tensor_type gradient{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            auto face_value = cell_value;
            if (mesh.is_interior_face(face_id))
            {
                const auto other_id = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                const auto other_lid = packed_cell_local_id(mesh, other_id);
                const auto other_value =
                    vec_type{local_values(other_lid, 0), local_values(other_lid, 1), local_values(other_lid, 2)};
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
        for (size_t component = 0; component < 3; ++component)
        {
            gradient_values(cell_lid, component * 3 + 0) = gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
        }
    }
}

/** @brief Gauss-linear scalar reconstruction through borrowed region geometry. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_region_gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    const auto boundary_locations = boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    mesh.visit_owned_region_cells(
        [&](local_ordinal_type cell_lid, auto canonical_cell, auto native_cell, const auto& region)
        {
            // Gauss reads the published overlap center, just like the generic path.
            const auto cell_value = local_values(cell_lid, 0);
            vec_type gradient{};
            region.visit_cell_faces(native_cell,
                [&](const auto& face)
                {
                    scalar_type face_value = cell_value;
                    if (face.interior())
                    {
                        const auto other_id = face.opposite_cell(canonical_cell);
                        const auto other_lid = mesh.region_cell_local_id(other_id);
                        if (other_lid == mesh.invalid_local_id())
                            throw std::logic_error("Region gradient requires the adjacent cell in the existing halo.");
                        const auto cell_distance = face.face_center_vector(canonical_cell).norm();
                        const auto other_distance = face.face_center_vector(other_id).norm();
                        const auto total_distance = cell_distance + other_distance;
                        face_value = total_distance > scalar_type{}
                                         ? (other_distance * cell_value + cell_distance * local_values(other_lid, 0)) /
                                               total_distance
                                         : scalar_type{0.5} * (cell_value + local_values(other_lid, 0));
                    }
                    else
                    {
                        const auto face_lid = mesh.region_face_local_id(face.face);
                        const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                        if (location.active)
                        {
                            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                            if (condition.type == BoundaryConditionType::Dirichlet)
                                face_value = boundary_value(location.batch_id, location.in_batch_id);
                            else if (condition.type == BoundaryConditionType::Neumann)
                            {
                                const ResolvedDiffusionGeometry<std::remove_cvref_t<decltype(face)>, scalar_type>
                                    geometry(face);
                                face_value = cell_value + static_cast<scalar_type>(condition.value) *
                                                              static_cast<scalar_type>(boundary_normal_distance(
                                                                  geometry, face.face, canonical_cell));
                            }
                            else
                                throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                            "Dirichlet and Neumann boundaries.");
                        }
                    }
                    gradient = gradient + face.face_area_vector_outward(canonical_cell) * face_value;
                });
            gradient = gradient / static_cast<scalar_type>(region.cell_volume(native_cell));
            gradient_values(cell_lid, 0) = gradient.x;
            gradient_values(cell_lid, 1) = gradient.y;
            gradient_values(cell_lid, 2) = gradient.z;
        });
}

/** @brief Gauss-linear vector reconstruction through borrowed region geometry. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_region_gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "gauss_linear_cell_gradient");
    const auto& mesh = field.mesh();
    const auto execution = acquire_mesh_execution(mesh);
    const auto boundary_locations = boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    mesh.visit_owned_region_cells(
        [&](local_ordinal_type cell_lid, auto canonical_cell, auto native_cell, const auto& region)
        {
            const vec_type cell_value{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
            tensor_type gradient{};
            region.visit_cell_faces(native_cell,
                [&](const auto& face)
                {
                    auto face_value = cell_value;
                    if (face.interior())
                    {
                        const auto other_id = face.opposite_cell(canonical_cell);
                        const auto other_lid = mesh.region_cell_local_id(other_id);
                        if (other_lid == mesh.invalid_local_id())
                            throw std::logic_error("Region gradient requires the adjacent cell in the existing halo.");
                        const vec_type other_value{
                            local_values(other_lid, 0), local_values(other_lid, 1), local_values(other_lid, 2)};
                        const auto cell_distance = face.face_center_vector(canonical_cell).norm();
                        const auto other_distance = face.face_center_vector(other_id).norm();
                        const auto total_distance = cell_distance + other_distance;
                        face_value = total_distance > scalar_type{}
                                         ? (cell_value * other_distance + other_value * cell_distance) / total_distance
                                         : (cell_value + other_value) / scalar_type{2};
                    }
                    else
                    {
                        const auto face_lid = mesh.region_face_local_id(face.face);
                        const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
                        if (location.active)
                            face_value = boundary_value(location.batch_id, location.in_batch_id);
                    }
                    const auto area = face.face_area_vector_outward(canonical_cell);
                    for (size_t component = 0; component < 3; ++component)
                        gradient[component] = gradient[component] + area * face_value.component(component);
                });
            const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(region.cell_volume(native_cell));
            for (auto& component_gradient : gradient)
                component_gradient = component_gradient * inverse_volume;
            for (size_t component = 0; component < 3; ++component)
            {
                gradient_values(cell_lid, component * 3) = gradient[component].x;
                gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
                gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
            }
        });
}

/** @brief Select region geometry while retaining the checked generic fallback. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    if constexpr (requires { field.mesh().supports_region_execution(); })
        if (field.mesh().supports_region_execution())
            return stored_region_gauss_linear_cell_gradient(
                field, std::move(boundary_condition), std::move(boundary_value), gradients);
    stored_gauss_linear_cell_gradient_reference(
        field, std::move(boundary_condition), std::move(boundary_value), gradients);
}

/** @brief Select region geometry for vector gradients when available. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void stored_gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    if constexpr (requires { field.mesh().supports_region_execution(); })
        if (field.mesh().supports_region_execution())
            return stored_region_gauss_linear_cell_gradient(field, std::move(boundary_value), gradients);
    stored_gauss_linear_cell_gradient_reference(field, std::move(boundary_value), gradients);
}

/** @brief Net owner-oriented flux using a scoped overlap or owned face view. */
template<TpetraTypePack Pack, class MeshType, class View>
auto stored_cell_flux_balance(const MeshType& mesh, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    const View& face_values, typename Pack::local_ordinal_type cell_lid) -> typename Pack::scalar_type
{
    using scalar_type = typename Pack::scalar_type;
    const bool local_view = face_values.extent(0) == face_fluxes.num_local_entries();
    if (face_values.extent(1) != 1 ||
        (!local_view && face_values.extent(0) != face_fluxes.num_owned_entries()))
        throw std::invalid_argument("cell_flux_balance requires an owned or overlap scalar face view.");
    // Owned-only compatibility callers still need published processor faces.
    const auto overlap = local_view ? decltype(face_fluxes.local_read_view()){} : face_fluxes.local_read_view();
    scalar_type balance{};
    const auto cell_id = query_cell_id(mesh, cell_lid);
    for (const auto face_id : mesh.faces(cell_id))
    {
        const auto face_lid = static_cast<typename Pack::local_ordinal_type>(packed_face_local_id(mesh, face_id));
        const auto owner_lid = packed_cell_local_id(mesh, mesh.owner_cell(face_id));
        const auto sign = owner_lid == cell_lid ? scalar_type{1} : scalar_type{-1};
        const auto value = local_view || static_cast<size_t>(face_lid) < face_fluxes.num_owned_entries()
            ? face_values(face_lid, 0) : overlap(face_lid, 0);
        balance += sign * value;
    }
    return balance;
}

/** @brief Net owner-oriented stored face flux around one cell. */
template<TpetraTypePack Pack, class MeshType>
auto stored_cell_flux_balance(const MeshType& mesh, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid) -> typename Pack::scalar_type
{
    return stored_cell_flux_balance(mesh, face_fluxes, face_fluxes.local_read_view(), cell_lid);
}

} // namespace SimpleFluid::FVM::detail
