/** @file FieldStoredCellOperators.tcc
 * @brief Compiled stored-field implementation kernels.
 */
#pragma once

#include "FVM/details/FieldStoredCellOperators.hh"

namespace SimpleFluid::FVM::detail
{

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
        const auto gradient = apply_stored_scalar_gradient<vec_type>(stencils[owned], local_values);
        gradient_values(static_cast<local_ordinal_type>(owned), 0) = gradient.x;
        gradient_values(static_cast<local_ordinal_type>(owned), 1) = gradient.y;
        gradient_values(static_cast<local_ordinal_type>(owned), 2) = gradient.z;
    }
}

template<TpetraTypePack Pack, class MeshType, class Cache>
void stored_cached_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    visit_cached_gradient_rows<false>(cache,
        [&](local_ordinal_type cell_lid, const auto& geometry)
        {
            const auto value_p = owned_values(cell_lid, 0);
            vec_type gradient{};
            for (const auto& sample : geometry.interior_samples)
            {
                gradient = gradient + sample.weight * (local_values(sample.other_lid, 0) - value_p);
            }
            gradient_values(cell_lid, 0) = gradient.x;
            gradient_values(cell_lid, 1) = gradient.y;
            gradient_values(cell_lid, 2) = gradient.z;
        });
}

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

template<TpetraTypePack Pack, class MeshType, class Cache>
void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::value_type;

    require_same_stored_mesh(field, gradients, "cell_gradient");
    cache.require_mesh(field.mesh());
    const auto local_values = field.local_read_view();
    const auto owned_values = field.owned_read_view();
    auto gradient_values = gradients.owned_write_view();
    visit_cached_gradient_rows<false>(cache,
        [&](local_ordinal_type cell_lid, const auto& geometry)
        {
            const auto value_p =
                vec_type{owned_values(cell_lid, 0), owned_values(cell_lid, 1), owned_values(cell_lid, 2)};
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
            for (size_t component = 0; component < 3; ++component)
            {
                gradient_values(cell_lid, component * 3 + 0) = gradient[component].x;
                gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
                gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
            }
        });
}

} // namespace SimpleFluid::FVM::detail
