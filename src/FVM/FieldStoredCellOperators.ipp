/** @brief Caller-defined mapped gradient callbacks. */
#pragma once

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
    requires requires(
        BoundaryConditionProvider condition, BoundaryValueProvider value, int batch_id, size_t in_batch_id) {
        { condition(batch_id, in_batch_id) } -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) } -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field, BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_scalar_cell_gradient(field, boundary_condition, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
    requires requires(
        BoundaryConditionProvider condition, BoundaryValueProvider value, int batch_id, size_t in_batch_id) {
        { condition(batch_id, in_batch_id) } -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) } -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field, BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache)
{
    detail::stored_cached_scalar_cell_gradient(field, boundary_condition, boundary_value, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider, int batch_id, size_t in_batch_id) {
        {
            provider(batch_id, in_batch_id)
        } -> std::convertible_to<typename VectorCellFieldStored<Pack, MeshType>::value_type>;
    }
void cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field, BoundaryValueProvider boundary_value,
    TensorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_vector_cell_gradient(field, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider, int batch_id, size_t in_batch_id) {
        {
            provider(batch_id, in_batch_id)
        } -> std::convertible_to<typename VectorCellFieldStored<Pack, MeshType>::value_type>;
    }
void cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field, BoundaryValueProvider boundary_value,
    TensorCellFieldStored<Pack, MeshType>& gradients, const CellGradientCache<Pack, MeshType>& cache)
{
    detail::stored_cached_vector_cell_gradient(field, boundary_value, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
void gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_gauss_linear_cell_gradient(field, boundary_condition, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
void gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_gauss_linear_cell_gradient(field, boundary_value, gradients);
}

} // namespace SimpleFluid::FVM
