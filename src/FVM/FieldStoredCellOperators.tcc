/** @brief Compiled mapped cell operators and custom-provider opt-in. */
#pragma once

#include "FVM/CellOperators.hh"
#include "FVM/details/FieldStoredCellOperators.tcc"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_scalar_cell_gradient(field, gradients);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const CellGradientCache<Pack, MeshType>& cache)
{
    detail::stored_cached_scalar_cell_gradient(field, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end() ? BoundaryCondition{} : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    { return static_cast<typename Pack::scalar_type>(boundary_condition(batch_id, in_batch_id).value); };
    detail::stored_scalar_cell_gradient(field, boundary_condition, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache)
{
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end() ? BoundaryCondition{} : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    { return static_cast<typename Pack::scalar_type>(boundary_condition(batch_id, in_batch_id).value); };
    detail::stored_cached_scalar_cell_gradient(field, boundary_condition, boundary_value, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field, TensorCellFieldStored<Pack, MeshType>& gradients)
{
    detail::stored_vector_cell_gradient(field, gradients);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const CellGradientCache<Pack, MeshType>& cache)
{
    detail::stored_cached_vector_cell_gradient(field, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end() ? BoundaryCondition{} : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    { return static_cast<typename Pack::scalar_type>(boundary_condition(batch_id, in_batch_id).value); };
    detail::stored_gauss_linear_cell_gradient(field, boundary_condition, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void gauss_linear_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients)
{
    auto boundary_condition = [](int, size_t) { return BoundaryCondition{}; };
    auto boundary_value = [](int, size_t) { return typename Pack::scalar_type{}; };
    detail::stored_gauss_linear_cell_gradient(field, boundary_condition, boundary_value, gradients);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache, CellGradientScheme scheme)
{
    if (scheme == CellGradientScheme::GaussLinear)
    {
        gauss_linear_cell_gradient(field, boundary_conditions, gradients);
        return;
    }
    cell_gradient(field, boundary_conditions, gradients, cache);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT std::vector<typename Pack::scalar_type> cell_divergence_from_fluxes(
    const MeshType& mesh, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    if (&mesh != &face_fluxes.mesh())
    {
        throw std::invalid_argument("cell_divergence_from_fluxes requires the face field mesh.");
    }
    std::vector<scalar_type> divergence(mesh.num_owned_cells());
    const auto face_values = face_fluxes.local_read_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = detail::query_cell_id(mesh, cell_lid);
        divergence[owned] = detail::stored_cell_flux_balance(mesh, face_fluxes, face_values, cell_lid) /
                            static_cast<scalar_type>(mesh.cell_volume(cell_id));
    }
    return divergence;
}

} // namespace SimpleFluid::FVM
