/** @file FVM/details/TransportRhs.ipp
 * @brief Inline cell and boundary updates for basic transport RHS adapters.
 */
#pragma once

namespace SimpleFluid::FVM::detail
{

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline BasicVectorTransportRhs<Pack, MeshType>::BasicVectorTransportRhs(
    const VectorCellFieldStored<Pack, MeshType>& field,
    const VectorBoundaryConditionProvider<Pack>& condition, const VectorBoundaryValueProvider<Pack>& boundary_value,
    const VectorCellValueProvider<Pack>& source)
    : field(field), condition_provider(condition), boundary_provider(boundary_value), source_provider(source)
{
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void BasicVectorTransportRhs<Pack, MeshType>::begin_cell(local_ordinal_type cell, scalar_type, scalar_type)
{
    cell_source = source_provider(cell);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void BasicVectorTransportRhs<Pack, MeshType>::finish_cell(
    local_ordinal_type cell, scalar_type transient, scalar_type volume) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->replaceLocalValue(
            cell, comp, transient * (*old_data)(cell, comp) + volume * cell_source.component(comp));
    }
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline BoundaryConditionType BasicVectorTransportRhs<Pack, MeshType>::begin_boundary(int batch, size_t face)
{
    boundary_value = boundary_provider(batch, face);
    condition = condition_provider(batch, face);
    return condition.type;
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void BasicVectorTransportRhs<Pack, MeshType>::add_boundary_value(
    local_ordinal_type owner, scalar_type factor) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->sumIntoLocalValue(owner, comp, factor * boundary_value.component(comp));
    }
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
template<class Metrics>
inline void BasicVectorTransportRhs<Pack, MeshType>::add_neumann(
    local_ordinal_type owner, scalar_type diffusivity, const Metrics& metrics, local_ordinal_type face) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->sumIntoLocalValue(owner, comp, diffusivity * condition.value.component(comp) * metrics.face_area(face));
    }
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void MultiColumnWeightedTransportRhs<Pack, MeshType>::observe_value(std::size_t column, scalar_type value)
{
    if (!std::isfinite(value))
    {
        local_max_abs[column] = std::numeric_limits<double>::infinity();
        return;
    }
    const double abs_val = static_cast<double>(std::abs(value));
    if (std::isfinite(local_max_abs[column]) && abs_val > local_max_abs[column])
    {
        local_max_abs[column] = abs_val;
    }
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void MultiColumnWeightedTransportRhs<Pack, MeshType>::set_cell_value(
    local_ordinal_type cell, std::size_t column, scalar_type value)
{
    values->replaceLocalValue(cell, column, value);
    observe_value(column, value);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
inline void MultiColumnWeightedTransportRhs<Pack, MeshType>::add_cell_value(
    local_ordinal_type cell, std::size_t column, scalar_type value) const
{
    values->sumIntoLocalValue(cell, column, value);
}

} // namespace SimpleFluid::FVM::detail
