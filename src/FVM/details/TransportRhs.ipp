/** @file FVM/details/TransportRhs.ipp
 * @brief Inline cell and boundary updates for basic transport RHS adapters.
 */
#pragma once

namespace SimpleFluid::FVM::detail
{

template<TpetraTypePack Pack>
inline BasicScalarTransportRhs<Pack>::BasicScalarTransportRhs(const CellField<Pack>& field,
    const ScalarBoundaryConditionProvider<Pack>& condition, const ScalarBoundaryValueProvider<Pack>& boundary_value,
    const ScalarCellValueProvider<Pack>& source)
    : field(field), condition_provider(condition), boundary_provider(boundary_value), source_provider(source)
{
}

template<TpetraTypePack Pack>
inline void BasicScalarTransportRhs<Pack>::begin_cell(
    local_ordinal_type cell, scalar_type transient, scalar_type volume)
{
    const auto old_value = (*old_data)(cell, 0);
    cell_rhs = transient * old_value + volume * source_provider(cell);
}

template<TpetraTypePack Pack>
inline void BasicScalarTransportRhs<Pack>::finish_cell(local_ordinal_type cell, scalar_type, scalar_type) const
{
    values->replaceLocalValue(cell, cell_rhs);
}

template<TpetraTypePack Pack>
inline BoundaryConditionType BasicScalarTransportRhs<Pack>::begin_boundary(int batch, size_t face)
{
    boundary_value = boundary_provider(batch, face);
    condition = condition_provider(batch, face);
    return condition.type;
}

template<TpetraTypePack Pack>
inline void BasicScalarTransportRhs<Pack>::add_boundary_value(local_ordinal_type owner, scalar_type factor) const
{
    values->sumIntoLocalValue(owner, factor * boundary_value);
}

template<TpetraTypePack Pack>
template<class Metrics>
inline void BasicScalarTransportRhs<Pack>::add_neumann(
    local_ordinal_type owner, scalar_type diffusivity, const Metrics& metrics, local_ordinal_type face) const
{
    values->sumIntoLocalValue(owner, diffusivity * condition.value * metrics.face_area(face));
}

template<TpetraTypePack Pack>
inline BasicVectorTransportRhs<Pack>::BasicVectorTransportRhs(const VectorCellField<Pack>& field,
    const VectorBoundaryConditionProvider<Pack>& condition, const VectorBoundaryValueProvider<Pack>& boundary_value,
    const VectorCellValueProvider<Pack>& source)
    : field(field), condition_provider(condition), boundary_provider(boundary_value), source_provider(source)
{
}

template<TpetraTypePack Pack>
inline void BasicVectorTransportRhs<Pack>::begin_cell(local_ordinal_type cell, scalar_type, scalar_type)
{
    cell_source = source_provider(cell);
}

template<TpetraTypePack Pack>
inline void BasicVectorTransportRhs<Pack>::finish_cell(
    local_ordinal_type cell, scalar_type transient, scalar_type volume) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->replaceLocalValue(
            cell, comp, transient * (*old_data)(cell, comp) + volume * cell_source.component(comp));
    }
}

template<TpetraTypePack Pack>
inline BoundaryConditionType BasicVectorTransportRhs<Pack>::begin_boundary(int batch, size_t face)
{
    boundary_value = boundary_provider(batch, face);
    condition = condition_provider(batch, face);
    return condition.type;
}

template<TpetraTypePack Pack>
inline void BasicVectorTransportRhs<Pack>::add_boundary_value(local_ordinal_type owner, scalar_type factor) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->sumIntoLocalValue(owner, comp, factor * boundary_value.component(comp));
    }
}

template<TpetraTypePack Pack>
template<class Metrics>
inline void BasicVectorTransportRhs<Pack>::add_neumann(
    local_ordinal_type owner, scalar_type diffusivity, const Metrics& metrics, local_ordinal_type face) const
{
    for (size_t comp = 0; comp < num_components; ++comp)
    {
        values->sumIntoLocalValue(owner, comp, diffusivity * condition.value.component(comp) * metrics.face_area(face));
    }
}

} // namespace SimpleFluid::FVM::detail
