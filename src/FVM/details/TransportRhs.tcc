/** @file FVM/details/TransportRhs.tcc
 * @brief Allocation and error paths for basic transport RHS adapters.
 */
#pragma once
#include "utils/export_helper.hh"

#include "FVM/details/TransportRhs.hh"

#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

// Allocate temporary RHS/view storage during local staging, before cached-matrix mutation.
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
void BasicVectorTransportRhs<Pack, MeshType>::initialize(const MeshType& mesh)
{
    values = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), num_components, true));
    old_data.emplace(field.owned_read_view());
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
[[noreturn]] void BasicVectorTransportRhs<Pack, MeshType>::throw_robin() const
{
    throw std::runtime_error("Robin boundary conditions are not yet implemented in vector transport_system.");
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
void MultiColumnWeightedTransportRhs<Pack, MeshType>::initialize(
    const MeshType& mesh, std::size_t num_columns, Teuchos::RCP<multi_vector_type> cached_rhs)
{
    values = cached_rhs.is_null()
        ? Teuchos::rcp(new multi_vector_type(mesh.owned_cell_map(), num_columns, true))
        : std::move(cached_rhs);
    values->putScalar(scalar_type{});
    local_max_abs.assign(num_columns, 0.0);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
std::vector<bool> MultiColumnWeightedTransportRhs<Pack, MeshType>::finalize_active_columns(const MeshType& mesh)
{
    const std::size_t num_columns = local_max_abs.size();
    const std::size_t owned_cells = mesh.num_owned_cells();
    for (std::size_t column = 0; column < num_columns; ++column)
    {
        const auto data = values->getData(column);
        for (std::size_t row = 0; row < owned_cells; ++row)
        {
            observe_value(column, data[row]);
        }
    }
    std::vector<double> global_max_abs = local_max_abs;
    const auto comm = mesh.owned_cell_map()->getComm();
    if (comm->getSize() > 1 && num_columns > 0)
    {
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(num_columns),
            local_max_abs.data(), global_max_abs.data());
    }
    std::vector<bool> active(num_columns, false);
    for (std::size_t column = 0; column < num_columns; ++column)
    {
        active[column] = (global_max_abs[column] > 0.0) || !std::isfinite(global_max_abs[column]);
    }
    return active;
}

} // namespace SimpleFluid::FVM::detail
