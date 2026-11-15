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

} // namespace SimpleFluid::FVM::detail
