/** @file FVM/details/TransportRhs.tcc
 * @brief Allocation and error paths for basic transport RHS adapters.
 */
#pragma once

#include "FVM/details/TransportRhs.hh"

#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

// Retain allocation and view acquisition after preflight and matrix setup.
template<TpetraTypePack Pack> void BasicScalarTransportRhs<Pack>::initialize(const Mesh<Pack>& mesh)
{
    values = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    old_data.emplace(field.owned_read_view());
}

template<TpetraTypePack Pack> [[noreturn]] void BasicScalarTransportRhs<Pack>::throw_robin() const
{
    throw std::runtime_error("Robin boundary conditions are not yet implemented in transport_system.");
}

template<TpetraTypePack Pack> void BasicVectorTransportRhs<Pack>::initialize(const Mesh<Pack>& mesh)
{
    values = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), num_components, true));
    old_data.emplace(field.owned_read_view());
}

template<TpetraTypePack Pack> [[noreturn]] void BasicVectorTransportRhs<Pack>::throw_robin() const
{
    throw std::runtime_error("Robin boundary conditions are not yet implemented in vector transport_system.");
}

} // namespace SimpleFluid::FVM::detail
