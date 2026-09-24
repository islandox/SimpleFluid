#pragma once

#include "FVM/details/OperatorDetails.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FVM::detail
{

template<class MeshType, size_t StateCount>
std::array<int, StateCount> reduce_transport_validation_state(
    const MeshType& mesh, const std::array<int, StateCount>& local_state)
{
    auto global_state = local_state;
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
        Teuchos::reduceAll(
            *communicator, Teuchos::REDUCE_MAX, static_cast<int>(StateCount), local_state.data(), global_state.data());
    return global_state;
}

template<TpetraTypePack Pack> struct TransportMatrixRow
{
    std::vector<typename Pack::local_ordinal_type> columns;
    std::vector<typename Pack::scalar_type> values;
};

template<TpetraTypePack Pack>
TransportMatrixRow<Pack> capture_transport_row(
    const FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values)
{
    TransportMatrixRow<Pack> row;
    row.columns.assign(row_values.column_data(), row_values.column_data() + row_values.size());
    row.values.assign(row_values.value_data(), row_values.value_data() + row_values.size());
    return row;
}

// All ranks must enter the same fixed-shape reductions. Map::isSameAs may
// short-circuit on one rank and reduce on another, even for equivalent maps.
template<TpetraTypePack Pack, class MeshType>
bool compatible_present_transport_matrix_maps(
    const MeshType& mesh, const typename Pack::matrix_type& matrix, size_t row_count)
{
    const auto row = matrix.getRowMap();
    const auto col = matrix.getColMap();
    const auto domain = matrix.getDomainMap();
    const auto range = matrix.getRangeMap();
    const auto structural = reduce_transport_validation_state(
        mesh, std::array<int, 1>{!matrix.isFillComplete() || row.is_null() || col.is_null() || domain.is_null() ||
                                         range.is_null() || row_count != mesh.num_owned_cells()
                                     ? 1
                                     : 0});
    if (structural[0])
        return false;

    const auto locally_same = [](const auto& candidate, const auto& expected)
    {
        return candidate->getComm()->getSize() == expected->getComm()->getSize() &&
               Tpetra::Details::congruent(*candidate->getComm(), *expected->getComm()) &&
               candidate->getGlobalNumElements() == expected->getGlobalNumElements() &&
               candidate->getIndexBase() == expected->getIndexBase() && candidate->locallySameAs(*expected);
    };
    int invalid = 0;
    try
    {
        invalid = !locally_same(row, mesh.owned_cell_map()) || !locally_same(col, mesh.overlap_cell_map()) ||
                  !locally_same(domain, mesh.owned_cell_map()) || !locally_same(range, mesh.owned_cell_map());
    }
    catch (const std::exception&)
    {
        invalid = 1;
    }
    return reduce_transport_validation_state(mesh, std::array<int, 1>{invalid})[0] == 0;
}

template<TpetraTypePack Pack, class MeshType>
void validate_transport_matrix_maps(
    const MeshType& mesh, const Teuchos::RCP<typename Pack::matrix_type>& matrix, size_t row_count)
{
    const int present = matrix.is_null() ? 0 : 1;
    const auto presence = reduce_transport_validation_state(mesh, std::array<int, 2>{present, -present});
    if (presence[0] != -presence[1])
        throw std::invalid_argument("transport_system requires every rank to use the same cached-matrix category.");
    if (present && !compatible_present_transport_matrix_maps<Pack>(mesh, *matrix, row_count))
        throw std::invalid_argument("transport_system cached matrix is incompatible with the mesh.");
}

template<class MeshType, class Scalar>
void validate_transport_scalar_inputs(
    const MeshType& mesh, bool face_mesh_matches, Scalar time_step, Scalar diffusivity)
{
    const auto invalid = reduce_transport_validation_state(
        mesh, std::array<int, 3>{face_mesh_matches ? 0 : 1, !std::isfinite(time_step) || time_step <= Scalar{} ? 1 : 0,
                  !std::isfinite(diffusivity) || diffusivity < Scalar{} ? 1 : 0});
    if (invalid[0])
        throw std::invalid_argument("transport_system requires face fluxes on the old-value mesh.");
    if (invalid[1])
        throw std::invalid_argument("transport_system requires a finite positive time step.");
    if (invalid[2])
        throw std::invalid_argument("transport_system requires finite non-negative diffusivity.");
}

} // namespace SimpleFluid::FVM::detail
