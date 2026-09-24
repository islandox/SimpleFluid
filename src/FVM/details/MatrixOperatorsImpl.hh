/**
 * @file FVM/details/MatrixOperatorsImpl.hh
 * @brief Mesh-generic implementation kernels for FVM matrix operators.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/FaceStencilMatrix.hh"
#include "FVM/details/TransportValidation.hh"
#include "dataclass/TpetraTypes.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <optional>
#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

/** @brief Assemble the mesh-generic two-point diffusion matrix. */
template<TpetraTypePack Pack, class MeshType>
Teuchos::RCP<typename Pack::matrix_type> diffusion_matrix_impl(
    const MeshType& mesh, typename Pack::scalar_type diffusivity)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = make_face_stencil_matrix<Pack>(mesh);
    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(32);
    values.reserve(32);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        columns.clear();
        values.clear();
        scalar_type diagonal{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            if (!mesh.is_interior_face(face_id))
            {
                continue;
            }
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            const auto other_lid =
                packed_cell_local_id(mesh, mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id));
            const auto coefficient = interior_diffusion_coefficient(mesh, face_lid, cell_lid, other_lid, diffusivity);
            diagonal += coefficient;
            columns.push_back(other_lid);
            values.push_back(-coefficient);
        }
        columns.push_back(cell_lid);
        values.push_back(diagonal);
        matrix->insertLocalValues(cell_lid, columns(), values());
    }
    matrix->fillComplete();
    return matrix;
}

/** @brief Assemble upwind convection from a generic face-flux accessor. */
template<TpetraTypePack Pack, class MeshType, class IsOwnedFace, class FaceValue>
Teuchos::RCP<typename Pack::matrix_type> upwind_convection_matrix_impl(
    const MeshType& mesh, IsOwnedFace is_owned_face, FaceValue face_value)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = make_face_stencil_matrix<Pack>(mesh);
    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(32);
    values.reserve(32);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        columns.clear();
        values.clear();
        scalar_type diagonal{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            const auto owner_flux =
                is_owned_face(face_lid) ? static_cast<scalar_type>(face_value(face_lid)) : scalar_type{};
            const auto owner_lid = packed_cell_local_id(mesh, mesh.owner_cell(face_id));
            const auto outward_flux = owner_lid == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                diagonal += outward_flux;
            }
            else if (mesh.is_interior_face(face_id))
            {
                const auto other_lid =
                    packed_cell_local_id(mesh, mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id));
                columns.push_back(other_lid);
                values.push_back(outward_flux);
            }
        }
        columns.push_back(cell_lid);
        values.push_back(diagonal);
        matrix->insertLocalValues(cell_lid, columns(), values());
    }
    matrix->fillComplete();
    return matrix;
}

/** @brief Visit pressure rows in the same face order for fresh and reused graphs. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class RowVisitor>
void visit_pressure_poisson_rows(const MeshType& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid,
    BoundaryConditionProvider boundary_condition, RowVisitor visit_row)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(32);
    values.reserve(32);
    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_id = query_cell_id(mesh, cell_lid);
        const auto row_gid = mesh.owned_cell_map()->getGlobalElement(cell_lid);
        columns.clear();
        values.clear();
        if (gauge_cell_gid && row_gid == *gauge_cell_gid)
        {
            columns.push_back(cell_lid);
            values.push_back(scalar_type{1});
            visit_row(cell_lid, columns, values);
            continue;
        }

        scalar_type diagonal{};
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = static_cast<local_ordinal_type>(packed_face_local_id(mesh, face_id));
            if (mesh.is_interior_face(face_id))
            {
                const auto other_lid =
                    packed_cell_local_id(mesh, mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id));
                const auto coefficient =
                    interior_diffusion_coefficient(mesh, face_lid, cell_lid, other_lid, scalar_type{1});
                diagonal += coefficient;
                columns.push_back(other_lid);
                values.push_back(-coefficient);
                continue;
            }
            if (!mesh.is_boundary_face(face_id))
            {
                continue;
            }
            const auto location = boundary_locations.at(static_cast<size_t>(face_lid));
            if (!location.active)
            {
                continue;
            }
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                diagonal += boundary_diffusion_coefficient(mesh, face_lid, cell_lid, scalar_type{1});
            }
            else if (condition.type != BoundaryConditionType::Neumann)
            {
                throw std::invalid_argument("pressure_poisson_matrix supports only Dirichlet and "
                                            "Neumann pressure boundary conditions.");
            }
        }
        columns.push_back(cell_lid);
        values.push_back(diagonal > scalar_type{} ? diagonal : scalar_type{1});
        visit_row(cell_lid, columns, values);
    }
}

/** @brief Assemble pressure Poisson on a generic mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider>
Teuchos::RCP<typename Pack::matrix_type> pressure_poisson_matrix_impl(const MeshType& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid, BoundaryConditionProvider boundary_condition)
{
    auto matrix = make_face_stencil_matrix<Pack>(mesh);
    visit_pressure_poisson_rows<Pack>(mesh, gauge_cell_gid, boundary_condition,
        [&](auto row, const auto& columns, const auto& values)
        {
            matrix->insertLocalValues(row, columns(), values());
        });
    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Refresh a compatible pressure graph without rebuilding its maps/imports.
 * Stage every coefficient before publication. Both graph compatibility and exact
 * numeric equality are collective, including ranks whose gauge row never changes.
 * Incompatible maps/entries leave the matrix untouched. False requires a fresh
 * graph, also if Tpetra unexpectedly rejects a staged replacement. The caller
 * must invalidate numeric preconditioners whenever values_changed is true.
 */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider>
bool refresh_pressure_poisson_matrix_values(const MeshType& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid,
    BoundaryConditionProvider boundary_condition, bool symmetric_gauge,
    typename Pack::matrix_type& matrix, bool& values_changed)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    const auto communicator = mesh.owned_cell_map()->getComm();
    values_changed = false;
    if (!compatible_present_transport_matrix_maps<Pack>(mesh, matrix, mesh.num_owned_cells())) return false;
    int local_invalid = 0;
    int global_invalid = 0;

    std::vector<Teuchos::Array<scalar_type>> staged(mesh.num_owned_cells());
    int local_changed = 0;
    visit_pressure_poisson_rows<Pack>(mesh, gauge_cell_gid, boundary_condition,
        [&](auto row, const auto& columns, const auto& values)
        {
            typename Pack::matrix_type::local_inds_host_view_type previous_columns;
            typename Pack::matrix_type::values_host_view_type previous_values;
            matrix.getLocalRowView(row, previous_columns, previous_values);
            auto& refreshed = staged[static_cast<size_t>(row)];
            refreshed.resize(previous_values.extent(0), scalar_type{});
            std::vector<bool> visited(previous_values.extent(0), false);
            for (int entry = 0; entry < columns.size(); ++entry)
            {
                size_t slot = 0;
                while (slot < previous_columns.extent(0) && previous_columns(slot) != columns[entry]) ++slot;
                if (slot == previous_columns.extent(0))
                {
                    local_invalid = 1;
                    continue;
                }
                visited[slot] = true;
                refreshed[slot] += values[entry];
            }
            for (size_t slot = 0; slot < previous_columns.extent(0); ++slot)
            {
                if (!visited[slot]) local_invalid = 1;
                if (symmetric_gauge && gauge_cell_gid &&
                    mesh.owned_cell_map()->getGlobalElement(row) != *gauge_cell_gid &&
                    matrix.getColMap()->getGlobalElement(previous_columns(slot)) == *gauge_cell_gid)
                    refreshed[slot] = scalar_type{};
                if (refreshed[slot] != previous_values(slot)) local_changed = 1;
            }
        });
    const int local_status[2]{local_invalid, local_changed};
    int global_status[2]{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 2, local_status, global_status);
    if (global_status[0] != 0) return false;
    values_changed = global_status[1] != 0;
    if (!values_changed) return true;
    const auto domain_map = matrix.getDomainMap();
    const auto range_map = matrix.getRangeMap();
    // Tpetra's host replacement API requires fill-active state even when
    // every entry already exists. It otherwise returns invalid without writing.
    matrix.resumeFill();
    int local_replacement_failed = 0;
    for (size_t owned = 0; owned < staged.size(); ++owned)
    {
        const auto row = static_cast<local_ordinal_type>(owned);
        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type previous_values;
        matrix.getLocalRowView(row, columns, previous_values);
        const auto entries = static_cast<local_ordinal_type>(columns.extent(0));
        const auto replaced = matrix.replaceLocalValues(row, entries,
            staged[owned].getRawPtr(), columns.data());
        if (replaced != entries) local_replacement_failed = 1;
    }
    matrix.fillComplete(domain_map, range_map);
    int global_replacement_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1,
        &local_replacement_failed, &global_replacement_failed);
    return global_replacement_failed == 0;
}

} // namespace SimpleFluid::FVM::detail
