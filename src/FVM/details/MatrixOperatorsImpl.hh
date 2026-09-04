/**
 * @file FVM/details/MatrixOperatorsImpl.hh
 * @brief Mesh-generic implementation kernels for FVM matrix operators.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "dataclass/TpetraTypes.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

/** @brief Assemble the mesh-generic two-point diffusion matrix. */
template<TpetraTypePack Pack, class MeshType>
Teuchos::RCP<typename Pack::matrix_type> diffusion_matrix_impl(
    const MeshType& mesh, typename Pack::scalar_type diffusivity)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
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
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
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

/** @brief Assemble pressure Poisson on a generic mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider>
Teuchos::RCP<typename Pack::matrix_type> pressure_poisson_matrix_impl(const MeshType& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid, BoundaryConditionProvider boundary_condition)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
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
            matrix->insertLocalValues(cell_lid, columns(), values());
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
        matrix->insertLocalValues(cell_lid, columns(), values());
    }
    matrix->fillComplete();
    return matrix;
}

} // namespace SimpleFluid::FVM::detail
