/**
 * @file FVM/details/DiffusionSystemImpl.hh
 * @brief Mesh-generic scalar and vector diffusion assembly kernels.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "dataclass/TpetraTypes.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <stdexcept>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack> struct DiffusionSystem;

template<TpetraTypePack Pack> struct VectorDiffusionSystem;

} // namespace SimpleFluid::FVM

namespace SimpleFluid::FVM::detail
{

/** @brief Assemble scalar orthogonal diffusion for any mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class SourceProvider>
DiffusionSystem<Pack> diffusion_system_impl(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition, SourceProvider right_hand_source)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (diffusivity < scalar_type{})
    {
        throw std::invalid_argument("diffusion_system requires non-negative diffusivity.");
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
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
        rhs->replaceLocalValue(
            cell_lid, static_cast<scalar_type>(mesh.cell_volume(cell_id)) * right_hand_source(cell_lid));
    }

    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < boundary_locations.size(); ++face)
    {
        const auto& location = boundary_locations[face];
        if (!location.active)
        {
            continue;
        }
        const auto face_lid = static_cast<local_ordinal_type>(face);
        const auto face_id = query_face_id(mesh, face_lid);
        if (!mesh.is_owned_face(face_id) || !mesh.is_boundary_face(face_id))
        {
            continue;
        }
        const auto owner_lid = packed_cell_local_id(mesh, mesh.owner_cell(face_id));
        const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
        if (condition.type == BoundaryConditionType::Dirichlet)
        {
            const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, owner_lid, diffusivity);
            if (coefficient > scalar_type{})
            {
                auto column = owner_lid;
                matrix->sumIntoLocalValues(
                    owner_lid, Teuchos::arrayView(&column, 1), Teuchos::arrayView(&coefficient, 1));
                rhs->sumIntoLocalValue(owner_lid, coefficient * condition.value);
            }
        }
        else if (condition.type == BoundaryConditionType::Neumann)
        {
            rhs->sumIntoLocalValue(
                owner_lid, diffusivity * condition.value * static_cast<scalar_type>(mesh.face_area(face_id)));
        }
        else if (condition.type == BoundaryConditionType::Robin)
        {
            throw std::runtime_error("Robin boundary conditions are not yet implemented in "
                                     "diffusion_system.");
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/** @brief Assemble vector orthogonal diffusion for any mapped mesh. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class SourceProvider>
VectorDiffusionSystem<Pack> vector_diffusion_system_impl(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition, SourceProvider right_hand_source)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t num_components = 3;

    if (diffusivity < scalar_type{})
    {
        throw std::invalid_argument("vector_diffusion_system requires non-negative diffusivity.");
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 8));
    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), num_components, true));
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

        const auto source = right_hand_source(cell_lid);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_id));
        for (size_t component = 0; component < num_components; ++component)
        {
            rhs->replaceLocalValue(cell_lid, component, volume * source.component(component));
        }
    }

    const auto boundary_locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < boundary_locations.size(); ++face)
    {
        const auto& location = boundary_locations[face];
        if (!location.active)
        {
            continue;
        }
        const auto face_lid = static_cast<local_ordinal_type>(face);
        const auto face_id = query_face_id(mesh, face_lid);
        if (!mesh.is_owned_face(face_id) || !mesh.is_boundary_face(face_id))
        {
            continue;
        }
        const auto owner_lid = packed_cell_local_id(mesh, mesh.owner_cell(face_id));
        const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
        if (condition.type == BoundaryConditionType::Dirichlet || condition.type == BoundaryConditionType::NoSlip)
        {
            const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, owner_lid, diffusivity);
            if (coefficient > scalar_type{})
            {
                auto column = owner_lid;
                matrix->sumIntoLocalValues(
                    owner_lid, Teuchos::arrayView(&column, 1), Teuchos::arrayView(&coefficient, 1));
                for (size_t component = 0; component < num_components; ++component)
                {
                    rhs->sumIntoLocalValue(owner_lid, component, coefficient * condition.value.component(component));
                }
            }
        }
        else if (condition.type == BoundaryConditionType::Neumann)
        {
            const auto area = static_cast<scalar_type>(mesh.face_area(face_id));
            for (size_t component = 0; component < num_components; ++component)
            {
                rhs->sumIntoLocalValue(owner_lid, component, diffusivity * condition.value.component(component) * area);
            }
        }
        else if (condition.type == BoundaryConditionType::Robin)
        {
            throw std::runtime_error("Robin boundary conditions are not yet implemented in "
                                     "vector_diffusion_system.");
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

} // namespace SimpleFluid::FVM::detail
