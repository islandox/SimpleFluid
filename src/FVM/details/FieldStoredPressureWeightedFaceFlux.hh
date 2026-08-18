/**
 * @file FVM/details/FieldStoredPressureWeightedFaceFlux.hh
 * @brief Internal Rhie-Chow face-flux kernels for mesh-aware stored fields.
 */
#pragma once

#include "FVM/CellOperators.hh"
#include "FVM/details/FieldStoredFaceFlux.hh"
#include "fields/FieldStored.hh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace SimpleFluid::FVM::detail
{

/** @brief Validate pressure/velocity boundary compatibility for any cache. */
template<class VelocityBoundaryCacheType>
void validate_pressure_velocity_boundary_compatibility(
    const VelocityBoundaryCacheType& velocity_boundary_cache, const BoundaryConditionMap& pressure_boundary_conditions)
{
    for (const auto& [name, pressure_condition] : pressure_boundary_conditions)
    {
        if (pressure_condition.type != BoundaryConditionType::Dirichlet &&
            pressure_condition.type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument("pressure_weighted_face_fluxes supports only Dirichlet "
                                        "and Neumann pressure boundary conditions.");
        }
        if (pressure_condition.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        const auto velocity_iter = velocity_boundary_cache.type_by_name.find(name);
        const auto velocity_type = velocity_iter == velocity_boundary_cache.type_by_name.end()
                                       ? BoundaryConditionType::Neumann
                                       : velocity_iter->second;
        if (velocity_type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument("Dirichlet pressure boundary '" + name +
                                        "' requires a Neumann velocity boundary so owner-cell "
                                        "velocity can be extrapolated to the open face.");
        }
    }
}

/**
 * @brief Compute Rhie-Chow face fluxes on a mapped mesh using stored fields.
 *
 * @tparam Pack Tpetra type pack used by the fields.
 * @tparam MeshType Runtime or statically dispatched mapped mesh.
 * @tparam VelocityBoundaryCacheType Stored-field velocity boundary cache.
 * @tparam Workspace Stored-field pressure-flux workspace.
 */
template<TpetraTypePack Pack, class MeshType, class VelocityBoundaryCacheType, class Workspace>
void pressure_weighted_stored_face_fluxes_impl(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCacheType& boundary_cache, const BoundaryConditionMap* pressure_boundary_conditions,
    Workspace& workspace, VectorCellFieldStored<Pack, MeshType>* precomputed_pressure_gradient,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes,
    CellGradientScheme gradient_scheme = CellGradientScheme::LeastSquares)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    if (pressure.mesh_ptr().get() != velocity.mesh_ptr().get() || fluxes.mesh_ptr().get() != velocity.mesh_ptr().get())
    {
        throw std::invalid_argument("pressure_weighted_face_fluxes requires fields on one mesh.");
    }
    if (workspace.mesh_ptr().get() != velocity.mesh_ptr().get() ||
        workspace.boundary_locations().size() != velocity.mesh().num_faces())
    {
        throw std::invalid_argument("pressure_weighted_face_fluxes received a workspace for "
                                    "another mesh.");
    }
    if (precomputed_pressure_gradient != nullptr &&
        precomputed_pressure_gradient->mesh_ptr().get() != velocity.mesh_ptr().get())
    {
        throw std::invalid_argument("pressure_weighted_face_fluxes received a pressure gradient "
                                    "for another mesh.");
    }
    if (pressure_coefficient < scalar_type{})
    {
        throw std::invalid_argument("pressure_weighted_face_fluxes requires a non-negative "
                                    "pressure coefficient.");
    }
    if (pressure_boundary_conditions != nullptr)
    {
        validate_pressure_velocity_boundary_compatibility(boundary_cache, *pressure_boundary_conditions);
    }

    assemble_stored_normal_face_fluxes(velocity, &boundary_cache, fluxes);
    if (pressure_coefficient == scalar_type{})
    {
        fluxes.sync_ghosts();
        return;
    }

    auto& pressure_gradient =
        precomputed_pressure_gradient == nullptr ? workspace.pressure_gradient() : *precomputed_pressure_gradient;
    if (precomputed_pressure_gradient == nullptr && pressure_boundary_conditions == nullptr)
    {
        if (gradient_scheme == CellGradientScheme::GaussLinear)
        {
            ::SimpleFluid::FVM::gauss_linear_cell_gradient(pressure, pressure_gradient);
        }
        else
        {
            ::SimpleFluid::FVM::cell_gradient(pressure, pressure_gradient, workspace.gradient_cache());
        }
    }
    else if (precomputed_pressure_gradient == nullptr)
    {
        ::SimpleFluid::FVM::cell_gradient(
            pressure, *pressure_boundary_conditions, pressure_gradient, workspace.gradient_cache(), gradient_scheme);
    }
    pressure_gradient.sync_ghosts();

    const auto& mesh = velocity.mesh();
    const auto& boundary_locations = workspace.boundary_locations();
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!fluxes.is_owned(face_lid))
        {
            continue;
        }

        const auto face_id = query_face_id(mesh, face_lid);
        const auto owner_id = mesh.owner_cell(face_id);
        const auto owner_lid = packed_cell_local_id(mesh, owner_id);
        if (!velocity.is_owned(owner_lid))
        {
            continue;
        }

        const auto area_vector = mesh.face_area_vector_outward(face_id, owner_id);
        if (!mesh.is_interior_face(face_id))
        {
            if (pressure_boundary_conditions == nullptr || !mesh.is_boundary_face(face_id) ||
                face >= boundary_locations.size())
            {
                continue;
            }
            const auto location = boundary_locations[face];
            if (!location.active)
            {
                continue;
            }
            const auto& name = mesh.boundary_batch_name(location.batch_id);
            const auto condition_iter = pressure_boundary_conditions->find(name);
            const auto condition =
                condition_iter == pressure_boundary_conditions->end() ? BoundaryCondition{} : condition_iter->second;
            if (condition.type == BoundaryConditionType::Neumann)
            {
                continue;
            }
            if (condition.type != BoundaryConditionType::Dirichlet)
            {
                throw std::invalid_argument("pressure_weighted_face_fluxes supports only Dirichlet "
                                            "and Neumann pressure boundary conditions.");
            }

            const auto owner_velocity = velocity.local_value(owner_lid);
            const auto direct_gradient_flux =
                (static_cast<scalar_type>(condition.value) - pressure.local_value(owner_lid)) *
                boundary_diffusion_coefficient(mesh, face_lid, owner_lid, scalar_type{1});
            const auto interpolated_gradient_flux = pressure_gradient.local_value(owner_lid).dot(area_vector);
            fluxes.set_owned_value(
                face_lid, owner_velocity.dot(area_vector) -
                              pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux));
            continue;
        }

        const auto neighbor_id = mesh.opposite_or_periodic_neighbor_cell(face_id, owner_id);
        const auto neighbor_lid = packed_cell_local_id(mesh, neighbor_id);
        const auto center_delta = mesh.cell_centroid(neighbor_id) - mesh.cell_centroid(owner_id);
        const auto distance_squared = center_delta.dot(center_delta);
        if (distance_squared <= scalar_type{})
        {
            continue;
        }

        const auto direct_gradient_flux = (pressure.local_value(neighbor_lid) - pressure.local_value(owner_lid)) *
                                          area_vector.dot(center_delta) / distance_squared;
        const auto [owner_weight, neighbor_weight] =
            stored_interior_face_linear_weights(mesh, face_lid, owner_lid, neighbor_lid);
        const auto interpolated_gradient = pressure_gradient.local_value(owner_lid) * owner_weight +
                                           pressure_gradient.local_value(neighbor_lid) * neighbor_weight;
        const auto interpolated_gradient_flux = interpolated_gradient.dot(area_vector);
        fluxes.set_owned_value(face_lid,
            fluxes.value(face_lid) - pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux));
    }
    fluxes.sync_ghosts();
}

} // namespace SimpleFluid::FVM::detail
