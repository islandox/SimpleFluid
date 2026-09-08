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
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** @brief Ordered owned-face topology and epoch-dependent Rhie--Chow metrics. */
template<TpetraTypePack Pack>
struct StoredPressureFaceGeometry
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;

    local_ordinal_type face{}, owner{}, neighbor{};
    bool interior = false;
    bool boundary = false;
    bool owned_owner = false;
    vec_type normal{}, owner_normal{}, area_vector{};
    scalar_type area{}, owner_weight{}, neighbor_weight{};
    // Keep numerator and denominator separate to preserve (dp * dot) / d2.
    scalar_type center_projection{}, distance_squared{}, boundary_diffusion{};
};

/** @brief Rebuild numeric geometry without retaining field or boundary values. */
template<TpetraTypePack Pack, class MeshType>
auto stored_pressure_face_geometry(const MeshType& mesh)
    -> std::vector<StoredPressureFaceGeometry<Pack>>
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    std::vector<StoredPressureFaceGeometry<Pack>> result;
    result.reserve(mesh.num_owned_faces());
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!mesh.is_owned_face(face_lid))
        {
            continue;
        }
        const auto face_id = query_face_id(mesh, face_lid);
        const auto owner_id = mesh.owner_cell(face_id);
        StoredPressureFaceGeometry<Pack> entry;
        entry.face = face_lid;
        entry.owner = packed_cell_local_id(mesh, owner_id);
        entry.owned_owner = static_cast<size_t>(entry.owner) < mesh.num_owned_cells();
        entry.interior = mesh.is_interior_face(face_id);
        entry.boundary = mesh.is_boundary_face(face_id);
        entry.normal = mesh.face_normal(face_id);
        entry.owner_normal = mesh.face_normal_outward(face_id, owner_id);
        entry.area = static_cast<scalar_type>(mesh.face_area(face_id));
        entry.area_vector = mesh.face_area_vector_outward(face_id, owner_id);
        if (entry.interior)
        {
            const auto neighbor_id = mesh.opposite_or_periodic_neighbor_cell(face_id, owner_id);
            entry.neighbor = packed_cell_local_id(mesh, neighbor_id);
            const auto weights = stored_interior_face_linear_weights(mesh, face_lid, entry.owner, entry.neighbor);
            entry.owner_weight = weights.first;
            entry.neighbor_weight = weights.second;
            const auto center_delta = mesh.cell_centroid(neighbor_id) - mesh.cell_centroid(owner_id);
            entry.distance_squared = center_delta.dot(center_delta);
            entry.center_projection = entry.area_vector.dot(center_delta);
        }
        else if (entry.boundary && entry.owned_owner)
        {
            entry.boundary_diffusion = boundary_diffusion_coefficient(mesh, face_lid, entry.owner, scalar_type{1});
        }
        result.push_back(entry);
    }
    return result;
}

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
    using scalar_type = typename Pack::scalar_type;
    using vec_type = typename VectorCellFieldStored<Pack, MeshType>::value_type;

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

    if (pressure_coefficient == scalar_type{})
    {
        assemble_stored_normal_face_fluxes(velocity, &boundary_cache, fluxes);
        fluxes.sync_ghosts();
        return;
    }
    if (boundary_cache.mesh.get() != velocity.mesh_ptr().get())
    {
        throw std::invalid_argument("face_fluxes received a boundary cache for another mesh.");
    }
    const auto& face_geometry = workspace.face_geometry();

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
    {
        // Release host views before importing the final owned fluxes below.
        const auto velocity_values = velocity.local_read_view();
        const auto pressure_values = pressure.local_read_view();
        const auto gradient_values = pressure_gradient.local_read_view();
        auto flux_values = fluxes.owned_write_view();
        for (const auto& entry : face_geometry)
        {
            const auto face_lid = entry.face;
            const auto face = static_cast<size_t>(face_lid);
            const auto owner_lid = entry.owner;
            const auto neighbor_lid = entry.neighbor;
            const vec_type owner_velocity{
                velocity_values(owner_lid, 0), velocity_values(owner_lid, 1), velocity_values(owner_lid, 2)};
            vec_type face_velocity{};
            if (entry.interior)
            {
                const vec_type neighbor_velocity{velocity_values(neighbor_lid, 0), velocity_values(neighbor_lid, 1),
                    velocity_values(neighbor_lid, 2)};
                face_velocity = owner_velocity * entry.owner_weight + neighbor_velocity * entry.neighbor_weight;
            }
            else if (entry.boundary)
            {
                const auto location = boundary_locations[face];
                if (location.active)
                {
                    const auto value_iter = boundary_cache.value.find(location.batch_id);
                    if (value_iter != boundary_cache.value.end() && location.in_batch_id < value_iter->second.size())
                    {
                        const auto type_iter = boundary_cache.type.find(location.batch_id);
                        const auto type = type_iter == boundary_cache.type.end() ? BoundaryConditionType::Neumann
                                                                                               : type_iter->second;
                        face_velocity = type == BoundaryConditionType::Slip
                                            ? owner_velocity - entry.owner_normal * owner_velocity.dot(entry.owner_normal)
                                            : value_iter->second[location.in_batch_id];
                    }
                }
            }
            // Preserve the normal-then-area order of the standalone kernel.
            flux_values(face_lid, 0) = face_velocity.dot(entry.normal) * entry.area;
            if (!entry.owned_owner)
            {
                continue;
            }

            const auto& area_vector = entry.area_vector;
            if (!entry.interior)
            {
                if (pressure_boundary_conditions == nullptr || !entry.boundary ||
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
                const auto condition = condition_iter == pressure_boundary_conditions->end() ? BoundaryCondition{}
                                                                                            : condition_iter->second;
                if (condition.type == BoundaryConditionType::Neumann)
                {
                    continue;
                }
                if (condition.type != BoundaryConditionType::Dirichlet)
                {
                    throw std::invalid_argument("pressure_weighted_face_fluxes supports only Dirichlet "
                                                "and Neumann pressure boundary conditions.");
                }

                const vec_type owner_gradient{
                    gradient_values(owner_lid, 0), gradient_values(owner_lid, 1), gradient_values(owner_lid, 2)};
                const auto direct_gradient_flux =
                    (static_cast<scalar_type>(condition.value) - pressure_values(owner_lid, 0)) *
                    entry.boundary_diffusion;
                const auto interpolated_gradient_flux = owner_gradient.dot(area_vector);
                flux_values(face_lid, 0) = owner_velocity.dot(area_vector) -
                                         pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux);
                continue;
            }

            if (entry.distance_squared <= scalar_type{})
            {
                continue;
            }

            const auto direct_gradient_flux = (pressure_values(neighbor_lid, 0) - pressure_values(owner_lid, 0)) *
                                              entry.center_projection / entry.distance_squared;
            const vec_type owner_gradient{
                gradient_values(owner_lid, 0), gradient_values(owner_lid, 1), gradient_values(owner_lid, 2)};
            const vec_type neighbor_gradient{
                gradient_values(neighbor_lid, 0), gradient_values(neighbor_lid, 1), gradient_values(neighbor_lid, 2)};
            const auto interpolated_gradient =
                owner_gradient * entry.owner_weight + neighbor_gradient * entry.neighbor_weight;
            const auto interpolated_gradient_flux = interpolated_gradient.dot(area_vector);
            flux_values(face_lid, 0) =
                flux_values(face_lid, 0) - pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux);
        }
    }
    fluxes.sync_ghosts();
}

} // namespace SimpleFluid::FVM::detail
