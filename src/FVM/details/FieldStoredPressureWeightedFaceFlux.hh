/**
 * @file FVM/details/FieldStoredPressureWeightedFaceFlux.hh
 * @brief Internal Rhie-Chow face-flux kernels for mesh-aware stored fields.
 */
#pragma once

#include "FVM/CellOperators.hh"
#include "FVM/details/FieldStoredFaceFlux.hh"
#include "FVM/details/ResolvedTransportGeometry.hh"
#include "fields/FieldStored.hh"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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
    const auto execution = acquire_mesh_execution(mesh);
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
        const auto append = [&](local_ordinal_type owner, local_ordinal_type neighbor,
                                bool interior, bool boundary, const auto& metrics)
        {
            StoredPressureFaceGeometry<Pack> entry;
            entry.face = face_lid;
            entry.owner = owner;
            entry.owned_owner = static_cast<size_t>(entry.owner) < mesh.num_owned_cells();
            entry.interior = interior;
            entry.boundary = boundary;
            entry.normal = metrics.face_normal(query_face_id(metrics, face_lid));
            entry.owner_normal = metrics.face_normal_outward(
                query_face_id(metrics, face_lid), query_cell_id(metrics, owner));
            entry.area = static_cast<scalar_type>(metrics.face_area(query_face_id(metrics, face_lid)));
            entry.area_vector = metrics.face_area_vector_outward(
                query_face_id(metrics, face_lid), query_cell_id(metrics, owner));
            if (entry.interior)
            {
                entry.neighbor = neighbor;
                const auto weights = stored_interior_face_linear_weights(metrics, face_lid, owner, neighbor);
                entry.owner_weight = weights.first;
                entry.neighbor_weight = weights.second;
                const auto center_delta = metrics.cell_center_vector(
                    query_face_id(metrics, face_lid), query_cell_id(metrics, owner));
                entry.distance_squared = center_delta.dot(center_delta);
                entry.center_projection = entry.area_vector.dot(center_delta);
            }
            else if (entry.boundary && entry.owned_owner)
            {
                entry.boundary_diffusion = boundary_diffusion_coefficient(metrics, face_lid, owner, scalar_type{1});
            }
            result.push_back(entry);
        };
        if constexpr (requires { mesh.supports_region_execution(); })
        {
            if (mesh.supports_region_execution())
            {
                const auto resolved = mesh.resolve_region_face(face_lid);
                const auto owner = mesh.region_cell_local_id(resolved.owner);
                const auto neighbor = resolved.interior()
                    ? mesh.region_cell_local_id(resolved.neighbor) : local_ordinal_type{};
                const ResolvedTransportGeometry<MeshType, std::remove_cvref_t<decltype(resolved)>>
                    metrics(resolved, owner, resolved.owner);
                append(owner, neighbor, resolved.interior(),
                    !resolved.interior() && mesh.is_boundary_face(face_lid), metrics);
                continue;
            }
        }
        const auto face_id = query_face_id(mesh, face_lid);
        const auto owner_id = mesh.owner_cell(face_id);
        const auto interior = mesh.is_interior_face(face_id);
        const auto neighbor = interior
            ? packed_cell_local_id(mesh, mesh.opposite_or_periodic_neighbor_cell(face_id, owner_id))
            : local_ordinal_type{};
        append(packed_cell_local_id(mesh, owner_id), neighbor, interior, mesh.is_boundary_face(face_id), mesh);
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
 * Interior faces and physical boundary batches are evaluated separately.
 * Output storage must not alias input storage across these independent passes.
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

    const auto validate_inputs = [&]
    {
        if (pressure.mesh_ptr().get() != velocity.mesh_ptr().get() || fluxes.mesh_ptr().get() != velocity.mesh_ptr().get())
            throw std::invalid_argument("pressure_weighted_face_fluxes requires fields on one mesh.");
        if (workspace.mesh_ptr().get() != velocity.mesh_ptr().get())
            throw std::invalid_argument("pressure_weighted_face_fluxes received a workspace for another mesh.");
        if (precomputed_pressure_gradient != nullptr &&
            precomputed_pressure_gradient->mesh_ptr().get() != velocity.mesh_ptr().get())
            throw std::invalid_argument("pressure_weighted_face_fluxes received a pressure gradient for another mesh.");
        if (pressure_coefficient < scalar_type{})
            throw std::invalid_argument("pressure_weighted_face_fluxes requires a non-negative pressure coefficient.");
        if (pressure_boundary_conditions != nullptr)
            validate_pressure_velocity_boundary_compatibility(boundary_cache, *pressure_boundary_conditions);
    };

    if (pressure_coefficient == scalar_type{})
    {
        validate_inputs();
        if (workspace.boundary_locations().size() != velocity.mesh().num_faces())
            throw std::invalid_argument("pressure face-flux workspace has incompatible boundary locations.");
        assemble_stored_normal_face_fluxes(velocity, &boundary_cache, fluxes);
        fluxes.sync_ghosts();
        return;
    }

    const auto& mesh = velocity.mesh();
    using execution_type = decltype(workspace.acquire_execution());
    struct ExecutionFactory
    {
        const Workspace& workspace;
        operator execution_type() const { return workspace.acquire_execution(); }
    };
    std::optional<execution_type> execution;
    int local_error = 0;
    try
    {
        validate_inputs();
        if (boundary_cache.mesh.get() != velocity.mesh_ptr().get())
            throw std::invalid_argument("face_fluxes received a boundary cache for another mesh.");
        // Epoch, topology, retained construction failures, and lease acquisition
        // all reject collectively before gradient synchronization or writes.
        execution.emplace(ExecutionFactory{workspace});
        if (workspace.boundary_locations().size() != mesh.num_faces())
            throw std::invalid_argument("pressure face-flux workspace has incompatible boundary locations.");
    }
    catch (...)
    {
        local_error = 1;
    }
    int any_error = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_error, &any_error);
    if (any_error != 0)
        throw std::invalid_argument("pressure face-flux execution requires valid fields, current geometry, "
                                    "and matching topology on every rank.");

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

    {
        // Release host views before importing the final owned fluxes below.
        const auto velocity_values = velocity.local_read_view();
        const auto pressure_values = pressure.local_read_view();
        const auto gradient_values = pressure_gradient.local_read_view();
        auto flux_values = fluxes.owned_write_view();

        execution->interior().apply([&](const auto& face, const auto& metrics)
        {
            const auto face_lid = face.face;
            const auto owner_lid = face.owner;
            const auto neighbor_lid = face.neighbor;
            const vec_type owner_velocity{
                velocity_values(owner_lid, 0), velocity_values(owner_lid, 1), velocity_values(owner_lid, 2)};
            const vec_type neighbor_velocity{
                velocity_values(neighbor_lid, 0), velocity_values(neighbor_lid, 1), velocity_values(neighbor_lid, 2)};
            const auto face_velocity = owner_velocity * metrics.owner_weight + neighbor_velocity * metrics.neighbor_weight;
            // Preserve the normal-then-area order of the standalone kernel.
            flux_values(face_lid, 0) = face_velocity.dot(metrics.normal) * metrics.area;
            if (!face.owned_owner || metrics.distance_squared <= scalar_type{}) return;

            const auto direct_gradient_flux = (pressure_values(neighbor_lid, 0) - pressure_values(owner_lid, 0)) *
                                              metrics.center_projection / metrics.distance_squared;
            const vec_type owner_gradient{
                gradient_values(owner_lid, 0), gradient_values(owner_lid, 1), gradient_values(owner_lid, 2)};
            const vec_type neighbor_gradient{
                gradient_values(neighbor_lid, 0), gradient_values(neighbor_lid, 1), gradient_values(neighbor_lid, 2)};
            const auto interpolated_gradient =
                owner_gradient * metrics.owner_weight + neighbor_gradient * metrics.neighbor_weight;
            const auto interpolated_gradient_flux = interpolated_gradient.dot(metrics.area_vector);
            flux_values(face_lid, 0) =
                flux_values(face_lid, 0) - pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux);
        });

        execution->zero_flux_faces().apply([&](const auto& face, const auto& metrics)
        {
            flux_values(face.face, 0) = vec_type{}.dot(metrics.normal) * metrics.area;
        });

        for (const auto& batch : execution->boundary_groups())
        {
            const auto value_iter = boundary_cache.value.find(batch.id);
            const auto* values = value_iter == boundary_cache.value.end() ? nullptr : &value_iter->second;
            const auto type_iter = boundary_cache.type.find(batch.id);
            const auto velocity_type = type_iter == boundary_cache.type.end()
                ? BoundaryConditionType::Neumann : type_iter->second;
            BoundaryCondition pressure_condition{};
            if (pressure_boundary_conditions != nullptr)
            {
                const auto condition_iter = pressure_boundary_conditions->find(batch.name);
                if (condition_iter != pressure_boundary_conditions->end()) pressure_condition = condition_iter->second;
            }
            const bool correct_pressure = pressure_condition.type == BoundaryConditionType::Dirichlet;
            const bool complete_values = values != nullptr && values->size() >= batch.required_values;
            const auto faces = execution->boundary(batch);

            const auto apply_boundary = [&]<bool Slip, bool CorrectPressure, bool CompleteValues>()
            {
                faces.apply([&](const auto& face, const auto& metrics)
                {
                    const auto face_lid = face.face;
                    const auto owner_lid = face.owner;
                    const vec_type owner_velocity{
                        velocity_values(owner_lid, 0), velocity_values(owner_lid, 1), velocity_values(owner_lid, 2)};
                    vec_type face_velocity{};
                    const auto boundary_velocity = [&]
                    {
                        if constexpr (Slip)
                            return owner_velocity - metrics.owner_normal * owner_velocity.dot(metrics.owner_normal);
                        else
                            return (*values)[face.in_batch_id];
                    };
                    if constexpr (CompleteValues)
                        face_velocity = boundary_velocity();
                    else if (values != nullptr && face.in_batch_id < values->size())
                        face_velocity = boundary_velocity();
                    flux_values(face_lid, 0) = face_velocity.dot(metrics.normal) * metrics.area;

                    if constexpr (CorrectPressure)
                    {
                        if (!face.owned_owner) return;
                        // A pressure correction does not require a velocity
                        // value slot, including for missing/truncated arrays.
                        const vec_type owner_gradient{
                            gradient_values(owner_lid, 0), gradient_values(owner_lid, 1), gradient_values(owner_lid, 2)};
                        const auto direct_gradient_flux =
                            (static_cast<scalar_type>(pressure_condition.value) - pressure_values(owner_lid, 0)) *
                            metrics.boundary_diffusion;
                        const auto interpolated_gradient_flux = owner_gradient.dot(metrics.area_vector);
                        flux_values(face_lid, 0) = owner_velocity.dot(metrics.area_vector) -
                            pressure_coefficient * (direct_gradient_flux - interpolated_gradient_flux);
                    }
                });
            };
            const auto select_values = [&]<bool Slip, bool CorrectPressure>()
            {
                if (complete_values) apply_boundary.template operator()<Slip, CorrectPressure, true>();
                else apply_boundary.template operator()<Slip, CorrectPressure, false>();
            };
            const auto select_pressure = [&]<bool Slip>()
            {
                if (correct_pressure) select_values.template operator()<Slip, true>();
                else select_values.template operator()<Slip, false>();
            };
            if (velocity_type == BoundaryConditionType::Slip) select_pressure.template operator()<true>();
            else select_pressure.template operator()<false>();
        }
    }
    fluxes.sync_ghosts();
}

} // namespace SimpleFluid::FVM::detail
