/** @brief Compiled mapped face operators, workspace setup and custom-provider opt-in. */
#pragma once

#include "FVM/FaceFlux.hh"
#include "FVM/details/FieldStoredFaceFlux.tcc"
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.tcc"
#include "FVM/details/FieldStoredVelocityBoundaryCache.tcc"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::SharedGeometry::SharedGeometry(SP<const mesh_type> mesh,
    std::shared_ptr<const topology_type> topology, std::shared_ptr<const traverse_plan_type> gradient_traverse)
    : d_mesh(require_mesh(std::move(mesh))), d_topology(std::move(topology)),
      d_geometry_identity(detail::ale_geometry_identity(*d_mesh))
{
    // A local failure must reach the numerical collective preflight,
    // before another rank enters imports or writes any output values.
    try
    {
        const auto backend = ::SimpleFluid::detail::retain_traverse_backend(*d_mesh);
        const auto native_lease = detail::acquire_pressure_native_geometry(*d_mesh);
        const auto lease = acquire_mesh_execution(*d_mesh);
        d_geometry_epoch = mesh_geometry_epoch(*d_mesh);
        if (d_topology)
            d_topology->require_valid(*d_mesh);
        d_gradient_cache = std::make_unique<CellGradientCache<Pack, mesh_type>>(d_mesh,
            typename CellGradientCache<Pack, mesh_type>::boundary_direction_provider_type{},
            std::move(gradient_traverse));
        if (!d_topology)
            d_topology = detail::make_pressure_face_topology<Pack>(*d_mesh, d_gradient_cache->boundary_locations());
        d_metrics = detail::pressure_face_metrics<Pack>(*d_mesh, *d_topology);
    }
    catch (...)
    {
        d_error = std::current_exception();
    }
}

template<TpetraTypePack Pack, class MeshType>
FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::FieldStoredPressureWeightedFaceFluxWorkspace(
    SP<const mesh_type> mesh)
    : FieldStoredPressureWeightedFaceFluxWorkspace(mesh, std::make_shared<SharedGeometry>(mesh))
{
}

template<TpetraTypePack Pack, class MeshType>
FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::FieldStoredPressureWeightedFaceFluxWorkspace(
    SP<const mesh_type> mesh, shared_geometry_type geometry)
    : d_mesh(require_mesh(std::move(mesh))), d_geometry(require_geometry(d_mesh, std::move(geometry))),
      d_pressure_gradient(VectorCellFieldDescriptor<Pack>("rhie_chow_pressure_gradient_workspace"), d_mesh, false)
{
}

template<TpetraTypePack Pack, class MeshType>
const std::vector<typename FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::face_geometry_type>&
FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::face_geometry() const
{
    const auto execution = acquire_execution();
    std::call_once(d_geometry->d_legacy_geometry_once,
        [&]
        {
            d_geometry->d_legacy_geometry = detail::stored_pressure_face_geometry<Pack>(*d_mesh);
            d_geometry->d_materialized_geometry.store(true, std::memory_order_release);
        });
    return d_geometry->d_legacy_geometry;
}

template<TpetraTypePack Pack, class MeshType>
const typename FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::FaceBatches&
FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::face_batches() const
{
    const auto& geometry = face_geometry();
    std::call_once(d_geometry->d_legacy_batches_once,
        [&]
        {
            const auto& locations = boundary_locations();
            FaceBatches result;
            for (size_t i = 0; i < geometry.size(); ++i)
            {
                const auto& face = geometry[i];
                if (face.interior)
                {
                    result.interior.push_back(i);
                    continue;
                }
                const auto& location = locations[static_cast<size_t>(face.face)];
                const auto id = d_mesh->boundary_id(detail::query_face_id(*d_mesh, face.face));
                auto& batch = result.boundary.try_emplace(id, BoundaryBatch{id, {}, {}}).first->second;
                const bool active = face.boundary && location.active;
                batch.faces.push_back({i, location.in_batch_id, active});
                batch.all_active = batch.all_active && active;
                if (active)
                    batch.required_values = std::max(batch.required_values, location.in_batch_id + 1);
            }
            for (auto& [id, batch] : result.boundary)
                if (batch.required_values)
                    batch.name = d_mesh->boundary_batch_name(id);
            d_geometry->d_legacy_batches = std::move(result);
        });
    return d_geometry->d_legacy_batches;
}

template<TpetraTypePack Pack, class MeshType>
void FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::refresh_geometry()
{
    require_ready_geometry(*d_geometry);
    d_geometry->d_topology->require_valid(*d_mesh);
    if (detail::ale_geometry_identity(*d_mesh) == d_geometry->d_geometry_identity &&
        mesh_geometry_epoch(*d_mesh) == d_geometry->d_geometry_epoch)
        return;
    auto next = shared_geometry_type(
        new SharedGeometry(d_mesh, d_geometry->d_topology, d_geometry->d_gradient_cache->shared_traverse_plan()));
    require_ready_geometry(*next);
    d_geometry = std::move(next);
}

template<TpetraTypePack Pack, class MeshType>
void FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>::refresh_geometry(shared_geometry_type geometry)
{
    auto next = require_geometry(d_mesh, std::move(geometry));
    require_ready_geometry(*next);
    next->d_topology->require_valid(*d_mesh);
    d_geometry = std::move(next);
}

template<TpetraTypePack Pack, class MeshType>
    requires(!std::same_as<std::remove_cv_t<MeshType>, Mesh<Pack>>)
SIMPLEFLUID_FVM_EXPORT FieldStoredVelocityBoundaryCache<Pack, MeshType> cache_velocity_boundary_conditions(
    SP<const MeshType> mesh, const BoundaryConditionSet& boundary_conditions)
{
    using cache_type = FieldStoredVelocityBoundaryCache<Pack, MeshType>;
    return detail::cache_field_stored_velocity_boundary_conditions_impl<Pack, MeshType, cache_type>(
        std::move(mesh), boundary_conditions);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void face_velocities(
    const VectorCellFieldStored<Pack, MeshType>& velocity, VectorFaceFieldStored<Pack, MeshType>& face_velocity)
{
    using cache_type = FieldStoredVelocityBoundaryCache<Pack, MeshType>;
    detail::assemble_stored_face_velocities<Pack, MeshType, cache_type>(velocity, nullptr, face_velocity);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void face_velocities(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    VectorFaceFieldStored<Pack, MeshType>& face_velocity)
{
    detail::assemble_stored_face_velocities(velocity, &boundary_cache, face_velocity);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void normal_face_fluxes(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    detail::stored_normal_face_fluxes(face_velocity, fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    using cache_type = FieldStoredVelocityBoundaryCache<Pack, MeshType>;
    detail::assemble_stored_normal_face_fluxes<Pack, MeshType, cache_type>(velocity, nullptr, fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    detail::assemble_stored_normal_face_fluxes(velocity, &boundary_cache, fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void mesh_relative_face_fluxes(const ScalarFaceFieldStored<Pack, MeshType>& absolute_fluxes,
    const ALEControlVolumeState& ale, ScalarFaceFieldStored<Pack, MeshType>& relative_fluxes)
{
    detail::stored_mesh_relative_face_fluxes(absolute_fluxes, ale, relative_fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme)
{
    detail::pressure_weighted_stored_face_fluxes_impl(velocity, pressure, pressure_coefficient, boundary_cache, nullptr,
        workspace, static_cast<VectorCellFieldStored<Pack, MeshType>*>(nullptr), fluxes, gradient_scheme);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType> workspace(velocity.mesh_ptr());
    pressure_weighted_face_fluxes(velocity, pressure, pressure_coefficient, boundary_cache, workspace, fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme)
{
    detail::pressure_weighted_stored_face_fluxes_impl(velocity, pressure, pressure_coefficient, boundary_cache,
        &pressure_boundary_conditions, workspace, static_cast<VectorCellFieldStored<Pack, MeshType>*>(nullptr), fluxes,
        gradient_scheme);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_flux_components(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& total, ScalarFaceFieldStored<Pack, MeshType>& velocity_component,
    ScalarFaceFieldStored<Pack, MeshType>& pressure_component, CellGradientScheme gradient_scheme)
{
    detail::pressure_weighted_stored_face_fluxes_impl(velocity, pressure, pressure_coefficient, boundary_cache,
        &pressure_boundary_conditions, workspace, static_cast<VectorCellFieldStored<Pack, MeshType>*>(nullptr), total,
        gradient_scheme, &velocity_component, &pressure_component);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, VectorCellFieldStored<Pack, MeshType>& pressure_gradient,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    detail::pressure_weighted_stored_face_fluxes_impl(velocity, pressure, pressure_coefficient, boundary_cache,
        &pressure_boundary_conditions, workspace, &pressure_gradient, fluxes);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions, ScalarFaceFieldStored<Pack, MeshType>& fluxes)
{
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType> workspace(velocity.mesh_ptr());
    pressure_weighted_face_fluxes(
        velocity, pressure, pressure_coefficient, boundary_cache, pressure_boundary_conditions, workspace, fluxes);
}

} // namespace SimpleFluid::FVM
