/**
 * @file FaceFlux.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Face-flux assembly and velocity-boundary cache helpers.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
#include "utils/export_helper.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"
#include "geometry/MeshTraverse.hh"
#include "FVM/CellOperators.hh"
#include "FVM/details/FieldStoredFaceFlux.hh"
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.hh"
#include "FVM/details/PressureFaceBatches.hh"
#include "geometry/DirectBatchRange.hh"
#include "FVM/details/FieldStoredVelocityBoundaryCache.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Boundary cache for mesh-aware stored velocity fields.
 * @tparam Pack Tpetra type pack used by the fields.
 * @tparam MeshType Runtime or statically dispatched mapped mesh.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
struct FieldStoredVelocityBoundaryCache
{
    using vec_type = vec3<typename Pack::scalar_type>;

    explicit FieldStoredVelocityBoundaryCache(SP<const MeshType> mesh)
        : mesh(std::move(mesh))
    {
    }

    std::unordered_map<int, Arr<vec_type>> value;
    std::unordered_map<int, BoundaryConditionType> type;
    std::unordered_map<std::string, BoundaryConditionType> type_by_name;
    SP<const MeshType> mesh;
};

/**
 * @brief Private Rhie-Chow scratch with shared topology and geometry snapshots.
 *
 * Face/cell IDs and boundary slots are retained independently of geometry.
 * Each epoch owns category-specific metric arrays in exactly the same order.
 * Acquiring an execution pins and validates the snapshot once, then exposes
 * direct paired ranges. Fixed-topology ALE refresh preserves topology identity.
 * A workspace is tied to one exact mesh and is not safe for concurrent use.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType = MeshHandle<Pack>>
class SIMPLEFLUID_FVM_EXPORT FieldStoredPressureWeightedFaceFluxWorkspace
{
public:
    using mesh_type = MeshType;
    using face_geometry_type = detail::StoredPressureFaceGeometry<Pack>;
    using traverse_plan_type = MeshTraversePlan<mesh_type>;
    using topology_type = detail::PressureFaceTopology<Pack, mesh_type>;
    using boundary_group_type = typename topology_type::boundary_group_type;
    using boundary_location_type = detail::BoundaryFaceLocation<mesh_type>;
    using interior_range_type = DirectBatchRange<typename topology_type::interior_type,
        detail::InteriorPressureFaceMetrics<Pack>>;
    using boundary_range_type = DirectBatchRange<typename topology_type::boundary_type,
        detail::BoundaryPressureFaceMetrics<Pack>>;
    using zero_range_type = DirectBatchRange<typename topology_type::zero_type,
        detail::ZeroPressureFaceMetrics<Pack>>;

    // Compatibility descriptors are materialized only on explicit request.
    struct BoundaryFaceIndex { size_t geometry_index, in_batch_id; bool active; };
    struct BoundaryBatch
    {
        int id;
        std::string name;
        std::vector<BoundaryFaceIndex> faces;
        size_t required_values = 0;
        bool all_active = true;
    };
    struct FaceBatches
    {
        std::vector<size_t> interior;
        std::map<int, BoundaryBatch> boundary;
    };

    /** Immutable topology/metric snapshot; lazy legacy views are not on the numerical path. */
    class SharedGeometry
    {
        friend class FieldStoredPressureWeightedFaceFluxWorkspace;
    public:
        explicit SharedGeometry(SP<const mesh_type> mesh)
            : SharedGeometry(std::move(mesh), {}) {}

    private:
        SharedGeometry(SP<const mesh_type> mesh, std::shared_ptr<const topology_type> topology,
            std::shared_ptr<const traverse_plan_type> gradient_traverse = {});

        const SP<const mesh_type> d_mesh;
        std::shared_ptr<const topology_type> d_topology;
        std::unique_ptr<CellGradientCache<Pack, mesh_type>> d_gradient_cache;
        detail::PressureFaceMetrics<Pack> d_metrics;
        const void* const d_geometry_identity;
        std::uint64_t d_geometry_epoch = 0;
        std::exception_ptr d_error;
        mutable std::once_flag d_legacy_geometry_once, d_legacy_batches_once;
        mutable std::vector<face_geometry_type> d_legacy_geometry;
        mutable FaceBatches d_legacy_batches;
        mutable std::atomic<bool> d_materialized_geometry{false};
    };
    using shared_geometry_type = std::shared_ptr<const SharedGeometry>;

    /** Snapshot ownership and read leases outlive every borrowed direct range. */
    class SIMPLEFLUID_LOCAL Execution
    {
        friend class FieldStoredPressureWeightedFaceFluxWorkspace;
    public:
        Execution(const Execution&) = delete;
        Execution& operator=(const Execution&) = delete;
        Execution(Execution&&) = delete;
        Execution& operator=(Execution&&) = delete;

        interior_range_type interior() const & noexcept { return d_interior; }
        zero_range_type zero_flux_faces() const & noexcept { return d_zero; }
        const std::vector<boundary_group_type>& boundary_groups() const & noexcept
        { return d_geometry->d_topology->boundary_groups; }
        boundary_range_type boundary(const boundary_group_type& group) const & noexcept
        { return d_boundary.subrange(group.begin, group.count); }
        interior_range_type interior() const && = delete;
        zero_range_type zero_flux_faces() const && = delete;
        const std::vector<boundary_group_type>& boundary_groups() const && = delete;
        boundary_range_type boundary(const boundary_group_type&) const && = delete;

    private:
        explicit Execution(shared_geometry_type geometry)
            : d_geometry(checked_geometry(std::move(geometry))),
              d_backend(::SimpleFluid::detail::retain_traverse_backend(*d_geometry->d_mesh)),
              d_native_lease(detail::acquire_pressure_native_geometry(*d_geometry->d_mesh)),
              d_lease(acquire_mesh_execution(*d_geometry->d_mesh)),
              d_interior(interior_range_type::checked(d_geometry->d_topology->interior, d_geometry->d_metrics.interior)),
              d_boundary(boundary_range_type::checked(d_geometry->d_topology->boundary, d_geometry->d_metrics.boundary)),
              d_zero(zero_range_type::checked(d_geometry->d_topology->zero_flux_faces, d_geometry->d_metrics.zero_flux_faces))
        {}

        static shared_geometry_type checked_geometry(shared_geometry_type geometry)
        {
            require_ready_geometry(*geometry);
            require_current_geometry(*geometry->d_mesh, *geometry);
            geometry->d_topology->require_valid(*geometry->d_mesh);
            return geometry;
        }
        shared_geometry_type d_geometry;
        [[maybe_unused]] decltype(::SimpleFluid::detail::retain_traverse_backend(std::declval<const mesh_type&>())) d_backend;
        [[maybe_unused]] std::optional<GeometryExecutionGuard::ReadLease> d_native_lease;
        decltype(acquire_mesh_execution(std::declval<const mesh_type&>())) d_lease;
        interior_range_type d_interior;
        boundary_range_type d_boundary;
        zero_range_type d_zero;
    };

    explicit FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh);

    /** Allocate private scratch while reusing the exact mesh's current snapshot. */
    FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh, shared_geometry_type geometry);

    FieldStoredPressureWeightedFaceFluxWorkspace(const FieldStoredPressureWeightedFaceFluxWorkspace&) = delete;
    FieldStoredPressureWeightedFaceFluxWorkspace& operator=(const FieldStoredPressureWeightedFaceFluxWorkspace&) = delete;
    FieldStoredPressureWeightedFaceFluxWorkspace(FieldStoredPressureWeightedFaceFluxWorkspace&&) = default;
    FieldStoredPressureWeightedFaceFluxWorkspace& operator=(FieldStoredPressureWeightedFaceFluxWorkspace&&) = default;

    const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }
    VectorCellFieldStored<Pack, mesh_type>& pressure_gradient() noexcept { return d_pressure_gradient; }
    const VectorCellFieldStored<Pack, mesh_type>& pressure_gradient() const noexcept { return d_pressure_gradient; }
    const std::vector<boundary_location_type>& boundary_locations() const
    { return gradient_cache().boundary_locations(); }
    const CellGradientCache<Pack, mesh_type>& gradient_cache() const
    {
        require_ready_geometry(*d_geometry);
        return *d_geometry->d_gradient_cache;
    }
    const shared_geometry_type& shared_geometry() const noexcept { return d_geometry; }
    const topology_type& topology_groups() const
    {
        require_ready_geometry(*d_geometry);
        d_geometry->d_topology->require_valid(*d_mesh);
        return *d_geometry->d_topology;
    }
    [[nodiscard]] SIMPLEFLUID_LOCAL Execution acquire_execution() const & { return Execution(d_geometry); }
    Execution acquire_execution() const && = delete;

    /** Explicit invalidation propagates to all snapshots sharing this topology. */
    void invalidate_topology() noexcept
    { if (d_geometry->d_topology) d_geometry->d_topology->invalidate(); }

    /** Compatibility only: do not use in numerical traversal. */
    const traverse_plan_type& traverse_plan() const
    {
        return topology_groups().legacy_traverse(*d_mesh);
    }
    bool face_traverse_matches_geometry() const noexcept
    {
        try
        {
            require_current_geometry(*d_mesh, *d_geometry);
            topology_groups();
            return true;
        }
        catch (...) { return false; }
    }
    bool has_materialized_face_geometry() const noexcept
    { return d_geometry->d_materialized_geometry.load(std::memory_order_acquire); }

    /** Lazily rebuild the legacy face-ordered vector only when explicitly requested. */
    const std::vector<face_geometry_type>& face_geometry() const;

    /** Legacy index grouping, materialized independently of production storage. */
    const FaceBatches& face_batches() const;

    /** Publish replacement metrics only after successful fixed-topology refresh. */
    void refresh_geometry();
    void refresh_geometry(shared_geometry_type geometry);

private:
    static void require_ready_geometry(const SharedGeometry& geometry)
    {
        if (geometry.d_error) std::rethrow_exception(geometry.d_error);
        if (!geometry.d_topology || !geometry.d_gradient_cache)
            throw std::invalid_argument("Pressure face-flux snapshot construction is incomplete.");
    }
    static void require_current_geometry(const mesh_type& mesh, const SharedGeometry& geometry)
    {
        if (&mesh != geometry.d_mesh.get())
            throw std::invalid_argument("Pressure face-flux geometry belongs to another mesh.");
        if (detail::ale_geometry_identity(mesh) != geometry.d_geometry_identity
            || mesh_geometry_epoch(mesh) != geometry.d_geometry_epoch)
            throw std::invalid_argument("Pressure face-flux geometry is stale for the mesh geometry epoch.");
    }
    static shared_geometry_type require_geometry(const SP<const mesh_type>& mesh, shared_geometry_type geometry)
    {
        if (!geometry) throw std::invalid_argument("Pressure face-flux geometry snapshot must not be null.");
        if (mesh.get() != geometry->d_mesh.get())
            throw std::invalid_argument("Pressure face-flux geometry belongs to another mesh.");
        if (!geometry->d_error)
        {
            require_current_geometry(*mesh, *geometry);
            if (geometry->d_topology) geometry->d_topology->require_valid(*mesh);
        }
        return geometry;
    }
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh) throw std::invalid_argument("FieldStoredPressureWeightedFaceFluxWorkspace requires a non-null mesh.");
        return mesh;
    }
    SP<const mesh_type> d_mesh;
    shared_geometry_type d_geometry;
    VectorCellFieldStored<Pack, mesh_type> d_pressure_gradient;
};

// Give the runtime constructor one exported, non-COMDAT owner. It is called
// across the FVM/solver DSO boundary, including optimized Darwin builds.
template<>
SIMPLEFLUID_FVM_EXPORT FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh);

// Explicit cold specializations override hidden static-mesh argument visibility
// without suppressing header-owned execution/lease methods.
template<>
SIMPLEFLUID_FVM_EXPORT FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D,
        DefaultTpetraTypes>>::SharedGeometry::SharedGeometry(SP<const mesh_type> mesh,
    std::shared_ptr<const topology_type> topology, std::shared_ptr<const traverse_plan_type> gradient_traverse);

template<>
SIMPLEFLUID_FVM_EXPORT FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D,
        DefaultTpetraTypes>>::FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh);

template<>
SIMPLEFLUID_FVM_EXPORT FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::
    FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh, shared_geometry_type geometry);

template<>
SIMPLEFLUID_FVM_EXPORT const std::vector<typename FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::face_geometry_type>&
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::face_geometry() const;

template<>
SIMPLEFLUID_FVM_EXPORT const typename FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::FaceBatches&
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::face_batches() const;

template<>
SIMPLEFLUID_FVM_EXPORT void FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::refresh_geometry();

template<>
SIMPLEFLUID_FVM_EXPORT void FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>::refresh_geometry(shared_geometry_type
        geometry);

/** Native default-mesh cache and pressure-workspace spellings. */
template<TpetraTypePack Pack>
using VelocityBoundaryCache = FieldStoredVelocityBoundaryCache<Pack, MeshHandle<Pack>>;

template<TpetraTypePack Pack>
using PressureWeightedFaceFluxWorkspace = FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshHandle<Pack>>;

/** @brief Build a boundary cache for a mesh-aware stored velocity field. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT FieldStoredVelocityBoundaryCache<Pack, MeshType>
cache_velocity_boundary_conditions(
    SP<const MeshType> mesh,
    const BoundaryConditionSet& boundary_conditions);

/** Forward a mutable native mesh to the const cache-provider entry. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
    requires (!std::is_const_v<MeshType>)
FieldStoredVelocityBoundaryCache<Pack, MeshType> cache_velocity_boundary_conditions(
    SP<MeshType> mesh, const BoundaryConditionSet& boundary_conditions)
{
    return cache_velocity_boundary_conditions<Pack, MeshType>(
        SP<const MeshType>(std::move(mesh)), boundary_conditions);
}

/**
 * @brief Interpolate a mesh-aware stored cell velocity onto stored faces.
 *
 * The mesh may be a runtime `MeshHandle` over an orthogonal or
 * semi-structured geometry, or a statically typed partitioned mesh.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void face_velocities(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    VectorFaceFieldStored<Pack, MeshType>& face_velocity);

/** @brief Interpolate stored velocities using cached boundary values. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void face_velocities(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    VectorFaceFieldStored<Pack, MeshType>& face_velocity);

/** @brief Project a stored face velocity onto owner-oriented face areas. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void normal_face_fluxes(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/** @brief Assemble stored normal face fluxes without boundary treatment. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/** @brief Assemble stored normal face fluxes with cached boundary values. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/**
 * @brief Subtract the exact swept-volume rate from an absolute face flux.
 *
 * Both input and output use mesh-owner orientation and units of m^3/s. Owned
 * values are authoritative; the completed relative flux is synchronized to
 * overlap faces before returning. Input and output must be distinct so an
 * absolute flux retained by pressure correction and continuity diagnostics is
 * never overwritten in place.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void mesh_relative_face_fluxes(
    const ScalarFaceFieldStored<Pack, MeshType>& absolute_fluxes,
    const ALEControlVolumeState& ale,
    ScalarFaceFieldStored<Pack, MeshType>& relative_fluxes);

/**
 * @brief Compute stored-field Rhie-Chow fluxes with reusable scratch storage.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes,
    CellGradientScheme gradient_scheme =
        CellGradientScheme::LeastSquares);

/** @brief Compute stored-field Rhie-Chow fluxes with one-shot scratch. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/**
 * @brief Compute stored-field Rhie-Chow fluxes with pressure boundaries.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes,
    CellGradientScheme gradient_scheme =
        CellGradientScheme::LeastSquares);

/** Compute total and its velocity/pressure components in the identical face traversal.
 * Pressure component is the Rhie--Chow stabilization contribution, not a pressure-only velocity.
 * Outputs must be distinct and are synchronized with the same ownership policy.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_flux_components(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& total,
    ScalarFaceFieldStored<Pack, MeshType>& velocity_component,
    ScalarFaceFieldStored<Pack, MeshType>& pressure_component,
    CellGradientScheme gradient_scheme = CellGradientScheme::LeastSquares);

/**
 * @brief Compute stored-field Rhie-Chow fluxes from a precomputed gradient.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    VectorCellFieldStored<Pack, MeshType>& pressure_gradient,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

/**
 * @brief Compute stored-field Rhie-Chow fluxes with boundaries and one-shot
 *        scratch storage.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void pressure_weighted_face_fluxes(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

} // namespace SimpleFluid::FVM
