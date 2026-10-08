/**
 * @file FVM/ALEControlVolumeState.hh
 * @brief Borrowed or immutable validated geometry and swept-flux state for ALE assembly.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "dataclass/TpetraTypes.hh"
#include "geometry/GeometryEpoch.hh"
#include "geometry/MeshMotionModel.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{
template<TpetraTypePack Pack> class SIMPLEFLUID_PUBLIC_TYPE PlanarALEMeshMotion;
}

namespace SimpleFluid::FVM
{

namespace detail
{

/** @brief Return the identity of the concrete geometry behind a mesh view. */
template<class MeshType> const void* ale_geometry_identity(const MeshType& mesh) noexcept
{
    if constexpr (requires {
                      { mesh.geometry_identity() } -> std::convertible_to<const void*>;
                  })
    {
        return mesh.geometry_identity();
    }
    else
    {
        // Static mesh types and legacy meshes are their own geometry identity.
        return std::addressof(mesh);
    }
}

} // namespace detail

/**
 * @brief Accepted-old/trial-new control-volume state for one live ALE trial.
 *
 * Cell spans use mesh-local cell order, including overlap cells. Face swept
 * rates use mesh-local face order and are positive along the mesh owner normal.
 * Generic factories borrow the motion arrays. The validated planar factory
 * owns immutable copies and may reuse their successful GCL validation. Both
 * forms borrow the originating motion object, which must outlive the state;
 * its exact trial must remain active for every assembly that consumes it.
 */
class ALEControlVolumeState
{
public:
    ALEControlVolumeState(const ALEControlVolumeState&) noexcept = default;
    ALEControlVolumeState& operator=(const ALEControlVolumeState&) noexcept = default;

    /** Moving preserves the source view and shares immutable ownership, so a
     * moved-from state cannot fall back to dangling borrowed span storage.
     */
    ALEControlVolumeState(ALEControlVolumeState&& other) noexcept
        : ALEControlVolumeState(static_cast<const ALEControlVolumeState&>(other)) {}
    ALEControlVolumeState& operator=(ALEControlVolumeState&& other) noexcept
    {
        return operator=(static_cast<const ALEControlVolumeState&>(other));
    }

    /** Accepted-old local cell volumes [m^3]. */
    std::span<const real_t> old_cell_volumes() const noexcept { return d_old_cell_volumes; }

    /** Trial-new local cell volumes [m^3]. */
    std::span<const real_t> new_cell_volumes() const noexcept { return d_new_cell_volumes; }

    /** Owner-oriented swept-volume face rates [m^3/s]. */
    std::span<const real_t> face_mesh_fluxes() const noexcept { return d_face_mesh_fluxes; }

    real_t time_step() const noexcept { return d_time_step; }
    std::uint64_t old_geometry_epoch() const noexcept { return d_old_geometry_epoch; }
    std::uint64_t new_geometry_epoch() const noexcept { return d_new_geometry_epoch; }
    const void* geometry_identity() const noexcept { return d_geometry_identity; }

    /**
     * @brief Validate identity, active-trial state, dimensions, and the GCL.
     *
     * Metadata and live-trial checks are collective at every assembly boundary.
     * Borrowed views always repeat the full finite-data/GCL validation. Owned
     * planar snapshots reuse it only when every rank has a validated snapshot;
     * mixed owned/borrowed calls repeat the full checks on every rank.
     * Mesh, MeshHandle and SolidSubdomain with DefaultTpetraTypes are compiled
     * into FVM. Other mesh types must include FVM/ALEControlVolumeState.tcc.
     */
    template<class MeshType>
    SIMPLEFLUID_FVM_EXPORT void validate(const MeshType& mesh) const;

    /** @brief Also require the assembly timestep to equal the motion timestep. */
    template<class MeshType, class Scalar> void validate(const MeshType& mesh, Scalar assembly_time_step) const
    {
        validate(mesh);
        const auto value = static_cast<real_t>(assembly_time_step);
        const int local_mismatch = !std::isfinite(value) || value != d_time_step ? 1 : 0;
        int global_mismatch = 0;
        Teuchos::reduceAll(
            *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_mismatch, &global_mismatch);
        if (global_mismatch != 0)
        {
            throw std::invalid_argument("ALE transport timestep must exactly match the active mesh-motion trial.");
        }
    }

private:
    template<class MeshType, class MotionType>
    friend ALEControlVolumeState make_ale_control_volume_state(const MeshType&, const MotionType&);
    template<class MeshType, TpetraTypePack Pack>
    friend ALEControlVolumeState make_validated_planar_ale_control_volume_state(
        const MeshType&, const PlanarALEMeshMotion<Pack>&);

    /** Immutable storage is published as a certificate only after its copied
     * bytes pass full collective validation using the original GCL kernel.
     */
    struct ValidatedGeometry
    {
        ValidatedGeometry(std::span<const real_t> old_volumes, std::span<const real_t> new_volumes,
            std::span<const real_t> swept_fluxes)
            : old_cell_volumes(old_volumes.begin(), old_volumes.end()),
              new_cell_volumes(new_volumes.begin(), new_volumes.end()),
              face_mesh_fluxes(swept_fluxes.begin(), swept_fluxes.end()) {}
        const std::vector<real_t> old_cell_volumes;
        const std::vector<real_t> new_cell_volumes;
        const std::vector<real_t> face_mesh_fluxes;
    };

    template<class MeshType, class MotionType>
    static ALEControlVolumeState bind_to_motion(const MeshType& mesh, const MotionType& motion);

    ALEControlVolumeState(const MeshMotionModel& motion, const void* mesh_view_identity, const void* geometry_identity,
        real_t gcl_absolute_tolerance, real_t gcl_relative_tolerance)
        : d_motion(std::addressof(motion)), d_mesh_view_identity(mesh_view_identity),
          d_geometry_identity(geometry_identity), d_old_cell_volumes(motion.old_cell_volumes()),
          d_new_cell_volumes(motion.new_cell_volumes()), d_face_mesh_fluxes(motion.face_mesh_fluxes()),
          d_time_step(motion.diagnostics().time_step), d_old_geometry_epoch(motion.diagnostics().old_geometry_epoch),
          d_new_geometry_epoch(motion.diagnostics().new_geometry_epoch),
          d_gcl_absolute_tolerance(gcl_absolute_tolerance), d_gcl_relative_tolerance(gcl_relative_tolerance)
    {
    }

    const MeshMotionModel* d_motion = nullptr;
    const void* d_mesh_view_identity = nullptr;
    const void* d_geometry_identity = nullptr;
    std::span<const real_t> d_old_cell_volumes;
    std::span<const real_t> d_new_cell_volumes;
    std::span<const real_t> d_face_mesh_fluxes;
    real_t d_time_step = {};
    std::uint64_t d_old_geometry_epoch = 0;
    std::uint64_t d_new_geometry_epoch = 0;
    real_t d_gcl_absolute_tolerance = 1.0e-12;
    real_t d_gcl_relative_tolerance = 1.0e-10;
    std::shared_ptr<const ValidatedGeometry> d_validated_geometry;
};

/**
 * @brief Check shared mesh/motion binding before selecting storage ownership.
 *
 * The concrete motion type must expose mesh_ptr(); this prevents callers from
 * pairing valid-looking spans from one moving geometry with another mesh.
 */
template<class MeshType, class MotionType>
ALEControlVolumeState ALEControlVolumeState::bind_to_motion(const MeshType& mesh, const MotionType& motion)
{
    static_assert(std::derived_from<std::remove_cvref_t<MotionType>, MeshMotionModel>);
    static_assert(requires(const MotionType& candidate) { candidate.mesh_ptr(); });
    const auto communicator = mesh.owned_cell_map()->getComm();
    const auto& motion_mesh_ptr = motion.mesh_ptr();
    const auto mesh_identity = detail::ale_geometry_identity(mesh);
    const std::array<int, 2> local_binding_error{motion_mesh_ptr ? 0 : 1,
        motion_mesh_ptr && detail::ale_geometry_identity(*motion_mesh_ptr) != mesh_identity ? 1 : 0};
    std::array<int, 2> global_binding_error{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_binding_error.size()),
        local_binding_error.data(), global_binding_error.data());
    if (global_binding_error[0] != 0)
    {
        throw std::invalid_argument("ALE control-volume state requires a motion model with a live mesh.");
    }
    if (global_binding_error[1] != 0)
    {
        throw std::invalid_argument(
            "ALE control-volume state requires the motion model and fields to share one concrete geometry.");
    }

    const auto& motion_mesh = *motion_mesh_ptr;
    int local_layout_error = mesh.num_owned_cells() != motion_mesh.num_owned_cells() ||
                                     mesh.num_local_cells() != motion_mesh.num_local_cells() ||
                                     mesh.num_owned_faces() != motion_mesh.num_owned_faces() ||
                                     mesh.num_faces() != motion_mesh.num_faces()
                                 ? 1
                                 : 0;
    int global_layout_error = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_layout_error, &global_layout_error);
    if (global_layout_error != 0)
    {
        throw std::invalid_argument(
            "ALE control-volume state requires matching owned and local cell/face counts on every rank.");
    }

    const std::array<int, 8> local_map_presence_error{mesh.owned_cell_map() ? 0 : 1,
        motion_mesh.owned_cell_map() ? 0 : 1, mesh.overlap_cell_map() ? 0 : 1, motion_mesh.overlap_cell_map() ? 0 : 1,
        mesh.owned_face_map() ? 0 : 1, motion_mesh.owned_face_map() ? 0 : 1, mesh.overlap_face_map() ? 0 : 1,
        motion_mesh.overlap_face_map() ? 0 : 1};
    std::array<int, 8> global_map_presence_error{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_map_presence_error.size()),
        local_map_presence_error.data(), global_map_presence_error.data());
    if (std::any_of(
            global_map_presence_error.begin(), global_map_presence_error.end(), [](int error) { return error != 0; }))
    {
        throw std::invalid_argument("ALE control-volume state requires complete owned/overlap cell and face maps.");
    }

    local_layout_error = !mesh.owned_cell_map()->isSameAs(*motion_mesh.owned_cell_map()) ||
                         !mesh.overlap_cell_map()->isSameAs(*motion_mesh.overlap_cell_map()) ||
                         !mesh.owned_face_map()->isSameAs(*motion_mesh.owned_face_map()) ||
                         !mesh.overlap_face_map()->isSameAs(*motion_mesh.overlap_face_map());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_layout_error, &global_layout_error);
    if (global_layout_error != 0)
    {
        throw std::invalid_argument("ALE control-volume state requires matching owned/overlap cell and face maps.");
    }

    local_layout_error = 0;
    for (size_t local = 0; local < mesh.num_local_cells(); ++local)
    {
        const auto lid = static_cast<typename MeshType::local_ordinal_type>(local);
        local_layout_error = local_layout_error || mesh.cell_global_id(lid) != motion_mesh.cell_global_id(lid);
    }
    for (size_t local = 0; local < mesh.num_faces(); ++local)
    {
        const auto lid = static_cast<typename MeshType::local_ordinal_type>(local);
        local_layout_error = local_layout_error || mesh.face_global_id(lid) != motion_mesh.face_global_id(lid);
    }
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_layout_error, &global_layout_error);
    if (global_layout_error != 0)
    {
        throw std::invalid_argument("ALE control-volume state requires identical mesh-local global-ID order.");
    }

    real_t absolute_tolerance = 1.0e-12;
    real_t relative_tolerance = 1.0e-10;
    if constexpr (requires { motion.options().gcl_absolute_tolerance; })
    {
        absolute_tolerance = motion.options().gcl_absolute_tolerance;
        relative_tolerance = motion.options().gcl_relative_tolerance;
    }
    return ALEControlVolumeState(motion, std::addressof(mesh), mesh_identity, absolute_tolerance, relative_tolerance);
}

/** @brief Bind borrowed arrays to a motion trial and validate them in full.
 * Every later consumer repeats full validation, including for custom motion
 * models whose borrowed arrays can change without a geometry epoch change.
 */
template<class MeshType, class MotionType>
ALEControlVolumeState make_ale_control_volume_state(const MeshType& mesh, const MotionType& motion)
{
    auto result = ALEControlVolumeState::bind_to_motion(mesh, motion);
    result.validate(mesh);
    return result;
}

/** @brief Own and certify immutable arrays from one concrete planar trial.
 *
 * Copy allocation failures are propagated collectively before validation.
 * Full finite-data/GCL checks run once on the copied bytes before the proof is
 * published. Later consumers still validate identity, active-trial state,
 * dimensions, static constituents, epochs and timestep. Planar motion publishes
 * monotonic epochs on begin/rollback/restore, including identical trial replay.
 * Copies and moves share the immutable storage; the motion must still outlive
 * every state. Modifying the original motion arrays cannot change this snapshot.
 */
template<class MeshType, TpetraTypePack Pack>
ALEControlVolumeState make_validated_planar_ale_control_volume_state(
    const MeshType& mesh, const PlanarALEMeshMotion<Pack>& motion)
{
    auto result = ALEControlVolumeState::bind_to_motion(mesh, motion);
    std::shared_ptr<const ALEControlVolumeState::ValidatedGeometry> storage;
    std::exception_ptr allocation_error;
    try
    {
        storage = std::make_shared<const ALEControlVolumeState::ValidatedGeometry>(
            result.d_old_cell_volumes, result.d_new_cell_volumes, result.d_face_mesh_fluxes);
    }
    catch (...)
    {
        allocation_error = std::current_exception();
    }
    const int local_failure = allocation_error ? 1 : 0;
    int any_failure = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
        1, &local_failure, &any_failure);
    if (any_failure)
    {
        if (allocation_error) std::rethrow_exception(allocation_error);
        throw std::runtime_error("ALE immutable geometry snapshot allocation failed on another rank.");
    }
    result.d_old_cell_volumes = storage->old_cell_volumes;
    result.d_new_cell_volumes = storage->new_cell_volumes;
    result.d_face_mesh_fluxes = storage->face_mesh_fluxes;
    // No certificate is attached yet: validate the owned bytes in full once.
    result.validate(mesh);
    result.d_validated_geometry = std::move(storage);
    return result;
}

} // namespace SimpleFluid::FVM
