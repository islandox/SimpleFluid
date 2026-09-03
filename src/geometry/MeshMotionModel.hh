/**
 * @file MeshMotionModel.hh
 * @brief Fixed-topology mesh-motion state and transaction interface.
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/MeshQuality.hh"

#include <cstdint>
#include <span>
#include <string_view>

namespace SimpleFluid
{

/** @brief Immutable diagnostics for the current or most recent motion state. */
struct MeshMotionDiagnostics
{
    real_t old_surface_elevation = {};
    real_t new_surface_elevation = {};
    real_t time_step = {}; ///< [s]
    std::uint64_t old_geometry_epoch = 0;
    std::uint64_t new_geometry_epoch = 0;
    real_t maximum_absolute_gcl_residual = {}; ///< [m^3/s]
    real_t maximum_normalized_gcl_residual = {};
    MeshQualityMetrics mesh_quality;
    bool trial_active = false;
};

/**
 * @brief Transactional fixed-topology mesh-motion interface.
 *
 * Cell arrays use mesh-local cell order and face fluxes use mesh-local face
 * order with the mesh owner-normal sign convention. Implementations mutate
 * geometry only during begin_trial(); callers must either accept or roll back
 * the trial before beginning another one.
 */
class MeshMotionModel
{
public:
    virtual ~MeshMotionModel() = default;

    /** Apply and validate a trial surface elevation collectively. */
    virtual void begin_trial(real_t surface_elevation, real_t time_step) = 0;

    /** Commit the active geometry trial collectively. */
    virtual void accept_trial() = 0;

    /** Restore the pre-trial geometry collectively. */
    virtual void rollback_trial() = 0;

    [[nodiscard]] virtual bool has_active_trial() const noexcept = 0;
    [[nodiscard]] virtual std::string_view mesh_family() const noexcept = 0;

    /** Accepted-old cell volumes for the current motion state [m^3]. */
    [[nodiscard]] virtual std::span<const real_t> old_cell_volumes() const noexcept = 0;

    /** Trial/new cell volumes for the current motion state [m^3]. */
    [[nodiscard]] virtual std::span<const real_t> new_cell_volumes() const noexcept = 0;

    /** Owner-oriented swept-volume rates for locally visible faces [m^3/s]. */
    [[nodiscard]] virtual std::span<const real_t> face_mesh_fluxes() const noexcept = 0;

    [[nodiscard]] virtual const MeshMotionDiagnostics& diagnostics() const noexcept = 0;
};

} // namespace SimpleFluid
