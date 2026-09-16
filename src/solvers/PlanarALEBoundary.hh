/**
 * @file PlanarALEBoundary.hh
 * @brief Validated flat moving-boundary contract for planar ALE solvers.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/FaceFlux.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief One planar patch whose absolute carrier flux follows the mesh exactly.
 *
 * The selected faces remain fixed by topology. Geometry values are queried at
 * every trial so no normal, area, centroid, or velocity survives an epoch
 * transition. The liquid kinematic condition is imposed in flux form as
 * `phi_abs = phi_mesh`, hence the boundary ALE transport flux is exactly zero.
 * Construction, refresh, and both enforcement operations are collective on
 * the mesh communicator. Every rank must supply a non-null mesh, and boundary
 * controls and vessel-map evaluations must agree exactly across ranks.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes> class SIMPLEFLUID_SOLVERS_EXPORT PlanarALEBoundary
{
public:
    using mesh_type = MeshHandle<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using face_flux_field_type = ScalarFaceFieldStored<Pack, mesh_type>;
    using velocity_boundary_cache_type = FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>;

    struct Diagnostics
    {
        scalar_type surface_elevation = {};  ///< [m]
        scalar_type global_area = {};        ///< [m^2]
        scalar_type global_mesh_volume = {}; ///< [m^3]
        scalar_type mapped_pool_volume = {}; ///< [m^3]
        scalar_type volume_mismatch = {};    ///< [m^3]
    };

    PlanarALEBoundary(SP<const mesh_type> mesh, std::string boundary_name, Dimension axis,
        const VesselVolumeMap& volume_map, scalar_type volume_absolute_tolerance, scalar_type relative_tolerance);

    const std::string& name() const noexcept { return d_boundary_name; }
    int batch_id() const noexcept { return d_batch_id; }
    Dimension axis() const noexcept { return d_axis; }
    const Diagnostics& diagnostics() const noexcept { return d_diagnostics; }

    /** Revalidate current trial geometry against the same vessel map. */
    void refresh(const VesselVolumeMap& volume_map);

    /**
     * Mark the moving patch as slip and retain its mesh-normal velocity.
     * Tangential face velocity remains owner-extrapolated; the exact normal
     * kinematic condition is imposed independently by enforce_kinematic_flux().
     */
    void apply_kinematic_velocity(const FVM::ALEControlVolumeState& ale, velocity_boundary_cache_type& cache) const;

    /** Enforce the accepted absolute boundary-flux form of the kinematic BC. */
    void enforce_kinematic_flux(const FVM::ALEControlVolumeState& ale, face_flux_field_type& absolute_flux) const;

    bool contains(local_ordinal_type face_lid) const noexcept
    {
        return std::ranges::find(d_face_lids, face_lid) != d_face_lids.end();
    }

private:
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_collective_controls() const;

    SIMPLEFLUID_SOLVERS_LOCAL
    void validate(const VesselVolumeMap& volume_map);

    SP<const mesh_type> d_mesh;
    std::string d_boundary_name;
    Dimension d_axis = Dimension::Z;
    scalar_type d_volume_absolute_tolerance = {}; ///< [m^3]
    scalar_type d_relative_tolerance = {};
    int d_batch_id = -1;
    std::vector<local_ordinal_type> d_face_lids;
    Diagnostics d_diagnostics;
};
extern template class PlanarALEBoundary<DefaultTpetraTypes>;


} // namespace SimpleFluid
