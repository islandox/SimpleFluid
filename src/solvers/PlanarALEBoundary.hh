/**
 * @file PlanarALEBoundary.hh
 * @brief Validated flat moving-boundary contract for planar ALE solvers.
 */

#pragma once

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
template<TpetraTypePack Pack = DefaultTpetraTypes> class PlanarALEBoundary
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
        const VesselVolumeMap& volume_map, scalar_type volume_absolute_tolerance, scalar_type relative_tolerance)
        : d_mesh(std::move(mesh)), d_boundary_name(std::move(boundary_name)), d_axis(axis),
          d_volume_absolute_tolerance(volume_absolute_tolerance), d_relative_tolerance(relative_tolerance)
    {
        if (!d_mesh)
        {
            throw std::invalid_argument("PlanarALEBoundary requires a mesh.");
        }
        validate_collective_controls();

        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            if (d_mesh->boundary_batch_name(batch_id) == d_boundary_name)
            {
                d_batch_id = batch_id;
                d_face_lids = batch.face_lids;
                break;
            }
        }
        validate(volume_map);
    }

    const std::string& name() const noexcept { return d_boundary_name; }
    int batch_id() const noexcept { return d_batch_id; }
    Dimension axis() const noexcept { return d_axis; }
    const Diagnostics& diagnostics() const noexcept { return d_diagnostics; }

    /** Revalidate current trial geometry against the same vessel map. */
    void refresh(const VesselVolumeMap& volume_map) { validate(volume_map); }

    /**
     * Mark the moving patch as slip and retain its mesh-normal velocity.
     * Tangential face velocity remains owner-extrapolated; the exact normal
     * kinematic condition is imposed independently by enforce_kinematic_flux().
     */
    void apply_kinematic_velocity(const FVM::ALEControlVolumeState& ale, velocity_boundary_cache_type& cache) const
    {
        ale.validate(*d_mesh);
        int local_cache_mismatch = cache.mesh.get() != d_mesh.get() ? 1 : 0;
        int local_invalid_area = 0;
        for (const auto face_lid : d_face_lids)
        {
            const auto area = static_cast<scalar_type>(d_mesh->face_area(face_lid));
            local_invalid_area = local_invalid_area || !std::isfinite(area) || !(area > scalar_type{});
        }
        std::array<int, 2> local_validation{local_cache_mismatch, local_invalid_area};
        std::array<int, 2> global_validation{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
            static_cast<int>(local_validation.size()), local_validation.data(), global_validation.data());
        if (global_validation[0] != 0)
        {
            throw std::invalid_argument("PlanarALEBoundary velocity cache belongs to another mesh handle.");
        }
        if (global_validation[1] != 0)
        {
            throw std::runtime_error("PlanarALEBoundary moving face has non-finite or non-positive area.");
        }
        cache.type_by_name[d_boundary_name] = BoundaryConditionType::Slip;
        if (d_batch_id < 0)
        {
            return;
        }
        auto& values = cache.value[d_batch_id];
        values.resize(d_face_lids.size());
        cache.type[d_batch_id] = BoundaryConditionType::Slip;
        const auto mesh_fluxes = ale.face_mesh_fluxes();
        for (size_t in_batch = 0; in_batch < d_face_lids.size(); ++in_batch)
        {
            const auto face_lid = d_face_lids[in_batch];
            const auto area = static_cast<scalar_type>(d_mesh->face_area(face_lid));
            const auto normal = d_mesh->face_normal(face_lid);
            values[in_batch] = normal * (static_cast<scalar_type>(mesh_fluxes[static_cast<size_t>(face_lid)]) / area);
        }
    }

    /** Enforce the accepted absolute boundary-flux form of the kinematic BC. */
    void enforce_kinematic_flux(const FVM::ALEControlVolumeState& ale, face_flux_field_type& absolute_flux) const
    {
        ale.validate(*d_mesh);
        const int local_mesh_mismatch = absolute_flux.mesh_ptr().get() != d_mesh.get() ? 1 : 0;
        int any_mesh_mismatch = 0;
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_mesh_mismatch, &any_mesh_mismatch);
        if (any_mesh_mismatch != 0)
        {
            throw std::invalid_argument("PlanarALEBoundary face flux belongs to another mesh handle.");
        }
        const auto mesh_fluxes = ale.face_mesh_fluxes();
        for (const auto face_lid : d_face_lids)
        {
            if (absolute_flux.is_owned_face(face_lid))
            {
                absolute_flux.set_owned_value(
                    face_lid, static_cast<scalar_type>(mesh_fluxes[static_cast<size_t>(face_lid)]));
            }
        }
        absolute_flux.sync_ghosts();
    }

    bool contains(local_ordinal_type face_lid) const noexcept
    {
        return std::ranges::find(d_face_lids, face_lid) != d_face_lids.end();
    }

private:
    void validate_collective_controls() const
    {
        const auto communicator = d_mesh->owned_cell_map()->getComm();
        const auto axis_index = static_cast<int>(d_axis);
        const std::array<int, 4> local_validity{d_boundary_name.empty() ? 1 : 0,
            d_boundary_name.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0,
            axis_index < static_cast<int>(Dimension::X) || axis_index > static_cast<int>(Dimension::Z) ? 1 : 0,
            !std::isfinite(d_volume_absolute_tolerance) || d_volume_absolute_tolerance < scalar_type{} ||
                    !std::isfinite(d_relative_tolerance) || d_relative_tolerance < scalar_type{}
                ? 1
                : 0};
        std::array<int, 4> global_validity{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_validity.size()),
            local_validity.data(), global_validity.data());
        if (global_validity[0] != 0 || global_validity[1] != 0)
        {
            throw std::invalid_argument(
                "PlanarALEBoundary requires one non-empty boundary name of representable length on every rank.");
        }
        if (global_validity[2] != 0)
        {
            throw std::invalid_argument("PlanarALEBoundary axis must be X, Y, or Z on every rank.");
        }
        if (global_validity[3] != 0)
        {
            throw std::invalid_argument(
                "PlanarALEBoundary volume-absolute and relative tolerances must be finite and non-negative on every "
                "rank.");
        }

        const auto local_name_size = static_cast<int>(d_boundary_name.size());
        int minimum_name_size = 0;
        int maximum_name_size = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_name_size, &minimum_name_size);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_name_size, &maximum_name_size);
        if (minimum_name_size != maximum_name_size)
        {
            throw std::invalid_argument("PlanarALEBoundary boundary name must match exactly on every rank.");
        }
        std::vector<int> local_name(static_cast<size_t>(local_name_size));
        std::transform(d_boundary_name.begin(), d_boundary_name.end(), local_name.begin(),
            [](unsigned char byte) { return static_cast<int>(byte); });
        std::vector<int> minimum_name(local_name.size());
        std::vector<int> maximum_name(local_name.size());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, local_name_size, local_name.data(), minimum_name.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, local_name_size, local_name.data(), maximum_name.data());
        if (minimum_name != maximum_name)
        {
            throw std::invalid_argument("PlanarALEBoundary boundary name must match exactly on every rank.");
        }

        int minimum_axis = 0;
        int maximum_axis = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &axis_index, &minimum_axis);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &axis_index, &maximum_axis);
        const std::array<scalar_type, 2> local_tolerances{d_volume_absolute_tolerance, d_relative_tolerance};
        std::array<scalar_type, 2> minimum_tolerances{};
        std::array<scalar_type, 2> maximum_tolerances{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_tolerances.size()),
            local_tolerances.data(), minimum_tolerances.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_tolerances.size()),
            local_tolerances.data(), maximum_tolerances.data());
        if (minimum_axis != maximum_axis || minimum_tolerances != maximum_tolerances)
        {
            throw std::invalid_argument("PlanarALEBoundary axis and tolerances must match exactly on every rank.");
        }
    }

    void validate(const VesselVolumeMap& volume_map)
    {
        const auto communicator = d_mesh->owned_cell_map()->getComm();
        const auto axis = static_cast<size_t>(d_axis);
        int local_named_batch = d_batch_id >= 0 ? 1 : 0;
        int global_named_batch = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_named_batch, &global_named_batch);
        if (global_named_batch == 0)
        {
            throw std::invalid_argument(
                "Planar ALE top boundary '" + d_boundary_name + "' does not exist on the mesh.");
        }

        int local_invalid = 0;
        scalar_type local_area{};
        scalar_type local_min_elevation = std::numeric_limits<scalar_type>::infinity();
        scalar_type local_max_elevation = -std::numeric_limits<scalar_type>::infinity();
        long long local_owned_faces = 0;
        for (const auto face_lid : d_face_lids)
        {
            local_invalid = local_invalid || !d_mesh->is_boundary_face(face_lid);
            const auto normal = d_mesh->face_normal(face_lid);
            const auto normal_axis = static_cast<scalar_type>(normal.component(axis));
            scalar_type tangential_squared{};
            for (size_t component = 0; component < 3; ++component)
            {
                if (component != axis)
                {
                    const auto value = static_cast<scalar_type>(normal.component(component));
                    tangential_squared += value * value;
                }
            }
            local_invalid = local_invalid || !std::isfinite(normal_axis) || normal_axis <= scalar_type{} ||
                            std::sqrt(tangential_squared) > scalar_type{1.0e-10};
            const auto elevation = static_cast<scalar_type>(d_mesh->face_centroid(face_lid).component(axis));
            local_invalid = local_invalid || !std::isfinite(elevation);
            local_min_elevation = std::min(local_min_elevation, elevation);
            local_max_elevation = std::max(local_max_elevation, elevation);
            if (d_mesh->is_owned_face(face_lid))
            {
                ++local_owned_faces;
                local_area += static_cast<scalar_type>(d_mesh->face_area(face_lid));
            }
        }

        scalar_type local_mesh_volume{};
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            local_mesh_volume += static_cast<scalar_type>(d_mesh->cell_volume(static_cast<local_ordinal_type>(owned)));
        }
        std::array<scalar_type, 2> local_sums{local_area, local_mesh_volume};
        std::array<scalar_type, 2> global_sums{};
        scalar_type global_min_elevation{};
        scalar_type global_max_elevation{};
        long long global_owned_faces = 0;
        int any_invalid = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 2, local_sums.data(), global_sums.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_owned_faces, &global_owned_faces);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_min_elevation, &global_min_elevation);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_max_elevation, &global_max_elevation);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        if (any_invalid != 0 || global_owned_faces == 0 || !(global_sums[0] > scalar_type{}))
        {
            throw std::invalid_argument(
                "Planar ALE top boundary must contain finite outward axis-normal boundary faces.");
        }

        const auto planar_scale =
            std::max({scalar_type{1}, std::abs(global_min_elevation), std::abs(global_max_elevation)});
        const auto planarity_absolute_tolerance = d_volume_absolute_tolerance / global_sums[0];
        if (std::abs(global_max_elevation - global_min_elevation) >
            planarity_absolute_tolerance + d_relative_tolerance * planar_scale)
        {
            throw std::invalid_argument("Planar ALE top boundary faces are not coplanar.");
        }
        const auto surface = scalar_type{0.5} * (global_min_elevation + global_max_elevation);
        const auto bottom_elevation = static_cast<scalar_type>(volume_map.bottomElevation());
        const auto top_elevation = static_cast<scalar_type>(volume_map.topElevation());
        const auto total_usable_volume = static_cast<scalar_type>(volume_map.totalUsableVolume());
        const auto range_policy = static_cast<int>(volume_map.rangePolicy());
        scalar_type mapped_pool_volume{};
        scalar_type mapped_area{};
        int local_invalid_map = !std::isfinite(bottom_elevation) || !std::isfinite(top_elevation) ||
                                        !(top_elevation > bottom_elevation) || !std::isfinite(total_usable_volume) ||
                                        !(total_usable_volume > scalar_type{}) ||
                                        (range_policy != static_cast<int>(FreeSurfaceRangePolicy::Error) &&
                                            range_policy != static_cast<int>(FreeSurfaceRangePolicy::ClampAndReport))
                                    ? 1
                                    : 0;
        if (local_invalid_map == 0)
        {
            try
            {
                mapped_pool_volume = static_cast<scalar_type>(volume_map.volumeBelow(surface));
                mapped_area = static_cast<scalar_type>(volume_map.areaAt(surface));
                local_invalid_map = !std::isfinite(mapped_pool_volume) || mapped_pool_volume < scalar_type{} ||
                                            !std::isfinite(mapped_area) || !(mapped_area > scalar_type{})
                                        ? 1
                                        : 0;
            }
            catch (...)
            {
                local_invalid_map = 1;
            }
        }
        int any_invalid_map = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_map, &any_invalid_map);
        if (any_invalid_map != 0)
        {
            throw std::invalid_argument(
                "PlanarALEBoundary vessel map must be valid at the moving surface on every rank.");
        }
        const std::array<scalar_type, 5> local_map{
            bottom_elevation, top_elevation, total_usable_volume, mapped_pool_volume, mapped_area};
        std::array<scalar_type, 5> minimum_map{};
        std::array<scalar_type, 5> maximum_map{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_map.size()), local_map.data(),
            minimum_map.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_map.size()), local_map.data(),
            maximum_map.data());
        int minimum_range_policy = 0;
        int maximum_range_policy = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &range_policy, &minimum_range_policy);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &range_policy, &maximum_range_policy);
        if (minimum_map != maximum_map || minimum_range_policy != maximum_range_policy)
        {
            throw std::invalid_argument(
                "PlanarALEBoundary vessel-map values and range policy must match exactly on every rank.");
        }
        const auto area_scale = std::max({scalar_type{1}, std::abs(mapped_area), std::abs(global_sums[0])});
        const auto vessel_height = top_elevation - bottom_elevation;
        const auto area_absolute_tolerance = d_volume_absolute_tolerance / vessel_height;
        if (std::abs(mapped_area - global_sums[0]) > area_absolute_tolerance + d_relative_tolerance * area_scale)
        {
            throw std::invalid_argument(
                "Planar ALE moving-patch area disagrees with the configured vessel cross-section.");
        }
        const auto volume_scale = std::max({scalar_type{1}, std::abs(mapped_pool_volume), std::abs(global_sums[1])});
        const auto mismatch = global_sums[1] - mapped_pool_volume;
        if (std::abs(mismatch) > d_volume_absolute_tolerance + d_relative_tolerance * volume_scale)
        {
            throw std::invalid_argument(
                "Planar ALE mesh volume disagrees with vessel-map volume below the moving surface.");
        }
        d_diagnostics = {surface, global_sums[0], global_sums[1], mapped_pool_volume, mismatch};
    }

    SP<const mesh_type> d_mesh;
    std::string d_boundary_name;
    Dimension d_axis = Dimension::Z;
    scalar_type d_volume_absolute_tolerance = {}; ///< [m^3]
    scalar_type d_relative_tolerance = {};
    int d_batch_id = -1;
    std::vector<local_ordinal_type> d_face_lids;
    Diagnostics d_diagnostics;
};

} // namespace SimpleFluid
