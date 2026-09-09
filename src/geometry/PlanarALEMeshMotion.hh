/**
 * @file PlanarALEMeshMotion.hh
 * @brief Transactional planar motion for structured and extruded meshes.
 */

#pragma once

#include "geometry/MeshHandle.hh"
#include "geometry/MeshMotionModel.hh"

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack> class PlanarALEMeshMotion;

} // namespace SimpleFluid

namespace SimpleFluid::Meshes
{

/** @brief Narrow friend seam for geometry-only, same-count edge updates. */
class PlanarALEGeometryAccess
{
    template<::SimpleFluid::TpetraTypePack Pack> friend class ::SimpleFluid::PlanarALEMeshMotion;

private:
    template<class Mesh> static bool motion_available(const Mesh& mesh) noexcept
    {
        return mesh.d_geometry_state.motion_owner == nullptr;
    }

    template<class Mesh> static bool motion_owned_by(const Mesh& mesh, const void* owner) noexcept
    {
        return mesh.d_geometry_state.motion_owner == owner;
    }

    template<class Mesh> static void claim_motion(Mesh& mesh, const void* owner)
    {
        if (mesh.d_geometry_state.motion_owner != nullptr)
        {
            throw std::logic_error("Planar ALE geometry already has a motion controller.");
        }
        mesh.d_geometry_state.motion_owner = owner;
    }

    template<class Mesh> static void release_motion(Mesh& mesh, const void* owner) noexcept
    {
        if (mesh.d_geometry_state.motion_owner == owner)
        {
            mesh.d_geometry_state.motion_owner = nullptr;
        }
    }

    template<class Mesh> static void require_motion_owner(const Mesh& mesh, const void* owner)
    {
        if (mesh.d_geometry_state.motion_owner != owner)
        {
            throw std::logic_error("Planar ALE geometry is owned by another motion controller.");
        }
    }

    static void replace_axis_edges(OrthogonalCartesian3D& mesh, size_t axis, Arr<real_t> edges)
    {
        mesh.replace_axis_edges_fixed_topology(axis, std::move(edges));
    }

    static void replace_axial_edges(OrthogonalCylindrial3D& mesh, Arr<real_t> edges)
    {
        mesh.replace_axial_edges_fixed_topology(std::move(edges));
    }

    static void replace_axial_edges(SemiStructuredXY_Z& mesh, Arr<real_t> edges)
    {
        mesh.replace_axial_edges_fixed_topology(std::move(edges));
    }
};

} // namespace SimpleFluid::Meshes

namespace SimpleFluid
{

/** @brief Configuration for exact extruded-column planar motion. */
struct PlanarALEMeshMotionOptions
{
    Dimension axis = Dimension::Z;
    /** Elevation below which reference points remain fixed [m]. */
    std::optional<real_t> deformation_start_elevation;
    /** Optional accepted-to-trial surface displacement limit [m]. */
    std::optional<real_t> maximum_level_change;
    MeshQualityLimits quality_limits;
    real_t gcl_absolute_tolerance = 1.0e-12; ///< [m^3/s]
    real_t gcl_relative_tolerance = 1.0e-10;
};

/**
 * @brief Fixed-topology planar deformation with exact swept flux for extrusions.
 *
 * Cartesian meshes support any Cartesian axis. Cylindrical and
 * SemiStructuredXY_Z meshes support only their axial Z direction. Structured
 * Cartesian/cylindrical handles retain their existing MPI partition and maps;
 * SemiStructuredXY_Z retains its established serial-only contract.
 *
 * This class supplies geometry/GCL primitives only. It does not alter finite-
 * volume transport, pressure coupling, or free-surface boundary conditions.
 * Exactly one controller leases a concrete geometry object at a time, even
 * when multiple MeshHandle objects share it. Callers should resolve every
 * trial explicitly; destruction rolls back only when the geometry still
 * matches that controller's active trial. Direct mutation through
 * MeshHandle::visit_mutable() cannot publish an epoch and remains unsupported
 * after fields or caches are constructed.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_PUBLIC_TYPE PlanarALEMeshMotion final : public MeshMotionModel
{
public:
    using mesh_type = MeshHandle<Pack>;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    PlanarALEMeshMotion(SP<mesh_type> mesh, PlanarALEMeshMotionOptions options = {});
    ~PlanarALEMeshMotion() override;

    PlanarALEMeshMotion(const PlanarALEMeshMotion&) = delete;
    PlanarALEMeshMotion& operator=(const PlanarALEMeshMotion&) = delete;
    PlanarALEMeshMotion(PlanarALEMeshMotion&&) = delete;
    PlanarALEMeshMotion& operator=(PlanarALEMeshMotion&&) = delete;

    void begin_trial(real_t surface_elevation, real_t time_step) override;
    void accept_trial() override;
    void rollback_trial() override;

    [[nodiscard]] bool has_active_trial() const noexcept override { return d_trial_active; }

    [[nodiscard]] std::string_view mesh_family() const noexcept override { return d_family_name; }

    [[nodiscard]] std::span<const real_t> old_cell_volumes() const noexcept override { return d_old_cell_volumes; }

    [[nodiscard]] std::span<const real_t> new_cell_volumes() const noexcept override { return d_new_cell_volumes; }

    [[nodiscard]] std::span<const real_t> face_mesh_fluxes() const noexcept override { return d_face_mesh_fluxes; }

    [[nodiscard]] const MeshMotionDiagnostics& diagnostics() const noexcept override { return d_diagnostics; }

    [[nodiscard]] const SP<mesh_type>& mesh_ptr() const noexcept { return d_mesh; }

    [[nodiscard]] const PlanarALEMeshMotionOptions& options() const noexcept { return d_options; }

private:
    enum class Family : int
    {
        Unsupported = 0,
        Cartesian = 1,
        Cylindrical = 2,
        SemiStructured = 3
    };

    enum class TransactionAction : int
    {
        Accept = 1,
        Rollback = 2
    };

    [[nodiscard]] SIMPLEFLUID_LOCAL Family detect_family() const noexcept;
    [[nodiscard]] SIMPLEFLUID_LOCAL ArrReal current_axis_edges() const;
    [[nodiscard]] SIMPLEFLUID_LOCAL std::array<ArrReal, 3> geometry_edge_coordinates() const;
    [[nodiscard]] SIMPLEFLUID_LOCAL bool geometry_motion_available() const noexcept;
    [[nodiscard]] SIMPLEFLUID_LOCAL bool geometry_motion_owned() const noexcept;
    SIMPLEFLUID_LOCAL void claim_geometry_motion();
    SIMPLEFLUID_LOCAL void release_geometry_motion() noexcept;
    SIMPLEFLUID_LOCAL void replace_axis_edges(ArrReal edges);
    [[nodiscard]] SIMPLEFLUID_LOCAL ArrReal candidate_axis_edges(real_t surface_elevation) const;
    [[nodiscard]] SIMPLEFLUID_LOCAL std::array<ArrReal, 3> candidate_geometry_edges(real_t surface_elevation) const;
    [[nodiscard]] SIMPLEFLUID_LOCAL std::vector<real_t> capture_cell_volumes() const;
    [[nodiscard]] SIMPLEFLUID_LOCAL std::vector<typename mesh_type::Vec3> capture_face_centroids() const;
    SIMPLEFLUID_LOCAL void validate_collective_construction();
    SIMPLEFLUID_LOCAL void validate_collective_trial(real_t surface_elevation, real_t time_step) const;
    SIMPLEFLUID_LOCAL void validate_collective_transaction(TransactionAction action) const;
    SIMPLEFLUID_LOCAL void compute_trial_state(
        const std::vector<typename mesh_type::Vec3>& old_face_centroids, real_t time_step);
    SIMPLEFLUID_LOCAL void rollback_impl();
    SIMPLEFLUID_LOCAL void reset_stationary_state();

    SP<mesh_type> d_mesh;
    PlanarALEMeshMotionOptions d_options;
    Family d_family = Family::Unsupported;
    std::string d_family_name;
    std::array<ArrReal, 3> d_reference_geometry_edges;
    ArrReal d_reference_axis_edges;
    ArrReal d_deformation_weights;
    ArrReal d_pre_trial_axis_edges;
    std::array<ArrReal, 3> d_trial_geometry_edges;
    real_t d_reference_surface_elevation = {};
    real_t d_accepted_surface_elevation = {};
    std::uint64_t d_expected_geometry_epoch = 0;
    bool d_geometry_motion_claimed = false;
    bool d_trial_active = false;
    MeshQualityMetrics d_accepted_quality;
    std::vector<real_t> d_old_cell_volumes;
    std::vector<real_t> d_new_cell_volumes;
    std::vector<real_t> d_face_mesh_fluxes;
    MeshMotionDiagnostics d_diagnostics;
};

extern template class PlanarALEMeshMotion<DefaultTpetraTypes>;

} // namespace SimpleFluid

#include "geometry/PlanarALEMeshMotion.tcc"
