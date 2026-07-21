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

#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"
#include "FVM/CellOperators.hh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Cache of velocity values prescribed on boundary faces.
 *
 * @tparam Pack The Tpetra type pack governing local ordinals and scalars.
 */
template<TpetraTypePack Pack>
struct VelocityBoundaryCache
{
    using vec_type = vec3<typename Pack::scalar_type>;

    /**
     * @brief Construct a velocity-boundary cache backed by the given mesh.
     *
     * @param mesh The mesh whose boundary faces will be cached.
     */
    explicit VelocityBoundaryCache(SP<const Mesh<Pack>> mesh)
        : value(), type(), type_by_name(), mesh(mesh)
    {
    }

    std::unordered_map<int, Arr<vec_type>> value;
    std::unordered_map<int, BoundaryConditionType> type;
    /// Configured velocity types replicated by name on every rank.
    std::unordered_map<std::string, BoundaryConditionType> type_by_name;
    SP<const Mesh<Pack>> mesh;
};

/**
 * @brief Reusable scratch storage for Rhie-Chow face-flux reconstruction.
 *
 * The workspace owns only temporary pressure-gradient values. Every
 * invocation of pressure_weighted_face_fluxes overwrites the values needed by
 * that call; no pressure-derived result is reused. Boundary-face locations
 * depend only on the immutable mesh topology and are cached at construction.
 *
 * A workspace is tied to one exact mesh instance. It is move-only to prevent
 * scratch storage from being shallow-copied, and is not safe for concurrent
 * use by multiple face-flux evaluations.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
class PressureWeightedFaceFluxWorkspace
{
public:
    using mesh_type = Mesh<Pack>;
    using boundary_location_type =
        detail::BoundaryFaceLocation<mesh_type>;

    /**
     * @brief Allocate scratch fields and cache boundary locations for a mesh.
     * @param mesh Shared pointer to an assembled computational mesh.
     * @throws std::invalid_argument if @p mesh is null.
     */
    explicit PressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh)
        : d_mesh(require_mesh(std::move(mesh))),
          d_pressure_gradient(
              d_mesh, "rhie_chow_pressure_gradient_workspace", false),
          d_boundary_locations(
              detail::boundary_face_locations(*d_mesh))
    {
    }

    PressureWeightedFaceFluxWorkspace(
        const PressureWeightedFaceFluxWorkspace&) = delete;
    PressureWeightedFaceFluxWorkspace& operator=(
        const PressureWeightedFaceFluxWorkspace&) = delete;
    PressureWeightedFaceFluxWorkspace(
        PressureWeightedFaceFluxWorkspace&&) = default;
    PressureWeightedFaceFluxWorkspace& operator=(
        PressureWeightedFaceFluxWorkspace&&) = default;

    const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }

    VectorCellField<Pack>& pressure_gradient() noexcept
    {
        return d_pressure_gradient;
    }

    const VectorCellField<Pack>& pressure_gradient() const noexcept
    {
        return d_pressure_gradient;
    }

    const std::vector<boundary_location_type>&
    boundary_locations() const noexcept
    {
        return d_boundary_locations;
    }

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "PressureWeightedFaceFluxWorkspace requires a non-null "
                "mesh.");
        }
        return mesh;
    }

    SP<const mesh_type> d_mesh;
    VectorCellField<Pack> d_pressure_gradient;
    std::vector<boundary_location_type> d_boundary_locations;
};

/**
 * @brief Build a velocity-boundary cache from a shared mesh pointer and a
 *        set of boundary conditions.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param boundary_conditions The boundary-condition set to evaluate.
 * @return VelocityBoundaryCache populated with prescribed non-periodic
 *         boundary values. Velocity Neumann data must be homogeneous;
 *         periodic faces are left to the paired-cell interpolation path.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack>
VelocityBoundaryCache<Pack> cache_velocity_boundary_conditions(
    SP<const Mesh<Pack>> mesh,
    const BoundaryConditionSet& boundary_conditions)
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "cache_velocity_boundary_conditions requires a non-null mesh.");
    }

    VelocityBoundaryCache<Pack> cache(mesh);
    for (const auto& [name, condition] : boundary_conditions.velocity)
    {
        cache.type_by_name[name] = condition.type;
    }

    for (const auto& [batch_id, boundary_batch] : mesh->boundary_batches())
    {
        typename VelocityBoundaryCache<Pack>::vec_type prescribed_value{};
        auto boundary_type = BoundaryConditionType::Neumann;

        const auto iter =
            boundary_conditions.velocity.find(mesh->boundary_batch_name(batch_id));
        if (iter != boundary_conditions.velocity.end())
        {
            boundary_type = iter->second.type;
            if (boundary_type == BoundaryConditionType::Periodic)
            {
                prescribed_value = {};
            }
            else if (boundary_type == BoundaryConditionType::NoSlip)
            {
                prescribed_value = {0.0, 0.0, 0.0};
            }
            else if (boundary_type == BoundaryConditionType::Dirichlet)
            {
                prescribed_value = iter->second.value;
            }
            else if (boundary_type == BoundaryConditionType::Neumann
                     && (iter->second.value.x != 0.0
                         || iter->second.value.y != 0.0
                         || iter->second.value.z != 0.0))
            {
                throw std::invalid_argument(
                    "Cache-based incompressible velocity transport supports "
                    "homogeneous Neumann outlet conditions only.");
            }
        }

        Arr<typename VelocityBoundaryCache<Pack>::vec_type> batch_values(
            boundary_batch.face_lids.size(), prescribed_value);
        cache.value[batch_id] = std::move(batch_values);
        cache.type[batch_id] = boundary_type;
    }

    return cache;
}

namespace detail
{

/**
 * @brief Validate that the optional boundary cache matches the velocity
 *        field mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @throws std::invalid_argument if the velocity field or boundary cache
 *         is not associated with the velocity mesh.
 */
template<TpetraTypePack Pack>
void validate_face_flux_inputs(
    const VectorCellField<Pack>& velocity,
    const VelocityBoundaryCache<Pack>* boundary_cache)
{
    const auto& mesh = velocity.mesh();
    if (boundary_cache != nullptr && boundary_cache->value.size() != mesh.boundary_batches().size())
    {
        throw std::invalid_argument("face_fluxes received the wrong boundary-cache size.");
    }
    if (boundary_cache != nullptr && boundary_cache->type.size() != mesh.boundary_batches().size())
    {
        throw std::invalid_argument("face_fluxes received the wrong boundary-cache type size.");
    }
    if (boundary_cache != nullptr && boundary_cache->mesh != velocity.mesh_ptr())
    {
        throw std::invalid_argument("face_fluxes received a boundary cache for another mesh.");
    }
}

/**
 * @brief Validate that the output face-velocity field is associated with
 *        the velocity mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param face_velocity Face-centered velocity field to validate.
 * @throws std::invalid_argument if @p face_velocity is not on the velocity mesh.
 */
template<TpetraTypePack Pack>
void validate_face_velocity_output(
    const VectorCellField<Pack>& velocity,
    const VectorFaceField<Pack>& face_velocity)
{
    if (&face_velocity.mesh() != &velocity.mesh())
    {
        throw std::invalid_argument(
            "face_velocities requires output on the velocity mesh.");
    }
}

/**
 * @brief Validate that normal-flux output is on the face-velocity mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param face_velocity Face-centered velocity field to validate.
 * @param fluxes Scalar face-flux field to validate.
 * @throws std::invalid_argument if @p fluxes is not on the face-velocity mesh.
 */
template<TpetraTypePack Pack>
void validate_normal_flux_inputs(
    const VectorFaceField<Pack>& face_velocity,
    const FaceField<Pack>& fluxes)
{
    if (&fluxes.mesh() != &face_velocity.mesh())
    {
        throw std::invalid_argument(
            "normal_face_fluxes requires output on the face-velocity mesh.");
    }
}

/**
 * @brief Project owner-cell velocity onto the tangent plane of a boundary face.
 *
 * @tparam Pack Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param face_lid Local ID of the boundary face.
 * @return Velocity component tangential to the boundary face.
 */
template<TpetraTypePack Pack>
auto slip_face_velocity(const VectorCellField<Pack>& velocity,
                        typename Pack::local_ordinal_type face_lid)
    -> typename VectorCellField<Pack>::vec_type
{
    const auto& mesh = velocity.mesh();
    const auto owner = mesh.owner_cell(face_lid);
    const auto cell_velocity = velocity.local_value(owner);
    const auto& normal = mesh.face_normal_outward(face_lid, owner);

    return cell_velocity - normal * cell_velocity.dot(normal);
}

/**
 * @brief Retrieve the prescribed velocity on boundary faces from the
 *        cache, if available.
 *
 * @tparam Pack Tpetra type pack.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @param velocity Cell-centered velocity field (used for slip condition
 *        computation).
 * @param[out] face_velocity On return, the cached boundary velocity.
 */
template<TpetraTypePack Pack>
void load_boundary_face_velocity(
    const VelocityBoundaryCache<Pack>* boundary_cache,
    const VectorCellField<Pack>& velocity,
    VectorFaceField<Pack>& face_velocity)
{
    if (boundary_cache == nullptr) return;

    const auto& mesh = face_velocity.mesh();
    for (auto [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        if (boundary_batch.face_lids.empty())
        {
            continue;
        }

        const auto iter = boundary_cache->value.find(batch_id);
        if (iter == boundary_cache->value.end())
        {
            continue;
        }
        const auto type_iter = boundary_cache->type.find(batch_id);
        const auto boundary_type =
            type_iter == boundary_cache->type.end()
          ? BoundaryConditionType::Neumann
          : type_iter->second;

        for (size_t i = 0; i < boundary_batch.face_lids.size(); ++i)
        {
            const auto face_lid = boundary_batch.face_lids[i];
            if (!face_velocity.is_owned_face(face_lid)) continue;
            if (!mesh.is_boundary_face(face_lid)) continue;

            if (boundary_type == BoundaryConditionType::Slip)
            {
                face_velocity.set_value(face_lid,
                    slip_face_velocity(velocity, face_lid));
            }
            else
            {
                face_velocity.set_value(face_lid, iter->second[i]);
            }
        }
    }
}

/**
 * @brief Assemble face-centered velocities by averaging cell-centered
 *        values at interior faces and applying boundary conditions at
 *        boundary faces.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
void assemble_face_velocities(const VectorCellField<Pack>& velocity,
                              const VelocityBoundaryCache<Pack>* boundary_cache,
                              VectorFaceField<Pack>& face_velocity)
{
    validate_face_flux_inputs(velocity, boundary_cache);
    validate_face_velocity_output(velocity, face_velocity);
    const auto& mesh = velocity.mesh();
    face_velocity.put_scalar(typename Mesh<Pack>::Vec3{});

    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        if (!face_velocity.is_owned_face(face_lid))
        {
            continue;
        }

        const auto owner = mesh.owner_cell(face_lid);
        auto value = velocity.local_value(owner);

        if (mesh.is_interior_face(face_lid))
        {
            const auto neighbor =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, owner);
            value = (value + velocity.local_value(neighbor)) / 2.0;
            face_velocity.set_value(face_lid, value);
        }
    }
    load_boundary_face_velocity(boundary_cache, velocity, face_velocity);
}

/**
 * @brief Assemble normal face fluxes directly from cell velocities.
 *
 * This is algebraically equivalent to assemble_face_velocities followed by
 * normal_face_fluxes, but avoids materializing and rereading a three-component
 * face-velocity field. A null @p boundary_cache skips boundary treatment.
 */
template<TpetraTypePack Pack>
void assemble_normal_face_fluxes(
    const VectorCellField<Pack>& velocity,
    const VelocityBoundaryCache<Pack>* boundary_cache,
    FaceField<Pack>& fluxes)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    validate_face_flux_inputs(velocity, boundary_cache);
    if (&fluxes.mesh() != &velocity.mesh())
    {
        throw std::invalid_argument(
            "face_fluxes requires output on the velocity mesh.");
    }

    const auto& mesh = velocity.mesh();
    fluxes.put_scalar(scalar_type{});

    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!fluxes.is_owned_face(face_lid)
            || !mesh.is_interior_face(face_lid))
        {
            continue;
        }

        const auto owner = mesh.owner_cell(face_lid);
        const auto neighbor =
            mesh.opposite_or_periodic_neighbor_cell(face_lid, owner);
        const auto face_velocity =
            (velocity.local_value(owner)
             + velocity.local_value(neighbor)) / 2.0;
        fluxes.set_value(
            face_lid,
            face_velocity.dot(mesh.face_normal(face_lid))
          * mesh.face_area(face_lid));
    }

    if (boundary_cache == nullptr)
    {
        return;
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        const auto value_iter = boundary_cache->value.find(batch_id);
        if (value_iter == boundary_cache->value.end())
        {
            continue;
        }
        const auto type_iter = boundary_cache->type.find(batch_id);
        const auto boundary_type =
            type_iter == boundary_cache->type.end()
          ? BoundaryConditionType::Neumann
          : type_iter->second;

        for (size_t i = 0; i < boundary_batch.face_lids.size(); ++i)
        {
            const auto face_lid = boundary_batch.face_lids[i];
            if (!fluxes.is_owned_face(face_lid)
                || !mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto face_velocity =
                boundary_type == BoundaryConditionType::Slip
              ? slip_face_velocity(velocity, face_lid)
              : value_iter->second[i];
            fluxes.set_value(
                face_lid,
                face_velocity.dot(mesh.face_normal(face_lid))
              * mesh.face_area(face_lid));
        }
    }
}

} // namespace detail

/**
 * @brief Assemble face velocities from cell-centered velocities without
 *        boundary-condition treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const VectorCellField<Pack>& velocity,
                            VectorFaceField<Pack>& face_velocity)
{
    detail::assemble_face_velocities<Pack>(velocity, nullptr,
                                           face_velocity);
}

/**
 * @brief Assemble face velocities using a pre-built velocity-boundary
 *        cache.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const VectorCellField<Pack>& velocity,
                            const VelocityBoundaryCache<Pack>& boundary_cache,
                            VectorFaceField<Pack>& face_velocity)
{
    detail::assemble_face_velocities(velocity, &boundary_cache,
                                     face_velocity);
}

/**
 * @brief Compute the normal volumetric flux (velocity dot normal times
 *        area) at every owned face.
 *
 * @tparam Pack The Tpetra type pack.
 * @param face_velocity Face-centered velocity field.
 * @param[out] fluxes Pre-allocated FaceField to receive normal fluxes.
 */
template<TpetraTypePack Pack>
void normal_face_fluxes(
    const VectorFaceField<Pack>& face_velocity,
    FaceField<Pack>& fluxes)
{
    detail::validate_normal_flux_inputs(face_velocity, fluxes);
    const auto& mesh = face_velocity.mesh();
    fluxes.put_scalar(typename Pack::scalar_type{});

    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        if (!fluxes.is_owned_face(face_lid))
        {
            continue;
        }

        fluxes.set_value(face_lid,
            face_velocity.value(face_lid).dot(mesh.face_normal(face_lid))
          * mesh.face_area(face_lid));
    }
}

/**
 * @brief Compute face fluxes from cell-centered velocities without
 *        boundary-condition treatment, writing to a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param[out] fluxes Pre-allocated FaceField to receive fluxes.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const VectorCellField<Pack>& velocity,
                        FaceField<Pack>& fluxes)
{
    detail::assemble_normal_face_fluxes<Pack>(velocity, nullptr, fluxes);
}

/**
 * @brief Compute face fluxes from cell-centered velocities using a
 *        pre-built velocity-boundary cache, writing to a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[out] fluxes Pre-allocated FaceField to receive fluxes.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const VectorCellField<Pack>& velocity,
                        const VelocityBoundaryCache<Pack>& boundary_cache,
                        FaceField<Pack>& fluxes)
{
    detail::assemble_normal_face_fluxes(
        velocity, &boundary_cache, fluxes);
}

namespace detail
{

/**
 * @brief Validate the open-boundary contract used for pressure outlets.
 *
 * Dirichlet pressure extrapolates owner-cell velocity to the boundary face.
 * It therefore requires a Neumann velocity condition; a prescribed velocity
 * would otherwise be silently discarded by the pressure-flux reconstruction.
 *
 * @throws std::invalid_argument if a pressure condition is unsupported or a
 *         Dirichlet pressure boundary has a non-Neumann velocity condition.
 */
template<TpetraTypePack Pack>
void validate_pressure_velocity_boundary_compatibility(
    const VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions)
{
    // Both maps come directly from the globally configured boundary data, so
    // every rank makes the same decision before a caller enters collectives.
    for (const auto& [name, pressure_condition] :
         pressure_boundary_conditions)
    {
        if (pressure_condition.type != BoundaryConditionType::Dirichlet
            && pressure_condition.type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument(
                "pressure_weighted_face_fluxes supports only Dirichlet "
                "and Neumann pressure boundary conditions.");
        }
        if (pressure_condition.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        const auto velocity_iter =
            velocity_boundary_cache.type_by_name.find(name);
        const auto velocity_type =
            velocity_iter == velocity_boundary_cache.type_by_name.end()
          ? BoundaryConditionType::Neumann
          : velocity_iter->second;
        if (velocity_type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument(
                "Dirichlet pressure boundary '" + name
                + "' requires a Neumann velocity boundary so owner-cell "
                  "velocity can be extrapolated to the open face.");
        }
    }
}

/**
 * @brief Compute Rhie-Chow pressure-weighted face fluxes.
 *
 * The correction replaces the interpolated cell pressure gradient in the
 * normal face velocity with the direct owner-neighbor pressure difference.
 * It vanishes for a linearly reconstructed pressure field and couples
 * alternating collocated-cell pressure modes to continuity.
 *
 * @tparam Pack The Tpetra type pack.
 * @param velocity Cell-centered velocity field.
 * @param pressure Cell-centered pressure field.
 * @param pressure_coefficient Velocity-pressure coefficient. For pressure
 *        stored in Pa, this is normally the time step divided by the
 *        reference density.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[in,out] workspace Scratch storage tied to the field mesh.
 * @param[out] fluxes Pre-allocated FaceField to receive stabilized fluxes.
 */
template<TpetraTypePack Pack>
void pressure_weighted_face_fluxes_impl(
    const VectorCellField<Pack>& velocity,
    const CellField<Pack>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCache<Pack>& boundary_cache,
    const BoundaryConditionMap* pressure_boundary_conditions,
    PressureWeightedFaceFluxWorkspace<Pack>& workspace,
    FaceField<Pack>& fluxes)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    if (&pressure.mesh() != &velocity.mesh()
        || &fluxes.mesh() != &velocity.mesh())
    {
        throw std::invalid_argument(
            "pressure_weighted_face_fluxes requires fields on one mesh.");
    }
    if (workspace.mesh_ptr() != velocity.mesh_ptr()
        || workspace.boundary_locations().size()
           != velocity.mesh().num_faces())
    {
        throw std::invalid_argument(
            "pressure_weighted_face_fluxes received a workspace for "
            "another mesh.");
    }
    if (pressure_coefficient < scalar_type{})
    {
        throw std::invalid_argument(
            "pressure_weighted_face_fluxes requires a non-negative "
            "pressure coefficient.");
    }
    if (pressure_boundary_conditions != nullptr)
    {
        validate_pressure_velocity_boundary_compatibility(
            boundary_cache,
            *pressure_boundary_conditions);
    }

    assemble_normal_face_fluxes(
        velocity, &boundary_cache, fluxes);
    if (pressure_coefficient == scalar_type{})
    {
        return;
    }

    const auto& mesh = velocity.mesh();
    auto& pressure_gradient = workspace.pressure_gradient();
    if (pressure_boundary_conditions == nullptr)
    {
        cell_gradient(pressure, pressure_gradient);
    }
    else
    {
        scalar_cell_gradient(
            pressure,
            pressure_boundary_conditions,
            pressure_gradient,
            &workspace.boundary_locations());
    }
    pressure_gradient.sync_ghosts();
    const auto& boundary_locations = workspace.boundary_locations();

    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!fluxes.is_owned_face(face_lid))
        {
            continue;
        }

        if (!mesh.is_interior_face(face_lid))
        {
            if (pressure_boundary_conditions == nullptr
                || !mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                   >= boundary_locations.size())
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            if (!location.active)
            {
                continue;
            }
            const auto name =
                mesh.boundary_batch_name(location.batch_id);
            const auto condition_iter =
                pressure_boundary_conditions->find(name);
            const auto condition =
                condition_iter == pressure_boundary_conditions->end()
              ? BoundaryCondition{}
              : condition_iter->second;
            if (condition.type == BoundaryConditionType::Neumann)
            {
                continue;
            }
            if (condition.type != BoundaryConditionType::Dirichlet)
            {
                throw std::invalid_argument(
                    "pressure_weighted_face_fluxes supports only Dirichlet "
                    "and Neumann pressure boundary conditions.");
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto area_vector =
                mesh.face_area_vector_outward(face_lid, owner);
            fluxes.set_value(
                face_lid,
                velocity.local_value(owner).dot(area_vector));
            const auto direct_gradient_flux =
                (condition.value
                 - pressure.local_value(owner))
              * boundary_diffusion_coefficient(
                    mesh, face_lid, owner, scalar_type{1});
            const auto interpolated_gradient_flux =
                pressure_gradient.local_value(owner).dot(area_vector);
            fluxes.sum_into_value(
                face_lid,
                -pressure_coefficient
                * (direct_gradient_flux
                   - interpolated_gradient_flux));
            continue;
        }

        const auto owner = mesh.owner_cell(face_lid);
        const auto neighbor =
            mesh.opposite_or_periodic_neighbor_cell(face_lid, owner);
        const auto center_delta =
            mesh.cell_center_vector(face_lid, owner);
        const auto distance_squared = center_delta.dot(center_delta);
        if (distance_squared <= scalar_type{})
        {
            continue;
        }

        const auto area_vector = mesh.face_area_vector(face_lid);
        const auto direct_gradient_flux =
            (pressure.local_value(neighbor)
             - pressure.local_value(owner))
            * area_vector.dot(center_delta) / distance_squared;
        const auto interpolated_gradient =
            (pressure_gradient.local_value(owner)
             + pressure_gradient.local_value(neighbor))
            / scalar_type{2};
        const auto interpolated_gradient_flux =
            interpolated_gradient.dot(area_vector);

        fluxes.sum_into_value(
            face_lid,
            -pressure_coefficient
            * (direct_gradient_flux - interpolated_gradient_flux));
    }
}

} // namespace detail

/**
 * @brief Compute Rhie-Chow pressure-weighted face fluxes using cell-only
 *        pressure reconstruction and reusable scratch storage.
 *
 * @param velocity Cell-centered velocity field.
 * @param pressure Cell-centered pressure field.
 * @param pressure_coefficient Velocity-pressure coefficient.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[in,out] workspace Scratch storage tied to the field mesh.
 * @param[out] fluxes Pre-allocated face-flux output.
 */
template<TpetraTypePack Pack>
void pressure_weighted_face_fluxes(
    const VectorCellField<Pack>& velocity,
    const CellField<Pack>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCache<Pack>& boundary_cache,
    PressureWeightedFaceFluxWorkspace<Pack>& workspace,
    FaceField<Pack>& fluxes)
{
    detail::pressure_weighted_face_fluxes_impl(
        velocity, pressure, pressure_coefficient,
        boundary_cache, nullptr, workspace, fluxes);
}

/**
 * @brief Compute Rhie-Chow pressure-weighted face fluxes using cell-only
 *        pressure reconstruction and one-shot scratch storage.
 */
template<TpetraTypePack Pack>
void pressure_weighted_face_fluxes(
    const VectorCellField<Pack>& velocity,
    const CellField<Pack>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCache<Pack>& boundary_cache,
    FaceField<Pack>& fluxes)
{
    PressureWeightedFaceFluxWorkspace<Pack> workspace(
        velocity.mesh_ptr());
    pressure_weighted_face_fluxes(
        velocity, pressure, pressure_coefficient,
        boundary_cache, workspace, fluxes);
}

/**
 * @brief Compute pressure-weighted face fluxes including pressure boundary
 *        reconstruction.
 *
 * A Dirichlet pressure face is treated as an open boundary: owner velocity
 * is extrapolated to the face, then its pressure-gradient flux is replaced
 * by the direct owner-to-boundary gradient. Such a face must have a Neumann
 * velocity condition; prescribed velocity conditions are rejected. Missing
 * pressure batch names, including every batch in an empty map, default to
 * homogeneous Neumann conditions.
 *
 * @param velocity Cell-centered velocity field.
 * @param pressure Cell-centered pressure field.
 * @param pressure_coefficient Velocity-pressure coefficient.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param pressure_boundary_conditions Pressure boundary-condition map.
 * @param[in,out] workspace Scratch storage tied to the field mesh.
 * @param[out] fluxes Pre-allocated face-flux output.
 */
template<TpetraTypePack Pack>
void pressure_weighted_face_fluxes(
    const VectorCellField<Pack>& velocity,
    const CellField<Pack>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCache<Pack>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    PressureWeightedFaceFluxWorkspace<Pack>& workspace,
    FaceField<Pack>& fluxes)
{
    detail::pressure_weighted_face_fluxes_impl(
        velocity, pressure, pressure_coefficient,
        boundary_cache, &pressure_boundary_conditions, workspace, fluxes);
}

/**
 * @brief Compute pressure-weighted face fluxes including pressure boundary
 *        reconstruction and one-shot scratch storage.
 */
template<TpetraTypePack Pack>
void pressure_weighted_face_fluxes(
    const VectorCellField<Pack>& velocity,
    const CellField<Pack>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const VelocityBoundaryCache<Pack>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FaceField<Pack>& fluxes)
{
    PressureWeightedFaceFluxWorkspace<Pack> workspace(
        velocity.mesh_ptr());
    pressure_weighted_face_fluxes(
        velocity, pressure, pressure_coefficient,
        boundary_cache, pressure_boundary_conditions,
        workspace, fluxes);
}

} // namespace SimpleFluid::FVM
