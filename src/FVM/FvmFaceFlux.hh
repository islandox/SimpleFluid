/**
 * @file FvmFaceFlux.hh
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
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace SimpleFluid::FvmOperators
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
        : value(), mesh(mesh)
    {
    }

    std::unordered_map<int, Arr<vec_type>> value;
    SP<const Mesh<Pack>> mesh;
};

/**
 * @brief Build a velocity-boundary cache from a shared mesh pointer and a
 *        set of boundary conditions.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param boundary_conditions The boundary-condition set to evaluate.
 * @return VelocityBoundaryCache populated with prescribed non-periodic
 *         boundary values. Periodic faces are left to the paired-cell
 *         interpolation path.
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

    for (const auto& [patch_id, boundary_patch] : mesh->boundary_patches())
    {
        typename VelocityBoundaryCache<Pack>::vec_type prescribed_value{};

        const auto iter =
            boundary_conditions.velocity.find(mesh->boundary_patch_name(patch_id));
        if (iter != boundary_conditions.velocity.end())
        {
            if (iter->second.type == BoundaryConditionType::Periodic)
            {
                prescribed_value = {};
            }
            else if (iter->second.type == BoundaryConditionType::NoSlip)
            {
                prescribed_value = {0.0, 0.0, 0.0};
            }
            else if (iter->second.type == BoundaryConditionType::Dirichlet)
            {
                prescribed_value = iter->second.value;
            }
        }

        Arr<typename VelocityBoundaryCache<Pack>::vec_type> patch_values(
            boundary_patch.face_lids.size(), prescribed_value);
        cache.value[patch_id] = std::move(patch_values);
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
    if (boundary_cache != nullptr && boundary_cache->value.size() != mesh.boundary_patches().size())
    {
        throw std::invalid_argument("face_fluxes received the wrong boundary-cache size.");
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
 * @brief Retrieve the prescribed velocity on boundary faces from the
 *        cache, if available.
 *
 * @tparam Pack The Tpetra type pack.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @param[out] face_velocity On return, the cached boundary velocity.
 */
template<TpetraTypePack Pack>
void load_boundary_face_velocity(
    const VelocityBoundaryCache<Pack>* boundary_cache,
    VectorFaceField<Pack>& face_velocity)
{
    if (boundary_cache == nullptr) return;

    const auto& mesh = face_velocity.mesh();
    for (auto [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        if (boundary_patch.face_lids.empty())
        {
            continue;
        }

        const auto iter = boundary_cache->value.find(patch_id);
        if (iter == boundary_cache->value.end())
        {
            continue;
        }

        for (size_t i = 0; i < boundary_patch.face_lids.size(); ++i)
        {
            const auto face_lid = boundary_patch.face_lids[i];
            if (!face_velocity.is_owned_face(face_lid)) continue;
            if (mesh.is_periodic_boundary_face(face_lid)) continue;

            face_velocity.set_value(face_lid, iter->second[i]);
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

    for (std::size_t face = 0; face < mesh.num_faces(); ++face)
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
            const auto neighbor = mesh.neighbor_cell(face_lid);
            value = (value + velocity.local_value(neighbor)) / 2.0;
            face_velocity.set_value(face_lid, value);
        }
        else if (mesh.is_periodic_boundary_face(face_lid))
        {
            const auto paired = mesh.periodic_neighbor_cell(face_lid);
            value = (value + velocity.local_value(paired)) / 2.0;
            face_velocity.set_value(face_lid, value);
        }
    }
    load_boundary_face_velocity(boundary_cache, face_velocity);
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

    for (std::size_t face = 0; face < mesh.num_faces(); ++face)
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
    VectorFaceField<Pack> face_velocity(velocity.mesh_ptr(), "face_velocity");
    face_velocities(velocity, face_velocity);
    normal_face_fluxes(face_velocity, fluxes);
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
    VectorFaceField<Pack> face_velocity(velocity.mesh_ptr(), "face_velocity");
    face_velocities(velocity, boundary_cache, face_velocity);
    normal_face_fluxes(face_velocity, fluxes);
}

} // namespace SimpleFluid::FvmOperators
