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
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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
    using vec_type = typename Mesh<Pack>::Vec3;

    /**
     * @brief Construct a velocity-boundary cache backed by the given mesh.
     *
     * @param mesh The mesh whose boundary faces will be cached.
     */
    explicit VelocityBoundaryCache(SP<const Mesh<Pack>> mesh)
        : value(std::move(mesh), "velocity_boundary")
    {
    }

    VectorFaceField<Pack> value;
    std::vector<std::uint8_t> has_value;
};

/**
 * @brief Build a velocity-boundary cache from a shared mesh pointer and a
 *        set of boundary conditions.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param boundary_conditions The boundary-condition set to evaluate.
 * @return VelocityBoundaryCache populated with Dirichlet and no-slip values.
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
    cache.has_value.assign(mesh->num_faces(), 0);

    for (std::size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        if (!mesh->is_boundary_face(face_lid))
        {
            continue;
        }

        const auto iter =
            boundary_conditions.velocity.find(mesh->boundary_name(face_lid));
        if (iter == boundary_conditions.velocity.end())
        {
            continue;
        }

        if (iter->second.type == BoundaryConditionType::NoSlip)
        {
            if (!cache.value.is_owned_face(face_lid))
            {
                continue;
            }
            cache.value.set_value(face_lid, {});
            cache.has_value[face] = 1;
        }
        else if (iter->second.type == BoundaryConditionType::Dirichlet)
        {
            if (!cache.value.is_owned_face(face_lid))
            {
                continue;
            }
            cache.value.set_value(face_lid, iter->second.value);
            cache.has_value[face] = 1;
        }
    }

    return cache;
}

/**
 * @brief Build a velocity-boundary cache from a mesh reference.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh Reference to the computational mesh.
 * @param boundary_conditions The boundary-condition set to evaluate.
 * @return VelocityBoundaryCache populated with Dirichlet and no-slip values.
 */
template<TpetraTypePack Pack>
inline VelocityBoundaryCache<Pack> cache_velocity_boundary_conditions(
    const Mesh<Pack>& mesh,
    const BoundaryConditionSet& boundary_conditions)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    return cache_velocity_boundary_conditions<Pack>(mesh_ptr, boundary_conditions);
}

namespace detail
{

/**
 * @brief Validate that the velocity field and optional boundary cache are
 *        consistent with the supplied mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @throws std::invalid_argument if the velocity field or boundary cache
 *         is not associated with @p mesh.
 */
template<TpetraTypePack Pack>
void validate_face_flux_inputs(
    const Mesh<Pack>& mesh,
    const VectorCellField<Pack>& velocity,
    const VelocityBoundaryCache<Pack>* boundary_cache)
{
    if (&velocity.mesh() != &mesh)
    {
        throw std::invalid_argument("face_fluxes requires a velocity field on the input mesh.");
    }
    if (boundary_cache != nullptr
        && (boundary_cache->has_value.size() != mesh.num_faces()
            || &boundary_cache->value.mesh() != &mesh))
    {
        throw std::invalid_argument("face_fluxes received the wrong boundary-cache size.");
    }
}

/**
 * @brief Validate that the output face-velocity field is associated with
 *        the given mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_velocity Face-centered velocity field to validate.
 * @throws std::invalid_argument if @p face_velocity is not on @p mesh.
 */
template<TpetraTypePack Pack>
void validate_face_velocity_output(
    const Mesh<Pack>& mesh,
    const VectorFaceField<Pack>& face_velocity)
{
    if (&face_velocity.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "face_velocities requires an output field on the input mesh.");
    }
}

/**
 * @brief Validate that the face-velocity field used for normal-flux
 *        computation is associated with the given mesh.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_velocity Face-centered velocity field to validate.
 * @throws std::invalid_argument if @p face_velocity is not on @p mesh.
 */
template<TpetraTypePack Pack>
void validate_normal_flux_inputs(
    const Mesh<Pack>& mesh,
    const VectorFaceField<Pack>& face_velocity)
{
    if (&face_velocity.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "normal_face_fluxes requires a face-velocity field on the input mesh.");
    }
}

/**
 * @brief Retrieve the prescribed velocity on a boundary face from the
 *        cache, if available.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_lid Local index of the face to query.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @param[out] face_velocity On return, the cached boundary velocity.
 * @return true if a cached value was written to @p face_velocity, false
 *         otherwise.
 */
template<TpetraTypePack Pack>
bool boundary_face_velocity(
    const Mesh<Pack>& mesh,
    typename Pack::local_ordinal_type face_lid,
    const VelocityBoundaryCache<Pack>* boundary_cache,
    typename Mesh<Pack>::Vec3& face_velocity)
{
    if (boundary_cache == nullptr || !mesh.is_boundary_face(face_lid))
    {
        return false;
    }

    const auto face = static_cast<std::size_t>(face_lid);
    if (!boundary_cache->has_value[face])
    {
        return false;
    }

    face_velocity = boundary_cache->value.value(face_lid);
    return true;
}

/**
 * @brief Assemble face-centered velocities by averaging cell-centered
 *        values at interior faces and applying boundary conditions at
 *        boundary faces.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pointer to a velocity-boundary cache, or nullptr.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
void assemble_face_velocities(const Mesh<Pack>& mesh,
                              const VectorCellField<Pack>& velocity,
                              const VelocityBoundaryCache<Pack>* boundary_cache,
                              VectorFaceField<Pack>& face_velocity)
{
    validate_face_flux_inputs(mesh, velocity, boundary_cache);
    validate_face_velocity_output(mesh, face_velocity);
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
        }
        else if (!boundary_face_velocity(mesh, face_lid, boundary_cache, value))
        {
            continue;
        }

        face_velocity.set_value(face_lid, value);
    }
}

} // namespace detail

/**
 * @brief Assemble face velocities from cell-centered velocities without
 *        boundary-condition treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const Mesh<Pack>& mesh,
                     const VectorCellField<Pack>& velocity,
                     VectorFaceField<Pack>& face_velocity)
{
    detail::assemble_face_velocities<Pack>(mesh, velocity, nullptr,
                                           face_velocity);
}

/**
 * @brief Assemble face velocities using a pre-built velocity-boundary
 *        cache.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const Mesh<Pack>& mesh,
                     const VectorCellField<Pack>& velocity,
                     const VelocityBoundaryCache<Pack>& boundary_cache,
                     VectorFaceField<Pack>& face_velocity)
{
    detail::assemble_face_velocities(mesh, velocity, &boundary_cache,
                                     face_velocity);
}

/**
 * @brief Assemble face velocities using a boundary-condition set,
 *        building a temporary velocity-boundary cache internally.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions The boundary-condition set to apply.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const Mesh<Pack>& mesh,
                     const VectorCellField<Pack>& velocity,
                     const BoundaryConditionSet& boundary_conditions,
                     VectorFaceField<Pack>& face_velocity)
{
    const auto cache =
        cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    face_velocities(mesh, velocity, cache, face_velocity);
}

/**
 * @brief Assemble face velocities with optional boundary conditions
 *        supplied as a pointer (nullptr means no boundary treatment).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions Pointer to boundary-condition set, may be
 *        nullptr.
 * @param[in,out] face_velocity On output, the assembled face velocities.
 */
template<TpetraTypePack Pack>
inline void face_velocities(const Mesh<Pack>& mesh,
                     const VectorCellField<Pack>& velocity,
                     const BoundaryConditionSet* boundary_conditions,
                     VectorFaceField<Pack>& face_velocity)
{
    if (boundary_conditions == nullptr)
    {
        face_velocities(mesh, velocity, face_velocity);
        return;
    }

    face_velocities(mesh, velocity, *boundary_conditions, face_velocity);
}

/**
 * @brief Assemble and return face velocities without boundary-condition
 *        treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @return Newly allocated VectorFaceField with the assembled face
 *         velocities.
 */
template<TpetraTypePack Pack>
inline VectorFaceField<Pack>
face_velocities(const Mesh<Pack>& mesh,
                const VectorCellField<Pack>& velocity)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    VectorFaceField<Pack> face_velocity(mesh_ptr, "face_velocity");
    face_velocities(mesh, velocity, face_velocity);

    return face_velocity;
}

/**
 * @brief Assemble and return face velocities with boundary-condition
 *        treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions The boundary-condition set to apply.
 * @return Newly allocated VectorFaceField with the assembled face
 *         velocities.
 */
template<TpetraTypePack Pack>
inline VectorFaceField<Pack>
face_velocities(const Mesh<Pack>& mesh,
                const VectorCellField<Pack>& velocity,
                const BoundaryConditionSet& boundary_conditions)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    VectorFaceField<Pack> face_velocity(mesh_ptr, "face_velocity");
    face_velocities(mesh, velocity, boundary_conditions, face_velocity);

    return face_velocity;
}

/**
 * @brief Assemble and return face velocities with optional boundary
 *        conditions (nullptr means no boundary treatment).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions Pointer to boundary-condition set, may be
 *        nullptr.
 * @return Newly allocated VectorFaceField with the assembled face
 *         velocities.
 */
template<TpetraTypePack Pack>
inline VectorFaceField<Pack>
face_velocities(const Mesh<Pack>& mesh,
                const VectorCellField<Pack>& velocity,
                const BoundaryConditionSet* boundary_conditions)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    VectorFaceField<Pack> face_velocity(mesh_ptr, "face_velocity");
    face_velocities(mesh, velocity, boundary_conditions, face_velocity);

    return face_velocity;
}

/**
 * @brief Assemble and return face velocities using a pre-built
 *        velocity-boundary cache.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @return Newly allocated VectorFaceField with the assembled face
 *         velocities.
 */
template<TpetraTypePack Pack>
inline VectorFaceField<Pack>
face_velocities(const Mesh<Pack>& mesh,
                const VectorCellField<Pack>& velocity,
                const VelocityBoundaryCache<Pack>& boundary_cache)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    VectorFaceField<Pack> face_velocity(mesh_ptr, "face_velocity");
    face_velocities(mesh, velocity, boundary_cache, face_velocity);

    return face_velocity;
}

/**
 * @brief Compute the normal volumetric flux (velocity dot normal times
 *        area) at every owned face.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_velocity Face-centered velocity field.
 * @param[out] fluxes Pre-allocated vector to receive the flux at each
 *        face.
 */
template<TpetraTypePack Pack>
void normal_face_fluxes(
    const Mesh<Pack>& mesh,
    const VectorFaceField<Pack>& face_velocity,
    std::vector<typename Pack::scalar_type>& fluxes)
{
    detail::validate_normal_flux_inputs(mesh, face_velocity);
    fluxes.assign(mesh.num_faces(), typename Pack::scalar_type{});

    for (std::size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        if (!face_velocity.is_owned_face(face_lid))
        {
            continue;
        }

        fluxes[face] = face_velocity.value(face_lid).dot(mesh.face_normal(face_lid))
                     * mesh.face_area(face_lid);
    }
}

/**
 * @brief Compute and return the normal volumetric flux at every owned
 *        face.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_velocity Face-centered velocity field.
 * @return Vector of normal fluxes indexed by face local ID.
 */
template<TpetraTypePack Pack>
inline std::vector<typename Pack::scalar_type>
normal_face_fluxes(const Mesh<Pack>& mesh,
                   const VectorFaceField<Pack>& face_velocity)
{
    std::vector<typename Pack::scalar_type> fluxes;
    normal_face_fluxes(mesh, face_velocity, fluxes);

    return fluxes;
}

/**
 * @brief Compute face fluxes from cell-centered velocities without
 *        boundary-condition treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param[out] fluxes Pre-allocated vector to receive the flux at each
 *        face.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    const auto face_velocity = face_velocities(mesh, velocity);
    normal_face_fluxes(mesh, face_velocity, fluxes);
}

/**
 * @brief Compute face fluxes from cell-centered velocities using a
 *        pre-built velocity-boundary cache.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @param[out] fluxes Pre-allocated vector to receive the flux at each
 *        face.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 const VelocityBoundaryCache<Pack>& boundary_cache,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    const auto face_velocity = face_velocities(mesh, velocity, boundary_cache);
    normal_face_fluxes(mesh, face_velocity, fluxes);
}

/**
 * @brief Compute face fluxes from cell-centered velocities using a
 *        boundary-condition set.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions The boundary-condition set to apply.
 * @param[out] fluxes Pre-allocated vector to receive the flux at each
 *        face.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 const BoundaryConditionSet& boundary_conditions,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    const auto cache =
        cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    face_fluxes(mesh, velocity, cache, fluxes);
}

/**
 * @brief Compute face fluxes from cell-centered velocities with optional
 *        boundary conditions (nullptr means no boundary treatment).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions Pointer to boundary-condition set, may be
 *        nullptr.
 * @param[out] fluxes Pre-allocated vector to receive the flux at each
 *        face.
 */
template<TpetraTypePack Pack>
inline void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 const BoundaryConditionSet* boundary_conditions,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    if (boundary_conditions == nullptr)
    {
        face_fluxes(mesh, velocity, fluxes);
        return;
    }

    face_fluxes(mesh, velocity, *boundary_conditions, fluxes);
}

/**
 * @brief Compute and return face fluxes from cell-centered velocities
 *        without boundary-condition treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @return Vector of face fluxes indexed by face local ID.
 */
template<TpetraTypePack Pack>
inline std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, fluxes);

    return fluxes;
}

/**
 * @brief Compute and return face fluxes from cell-centered velocities
 *        with boundary-condition treatment.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions The boundary-condition set to apply.
 * @return Vector of face fluxes indexed by face local ID.
 */
template<TpetraTypePack Pack>
inline std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const BoundaryConditionSet& boundary_conditions)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_conditions, fluxes);

    return fluxes;
}

/**
 * @brief Compute and return face fluxes from cell-centered velocities
 *        with optional boundary conditions (nullptr means no boundary
 *        treatment).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_conditions Pointer to boundary-condition set, may be
 *        nullptr.
 * @return Vector of face fluxes indexed by face local ID.
 */
template<TpetraTypePack Pack>
inline std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const BoundaryConditionSet* boundary_conditions)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_conditions, fluxes);

    return fluxes;
}

/**
 * @brief Compute and return face fluxes from cell-centered velocities
 *        using a pre-built velocity-boundary cache.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param velocity Cell-centered velocity field.
 * @param boundary_cache Pre-computed velocity-boundary cache.
 * @return Vector of face fluxes indexed by face local ID.
 */
template<TpetraTypePack Pack>
inline std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const VelocityBoundaryCache<Pack>& boundary_cache)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_cache, fluxes);

    return fluxes;
}

} // namespace SimpleFluid::FvmOperators
