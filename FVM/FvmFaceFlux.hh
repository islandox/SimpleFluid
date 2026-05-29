/**
 * @file FvmFaceFlux.hh
 * @brief Face-flux assembly and velocity-boundary cache helpers.
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

template<TpetraTypePack Pack>
struct VelocityBoundaryCache
{
    using vec_type = typename Mesh<Pack>::Vec3;

    explicit VelocityBoundaryCache(SP<const Mesh<Pack>> mesh)
        : value(std::move(mesh), "velocity_boundary")
    {
    }

    VectorFaceField<Pack> value;
    std::vector<std::uint8_t> has_value;
};

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

template<TpetraTypePack Pack>
VelocityBoundaryCache<Pack> cache_velocity_boundary_conditions(
    const Mesh<Pack>& mesh,
    const BoundaryConditionSet& boundary_conditions)
{
    SP<const Mesh<Pack>> mesh_ptr(&mesh, [](const Mesh<Pack>*) {});
    return cache_velocity_boundary_conditions<Pack>(mesh_ptr, boundary_conditions);
}

namespace detail
{

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

template<TpetraTypePack Pack>
void assemble_face_fluxes(const Mesh<Pack>& mesh,
                          const VectorCellField<Pack>& velocity,
                          const VelocityBoundaryCache<Pack>* boundary_cache,
                          std::vector<typename Pack::scalar_type>& fluxes)
{
    validate_face_flux_inputs(mesh, velocity, boundary_cache);
    fluxes.assign(mesh.num_faces(), typename Pack::scalar_type{});

    for (std::size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        const auto owner = mesh.owner_cell(face_lid);
        auto face_velocity = velocity.local_value(owner);

        if (mesh.is_interior_face(face_lid))
        {
            const auto neighbor = mesh.neighbor_cell(face_lid);
            face_velocity = (face_velocity + velocity.local_value(neighbor)) / 2.0;
        }
        else if (!boundary_face_velocity(mesh, face_lid, boundary_cache,
                                         face_velocity))
        {
            continue;
        }

        fluxes[face] = face_velocity.dot(mesh.face_normal(face_lid))
                     * mesh.face_area(face_lid);
    }
}

} // namespace detail

template<TpetraTypePack Pack>
void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    detail::assemble_face_fluxes<Pack>(mesh, velocity, nullptr, fluxes);
}

template<TpetraTypePack Pack>
void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 const VelocityBoundaryCache<Pack>& boundary_cache,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    detail::assemble_face_fluxes(mesh, velocity, &boundary_cache, fluxes);
}

template<TpetraTypePack Pack>
void face_fluxes(const Mesh<Pack>& mesh,
                 const VectorCellField<Pack>& velocity,
                 const BoundaryConditionSet& boundary_conditions,
                 std::vector<typename Pack::scalar_type>& fluxes)
{
    const auto cache =
        cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    face_fluxes(mesh, velocity, cache, fluxes);
}

template<TpetraTypePack Pack>
void face_fluxes(const Mesh<Pack>& mesh,
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

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, fluxes);

    return fluxes;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const BoundaryConditionSet& boundary_conditions)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_conditions, fluxes);

    return fluxes;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const BoundaryConditionSet* boundary_conditions)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_conditions, fluxes);

    return fluxes;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const VectorCellField<Pack>& velocity,
            const VelocityBoundaryCache<Pack>& boundary_cache)
{
    std::vector<typename Pack::scalar_type> fluxes;
    face_fluxes(mesh, velocity, boundary_cache, fluxes);

    return fluxes;
}

} // namespace SimpleFluid::FvmOperators
