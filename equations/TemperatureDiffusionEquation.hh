/**
 * @file TemperatureDiffusionEquation.hh
 * @brief Explicit finite-volume temperature diffusion equation.
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Explicit finite-volume heat equation for cell-centered temperature.
 *
 * The class owns the boundary-condition lookup needed by the equation while
 * the caller owns field storage and time integration order.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class TemperatureDiffusionEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    TemperatureDiffusionEquation(SP<const mesh_type> mesh,
                                 const BoundaryConditionSet& boundary_conditions);

    void refresh_boundary_cache(const BoundaryConditionSet& boundary_conditions);

    void advance_explicit(const std::vector<scalar_type>& old_temperature,
                          scalar_type time_step,
                          scalar_type thermal_diffusivity,
                          field_type& temperature) const;

private:
    scalar_type cached_boundary_value(local_ordinal_type face_lid,
                                      scalar_type fallback) const;

    SP<const mesh_type> d_mesh;
    std::vector<scalar_type> d_face_boundary_temperature;
    std::vector<std::uint8_t> d_face_has_dirichlet_temperature;
};

template<TpetraTypePack Pack>
TemperatureDiffusionEquation<Pack>::TemperatureDiffusionEquation(
    SP<const mesh_type> mesh,
    const BoundaryConditionSet& boundary_conditions)
    : d_mesh(std::move(mesh))
{
    if (!d_mesh)
    {
        throw std::invalid_argument("TemperatureDiffusionEquation requires a non-null mesh.");
    }

    refresh_boundary_cache(boundary_conditions);
}

/**
 * @brief Rebuild the cached lookup of Dirichlet temperature boundary values.
 *
 * @tparam Pack Tpetra type pack.
 * @param boundary_conditions Current boundary-condition set.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::refresh_boundary_cache(
    const BoundaryConditionSet& boundary_conditions)
{
    d_face_boundary_temperature.assign(d_mesh->num_faces(), scalar_type{});
    d_face_has_dirichlet_temperature.assign(d_mesh->num_faces(), std::uint8_t{0});

    for (std::size_t face = 0; face < d_mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!d_mesh->is_boundary_face(face_lid))
        {
            continue;
        }

        const auto iter =
            boundary_conditions.temperature.find(d_mesh->boundary_name(face_lid));
        if (iter == boundary_conditions.temperature.end()
            || iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        d_face_boundary_temperature[face] = iter->second.value;
        d_face_has_dirichlet_temperature[face] = 1;
    }
}

/**
 * @brief Retrieve the cached Dirichlet temperature for a face, or a fallback value.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_lid Local face index.
 * @param fallback Value returned when the face has no Dirichlet condition.
 * @return Cached boundary temperature or the fallback scalar.
 */
template<TpetraTypePack Pack>
auto TemperatureDiffusionEquation<Pack>::cached_boundary_value(
    local_ordinal_type face_lid,
    scalar_type fallback) const -> scalar_type
{
    const auto index = static_cast<std::size_t>(face_lid);
    return d_face_has_dirichlet_temperature[index]
         ? d_face_boundary_temperature[index]
         : fallback;
}

/**
 * @brief Advance temperature with explicit thermal diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Local-cell temperature values from the previous time level.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Constant thermal diffusivity.
 * @param temperature Output temperature field over owned cells.
 * @throws std::invalid_argument if the field belongs to a different mesh or
 *         coefficients are non-physical.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_explicit(
    const std::vector<scalar_type>& old_temperature,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature) const
{
    if (&temperature.mesh() != d_mesh.get())
    {
        throw std::invalid_argument("TemperatureDiffusionEquation field mesh mismatch.");
    }
    if (time_step < 0.0)
    {
        throw std::invalid_argument("TemperatureDiffusionEquation requires non-negative time step.");
    }
    if (thermal_diffusivity < 0.0)
    {
        throw std::invalid_argument("TemperatureDiffusionEquation requires non-negative diffusivity.");
    }
    if (old_temperature.size() < d_mesh->num_local_cells())
    {
        throw std::invalid_argument("TemperatureDiffusionEquation old temperature cache is too small.");
    }

    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto temp_p = old_temperature[cell];
        scalar_type laplacian = 0.0;

        const auto& faces = d_mesh->faces(cell_lid);
        const auto& face_distances = d_mesh->face_distances(cell_lid);
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
        {
            const auto face_lid = faces[face_index];

            if (d_mesh->is_interior_face(face_lid))
            {
                const auto other = d_mesh->opposite_cell(face_lid, cell_lid);
                const auto distance = d_mesh->face_cell_center_distance(face_lid);
                if (distance > 0.0)
                {
                    laplacian +=
                        (old_temperature[static_cast<std::size_t>(other)] - temp_p)
                      * d_mesh->face_area(face_lid)
                      / distance;
                }
            }
            else if (const auto distance_to_face = face_distances[face_index];
                     distance_to_face > 0.0)
            {
                const auto boundary_temperature =
                    cached_boundary_value(face_lid, temp_p);
                laplacian += (boundary_temperature - temp_p)
                           * d_mesh->face_area(face_lid)
                           / distance_to_face;
            }
        }

        laplacian /= d_mesh->cell_volume(cell_lid);
        temperature.set_value(cell_lid,
                              temp_p
                            + time_step * thermal_diffusivity * laplacian);
    }
}

} // namespace SimpleFluid
