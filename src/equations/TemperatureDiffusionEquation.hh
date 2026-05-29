/**
 * @file TemperatureDiffusionEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume temperature diffusion and convection equation.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "FVM/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Finite-volume heat equation for cell-centered temperature.
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

    void advance_semi_implicit(
        const std::vector<scalar_type>& old_temperature,
        const std::vector<scalar_type>& face_fluxes,
        scalar_type time_step,
        scalar_type thermal_diffusivity,
        field_type& temperature,
        const LinearSolverOptions& linear_options = {}) const;

private:
    /**
     * @brief Cached Dirichlet boundary temperature value for a face.
     */
    struct BoundaryTemperature
    {
        scalar_type value = scalar_type{};
        std::uint8_t has_dirichlet = 0;
    };

    scalar_type cached_boundary_value(local_ordinal_type face_lid,
                                      scalar_type fallback) const;

    SP<const mesh_type> d_mesh;
    std::vector<BoundaryTemperature> d_face_boundary_temperature;
};

/**
 * @brief Construct a temperature diffusion equation with mesh and boundary conditions.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the assembled mesh.
 * @param boundary_conditions Boundary condition set for temperature field.
 * @throws std::invalid_argument if the mesh is null.
 */
template<TpetraTypePack Pack>
TemperatureDiffusionEquation<Pack>::TemperatureDiffusionEquation(
    SP<const mesh_type> mesh,
    const BoundaryConditionSet& boundary_conditions)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "TemperatureDiffusionEquation"))
{
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
    d_face_boundary_temperature.assign(d_mesh->num_faces(), BoundaryTemperature{});

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

        d_face_boundary_temperature[face] = {iter->second.value, 1};
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
    const auto& boundary = d_face_boundary_temperature[index];
    return boundary.has_dirichlet
         ? boundary.value
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
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step",
                                             "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity",
                                             "TemperatureDiffusionEquation");
    EquationValidation::assert_sufficient_cache_size(old_temperature.size(),
                                                     d_mesh->num_local_cells());

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
        temperature.set_owned_value(cell_lid,
                                    temp_p
                                  + time_step * thermal_diffusivity * laplacian);
    }
}

/**
 * @brief Advance temperature with semi-implicit upwind convection and diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_temperature Local-cell temperature values from the previous time level.
 * @param face_fluxes Owner-oriented integrated mass fluxes.
 * @param time_step Time-step size.
 * @param thermal_diffusivity Constant thermal diffusivity.
 * @param temperature Output temperature field over owned cells.
 * @param linear_options Belos solver options for the transport solve.
 */
template<TpetraTypePack Pack>
void TemperatureDiffusionEquation<Pack>::advance_semi_implicit(
    const std::vector<scalar_type>& old_temperature,
    const std::vector<scalar_type>& face_fluxes,
    scalar_type time_step,
    scalar_type thermal_diffusivity,
    field_type& temperature,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(time_step, "time step",
                                             "TemperatureDiffusionEquation");
    EquationValidation::require_non_negative(thermal_diffusivity, "diffusivity",
                                             "TemperatureDiffusionEquation");
    EquationValidation::assert_sufficient_cache_size(old_temperature.size(),
                                                     d_mesh->num_local_cells());

    auto boundary_value =
        [&](local_ordinal_type face_lid,
            scalar_type /*fallback*/) -> std::optional<scalar_type>
    {
        const auto& boundary =
            d_face_boundary_temperature[static_cast<std::size_t>(face_lid)];
        if (!boundary.has_dirichlet)
        {
            return std::nullopt;
        }
        return boundary.value;
    };

    auto system = FvmOperators::transport_system<Pack>(
        *d_mesh, old_temperature, face_fluxes, time_step,
        thermal_diffusivity, boundary_value);

    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto converged =
        solve_linear_system<Pack>(matrix, system.rhs,
                                  temperature.owned_data(), linear_options);
    if (!converged)
    {
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            if (!std::isfinite(temperature.value(cell_lid)))
            {
                throw std::runtime_error(
                    "TemperatureDiffusionEquation transport solve produced a non-finite value.");
            }
        }
    }
    temperature.sync_ghosts();
}

} // namespace SimpleFluid
