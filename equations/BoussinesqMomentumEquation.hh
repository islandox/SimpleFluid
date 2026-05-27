/**
 * @file BoussinesqMomentumEquation.hh
 * @brief Boussinesq buoyancy update for vertical velocity.
 */
#pragma once

#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Explicit Boussinesq momentum update for the vertical velocity component.
 *
 * The current solver stores velocity component-wise. This equation class keeps
 * the existing vertical-buoyancy model isolated from solver orchestration.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqMomentumEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    void advance_vertical_velocity(const std::vector<scalar_type>& old_velocity_z,
                                   const field_type& temperature,
                                   const TimeStepperOptions& options,
                                   field_type& velocity_z) const;

private:
    SP<const mesh_type> d_mesh;
};

template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(std::move(mesh))
{
    if (!d_mesh)
    {
        throw std::invalid_argument("BoussinesqMomentumEquation requires a non-null mesh.");
    }
}

/**
 * @brief Advance vertical velocity using Boussinesq buoyancy and linear damping.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity_z Local-cell velocity values from the previous time level.
 * @param temperature Updated temperature field.
 * @param options Physical and time-stepping parameters.
 * @param velocity_z Output vertical velocity field over owned cells.
 */
template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_vertical_velocity(
    const std::vector<scalar_type>& old_velocity_z,
    const field_type& temperature,
    const TimeStepperOptions& options,
    field_type& velocity_z) const
{
    if (&temperature.mesh() != d_mesh.get() || &velocity_z.mesh() != d_mesh.get())
    {
        throw std::invalid_argument("BoussinesqMomentumEquation field mesh mismatch.");
    }
    if (options.time_step < 0.0)
    {
        throw std::invalid_argument("BoussinesqMomentumEquation requires non-negative time step.");
    }
    if (options.kinematic_viscosity < 0.0)
    {
        throw std::invalid_argument("BoussinesqMomentumEquation requires non-negative viscosity.");
    }
    if (old_velocity_z.size() < d_mesh->num_local_cells())
    {
        throw std::invalid_argument("BoussinesqMomentumEquation velocity cache is too small.");
    }

    const auto damping =
        1.0 / (1.0 + options.time_step * options.kinematic_viscosity);

    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto buoyancy =
            options.thermal_expansion
          * (temperature.value(cell_lid) - options.reference_temperature)
          * (-options.gravity_z);

        velocity_z.set_value(cell_lid,
                             (old_velocity_z[cell]
                            + options.time_step * buoyancy) * damping);
    }
}

} // namespace SimpleFluid
