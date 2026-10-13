#pragma once

#include "equations/LumpedInnerCavityModel.hh"

namespace SimpleFluid
{
struct VentedLumpedCavityOptions
{
    real_t ambient_pressure = 101325.0;
    real_t ambient_temperature = 300.0;
    real_t dry_air_cp = 1005.0;
    real_t dry_air_gas_constant = 287.05;
    real_t minimum_temperature = 200.0;
    real_t maximum_temperature = 1000.0;
};

struct VentedLumpedCavityState
{
    real_t temperature = 300.0;
    real_t volume = 1.0;
};

struct VentedLumpedCavityStep
{
    VentedLumpedCavityState state;
    LumpedInnerCavityExchange exchange;
    real_t vent_mass_transfer = 0.0;     ///< kg, positive enters the cavity.
    real_t vent_enthalpy_transfer = 0.0; ///< J, positive enters; h = cp T.
    real_t volume_work = 0.0;           ///< J, p (V_new - V_old), positive done by gas.
    real_t mass_balance_residual = 0.0;
    real_t energy_balance_residual = 0.0; ///< U_new - U_old - Q - H_vent + work, J.
};

/** Ambient-pressure, perfectly mixed ideal dry air with a prescribed moving volume.
 * Backward Euler uses h frozen at the saved air temperature and implicit surface
 * temperature differences. Net vent inflow has ambient enthalpy; net outflow has
 * final cavity enthalpy. Thus H_new-H_old = dt Q + dm cp T_upwind, with
 * H = m cp T and U = H-pV. Surface heat rates are positive leaving the surface.
 * Water exchange and radiation are disabled. The caller owns all other gas and
 * liquid inventories. preview is pure, including on failure; advance commits
 * only successful steps. Scalar inputs require caller-owned MPI reductions.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT VentedLumpedCavityModel
{
public:
    VentedLumpedCavityModel(VentedLumpedCavityOptions options, VentedLumpedCavityState state);
    const VentedLumpedCavityOptions& options() const noexcept { return d_options; }
    const VentedLumpedCavityState& state() const noexcept { return d_state; }
    void set_state(VentedLumpedCavityState state);
    real_t pressure() const noexcept { return d_options.ambient_pressure; }
    real_t dry_air_mass() const;
    real_t enthalpy() const;
    real_t internal_energy() const;
    VentedLumpedCavityStep preview(real_t time_step, real_t new_volume,
        std::span<const InnerCavitySurface> surfaces) const;
    VentedLumpedCavityStep advance(real_t time_step, real_t new_volume,
        std::span<const InnerCavitySurface> surfaces);
private:
    VentedLumpedCavityOptions d_options;
    VentedLumpedCavityState d_state;
};
} // namespace SimpleFluid
