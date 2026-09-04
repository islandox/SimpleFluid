/** @file IF97Water.hh
 * @brief Optional pure-water material properties in SI units.
 */
#pragma once

#include <string>

namespace SimpleFluid::IF97Water
{

/** A homogeneous water/steam state. No void or mixture correction is applied. */
struct Properties
{
    double temperature;            ///< K
    double absolute_pressure;      ///< Pa, never gauge or kinematic pressure
    double density;                ///< kg/m^3
    double specific_heat_capacity; ///< Isobaric cp, J/(kg K)
    double specific_enthalpy;      ///< J/kg, with the IF97 reference state
    double specific_entropy;       ///< J/(kg K), with the IF97 reference state
    double dynamic_viscosity;      ///< Pa s
    double thermal_conductivity;   ///< W/(m K)

    double kinematic_viscosity() const noexcept { return dynamic_viscosity / density; }
    double thermal_diffusivity() const noexcept { return thermal_conductivity / (density * specific_heat_capacity); }
};

/** Separate saturated phases at one pressure, below the critical point. */
struct SaturationProperties
{
    Properties liquid;
    Properties vapor;
    double surface_tension; ///< N/m

    double latent_heat() const noexcept { return vapor.specific_enthalpy - liquid.specific_enthalpy; }
};

/**
 * Evaluate the stable homogeneous phase at temperature [K], pressure [Pa].
 * The material API covers 273.15 <= T <= 1073.15 K and
 * 611.213 <= p <= 100 MPa (IF97 regions 1, 2, 3). Region 5 is excluded
 * because this API also returns transport properties.
 * Saturation states are ambiguous in (T,p); use saturation_at_pressure().
 * Non-finite/non-positive inputs throw std::invalid_argument; unsupported
 * states throw std::out_of_range. No extrapolation or clipping is performed.
 */
Properties evaluate(double temperature, double absolute_pressure);

/**
 * Evaluate liquid water, including saturated liquid. Reject vapor and
 * T >= 647.096 K instead of silently supplying a different phase.
 */
Properties liquid(double temperature, double absolute_pressure);

/** Both saturation phases for 611.213 <= p < 22.064 MPa. */
SaturationProperties saturation_at_pressure(double absolute_pressure);

/** Saturation line including its critical endpoint; inputs/outputs are SI. */
double saturation_temperature(double absolute_pressure);
double saturation_pressure(double temperature);

/** Surface tension along saturation for 273.15 <= T <= 647.096 K [N/m]. */
double surface_tension(double temperature);

/**
 * Liquid isobaric expansion coefficient -(1/rho) d(rho)/dT [1/K].
 * A second-order density difference uses a 0.001 K increment, with a
 * one-sided stencil at domain, saturation, and region 1/3 boundaries. Intended for
 * reference-state Boussinesq linearization; can be negative near freezing.
 */
double liquid_thermal_expansion(double temperature, double absolute_pressure);

/** Version of the linked CoolProp IF97 implementation. */
std::string backend_version();

} // namespace SimpleFluid::IF97Water
