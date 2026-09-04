#include "materials/IF97Water.hh"

// This private dependency must never change the public SI-unit contract.
#ifdef IAPWS_UNITS
#error "SimpleFluid IF97 requires SI units; remove the IAPWS_UNITS definition."
#endif
#include <IF97.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace SimpleFluid::IF97Water
{
namespace
{
void require_positive(double value, const char* label)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string("IF97 water ") + label + " must be finite and positive.");
}

void validate_state(double temperature, double pressure)
{
    require_positive(temperature, "temperature [K]");
    require_positive(pressure, "absolute pressure [Pa]");
    if (temperature < IF97::Tmin || temperature > IF97::Tmax || pressure < IF97::Pmin || pressure > IF97::Pmax)
        throw std::out_of_range("IF97 water material range is 273.15..1073.15 K and 611.213..1e8 Pa.");
}

bool on_saturation(double temperature, double pressure)
{
    return pressure <= IF97::Pcrit && std::abs(temperature - IF97::Tsat97(pressure)) <= 1.0e-7;
}

Properties validate_properties(Properties result)
{
    const double positive[]{
        result.density, result.specific_heat_capacity, result.dynamic_viscosity, result.thermal_conductivity};
    for (const auto value : positive)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::out_of_range("IF97 water returned an invalid material property.");
    if (!std::isfinite(result.specific_enthalpy) || !std::isfinite(result.specific_entropy))
        throw std::out_of_range("IF97 water returned a non-finite thermodynamic property.");
    return result;
}

Properties saturated_phase(double temperature, double pressure, bool vapor)
{
    return validate_properties({temperature, pressure, vapor ? IF97::rhovap_p(pressure) : IF97::rholiq_p(pressure),
        vapor ? IF97::cpvap_p(pressure) : IF97::cpliq_p(pressure),
        vapor ? IF97::hvap_p(pressure) : IF97::hliq_p(pressure),
        vapor ? IF97::svap_p(pressure) : IF97::sliq_p(pressure),
        vapor ? IF97::viscvap_p(pressure) : IF97::viscliq_p(pressure),
        vapor ? IF97::tcondvap_p(pressure) : IF97::tcondliq_p(pressure)});
}
} // namespace

Properties evaluate(double temperature, double absolute_pressure)
{
    validate_state(temperature, absolute_pressure);
    if (on_saturation(temperature, absolute_pressure))
        throw std::out_of_range("IF97 (T,p) state is on saturation; use saturation_at_pressure to select a phase.");
    return validate_properties({temperature, absolute_pressure, IF97::rhomass_Tp(temperature, absolute_pressure),
        IF97::cpmass_Tp(temperature, absolute_pressure), IF97::hmass_Tp(temperature, absolute_pressure),
        IF97::smass_Tp(temperature, absolute_pressure), IF97::visc_Tp(temperature, absolute_pressure),
        IF97::tcond_Tp(temperature, absolute_pressure)});
}

Properties liquid(double temperature, double absolute_pressure)
{
    validate_state(temperature, absolute_pressure);
    if (temperature >= IF97::Tcrit)
        throw std::out_of_range("IF97 liquid water requires a temperature below the critical point.");
    if (on_saturation(temperature, absolute_pressure))
        return saturated_phase(saturation_temperature(absolute_pressure), absolute_pressure, false);
    if (absolute_pressure < IF97::psat97(temperature))
        throw std::out_of_range("IF97 liquid water received a vapor state.");
    return evaluate(temperature, absolute_pressure);
}

double saturation_temperature(double absolute_pressure)
{
    require_positive(absolute_pressure, "absolute pressure [Pa]");
    if (absolute_pressure < IF97::Pmin || absolute_pressure > IF97::Pcrit)
        throw std::out_of_range("IF97 saturation pressure must be in 611.213..22064000 Pa.");
    return IF97::Tsat97(absolute_pressure);
}

double saturation_pressure(double temperature)
{
    require_positive(temperature, "temperature [K]");
    if (temperature < IF97::Tmin || temperature > IF97::Tcrit)
        throw std::out_of_range("IF97 saturation temperature must be in 273.15..647.096 K.");
    // Keep the rounded published endpoints invertible by Tsat97().
    return std::clamp(IF97::psat97(temperature), IF97::Pmin, IF97::Pcrit);
}

double surface_tension(double temperature)
{
    (void) saturation_pressure(temperature);
    return IF97::sigma97(temperature);
}

SaturationProperties saturation_at_pressure(double absolute_pressure)
{
    const auto temperature = saturation_temperature(absolute_pressure);
    if (absolute_pressure >= IF97::Pcrit)
        throw std::out_of_range("IF97 separate saturation phases require pressure below the critical point.");
    return {saturated_phase(temperature, absolute_pressure, false),
        saturated_phase(temperature, absolute_pressure, true), surface_tension(temperature)};
}

double liquid_thermal_expansion(double temperature, double absolute_pressure)
{
    const auto state = liquid(temperature, absolute_pressure);
    temperature = state.temperature;
    double lower = IF97::Tmin;
    double upper = absolute_pressure < IF97::Pcrit ? saturation_temperature(absolute_pressure) : IF97::Tcrit;
    // The region 1/3 density fits have a small allowed discontinuity at
    // 623.15 K. Differencing across it would create a spurious expansion.
    if (temperature <= IF97::T23min)
        upper = std::min(upper, IF97::T23min);
    else
        lower = IF97::T23min;
    const double step = std::min(0.001, (upper - lower) / 4.0);
    if (step < 1.0e-6)
        throw std::out_of_range("IF97 liquid interval is too narrow for a density derivative.");
    // Phase validation above and the bounded stencil keep these calls on
    // the same liquid branch without evaluating a vapor density.
    const auto rho = [absolute_pressure](double t) { return IF97::rhomass_Tp(t, absolute_pressure); };
    double derivative;
    if (temperature - step <= lower)
        derivative =
            (-3.0 * state.density + 4.0 * rho(temperature + step) - rho(temperature + 2.0 * step)) / (2.0 * step);
    else if (temperature + step >= upper)
        derivative =
            (3.0 * state.density - 4.0 * rho(temperature - step) + rho(temperature - 2.0 * step)) / (2.0 * step);
    else
        derivative = (rho(temperature + step) - rho(temperature - step)) / (2.0 * step);
    const double expansion = -derivative / state.density;
    if (!std::isfinite(expansion))
        throw std::out_of_range("IF97 liquid density derivative is non-finite.");
    return expansion;
}

std::string backend_version()
{
    return IF97::get_if97_version();
}

} // namespace SimpleFluid::IF97Water
