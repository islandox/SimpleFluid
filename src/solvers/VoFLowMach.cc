/** @file VoFLowMach.cc
 * @brief Validated caloric EOS wrappers for low-Mach VoF.
 */
#include "solvers/VoFLowMach.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::VoF
{
namespace
{
void validate_bounds(const ThermodynamicBounds& b)
{
    if (!std::isfinite(b.minimum_temperature) || !std::isfinite(b.maximum_temperature) ||
        !std::isfinite(b.minimum_pressure) || !std::isfinite(b.maximum_pressure) || b.minimum_temperature <= 0 ||
        b.maximum_temperature <= b.minimum_temperature || b.minimum_pressure <= 0 ||
        b.maximum_pressure <= b.minimum_pressure)
        throw std::invalid_argument("Low-Mach liquid EOS requires explicit positive finite T/P validity bounds.");
}
} // namespace

LiquidThermodynamics::LiquidThermodynamics(ThermodynamicBounds bounds, Evaluator evaluator)
    : bounds_(bounds), evaluator_(std::move(evaluator))
{
    validate_bounds(bounds_);
    if (!evaluator_)
        throw std::invalid_argument("Low-Mach liquid EOS callback is null.");
}

const ThermodynamicBounds& LiquidThermodynamics::bounds() const noexcept
{
    return bounds_;
}

LiquidThermodynamicPoint LiquidThermodynamics::evaluate(double temperature, double pressure) const
{
    if (!std::isfinite(temperature) || !std::isfinite(pressure) || temperature < bounds_.minimum_temperature ||
        temperature > bounds_.maximum_temperature || pressure < bounds_.minimum_pressure ||
        pressure > bounds_.maximum_pressure)
        throw std::out_of_range("Low-Mach liquid EOS state is outside its declared T/P range.");
    const auto p = evaluator_(temperature, pressure);
    const std::array<double, 6> values{p.density, p.internal_energy, p.enthalpy, p.internal_energy_temperature,
        p.dynamic_viscosity, p.thermal_conductivity};
    if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }) || p.density <= 0 ||
        p.internal_energy_temperature <= 0 || p.dynamic_viscosity < 0 || p.thermal_conductivity < 0)
        throw std::runtime_error(
            "Low-Mach liquid EOS returned invalid density, caloric derivative, or transport properties.");
    const double pv = pressure / p.density;
    if (std::abs(p.enthalpy - p.internal_energy - pv) >
        2e-11 * std::max({1.0, std::abs(p.enthalpy), std::abs(p.internal_energy), std::abs(pv)}))
        throw std::runtime_error("Low-Mach liquid EOS violates h=e+p/rho.");
    return p;
}

std::shared_ptr<const LiquidThermodynamics> make_linear_compressible_liquid(LinearCompressibleLiquid p)
{
    validate_bounds(p.bounds);
    const std::array<double, 9> values{p.reference_temperature, p.reference_pressure, p.reference_density,
        p.isobaric_heat_capacity, p.thermal_expansion, p.compressibility, p.reference_internal_energy,
        p.dynamic_viscosity, p.thermal_conductivity};
    if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }) ||
        p.reference_temperature <= 0 || p.reference_pressure <= 0 || p.reference_density <= 0 ||
        p.isobaric_heat_capacity <= 0 || p.compressibility <= 0 || p.dynamic_viscosity < 0 ||
        p.thermal_conductivity < 0)
        throw std::invalid_argument("Linear-compressible liquid parameters are invalid.");
    const double v0 = 1.0 / p.reference_density;
    if (p.isobaric_heat_capacity -
            p.bounds.maximum_temperature * v0 * p.thermal_expansion * p.thermal_expansion / p.compressibility <=
        0)
        throw std::invalid_argument(
            "Linear-compressible liquid Gibbs potential has nonpositive isochoric heat capacity.");
    auto evaluate = [p, v0](double temperature, double pressure)
    {
        const double dT = temperature - p.reference_temperature, dP = pressure - p.reference_pressure;
        const double v = v0 * (1 + p.thermal_expansion * dT - p.compressibility * dP);
        const double h =
            p.reference_internal_energy + p.reference_pressure * v0 + p.isobaric_heat_capacity * dT +
            v0 * ((1 - p.thermal_expansion * p.reference_temperature) * dP - 0.5 * p.compressibility * dP * dP);
        // These derive from one Gibbs potential: v=g_p, h=g-T*g_T.
        return LiquidThermodynamicPoint{1.0 / v, h - pressure * v, h,
            p.isobaric_heat_capacity - pressure * v0 * p.thermal_expansion, p.dynamic_viscosity,
            p.thermal_conductivity};
    };
    auto result = std::make_shared<const LiquidThermodynamics>(p.bounds, std::move(evaluate));
    for (double t : {p.bounds.minimum_temperature, p.bounds.maximum_temperature})
        for (double pressure : {p.bounds.minimum_pressure, p.bounds.maximum_pressure})
            (void) result->evaluate(t, pressure);
    (void) result->evaluate(p.reference_temperature, p.reference_pressure);
    return result;
}

} // namespace SimpleFluid::VoF
