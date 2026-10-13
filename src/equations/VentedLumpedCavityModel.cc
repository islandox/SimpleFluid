#include "equations/VentedLumpedCavityModel.hh"

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
namespace
{
bool positive(real_t x) { return std::isfinite(x) && x > 0.0; }
real_t finite(real_t x)
{
    if (!std::isfinite(x)) throw std::domain_error("Vented cavity arithmetic overflow.");
    return x;
}
}
VentedLumpedCavityModel::VentedLumpedCavityModel(
    VentedLumpedCavityOptions options, VentedLumpedCavityState state) : d_options(std::move(options))
{
    if (!positive(d_options.ambient_pressure) || !positive(d_options.ambient_temperature) ||
        !positive(d_options.dry_air_gas_constant) || !positive(d_options.dry_air_cp) ||
        d_options.dry_air_cp <= d_options.dry_air_gas_constant ||
        !positive(d_options.minimum_temperature) || !positive(d_options.maximum_temperature) ||
        d_options.maximum_temperature <= d_options.minimum_temperature)
        throw std::invalid_argument("Vented cavity requires positive physical options and cp > R.");
    set_state(state);
}
void VentedLumpedCavityModel::set_state(VentedLumpedCavityState state)
{
    if (!positive(state.volume) || !positive(state.temperature) ||
        state.temperature < d_options.minimum_temperature || state.temperature > d_options.maximum_temperature)
        throw std::invalid_argument("Vented cavity state lies outside its physical temperature/volume range.");
    if (!positive(finite(d_options.ambient_pressure * state.volume / d_options.dry_air_gas_constant / state.temperature)))
        throw std::invalid_argument("Vented cavity requires a representable positive dry-air mass.");
    finite(d_options.ambient_pressure * state.volume * d_options.dry_air_cp / d_options.dry_air_gas_constant);
    d_state = state;
}
real_t VentedLumpedCavityModel::dry_air_mass() const
{
    return d_options.ambient_pressure * d_state.volume / d_options.dry_air_gas_constant / d_state.temperature;
}
real_t VentedLumpedCavityModel::enthalpy() const
{
    return d_options.ambient_pressure * d_state.volume * d_options.dry_air_cp / d_options.dry_air_gas_constant;
}
real_t VentedLumpedCavityModel::internal_energy() const { return enthalpy() - pressure() * d_state.volume; }

VentedLumpedCavityStep VentedLumpedCavityModel::preview(
    real_t dt, real_t volume, std::span<const InnerCavitySurface> surfaces) const
{
    if (!positive(dt) || !positive(volume))
        throw std::invalid_argument("Vented cavity requires positive finite time step and new volume.");
    VentedLumpedCavityStep result;
    result.exchange.surfaces.resize(surfaces.size());
    real_t conductance = 0.0, source = 0.0;
    for (std::size_t i = 0; i < surfaces.size(); ++i)
    {
        const auto& s = surfaces[i];
        if (!positive(s.temperature) || !std::isfinite(s.area) || s.area < 0.0 ||
            s.moisture_exchange != CavityMoistureExchange::Impermeable || s.mass_transfer_coefficient != 0.0)
            throw std::invalid_argument("Vented sensible cavity requires physical surfaces with water exchange disabled.");
        auto boundary = s.convection;
        boundary.ambient_temperature = d_state.temperature;
        const auto h = s.convection_coefficient ? s.convection_coefficient(s.temperature, d_state.temperature)
            : evaluate_convection(boundary, s.temperature).heat_transfer_coefficient;
        if (!std::isfinite(h) || h < 0.0)
            throw std::invalid_argument("Vented cavity convection coefficient must be finite and nonnegative.");
        result.exchange.surfaces[i].heat_transfer_coefficient = h;
        conductance = finite(conductance + finite(s.area * h));
        source = finite(source + finite(s.area * h * s.temperature));
    }
    const auto pv_r = finite(pressure() * volume / d_options.dry_air_gas_constant);
    const auto old_mass = dry_air_mass();
    const auto cp = d_options.dry_air_cp;
    const auto delta_h = finite(cp * pressure() * (volume - d_state.volume) / d_options.dry_air_gas_constant);
    const auto residual = [&](real_t temperature)
    {
        const auto dm = pv_r / temperature - old_mass;
        const auto vent = finite(dm * cp * (dm >= 0.0 ? d_options.ambient_temperature : temperature));
        return finite(delta_h - finite(dt * finite(source - conductance * temperature)) - vent);
    };
    real_t lo = d_options.minimum_temperature, hi = d_options.maximum_temperature;
    auto flo = residual(lo), fhi = residual(hi);
    if (flo > 0.0 || fhi < 0.0)
        throw std::domain_error("Vented cavity implicit temperature has no root in the configured bracket.");
    real_t temperature = d_state.temperature;
    if (residual(temperature) != 0.0)
    {
        for (int i = 0; i < 200; ++i)
        {
            temperature = std::midpoint(lo, hi);
            const auto f = residual(temperature);
            if (f == 0.0 || hi - lo <= 8.0 * std::numeric_limits<real_t>::epsilon() * temperature) break;
            if (f < 0.0) lo = temperature;
            else hi = temperature;
        }
    }
    result.state = {temperature, volume};
    if (!positive(pv_r / temperature))
        throw std::domain_error("Vented cavity final dry-air mass must be positive.");
    result.vent_mass_transfer = pv_r / temperature - old_mass;
    result.vent_enthalpy_transfer = finite(result.vent_mass_transfer * cp *
        (result.vent_mass_transfer >= 0.0 ? d_options.ambient_temperature : temperature));
    result.volume_work = finite(pressure() * (volume - d_state.volume));
    for (std::size_t i = 0; i < surfaces.size(); ++i)
    {
        auto& e = result.exchange.surfaces[i];
        e.sensible_heat_rate = finite(surfaces[i].area * e.heat_transfer_coefficient *
            (surfaces[i].temperature - temperature));
        e.energy_rate = e.heat_rate = e.sensible_heat_rate;
        result.exchange.air_energy_rate = finite(result.exchange.air_energy_rate + e.energy_rate);
    }
    result.mass_balance_residual = pv_r / temperature - old_mass - result.vent_mass_transfer;
    result.energy_balance_residual = finite(delta_h - dt * result.exchange.air_energy_rate - result.vent_enthalpy_transfer);
    const auto energy_scale = std::max({real_t(1.0), std::abs(enthalpy()), std::abs(finite(pv_r * cp)), std::abs(delta_h),
        std::abs(dt * result.exchange.air_energy_rate), std::abs(result.vent_enthalpy_transfer)});
    if (std::abs(result.energy_balance_residual) > 256.0 * std::numeric_limits<real_t>::epsilon() * energy_scale)
        throw std::domain_error("Vented cavity implicit solve failed its energy balance tolerance.");
    return result;
}
VentedLumpedCavityStep VentedLumpedCavityModel::advance(
    real_t dt, real_t volume, std::span<const InnerCavitySurface> surfaces)
{
    auto result = preview(dt, volume, surfaces);
    d_state = result.state;
    return result;
}
} // namespace SimpleFluid
