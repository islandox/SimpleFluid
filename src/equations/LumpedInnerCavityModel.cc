#include "equations/LumpedInnerCavityModel.hh"

#include <Teuchos_LAPACK.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace SimpleFluid
{
namespace
{
constexpr real_t stefan_boltzmann = 5.670374419e-8;

real_t finite(real_t value)
{
    if (!std::isfinite(value))
        throw std::overflow_error("Lumped inner cavity produced a non-finite value.");
    return value;
}

void positive(real_t value, const char* label)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string("Lumped inner cavity requires positive finite ") + label + '.');
}

void nonnegative(real_t value, const char* label)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string("Lumped inner cavity requires nonnegative finite ") + label + '.');
}

real_t saturation(const LumpedInnerCavityOptions& o, real_t temperature)
{
    const auto value = o.saturation_pressure(temperature);
    positive(value, "saturation pressure");
    return value;
}

real_t vapor_enthalpy(const LumpedInnerCavityOptions& o, real_t temperature)
{
    return finite(o.reference_latent_heat + o.vapor_cp * (temperature - o.reference_temperature));
}

real_t liquid_enthalpy(const LumpedInnerCavityOptions& o, real_t temperature)
{
    return finite(o.liquid_cp * (temperature - o.reference_temperature));
}

real_t energy(const LumpedInnerCavityOptions& o, real_t temperature, real_t mass)
{
    return finite(o.dry_air_mass * (o.dry_air_cp - o.dry_air_gas_constant) * (temperature - o.reference_temperature) +
                  mass * (vapor_enthalpy(o, temperature) - o.vapor_gas_constant * temperature));
}

void validate_state(const LumpedInnerCavityOptions& o, const LumpedInnerCavityState& state)
{
    positive(state.temperature, "air temperature");
    nonnegative(state.humidity_ratio, "humidity ratio");
    if (state.temperature < o.minimum_temperature || state.temperature > o.maximum_temperature)
        throw std::out_of_range("Lumped inner cavity air temperature is outside its configured bracket.");
    const auto mass = finite(o.dry_air_mass * state.humidity_ratio);
    const auto vapor_pressure = finite(mass * o.vapor_gas_constant * state.temperature / o.volume);
    positive((o.dry_air_mass * o.dry_air_gas_constant + mass * o.vapor_gas_constant) * state.temperature / o.volume,
        "absolute pressure");
    finite(vapor_pressure / saturation(o, state.temperature));
    energy(o, state.temperature, mass);
}

// Column-major enclosure solve through Trilinos' LAPACK backend.
std::vector<real_t> solve_radiosity(std::vector<real_t> a, std::vector<real_t> rhs)
{
    if (rhs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("Lumped inner cavity radiosity matrix exceeds the LAPACK ordinal range.");
    const auto n = static_cast<int>(rhs.size());
    const Teuchos::LAPACK<int, real_t> lapack;
    std::vector<int> pivots(rhs.size());
    int info = 0;
    lapack.GETRF(n, n, a.data(), n, pivots.data(), &info);
    if (info < 0)
        throw std::runtime_error("Lumped inner cavity LAPACK GETRF rejected argument " + std::to_string(-info) + '.');
    if (info > 0)
        throw std::domain_error("Lumped inner cavity radiosity matrix is singular.");
    // Retain the existing near-singularity guard in addition to LAPACK's exact
    // zero-pivot check. GETRF stores the U diagonal in the factored matrix.
    for (std::size_t i = 0; i < rhs.size(); ++i)
        if (std::abs(finite(a[i * rhs.size() + i])) <= 64.0 * std::numeric_limits<real_t>::epsilon())
            throw std::domain_error("Lumped inner cavity radiosity matrix is numerically singular.");
    lapack.GETRS('N', n, 1, a.data(), n, pivots.data(), rhs.data(), n, &info);
    if (info != 0)
        throw std::runtime_error("Lumped inner cavity LAPACK GETRS failed with INFO=" + std::to_string(info) + '.');
    for (const auto value : rhs)
        finite(value);
    return rhs;
}

struct PreparedSurface
{
    real_t h, heat_conductance, mass_conductance, equilibrium_pressure, hv, hl;
    real_t radiation = 0.0;
};

std::vector<PreparedSurface> prepare(const LumpedInnerCavityOptions& o, real_t air_temperature,
    std::span<const cavity_detail::SurfaceSample> surfaces, std::span<const real_t> factors)
{
    const auto n = surfaces.size();
    if (!factors.empty() && (n == 0 || factors.size() / n != n || factors.size() % n != 0))
        throw std::invalid_argument("Lumped inner cavity requires an N by N view-factor matrix.");
    std::vector<PreparedSurface> result;
    for (const auto& sample : surfaces)
    {
        const auto& s = *sample.properties;
        positive(sample.area, "surface area");
        positive(sample.temperature, "surface temperature");
        nonnegative(s.mass_transfer_coefficient, "mass transfer coefficient");
        if (s.moisture_exchange != CavityMoistureExchange::Impermeable &&
            s.moisture_exchange != CavityMoistureExchange::LiquidSurface &&
            s.moisture_exchange != CavityMoistureExchange::CondensationOnly)
            throw std::invalid_argument("Unknown inner cavity moisture exchange mode.");
        auto boundary = s.convection;
        boundary.ambient_temperature = air_temperature;
        if (!s.convection_coefficient && boundary.correlation != ConvectionCorrelation::Constant &&
            boundary.correlation != ConvectionCorrelation::ChurchillChuVerticalPlate)
            throw std::invalid_argument("Inner cavity convection requires constant h or natural convection.");
        const auto h = s.convection_coefficient
                           ? s.convection_coefficient(sample.temperature, air_temperature)
                           : evaluate_convection(boundary, sample.temperature).heat_transfer_coefficient;
        nonnegative(h, "heat transfer coefficient");
        const bool wet = s.moisture_exchange != CavityMoistureExchange::Impermeable;
        const auto b = wet ? finite(sample.area * s.mass_transfer_coefficient /
                                    (o.vapor_gas_constant * (0.5 * sample.temperature + 0.5 * air_temperature)))
                           : 0.0;
        const auto hv = vapor_enthalpy(o, sample.temperature);
        const auto hl = liquid_enthalpy(o, sample.temperature);
        if (wet)
            positive(hv - hl, "latent heat at the surface");
        result.push_back({h, finite(sample.area * h), b, wet ? saturation(o, sample.temperature) : 0.0, hv, hl});
    }
    if (factors.empty())
        return result;

    constexpr real_t tolerance = 1.0e-10;
    std::vector<real_t> matrix(factors.size()), rhs(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        positive(surfaces[i].properties->emissivity, "surface emissivity");
        if (surfaces[i].properties->emissivity > 1.0)
            throw std::invalid_argument("Lumped inner cavity emissivity must be at most one.");
        real_t sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
        {
            const auto f = factors[i * n + j];
            nonnegative(f, "view factor");
            if (f > 1.0)
                throw std::invalid_argument("Lumped inner cavity view factors must be at most one.");
            sum += f;
            const auto aij = finite(surfaces[i].area * f);
            const auto aji = finite(surfaces[j].area * factors[j * n + i]);
            if (std::abs(aij - aji) > tolerance * std::max(aij, aji))
                throw std::invalid_argument("Lumped inner cavity view factors violate area reciprocity.");
            // View factors remain row-major at the API; LAPACK stores columns.
            matrix[j * n + i] = (i == j ? 1.0 : 0.0) - (1.0 - surfaces[i].properties->emissivity) * f;
        }
        if (std::abs(sum - 1.0) > tolerance)
            throw std::invalid_argument("Lumped inner cavity view factors must close each enclosure row.");
        rhs[i] = finite(surfaces[i].properties->emissivity * stefan_boltzmann * std::pow(surfaces[i].temperature, 4));
    }
    const auto radiosity = solve_radiosity(std::move(matrix), std::move(rhs));
    for (std::size_t i = 0; i < n; ++i)
        for (auto j = i + 1; j < n; ++j)
        {
            // Symmetric exchange area preserves energy despite roundoff in reciprocity.
            const auto area = 0.5 * surfaces[i].area * factors[i * n + j] + 0.5 * surfaces[j].area * factors[j * n + i];
            const auto heat = finite(area * (radiosity[i] - radiosity[j]));
            result[i].radiation = finite(result[i].radiation + heat);
            result[j].radiation = finite(result[j].radiation - heat);
        }
    return result;
}

real_t mass_rate(const InnerCavitySurface& s, const PreparedSurface& p, real_t vapor_pressure)
{
    const auto rate = finite(p.mass_conductance * (p.equilibrium_pressure - vapor_pressure));
    return s.moisture_exchange == CavityMoistureExchange::CondensationOnly ? std::min(0.0, rate) : rate;
}

LumpedInnerCavityExchange exchange(const LumpedInnerCavityOptions& o, real_t temperature, real_t mass,
    std::span<const cavity_detail::SurfaceSample> surfaces, const std::vector<PreparedSurface>& prepared)
{
    LumpedInnerCavityExchange result;
    const auto pv = finite(mass * o.vapor_gas_constant * temperature / o.volume);
    for (std::size_t i = 0; i < surfaces.size(); ++i)
    {
        const auto& p = prepared[i];
        InnerCavitySurfaceExchange s;
        s.heat_transfer_coefficient = p.h;
        s.vapor_mass_rate = mass_rate(*surfaces[i].properties, p, pv);
        s.sensible_heat_rate = finite(p.heat_conductance * (surfaces[i].temperature - temperature));
        s.latent_heat_rate = finite(s.vapor_mass_rate * (p.hv - p.hl));
        s.radiation_heat_rate = p.radiation;
        s.heat_rate = finite(s.sensible_heat_rate + s.latent_heat_rate + s.radiation_heat_rate);
        s.energy_rate = finite(s.heat_rate + s.vapor_mass_rate * p.hl);
        result.vapor_mass_rate = finite(result.vapor_mass_rate + s.vapor_mass_rate);
        result.air_energy_rate = finite(result.air_energy_rate + s.sensible_heat_rate + s.vapor_mass_rate * p.hv);
        result.surfaces.push_back(s);
    }
    return result;
}

// The implicit mass residual is continuous and strictly increasing, including
// condensation-only switches. Bracketing makes vapor depletion impossible.
real_t implicit_mass(const LumpedInnerCavityOptions& o, real_t temperature, real_t old_mass, real_t dt,
    std::span<const cavity_detail::SurfaceSample> surfaces, const std::vector<PreparedSurface>& prepared)
{
    const auto pressure_per_mass = finite(o.vapor_gas_constant * temperature / o.volume);
    real_t lower = 0.0, upper = old_mass;
    bool active = false;
    for (const auto& p : prepared)
        if (p.mass_conductance > 0.0)
        {
            active = true;
            upper = std::max(upper, finite(p.equilibrium_pressure / pressure_per_mass));
        }
    if (!active)
        return old_mass;
    const auto residual_at = [&](real_t mass)
    {
        real_t rate = 0.0;
        for (std::size_t i = 0; i < surfaces.size(); ++i)
            rate = finite(rate + mass_rate(*surfaces[i].properties, prepared[i], finite(pressure_per_mass * mass)));
        return finite(mass - old_mass - dt * rate);
    };
    // An inactive condensation-only wall must not manufacture traces of vapor
    // by stopping a bisection near the exact dry endpoint.
    if (residual_at(old_mass) == 0.0)
        return old_mass;
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        const auto mass = std::midpoint(lower, upper);
        if (mass == lower || mass == upper)
            break;
        const auto residual = residual_at(mass);
        if (residual == 0.0)
            return mass;
        if (residual > 0.0)
            upper = mass;
        else
            lower = mass;
    }
    return std::midpoint(lower, upper);
}
} // namespace

LumpedInnerCavityModel::LumpedInnerCavityModel(LumpedInnerCavityOptions options, LumpedInnerCavityState initial_state)
    : d_options(std::move(options)), d_state(initial_state)
{
    const auto& o = d_options;
    positive(o.volume, "gas volume");
    positive(o.dry_air_mass, "dry-air mass");
    positive(o.dry_air_gas_constant, "dry-air gas constant");
    positive(o.vapor_gas_constant, "vapor gas constant");
    positive(o.dry_air_cp - o.dry_air_gas_constant, "dry-air cv");
    positive(o.vapor_cp - o.vapor_gas_constant, "vapor cv");
    positive(o.liquid_cp, "liquid cp");
    positive(o.reference_temperature, "reference temperature");
    positive(o.reference_latent_heat, "reference latent heat");
    positive(o.minimum_temperature, "minimum temperature");
    positive(o.maximum_temperature, "maximum temperature");
    if (o.minimum_temperature >= o.maximum_temperature || !o.saturation_pressure)
        throw std::invalid_argument("Lumped inner cavity requires an ordered temperature bracket and saturation law.");
    validate_state(o, d_state);
}

void LumpedInnerCavityModel::set_state(LumpedInnerCavityState state)
{
    validate_state(d_options, state);
    d_state = state;
}

real_t LumpedInnerCavityModel::vapor_mass() const
{
    return d_options.dry_air_mass * d_state.humidity_ratio;
}
real_t LumpedInnerCavityModel::vapor_pressure() const
{
    return vapor_mass() * d_options.vapor_gas_constant * d_state.temperature / d_options.volume;
}
real_t LumpedInnerCavityModel::pressure() const
{
    return (d_options.dry_air_mass * d_options.dry_air_gas_constant + vapor_mass() * d_options.vapor_gas_constant) *
           d_state.temperature / d_options.volume;
}
real_t LumpedInnerCavityModel::relative_humidity() const
{
    return finite(vapor_pressure() / saturation(d_options, d_state.temperature));
}
real_t LumpedInnerCavityModel::internal_energy() const
{
    return energy(d_options, d_state.temperature, vapor_mass());
}

namespace
{
std::vector<cavity_detail::SurfaceSample> surface_samples(std::span<const InnerCavitySurface> surfaces)
{
    std::unordered_set<std::string> names;
    std::vector<cavity_detail::SurfaceSample> samples;
    samples.reserve(surfaces.size());
    for (const auto& surface : surfaces)
    {
        if (surface.name.empty() || !names.insert(surface.name).second)
            throw std::invalid_argument("Lumped inner cavity surface names must be nonempty and unique.");
        samples.push_back({&surface, surface.area, surface.temperature});
    }
    return samples;
}
} // namespace

LumpedInnerCavityExchange LumpedInnerCavityModel::evaluate(
    std::span<const InnerCavitySurface> surfaces, std::span<const real_t> factors) const
{
    return evaluate_samples(surface_samples(surfaces), factors);
}

LumpedInnerCavityStep LumpedInnerCavityModel::preview(
    real_t dt, std::span<const InnerCavitySurface> surfaces, std::span<const real_t> factors) const
{
    return preview_samples(dt, surface_samples(surfaces), factors);
}

LumpedInnerCavityExchange LumpedInnerCavityModel::evaluate_samples(
    std::span<const cavity_detail::SurfaceSample> surfaces, std::span<const real_t> view_factors) const
{
    const auto prepared = prepare(d_options, d_state.temperature, surfaces, view_factors);
    return exchange(d_options, d_state.temperature, vapor_mass(), surfaces, prepared);
}

LumpedInnerCavityStep LumpedInnerCavityModel::preview_samples(real_t time_step,
    std::span<const cavity_detail::SurfaceSample> surfaces, std::span<const real_t> view_factors) const
{
    positive(time_step, "time step");
    const auto prepared = prepare(d_options, d_state.temperature, surfaces, view_factors);
    const auto old_mass = vapor_mass();
    const auto old_energy = internal_energy();
    const auto trial = [&](real_t temperature)
    {
        LumpedInnerCavityStep step;
        const auto mass = implicit_mass(d_options, temperature, old_mass, time_step, surfaces, prepared);
        step.state = {temperature, finite(mass / d_options.dry_air_mass)};
        step.exchange = exchange(d_options, temperature, mass, surfaces, prepared);
        step.mass_balance_residual = finite(mass - old_mass - time_step * step.exchange.vapor_mass_rate);
        step.energy_balance_residual =
            finite(energy(d_options, temperature, mass) - old_energy - time_step * step.exchange.air_energy_rate);
        return step;
    };
    // Preserve exact equilibrium and an empty/insulated cavity without bisection drift.
    auto result = trial(d_state.temperature);
    if (result.energy_balance_residual != 0.0)
    {
        auto lower = d_options.minimum_temperature;
        auto upper = d_options.maximum_temperature;
        auto low = trial(lower), high = trial(upper);
        if (low.energy_balance_residual > 0.0 || high.energy_balance_residual < 0.0)
            throw std::out_of_range("Lumped inner cavity advance leaves the configured temperature bracket.");
        result = std::abs(low.energy_balance_residual) < std::abs(high.energy_balance_residual) ? low : high;
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            const auto temperature = std::midpoint(lower, upper);
            if (temperature == lower || temperature == upper || result.energy_balance_residual == 0.0)
                break;
            auto middle = trial(temperature);
            if (std::abs(middle.energy_balance_residual) < std::abs(result.energy_balance_residual))
                result = middle;
            if (middle.energy_balance_residual > 0.0)
                upper = temperature;
            else
                lower = temperature;
        }
    }
    validate_state(d_options, result.state);
    // Reject loss of closure from overflow/cancellation rather than committing an
    // apparently successful step. Tolerances are relative to stored/transferred quantities.
    const auto mass_scale = std::max({old_mass, d_options.dry_air_mass * result.state.humidity_ratio,
        std::abs(time_step * result.exchange.vapor_mass_rate), real_t{1.0e-12}});
    const auto energy_scale = std::max({std::abs(old_energy), std::abs(time_step * result.exchange.air_energy_rate),
        d_options.dry_air_mass * (d_options.dry_air_cp - d_options.dry_air_gas_constant) * result.state.temperature});
    if (std::abs(result.mass_balance_residual) > 1.0e-10 * mass_scale ||
        std::abs(result.energy_balance_residual) > 1.0e-10 * energy_scale)
        throw std::runtime_error("Lumped inner cavity implicit balance did not converge; reduce the time step.");
    return result;
}

LumpedInnerCavityStep LumpedInnerCavityModel::advance(
    real_t time_step, std::span<const InnerCavitySurface> surfaces, std::span<const real_t> view_factors)
{
    auto result = preview(time_step, surfaces, view_factors);
    d_state = result.state;
    return result;
}

} // namespace SimpleFluid
