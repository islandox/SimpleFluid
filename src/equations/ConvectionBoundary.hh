/** @file ConvectionBoundary.hh
 *  @brief Exterior convection closures; all inputs use SI units.
 */
#pragma once

#include "dataclass/typedefs.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

enum class ConvectionCorrelation
{
    Constant,
    ChurchillChuVerticalPlate,
    ChurchillBernsteinCylinder
};

/**
 * @brief Exterior heat loss q_out = h (T_surface - T_ambient).
 *
 * Ambient properties are caller-supplied film-temperature values, distinct
 * from the material inside the modeled domain. Correlations return a mean
 * coefficient, used facewise with the specified full height or diameter.
 * No material-property law, geometry inference, or mixed-convection blend
 * is implied. See docs/modeling/exterior_convection.md for applicability.
 */
struct ConvectionBoundary
{
    ConvectionCorrelation correlation = ConvectionCorrelation::Constant;
    real_t ambient_temperature = 300.0;      ///< K
    real_t heat_transfer_coefficient = 0.0;  ///< Constant h, W/(m^2 K)
    real_t characteristic_length = 1.0;      ///< Plate height or cylinder diameter, m
    real_t fluid_thermal_conductivity = 0.0; ///< Exterior fluid k, W/(m K)
    real_t kinematic_viscosity = 0.0;        ///< Exterior fluid nu, m^2/s
    real_t thermal_diffusivity = 0.0;        ///< Exterior fluid alpha, m^2/s
    real_t thermal_expansion = 0.0;          ///< Exterior fluid beta, 1/K
    real_t gravity = 9.80665;                ///< Magnitude, m/s^2
    real_t free_stream_speed = 0.0;          ///< Crossflow speed, m/s

    bool operator==(const ConvectionBoundary&) const = default;
};

using ConvectionBoundaryMap = std::unordered_map<std::string, ConvectionBoundary>;

namespace convection_detail
{
/** Stable representation for collective configuration and coupling checkpoints. */
inline std::string configuration_signature(const ConvectionBoundaryMap& boundaries)
{
    std::vector<std::string> names;
    for (const auto& [name, boundary] : boundaries)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    std::ostringstream stream;
    stream << std::setprecision(17);
    for (const auto& name : names)
    {
        const auto& b = boundaries.at(name);
        stream << std::quoted(name) << ' ' << static_cast<int>(b.correlation) << ' ' << b.ambient_temperature << ' '
               << b.heat_transfer_coefficient << ' ' << b.characteristic_length << ' ' << b.fluid_thermal_conductivity
               << ' ' << b.kinematic_viscosity << ' ' << b.thermal_diffusivity << ' ' << b.thermal_expansion << ' '
               << b.gravity << ' ' << b.free_stream_speed << '\n';
    }
    return stream.str();
}
} // namespace convection_detail

struct ConvectionEvaluation
{
    real_t heat_transfer_coefficient = 0.0;
    real_t nusselt = 0.0;
    real_t prandtl = 0.0;
    real_t rayleigh = 0.0;
    real_t reynolds = 0.0;
};

/** @brief Evaluate the documented mean correlation, rejecting invalid inputs/ranges. */
inline ConvectionEvaluation evaluate_convection(const ConvectionBoundary& boundary, real_t surface_temperature)
{
    const auto positive = [](real_t value) { return std::isfinite(value) && value > 0.0; };
    if (!positive(surface_temperature) || !positive(boundary.ambient_temperature))
        throw std::invalid_argument("Exterior convection requires finite positive temperatures in K.");
    ConvectionEvaluation result;
    if (boundary.correlation == ConvectionCorrelation::Constant)
    {
        if (!std::isfinite(boundary.heat_transfer_coefficient) || boundary.heat_transfer_coefficient < 0.0)
            throw std::invalid_argument("Exterior convection requires finite non-negative h.");
        result.heat_transfer_coefficient = boundary.heat_transfer_coefficient;
        return result;
    }
    if (!positive(boundary.characteristic_length) || !positive(boundary.fluid_thermal_conductivity) ||
        !positive(boundary.kinematic_viscosity) || !positive(boundary.thermal_diffusivity))
        throw std::invalid_argument("Exterior convection requires positive length and exterior fluid properties.");
    const auto length = boundary.characteristic_length;
    result.prandtl = boundary.kinematic_viscosity / boundary.thermal_diffusivity;
    if (!positive(result.prandtl))
        throw std::invalid_argument("Exterior convection requires a finite positive Prandtl number.");
    switch (boundary.correlation)
    {
        case ConvectionCorrelation::ChurchillChuVerticalPlate: {
            if (!positive(boundary.gravity) || !positive(boundary.thermal_expansion))
                throw std::invalid_argument("Natural convection requires positive gravity and thermal expansion.");
            result.rayleigh = boundary.gravity * boundary.thermal_expansion *
                              std::abs(surface_temperature - boundary.ambient_temperature) * length * length * length /
                              (boundary.kinematic_viscosity * boundary.thermal_diffusivity);
            // Conservative supported range; Ra=0 retains the continuous zero-flux limit.
            if (!std::isfinite(result.rayleigh) || result.rayleigh < 0.0 || result.rayleigh > 1.0e12)
                throw std::domain_error("Churchill-Chu vertical plate requires 0 <= Ra <= 1e12.");
            const auto root_nu = 0.825 + 0.387 * std::pow(result.rayleigh, 1.0 / 6.0) /
                                             std::pow(1.0 + std::pow(0.492 / result.prandtl, 9.0 / 16.0), 8.0 / 27.0);
            result.nusselt = root_nu * root_nu;
            break;
        }
        case ConvectionCorrelation::ChurchillBernsteinCylinder:
            if (!std::isfinite(boundary.free_stream_speed) || boundary.free_stream_speed < 0.0)
                throw std::invalid_argument("Forced convection requires a finite non-negative crossflow speed.");
            result.reynolds = boundary.free_stream_speed * length / boundary.kinematic_viscosity;
            if (!std::isfinite(result.reynolds) || !std::isfinite(result.reynolds * result.prandtl) ||
                result.reynolds * result.prandtl <= 0.2)
                throw std::domain_error("Churchill-Bernstein cylinder requires Re Pr > 0.2.");
            result.nusselt = 0.3 + 0.62 * std::sqrt(result.reynolds) * std::cbrt(result.prandtl) /
                                       std::pow(1.0 + std::pow(0.4 / result.prandtl, 2.0 / 3.0), 0.25) *
                                       std::pow(1.0 + std::pow(result.reynolds / 282000.0, 5.0 / 8.0), 4.0 / 5.0);
            break;
        default:
            throw std::invalid_argument("Unknown exterior convection correlation.");
    }
    result.heat_transfer_coefficient = result.nusselt * boundary.fluid_thermal_conductivity / length;
    if (!positive(result.heat_transfer_coefficient))
        throw std::overflow_error("Exterior convection produced a non-finite or non-positive coefficient.");
    return result;
}

/**
 * @brief Reconstruct the old wall temperature through the cell-to-face resistance.
 * The returned coefficient is frozen during the next implicit temperature solve.
 * conductance is k/d [W/(m^2 K)], not the exterior fluid conductivity.
 */
inline ConvectionEvaluation evaluate_convection_at_cell(
    const ConvectionBoundary& boundary, real_t cell_temperature, real_t conductance)
{
    if (!std::isfinite(conductance) || conductance < 0.0)
        throw std::invalid_argument("Exterior convection requires finite non-negative wall conductance.");
    // Validate the full cell-to-ambient interval before solving for its wall value.
    auto result = evaluate_convection(boundary, cell_temperature);
    if (boundary.correlation != ConvectionCorrelation::ChurchillChuVerticalPlate)
        return result;
    if (conductance == 0.0)
        return evaluate_convection(boundary, boundary.ambient_temperature);
    real_t lower = 0.0, upper = 1.0;
    for (int iteration = 0; iteration < 48; ++iteration)
    {
        const auto fraction = 0.5 * (lower + upper);
        result = evaluate_convection(
            boundary, boundary.ambient_temperature + fraction * (cell_temperature - boundary.ambient_temperature));
        // Cancel the signed temperature difference, valid for heating and cooling.
        if (conductance * (1.0 - fraction) > result.heat_transfer_coefficient * fraction)
            lower = fraction;
        else
            upper = fraction;
    }
    return result;
}

} // namespace SimpleFluid
