/**
 * @file equations/PressureVelocityCoupling.hh
 * @brief Runtime switches and residuals for segregated pressure-velocity coupling.
 */
#pragma once

#include <stdexcept>
#include <string_view>

namespace SimpleFluid
{

/**
 * @brief Available segregated pressure-velocity coupling drivers.
 */
enum class PressureVelocityCoupling
{
    SIMPLE,
    PISO,
    PIMPLE
};

/**
 * @brief Parse the pressure-velocity coupling switch used by input files.
 */
inline PressureVelocityCoupling
pressure_velocity_coupling_from_string(std::string_view value)
{
    if (value == "SIMPLE" || value == "simple")
    {
        return PressureVelocityCoupling::SIMPLE;
    }
    if (value == "PISO" || value == "piso")
    {
        return PressureVelocityCoupling::PISO;
    }
    if (value == "PIMPLE" || value == "pimple")
    {
        return PressureVelocityCoupling::PIMPLE;
    }

    throw std::invalid_argument(
        "Unknown pressure-velocity coupling; expected SIMPLE, PISO, or PIMPLE.");
}

/**
 * @brief Return the input-file spelling for a pressure-velocity coupling mode.
 */
inline std::string_view
to_string(PressureVelocityCoupling coupling)
{
    switch (coupling)
    {
        case PressureVelocityCoupling::SIMPLE: return "SIMPLE";
        case PressureVelocityCoupling::PISO:   return "PISO";
        case PressureVelocityCoupling::PIMPLE: return "PIMPLE";
    }

    throw std::invalid_argument("Unknown PressureVelocityCoupling value.");
}

/**
 * @brief Last-step residual norms for segregated pressure-velocity coupling.
 */
template<class Scalar>
struct PressureVelocityResiduals
{
    Scalar momentum = {};
    Scalar pressure = {};
    Scalar continuity = {};
};

} // namespace SimpleFluid
