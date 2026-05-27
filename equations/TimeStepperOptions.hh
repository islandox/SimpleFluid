/**
 * @file TimeStepperOptions.hh
 * @brief Time-stepping and physical parameters used by transient equations.
 */
#pragma once

#include "dataclass/typedefs.hh"

namespace SimpleFluid
{

/**
 * @brief Time stepping and physical parameters for Boussinesq-style updates.
 */
struct TimeStepperOptions
{
    real_t time_step = 1.0e-3;
    int steps = 1;
    real_t thermal_diffusivity = 1.0e-3;
    real_t kinematic_viscosity = 1.0e-3;
    real_t thermal_expansion = 3.0e-3;
    real_t gravity_z = -9.81;
    real_t reference_temperature = 0.5;
};

} // namespace SimpleFluid
