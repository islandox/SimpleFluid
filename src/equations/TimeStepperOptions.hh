/**
 * @file TimeStepperOptions.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Time-stepping and physical parameters used by transient equations.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/vec3.hh"
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
    real_t gravity_x = 0.0;
    real_t gravity_y = 0.0;
    real_t gravity_z = -9.81;
    real_t reference_temperature = 0.5;

    vec3<real_t> gravity_vector() const noexcept
    {
        return {gravity_x, gravity_y, gravity_z};
    }
};

} // namespace SimpleFluid
