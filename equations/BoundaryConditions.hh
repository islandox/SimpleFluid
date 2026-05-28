/**
 * @file BoundaryConditions.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Boundary-condition types shared by physical equation classes.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/vec3.hh"
#include "dataclass/typedefs.hh"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace SimpleFluid
{

/**
 * @brief Enumeration of supported boundary condition types.
 */
enum class BoundaryConditionType : std::uint8_t
{
    Dirichlet = 0,
    Neumann = 1,
    NoSlip = 2
};

/**
 * @brief Stores a single boundary condition with type and value.
 */
struct BoundaryCondition
{
    BoundaryConditionType type = BoundaryConditionType::Neumann;
    real_t value = 0.0;
};

/**
 * @brief Stores a vector boundary condition for velocity fields.
 */
struct VectorBoundaryCondition
{
    BoundaryConditionType type = BoundaryConditionType::Neumann;
    vec3<real_t> value{};
};

/**
 * @brief Collection of boundary conditions for temperature, velocity, and pressure.
 */
struct BoundaryConditionSet
{
    std::unordered_map<std::string, BoundaryCondition> temperature;
    std::unordered_map<std::string, VectorBoundaryCondition> velocity;
    std::unordered_map<std::string, BoundaryCondition> pressure;
};

} // namespace SimpleFluid
