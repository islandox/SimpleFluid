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
    Neumann   = 1,
    NoSlip    = 2,
    Robin     = 3,
    Periodic  = 4,
    Slip      = 5
};

/**
 * @brief Stores a single boundary condition with type and value.
 */
struct BoundaryCondition
{
    BoundaryConditionType type = BoundaryConditionType::Neumann;
    real_t value = 0.0;
    real_t robin_coefficient = 0.0; // Used only for Robin conditions
};

/**
 * @brief Stores a vector boundary condition for velocity fields.
 */
struct VectorBoundaryCondition
{
    BoundaryConditionType type = BoundaryConditionType::Neumann;
    vec3<real_t> value{};
    real_t robin_coefficient = 0.0; // Used only for Robin conditions
};

using BoundaryConditionMap = std::unordered_map<std::string, BoundaryCondition>;
using VectorBoundaryConditionMap = std::unordered_map<std::string, VectorBoundaryCondition>;

/**
 * @brief Scalar boundary conditions used by two-equation turbulence models.
 *
 * Missing entries use the transport-equation default of homogeneous Neumann.
 * Turbulence scalar equations accept Dirichlet and Neumann conditions only.
 */
struct TurbulenceBoundaryConditionSet
{
    BoundaryConditionMap turbulent_kinetic_energy;
    BoundaryConditionMap dissipation_rate;
    BoundaryConditionMap specific_dissipation_rate;
};

/**
 * @brief Collection of boundary conditions for all fluid unknowns.
 */
struct BoundaryConditionSet
{
    BoundaryConditionMap temperature; ///< Dirichlet K; Neumann K/m.
    VectorBoundaryConditionMap velocity;
    BoundaryConditionMap pressure; ///< Dirichlet Pa; Neumann Pa/m.
    TurbulenceBoundaryConditionSet turbulence;
};

} // namespace SimpleFluid
