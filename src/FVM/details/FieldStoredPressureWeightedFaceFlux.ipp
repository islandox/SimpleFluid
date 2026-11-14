/** @file FieldStoredPressureWeightedFaceFlux.ipp
 * @brief Caller-dependent stored-field queries and kernels.
 */
#pragma once

namespace SimpleFluid::FVM::detail
{

template<class VelocityBoundaryCacheType>
void validate_pressure_velocity_boundary_compatibility(
    const VelocityBoundaryCacheType& velocity_boundary_cache, const BoundaryConditionMap& pressure_boundary_conditions)
{
    for (const auto& [name, pressure_condition] : pressure_boundary_conditions)
    {
        if (pressure_condition.type != BoundaryConditionType::Dirichlet &&
            pressure_condition.type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument("pressure_weighted_face_fluxes supports only Dirichlet "
                                        "and Neumann pressure boundary conditions.");
        }
        if (pressure_condition.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        const auto velocity_iter = velocity_boundary_cache.type_by_name.find(name);
        const auto velocity_type = velocity_iter == velocity_boundary_cache.type_by_name.end()
                                       ? BoundaryConditionType::Neumann
                                       : velocity_iter->second;
        if (velocity_type != BoundaryConditionType::Neumann)
        {
            throw std::invalid_argument("Dirichlet pressure boundary '" + name +
                                        "' requires a Neumann velocity boundary so owner-cell "
                                        "velocity can be extrapolated to the open face.");
        }
    }
}

} // namespace SimpleFluid::FVM::detail
