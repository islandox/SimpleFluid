/**
 * @file CoupledPressureVelocitySolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiation for the coupled pressure-velocity solver.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "solvers/CoupledPressureVelocitySolver.tcc"

namespace SimpleFluid
{

template class detail::CoupledSchurPreconditioner<DefaultTpetraTypes>;
template auto detail::pressure_gradient_stencils<DefaultTpetraTypes>(const Mesh<DefaultTpetraTypes>&,
    const BoundaryConditionMap&, DefaultTpetraTypes::scalar_type)
    -> std::vector<detail::AffinePressureGradientStencil<DefaultTpetraTypes>>;
template class CoupledPressureVelocitySolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
