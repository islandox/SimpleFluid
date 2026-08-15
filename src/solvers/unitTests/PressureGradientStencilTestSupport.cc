#include "solvers/CoupledPressureVelocitySolver.tcc"

namespace SimpleFluid
{

template auto detail::pressure_gradient_stencils<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    const BoundaryConditionMap&,
    DefaultTpetraTypes::scalar_type)
    -> std::vector<
        detail::AffinePressureGradientStencil<DefaultTpetraTypes>>;

} // namespace SimpleFluid
