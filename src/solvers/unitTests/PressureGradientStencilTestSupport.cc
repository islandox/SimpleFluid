#include "solvers/CoupledPressureVelocitySolver.tcc"

namespace SimpleFluid
{

template auto detail::pressure_gradient_stencils<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>(
    const Mesh<DefaultTpetraTypes>&, const BoundaryConditionMap&, DefaultTpetraTypes::scalar_type)
    -> std::vector<detail::AffinePressureGradientStencil<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>>;

template auto detail::pressure_gradient_stencils<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>(
    const MeshHandle<DefaultTpetraTypes>&, const BoundaryConditionMap&, DefaultTpetraTypes::scalar_type)
    -> std::vector<detail::AffinePressureGradientStencil<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>>;

} // namespace SimpleFluid
