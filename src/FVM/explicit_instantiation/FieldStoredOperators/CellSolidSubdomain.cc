/** @brief Compiled mapped cell provider for SolidSubdomain. */
#include "FVM/FieldStoredCellOperators.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = SolidSubdomain<Pack>;
} // namespace

template void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients);

template void cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    VectorCellFieldStored<Pack, Mesh>& gradients, const CellGradientCache<Pack, Mesh>& cache);

template void cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients);

template void cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients,
    const CellGradientCache<Pack, Mesh>& cache);

template void gauss_linear_cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients);

template void gauss_linear_cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients);

template void cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients,
    const CellGradientCache<Pack, Mesh>& cache, CellGradientScheme scheme);

} // namespace SimpleFluid::FVM
