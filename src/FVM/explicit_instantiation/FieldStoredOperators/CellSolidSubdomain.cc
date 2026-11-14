/** @brief Compiled mapped cell provider for SolidSubdomain. */
#include "FVM/FieldStoredCellOperators.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = SolidSubdomain<Pack>;
} // namespace

template void cell_gradient<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients);

template void cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const CellGradientCache<Pack, MeshType>& cache);

template void cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients);

template void cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

template void gauss_linear_cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients);

template void gauss_linear_cell_gradient<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients);

template void cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache, CellGradientScheme scheme);

} // namespace SimpleFluid::FVM
