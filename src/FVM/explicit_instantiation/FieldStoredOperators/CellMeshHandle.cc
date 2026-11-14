/** @brief Compiled mapped cell provider for MeshHandle. */
#include "FVM/FieldStoredCellOperators.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
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

template void cell_gradient<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& field, TensorCellFieldStored<Pack, MeshType>& gradients);

template void cell_gradient<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const CellGradientCache<Pack, MeshType>& cache);

template void gauss_linear_cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients);

template void gauss_linear_cell_gradient<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients);

template void cell_gradient<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache, CellGradientScheme scheme);

template std::vector<typename Pack::scalar_type> cell_divergence_from_fluxes<Pack, MeshType>(
    const MeshType& mesh, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes);

} // namespace SimpleFluid::FVM
