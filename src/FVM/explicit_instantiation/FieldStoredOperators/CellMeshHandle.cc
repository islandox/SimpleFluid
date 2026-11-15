/** @brief Compiled mapped cell provider for MeshHandle. */
#include "FVM/FieldStoredCellOperators.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
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

template void cell_gradient<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& field, TensorCellFieldStored<Pack, Mesh>& gradients);

template void cell_gradient<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& field,
    TensorCellFieldStored<Pack, Mesh>& gradients, const CellGradientCache<Pack, Mesh>& cache);

template void gauss_linear_cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients);

template void gauss_linear_cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients);

template void cell_gradient<Pack, Mesh>(const ScalarCellFieldStored<Pack, Mesh>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, Mesh>& gradients,
    const CellGradientCache<Pack, Mesh>& cache, CellGradientScheme scheme);

template std::vector<typename Pack::scalar_type> cell_divergence_from_fluxes<Pack, Mesh>(
    const Mesh& mesh, const ScalarFaceFieldStored<Pack, Mesh>& face_fluxes);

} // namespace SimpleFluid::FVM
