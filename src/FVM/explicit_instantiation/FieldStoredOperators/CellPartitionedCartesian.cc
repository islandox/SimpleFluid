/** @brief Compiled mapped cell provider for PartitionedCartesian. */
#include "FVM/FieldStoredCellOperators.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// Only these public static-mesh specializations override hidden argument
// visibility. GNU attributes on instantiations are accepted by GCC and Clang;
// standard [[gnu::visibility]] attributes are not valid there in Clang.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT
#endif

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients,
    const CellGradientCache<Pack, Mesh>& cache);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, Mesh>& gradients);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, Mesh>& gradients, const CellGradientCache<Pack, Mesh>& cache);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& field, TensorCellFieldStored<Pack, Mesh>& gradients);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& field, TensorCellFieldStored<Pack, Mesh>& gradients,
    const CellGradientCache<Pack, Mesh>& cache);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void gauss_linear_cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, Mesh>& gradients);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void gauss_linear_cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, VectorCellFieldStored<Pack, Mesh>& gradients);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void cell_gradient<Pack, Mesh>(
    const ScalarCellFieldStored<Pack, Mesh>& field, const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, Mesh>& gradients, const CellGradientCache<Pack, Mesh>& cache,
    CellGradientScheme scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes<Pack, Mesh>(
    const Mesh& mesh, const ScalarFaceFieldStored<Pack, Mesh>& face_fluxes);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
