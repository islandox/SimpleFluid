/**
 * @file FVM/details/FieldStoredCellOperators.hh
 * @brief Internal cell-operator kernels for mesh-aware stored fields.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/details/CachedGradientRows.hh"
#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/ResolvedDiffusionGeometry.hh"
#include "FVM/details/StoredMeshIdentity.hh"
#include "FVM/details/StoredScalarGradientKernel.hh"
#include "fields/FieldStored.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** @brief Evaluate interior least-squares scalar-gradient stencils. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL void stored_scalar_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field, VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Evaluate boundary-aware affine scalar-gradient stencils. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Evaluate cached interior scalar-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache>
SIMPLEFLUID_LOCAL void stored_cached_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache);

/** @brief Evaluate cached boundary-aware scalar-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache, class BoundaryConditionProvider, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_cached_scalar_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache);

/** @brief Evaluate interior vector-gradient stencils into tensor storage. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL void stored_vector_cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field, TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Evaluate boundary-aware vector-gradient stencils. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Evaluate cached interior vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache>
SIMPLEFLUID_LOCAL void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache);

/** @brief Evaluate cached boundary-aware vector-gradient geometry. */
template<TpetraTypePack Pack, class MeshType, class Cache, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_cached_vector_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients, const Cache& cache);

/** @brief Gauss-linear scalar gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_gauss_linear_cell_gradient_reference(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Gauss-linear vector gradient for stored fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_gauss_linear_cell_gradient_reference(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Gauss-linear scalar reconstruction through borrowed region geometry. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_region_gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Gauss-linear vector reconstruction through borrowed region geometry. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_region_gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Select region geometry while retaining the checked generic fallback. */
template<TpetraTypePack Pack, class MeshType, class BoundaryConditionProvider, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Select region geometry for vector gradients when available. */
template<TpetraTypePack Pack, class MeshType, class BoundaryValueProvider>
SIMPLEFLUID_LOCAL void stored_gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Net owner-oriented flux using a scoped overlap or owned face view. */
template<TpetraTypePack Pack, class MeshType, class View>
SIMPLEFLUID_LOCAL auto stored_cell_flux_balance(const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, const View& face_values,
    typename Pack::local_ordinal_type cell_lid) -> typename Pack::scalar_type;

/** @brief Net owner-oriented stored face flux around one cell. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL auto stored_cell_flux_balance(const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::local_ordinal_type cell_lid) ->
    typename Pack::scalar_type
{
    return stored_cell_flux_balance(mesh, face_fluxes, face_fluxes.local_read_view(), cell_lid);
}

} // namespace SimpleFluid::FVM::detail

#include "FVM/details/FieldStoredCellOperators.ipp"
