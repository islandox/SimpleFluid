/**
 * @file FvmInstantiations.cc
 * @brief Explicit template instantiations for default finite-volume helpers.
 */

#include "FVM/BoundaryCache.hh"
#include "FVM/CellOperators.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/FaceFlux.hh"
#include "FVM/MatrixOperators.hh"
#include "FVM/TransportSystem.tcc"

namespace SimpleFluid
{

template struct BoundaryCache<DefaultTpetraTypes>;
template BoundaryCache<DefaultTpetraTypes>
cache_boundary_conditions<DefaultTpetraTypes>(
    SP<const Mesh<DefaultTpetraTypes>>,
    const BoundaryConditionMap&);

namespace FVM
{

template struct DiffusionSystem<DefaultTpetraTypes>;
template struct VectorDiffusionSystem<DefaultTpetraTypes>;
template struct TransportSystem<DefaultTpetraTypes>;
template struct VectorTransportSystem<DefaultTpetraTypes>;

template void cell_gradient<DefaultTpetraTypes>(
    const CellField<DefaultTpetraTypes>&,
    VectorCellField<DefaultTpetraTypes>&);
template void cell_gradient<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    TensorCellField<DefaultTpetraTypes>&);
template DefaultTpetraTypes::scalar_type
cell_flux_balance<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    const FaceField<DefaultTpetraTypes>&,
    DefaultTpetraTypes::local_ordinal_type);
template std::vector<DefaultTpetraTypes::scalar_type>
cell_divergence_from_fluxes<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    const FaceField<DefaultTpetraTypes>&);

template Teuchos::RCP<DefaultTpetraTypes::matrix_type>
identity_matrix<DefaultTpetraTypes>(
    const Teuchos::RCP<const DefaultTpetraTypes::map_type>&,
    DefaultTpetraTypes::scalar_type);
template Teuchos::RCP<DefaultTpetraTypes::matrix_type>
diffusion_matrix<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    DefaultTpetraTypes::scalar_type);
template Teuchos::RCP<DefaultTpetraTypes::matrix_type>
upwind_convection_matrix<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    const FaceField<DefaultTpetraTypes>&);
template Teuchos::RCP<DefaultTpetraTypes::matrix_type>
pressure_poisson_matrix<DefaultTpetraTypes>(
    const Mesh<DefaultTpetraTypes>&,
    DefaultTpetraTypes::global_ordinal_type);

template struct VelocityBoundaryCache<DefaultTpetraTypes>;
template VelocityBoundaryCache<DefaultTpetraTypes>
cache_velocity_boundary_conditions<DefaultTpetraTypes>(
    SP<const Mesh<DefaultTpetraTypes>>,
    const BoundaryConditionSet&);
template VectorCellField<DefaultTpetraTypes>::vec_type
slip_face_velocity<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    DefaultTpetraTypes::local_ordinal_type);

namespace detail
{

template struct PreparedTransportMatrix<DefaultTpetraTypes>;
template void validate_face_flux_inputs<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const VelocityBoundaryCache<DefaultTpetraTypes>*);
template void validate_face_velocity_output<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const VectorFaceField<DefaultTpetraTypes>&);
template void validate_normal_flux_inputs<DefaultTpetraTypes>(
    const VectorFaceField<DefaultTpetraTypes>&,
    const FaceField<DefaultTpetraTypes>&);
template void load_boundary_face_velocity<DefaultTpetraTypes>(
    const VelocityBoundaryCache<DefaultTpetraTypes>*,
    const VectorCellField<DefaultTpetraTypes>&,
    VectorFaceField<DefaultTpetraTypes>&);
template void assemble_face_velocities<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const VelocityBoundaryCache<DefaultTpetraTypes>*,
    VectorFaceField<DefaultTpetraTypes>&);

template PreparedTransportMatrix<DefaultTpetraTypes>
prepare_transport_matrix<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>(
    const Mesh<DefaultTpetraTypes>&,
    Teuchos::RCP<DefaultTpetraTypes::matrix_type>,
    size_t);
template void add_transport_values<DefaultTpetraTypes>(
    const PreparedTransportMatrix<DefaultTpetraTypes>&,
    DefaultTpetraTypes::local_ordinal_type,
    const Teuchos::ArrayView<const DefaultTpetraTypes::local_ordinal_type>&,
    const Teuchos::ArrayView<const DefaultTpetraTypes::scalar_type>&);

} // namespace detail

template void face_velocities<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    VectorFaceField<DefaultTpetraTypes>&);
template void face_velocities<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const VelocityBoundaryCache<DefaultTpetraTypes>&,
    VectorFaceField<DefaultTpetraTypes>&);
template void normal_face_fluxes<DefaultTpetraTypes>(
    const VectorFaceField<DefaultTpetraTypes>&,
    FaceField<DefaultTpetraTypes>&);
template void face_fluxes<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    FaceField<DefaultTpetraTypes>&);
template void face_fluxes<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const VelocityBoundaryCache<DefaultTpetraTypes>&,
    FaceField<DefaultTpetraTypes>&);
template void pressure_weighted_face_fluxes<DefaultTpetraTypes>(
    const VectorCellField<DefaultTpetraTypes>&,
    const CellField<DefaultTpetraTypes>&,
    DefaultTpetraTypes::scalar_type,
    const VelocityBoundaryCache<DefaultTpetraTypes>&,
    FaceField<DefaultTpetraTypes>&);

} // namespace FVM
} // namespace SimpleFluid
