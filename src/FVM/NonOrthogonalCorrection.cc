/**
 * @file FVM/NonOrthogonalCorrection.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiations for non-orthogonal FVM corrections.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FVM/NonOrthogonalCorrection.tcc"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Scalar = Pack::scalar_type;
using MatrixVector = Pack::vector_type;
using MatrixMultiVector = Pack::multi_vector_type;
using ScalarField = CellField<Pack>;
using VectorField = VectorCellField<Pack>;
using MeshType = Mesh<Pack>;
using ScalarAffineStencil = detail::AffineLeastSquaresGradientStencil<MeshType>;
using VectorAffineStencil = detail::VectorAffineLeastSquaresGradientStencil<MeshType>;
using InteriorStencil = detail::LeastSquaresGradientStencil<MeshType>;
using BoundaryLocation = detail::BoundaryFaceLocation<MeshType>;
} // namespace

template void add_explicit_non_orthogonal_correction<Pack>(
    const ScalarField&, Scalar, ScalarBoundaryConditionProvider<Pack>, MatrixVector&, Scalar);

template void add_explicit_non_orthogonal_correction<Pack>(
    const VectorField&, Scalar, MatrixMultiVector&, Scalar, BoundaryFaceSelector, const std::vector<InteriorStencil>*);

template void add_variable_explicit_non_orthogonal_correction<Pack>(const ScalarField&, const ScalarField&,
    ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, MatrixVector&, Scalar,
    BoundaryCoefficientProvider<Pack>, const std::vector<ScalarAffineStencil>*, FaceCoefficientInterpolation);

template void add_variable_explicit_non_orthogonal_correction<Pack>(const ScalarField&, const ScalarField&,
    ScalarBoundaryConditionProvider<Pack>, MatrixVector&, Scalar, BoundaryCoefficientProvider<Pack>);

template void add_variable_explicit_non_orthogonal_correction<Pack>(const VectorField&, const ScalarField&,
    VectorBoundaryValueProvider<Pack>, MatrixMultiVector&, Scalar, BoundaryFaceSelector,
    BoundaryCoefficientProvider<Pack>, const std::vector<VectorAffineStencil>*, const std::vector<BoundaryLocation>*,
    FaceCoefficientInterpolation);

template void add_variable_explicit_non_orthogonal_correction<Pack>(const VectorField&, const ScalarField&,
    MatrixMultiVector&, Scalar, BoundaryFaceSelector, BoundaryCoefficientProvider<Pack>);

template void add_explicit_deviatoric_transpose_gradient_stress<Pack>(const VectorField&, const ScalarField&, Scalar,
    VectorBoundaryValueProvider<Pack>, MatrixMultiVector&, BoundaryFaceSelector, BoundaryCoefficientProvider<Pack>,
    const std::vector<VectorAffineStencil>*, const std::vector<BoundaryLocation>*, FaceCoefficientInterpolation);

template DiffusionSystem<Pack> explicit_non_orthogonal_diffusion_system<Pack>(
    const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, const ScalarField&);

template DiffusionSystem<Pack> implicit_non_orthogonal_diffusion_system<Pack>(const MeshType&, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, Scalar, const ScalarField*);

template DiffusionSystem<Pack> fully_implicit_non_orthogonal_diffusion_system<Pack>(
    const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, const ScalarField*);

template DiffusionSystem<Pack> non_orthogonal_diffusion_system<Pack>(const MeshType&, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*);

template Teuchos::RCP<MatrixVector> full_diffusion_residual<Pack>(
    const ScalarField&, Scalar, ScalarBoundaryConditionProvider<Pack>);

template bool solve_explicit_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, ScalarField&, int,
    const LinearSolverOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarCellValueProvider<Pack>, ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&);

template bool solve_explicit_non_orthogonal_diffusion<Pack>(
    const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>, ScalarField&, int, const LinearSolverOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&);

} // namespace SimpleFluid::FVM
