/**
 * @file solvers/explicit_instantiation/NonOrthogonalDiffusionSolve.cc
 * @brief Explicit instantiations for solver-owned steady non-orthogonal diffusion solves.
 */

#include "solvers/NonOrthogonalDiffusionSolve.tcc"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Scalar = Pack::scalar_type;
using ScalarField = CellField<Pack>;
using MeshType = Mesh<Pack>;
} // namespace

template bool solve_explicit_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarCellValueProvider<Pack>, ScalarField&, int,
    const LinearSolverOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarCellValueProvider<Pack>, ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarCellValueProvider<Pack>, ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&,
    const NonOrthogonalConvergenceOptions&);

template bool solve_explicit_non_orthogonal_diffusion<Pack>(
    const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>, ScalarField&, int, const LinearSolverOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&, const NonOrthogonalConvergenceOptions&);

template bool solve_non_orthogonal_diffusion<Pack>(const MeshType&, Scalar, ScalarBoundaryConditionProvider<Pack>,
    ScalarField&, NonOrthogonalTreatment, int, const LinearSolverOptions&);

} // namespace SimpleFluid::FVM
