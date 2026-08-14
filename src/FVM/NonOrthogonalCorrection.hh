/**
 * @file FVM/NonOrthogonalCorrection.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Non-orthogonal diffusion correction: explicit, implicit, hybrid treatments,
 *        residual evaluation, and solver wrappers.
 * @version 0.1
 * @date 2026-06-04
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/OperatorDetails.hh"
#include "equations/BoundaryConditions.hh"
#include "solvers/BelosLinearSolver.hh"

#include <vector>

namespace SimpleFluid::FVM
{

namespace detail
{

/** @brief Evaluate cached scalar affine reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_scalar_affine_gradients(const CellField<Pack>& field,
    const std::vector<AffineLeastSquaresGradientStencil<Mesh<Pack>>>& stencils, VectorCellField<Pack>& gradients);

/** @brief Evaluate cached vector affine reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_vector_affine_gradients(const VectorCellField<Pack>& field,
    const std::vector<VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>& stencils, TensorCellField<Pack>& gradients);

/** @brief Evaluate cached interior-only vector reconstruction coefficients. */
template<TpetraTypePack Pack>
void evaluate_vector_interior_gradients(const VectorCellField<Pack>& field,
    const std::vector<LeastSquaresGradientStencil<Mesh<Pack>>>& stencils, TensorCellField<Pack>& gradients);

} // namespace detail

/**
 * @brief Add the explicit non-orthogonal diffusion correction to an RHS.
 *
 * For each face, the area vector is split into @f$S_f = E_f + T_f@f$.
 * The implicit system owns the @f$E_f@f$ contribution; this routine adds
 * @f$\Gamma_f \nabla\phi_f\cdot T_f@f$ to the RHS for the explicit
 * correction of @f$-\nabla\cdot(\Gamma\nabla\phi)@f$.
 *
 * @tparam Pack Tpetra type pack.
 * @param correction_field Scalar field used to compute the gradient for
 *        the non-orthogonal correction.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param[in,out] rhs Owned-cell RHS vector; receives the correction.
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @throws std::invalid_argument if @p rhs is not on the owned-cell map.
 */
template<TpetraTypePack Pack>
void add_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    typename Pack::vector_type& rhs, typename Pack::scalar_type correction_weight = 1.0);

/**
 * @brief Add the explicit non-orthogonal diffusion correction for a
 *        vector field to a three-column RHS.
 *
 * The correction is applied component-wise using the vector least-squares
 * gradient reconstruction. Boundary faces are treated as prescribed-value
 * diffusion faces, matching vector transport-system assembly.
 *
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param gradient_stencils Optional cached interior reconstruction stencil.
 * @throws std::invalid_argument if @p rhs is incompatible with the mesh or
 *         does not contain three component vectors.
 */
template<TpetraTypePack Pack>
void add_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    typename Pack::scalar_type diffusivity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    const std::vector<detail::LeastSquaresGradientStencil<Mesh<Pack>>>* gradient_stencils = nullptr);

/**
 * @brief Add a scalar explicit non-orthogonal correction using a
 *        cell-centered variable diffusion coefficient and boundary samples.
 *
 * @param boundary_value Boundary-value provider used by gradient reconstruction.
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param boundary_coefficient Boundary-face coefficient provider receiving
 *        the owner-cell value as its fallback.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @throws std::invalid_argument if fields use different meshes, @p rhs uses
 *         an incompatible map, or a cell coefficient is negative.
 */
template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0, BoundaryCoefficientProvider<Pack> boundary_coefficient = {},
    const std::vector<detail::AffineLeastSquaresGradientStencil<Mesh<Pack>>>* gradient_stencils = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/** @brief Backward-compatible scalar overload omitting boundary samples. */
template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const CellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    typename Pack::vector_type& rhs, typename Pack::scalar_type correction_weight = 1.0,
    BoundaryCoefficientProvider<Pack> boundary_coefficient = {});

/**
 * @brief Add a vector explicit non-orthogonal correction using a
 *        cell-centered variable diffusion coefficient and boundary samples.
 *
 * @param boundary_value Boundary-value provider used by gradient reconstruction.
 * @param correction_weight Fraction of the correction added to @p rhs.
 * @param boundary_coefficient Boundary-face coefficient provider receiving
 *        the owner-cell value as its fallback.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @param cached_boundary_locations Optional mesh-bound boundary lookup.
 * @throws std::invalid_argument if fields use different meshes, @p rhs is
 *         incompatible, or a cell coefficient is negative.
 */
template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, VectorBoundaryValueProvider<Pack> boundary_value,
    typename Pack::multi_vector_type& rhs, typename Pack::scalar_type correction_weight = 1.0,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    BoundaryCoefficientProvider<Pack> boundary_coefficient = {},
    const std::vector<detail::VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>* gradient_stencils = nullptr,
    const std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>* cached_boundary_locations = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/** @brief Backward-compatible vector overload omitting boundary samples. */
template<TpetraTypePack Pack>
void add_variable_explicit_non_orthogonal_correction(const VectorCellField<Pack>& correction_field,
    const CellField<Pack>& coefficient_field, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight = 1.0,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    BoundaryCoefficientProvider<Pack> boundary_coefficient = {});

/**
 * @brief Add the explicit deviatoric transpose-gradient part of a symmetric
 *        viscous stress to a momentum RHS.
 *
 * The component Laplacian in physical_momentum_transport_system() supplies
 * @f$\nabla\cdot(\mu\nabla\mathbf{u})/\rho@f$ implicitly.  A Newtonian/RANS
 * stress is symmetric and deviatoric, so this routine adds the remaining
 * lagged term
 * @f$\nabla\cdot[\mu((\nabla\mathbf{u})^T
 * - 2/3\,\nabla\cdot\mathbf{u}\,I)]/\rho@f$ as face tractions. Cell gradients
 * are reconstructed from @p old_velocity and the supplied boundary values,
 * then interpolated to interior faces. The contribution is integrated over
 * each control volume because @p rhs stores integrated momentum balances.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity Lagged velocity used by the explicit stress term.
 * @param dynamic_viscosity Molecular or effective dynamic-viscosity field.
 * @param reference_density Constant reference density.
 * @param boundary_value Boundary-face velocity provider.
 * @param[in,out] rhs Three-component owned-cell momentum RHS.
 * @param boundary_stress Boundary-face stress selector.
 * @param gradient_stencils Optional materialized affine reconstruction.
 * @param cached_boundary_locations Optional mesh-bound boundary lookup.
 */
template<TpetraTypePack Pack>
void add_explicit_deviatoric_transpose_gradient_stress(const VectorCellField<Pack>& old_velocity,
    const CellField<Pack>& dynamic_viscosity, typename Pack::scalar_type reference_density,
    VectorBoundaryValueProvider<Pack> boundary_value, typename Pack::multi_vector_type& rhs,
    BoundaryFaceSelector boundary_stress = detail::AlwaysDiffuseBoundary{},
    BoundaryCoefficientProvider<Pack> boundary_coefficient = {},
    const std::vector<detail::VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>>* gradient_stencils = nullptr,
    const std::vector<detail::BoundaryFaceLocation<Mesh<Pack>>>* cached_boundary_locations = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/**
 * @brief Assemble a scalar diffusion system with an explicit
 *        non-orthogonal correction from a previous solution.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @param correction_field Scalar field whose gradient drives the
 *        non-orthogonal correction.
 * @return Matrix/RHS pair with the explicit correction included.
 * @throws std::invalid_argument if @p correction_field is not on the
 *         target mesh.
 */
template<TpetraTypePack Pack>
DiffusionSystem<Pack> explicit_non_orthogonal_diffusion_system(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, const CellField<Pack>& correction_field);

/**
 * @brief Assemble scalar diffusion with an implicit least-squares
 *        non-orthogonal contribution.
 *
 * The orthogonal two-point contribution is assembled exactly as in
 * diffusion_system(). The non-orthogonal part linearizes the same
 * least-squares gradient reconstruction used by cell_gradient().
 *
 * @param non_orthogonal_implicit_weight Fraction of the tangential term to
 *        place in the matrix. Use 1.0 for a fully implicit operator and
 *        0.5 for the built-in hybrid treatment.
 * @param partition_correction_field Lagged field used to reconstruct the
 *        remote half of a partition-face gradient. The owned half remains in
 *        the matrix. This field is required when a nonzero implicit weight is
 *        assembled on a distributed partition face because the remote
 *        cell's extended least-squares stencil is not locally addressable.
 * @throws std::invalid_argument if diffusivity is not finite and
 *         non-negative, the implicit weight is not finite or outside
 *         `[0, 1]`, ranks disagree about whether an implicit correction or
 *         correction field is present, the correction field uses another
 *         mesh, or a distributed partition face needs a missing field.
 */
template<TpetraTypePack Pack>
DiffusionSystem<Pack> implicit_non_orthogonal_diffusion_system(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, typename Pack::scalar_type non_orthogonal_implicit_weight = 1.0,
    const CellField<Pack>* partition_correction_field = nullptr);

/**
 * @brief Assemble a fully implicit scalar non-orthogonal diffusion system.
 *
 * On distributed meshes, @p partition_correction_field supplies the lagged
 * remote half of partition-face gradients.
 */
template<TpetraTypePack Pack>
DiffusionSystem<Pack> fully_implicit_non_orthogonal_diffusion_system(const Mesh<Pack>& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, const CellField<Pack>* partition_correction_field = nullptr);

/**
 * @brief Assemble a diffusion system selected by the runtime
 *        non-orthogonal treatment switch.
 *
 * Explicit and hybrid systems use @p correction_field for the explicit RHS
 * fraction when it is supplied. If no correction field is provided,
 * `explicit` falls back to the orthogonal matrix. A nonzero implicit fraction
 * additionally requires the field when the mesh has distributed partition
 * faces so their remote gradient half can be synchronized.
 *
 * @throws std::invalid_argument if @p correction_field uses another mesh or
 *         @p treatment is invalid.
 */
template<TpetraTypePack Pack>
DiffusionSystem<Pack> non_orthogonal_diffusion_system(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment, const CellField<Pack>* correction_field = nullptr);

/**
 * @brief Apply the full scalar non-orthogonal diffusion residual.
 *
 * The returned vector stores the integrated operator
 * @f$-\nabla\cdot(\Gamma\nabla\phi)@f$ using the orthogonal two-point
 * stencil plus the least-squares tangential correction. The input field's
 * overlap values must be synchronized before calling.
 *
 * @throws std::invalid_argument if @p diffusivity is negative.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::vector_type> full_diffusion_residual(const CellField<Pack>& field,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition);

/**
 * @brief Solve a scalar diffusion equation with explicit non-orthogonal
 *        correction sweeps.
 *
 * A value of zero performs only the orthogonal solve. Positive values run
 * that solve followed by @p nNonOrthogonalCorrectors explicit correction
 * solves using the latest solution gradient.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @param[in,out] solution Solution field; zeroed before each solve and
 *        updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 * @throws std::invalid_argument if @p solution is not on the target mesh
 *         or @p nNonOrthogonalCorrectors is negative.
 */
template<TpetraTypePack Pack>
bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    CellField<Pack>& solution, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {});

/**
 * @brief Solve scalar diffusion using the selected non-orthogonal treatment.
 *
 * For `explicit`, @p nNonOrthogonalCorrectors has the same meaning as in
 * solve_explicit_non_orthogonal_diffusion(). For `implicit`, one fully
 * implicit solve is performed. For `hybrid`, an initial half-implicit solve
 * is followed by @p nNonOrthogonalCorrectors correction solves that put the
 * remaining half of the tangential term on the RHS.
 *
 * @return `true` if every linear solve converged; otherwise `false`.
 * @throws std::invalid_argument if @p solution uses another mesh, the
 *         corrector count is negative, or @p treatment is invalid.
 */
template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    CellField<Pack>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {});

/**
 * @brief Solve a scalar diffusion equation with zero source and explicit
 *        non-orthogonal correction sweeps.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param[in,out] solution Solution field; updated on return.
 * @param nNonOrthogonalCorrectors Number of explicit non-orthogonal
 *        correction sweeps (zero for orthogonal-only solve).
 * @param linear_options Linear solver options.
 * @return true if all solves converged, false otherwise.
 */
template<TpetraTypePack Pack>
bool solve_explicit_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, CellField<Pack>& solution, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options = {});

/**
 * @brief Solve a zero-source scalar diffusion equation using the selected
 *        non-orthogonal treatment.
 */
template<TpetraTypePack Pack>
bool solve_non_orthogonal_diffusion(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, CellField<Pack>& solution,
    NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors, const LinearSolverOptions& linear_options = {});

} // namespace SimpleFluid::FVM
