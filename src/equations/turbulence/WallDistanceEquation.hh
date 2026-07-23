/**
 * @file WallDistanceEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Distributed Poisson reconstruction of cell-to-wall distance.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "FVM/NonOrthogonalTreatment.hh"
#include "dataclass/typedefs.hh"
#include "fields/CellField.hh"
#include "solvers/BelosLinearSolver.hh"

namespace SimpleFluid
{

/**
 * @brief Numerical controls for the Poisson wall-distance solve.
 *
 * The default explicit treatment performs two non-orthogonal correction
 * sweeps after the initial orthogonal solve.
 */
struct WallDistanceEquationOptions
{
    FVM::NonOrthogonalTreatment non_orthogonal_treatment =
        FVM::NonOrthogonalTreatment::Explicit;
    int non_orthogonal_correctors = 2;
    LinearSolverOptions linear_solver{};
};

/**
 * @brief Validate numerical controls for a Poisson wall-distance solve.
 * @throws std::invalid_argument for invalid treatment, corrector, or solver
 *         controls.
 */
void validate_wall_distance_equation_options(
    const WallDistanceEquationOptions& options);

/**
 * @brief Recover a positive cell-centered wall-distance field in parallel.
 *
 * The equation solves
 * @f[
 *   -\nabla^2\psi = 1
 * @f]
 * with @f$\psi=0@f$ on the selected wall batches and homogeneous Neumann
 * conditions on every other exterior boundary. The Eikonal distance is then
 * reconstructed with the cancellation-resistant form
 * @f[
 *   y = \frac{2\psi}
 *            {\sqrt{\lvert\nabla\psi\rvert^2 + 2\psi}
 *             + \lvert\nabla\psi\rvert}.
 * @f]
 *
 * Boundary names and numerical controls must agree on every rank. A selected
 * name is valid when at least one physical face with that name exists
 * globally; ranks without a local selected wall participate normally.
 *
 * @tparam Pack Tpetra type pack used by the mesh and fields.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes>
class PoissonWallDistanceEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;

    /**
     * @brief Construct the equation on a distributed mesh.
     * @throws std::invalid_argument if @p mesh is null.
     */
    explicit PoissonWallDistanceEquation(SP<const mesh_type> mesh);

    /**
     * @brief Solve for distance to the union of selected wall boundaries.
     *
     * The output is replaced atomically after a converged solve produces
     * finite, strictly positive values in every owned cell.
     *
     * @param wall_boundary_names Exterior batch names treated as walls.
     * @param[out] wall_distance Cell-centered distance to the selected walls.
     * @param options Non-orthogonal and linear-solver controls.
     * @throws std::invalid_argument for invalid or rank-inconsistent inputs.
     * @throws std::runtime_error if the solve fails or produces invalid data.
     */
    void solve(const ArrString& wall_boundary_names,
               field_type& wall_distance,
               const WallDistanceEquationOptions& options = {}) const;

private:
    SP<const mesh_type> d_mesh;
};

} // namespace SimpleFluid
