/**
 * @file BoussinesqMomentumEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Boussinesq buoyancy and velocity-transport updates.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Boussinesq momentum update for component-wise velocity fields.
 *
 * The solver stores velocity as a three-column MultiVector-backed field.
 * This equation class advances all velocity components together.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqMomentumEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_velocity_field_type = VectorFaceField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const field_type& temperature,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const field_type& temperature,
        const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const face_velocity_field_type& face_velocity,
        const field_type& temperature,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const face_velocity_field_type& face_velocity,
        const field_type& temperature,
        const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
    mutable std::vector<scalar_type> d_cached_old_component;
    mutable Teuchos::RCP<typename Pack::vector_type> d_cached_solution;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
};

} // namespace SimpleFluid
