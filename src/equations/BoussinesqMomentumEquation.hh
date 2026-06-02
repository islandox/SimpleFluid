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
#include <functional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Boussinesq momentum update for coupled three-component velocity fields.
 *
 * The solver stores velocity as a three-column MultiVector-backed field.
 * This equation class advances all velocity components in a single linear solve.
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
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type =
        std::function<typename velocity_field_type::vec_type(local_ordinal_type)>;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    void advance_velocity(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const field_type& temperature,
        const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_velocity(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const field_type& temperature,
        const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
};

} // namespace SimpleFluid
