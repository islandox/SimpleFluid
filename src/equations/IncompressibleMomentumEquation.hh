/**
 * @file IncompressibleMomentumEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Incompressible velocity-transport assembly and solution.
 * @version 0.1
 * @date 2026-06-21
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
#include "FVM/Operators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <functional>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Momentum update for incompressible three-component velocity fields.
 *
 * This class owns generic velocity transport assembly, boundary treatment,
 * non-orthogonal correction sweeps, and linear solves. Physics-specific
 * momentum equations provide their acceleration through the source callback.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class IncompressibleMomentumEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type =
        std::function<typename velocity_field_type::vec_type(local_ordinal_type)>;
    using system_type = FVM::VectorTransportSystem<Pack>;

    explicit IncompressibleMomentumEquation(SP<const mesh_type> mesh);
    virtual ~IncompressibleMomentumEquation() = default;

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    LinearSolveSummary advance_velocity(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    LinearSolveSummary advance_velocity(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

    system_type assemble_system(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const velocity_field_type* correction_field = nullptr) const;

    system_type assemble_system(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const source_type& right_hand_source,
        const velocity_field_type* correction_field = nullptr) const;

    LinearSolveSummary advance_velocity_physical(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const field_type& dynamic_viscosity,
        scalar_type reference_density,
        velocity_field_type& velocity,
        const source_type& acceleration_source,
        const LinearSolverOptions& linear_options = {}) const;

    system_type assemble_physical_system(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const field_type& dynamic_viscosity,
        scalar_type reference_density,
        const source_type& acceleration_source,
        const velocity_field_type* correction_field = nullptr) const;

private:
    void validate_transport_inputs(
        const velocity_field_type& old_velocity,
        const FaceField<Pack>& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const velocity_field_type* correction_field) const;

    SP<const mesh_type> d_mesh;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
    mutable Teuchos::RCP<typename Pack::matrix_type>
        d_cached_physical_transport_matrix;
    mutable bool
        d_cached_physical_graph_supports_non_orthogonal_correction = false;
    mutable BelosLinearSolver<Pack> d_linear_solver;
};

} // namespace SimpleFluid
