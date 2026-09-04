/**
 * @file PressureProjectionEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Pressure projection equation for the Boussinesq solver.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "FVM/CellGradientScheme.hh"
#include "FVM/Operators.hh"
#include "SimpleFluidExport.hh"
#include "equations/EquationValidation.hh"
#include "equations/VolumeContinuityTarget.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "fields/VectorCellField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack> class SIMPLEFLUID_SOLVERS_EXPORT FluidSolver;

namespace detail
{

/**
 * @brief Test-only accessor for pressure-projection implementation state.
 * @tparam Pack Tpetra type pack used by the equation.
 */
template<TpetraTypePack Pack> struct PressureProjectionEquationTestAccess;

} // namespace detail

/**
 * @brief Build the default linear-solver options for pressure projection.
 * @return Options selecting the MueLu preconditioner.
 */
inline LinearSolverOptions pressure_projection_linear_solver_options()
{
    LinearSolverOptions options;
    options.preconditioner = LinearPreconditioner::MueLu;
    return options;
}

/**
 * @brief Pressure-projection solve used to correct transient velocity fields.
 *
 * The class keeps the legacy pressure reset and the pressure-correction
 * projection used by the Boussinesq solver separated from orchestration.
 * Projection unknowns are normalized internally, while returned pressure
 * corrections are physical gauge pressures in Pa.
 *
 * @tparam Pack Tpetra type pack used for matrix/vector storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT PressureProjectionEquation
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using face_velocity_field_type = typename field_traits::vector_face_type;
    using velocity_boundary_cache_type = std::conditional_t<std::same_as<mesh_type, Mesh<Pack>>,
        FVM::VelocityBoundaryCache<Pack>, FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>>;
    using face_flux_workspace_type =
        std::conditional_t<std::same_as<mesh_type, Mesh<Pack>>, FVM::PressureWeightedFaceFluxWorkspace<Pack>,
            FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, mesh_type>>;
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type = std::function<scalar_type(local_ordinal_type)>;
    using fixed_boundary_flux_provider_type = std::function<scalar_type(int, size_t, local_ordinal_type)>;
    using continuity_target_type = VolumeContinuityTarget<Pack, mesh_type>;
    using continuity_residual_type = VolumeContinuityResiduals<scalar_type>;
    /** @brief Pressure-correction solve and continuity statistics. */
    struct ProjectionResult
    {
        scalar_type pressure_correction = {}; ///< L2 norm of the Pa update.
        scalar_type continuity = {};          ///< L2 continuity residual after projection.
        LinearSolveStatistics linear_solve;
        continuity_residual_type continuity_residuals;
    };

    explicit PressureProjectionEquation(SP<const mesh_type> mesh,
        LinearSolverOptions linear_options = pressure_projection_linear_solver_options(),
        BoundaryConditionMap pressure_boundary_conditions = {},
        FVM::CellGradientScheme gradient_scheme = FVM::CellGradientScheme::LeastSquares);

    void set_linear_solver_options(LinearSolverOptions options);

    const LinearSolverOptions& linear_solver_options() const noexcept;

    void rebuild_matrix() const;

    /** Refresh geometry-dependent workspaces and discard numeric solver state. */
    void refresh_geometry();

    /**
     * Prescribe exact owner-oriented boundary volume fluxes [m^3/s].
     *
     * The named physical pressure boundary remains unchanged, while its
     * pressure-correction condition becomes homogeneous Neumann. Predictor and
     * final flux reconstructions are overwritten from @p provider. Change
     * @p generation whenever any provided value changes.
     */
    void set_fixed_boundary_flux_provider(
        std::vector<std::string> boundary_names, fixed_boundary_flux_provider_type provider, std::uint64_t generation);

    /** Restore the configured physical pressure-correction boundary types. */
    void clear_fixed_boundary_flux_provider();

    /** Return a Rhie--Chow cache with fixed-flux faces validation-neutral. */
    velocity_boundary_cache_type pressure_flux_boundary_cache(
        const velocity_boundary_cache_type& boundary_cache) const;

    /** Overwrite registered owned boundary faces with their exact fluxes. */
    void apply_fixed_boundary_fluxes(face_flux_field_type& fluxes) const;

    void solve(field_type& pressure);

    ProjectionResult project(field_type& pressure, scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity);

    /** Project to the named integrated volume-continuity target [m^3/s]. */
    ProjectionResult project(field_type& pressure, scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity,
        const continuity_target_type& continuity_target);

    /**
     * @brief Project velocity while accumulating a physical pressure field.
     *
     * The predictor face flux is reconstructed from @p pressure before the
     * homogeneous correction is solved.  On return, @p pressure includes the
     * correction stored in @p pressure_correction, and
     * corrected_face_fluxes() is consistent with both updated fields.
     *
     * @param[in,out] pressure Accumulated physical gauge pressure in Pa.
     * @param[out] pressure_correction Physical pressure correction in Pa.
     * @param time_step Positive time-step size.
     * @param reference_density Positive pressure-normalization density.
     * @param velocity_boundary_cache Cached velocity boundary conditions.
     * @param[in,out] velocity Pressure-corrected cell velocity.
     * @return Correction, continuity, and linear-solve statistics.
     */
    ProjectionResult project(field_type& pressure, field_type& pressure_correction, scalar_type time_step,
        scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
        velocity_field_type& velocity);

    /** Accumulated-pressure projection to an integrated cell target [m^3/s]. */
    ProjectionResult project(field_type& pressure, field_type& pressure_correction, scalar_type time_step,
        scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
        velocity_field_type& velocity, const continuity_target_type& continuity_target);

    ProjectionResult project(field_type& pressure, scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity,
        const source_type& right_hand_source);

    /**
     * @brief Return the total-pressure-consistent flux from the latest project().
     *
     * The reference remains valid until the next project() call or equation
     * destruction.
     */
    const face_flux_field_type& corrected_face_fluxes() const noexcept { return d_cached_face_fluxes; }

private:
    friend struct detail::PressureProjectionEquationTestAccess<Pack>;
    friend class FluidSolver<Pack>;

    SIMPLEFLUID_EQUATIONS_LOCAL
    static Teuchos::RCP<const map_type> require_owned_cell_map(const SP<const mesh_type>& mesh);

    /**
     * @brief Project using the final face flux cached by the preceding
     *        accumulated-pressure projection as this predictor.
     *
     * This is restricted to FluidSolver's adjacent PISO correctors: pressure
     * and velocity must be unchanged since the preceding project() returned.
     */
    ProjectionResult project_reusing_cached_predictor(field_type& pressure, field_type& pressure_correction,
        scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity);

    /** Reuse an adjacent predictor only for the same target generation. */
    ProjectionResult project_reusing_cached_predictor(field_type& pressure, field_type& pressure_correction,
        scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity,
        const continuity_target_type& continuity_target);

    SIMPLEFLUID_EQUATIONS_LOCAL
    ProjectionResult project_impl(field_type& pressure_correction, scalar_type time_step, scalar_type reference_density,
        const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity,
        const continuity_target_type& continuity_target, field_type* accumulated_pressure,
        bool reuse_cached_predictor_flux = false, bool enforce_global_compatibility = true);

    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_fixed_boundary_flux_provider() const;

    SP<const mesh_type> d_mesh;
    LinearSolverOptions d_linear_options;
    BoundaryConditionMap d_pressure_boundary_conditions;
    BoundaryConditionMap d_pressure_correction_boundary_conditions;
    FVM::CellGradientScheme d_gradient_scheme = FVM::CellGradientScheme::LeastSquares;
    mutable std::optional<typename Pack::global_ordinal_type> d_pressure_gauge_gid;
    mutable face_flux_field_type d_cached_face_fluxes;
    mutable face_flux_workspace_type d_face_flux_workspace;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_pressure_matrix;
    mutable Teuchos::RCP<typename Pack::vector_type> d_cached_rhs;
    real_t d_rhs_norm_reference = {};
    bool d_cached_predictor_flux_valid = false;
    std::size_t d_cached_predictor_flux_reuse_count = 0;
    std::uint64_t d_cached_target_generation = 0;
    std::uint64_t d_cached_target_geometry_epoch = 0;
    scalar_type d_cached_predictor_time_step = {};
    scalar_type d_cached_predictor_reference_density = {};
    std::uint64_t d_cached_fixed_boundary_flux_generation = 0;
    std::vector<std::string> d_fixed_boundary_flux_names;
    fixed_boundary_flux_provider_type d_fixed_boundary_flux_provider;
    std::uint64_t d_fixed_boundary_flux_generation = 0;
    BelosLinearSolver<Pack> d_linear_solver;
};

extern template class PressureProjectionEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class PressureProjectionEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
