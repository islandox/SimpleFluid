/**
 * @file FluidSolver.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for FluidSolver.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FluidSolver.hh"
#include "geometry/MeshFactory.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Validate and return a legacy mesh pointer.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Candidate mesh pointer.
 * @return Validated non-null mesh.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "FluidSolver");
}

/**
 * @brief Obtain a legacy mesh from a type-erased mesh handle.
 *
 * Cartesian handles are rebuilt through the legacy structured-mesh path.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Candidate mesh handle.
 * @return Legacy mesh used by the current equation stack.
 * @throws std::invalid_argument if the handle is null, unsupported, or distributed Cartesian.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_legacy_mesh(
    const SP<const MeshHandle<Pack>>& mesh) -> SP<const mesh_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "FluidSolver requires a non-null mesh handle.");
    }
    auto legacy = mesh->legacy_mesh();
    if (legacy)
    {
        return legacy;
    }

    const auto* cartesian =
        std::get_if<typename MeshHandle<Pack>::CartesianPtr>(
            &mesh->variant());
    if (!cartesian)
    {
        throw std::invalid_argument(
            "FluidSolver currently supports STK and Cartesian mesh handles.");
    }
    if (mesh->num_owned_cells() != (*cartesian)->num_cells())
    {
        throw std::invalid_argument(
            "FluidSolver Cartesian compatibility currently requires a "
            "serial mesh.");
    }

    auto database = std::make_shared<Database>();
    database->set("dimension", 3);
    database->set("mesh_size", real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(MeshFactory::DomainType::BOX));
    database->set("X", ArrReal((*cartesian)->cell_edges()[0]));
    database->set("Y", ArrReal((*cartesian)->cell_edges()[1]));
    database->set("Z", ArrReal((*cartesian)->cell_edges()[2]));
    database->set(
        "domain_exterior_face_types",
        ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return MeshFactory(database).template build<Pack>();
}

/**
 * @brief Construct a fluid solver from a legacy mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh.
 * @param boundary_conditions Velocity and pressure boundary conditions.
 * @param time_options Time-stepping and coupling options.
 * @param linear_options Linear solver options.
 * @throws std::invalid_argument if the mesh or time-step options are invalid.
 */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh))),
          std::move(boundary_conditions),
          time_options,
          linear_options)
{
}

/**
 * @brief Construct a fluid solver from a type-erased mesh handle.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh handle.
 * @param boundary_conditions Velocity and pressure boundary conditions.
 * @param time_options Time-stepping and coupling options.
 * @param linear_options Linear solver options.
 * @throws std::invalid_argument if the mesh or time-step options are invalid.
 */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          true)
{
}

/**
 * @brief Construct a derived solver while deferring momentum-equation registration.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh handle.
 * @param boundary_conditions Velocity and pressure boundary conditions.
 * @param time_options Time-stepping and coupling options.
 * @param linear_options Linear solver options.
 * @param tag Deferred-registration dispatch tag.
 * @throws std::invalid_argument if the mesh or time-step options are invalid.
 */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    DeferredMomentumEquationTag)
    : FluidSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          false)
{
}

/**
 * @brief Initialize common fields, equations, and pressure-coupling workspaces.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh handle.
 * @param boundary_conditions Velocity and pressure boundary conditions.
 * @param time_options Time-stepping and coupling options.
 * @param linear_options Linear solver options.
 * @param register_momentum_equation Whether to register the base momentum equation.
 * @throws std::invalid_argument if the time step is non-positive.
 */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    bool register_momentum_equation)
    : d_mesh(require_legacy_mesh(mesh)),
      d_problem(std::make_shared<MeshHandle<Pack>>(d_mesh),
                std::move(boundary_conditions),
                time_options,
                linear_options)
{
    if (d_problem.time_options().time_step <= 0.0)
    {
        throw std::invalid_argument(
            "FluidSolver requires a positive time step.");
    }

    d_problem.template emplace_object<FVM::VelocityBoundaryCache<Pack>>(
        "velocity_boundary_cache",
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions()));
    d_problem.template emplace_object<face_flux_workspace_type>(
        "pressure_face_flux_workspace", d_mesh);
    if (register_momentum_equation)
    {
        d_problem.template emplace_object<
            IncompressibleMomentumEquation<Pack>>(
                "momentum_equation", d_mesh);
    }
    auto pressure_linear_options = d_problem.linear_options();
    pressure_linear_options.preconditioner =
        LinearPreconditioner::MueLu;
    // The pressure-Poisson matrix is cached by PressureProjectionEquation and
    // remains numerically unchanged until rebuild_matrix() replaces it.
    pressure_linear_options.reuse_preconditioner = true;
    d_problem.template emplace_object<PressureProjectionEquation<Pack>>(
        "pressure_projection",
        d_mesh,
        pressure_linear_options,
        d_problem.boundary_conditions().pressure);
    d_problem.template emplace_object<CoupledPressureVelocitySolver<Pack>>(
        "coupled_pressure_velocity_solver", d_mesh);
    d_problem.template emplace_object<field_type>(
        "pressure", d_mesh, "pressure");
    d_problem.template emplace_object<field_type>(
        "pressure_correction", d_mesh, "pressure_correction");
    d_problem.template emplace_object<velocity_field_type>(
        "velocity", d_mesh, "velocity");
    d_problem.template emplace_object<velocity_field_type>(
        "predictor_pressure_gradient",
        d_mesh,
        "predictor_pressure_gradient");
    d_problem.template emplace_object<velocity_field_type>(
        "pressure_velocity_predictor",
        d_mesh,
        "pressure_velocity_predictor");
    d_problem.template emplace_object<face_flux_field_type>(
        "old_face_flux", d_mesh, "old_face_flux");
    d_problem.template emplace_object<face_flux_field_type>(
        "projected_face_flux", d_mesh, "projected_face_flux");
    d_problem.template emplace_object<residual_type>(
        "pressure_velocity_residuals");
}

/**
 * @brief Return the immutable physical gauge-pressure field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned pressure field in Pa.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure() const noexcept -> const field_type&
{
    return d_problem.template object<field_type>("pressure");
}

/**
 * @brief Return the immutable cell-centered velocity field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned velocity field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity() const noexcept
    -> const velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

/**
 * @brief Return the mutable physical gauge-pressure field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned pressure field in Pa.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure() noexcept -> field_type&
{
    return d_problem.template object<field_type>("pressure");
}

/**
 * @brief Return the mutable cell-centered velocity field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned velocity field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity() noexcept -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

/**
 * @brief Return the latest pressure-velocity residual summary.
 *
 * @tparam Pack Tpetra type pack.
 * @return Residuals recorded by the most recent coupling solve.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::last_pressure_velocity_residuals() const noexcept
    -> const residual_type&
{
    return pressure_velocity_residuals();
}

/**
 * @brief Return the Problem-owned base momentum equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored incompressible momentum equation.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::momentum_equation()
    -> IncompressibleMomentumEquation<Pack>&
{
    return d_problem.template object<
        IncompressibleMomentumEquation<Pack>>("momentum_equation");
}

/**
 * @brief Return the Problem-owned pressure projection equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored pressure projection equation.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_projection()
    -> PressureProjectionEquation<Pack>&
{
    return d_problem.template object<
        PressureProjectionEquation<Pack>>("pressure_projection");
}

/**
 * @brief Return the monolithic pressure-velocity solver.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned coupled solver.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::coupled_pressure_velocity_solver()
    -> CoupledPressureVelocitySolver<Pack>&
{
    return d_problem.template object<
        CoupledPressureVelocitySolver<Pack>>(
            "coupled_pressure_velocity_solver");
}

/**
 * @brief Return cached velocity boundary values and metadata.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned velocity boundary cache.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity_boundary_cache()
    -> FVM::VelocityBoundaryCache<Pack>&
{
    return d_problem.template object<
        FVM::VelocityBoundaryCache<Pack>>("velocity_boundary_cache");
}

/**
 * @brief Return reusable pressure-weighted face-flux workspace.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned face-flux workspace.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_face_flux_workspace()
    -> face_flux_workspace_type&
{
    return d_problem.template object<face_flux_workspace_type>(
        "pressure_face_flux_workspace");
}

/**
 * @brief Return the mutable pressure-correction field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned pressure-correction field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_correction() -> field_type&
{
    return d_problem.template object<field_type>("pressure_correction");
}

/**
 * @brief Return the cached pressure gradient used by the momentum predictor.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned pressure-gradient field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::predictor_pressure_gradient()
    -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>(
        "predictor_pressure_gradient");
}

/**
 * @brief Return the velocity snapshot used to measure predictor updates.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned predictor velocity field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::predictor_velocity() -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>(
        "pressure_velocity_predictor");
}

/**
 * @brief Return face fluxes evaluated before the current correction.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned old face-flux field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::old_face_fluxes() -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>("old_face_flux");
}

/**
 * @brief Return pressure-corrected face fluxes.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned projected face-flux field.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::projected_face_fluxes() -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>(
        "projected_face_flux");
}

/**
 * @brief Return mutable pressure-velocity residual storage.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned residual summary.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_velocity_residuals() -> residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

/**
 * @brief Return immutable pressure-velocity residual storage.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned residual summary.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_velocity_residuals() const
    -> const residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

/**
 * @brief Compute the global volume-weighted norm of a velocity update.
 *
 * @tparam Pack Tpetra type pack.
 * @param before Velocity field before the update.
 * @param after Velocity field after the update.
 * @return Global L2-like update norm.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity_update_norm(
    const velocity_field_type& before,
    const velocity_field_type& after) const -> scalar_type
{
    EquationValidation::require_mesh_match(
        *d_mesh, before, "FluidSolver");
    EquationValidation::require_mesh_match(
        *d_mesh, after, "FluidSolver");

    scalar_type norm_squared = {};
    const auto before_values = before.owned_read_view();
    const auto after_values = after.owned_read_view();
    const auto& volumes = d_mesh->host_views().cell_geometry.volume;
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto delta_x =
            after_values(cell_lid, 0) - before_values(cell_lid, 0);
        const auto delta_y =
            after_values(cell_lid, 1) - before_values(cell_lid, 1);
        const auto delta_z =
            after_values(cell_lid, 2) - before_values(cell_lid, 2);
        norm_squared +=
            (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
            * volumes[owned];
    }

    using std::sqrt;
    return sqrt(global_sum(norm_squared));
}

/**
 * @brief Sum a scalar contribution across the mesh communicator.
 *
 * @tparam Pack Tpetra type pack.
 * @param local_value Local rank contribution.
 * @return Global sum across ranks.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::global_sum(
    scalar_type local_value) const -> scalar_type
{
    scalar_type global_value{};
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(),
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/**
 * @brief Advance the base momentum equation using the old pressure gradient.
 *
 * @tparam Pack Tpetra type pack.
 * @return Aggregated momentum linear-solve summary.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::advance_momentum() -> LinearSolveSummary
{
    FVM::cell_gradient(
        pressure(),
        d_problem.boundary_conditions().pressure,
        predictor_pressure_gradient(),
        pressure_face_flux_workspace().gradient_cache());
    const auto inverse_reference_density =
        scalar_type{1} / pressure_reference_density();
    const auto pressure_gradient_values =
        predictor_pressure_gradient().owned_read_view();
    auto pressure_source =
        [&](local_ordinal_type cell_lid) -> vec_type
    {
        return {
            pressure_gradient_values(cell_lid, 0)
                * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 1)
                * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 2)
                * (-inverse_reference_density)};
    };

    return momentum_equation().advance_velocity(
        velocity(),
        old_face_fluxes(),
        velocity_boundary_cache(),
        d_problem.time_options(),
        velocity(),
        pressure_source,
        d_problem.linear_options());
}

/**
 * @brief Run one momentum predictor and record its velocity residual.
 *
 * @tparam Pack Tpetra type pack.
 * @return Aggregated predictor linear-solve summary.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::run_momentum_predictor() -> LinearSolveSummary
{
    {
        const auto velocity_values = velocity().owned_read_view();
        auto predictor_values = predictor_velocity().owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            predictor_values(cell_lid, 0) =
                velocity_values(cell_lid, 0);
            predictor_values(cell_lid, 1) =
                velocity_values(cell_lid, 1);
            predictor_values(cell_lid, 2) =
                velocity_values(cell_lid, 2);
        }
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(),
        d_problem.time_options().time_step
            / pressure_reference_density(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions().pressure,
        pressure_face_flux_workspace(),
        old_face_fluxes());
    const auto linear_summary = advance_momentum();
    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
    return linear_summary;
}

/**
 * @brief Project velocity, update pressure, and return correction statistics.
 *
 * @tparam Pack Tpetra type pack.
 * @return Pressure projection result for the current corrector.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::run_pressure_correction(
    bool reuse_cached_predictor_flux)
    -> typename PressureProjectionEquation<Pack>::ProjectionResult
{
    auto& projection = pressure_projection();
    const auto result =
        reuse_cached_predictor_flux
      ? projection.project_reusing_cached_predictor(
            pressure(),
            pressure_correction(),
            d_problem.time_options().time_step,
            pressure_reference_density(),
            velocity_boundary_cache(),
            velocity())
      : projection.project(
            pressure(),
            pressure_correction(),
            d_problem.time_options().time_step,
            pressure_reference_density(),
            velocity_boundary_cache(),
            velocity());
    // Downstream transport must use the same conservative face-flux field
    // whose balance produced result.continuity.
    projected_face_fluxes().data().update(
        scalar_type{1},
        projection.corrected_face_fluxes().data(),
        scalar_type{0});
    return result;
}

/**
 * @brief Assemble the base monolithic velocity-pressure system.
 *
 * @tparam Pack Tpetra type pack.
 * @return Coupled system assembled from current fields and options.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::assemble_coupled_system()
    -> coupled_system_type
{
    return coupled_pressure_velocity_solver().assemble(
        momentum_equation(),
        velocity(),
        pressure(),
        old_face_fluxes(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions(),
        d_problem.time_options(),
        pressure_reference_density());
}

/**
 * @brief Solve one monolithic velocity-pressure Krylov system.
 *
 * @tparam Pack Tpetra type pack.
 * @throws std::runtime_error if the coupled linear solve does not converge.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::solve_coupled_krylov()
{
    {
        const auto velocity_values = velocity().owned_read_view();
        auto predictor_values = predictor_velocity().owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            predictor_values(cell_lid, 0) =
                velocity_values(cell_lid, 0);
            predictor_values(cell_lid, 1) =
                velocity_values(cell_lid, 1);
            predictor_values(cell_lid, 2) =
                velocity_values(cell_lid, 2);
        }
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(),
        d_problem.time_options().time_step
            / pressure_reference_density(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions().pressure,
        pressure_face_flux_workspace(),
        old_face_fluxes());
    const auto system = assemble_coupled_system();
    const auto result =
        coupled_pressure_velocity_solver().solve(
            system,
            velocity(),
            pressure(),
            d_problem.linear_options());
    if (!result.converged)
    {
        throw std::runtime_error(
            "FluidSolver coupled Krylov solve did not converge.");
    }

    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
    pressure_velocity_residuals().pressure =
        result.achieved_tolerance;
    pressure_velocity_residuals().achieved_tolerance =
        result.achieved_tolerance;
    pressure_velocity_residuals().linear_iterations =
        result.iterations;
    d_last_step_statistics.nonlinear_iterations = 1;
    d_last_step_statistics.add(LinearSolveStatistics{
        result.converged,
        result.iterations,
        result.achieved_tolerance});

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(),
        d_problem.time_options().time_step
            / pressure_reference_density(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions().pressure,
        pressure_face_flux_workspace(),
        projected_face_fluxes());
    scalar_type continuity_norm_squared = {};
    {
        const auto projected_flux_values =
            projected_face_fluxes().owned_read_view();
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto balance =
                FVM::cell_flux_balance<Pack>(
                    *d_mesh,
                    projected_face_fluxes(),
                    projected_flux_values,
                    cell_lid);
            continuity_norm_squared += balance * balance;
        }
    }
    using std::sqrt;
    pressure_velocity_residuals().continuity =
        sqrt(global_sum(continuity_norm_squared));
}

/**
 * @brief Execute the configured SIMPLE, PISO, PIMPLE, or coupled loop.
 *
 * @tparam Pack Tpetra type pack.
 * @throws std::invalid_argument if a segregated loop has no required correctors.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::solve_pressure_velocity_coupling()
{
    pressure_velocity_residuals() = {};
    if (d_problem.time_options().pressure_velocity_coupling
        == PressureVelocityCoupling::CoupledKrylov)
    {
        solve_coupled_krylov();
        return;
    }

    if (d_problem.time_options().n_pressure_correctors < 1)
    {
        throw std::invalid_argument(
            "FluidSolver requires at least one pressure corrector.");
    }
    if (d_problem.time_options().n_outer_correctors < 1)
    {
        throw std::invalid_argument(
            "FluidSolver requires at least one outer corrector.");
    }

    const auto pressure_corrections =
        d_problem.time_options().pressure_velocity_coupling
                == PressureVelocityCoupling::SIMPLE
          ? 1
          : d_problem.time_options().n_pressure_correctors;
    const auto outer_corrections =
        d_problem.time_options().pressure_velocity_coupling
                == PressureVelocityCoupling::PIMPLE
          ? d_problem.time_options().n_outer_correctors
          : 1;

    for (int outer = 0; outer < outer_corrections; ++outer)
    {
        ++d_last_step_statistics.nonlinear_iterations;
        d_last_step_statistics.add(run_momentum_predictor());

        typename PressureProjectionEquation<Pack>::ProjectionResult result;
        for (int corrector = 0;
             corrector < pressure_corrections;
             ++corrector)
        {
            result = run_pressure_correction(corrector != 0);
            d_last_step_statistics.add(result.linear_solve);
        }

        pressure_velocity_residuals().pressure =
            result.pressure_correction;
        pressure_velocity_residuals().continuity =
            result.continuity;
    }
    pressure_velocity_residuals().linear_iterations =
        d_last_step_statistics.krylov_iterations;
    pressure_velocity_residuals().achieved_tolerance =
        d_last_step_statistics.achieved_tolerance;
}

/**
 * @brief Reset per-step statistics and synchronize initial velocity ghosts.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::begin_step()
{
    d_last_step_statistics = {};
    if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(velocity());
    }
}

/**
 * @brief Finalize residual statistics, synchronization, time, and step index.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::finish_step()
{
    d_last_step_statistics.momentum =
        pressure_velocity_residuals().momentum;
    d_last_step_statistics.pressure =
        pressure_velocity_residuals().pressure;
    d_last_step_statistics.continuity =
        pressure_velocity_residuals().continuity;
    d_mesh->sync_periodic_boundaries(velocity());
    d_time += d_problem.time_options().time_step;
    ++d_step_index;
}

/**
 * @brief Advance the base fluid solver by one physical time step.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::step()
{
    begin_step();
    solve_pressure_velocity_coupling();
    finish_step();
}

/**
 * @brief Advance one step and emit rank-zero progress.
 *
 * @tparam Pack Tpetra type pack.
 * @param progress_output Progress stream receiving the summary.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::step(ProgressStream& progress_output)
{
    step();
    write_step_progress(
        progress_output, d_problem.time_options().steps);
}

/**
 * @brief Advance a requested number of physical steps.
 *
 * @tparam Pack Tpetra type pack.
 * @param steps Number of steps to execute.
 * @throws std::invalid_argument if @p steps is negative.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::run(int steps)
{
    if (steps < 0)
    {
        throw std::invalid_argument(
            "FluidSolver::run steps cannot be negative.");
    }

    for (int step_id = 0; step_id < steps; ++step_id)
    {
        step();
    }
}

/**
 * @brief Advance several steps and emit one rank-zero progress line per step.
 *
 * @tparam Pack Tpetra type pack.
 * @param steps Number of steps to execute.
 * @param progress_output Progress stream receiving summaries.
 * @throws std::invalid_argument if @p steps is negative.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::run(
    int steps,
    ProgressStream& progress_output)
{
    if (steps < 0)
    {
        throw std::invalid_argument(
            "FluidSolver::run steps cannot be negative.");
    }

    const int final_step = d_step_index + steps;
    for (int step_id = 0; step_id < steps; ++step_id)
    {
        step();
        write_step_progress(progress_output, final_step);
    }
}

/**
 * @brief Write the latest step statistics on communicator rank zero.
 *
 * @tparam Pack Tpetra type pack.
 * @param progress_output Progress stream receiving the summary.
 * @param total_steps Final absolute step index for the current run.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::write_step_progress(
    ProgressStream& progress_output,
    int total_steps) const
{
    if (d_mesh->owned_cell_map()->getComm()->getRank() == 0)
    {
        progress_output.write(
            step_index(), total_steps, time(), d_last_step_statistics);
    }
}

/**
 * @brief Gather one rank-owned scalar field into VTU cell-data order.
 *
 * @tparam Pack Tpetra type pack.
 * @param field Cell field to collect.
 * @return Scalar values ordered by owned local cell identifier.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::collect_scalar_field(
    const field_type& field) const -> VTUWriter::ScalarData
{
    VTUWriter::ScalarData values;
    values.reserve(d_mesh->num_owned_cells());
    for (size_t lid = 0; lid < d_mesh->num_owned_cells(); ++lid)
    {
        values.push_back(static_cast<real_t>(
            field.local_value(
                static_cast<local_ordinal_type>(lid))));
    }
    return values;
}

/**
 * @brief Build a VTU writer containing mesh, pressure, and velocity data.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured in-memory VTU writer.
 */
template<TpetraTypePack Pack>
VTUWriter FluidSolver<Pack>::fluid_solution_writer() const
{
    if (!d_vtu_topology)
    {
        std::unordered_map<global_ordinal_type, global_index_t> node_lid;
        VTUWriter::VectorData node_coords;
        VTUWriter::Int64Data cell_node_offsets;
        VTUWriter::Int64Data cell_node_ids;
        VTUWriter::UInt8Data cell_types;

        auto append_node =
            [&](global_ordinal_type node_gid) -> global_index_t
        {
            const auto iter = node_lid.find(node_gid);
            if (iter != node_lid.end())
            {
                return iter->second;
            }

            const auto lid =
                static_cast<global_index_t>(node_coords.size());
            node_lid.emplace(node_gid, lid);
            node_coords.push_back(d_mesh->node_coord(node_gid));
            return lid;
        };

        for (size_t lid = 0; lid < d_mesh->num_owned_cells(); ++lid)
        {
            const auto& cell_info =
                d_mesh->cell(static_cast<local_ordinal_type>(lid));
            for (const auto node_gid : cell_info.node_gids)
            {
                cell_node_ids.push_back(append_node(node_gid));
            }
            cell_node_offsets.push_back(
                static_cast<global_index_t>(cell_node_ids.size()));
            cell_types.push_back(static_cast<std::uint8_t>(
                MeshUtils::vtu_cell_type_code(cell_info.type)));
        }

        d_vtu_topology = VTUWriter::make_topology(
            std::move(node_coords),
            std::move(cell_node_ids),
            std::move(cell_node_offsets),
            std::move(cell_types));
    }

    VTUWriter::VectorData velocity_values;
    velocity_values.reserve(d_mesh->num_owned_cells());
    for (size_t lid = 0; lid < d_mesh->num_owned_cells(); ++lid)
    {
        velocity_values.push_back(
            velocity().local_value(
                static_cast<local_ordinal_type>(lid)));
    }

    VTUWriter writer(d_vtu_topology);
    writer.add_scalar_cell_data(
        "pressure", collect_scalar_field(pressure()));
    writer.add_vector_cell_data(
        "velocity", std::move(velocity_values));
    return writer;
}

/**
 * @brief Write core pressure and velocity fields to a VTU file.
 *
 * @tparam Pack Tpetra type pack.
 * @param filename Output VTU path.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::write_solution_vtu(
    const std::string& filename) const
{
    fluid_solution_writer().write(
        filename, VTUWriter::Encoding::AppendedBinary);
}

} // namespace SimpleFluid
