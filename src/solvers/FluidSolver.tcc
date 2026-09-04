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

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{
namespace
{
namespace fluid_solver_detail
{

/** @brief Compute a communicator-wide volume-weighted velocity-update norm. */
template<class MeshType, class VelocityField, class GlobalSum>
auto volume_weighted_velocity_update_norm(
    const MeshType& mesh,
    const VelocityField& before,
    const VelocityField& after,
    GlobalSum global_sum) -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;
    using local_ordinal_type = typename MeshType::local_ordinal_type;

    EquationValidation::require_mesh_match(
        mesh, before, "FluidSolver");
    EquationValidation::require_mesh_match(
        mesh, after, "FluidSolver");

    scalar_type norm_squared{};
    const auto before_values = before.owned_read_view();
    const auto after_values = after.owned_read_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
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
            * static_cast<scalar_type>(mesh.cell_volume(cell_lid));
    }

    using std::sqrt;
    return sqrt(global_sum(norm_squared));
}

/** @brief Collect rank-owned scalar values in mesh-local cell order. */
template<class MeshType, class Field>
auto collect_owned_scalar_values(
    const MeshType& mesh,
    const Field& field) -> VTUWriter::ScalarData
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;

    VTUWriter::ScalarData values;
    values.reserve(mesh.num_owned_cells());
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        values.push_back(static_cast<real_t>(
            field.local_value(
                static_cast<local_ordinal_type>(owned))));
    }
    return values;
}

/** @brief Reduce a backend-provided cell metric to a global L2 norm. */
template<class MeshType, class CellMetric, class GlobalSum>
auto global_cell_metric_norm(
    const MeshType& mesh,
    CellMetric cell_metric,
    GlobalSum global_sum) -> typename MeshType::scalar_type
{
    using scalar_type = typename MeshType::scalar_type;
    using local_ordinal_type = typename MeshType::local_ordinal_type;

    scalar_type norm_squared{};
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value = cell_metric(cell_lid);
        norm_squared += value * value;
    }

    using std::sqrt;
    return sqrt(global_sum(norm_squared));
}

/** @brief Publish one accepted coupled solve into residual/statistics state. */
template<class Result, class Scalar, class Residual, class Statistics>
void record_coupled_result(
    const Result& result,
    Scalar momentum_norm,
    Residual& residuals,
    Statistics& statistics)
{
    residuals.momentum = momentum_norm;
    residuals.pressure = result.achieved_tolerance;
    residuals.achieved_tolerance = result.achieved_tolerance;
    residuals.linear_iterations = result.iterations;
    statistics.nonlinear_iterations = 1;
    statistics.add(LinearSolveStatistics{
        result.converged,
        result.iterations,
        result.achieved_tolerance});
}

} // namespace fluid_solver_detail
} // namespace

/**
 * @brief Validate and return a legacy mesh pointer.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Candidate mesh pointer.
 * @return Validated non-null mesh.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_mesh(SP<const legacy_mesh_type> mesh)
    -> SP<const legacy_mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "FluidSolver");
}

/**
 * @brief Validate and return a runtime mesh handle.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Candidate mesh handle.
 * @return Validated non-null handle.
 * @throws std::invalid_argument if the handle is null.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_mesh_handle(
    SP<const mesh_type> mesh) -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "FluidSolver");
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
    SP<const legacy_mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          SP<const mesh_type>(std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh)))),
          std::move(boundary_conditions),
          time_options,
          linear_options)
{
}

/** Construct on a retained mutable native runtime handle. */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          SP<const mesh_type>(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          true,
          mesh)
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
    SP<const mesh_type> mesh,
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

/** Construct a derived solver while retaining controlled mutable ownership. */
template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    DeferredMomentumEquationTag)
    : FluidSolver(
          SP<const mesh_type>(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          false,
          mesh)
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
    SP<const mesh_type> mesh,
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
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    bool register_momentum_equation,
    SP<mesh_type> mutable_mesh)
    : d_mesh(require_mesh_handle(std::move(mesh))),
      d_legacy_mesh(d_mesh->legacy_mesh()),
      d_problem(d_mesh,
                std::move(boundary_conditions),
                time_options,
                linear_options)
{
    retain_mutable_mesh_handle(std::move(mutable_mesh));
    if (d_legacy_mesh && d_mesh->has_reordered_cells())
    {
        throw std::invalid_argument(
            "FluidSolver legacy compatibility fields do not support a "
            "reordered MeshHandle; use the layout through SolidSubdomain "
            "or a native mesh backend.");
    }
    if (d_problem.time_options().time_step <= 0.0)
    {
        throw std::invalid_argument(
            "FluidSolver requires a positive time step.");
    }

    auto pressure_linear_options = d_problem.linear_options();
    pressure_linear_options.preconditioner =
        LinearPreconditioner::MueLu;
    // The pressure-Poisson matrix is cached by PressureProjectionEquation and
    // remains numerically unchanged until rebuild_matrix() replaces it.
    pressure_linear_options.reuse_preconditioner = true;
    d_problem.add_field(ScalarCellFieldDescriptor<Pack>("pressure"));
    d_problem.add_field(
        ScalarCellFieldDescriptor<Pack>("pressure_correction"));
    d_problem.add_field(VectorCellFieldDescriptor<Pack>("velocity"));
    d_problem.add_field(
        VectorCellFieldDescriptor<Pack>("predictor_pressure_gradient"));
    d_problem.add_field(
        VectorCellFieldDescriptor<Pack>("pressure_velocity_predictor"));
    d_problem.add_field(ScalarFaceFieldDescriptor<Pack>("old_face_flux"));
    d_problem.add_field(
        ScalarFaceFieldDescriptor<Pack>("projected_face_flux"));

    if (d_legacy_mesh)
    {
        d_problem.template emplace_object<FVM::VelocityBoundaryCache<Pack>>(
            "velocity_boundary_cache",
            FVM::cache_velocity_boundary_conditions<Pack>(
                d_legacy_mesh, d_problem.boundary_conditions()));
        d_problem.template emplace_object<legacy_face_flux_workspace_type>(
            "pressure_face_flux_workspace", d_legacy_mesh);
        if (register_momentum_equation)
        {
            d_problem.template emplace_object<
                IncompressibleMomentumEquation<Pack>>(
                    "momentum_equation", d_legacy_mesh);
        }
        d_problem.template emplace_object<PressureProjectionEquation<Pack>>(
            "pressure_projection",
            d_legacy_mesh,
            pressure_linear_options,
            d_problem.boundary_conditions().pressure,
            d_problem.time_options().pressure_gradient_scheme);
        d_problem.template emplace_object<
            CoupledPressureVelocitySolver<Pack>>(
                "coupled_pressure_velocity_solver", d_legacy_mesh);
        d_problem.template emplace_object<legacy_field_type>(
            "legacy_pressure", d_legacy_mesh, "pressure");
        d_problem.template emplace_object<legacy_field_type>(
            "legacy_pressure_correction", d_legacy_mesh,
            "pressure_correction");
        d_problem.template emplace_object<legacy_velocity_field_type>(
            "legacy_velocity", d_legacy_mesh, "velocity");
        d_problem.template emplace_object<legacy_velocity_field_type>(
            "legacy_predictor_pressure_gradient", d_legacy_mesh,
            "predictor_pressure_gradient");
        d_problem.template emplace_object<legacy_velocity_field_type>(
            "legacy_pressure_velocity_predictor", d_legacy_mesh,
            "pressure_velocity_predictor");
        d_problem.template emplace_object<legacy_face_flux_field_type>(
            "legacy_old_face_flux", d_legacy_mesh, "old_face_flux");
        d_problem.template emplace_object<legacy_face_flux_field_type>(
            "legacy_projected_face_flux", d_legacy_mesh,
            "projected_face_flux");
    }
    else
    {
        d_problem.template emplace_object<velocity_boundary_cache_type>(
            "velocity_boundary_cache",
            FVM::cache_velocity_boundary_conditions<Pack>(
                d_mesh, d_problem.boundary_conditions()));
        d_problem.template emplace_object<native_face_flux_workspace_type>(
            "pressure_face_flux_workspace", d_mesh);
        if (register_momentum_equation)
        {
            d_problem.template emplace_object<momentum_equation_type>("momentum_equation", d_mesh);
        }
        d_problem.template emplace_object<pressure_projection_type>("pressure_projection", d_mesh,
            pressure_linear_options, d_problem.boundary_conditions().pressure,
            d_problem.time_options().pressure_gradient_scheme);
        d_problem.template emplace_object<native_coupled_solver_type>("coupled_pressure_velocity_solver", d_mesh);
    }
    d_problem.template emplace_object<residual_type>("pressure_velocity_residuals");
}

/** Retain a controlled mutable view of the exact native runtime handle. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::retain_mutable_mesh_handle(SP<mesh_type> mesh)
{
    d_mutable_mesh.reset();
    if (!mesh)
    {
        return;
    }

    const SP<const mesh_type> mutable_view = mesh;
    const bool same_owner =
        !mutable_view.owner_before(d_mesh)
        && !d_mesh.owner_before(mutable_view);
    if (mesh.get() != d_mesh.get() || !same_owner)
    {
        throw std::invalid_argument(
            "FluidSolver mutable and const mesh views must share the same "
            "runtime handle.");
    }

    // A mutable shared_ptr to the handle object alone is not a geometry-
    // mutation capability. Preserve historical const/legacy behavior unless
    // the handle retains an actual mutable native concrete mesh.
    if (!d_legacy_mesh && mesh->has_mutable_geometry())
    {
        d_mutable_mesh = std::move(mesh);
    }
}

/** Refresh momentum plus the common pressure/flux/output geometry state. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::refresh_geometry_dependent_state()
{
    if (uses_legacy_backend())
    {
        momentum_equation().refresh_geometry();
    }
    else
    {
        native_momentum_equation().refresh_geometry();
    }
    refresh_pressure_velocity_geometry_state();
}

/** Refresh base pressure coupling and invalidate every numeric reuse owner. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::refresh_pressure_velocity_geometry_state()
{
    if (uses_legacy_backend())
    {
        velocity_boundary_cache() = FVM::cache_velocity_boundary_conditions<Pack>(
            d_legacy_mesh, d_problem.boundary_conditions());
        pressure_face_flux_workspace().refresh_geometry();
        pressure_projection().refresh_geometry();
        coupled_pressure_velocity_solver().clear_cache();
    }
    else
    {
        native_velocity_boundary_cache() = FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions());
        native_pressure_face_flux_workspace().refresh_geometry();
        native_pressure_projection().refresh_geometry();
        native_coupled_pressure_velocity_solver().clear_cache();
    }
    d_vtu_topology.reset();
    d_vtu_geometry_epoch = d_mesh->geometry_epoch();
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
 * @brief Return the independently configurable pressure solver policy.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_linear_solver_options() const noexcept
    -> const LinearSolverOptions&
{
    if (uses_legacy_backend())
    {
        return d_problem.template object<
            PressureProjectionEquation<Pack>>(
                "pressure_projection").linear_solver_options();
    }
    return d_problem.template object<pressure_projection_type>(
        "pressure_projection").linear_solver_options();
}

/**
 * @brief Replace the pressure solver policy without changing transport
 *        equation choices.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::set_pressure_linear_solver_options(
    LinearSolverOptions options)
{
    if (uses_legacy_backend())
    {
        pressure_projection().set_linear_solver_options(
            std::move(options));
    }
    else
    {
        native_pressure_projection().set_linear_solver_options(
            std::move(options));
    }
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

/** @brief Return the mapped-mesh momentum equation. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::native_momentum_equation()
    -> momentum_equation_type&
{
    return d_problem.template object<momentum_equation_type>(
        "momentum_equation");
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

/** @brief Return the mapped-mesh pressure projection equation. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::native_pressure_projection()
    -> pressure_projection_type&
{
    return d_problem.template object<pressure_projection_type>(
        "pressure_projection");
}

/**
 * @brief Return the monolithic pressure-velocity solver.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned coupled solver.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::coupled_pressure_velocity_solver() -> CoupledPressureVelocitySolver<Pack>&
{
    return d_problem.template object<CoupledPressureVelocitySolver<Pack>>("coupled_pressure_velocity_solver");
}

/** @brief Return the mapped-mesh monolithic pressure-velocity solver. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::native_coupled_pressure_velocity_solver() -> native_coupled_solver_type&
{
    return d_problem.template object<native_coupled_solver_type>("coupled_pressure_velocity_solver");
}

/**
 * @brief Return cached velocity boundary values and metadata.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned velocity boundary cache.
 */
template<TpetraTypePack Pack> auto FluidSolver<Pack>::velocity_boundary_cache() -> FVM::VelocityBoundaryCache<Pack>&
{
    return d_problem.template object<
        FVM::VelocityBoundaryCache<Pack>>("velocity_boundary_cache");
}

/** @brief Return cached boundary data for stored velocity fields. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::native_velocity_boundary_cache()
    -> velocity_boundary_cache_type&
{
    return d_problem.template object<velocity_boundary_cache_type>(
        "velocity_boundary_cache");
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

/** @brief Return the mapped-mesh Rhie-Chow workspace. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::native_pressure_face_flux_workspace()
    -> native_face_flux_workspace_type&
{
    return d_problem.template object<native_face_flux_workspace_type>(
        "pressure_face_flux_workspace");
}

/** @brief Return the legacy Rhie-Chow workspace. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_pressure_face_flux_workspace()
    -> legacy_face_flux_workspace_type&
{
    return d_problem.template object<legacy_face_flux_workspace_type>(
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

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_pressure() -> legacy_field_type&
{
    return d_problem.template object<legacy_field_type>(
        "legacy_pressure");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_pressure_correction()
    -> legacy_field_type&
{
    return d_problem.template object<legacy_field_type>(
        "legacy_pressure_correction");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_velocity()
    -> legacy_velocity_field_type&
{
    return d_problem.template object<legacy_velocity_field_type>(
        "legacy_velocity");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_predictor_pressure_gradient()
    -> legacy_velocity_field_type&
{
    return d_problem.template object<legacy_velocity_field_type>(
        "legacy_predictor_pressure_gradient");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_predictor_velocity()
    -> legacy_velocity_field_type&
{
    return d_problem.template object<legacy_velocity_field_type>(
        "legacy_pressure_velocity_predictor");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_old_face_fluxes()
    -> legacy_face_flux_field_type&
{
    return d_problem.template object<legacy_face_flux_field_type>(
        "legacy_old_face_flux");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::legacy_projected_face_fluxes()
    -> legacy_face_flux_field_type&
{
    return d_problem.template object<legacy_face_flux_field_type>(
        "legacy_projected_face_flux");
}

/** @brief Publish runtime-handle primary fields to the legacy backend. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::sync_primary_fields_to_legacy()
{
    if (!uses_legacy_backend())
    {
        return;
    }
    legacy_pressure().owned_data().update(
        scalar_type{1}, pressure().owned_data(), scalar_type{0});
    legacy_pressure_correction().owned_data().update(
        scalar_type{1}, pressure_correction().owned_data(),
        scalar_type{0});
    legacy_velocity().owned_data().update(
        scalar_type{1}, velocity().owned_data(), scalar_type{0});
    legacy_old_face_fluxes().data().update(
        scalar_type{1}, old_face_fluxes().data(), scalar_type{0});
    legacy_projected_face_fluxes().data().update(
        scalar_type{1}, projected_face_fluxes().data(), scalar_type{0});
    d_legacy_mesh->sync_periodic_boundaries(legacy_pressure());
    d_legacy_mesh->sync_periodic_boundaries(
        legacy_pressure_correction());
    d_legacy_mesh->sync_periodic_boundaries(legacy_velocity());
}

/** @brief Publish accepted legacy state back to runtime-handle fields. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::sync_primary_fields_from_legacy()
{
    if (!uses_legacy_backend())
    {
        return;
    }
    pressure().owned_data().update(
        scalar_type{1}, legacy_pressure().owned_data(), scalar_type{0});
    pressure_correction().owned_data().update(
        scalar_type{1}, legacy_pressure_correction().owned_data(),
        scalar_type{0});
    velocity().owned_data().update(
        scalar_type{1}, legacy_velocity().owned_data(), scalar_type{0});
    old_face_fluxes().data().update(
        scalar_type{1}, legacy_old_face_fluxes().data(), scalar_type{0});
    projected_face_fluxes().data().update(
        scalar_type{1}, legacy_projected_face_fluxes().data(),
        scalar_type{0});
    pressure().sync_ghosts();
    pressure_correction().sync_ghosts();
    velocity().sync_ghosts();
    old_face_fluxes().sync_ghosts();
    projected_face_fluxes().sync_ghosts();
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
    return fluid_solver_detail::volume_weighted_velocity_update_norm(
        *d_mesh,
        before,
        after,
        [this](scalar_type local_value)
        { return global_sum(local_value); });
}

/** @brief Compute the legacy backend's volume-weighted velocity update. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity_update_norm(
    const legacy_velocity_field_type& before,
    const legacy_velocity_field_type& after) const -> scalar_type
{
    return fluid_solver_detail::volume_weighted_velocity_update_norm(
        *d_legacy_mesh,
        before,
        after,
        [this](scalar_type local_value)
        { return global_sum(local_value); });
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
    const auto inverse_reference_density =
        scalar_type{1} / pressure_reference_density();

    if (uses_legacy_backend())
    {
        sync_primary_fields_to_legacy();
        FVM::cell_gradient(
            legacy_pressure(),
            d_problem.boundary_conditions().pressure,
            legacy_predictor_pressure_gradient(),
            legacy_pressure_face_flux_workspace().gradient_cache(),
            d_problem.time_options().pressure_gradient_scheme);
        const auto pressure_gradient_values =
            legacy_predictor_pressure_gradient().owned_read_view();
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

        const auto result = momentum_equation().advance_velocity(
            legacy_velocity(),
            legacy_old_face_fluxes(),
            velocity_boundary_cache(),
            d_problem.time_options(),
            legacy_velocity(),
            pressure_source,
            d_problem.linear_options());
        sync_primary_fields_from_legacy();
        return result;
    }

    FVM::cell_gradient(
        pressure(),
        d_problem.boundary_conditions().pressure,
        predictor_pressure_gradient(),
        native_pressure_face_flux_workspace().gradient_cache(),
        d_problem.time_options().pressure_gradient_scheme);
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

    return native_momentum_equation().advance_velocity(
        velocity(),
        old_face_fluxes(),
        native_velocity_boundary_cache(),
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

    if (uses_legacy_backend())
    {
        sync_primary_fields_to_legacy();
        FVM::pressure_weighted_face_fluxes(
            legacy_velocity(), legacy_pressure(),
            d_problem.time_options().time_step
                / pressure_reference_density(),
            velocity_boundary_cache(),
            d_problem.boundary_conditions().pressure,
            legacy_pressure_face_flux_workspace(),
            legacy_old_face_fluxes(),
            d_problem.time_options().pressure_gradient_scheme);
        // Keep the public handle-backed state authoritative between virtual
        // momentum hooks and outer pressure-correction iterations.
        sync_primary_fields_from_legacy();
        const auto linear_summary = advance_momentum();
        // The virtual hook's public contract is the handle-backed velocity.
        // Refresh legacy mirrors for pressure projection without overwriting
        // updates made by a derived solver through velocity().
        sync_primary_fields_to_legacy();
        pressure_velocity_residuals().momentum =
            velocity_update_norm(predictor_velocity(), velocity());
        return linear_summary;
    }

    auto& projection = native_pressure_projection();
    const auto predictor_boundary_cache =
        projection.pressure_flux_boundary_cache(native_velocity_boundary_cache());
    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(),
        d_problem.time_options().time_step
            / pressure_reference_density(),
        predictor_boundary_cache,
        d_problem.boundary_conditions().pressure,
        native_pressure_face_flux_workspace(),
        old_face_fluxes(),
        d_problem.time_options().pressure_gradient_scheme);
    projection.apply_fixed_boundary_fluxes(old_face_fluxes());
    const auto linear_summary = advance_momentum();
    // A derived native hook may update only authoritative owned storage.
    // Pressure projection immediately consumes overlap values, so normalize
    // the public-field contract here instead of requiring every override to
    // remember an implementation-specific ghost exchange.
    d_mesh->sync_periodic_boundaries(velocity());
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
    if (uses_legacy_backend())
    {
        sync_primary_fields_to_legacy();
        auto& projection = pressure_projection();
        const auto result =
            reuse_cached_predictor_flux
          ? projection.project_reusing_cached_predictor(
                legacy_pressure(),
                legacy_pressure_correction(),
                d_problem.time_options().time_step,
                pressure_reference_density(),
                velocity_boundary_cache(),
                legacy_velocity())
          : projection.project(
                legacy_pressure(),
                legacy_pressure_correction(),
                d_problem.time_options().time_step,
                pressure_reference_density(),
                velocity_boundary_cache(),
                legacy_velocity());
        // Downstream transport must use the same conservative face-flux field
        // whose balance produced result.continuity.
        legacy_projected_face_fluxes().data().update(
            scalar_type{1},
            projection.corrected_face_fluxes().data(),
            scalar_type{0});
        sync_primary_fields_from_legacy();
        return result;
    }

    auto& projection = native_pressure_projection();
    const auto* target = volume_continuity_target();
    const auto native_result = target != nullptr
        ? (reuse_cached_predictor_flux
                  ? projection.project_reusing_cached_predictor(pressure(), pressure_correction(),
                        d_problem.time_options().time_step, pressure_reference_density(),
                        native_velocity_boundary_cache(), velocity(), *target)
                  : projection.project(pressure(), pressure_correction(), d_problem.time_options().time_step,
                        pressure_reference_density(), native_velocity_boundary_cache(), velocity(), *target))
        : (reuse_cached_predictor_flux
                  ? projection.project_reusing_cached_predictor(pressure(), pressure_correction(),
                        d_problem.time_options().time_step, pressure_reference_density(),
                        native_velocity_boundary_cache(), velocity())
                  : projection.project(pressure(), pressure_correction(), d_problem.time_options().time_step,
                        pressure_reference_density(), native_velocity_boundary_cache(), velocity()));
    projected_face_fluxes().data().update(
        scalar_type{1},
        projection.corrected_face_fluxes().data(),
        scalar_type{0});
    projected_face_fluxes().sync_ghosts();
    d_last_volume_continuity_residuals = native_result.continuity_residuals;
    return {
        native_result.pressure_correction,
        native_result.continuity,
        native_result.linear_solve,
        native_result.continuity_residuals};
}

/**
 * @brief Assemble the base monolithic velocity-pressure system.
 *
 * @tparam Pack Tpetra type pack.
 * @return Coupled system assembled from current fields and options.
 */
template<TpetraTypePack Pack> auto FluidSolver<Pack>::assemble_coupled_system() -> coupled_system_type
{
    if (!uses_legacy_backend())
    {
        if (const auto* target = volume_continuity_target())
        {
            return native_coupled_pressure_velocity_solver().assemble(native_momentum_equation(), velocity(),
                pressure(), old_face_fluxes(), native_velocity_boundary_cache(), d_problem.boundary_conditions(),
                d_problem.time_options(), *target, pressure_reference_density());
        }
        return native_coupled_pressure_velocity_solver().assemble(native_momentum_equation(), velocity(), pressure(),
            old_face_fluxes(), native_velocity_boundary_cache(), d_problem.boundary_conditions(),
            d_problem.time_options(), pressure_reference_density());
    }
    sync_primary_fields_to_legacy();
    return coupled_pressure_velocity_solver().assemble(momentum_equation(), legacy_velocity(), legacy_pressure(),
        legacy_old_face_fluxes(), velocity_boundary_cache(), d_problem.boundary_conditions(), d_problem.time_options(),
        pressure_reference_density());
}

/**
 * @brief Solve one monolithic velocity-pressure Krylov system.
 *
 * @tparam Pack Tpetra type pack.
 * @throws std::runtime_error if the coupled linear solve does not converge.
 */
template<TpetraTypePack Pack> void FluidSolver<Pack>::solve_coupled_krylov()
{
    if (!uses_legacy_backend())
    {
        predictor_velocity().owned_data().update(scalar_type{1}, velocity().owned_data(), scalar_type{0});
        predictor_velocity().sync_ghosts();

        auto& coupled_solver = native_coupled_pressure_velocity_solver();
        const auto predictor_boundary_cache =
            coupled_solver.pressure_flux_boundary_cache(native_velocity_boundary_cache());
        FVM::pressure_weighted_face_fluxes(velocity(), pressure(),
            d_problem.time_options().time_step / pressure_reference_density(), predictor_boundary_cache,
            d_problem.boundary_conditions().pressure, native_pressure_face_flux_workspace(), old_face_fluxes(),
            d_problem.time_options().pressure_gradient_scheme);
        coupled_solver.apply_fixed_boundary_fluxes(old_face_fluxes());
        const auto system = assemble_coupled_system();
        const auto result = coupled_solver.solve(system, velocity(), pressure(), d_problem.linear_options());
        if (!result.converged)
        {
            throw std::runtime_error("FluidSolver coupled Krylov solve did not converge.");
        }

        const auto momentum_norm = velocity_update_norm(predictor_velocity(), velocity());
        fluid_solver_detail::record_coupled_result(
            result, momentum_norm, pressure_velocity_residuals(), d_last_step_statistics);

        const auto final_boundary_cache = coupled_solver.pressure_flux_boundary_cache(native_velocity_boundary_cache());
        FVM::pressure_weighted_face_fluxes(velocity(), pressure(),
            d_problem.time_options().time_step / pressure_reference_density(), final_boundary_cache,
            d_problem.boundary_conditions().pressure, native_pressure_face_flux_workspace(), projected_face_fluxes(),
            d_problem.time_options().pressure_gradient_scheme);
        coupled_solver.apply_fixed_boundary_fluxes(projected_face_fluxes());

        // The coupled block-row residual is useful linear algebra evidence,
        // but the physical contract is defined from the reconstructed final
        // absolute face flux: R_c = sum_f(phi_abs,f) - Q_V,c.
        scalar_type local_norm_squared{};
        scalar_type local_scale_squared{};
        scalar_type local_maximum{};
        const auto flux_values = projected_face_fluxes().owned_read_view();
        const auto* target = volume_continuity_target();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto balance = FVM::cell_flux_balance<Pack>(*d_mesh, projected_face_fluxes(), flux_values, cell);
            const auto target_rate = target == nullptr ? scalar_type{} : target->integrated_rate(cell);
            const auto residual = balance - target_rate;
            local_norm_squared += residual * residual;
            local_scale_squared += std::max(balance * balance, target_rate * target_rate);
            local_maximum = std::max(local_maximum, std::abs(residual));
        }
        const std::array<scalar_type, 2> local_sums{local_norm_squared, local_scale_squared};
        std::array<scalar_type, 2> global_sums{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local_sums.size()), local_sums.data(), global_sums.data());
        scalar_type global_maximum{};
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &global_maximum);
        using std::sqrt;
        const auto l2 = sqrt(global_sums[0]);
        const auto normalization = sqrt(global_sums[1]);
        d_last_volume_continuity_residuals = {
            l2, global_maximum, normalization > scalar_type{} ? l2 / normalization : scalar_type{}, normalization};
        pressure_velocity_residuals().continuity = l2;
        return;
    }
    sync_primary_fields_to_legacy();
    {
        const auto velocity_values = legacy_velocity().owned_read_view();
        auto predictor_values = legacy_predictor_velocity().owned_write_view();
        for (size_t owned = 0; owned < d_legacy_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            predictor_values(cell_lid, 0) = velocity_values(cell_lid, 0);
            predictor_values(cell_lid, 1) = velocity_values(cell_lid, 1);
            predictor_values(cell_lid, 2) = velocity_values(cell_lid, 2);
        }
    }
    d_legacy_mesh->sync_periodic_boundaries(legacy_predictor_velocity());

    FVM::pressure_weighted_face_fluxes(legacy_velocity(), legacy_pressure(),
        d_problem.time_options().time_step / pressure_reference_density(), velocity_boundary_cache(),
        d_problem.boundary_conditions().pressure, legacy_pressure_face_flux_workspace(), legacy_old_face_fluxes(),
        d_problem.time_options().pressure_gradient_scheme);
    sync_primary_fields_from_legacy();
    const auto system = assemble_coupled_system();
    const auto result =
        coupled_pressure_velocity_solver().solve(
            system,
            legacy_velocity(),
            legacy_pressure(),
            d_problem.linear_options());
    if (!result.converged)
    {
        throw std::runtime_error(
            "FluidSolver coupled Krylov solve did not converge.");
    }

    const auto momentum_norm = velocity_update_norm(
        legacy_predictor_velocity(), legacy_velocity());
    fluid_solver_detail::record_coupled_result(
        result,
        momentum_norm,
        pressure_velocity_residuals(),
        d_last_step_statistics);

    FVM::pressure_weighted_face_fluxes(
        legacy_velocity(), legacy_pressure(),
        d_problem.time_options().time_step
            / pressure_reference_density(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions().pressure,
        legacy_pressure_face_flux_workspace(),
        legacy_projected_face_fluxes(),
        d_problem.time_options().pressure_gradient_scheme);
    const auto projected_flux_values =
        legacy_projected_face_fluxes().owned_read_view();
    pressure_velocity_residuals().continuity =
        fluid_solver_detail::global_cell_metric_norm(
            *d_legacy_mesh,
            [this, projected_flux_values](local_ordinal_type cell_lid)
            {
                return FVM::cell_flux_balance<Pack>(
                    *d_legacy_mesh,
                    legacy_projected_face_fluxes(),
                    projected_flux_values,
                    cell_lid);
            },
            [this](scalar_type local_value)
            { return global_sum(local_value); });
    sync_primary_fields_from_legacy();
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
    if (const auto* target = volume_continuity_target())
    {
        if (uses_legacy_backend())
        {
            throw std::invalid_argument(
                "A generalized volume-continuity target requires the native MeshHandle backend.");
        }
        target->validate(*d_mesh, "FluidSolver volume-continuity target");
    }
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

/** @brief Tighten an already predicted native state against the active target. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::refine_volume_continuity(int maximum_corrections,
    scalar_type maximum_residual) -> continuity_residual_type
{
    if (uses_legacy_backend() || volume_continuity_target() == nullptr)
    {
        throw std::logic_error(
            "Strict volume-continuity refinement requires a native active target.");
    }
    if (maximum_corrections < 0 || !std::isfinite(maximum_residual) || maximum_residual < scalar_type{})
    {
        throw std::invalid_argument(
            "Volume-continuity refinement controls must be finite and non-negative.");
    }
    auto residuals = d_last_volume_continuity_residuals;
    bool reuse_predictor = d_problem.time_options().pressure_velocity_coupling !=
                           PressureVelocityCoupling::CoupledKrylov;
    for (int correction = 0;
         correction < maximum_corrections && residuals.maximum > maximum_residual;
         ++correction)
    {
        const auto result = run_pressure_correction(reuse_predictor);
        reuse_predictor = true;
        d_last_step_statistics.add(result.linear_solve);
        pressure_velocity_residuals().pressure = result.pressure_correction;
        pressure_velocity_residuals().continuity = result.continuity;
        residuals = result.continuity_residuals;
    }
    d_last_volume_continuity_residuals = residuals;
    return residuals;
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
    d_last_volume_continuity_residuals = {};
    if (uses_legacy_backend())
    {
        sync_primary_fields_to_legacy();
        if (d_step_index == 0)
        {
            d_legacy_mesh->sync_periodic_boundaries(
                legacy_velocity());
            sync_primary_fields_from_legacy();
        }
    }
    else if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(velocity());
    }
}

/** @brief Retain one validated target for every corrector in a coupling solve. */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::set_volume_continuity_target(const continuity_target_type& target)
{
    if (uses_legacy_backend())
    {
        throw std::invalid_argument(
            "A generalized volume-continuity target requires the native MeshHandle backend.");
    }
    target.validate(*d_mesh, "FluidSolver volume-continuity target");
    d_volume_continuity_target = target;
}

/** @brief Clear the opt-in target without changing fixed-grid state. */
template<TpetraTypePack Pack> void FluidSolver<Pack>::clear_volume_continuity_target() noexcept
{
    d_volume_continuity_target.reset();
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
    if (uses_legacy_backend())
    {
        d_legacy_mesh->sync_periodic_boundaries(
            legacy_velocity());
        sync_primary_fields_from_legacy();
    }
    else
    {
        d_mesh->sync_periodic_boundaries(velocity());
    }
    d_time += d_problem.time_options().time_step;
    ++d_step_index;
}

/**
 * @brief Replace the time step used by subsequent solver steps.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Finite positive physical or pseudo-time step.
 */
template<TpetraTypePack Pack>
void FluidSolver<Pack>::set_time_step(
    scalar_type time_step)
{
    if (!std::isfinite(time_step)
        || time_step <= scalar_type{})
    {
        throw std::invalid_argument(
            "FluidSolver requires a finite positive time step.");
    }
    d_problem.time_options().time_step =
        static_cast<real_t>(time_step);
}

/**
 * @brief Compute the communicator-wide maximum accepted cell Courant number.
 *
 * @tparam Pack Tpetra type pack.
 * @return Maximum cell Courant number for the current time step and projected
 *         face flux.
 */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::maximum_courant_number() const
    -> scalar_type
{
    const auto& flux =
        d_problem.template object<face_flux_field_type>(
            "projected_face_flux");
    scalar_type local_maximum{};
    const auto current_time_step = time_step();
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        scalar_type absolute_flux_sum{};
        for (const auto face_lid : d_mesh->faces(cell_lid))
        {
            absolute_flux_sum +=
                std::abs(flux.local_value(face_lid));
        }
        local_maximum = std::max(
            local_maximum,
            scalar_type{0.5} * current_time_step
                * absolute_flux_sum
                / static_cast<scalar_type>(d_mesh->cell_volume(
                    cell_lid)));
    }

    scalar_type global_maximum{};
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(),
        Teuchos::REDUCE_MAX,
        1,
        &local_maximum,
        &global_maximum);
    return global_maximum;
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
    return fluid_solver_detail::collect_owned_scalar_values(
        *d_mesh, field);
}

/** @brief Gather a legacy backend scalar field into VTU cell-data order. */
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::collect_scalar_field(
    const legacy_field_type& field) const -> VTUWriter::ScalarData
{
    if (!d_legacy_mesh)
    {
        throw std::invalid_argument(
            "FluidSolver cannot collect a legacy field without a legacy "
            "mesh backend.");
    }
    return fluid_solver_detail::collect_owned_scalar_values(
        *d_legacy_mesh, field);
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
    const auto geometry_epoch = d_mesh->geometry_epoch();
    if (!d_vtu_topology || d_vtu_geometry_epoch != geometry_epoch)
    {
        d_vtu_topology = d_mesh->vtu_topology();
        d_vtu_geometry_epoch = geometry_epoch;
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
