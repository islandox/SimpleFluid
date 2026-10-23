/**
 * @file BoussinesqSolver.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for BoussinesqSolver.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "BoussinesqSolver.hh"
#include "solvers/CoupledNonlinearProblem.hh"
#include "solvers/FieldStateTransaction.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace SimpleFluid
{


/**
 * @brief Construct a Boussinesq solver with mesh, boundary conditions, and solver options.
 *
 * Initialises all field and equation objects.  Validates that the time step is positive.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh.
 * @param boundary_conditions Boundary condition set for velocity and temperature.
 * @param time_options Time-stepping parameters.
 * @param linear_options Linear solver parameters.
 * @throws std::invalid_argument If the time step is non-positive.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<const legacy_mesh_type> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options)
    : BoussinesqSolver(SP<const MeshHandle<Pack>>(
          std::make_shared<MeshHandle<Pack>>(require_mesh(std::move(mesh)))),
          std::move(boundary_conditions), time_options, linear_options,
          BoussinesqModelOptions::legacy_defaults(time_options), false, PhysicalModelTag{})
{
}

/**
 * @brief Construct a legacy-mesh solver with explicit physical model options.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh.
 * @param boundary_conditions Velocity and temperature boundary conditions.
 * @param time_options Time-stepping and transport options.
 * @param linear_options Linear solver options.
 * @param model_options Physical material model options.
 * @throws std::invalid_argument if the mesh or model options are invalid.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<const legacy_mesh_type> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options)
    : BoussinesqSolver(SP<const MeshHandle<Pack>>(
          std::make_shared<MeshHandle<Pack>>(require_mesh(std::move(mesh)))),
          std::move(boundary_conditions), time_options, linear_options, std::move(model_options), true,
          PhysicalModelTag{})
{
}

/** Construct a legacy-property solver while retaining mutable native ownership. */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options)
    : BoussinesqSolver(std::move(mesh), std::move(boundary_conditions), time_options, linear_options,
          BoussinesqModelOptions::legacy_defaults(time_options), false, PhysicalModelTag{})
{
}

/**
 * @brief Construct a mesh-handle solver with legacy transport properties.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Type-erased computational mesh handle.
 * @param boundary_conditions Velocity and temperature boundary conditions.
 * @param time_options Time-stepping and transport options.
 * @param linear_options Linear solver options.
 * @throws std::invalid_argument if the mesh or time-step options are invalid.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options)
    : BoussinesqSolver(std::move(mesh), std::move(boundary_conditions), time_options, linear_options,
          BoussinesqModelOptions::legacy_defaults(time_options), false, PhysicalModelTag{})
{
}

/** Construct physical Boussinesq transport while retaining mutable native ownership. */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options)
    : BoussinesqSolver(std::move(mesh), std::move(boundary_conditions), time_options, linear_options,
          std::move(model_options), true, PhysicalModelTag{})
{
}

/**
 * @brief Construct a mesh-handle solver with explicit physical model options.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Type-erased computational mesh handle.
 * @param boundary_conditions Velocity and temperature boundary conditions.
 * @param time_options Time-stepping and transport options.
 * @param linear_options Linear solver options.
 * @param model_options Physical material model options.
 * @throws std::invalid_argument if the mesh or model options are invalid.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options)
    : BoussinesqSolver(std::move(mesh), std::move(boundary_conditions), time_options, linear_options,
          std::move(model_options), true, PhysicalModelTag{})
{
}

/** Retain the mutable view after common const-observer construction. */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options,
    bool physical_model_enabled, PhysicalModelTag tag)
    : BoussinesqSolver(SP<const MeshHandle<Pack>>(mesh), std::move(boundary_conditions), time_options, linear_options,
          std::move(model_options), physical_model_enabled, tag)
{
    this->retain_mutable_mesh_handle(std::move(mesh));
}

/**
 * @brief Initialize the common Boussinesq fields, equations, and model registry.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Type-erased computational mesh handle.
 * @param boundary_conditions Velocity and temperature boundary conditions.
 * @param time_options Time-stepping and transport options.
 * @param linear_options Linear solver options.
 * @param model_options Physical material model options.
 * @param physical_model_enabled Whether dimensional physical transport is active.
 * @param tag Constructor-dispatch tag.
 * @throws std::invalid_argument if model options or configured sources are invalid.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options,
    bool physical_model_enabled, PhysicalModelTag)
    : base_type(std::move(mesh), std::move(boundary_conditions), time_options, linear_options,
          typename base_type::DeferredMomentumEquationTag{}),
      d_model_options(std::move(model_options)), d_physical_model_enabled(physical_model_enabled)
{
    detail::validate_model_options(d_model_options, d_problem.time_options());

    d_problem.add_field(ScalarCellFieldDescriptor<Pack>("temperature"));

    d_problem.template emplace_object<temperature_equation_type>(
        "temperature_equation", d_mesh, d_problem.boundary_conditions());
    d_problem.template emplace_object<boussinesq_momentum_equation_type>("boussinesq_momentum_equation", d_mesh);
    d_problem.template emplace_object<material_type>(
        "material_properties", d_mesh, d_model_options, d_problem.time_options());
    d_problem.template emplace_object<turbulence_model_type>(
        "turbulence_model", d_mesh, d_problem.boundary_conditions());
    auto& sources = d_problem.template emplace_object<temperature_source_registry_type>("temperature_sources", d_mesh);
    if (uses_legacy_backend())
    {
        d_problem.template emplace_object<canonical_velocity_boundary_cache_type>("boussinesq_velocity_boundary_cache",
            FVM::cache_velocity_boundary_conditions<Pack>(d_mesh, d_problem.boundary_conditions()));
        d_problem.template emplace_object<canonical_face_flux_workspace_type>(
            "boussinesq_pressure_face_flux_workspace", d_mesh);
        d_problem.template emplace_object<canonical_coupled_solver_type>(
            "boussinesq_coupled_pressure_velocity_solver", d_mesh);
        d_problem.template emplace_object<BoussinesqMomentumEquation<Pack>>("momentum_equation", d_legacy_mesh);
        d_problem.template emplace_object<legacy_field_type>("legacy_temperature", d_legacy_mesh, "temperature");
    }
    for (size_t index = 0; index < d_model_options.temperature_source_names.size(); ++index)
    {
        sources.add(
            d_model_options.temperature_source_names[index], d_model_options.temperature_source_power_densities[index]);
    }
}

/**
 * @brief Return the immutable temperature field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored cell-centered temperature field.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::temperature() const noexcept -> const field_type&
{
    return d_problem.template object<field_type>("temperature");
}

/**
 * @brief Return the mutable temperature field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored cell-centered temperature field.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::temperature() noexcept -> field_type&
{
    return d_problem.template object<field_type>("temperature");
}

/** @brief Return the legacy equation stack's temperature mirror. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::legacy_temperature() -> legacy_field_type&
{
    return d_problem.template object<legacy_field_type>("legacy_temperature");
}

/** @brief Return the immutable legacy temperature mirror. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::legacy_temperature() const -> const legacy_field_type&
{
    return d_problem.template object<legacy_field_type>("legacy_temperature");
}

/** @brief Publish the public stored temperature to the legacy equation field. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::sync_temperature_to_legacy()
{
    if (!uses_legacy_backend())
    {
        return;
    }
    legacy_temperature().owned_data().update(scalar_type{1}, temperature().owned_data(), scalar_type{0});
    d_legacy_mesh->sync_periodic_boundaries(legacy_temperature());
}

/**
 * @brief Return the internally stored mutable material-property fields.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::stored_material_properties() -> material_type&
{
    return d_problem.template object<material_type>("material_properties");
}

/**
 * @brief Return the internally stored immutable material-property fields.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::stored_material_properties() const -> const material_type&
{
    return d_problem.template object<material_type>("material_properties");
}

/**
 * @brief Enable physical transport and return mutable material properties.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::material_properties() -> material_type&
{
    d_physical_model_enabled = true;
    return stored_material_properties();
}

/**
 * @brief Return the immutable material-property fields.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::material_properties() const -> const material_type&
{
    return stored_material_properties();
}

/**
 * @brief Return the internally stored mutable turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned turbulence model.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::stored_turbulence_model() -> turbulence_model_type&
{
    return d_problem.template object<turbulence_model_type>("turbulence_model");
}

/**
 * @brief Return the internally stored immutable turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned turbulence model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_turbulence_model() const -> const turbulence_model_type&
{
    return d_problem.template object<turbulence_model_type>("turbulence_model");
}

/**
 * @brief Report whether dimensional or turbulent transport is active.
 *
 * @tparam Pack Tpetra type pack.
 * @return true when a physical transport path must be used.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::physical_transport_enabled() const noexcept
{
    return d_physical_model_enabled || stored_turbulence_model().enabled();
}

/**
 * @brief Configure the Problem-owned turbulence model from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Turbulence model configuration.
 * @return Configured turbulence model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_turbulence(const TurbulenceModelOptions& options) -> turbulence_model_type&
{
    auto& model = stored_turbulence_model();
    model.configure(options, stored_material_properties(), d_model_options.reference_density);
    return model;
}

/**
 * @brief Configure the Problem-owned turbulence model from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing turbulence options.
 * @return Configured turbulence model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_turbulence(const Database& database) -> turbulence_model_type&
{
    auto& model = stored_turbulence_model();
    model.configure(database, stored_material_properties(), d_model_options.reference_density);
    return model;
}

/**
 * @brief Disable the active turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if an enabled model was disabled.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_turbulence_model() noexcept
{
    return stored_turbulence_model().disable();
}

/**
 * @brief Find the mutable active turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active model, or nullptr in laminar mode.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::find_turbulence_model() noexcept -> turbulence_model_type*
{
    auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

/**
 * @brief Find the immutable active turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active model, or nullptr in laminar mode.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_turbulence_model() const noexcept -> const turbulence_model_type*
{
    const auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

/**
 * @brief Return the internally stored mutable temperature-source registry.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_temperature_sources() -> temperature_source_registry_type&
{
    return d_problem.template object<temperature_source_registry_type>("temperature_sources");
}

/**
 * @brief Return the internally stored immutable temperature-source registry.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_temperature_sources() const -> const temperature_source_registry_type&
{
    return d_problem.template object<temperature_source_registry_type>("temperature_sources");
}

/**
 * @brief Enable physical transport and return mutable temperature sources.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::temperature_sources() -> temperature_source_registry_type&
{
    d_physical_model_enabled = true;
    return stored_temperature_sources();
}

/**
 * @brief Return the immutable temperature-source registry.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_sources() const -> const temperature_source_registry_type&
{
    return stored_temperature_sources();
}

/**
 * @brief Add a named volumetric temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Unique source name.
 * @param initial_power_density Initial source power density.
 * @return Newly registered source.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::add_temperature_source(std::string name, scalar_type initial_power_density)
    -> volumetric_source_type&
{
    d_physical_model_enabled = true;
    return stored_temperature_sources().add(std::move(name), initial_power_density);
}

/**
 * @brief Remove a named volumetric temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return true if a source was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_temperature_source(const std::string& name)
{
    return stored_temperature_sources().remove(name);
}

/**
 * @brief Find a mutable named temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return Source pointer, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_temperature_source(const std::string& name) noexcept -> volumetric_source_type*
{
    return stored_temperature_sources().find(name);
}

/**
 * @brief Find an immutable named temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return Source pointer, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_temperature_source(const std::string& name) const noexcept
    -> const volumetric_source_type*
{
    return stored_temperature_sources().find(name);
}

/**
 * @brief Install a per-step material-property updater.
 *
 * @tparam Pack Tpetra type pack.
 * @param updater Callable that updates material fields from solver context.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::set_material_updater(typename material_type::updater_type updater)
{
    d_physical_model_enabled = true;
    stored_material_properties().set_updater(std::move(updater));
}

/**
 * @brief Remove the custom material-property updater.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::clear_material_updater() noexcept
{
    stored_material_properties().clear_updater();
}

/**
 * @brief Return the Problem-owned temperature equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored temperature diffusion equation.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::temperature_equation() -> temperature_equation_type&
{
    return d_problem.template object<temperature_equation_type>("temperature_equation");
}

/** Refresh laminar Boussinesq transport and shared pressure/output geometry. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::refresh_geometry_dependent_state()
{
    const auto refresh_timing = this->d_solver_timings.scope(SolverPhase::GeometryRefresh);
    if (!d_shared_transport_geometry)
        d_shared_transport_geometry.emplace(*d_mesh);
    else
        d_shared_transport_geometry->refresh();
    const auto& geometry = d_shared_transport_geometry->shared_geometry();
    boussinesq_momentum_equation().refresh_geometry(geometry);
    temperature_equation().refresh_geometry(geometry);
    if (uses_legacy_backend())
    {
        boussinesq_pressure_face_flux_workspace().refresh_geometry();
        boussinesq_coupled_pressure_velocity_solver().clear_cache();
    }
    this->refresh_pressure_velocity_geometry_state();
}

/** @brief Return the canonical handle-backed Boussinesq momentum equation. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::boussinesq_momentum_equation() -> boussinesq_momentum_equation_type&
{
    return d_problem.template object<boussinesq_momentum_equation_type>("boussinesq_momentum_equation");
}

/** @brief Return the canonical velocity-boundary cache on either backend. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::boussinesq_velocity_boundary_cache() -> canonical_velocity_boundary_cache_type&
{
    if (!uses_legacy_backend())
    {
        return native_velocity_boundary_cache();
    }
    return d_problem.template object<canonical_velocity_boundary_cache_type>("boussinesq_velocity_boundary_cache");
}

/** @brief Return the canonical pressure-flux workspace on either backend. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::boussinesq_pressure_face_flux_workspace() -> canonical_face_flux_workspace_type&
{
    if (!uses_legacy_backend())
    {
        return native_pressure_face_flux_workspace();
    }
    return d_problem.template object<canonical_face_flux_workspace_type>("boussinesq_pressure_face_flux_workspace");
}

/** @brief Return the canonical coupled assembler on either backend. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::boussinesq_coupled_pressure_velocity_solver() -> canonical_coupled_solver_type&
{
    if (!uses_legacy_backend())
    {
        return native_coupled_pressure_velocity_solver();
    }
    return d_problem.template object<canonical_coupled_solver_type>("boussinesq_coupled_pressure_velocity_solver");
}

/**
 * @brief Return the Problem-owned Boussinesq momentum equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored momentum equation.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::momentum_equation() -> BoussinesqMomentumEquation<Pack>&
{
    return d_problem.template object<BoussinesqMomentumEquation<Pack>>("momentum_equation");
}

/**
 * @brief Advance momentum with buoyancy and optional physical transport.
 *
 * @tparam Pack Tpetra type pack.
 * @return Aggregated linear solve summary.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::advance_momentum() -> LinearSolveSummary
{
    const auto* momentum_flux = &momentum_transport_flux();
    const auto* ale = d_flow_transport.ale;
    FVM::cell_gradient(pressure(), d_problem.boundary_conditions().pressure, predictor_pressure_gradient(),
        boussinesq_pressure_face_flux_workspace().gradient_cache(), d_problem.time_options().pressure_gradient_scheme);
    const auto inverse_reference_density = scalar_type{1} / pressure_reference_density();
    const auto* turbulence = find_turbulence_model();
    const auto pressure_gradient_values = predictor_pressure_gradient().owned_read_view();
    using gradient_view_type = decltype(predictor_pressure_gradient().owned_read_view());
    std::optional<gradient_view_type> turbulent_gradient_values;
    if (turbulence != nullptr)
    {
        turbulent_gradient_values.emplace(turbulence->turbulent_kinetic_energy_gradient().owned_read_view());
    }
    auto pressure_source = [&](local_ordinal_type cell_lid) -> vec_type
    {
        vec_type acceleration{pressure_gradient_values(cell_lid, 0) * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 1) * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 2) * (-inverse_reference_density)};
        if (turbulent_gradient_values)
        {
            constexpr scalar_type turbulent_pressure_factor{-2.0 / 3.0};
            acceleration.x += (*turbulent_gradient_values)(cell_lid, 0) * turbulent_pressure_factor;
            acceleration.y += (*turbulent_gradient_values)(cell_lid, 1) * turbulent_pressure_factor;
            acceleration.z += (*turbulent_gradient_values)(cell_lid, 2) * turbulent_pressure_factor;
        }
        return acceleration;
    };

    if (physical_transport_enabled())
    {
        return boussinesq_momentum_equation().advance_velocity_physical(velocity(), *momentum_flux, temperature(),
            boussinesq_velocity_boundary_cache(), d_problem.time_options(), stored_material_properties(),
            d_model_options.reference_density, d_model_options.density_feedback_enabled, velocity(), pressure_source,
            this->momentum_linear_solver_options(),
            turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
            turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr, ale);
    }
    return boussinesq_momentum_equation().advance_velocity(velocity(), *momentum_flux, temperature(),
        boussinesq_velocity_boundary_cache(), d_problem.time_options(), velocity(), pressure_source,
        this->momentum_linear_solver_options(), ale);
}

/**
 * @brief Return the density used to normalize pressure operators.
 *
 * @tparam Pack Tpetra type pack.
 * @return Positive reference density.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::pressure_reference_density() const noexcept -> scalar_type
{
    return d_model_options.reference_density;
}

template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::supports_coupled_nonlinear() const noexcept
{
    return typeid(*this) == typeid(BoussinesqSolver<Pack>) && d_physical_model_enabled;
}

template<TpetraTypePack Pack>
std::unique_ptr<CoupledNonlinearProblem> BoussinesqSolver<Pack>::make_coupled_nonlinear_problem()
{
    if constexpr (std::same_as<Pack, DefaultTpetraTypes>)
    {
        const auto* turbulence = find_turbulence_model();
        const CoupledNonlinearProblem::FrozenBoussinesqInput physical{temperature(), stored_material_properties(),
            d_model_options.density_feedback_enabled, turbulence ? &turbulence->effective_dynamic_viscosity() : nullptr,
            turbulence ? &turbulence->turbulent_kinetic_energy_gradient() : nullptr,
            turbulence ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr};
        return std::make_unique<CoupledNonlinearProblem>(d_mesh, velocity(), pressure(),
            d_problem.boundary_conditions(), d_problem.time_options(), d_problem.time_options().nonlinear,
            pressure_reference_density(), volume_continuity_target(), &physical, this->coupled_nonlinear_workspace());
    }
    throw std::invalid_argument("coupledNonlinear requires the default Tpetra pack.");
}

/**
 * @brief Assemble the coupled Boussinesq velocity-pressure system.
 *
 * @tparam Pack Tpetra type pack.
 * @return Monolithic coupled system with optional material and turbulence terms.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::assemble_coupled_system() -> coupled_system_type
{
    const auto* turbulence = find_turbulence_model();
    if (const auto* ale = d_flow_transport.ale)
    {
        const auto& momentum_flux = momentum_transport_flux();
        if (const auto* target = volume_continuity_target())
        {
            return boussinesq_coupled_pressure_velocity_solver().assemble(boussinesq_momentum_equation(), velocity(),
                pressure(), temperature(), momentum_flux, boussinesq_velocity_boundary_cache(),
                d_problem.boundary_conditions(), d_problem.time_options(), *target,
                physical_transport_enabled() ? &stored_material_properties() : nullptr,
                d_model_options.reference_density, d_model_options.density_feedback_enabled,
                turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
                turbulence != nullptr ? &turbulence->turbulent_kinetic_energy_gradient() : nullptr,
                turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr, ale);
        }
        throw std::logic_error("planarALE coupled assembly requires an active volume-continuity target.");
    }
    const continuity_target_type zero_target(d_mesh);
    const auto* target = volume_continuity_target();
    return boussinesq_coupled_pressure_velocity_solver().assemble(boussinesq_momentum_equation(), velocity(),
        pressure(), temperature(), old_face_fluxes(), boussinesq_velocity_boundary_cache(),
        d_problem.boundary_conditions(), d_problem.time_options(), target != nullptr ? *target : zero_target,
        physical_transport_enabled() ? &stored_material_properties() : nullptr, d_model_options.reference_density,
        d_model_options.density_feedback_enabled,
        turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
        turbulence != nullptr ? &turbulence->turbulent_kinetic_energy_gradient() : nullptr,
        turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr);
}

/**
 * @brief Initialise the temperature field as a linear ramp along a given direction.
 *
 * Sets temperature to hot_at_min at the minimum projection onto direction and
 * cold_at_max at the maximum.  Also zeroes velocity and sets a uniform initial
 * pressure.
 *
 * @tparam Pack Tpetra type pack.
 * @param direction Unit direction vector for the temperature gradient.
 * @param hot_at_min Temperature at the minimum projection point.
 * @param cold_at_max Temperature at the maximum projection point.
 * @param initial_pressure Uniform initial gauge pressure in Pa.
 * @throws std::invalid_argument If direction has zero norm.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_linear_temperature(
    const vec_type& direction, scalar_type hot_at_min, scalar_type cold_at_max, scalar_type initial_pressure)
{
    if (direction.norm() <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a nonzero initialization direction.");
    }
    auto local_min_projected = std::numeric_limits<scalar_type>::max();
    auto local_max_projected = std::numeric_limits<scalar_type>::lowest();
    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = static_cast<scalar_type>(d_mesh->cell_centroid(cell_lid).dot(direction));
        local_min_projected = std::min(local_min_projected, projected);
        local_max_projected = std::max(local_max_projected, projected);
    }

    auto min_projected = local_min_projected;
    auto max_projected = local_max_projected;
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_min_projected, &min_projected);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_max_projected, &max_projected);

    if (min_projected <= max_projected)
    {
        const auto width = max_projected > min_projected ? max_projected - min_projected : scalar_type{1};
        for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(cell);
            const auto projected = static_cast<scalar_type>(d_mesh->cell_centroid(cell_lid).dot(direction));
            const auto blend = (projected - min_projected) / width;
            temperature().set_owned_value(cell_lid, hot_at_min * (scalar_type{1} - blend) + cold_at_max * blend);
            pressure().set_owned_value(cell_lid, initial_pressure);
            velocity().set_owned_value(cell_lid, {});
        }
    }

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(pressure());
    d_mesh->sync_periodic_boundaries(velocity());
    d_primary_fields_initialized = true;
    if (uses_legacy_backend())
    {
        sync_primary_fields_to_legacy();
        sync_temperature_to_legacy();
    }
}

/**
 * @brief Initialise fields for a heated-box problem with temperature gradient along X.
 *
 * Delegates to initialize_linear_temperature with direction (1, 0, 0).
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the hot (xmin) boundary.
 * @param cold_temperature Temperature at the cold (xmax) boundary.
 * @param initial_pressure Uniform initial gauge pressure in Pa.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_heated_box(
    scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure)
{
    initialize_linear_temperature({1.0, 0.0, 0.0}, hot_temperature, cold_temperature, initial_pressure);
}

/**
 * @brief Initialise fields for a bottom-hot, top-cold problem with temperature gradient along Z.
 *
 * Delegates to initialize_linear_temperature with direction (0, 0, 1).
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the bottom (zmin) boundary.
 * @param cold_temperature Temperature at the top (zmax) boundary.
 * @param initial_pressure Uniform initial gauge pressure in Pa.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_bottom_hot_top_cold(
    scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure)
{
    initialize_linear_temperature({0.0, 0.0, 1.0}, hot_temperature, cold_temperature, initial_pressure);
}

/**
 * @brief Advance the active turbulence equations after flow coupling.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size captured after flow coupling.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::advance_turbulence(scalar_type time_step)
{
    const auto phase_timing = this->d_solver_timings.scope(SolverPhase::Turbulence);
    if (auto* turbulence = find_turbulence_model())
    {
        const auto gravity = d_problem.time_options().gravity_vector();
        const turbulence_buoyancy_context_type buoyancy_context{&temperature(),
            &d_problem.boundary_conditions().temperature,
            {static_cast<scalar_type>(gravity.x), static_cast<scalar_type>(gravity.y),
                static_cast<scalar_type>(gravity.z)},
            static_cast<scalar_type>(d_problem.time_options().thermal_expansion),
            d_model_options.density_feedback_enabled};
        const auto turbulence_statistics =
            turbulence->advance(velocity(), projected_face_fluxes(), boussinesq_velocity_boundary_cache(), time_step,
                stored_material_properties(), d_model_options.reference_density,
                d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options(), &buoyancy_context);
        d_last_step_statistics.add(turbulence_statistics);
    }
}

/**
 * @brief Advance temperature, record its solve, and synchronize its ghosts.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::advance_temperature_transport(
    scalar_type time_step, const TemperatureTransportInputs& transport)
{
    const auto phase_timing = this->d_solver_timings.scope(SolverPhase::Temperature);
    LinearSolveStatistics temperature_statistics;
    if (physical_transport_enabled())
    {
        const auto* turbulence = find_turbulence_model();
        auto total_power_density = [&](local_ordinal_type cell_lid)
        {
            auto total = stored_temperature_sources().total_power_density(cell_lid);
            if (transport.power_sink)
            {
                total += -transport.power_sink->value(cell_lid) +
                    (transport.power_release ? transport.power_release->value(cell_lid) : scalar_type{});
            }
            else if (transport.power_release)
            {
                total += transport.power_release->value(cell_lid);
            }
            return total;
        };
        const auto& transport_flux = (transport.flux ? *transport.flux : projected_face_fluxes());
        temperature_statistics = temperature_equation().advance_physical(temperature(), transport_flux, time_step,
            stored_material_properties(), temperature(), total_power_density,
            d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options(),
            turbulence != nullptr ? &turbulence->effective_thermal_conductivity() : nullptr,
            turbulence != nullptr ? turbulence->effective_thermal_conductivity_boundary_cache() : nullptr,
            d_problem.time_options().coefficient_interpolation, transport.ale, transport.liquid_mass_inventory,
            transport.old_heat_capacity, transport.temperature_density);
    }
    else
    {
        temperature_statistics = temperature_equation().advance_semi_implicit(temperature(), projected_face_fluxes(),
            time_step, d_problem.time_options().thermal_diffusivity, temperature(),
            d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options());
    }
    d_last_step_statistics.add(temperature_statistics);
    d_last_step_statistics.temperature = temperature_statistics.achieved_tolerance;

    temperature().sync_ghosts();
}

/** @brief Advance standalone flow, heat, and turbulence without optional-model orchestration. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::step()
{
    collective_detail::collective_local_validation(*d_mesh, "Standalone Boussinesq advancement",
        [&]
        {
            if (!d_external_step_controller.expired() || d_step_failed)
                throw std::logic_error(
                    "Advance the external multiphysics controller while it is bound to this solver.");
        });
    const auto timing = this->d_solver_timings.scope(SolverPhase::Step);
    this->validate_pressure_velocity_selection();
    validate_step_state();
    AcceptedStateRollback rollback;
    auto* turbulence = find_turbulence_model();
    const bool sas =
        turbulence && turbulence->type() == TurbulenceModelType::SSTKOmegaSAS && turbulence->options().sas.enabled;
    if (turbulence)
        turbulence->validate_time_mode(d_problem.time_options().physical_time);
    if (sas)
    {
        rollback.capture(temperature());
        rollback.capture(pressure());
        rollback.capture(this->pressure_correction());
        rollback.capture(velocity());
        rollback.capture(predictor_pressure_gradient());
        rollback.capture(this->predictor_velocity());
        rollback.capture(old_face_fluxes());
        rollback.capture(projected_face_fluxes());
        for (const auto& [name, source] : stored_temperature_sources().entries())
            rollback.capture(source->field());
        rollback.capture_model(stored_material_properties());
        rollback.capture_model(*turbulence);
        rollback.add(
            [this, residuals = pressure_velocity_residuals(), volume = this->d_last_volume_continuity_residuals,
                nonlinear = this->d_last_nonlinear_result, time = d_time, index = d_step_index,
                statistics = d_last_step_statistics]
            {
                pressure_velocity_residuals() = residuals;
                this->d_last_volume_continuity_residuals = volume;
                this->d_last_nonlinear_result = nonlinear;
                d_time = time;
                d_step_index = index;
                d_last_step_statistics = statistics;
            });
    }
    const bool nonlinear =
        d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::CoupledNonlinear;
    std::optional<typename material_type::StateSnapshot> material_snapshot;
    if (nonlinear && !sas)
        material_snapshot.emplace(stored_material_properties().snapshot());
    if (!nonlinear)
        begin_step();
    bool flow_accepted = false;
    try
    {
        if (d_step_index == 0)
            temperature().sync_ghosts();
        if (d_physical_model_enabled)
            refresh_physical_models();
        if (turbulence)
            turbulence->refresh_effective_properties(stored_material_properties(), d_model_options.reference_density);
        solve_pressure_velocity_coupling();
        flow_accepted = true;
        const auto dt = d_problem.time_options().time_step;
        advance_turbulence(dt);
        advance_temperature_transport(dt, {});
        if (turbulence)
            turbulence->refresh_effective_properties(stored_material_properties(), d_model_options.reference_density);
        if (uses_legacy_backend())
            sync_temperature_to_legacy();
        finish_step();
    }
    catch (...)
    {
        if (sas)
        {
            rollback.restore();
            if (uses_legacy_backend())
            {
                sync_primary_fields_to_legacy();
                sync_temperature_to_legacy();
            }
        }
        else if (nonlinear && !flow_accepted)
        {
            stored_material_properties().restore(*material_snapshot);
            if (turbulence)
                turbulence->refresh_effective_properties(
                    stored_material_properties(), d_model_options.reference_density);
        }
        if (nonlinear && flow_accepted && !sas)
            d_step_failed = true;
        throw;
    }
}

/**
 * @brief Write the current solution fields to a VTU file.
 *
 * Converts the internal mesh and cell fields (temperature, pressure, velocity)
 * into VTU format and writes to the specified path.
 *
 * @tparam Pack Tpetra type pack.
 * @param filename Output .vtu file path.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::write_solution_vtu(const std::string& filename) const
{
    write_solution_vtu(filename, {});
}

/**
 * @brief Write selected primary and optional multiphysics fields to VTU.
 *
 * @tparam Pack Tpetra type pack.
 * @param filename Output VTU path.
 * @param output_options Field-selection controls for optional model data.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::solution_writer(const SolutionOutputOptions& output_options,
    const std::map<std::string, const field_type*>* extra_material_fields) const -> VTUWriter
{
    auto writer = fluid_solution_writer();
    writer.add_scalar_cell_data("temperature", collect_scalar_field(temperature()));

    if (output_options.include_material_properties)
    {
        const auto& material = stored_material_properties();
        writer.add_scalar_cell_data("density", collect_scalar_field(material.density));
        writer.add_scalar_cell_data("specific_heat_capacity", collect_scalar_field(material.specific_heat_capacity));
        writer.add_scalar_cell_data("dynamic_viscosity", collect_scalar_field(material.dynamic_viscosity));
        writer.add_scalar_cell_data("thermal_conductivity", collect_scalar_field(material.thermal_conductivity));
        if (extra_material_fields)
            for (const auto& [name, field] : *extra_material_fields)
                writer.add_scalar_cell_data(name, collect_scalar_field(*field));
    }
    if (output_options.include_turbulence_fields)
    {
        if (const auto* turbulence = find_turbulence_model())
        {
            for (const auto& [name, field] : turbulence->output_fields())
            {
                writer.add_scalar_cell_data(name, collect_scalar_field(*field));
            }
        }
    }
    if (output_options.include_sources)
        for (const auto& [name, source] : stored_temperature_sources().entries())
            writer.add_scalar_cell_data(name, collect_scalar_field(source->field()));
    return writer;
}

/** @brief Write selected solution fields to one VTU piece. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_solution_vtu(
    const std::string& filename, const SolutionOutputOptions& output_options) const
{
    solution_writer(output_options).write(filename, VTUWriter::Encoding::AppendedBinary);
}

/**
 * @brief Write collision-free rank pieces and a rank-zero PVTU index.
 *
 * Every rank writes only its owned cells. Collective status reductions ensure
 * rank zero publishes the index only after every piece is complete and make
 * piece or index failures rank coherent.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_parallel_solution_vtu(
    const std::string& filename, const SolutionOutputOptions& output_options) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count > 1)
    {
        const auto option_mask = static_cast<int>(output_options.include_sources) |
                                 (static_cast<int>(output_options.include_material_properties) << 1) |
                                 (static_cast<int>(output_options.include_reacting_gas_fields) << 2) |
                                 (static_cast<int>(output_options.include_turbulence_fields) << 4) |
                                 (static_cast<int>(output_options.include_free_surface_fields) << 5);
        std::array<long long, 2> root_arguments{rank == 0 ? static_cast<long long>(filename.size()) : 0LL,
            rank == 0 ? static_cast<long long>(option_mask) : 0LL};
        Teuchos::broadcast(*communicator, 0, static_cast<int>(root_arguments.size()), root_arguments.data());
        if (root_arguments[0] < 0 || root_arguments[0] > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument("Parallel VTU filename is too long for MPI broadcast.");
        }

        std::string root_filename(static_cast<size_t>(root_arguments[0]), '\0');
        if (rank == 0)
        {
            root_filename = filename;
        }
        if (!root_filename.empty())
        {
            Teuchos::broadcast(*communicator, 0, static_cast<int>(root_filename.size()), root_filename.data());
        }

        const int local_arguments_mismatch =
            (filename != root_filename || option_mask != static_cast<int>(root_arguments[1])) ? 1 : 0;
        int any_arguments_mismatch = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_arguments_mismatch, &any_arguments_mismatch);
        if (any_arguments_mismatch != 0)
        {
            throw std::invalid_argument("Parallel VTU filename and output options must agree on "
                                        "every rank.");
        }
    }
    const auto piece_filename = VTUWriter::rank_piece_filename(filename, rank, rank_count);
    if (rank_count <= 1)
    {
        solution_writer(output_options).write(piece_filename, VTUWriter::Encoding::AppendedBinary);
        return;
    }

    std::optional<VTUWriter> writer;
    std::string local_schema_key;
    std::exception_ptr preparation_error;
    try
    {
        writer.emplace(solution_writer(output_options));
        local_schema_key = writer->cell_data_schema_key();
    }
    catch (...)
    {
        preparation_error = std::current_exception();
    }
    const int local_preparation_failed = preparation_error ? 1 : 0;
    int any_preparation_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_preparation_failed, &any_preparation_failed);
    if (any_preparation_failed != 0)
    {
        if (preparation_error)
        {
            std::rethrow_exception(preparation_error);
        }
        throw std::runtime_error("VTU output preparation failed on another MPI rank.");
    }

    const int local_schema_size_is_valid =
        local_schema_key.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
    int global_schema_size_is_valid = 0;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1, &local_schema_size_is_valid, &global_schema_size_is_valid);
    if (global_schema_size_is_valid == 0)
    {
        throw std::invalid_argument("Parallel VTU CellData schema is too large for MPI "
                                    "broadcast.");
    }

    int root_schema_size = rank == 0 ? static_cast<int>(local_schema_key.size()) : 0;
    Teuchos::broadcast(*communicator, 0, 1, &root_schema_size);
    std::string root_schema_key(static_cast<size_t>(root_schema_size), '\0');
    if (rank == 0)
    {
        root_schema_key = local_schema_key;
    }
    if (!root_schema_key.empty())
    {
        Teuchos::broadcast(*communicator, 0, root_schema_size, root_schema_key.data());
    }
    const int local_schema_mismatch = local_schema_key == root_schema_key ? 0 : 1;
    int any_schema_mismatch = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_schema_mismatch, &any_schema_mismatch);
    if (any_schema_mismatch != 0)
    {
        throw std::invalid_argument("Parallel VTU CellData names, types, and component counts "
                                    "must agree on every rank.");
    }

    std::exception_ptr piece_error;
    try
    {
        writer->write(piece_filename, VTUWriter::Encoding::AppendedBinary);
    }
    catch (...)
    {
        piece_error = std::current_exception();
    }
    const int local_piece_failed = piece_error ? 1 : 0;
    int any_piece_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_piece_failed, &any_piece_failed);
    if (any_piece_failed != 0)
    {
        if (piece_error)
        {
            std::rethrow_exception(piece_error);
        }
        throw std::runtime_error("A VTU piece failed to write on another MPI rank.");
    }

    std::exception_ptr index_error;
    if (rank == 0)
    {
        try
        {
            std::vector<std::string> piece_filenames;
            piece_filenames.reserve(static_cast<size_t>(rank_count));
            for (int piece_rank = 0; piece_rank < rank_count; ++piece_rank)
            {
                piece_filenames.push_back(VTUWriter::rank_piece_filename(filename, piece_rank, rank_count));
            }
            writer->write_parallel_index(VTUWriter::parallel_index_filename(filename), piece_filenames);
        }
        catch (...)
        {
            index_error = std::current_exception();
        }
    }
    int local_index_failed = index_error ? 1 : 0;
    int any_index_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_index_failed, &any_index_failed);
    if (any_index_failed != 0)
    {
        if (index_error)
        {
            std::rethrow_exception(index_error);
        }
        throw std::runtime_error("The PVTU index failed to write on another MPI rank.");
    }
}

/** @brief Refresh dimensional material fields and registered heat sources. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::refresh_physical_models()
{
    const auto timing = this->d_solver_timings.scope(SolverPhase::PhysicalModels);
    update_context_type context{d_time, d_step_index, *d_mesh, temperature(), pressure(), velocity()};
    stored_material_properties().update(context);
    stored_temperature_sources().update(context);
}

/** @brief Bind borrowed numeric inputs for one pressure/velocity solve. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::solve_flow(const FlowTransportInputs& inputs)
{
    const auto previous = d_flow_transport;
    d_flow_transport = inputs;
    try
    {
        solve_pressure_velocity_coupling();
    }
    catch (...)
    {
        d_flow_transport = previous;
        throw;
    }
    d_flow_transport = previous;
}

/** @brief Derive momentum carrier flux from the stage's control-volume geometry. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::momentum_transport_flux() -> const face_flux_field_type&
{
    if (!d_flow_transport.ale)
        return old_face_fluxes();
    FVM::mesh_relative_face_fluxes(old_face_fluxes(), *d_flow_transport.ale, *d_flow_transport.relative_flux);
    return *d_flow_transport.relative_flux;
}

/** @brief Reject invalid or rank-divergent accepted time and timestep before advancement. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::validate_step_state() const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto current_time_step = static_cast<scalar_type>(d_problem.time_options().time_step);
    const int local_invalid_temporal_state =
        (!std::isfinite(d_time) || d_step_index < 0 || !std::isfinite(current_time_step) ||
            current_time_step <= scalar_type{})
            ? 1
            : 0;
    int any_invalid_temporal_state = 0;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_temporal_state, &any_invalid_temporal_state);
    if (any_invalid_temporal_state != 0)
    {
        throw std::invalid_argument(
            "Boussinesq accepted time and step index must be valid and its timestep finite and positive on every "
            "mesh rank.");
    }

    int minimum_step_index = 0;
    int maximum_step_index = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &d_step_index, &minimum_step_index);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &d_step_index, &maximum_step_index);
    const std::array<scalar_type, 2> local_temporal_state{d_time, current_time_step};
    std::array<scalar_type, 2> minimum_temporal_state{};
    std::array<scalar_type, 2> maximum_temporal_state{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_temporal_state.size()),
        local_temporal_state.data(), minimum_temporal_state.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_temporal_state.size()),
        local_temporal_state.data(), maximum_temporal_state.data());
    if (minimum_step_index != maximum_step_index || minimum_temporal_state != maximum_temporal_state)
    {
        throw std::invalid_argument(
            "Boussinesq accepted time, step index, and positive timestep must match exactly on every mesh rank.");
    }
}

} // namespace SimpleFluid
