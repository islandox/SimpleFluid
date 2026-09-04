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

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

namespace SimpleFluid
{
namespace
{

/** @brief Deterministically encode every free-surface option for MPI parity. */
[[nodiscard]] std::string encode_free_surface_options(const FreeSurfaceOptions& options)
{
    std::ostringstream encoded;
    encoded << std::hexfloat;
    auto scalar = [&encoded](const auto value) { encoded << value << ';'; };
    auto string = [&encoded](const std::string& value) { encoded << value.size() << ':' << value << ';'; };
    auto array = [&scalar](const ArrReal& values)
    {
        scalar(values.size());
        for (const auto value : values)
        {
            scalar(value);
        }
    };
    auto optional_scalar = [&scalar](const std::optional<real_t>& value)
    {
        scalar(value.has_value());
        if (value)
        {
            scalar(*value);
        }
    };

    scalar(options.enabled);
    scalar(static_cast<int>(options.mode));
    scalar(static_cast<int>(options.gravity_axis));
    scalar(options.validity_warning_relative_level_change);
    scalar(static_cast<int>(options.range_policy));
    optional_scalar(options.initial_liquid_volume);
    optional_scalar(options.initial_clear_level);

    scalar(static_cast<int>(options.vessel.mode));
    scalar(options.vessel.bottom_elevation);
    scalar(options.vessel.top_elevation);
    scalar(options.vessel.cross_section_area);
    scalar(options.vessel.total_internal_volume);
    array(options.vessel.height_table);
    array(options.vessel.volume_table);

    scalar(static_cast<int>(options.liquid_mass.mode));
    optional_scalar(options.liquid_mass.initial_liquid_mass);
    scalar(static_cast<int>(options.liquid_mass.depletion_policy));

    scalar(static_cast<int>(options.headspace.mode));
    scalar(static_cast<int>(options.headspace.temperature_mode));
    scalar(options.headspace.ambient_pressure);
    scalar(options.headspace.initial_pressure);
    scalar(options.headspace.initial_temperature);
    scalar(options.headspace.gas_constant);
    scalar(options.headspace.compressibility_factor);
    scalar(options.headspace.total_internal_volume);
    scalar(options.headspace.initial_moles.size());
    for (const auto& [species, moles] : options.headspace.initial_moles)
    {
        string(species);
        scalar(moles);
    }
    scalar(options.headspace.infer_initial_moles);
    array(options.headspace.prescribed_temperature_times);
    array(options.headspace.prescribed_temperature_values);

    scalar(options.coupling.maximum_correctors);
    scalar(options.coupling.absolute_tolerance);
    scalar(options.coupling.relative_tolerance);
    scalar(options.coupling.relaxation);
    scalar(options.coupling.minimum_absolute_pressure);
    scalar(options.coupling.maximum_absolute_pressure);
    scalar(options.coupling.volume_absolute_tolerance);
    scalar(options.coupling.volume_relative_tolerance);
    scalar(options.coupling.gas_absolute_tolerance);
    scalar(options.coupling.gas_relative_tolerance);

    string(options.ale.top_boundary);
    optional_scalar(options.ale.deformation_start_elevation);
    optional_scalar(options.ale.maximum_level_change);
    scalar(options.ale.gcl_absolute_tolerance);
    scalar(options.ale.gcl_relative_tolerance);
    scalar(options.ale.maximum_correctors);
    scalar(options.ale.level_absolute_tolerance);
    scalar(options.ale.level_relative_tolerance);
    scalar(options.ale.relaxation);
    optional_scalar(options.ale.quality_limits.maximum_growth_ratio);
    optional_scalar(options.ale.quality_limits.maximum_non_orthogonality_degrees);
    optional_scalar(options.ale.quality_limits.maximum_skewness);
    optional_scalar(options.ale.quality_limits.maximum_aspect_ratio);
    return encoded.str();
}

/** Deterministically encode every rank-global choice used by planar ALE. */
[[nodiscard]] std::string encode_planar_ale_runtime_options(
    const TimeStepperOptions& time, const LinearSolverOptions& linear,
    const LinearSolverOptions& pressure_linear,
    const BoussinesqModelOptions& model,
    const MaterialFeedbackOptions* feedback,
    const RadiolyticGasOptions* gas,
    const BoundaryConditionSet& boundaries,
    const std::vector<std::pair<std::string, bool>>& temperature_sources,
    bool mutable_mesh, bool legacy_backend, bool physical_transport,
    bool turbulence_enabled, bool boiling_enabled, bool precursor_enabled,
    bool scalar_void_explicit, bool material_updater,
    bool dynamic_temperature_sources)
{
    std::ostringstream encoded;
    encoded << std::hexfloat;
    const auto scalar = [&encoded](const auto value)
    {
        encoded << value << ';';
    };
    const auto string = [&encoded](const std::string& value)
    {
        encoded << value.size() << ':' << value << ';';
    };
    const auto real_array = [&scalar](const auto& values)
    {
        scalar(values.size());
        for (const auto value : values)
        {
            scalar(value);
        }
    };
    const auto string_array = [&scalar, &string](const auto& values)
    {
        scalar(values.size());
        for (const auto& value : values)
        {
            string(value);
        }
    };
    const auto optional_real = [&scalar](const std::optional<real_t>& value)
    {
        scalar(value.has_value());
        if (value)
        {
            scalar(*value);
        }
    };
    const auto linear_options = [&scalar](const LinearSolverOptions& value)
    {
        scalar(value.max_iterations);
        scalar(value.tolerance);
        scalar(value.verbosity);
        scalar(static_cast<int>(value.backend));
        scalar(static_cast<int>(value.preconditioner));
        scalar(value.reuse_preconditioner);
    };
    const auto scalar_boundaries = [&scalar, &string](const BoundaryConditionMap& values)
    {
        std::vector<std::string> names;
        names.reserve(values.size());
        for (const auto& [name, condition] : values)
        {
            (void)condition;
            names.push_back(name);
        }
        std::ranges::sort(names);
        scalar(names.size());
        for (const auto& name : names)
        {
            const auto& condition = values.at(name);
            string(name);
            scalar(static_cast<int>(condition.type));
            scalar(condition.value);
            scalar(condition.robin_coefficient);
        }
    };
    const auto vector_boundaries = [&scalar, &string](const VectorBoundaryConditionMap& values)
    {
        std::vector<std::string> names;
        names.reserve(values.size());
        for (const auto& [name, condition] : values)
        {
            (void)condition;
            names.push_back(name);
        }
        std::ranges::sort(names);
        scalar(names.size());
        for (const auto& name : names)
        {
            const auto& condition = values.at(name);
            string(name);
            scalar(static_cast<int>(condition.type));
            scalar(condition.value.x);
            scalar(condition.value.y);
            scalar(condition.value.z);
            scalar(condition.robin_coefficient);
        }
    };

    scalar(time.time_step);
    scalar(time.steps);
    scalar(time.thermal_diffusivity);
    scalar(time.kinematic_viscosity);
    scalar(time.thermal_expansion);
    scalar(time.gravity_x);
    scalar(time.gravity_y);
    scalar(time.gravity_z);
    scalar(time.reference_temperature);
    scalar(static_cast<int>(time.non_orthogonal_treatment));
    scalar(static_cast<int>(time.pressure_gradient_scheme));
    scalar(static_cast<int>(time.coefficient_interpolation));
    scalar(time.n_non_orthogonal_correctors);
    scalar(static_cast<int>(time.pressure_velocity_coupling));
    scalar(time.n_pressure_correctors);
    scalar(time.n_outer_correctors);
    linear_options(linear);
    linear_options(pressure_linear);

    scalar(model.reference_density);
    scalar(model.density);
    scalar(model.specific_heat_capacity);
    optional_real(model.dynamic_viscosity);
    optional_real(model.thermal_conductivity);
    scalar(model.density_feedback_enabled);
    string_array(model.temperature_source_names);
    real_array(model.temperature_source_power_densities);

    scalar(feedback != nullptr);
    if (feedback != nullptr)
    {
        scalar(static_cast<int>(feedback->density_mode));
        scalar(static_cast<int>(feedback->viscosity_mode));
        scalar(feedback->reference_density);
        scalar(feedback->liquid_density);
        scalar(feedback->gas_density);
        scalar(feedback->reference_temperature);
        scalar(feedback->thermal_expansion);
        scalar(feedback->reference_dynamic_viscosity);
        scalar(feedback->min_density);
        scalar(feedback->min_viscosity);
    }

    scalar(gas != nullptr);
    if (gas != nullptr)
    {
        scalar(static_cast<int>(gas->mode));
        scalar(static_cast<int>(gas->pressure_mode));
        scalar(static_cast<int>(gas->dissolved_transport));
        scalar(static_cast<int>(gas->bubble_transport));
        scalar(static_cast<int>(gas->heaviside_mode));
        scalar(static_cast<int>(gas->rise_velocity_mode));
        scalar(static_cast<int>(gas->surface_tension_mode));
        scalar(static_cast<int>(gas->diffusivity_mode));
        scalar(gas->hydrogen_yield_mol_per_j);
        scalar(gas->gas_release_efficiency);
        scalar(gas->reference_pressure);
        scalar(gas->gas_constant);
        scalar(gas->alpha_min);
        scalar(gas->alpha_max);
        scalar(gas->max_source_alpha_rate);
        scalar(gas->henry_coefficient);
        scalar(gas->surface_tension);
        scalar(gas->hydrogen_diffusivity);
        scalar(gas->atmospheric_pressure);
        scalar(gas->uranium_concentration_mol_per_m3);
        scalar(gas->hydrogen_yield_molecules_per_100_ev);
        scalar(gas->microbubble_lifetime);
        scalar(gas->large_bubble_dissolution_time);
        scalar(gas->micro_to_large_conversion_coefficient);
        scalar(gas->smooth_heaviside_width);
        scalar(gas->constant_slip_velocity);
        scalar(gas->bubble_gas_density);
        scalar(gas->bubble_gravity);
        scalar(gas->rise_velocity_tolerance);
        scalar(gas->max_rise_velocity_iterations);
        scalar(gas->initial_dissolved_hydrogen);
        scalar(gas->initial_micro_number_density);
        scalar(gas->initial_micro_moles);
        scalar(gas->initial_large_number_density);
        scalar(gas->initial_large_moles);
        scalar(gas->min_radius);
        scalar(gas->max_radius);
        scalar(gas->min_population);
        scalar(gas->max_population);
        scalar(gas->max_concentration);
        scalar(gas->local_ode_tolerance);
        scalar(gas->max_subcycles);
        scalar(gas->max_radius_iterations);
        scalar(gas->liquid_compressibility);
        scalar(gas->liquid_thermal_expansion);
        scalar(gas->minimum_absolute_pressure);
        real_array(gas->pressure_history_times);
        real_array(gas->pressure_history_values);
        string_array(gas->free_surface_patches);
    }

    scalar_boundaries(boundaries.temperature);
    vector_boundaries(boundaries.velocity);
    scalar_boundaries(boundaries.pressure);
    scalar_boundaries(boundaries.turbulence.turbulent_kinetic_energy);
    scalar_boundaries(boundaries.turbulence.dissipation_rate);
    scalar_boundaries(boundaries.turbulence.specific_dissipation_rate);
    scalar(temperature_sources.size());
    for (const auto& [name, enabled] : temperature_sources)
    {
        string(name);
        scalar(enabled);
    }
    scalar(mutable_mesh);
    scalar(legacy_backend);
    scalar(physical_transport);
    scalar(turbulence_enabled);
    scalar(boiling_enabled);
    scalar(precursor_enabled);
    scalar(scalar_void_explicit);
    scalar(material_updater);
    scalar(dynamic_temperature_sources);
    return encoded.str();
}

/** @brief Return one diagnostic species inventory, treating absence as zero. */
[[nodiscard]] real_t free_surface_species_moles(const GasMolesBySpecies& values, std::string_view species)
{
    const auto iterator = values.find(std::string(species));
    return iterator == values.end() ? real_t{} : iterator->second;
}

} // namespace

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
 * @brief Create the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Newly created fission source.
 * @throws std::invalid_argument if a fission source already exists.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::add_fission_power_source() -> fission_power_source_type&
{
    if (d_fission_power_source)
    {
        throw std::invalid_argument("BoussinesqSolver already has a fission power source.");
    }
    d_physical_model_enabled = true;
    d_fission_power_source = std::make_unique<fission_power_source_type>(d_mesh, stored_temperature_sources());
    return *d_fission_power_source;
}

/**
 * @brief Configure or disable the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Fission source profile and normalization options.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::configure_fission_power_source(const FissionPowerSourceOptions& options)
{
    detail::validate_fission_power_options(options);
    if (options.profile == FissionPowerProfile::Disabled)
    {
        remove_fission_power_source();
        return;
    }

    auto& source = d_fission_power_source ? *d_fission_power_source : add_fission_power_source();
    source.configure(options);
}

/**
 * @brief Remove the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a source was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_fission_power_source() noexcept
{
    if (!d_fission_power_source)
    {
        return false;
    }
    d_fission_power_source.reset();
    return true;
}

/**
 * @brief Find the mutable fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured source, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_fission_power_source() noexcept -> fission_power_source_type*
{
    return d_fission_power_source.get();
}

/**
 * @brief Find the immutable fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured source, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_fission_power_source() const noexcept -> const fission_power_source_type*
{
    return d_fission_power_source.get();
}

/**
 * @brief Configure the optional radiolytic-gas model from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Radiolysis mode, kinetics, and void-fraction bounds.
 * @return Configured radiolytic-gas model.
 * @throws std::invalid_argument if options or coupled-model combinations are invalid.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_radiolytic_gas(const RadiolyticGasOptions& options) -> radiolytic_gas_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "radiolytic gas, then configure free surface last.");
    }
    validate_radiolytic_gas_options(options);
    if (d_free_surface_model && options.mode == RadiolyticGasMode::IdealGasSource)
    {
        throw std::invalid_argument("The planar free-surface budget cannot use idealGasSource "
                                    "radiolysis because it does not own a conservative gas inventory.");
    }
    if (d_free_surface_model && options.mode == RadiolyticGasMode::Sheng2024TwoPopulation &&
        options.pressure_mode != RadiolyticPressureMode::Constant &&
        options.pressure_mode != RadiolyticPressureMode::Reconstructed)
    {
        throw std::invalid_argument("Planar free-surface coupling requires constant or reconstructed "
                                    "radiolytic absolute pressure.");
    }
    bool scalar_void_was_reset = false;
    if (options.mode == RadiolyticGasMode::Sheng2024TwoPopulation && d_boiling_source_model &&
        d_boiling_source_model->enabled())
    {
        throw std::invalid_argument("Sheng two-population radiolysis cannot be combined with "
                                    "boiling until vapor mass is coupled to bubble inventories.");
    }
    d_physical_model_enabled = true;
    if (options.mode != RadiolyticGasMode::Disabled &&
        (!d_scalar_void_fraction_model || (!d_scalar_void_fraction_explicitly_configured && d_step_index == 0)))
    {
        ScalarVoidFractionOptions void_options;
        void_options.alpha_min = options.alpha_min;
        void_options.alpha_max = options.alpha_max;
        void_options.initial_alpha = options.alpha_min;
        if (!d_scalar_void_fraction_model)
        {
            d_scalar_void_fraction_model = std::make_unique<scalar_void_fraction_model_type>(d_mesh, void_options);
        }
        else
        {
            d_scalar_void_fraction_model->configure(void_options);
        }
        scalar_void_was_reset = true;
    }
    else
    {
        ensure_scalar_void_fraction_model();
    }
    if (options.mode == RadiolyticGasMode::Sheng2024TwoPopulation)
    {
        const auto& void_options = d_scalar_void_fraction_model->options();
        if (void_options.alpha_min != options.alpha_min || void_options.alpha_max != options.alpha_max)
        {
            throw std::invalid_argument("Sheng radiolysis and its scalar mirror require identical "
                                        "void-fraction bounds.");
        }
        if (std::isfinite(void_options.alpha_collapse_time))
        {
            throw std::invalid_argument("Sheng radiolysis cannot use scalar void collapse until "
                                        "bubble inventory removal is coupled conservatively.");
        }
    }
    if (!d_radiolytic_gas_model)
    {
        d_radiolytic_gas_model = std::make_unique<radiolytic_gas_model_type>(d_mesh, options);
    }
    else
    {
        d_radiolytic_gas_model->configure(options);
    }
    if (options.mode == RadiolyticGasMode::Sheng2024TwoPopulation && d_primary_fields_initialized)
    {
        initialize_radiolytic_gas_state();
    }
    else if (scalar_void_was_reset && d_precursor_model && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(d_scalar_void_fraction_model->alpha_l());
    }
    return *d_radiolytic_gas_model;
}

/**
 * @brief Configure the optional radiolytic-gas model from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing radiolysis options.
 * @return Configured radiolytic-gas model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_radiolytic_gas(const Database& database) -> radiolytic_gas_model_type&
{
    return configure_radiolytic_gas(radiolytic_gas_options_from_database(database));
}

/**
 * @brief Remove the optional radiolytic-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_radiolytic_gas_model()
{
    if (!d_radiolytic_gas_model)
        return false;
    if (d_free_surface_model)
    {
        remove_free_surface_model();
    }
    d_radiolytic_gas_model.reset();
    return true;
}

/**
 * @brief Find the mutable radiolytic-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() noexcept -> radiolytic_gas_model_type*
{
    return d_radiolytic_gas_model.get();
}

/**
 * @brief Find the immutable radiolytic-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() const noexcept -> const radiolytic_gas_model_type*
{
    return d_radiolytic_gas_model.get();
}

/**
 * @brief Configure the optional boiling source from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Bulk and wall boiling configuration.
 * @return Configured boiling source model.
 * @throws std::invalid_argument if options conflict with active radiolysis.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_boiling_source(const BoilingSourceOptions& options) -> boiling_source_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "boiling, then configure free surface last.");
    }
    validate_boiling_source_options(options);
    if (d_free_surface_model && d_free_surface_options.headspace.mode == HeadspaceMode::Closed &&
        (options.enable_bulk_boiling || options.enable_wall_boiling))
    {
        throw std::invalid_argument(
            "Closed free-surface coupling cannot use the fixed-temperature boiling model; pressure-dependent "
            "saturation from absolute headspace pressure is not implemented.");
    }
    if ((options.enable_bulk_boiling || options.enable_wall_boiling) && d_radiolytic_gas_model &&
        d_radiolytic_gas_model->supplies_void_fraction())
    {
        throw std::invalid_argument("Boiling cannot be combined with Sheng two-population "
                                    "radiolysis until vapor mass is coupled to bubble inventories.");
    }
    d_physical_model_enabled = true;
    ensure_scalar_void_fraction_model();
    if (!d_boiling_source_model)
    {
        d_boiling_source_model = std::make_unique<boiling_source_model_type>(d_mesh, options);
    }
    else
    {
        d_boiling_source_model->configure(options);
    }
    return *d_boiling_source_model;
}

/**
 * @brief Configure the optional boiling source from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing boiling options.
 * @return Configured boiling source model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_boiling_source(const Database& database) -> boiling_source_model_type&
{
    return configure_boiling_source(boiling_source_options_from_database(database));
}

/**
 * @brief Remove the optional boiling source model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_boiling_source_model()
{
    if (!d_boiling_source_model)
        return false;
    if (d_free_surface_model)
    {
        remove_free_surface_model();
    }
    d_boiling_source_model.reset();
    return true;
}

/**
 * @brief Find the mutable boiling source model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_boiling_source_model() noexcept -> boiling_source_model_type*
{
    return d_boiling_source_model.get();
}

/**
 * @brief Find the immutable boiling source model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_boiling_source_model() const noexcept -> const boiling_source_model_type*
{
    return d_boiling_source_model.get();
}

/**
 * @brief Configure the authoritative scalar void-fraction model.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Void bounds, initial value, and collapse options.
 * @return Configured scalar void-fraction model.
 * @throws std::invalid_argument if options conflict with Sheng radiolysis.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_scalar_void_fraction(const ScalarVoidFractionOptions& options)
    -> scalar_void_fraction_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "scalar void, then configure free surface last.");
    }
    validate_scalar_void_fraction_options(options);
    if (d_free_surface_model && options.initial_alpha != scalar_type{} &&
        !(d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction()))
    {
        throw std::invalid_argument("The planar free-surface budget rejects nonzero scalar void "
                                    "without a conservative gas-inventory owner.");
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction() &&
        (options.alpha_min != d_radiolytic_gas_model->options().alpha_min ||
            options.alpha_max != d_radiolytic_gas_model->options().alpha_max))
    {
        throw std::invalid_argument("A Sheng radiolysis mirror requires its scalar void bounds.");
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction() &&
        std::isfinite(options.alpha_collapse_time))
    {
        throw std::invalid_argument("Sheng radiolysis cannot use scalar void collapse until "
                                    "bubble inventory removal is coupled conservatively.");
    }
    d_physical_model_enabled = true;
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model = std::make_unique<scalar_void_fraction_model_type>(d_mesh, options);
    }
    else
    {
        d_scalar_void_fraction_model->configure(options);
    }
    d_scalar_void_fraction_explicitly_configured = true;
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction() &&
        d_radiolytic_gas_model->initial_state_initialized())
    {
        d_scalar_void_fraction_model->initialize_from(d_radiolytic_gas_model->alpha_g());
    }
    if (d_precursor_model && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(d_scalar_void_fraction_model->alpha_l());
    }
    return *d_scalar_void_fraction_model;
}

/**
 * @brief Configure the scalar void-fraction model from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing void-fraction options.
 * @return Configured scalar void-fraction model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_scalar_void_fraction(const Database& database)
    -> scalar_void_fraction_model_type&
{
    return configure_scalar_void_fraction(scalar_void_fraction_options_from_database(database));
}

/**
 * @brief Find the mutable scalar void-fraction model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() noexcept -> scalar_void_fraction_model_type*
{
    return d_scalar_void_fraction_model.get();
}

/**
 * @brief Find the immutable scalar void-fraction model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() const noexcept -> const scalar_void_fraction_model_type*
{
    return d_scalar_void_fraction_model.get();
}

/**
 * @brief Configure material-property feedback from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Density and viscosity feedback configuration.
 * @return Configured feedback model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_material_feedback(const MaterialFeedbackOptions& options)
    -> material_feedback_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "material feedback, then configure free surface last.");
    }
    validate_material_feedback_options(options);
    d_physical_model_enabled = true;
    if (!d_material_feedback_model)
    {
        d_material_feedback_model = std::make_unique<material_feedback_model_type>(d_mesh, options);
    }
    else
    {
        d_material_feedback_model->configure(options);
    }
    d_model_options.density_feedback_enabled = d_material_feedback_model->density_feedback_enabled();
    return *d_material_feedback_model;
}

/**
 * @brief Configure material-property feedback from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing feedback options.
 * @return Configured feedback model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_material_feedback(const Database& database) -> material_feedback_model_type&
{
    return configure_material_feedback(
        material_feedback_options_from_database(database, d_model_options, d_problem.time_options()));
}

/**
 * @brief Remove the optional material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_material_feedback_model()
{
    if (!d_material_feedback_model)
        return false;
    if (d_free_surface_model)
    {
        remove_free_surface_model();
    }
    d_material_feedback_model.reset();
    d_model_options.density_feedback_enabled = false;
    return true;
}

/**
 * @brief Find the mutable material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_material_feedback_model() noexcept -> material_feedback_model_type*
{
    return d_material_feedback_model.get();
}

/**
 * @brief Find the immutable material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_material_feedback_model() const noexcept -> const material_feedback_model_type*
{
    return d_material_feedback_model.get();
}

/**
 * @brief Configure delayed-neutron precursor groups from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Precursor group and transport configuration.
 * @return Configured precursor model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_precursors(const DelayedNeutronPrecursorOptions& options)
    -> precursor_model_type&
{
    d_physical_model_enabled = true;
    if (d_primary_fields_initialized)
    {
        initialize_radiolytic_gas_state();
    }
    ensure_scalar_void_fraction_model();
    if (!d_precursor_model)
    {
        d_precursor_model = std::make_unique<precursor_model_type>(d_mesh, options);
    }
    else
    {
        d_precursor_model->configure(options);
    }
    d_precursor_model->initialize_inventory(*active_alpha_l_field());
    return *d_precursor_model;
}

/**
 * @brief Configure delayed-neutron precursor groups from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing precursor options.
 * @return Configured precursor model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_precursors(const Database& database) -> precursor_model_type&
{
    DelayedNeutronPrecursorOptions options;
    int local_parse_failed = 0;
    std::string local_error;
    try
    {
        options = delayed_neutron_precursor_options_from_database(database);
    }
    catch (const std::invalid_argument& error)
    {
        local_parse_failed = 1;
        local_error = error.what();
    }

    int any_parse_failed = 0;
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_parse_failed, &any_parse_failed);
    if (any_parse_failed != 0)
    {
        if (d_mesh->owned_cell_map()->getComm()->getSize() == 1 && !local_error.empty())
        {
            throw std::invalid_argument(local_error);
        }
        throw std::invalid_argument("Precursor database options must be valid on every rank.");
    }
    return configure_precursors(options);
}

/**
 * @brief Remove the optional precursor model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_precursor_model() noexcept
{
    if (!d_precursor_model)
        return false;
    d_precursor_model.reset();
    return true;
}

/**
 * @brief Find the mutable precursor model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::find_precursor_model() noexcept -> precursor_model_type*
{
    return d_precursor_model.get();
}

/**
 * @brief Find the immutable precursor model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_precursor_model() const noexcept -> const precursor_model_type*
{
    return d_precursor_model.get();
}

/** @brief Configure optional fixed-grid or solver-integrated planar free-surface coupling. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_free_surface(const FreeSurfaceOptions& options) -> free_surface_model_type*
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto local_encoding = encode_free_surface_options(options);
    long long root_size = rank == 0 ? static_cast<long long>(local_encoding.size()) : 0LL;
    Teuchos::broadcast(*communicator, 0, 1, &root_size);
    if (root_size < 0 || root_size > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("Free-surface option encoding is too large for MPI broadcast.");
    }
    std::string root_encoding(static_cast<size_t>(root_size), '\0');
    if (rank == 0)
    {
        root_encoding = local_encoding;
    }
    if (!root_encoding.empty())
    {
        Teuchos::broadcast(*communicator, 0, static_cast<int>(root_encoding.size()), root_encoding.data());
    }
    const int local_mismatch = local_encoding == root_encoding ? 0 : 1;
    int any_mismatch = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_mismatch, &any_mismatch);
    if (any_mismatch != 0)
    {
        throw std::invalid_argument("Free-surface options must match on every mesh rank.");
    }

    validate_collective_model_state();
    if (d_free_surface_step_failed)
    {
        throw std::logic_error(
            "A prior step failed after upstream multiphysics state advanced; reconstruct the solver from an accepted "
            "state before configuring or advancing the free surface.");
    }
    validate_free_surface_configuration(options);
    if (options.enabled && options.mode == FreeSurfaceMode::PlanarALE)
    {
        PlanarALEMeshMotionOptions motion_options;
        motion_options.axis = options.gravity_axis;
        motion_options.deformation_start_elevation =
            options.ale.deformation_start_elevation;
        motion_options.maximum_level_change = options.ale.maximum_level_change;
        motion_options.quality_limits = options.ale.quality_limits;
        motion_options.gcl_absolute_tolerance =
            options.ale.gcl_absolute_tolerance;
        motion_options.gcl_relative_tolerance =
            options.ale.gcl_relative_tolerance;
        const auto volume_map = make_vessel_volume_map(options);
        if (d_ale_motion)
        {
            if (!d_ale_boundary || d_ale_boundary->axis() != options.gravity_axis ||
                d_ale_boundary->name() != options.ale.top_boundary)
            {
                throw std::invalid_argument(
                    "Remove the active planarALE model before changing its motion axis or top boundary.");
            }
            ale_boundary_type geometry_preflight(d_mesh,
                options.ale.top_boundary, options.gravity_axis, *volume_map,
                options.coupling.volume_absolute_tolerance,
                options.coupling.volume_relative_tolerance);
        }
        else
        {
            // Constructor/destructor acquire and release the motion lease
            // without changing geometry; the boundary check proves current
            // patch planarity, area, and vessel/mesh-volume consistency.
            ale_motion_type motion_preflight(
                this->mutable_mesh_handle(), std::move(motion_options));
            ale_boundary_type geometry_preflight(d_mesh,
                options.ale.top_boundary, options.gravity_axis, *volume_map,
                options.coupling.volume_absolute_tolerance,
                options.coupling.volume_relative_tolerance);
        }
    }
    if (!options.enabled)
    {
        remove_free_surface_model();
        d_free_surface_options = options;
        return nullptr;
    }
    if (!physical_transport_enabled())
    {
        throw std::invalid_argument(
            "The planar free-surface budget requires the dimensional physical-model Boussinesq constructor or an "
            "already configured physical model.");
    }

    auto model = make_planar_free_surface_model(options);
    auto inventory = std::make_unique<liquid_mass_inventory_type>(d_mesh, options.liquid_mass);
    auto clear_level = std::make_unique<field_type>(d_mesh, "clearLevel");
    auto pool_level = std::make_unique<field_type>(d_mesh, "poolLevel");
    auto headspace_pressure = std::make_unique<field_type>(d_mesh, "headspacePressure");
    auto pool_occupancy = std::make_unique<field_type>(d_mesh, "poolOccupancy");

    remove_free_surface_model();
    d_free_surface_options = options;
    d_free_surface_model = std::move(model);
    d_liquid_mass_inventory = std::move(inventory);
    d_clear_level = std::move(clear_level);
    d_pool_level = std::move(pool_level);
    d_headspace_pressure = std::move(headspace_pressure);
    d_pool_occupancy = std::move(pool_occupancy);
    d_pool_occupancy_volume_error = {};

    try
    {
        initialize_free_surface_if_needed();
        initialize_planar_ale_if_needed();
    }
    catch (...)
    {
        remove_free_surface_model();
        throw;
    }
    return d_free_surface_model.get();
}

/** @brief Configure planar free-surface coupling from a Database. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_free_surface(const Database& database) -> free_surface_model_type*
{
    FreeSurfaceOptions options;
    int local_parse_failed = 0;
    std::string local_error;
    try
    {
        options = free_surface_options_from_database(database);
    }
    catch (const std::exception& error)
    {
        local_parse_failed = 1;
        local_error = error.what();
    }
    int any_parse_failed = 0;
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_parse_failed, &any_parse_failed);
    if (any_parse_failed != 0)
    {
        if (communicator->getSize() == 1 && !local_error.empty())
        {
            throw std::invalid_argument(local_error);
        }
        throw std::invalid_argument("Free-surface database options must be valid on every rank.");
    }
    return configure_free_surface(options);
}

/** @brief Remove all optional free-surface state. */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_free_surface_model()
{
    const auto removed = static_cast<bool>(d_free_surface_model) || static_cast<bool>(d_liquid_mass_inventory);
    const bool removed_planar_ale = static_cast<bool>(d_ale_motion);
    if (removed && d_boiling_source_model)
    {
        const auto local_steam_mass = d_boiling_source_model->global_submerged_steam_mass();
        const int local_invalid_mass = !std::isfinite(local_steam_mass) || local_steam_mass < scalar_type{} ? 1 : 0;
        int any_invalid_mass = 0;
        const auto communicator = d_mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_mass, &any_invalid_mass);
        if (any_invalid_mass != 0)
        {
            throw std::logic_error(
                "Cannot remove the planar free-surface model while the boiling steam inventory is invalid.");
        }

        scalar_type minimum_steam_mass{};
        scalar_type maximum_steam_mass{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_steam_mass, &minimum_steam_mass);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_steam_mass, &maximum_steam_mass);
        if (minimum_steam_mass != maximum_steam_mass)
        {
            throw std::logic_error(
                "Cannot remove the planar free-surface model because the boiling steam inventory does not match on "
                "every mesh rank.");
        }
        if (maximum_steam_mass > scalar_type{})
        {
            throw std::logic_error(
                "Cannot remove the planar free-surface model while boiling owns nonzero submerged steam; retain the "
                "free-surface owner or reconstruct the solver from a state with zero submerged steam.");
        }
    }
    if (removed_planar_ale)
    {
        boussinesq_velocity_boundary_cache() = FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions());
    }
    clear_volume_continuity_target();
    clear_ale_pressure_boundary();
    d_ale_temperature_density = nullptr;
    d_active_ale.reset();
    d_ale_boundary.reset();
    d_volume_continuity_model.reset();
    d_mesh_relative_face_flux.reset();
    d_bubble_slip_volume_flux.reset();
    d_mesh_volume_rate.reset();
    d_continuity_residual.reset();
    d_ale_old_density.reset();
    d_ale_old_heat_capacity.reset();
    d_ale_motion.reset();
    d_free_surface_model.reset();
    d_liquid_mass_inventory.reset();
    d_clear_level.reset();
    d_pool_level.reset();
    d_headspace_pressure.reset();
    d_pool_occupancy.reset();
    d_free_surface_options = {};
    d_pool_occupancy_volume_error = {};
    d_free_surface_history.clear();
    d_planar_ale_diagnostics = {};
    d_ale_target_generation = 0;
    return removed;
}

/** @brief Find mutable configured free-surface state. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_free_surface_model() noexcept -> free_surface_model_type*
{
    return d_free_surface_model.get();
}

/** @brief Find immutable configured free-surface state. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_free_surface_model() const noexcept -> const free_surface_model_type*
{
    return d_free_surface_model.get();
}

/** @brief Return the accepted free-surface diagnostics. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::free_surface_diagnostics() const -> free_surface_diagnostics_type
{
    if (!d_free_surface_model)
    {
        throw std::logic_error("BoussinesqSolver has no configured free-surface model.");
    }
    return d_free_surface_model->diagnostics();
}

/** @brief Return mutable liquid-mass state. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::liquid_mass_inventory() -> liquid_mass_inventory_type&
{
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("BoussinesqSolver has no configured liquid-mass inventory.");
    }
    return *d_liquid_mass_inventory;
}

/** @brief Return immutable liquid-mass state. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::liquid_mass_inventory() const -> const liquid_mass_inventory_type&
{
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("BoussinesqSolver has no configured liquid-mass inventory.");
    }
    return *d_liquid_mass_inventory;
}

/** @brief Find mutable liquid-mass state. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_liquid_mass_inventory() noexcept -> liquid_mass_inventory_type*
{
    return d_liquid_mass_inventory.get();
}

/** @brief Find immutable liquid-mass state. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_liquid_mass_inventory() const noexcept -> const liquid_mass_inventory_type*
{
    return d_liquid_mass_inventory.get();
}

/** @brief Return the pure-liquid density field. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::rho_liquid() const -> const field_type&
{
    return liquid_mass_inventory().rhoLiquid();
}

/** @brief Return the replicated clear-level field. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::clear_level() const -> const field_type&
{
    if (!d_clear_level)
    {
        throw std::logic_error("BoussinesqSolver has no configured clear-level field.");
    }
    return *d_clear_level;
}

/** @brief Return the replicated pool-level field. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::pool_level() const -> const field_type&
{
    if (!d_pool_level)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-level field.");
    }
    return *d_pool_level;
}

/** @brief Return the replicated absolute headspace-pressure field. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::headspace_pressure() const -> const field_type&
{
    if (!d_headspace_pressure)
    {
        throw std::logic_error("BoussinesqSolver has no configured headspace-pressure field.");
    }
    return *d_headspace_pressure;
}

/** @brief Return the approximate cell-centre pool indicator. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::pool_occupancy() const -> const field_type&
{
    if (!d_pool_occupancy)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-occupancy field.");
    }
    return *d_pool_occupancy;
}

/** @brief Return the signed cell-centre occupancy-volume error. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::pool_occupancy_volume_error() const -> scalar_type
{
    if (!d_pool_occupancy)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-occupancy field.");
    }
    return d_pool_occupancy_volume_error;
}

/** @brief Return the accepted carrier flux relative to the moving mesh. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::mesh_relative_face_fluxes() const -> const face_flux_field_type&
{
    if (!d_mesh_relative_face_flux)
    {
        throw std::logic_error("BoussinesqSolver has no configured planar-ALE relative-flux field.");
    }
    return *d_mesh_relative_face_flux;
}

/** @brief Select the accepted ALE-relative flux for the Courant diagnostic. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::courant_transport_face_fluxes() const -> const face_flux_field_type&
{
    if (planar_ale_enabled() && d_mesh_relative_face_flux)
    {
        return *d_mesh_relative_face_flux;
    }
    return base_type::courant_transport_face_fluxes();
}

/** @brief Return immutable accepted free-surface history. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::free_surface_history() const noexcept -> const std::vector<FreeSurfaceHistoryRecord>&
{
    return d_free_surface_history;
}

/** @brief Write accepted free-surface history on mesh rank zero. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_free_surface_history_csv(const std::string& filename) const
{
    if (d_mesh->owned_cell_map()->getComm()->getRank() != 0)
    {
        return;
    }
    std::ofstream output(filename, std::ios::out | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("Could not create free-surface history CSV file '" + filename + "'.");
    }
    const bool include_planar_ale = d_free_surface_options.mode == FreeSurfaceMode::PlanarALE;
    output << "time_s,time_step_s,liquid_mass_kg,cumulative_evaporated_mass_kg,"
              "cumulative_condensed_mass_kg,dryout_mass_deficit_kg,liquid_mass_residual_kg,"
              "liquid_mass_residual_normalized,liquid_step_mass_residual_kg,"
              "liquid_step_mass_residual_normalized,liquid_volume_m3,submerged_bubble_volume_m3,pool_volume_m3,"
              "clear_level_m,pool_level_m,clear_level_rate_m_per_s,pool_level_rate_m_per_s,"
              "surface_area_m2,headspace_volume_m3,headspace_pressure_pa,headspace_temperature_k,"
              "headspace_total_moles,overflow_volume_m3,dryout_volume_deficit_m3,"
              "configured_level_underflow_m,configured_level_overflow_m,occupancy_volume_error_m3,"
              "volume_closure_residual_m3,volume_closure_residual_normalized,nonlinear_iterations,"
              "nonlinear_residual_pa,h2_generated_mol,h2_dissolved_mol,h2_microbubble_submerged_mol,"
              "h2_large_bubble_submerged_mol,h2_submerged_mol,h2_headspace_mol,h2_vented_mol,"
              "h2_escaped_this_step_mol,h2_other_sink_mol,h2_closure_mol,h2_closure_normalized,"
              "gas_closure_mol,gas_closure_normalized,"
              "boiling_requested_mass_kg,"
              "boiling_accepted_mass_kg,boiling_rejected_mass_kg,boiling_condensed_mass_kg,"
              "boiling_submerged_steam_mass_kg,boiling_submerged_steam_volume_m3,"
              "boiling_condensation_latent_energy_release_j,boiling_mass_residual_kg,"
              "boiling_void_residual_m3,boiling_latent_energy_residual_j";
    if (include_planar_ale)
    {
        output << ",ale_old_geometry_epoch,ale_new_geometry_epoch,ale_old_mesh_volume_m3,"
                  "ale_new_mesh_volume_m3,ale_mesh_pool_mismatch_m3,ale_gcl_max_m3_per_s,"
                  "ale_gcl_max_normalized,ale_outer_correctors,ale_level_residual_m,"
                  "ale_pressure_residual_pa,ale_material_state_residual,"
                  "ale_gas_state_residual,ale_material_source_m3_per_s,"
                  "ale_carrier_transport_m3_per_s,ale_bubble_slip_divergence_m3_per_s,"
                  "ale_bubble_escape_m3_per_s,ale_source_pool_closure_m3_per_s,"
                  "ale_continuity_l2_m3_per_s,ale_continuity_max_m3_per_s,"
                  "ale_continuity_normalized,ale_liquid_mass_residual_kg,"
                  "ale_gas_inventory_residual_mol,ale_energy_residual_j,"
                  "ale_energy_residual_normalized,ale_rejected_transactions";
    }
    output << '\n';
    output << std::setprecision(std::numeric_limits<scalar_type>::max_digits10);
    for (const auto& record : d_free_surface_history)
    {
        const auto& surface = record.free_surface;
        const auto& liquid = record.liquid_mass;
        const auto boiling = record.boiling.value_or(boiling_diagnostics_type{});
        auto gas = [&](const GasMolesBySpecies& values, std::string_view species)
        { return free_surface_species_moles(values, species); };
        output << surface.time << ',' << surface.time_step << ',' << liquid.total_mass << ','
               << liquid.cumulative_evaporated_mass << ',' << liquid.cumulative_condensed_mass << ','
               << liquid.dryout_mass_deficit << ',' << liquid.mass_balance_residual << ','
               << liquid.normalized_mass_balance_residual << ',' << liquid.step_mass_balance_residual << ','
               << liquid.normalized_step_mass_balance_residual << ',' << surface.liquid_volume << ','
               << surface.submerged_bubble_volume << ',' << surface.pool_volume << ',' << surface.clear_level << ','
               << surface.pool_level << ',' << surface.clear_level_rate << ',' << surface.pool_level_rate << ','
               << surface.surface_area << ',' << surface.headspace.volume << ',' << surface.headspace.pressure << ','
               << surface.headspace.temperature << ',' << surface.headspace.total_moles << ','
               << surface.overflow_volume << ',' << surface.dryout_deficit << ',' << surface.configured_level_underflow
               << ',' << surface.configured_level_overflow << ',' << record.pool_occupancy_volume_error << ','
               << surface.volume_closure_residual << ',' << surface.normalized_volume_closure_residual << ','
               << surface.nonlinear_iterations << ',' << surface.nonlinear_residual << ','
               << gas(surface.generated_gas_moles, "H2") << ',' << gas(surface.dissolved_gas_moles, "H2") << ','
               << record.microbubble_hydrogen_moles << ',' << record.large_bubble_hydrogen_moles << ','
               << gas(surface.submerged_gas_moles, "H2") << ',' << gas(surface.headspace_gas_moles, "H2") << ','
               << gas(surface.vented_gas_moles, "H2") << ',' << gas(surface.escaped_gas_moles_this_step, "H2") << ','
               << gas(surface.other_sink_gas_moles, "H2") << ',' << gas(surface.gas_closure_by_species, "H2") << ','
               << gas(surface.normalized_gas_closure_by_species, "H2") << ',' << surface.gas_closure_residual << ','
               << surface.normalized_gas_closure_residual << ',' << boiling.requested_evaporation_mass << ','
               << boiling.accepted_evaporation_mass << ',' << boiling.rejected_vapor_mass << ','
               << boiling.condensed_liquid_mass << ',' << boiling.submerged_steam_mass << ','
               << boiling.submerged_steam_volume << ',' << boiling.condensation_latent_energy_release << ','
               << boiling.mass_balance_residual << ',' << boiling.void_balance_residual << ','
               << boiling.latent_energy_balance_residual;
        if (include_planar_ale)
        {
            const auto ale = record.planar_ale.value_or(PlanarALEStepDiagnostics{});
            output << ',' << ale.old_geometry_epoch << ',' << ale.new_geometry_epoch << ','
                   << ale.old_mesh_volume << ',' << ale.new_mesh_volume << ',' << ale.mesh_vessel_mismatch << ','
                   << ale.maximum_gcl_residual << ',' << ale.maximum_normalized_gcl_residual << ','
                   << ale.outer_correctors << ',' << ale.level_residual << ',' << ale.pressure_residual << ','
                   << ale.material_state_residual << ',' << ale.gas_state_residual << ','
                   << ale.volume_source.global_material_source << ','
                   << ale.volume_source.global_carrier_transport << ','
                   << ale.volume_source.global_bubble_slip_divergence << ','
                   << ale.volume_source.bubble_escape_volume_rate << ','
                   << ale.volume_source.source_pool_closure_residual << ',' << ale.continuity.l2 << ','
                   << ale.continuity.maximum << ',' << ale.continuity.normalized_l2 << ','
                   << ale.liquid_mass_residual << ',' << ale.gas_inventory_residual << ','
                   << ale.energy_residual << ',' << ale.normalized_energy_residual << ','
                   << ale.rejected_transactions;
        }
        output << '\n';
    }
    if (!output)
    {
        throw std::runtime_error("Could not write free-surface history CSV file '" + filename + "'.");
    }
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
    if (d_liquid_mass_inventory && d_liquid_mass_inventory->initialized())
    {
        throw std::logic_error(
            "Configure material updates before initializing the free-surface liquid-mass reference state.");
    }
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
 * @brief Refresh material properties, feedback, and registered heat sources.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::refresh_physical_models()
{
    update_context_type context{d_time, d_step_index, *d_mesh, temperature(), pressure(), velocity()};
    stored_material_properties().update(context);
    if (d_free_surface_model && !d_free_surface_model->initialized() && d_material_feedback_model)
    {
        // Lazy free-surface initialization needs the same accepted material
        // state as explicit primary-field initialization before radiolytic
        // bubble radii and liquid reference mass are reconstructed.
        refresh_material_feedback(d_time);
    }
    initialize_radiolytic_gas_state(
        d_step_index == 0 && !(d_free_surface_model && d_free_surface_model->initialized()));
    refresh_material_feedback(d_time);
    stored_temperature_sources().update(context);
}

/**
 * @brief Initialize Sheng radiolysis inventories and dependent liquid fractions.
 *
 * @tparam Pack Tpetra type pack.
 * @param force Whether to reinitialize an already initialized model.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::initialize_radiolytic_gas_state(bool force)
{
    if (!d_radiolytic_gas_model || !d_radiolytic_gas_model->supplies_void_fraction() ||
        (!force && d_radiolytic_gas_model->initial_state_initialized()))
    {
        return;
    }

    d_radiolytic_gas_model->initialize_state(
        d_time, temperature(), pressure(), velocity(), stored_material_properties(), force);
    if (d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model->initialize_from(d_radiolytic_gas_model->alpha_g());
    }
    if (d_precursor_model && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(*active_alpha_l_field());
    }
}

/**
 * @brief Apply material feedback for the supplied physical time.
 *
 * @tparam Pack Tpetra type pack.
 * @param time Physical time passed to feedback callbacks.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::refresh_material_feedback(scalar_type time)
{
    if (!d_material_feedback_model)
    {
        return;
    }
    update_context_type context{time, d_step_index, *d_mesh, temperature(), pressure(), velocity()};
    d_material_feedback_model->apply(context, active_alpha_g_field(), stored_material_properties());
}

/** @brief Reject rank-divergent optional-model ownership before conditional collectives. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::validate_collective_model_state() const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    constexpr int absent = -1;
    const std::array<int, 28> local_model_state{static_cast<int>(d_free_surface_model != nullptr),
        d_free_surface_model ? static_cast<int>(d_free_surface_model->initialized()) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.enabled) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.mode) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.headspace.mode) : absent,
        static_cast<int>(d_liquid_mass_inventory != nullptr),
        d_liquid_mass_inventory ? static_cast<int>(d_liquid_mass_inventory->initialized()) : absent,
        static_cast<int>(d_clear_level != nullptr), static_cast<int>(d_pool_level != nullptr),
        static_cast<int>(d_headspace_pressure != nullptr), static_cast<int>(d_pool_occupancy != nullptr),
        static_cast<int>(d_radiolytic_gas_model != nullptr),
        d_radiolytic_gas_model ? static_cast<int>(d_radiolytic_gas_model->mode()) : absent,
        d_radiolytic_gas_model ? static_cast<int>(d_radiolytic_gas_model->options().pressure_mode) : absent,
        d_radiolytic_gas_model ? static_cast<int>(d_radiolytic_gas_model->initial_state_initialized()) : absent,
        static_cast<int>(d_boiling_source_model != nullptr),
        d_boiling_source_model ? static_cast<int>(d_boiling_source_model->enabled()) : absent,
        d_boiling_source_model ? static_cast<int>(d_boiling_source_model->phase_change_completion_pending()) : absent,
        static_cast<int>(d_scalar_void_fraction_model != nullptr),
        static_cast<int>(d_material_feedback_model != nullptr),
        d_material_feedback_model ? static_cast<int>(d_material_feedback_model->options().density_mode) : absent,
        static_cast<int>(d_precursor_model != nullptr),
        d_precursor_model ? static_cast<int>(d_precursor_model->enabled()) : absent,
        static_cast<int>(d_fission_power_source != nullptr), static_cast<int>(stored_turbulence_model().enabled()),
        static_cast<int>(d_physical_model_enabled), static_cast<int>(d_primary_fields_initialized),
        static_cast<int>(d_free_surface_step_failed)};
    std::array<int, 28> minimum_model_state{};
    std::array<int, 28> maximum_model_state{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_model_state.size()),
        local_model_state.data(), minimum_model_state.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_model_state.size()),
        local_model_state.data(), maximum_model_state.data());
    if (minimum_model_state != maximum_model_state)
    {
        throw std::invalid_argument(
            "Boussinesq optional-model presence, enabled state, modes, and dependent ownership must match on every "
            "mesh rank.");
    }
}

/**
 * @brief Validate free-surface ownership and pressure-coupling seams.
 *
 * Scalar void is accepted only when a conservative model owns it.  The
 * two-population radiolysis model owns its bounded mirror; boiling owns only
 * the globally tracked submerged-steam volume.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::validate_free_surface_configuration(const FreeSurfaceOptions& options) const
{
    validate_free_surface_options(options);
    if (!options.enabled)
    {
        return;
    }
    if (options.mode == FreeSurfaceMode::PlanarALE)
    {
        validate_planar_ale_support(options);
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->mode() == RadiolyticGasMode::IdealGasSource)
    {
        throw std::invalid_argument("The planar free-surface budget cannot use idealGasSource "
                                    "radiolysis because it does not own a conservative gas inventory.");
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->mode() == RadiolyticGasMode::Sheng2024TwoPopulation &&
        d_radiolytic_gas_model->options().pressure_mode != RadiolyticPressureMode::Constant &&
        d_radiolytic_gas_model->options().pressure_mode != RadiolyticPressureMode::Reconstructed)
    {
        throw std::invalid_argument("Planar free-surface coupling requires constant or reconstructed "
                                    "radiolytic absolute pressure.");
    }
    if (options.headspace.mode == HeadspaceMode::Closed && d_boiling_source_model && d_boiling_source_model->enabled())
    {
        throw std::invalid_argument(
            "Closed free-surface coupling cannot use the fixed-temperature boiling model; pressure-dependent "
            "saturation from absolute headspace pressure is not implemented.");
    }

    if (!d_scalar_void_fraction_model || (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
                                             d_radiolytic_gas_model->supplies_void_fraction()))
    {
        return;
    }

    const auto represented_void = scalar_void_volume();
    const auto owned_steam_volume = d_boiling_source_model && d_boiling_source_model->enabled()
                                        ? d_boiling_source_model->global_submerged_steam_volume()
                                        : scalar_type{};
    const auto scale = std::max(scalar_type{1}, std::max(std::abs(represented_void), std::abs(owned_steam_volume)));
    const auto tolerance = std::max(scalar_type{128} * std::numeric_limits<scalar_type>::epsilon() * scale,
        static_cast<scalar_type>(
            options.coupling.volume_absolute_tolerance + options.coupling.volume_relative_tolerance * scale));
    if (std::abs(represented_void - owned_steam_volume) > tolerance)
    {
        throw std::invalid_argument("The planar free-surface budget rejects nonzero scalar void "
                                    "that is not owned by a conservative submerged-gas inventory.");
    }
}

/** @brief Require every rank-global ALE runtime choice to match exactly. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::validate_planar_ale_runtime_parity() const
{
    const auto* feedback_options = d_material_feedback_model
        ? &d_material_feedback_model->options()
        : nullptr;
    const auto* gas_options = d_radiolytic_gas_model
        ? &d_radiolytic_gas_model->options()
        : nullptr;
    std::vector<std::pair<std::string, bool>> temperature_sources;
    temperature_sources.reserve(stored_temperature_sources().entries().size());
    for (const auto& [name, source] : stored_temperature_sources().entries())
    {
        temperature_sources.emplace_back(name, source->enabled());
    }
    const auto local_encoding = encode_planar_ale_runtime_options(
        d_problem.time_options(), d_problem.linear_options(),
        this->pressure_linear_solver_options(), d_model_options,
        feedback_options, gas_options, d_problem.boundary_conditions(),
        temperature_sources,
        this->has_mutable_mesh_handle(), uses_legacy_backend(),
        physical_transport_enabled(), stored_turbulence_model().enabled(),
        d_boiling_source_model && d_boiling_source_model->enabled(),
        d_precursor_model && d_precursor_model->enabled(),
        d_scalar_void_fraction_explicitly_configured,
        stored_material_properties().has_updater(),
        stored_temperature_sources().has_dynamic_updates());

    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    long long root_size = rank == 0
        ? static_cast<long long>(local_encoding.size())
        : 0LL;
    Teuchos::broadcast(*communicator, 0, 1, &root_size);
    if (root_size < 0 || root_size > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(
            "planarALE runtime-option encoding is too large for MPI broadcast.");
    }
    std::string root_encoding(static_cast<size_t>(root_size), '\0');
    if (rank == 0)
    {
        root_encoding = local_encoding;
    }
    if (!root_encoding.empty())
    {
        Teuchos::broadcast(*communicator, 0, static_cast<int>(root_encoding.size()), root_encoding.data());
    }
    const int local_mismatch = local_encoding == root_encoding ? 0 : 1;
    int any_mismatch = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_mismatch, &any_mismatch);
    if (any_mismatch != 0)
    {
        throw std::invalid_argument(
            "planarALE time, linear, material, gas, boundary, and active-model options must match on every rank.");
    }
}

/** @brief Fail closed unless every active equation belongs to the tested ALE matrix. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::validate_planar_ale_support(const FreeSurfaceOptions& options) const
{
    validate_planar_ale_runtime_parity();
    if (!this->has_mutable_mesh_handle() || uses_legacy_backend())
    {
        throw std::invalid_argument(
            std::string(planar_ale_unavailable_diagnostic) + " A const-only or legacy solver mesh cannot be moved.");
    }
    if (!physical_transport_enabled())
    {
        throw std::invalid_argument("planarALE requires dimensional physical Boussinesq temperature transport.");
    }
    if (options.vessel.mode != VesselVolumeMapMode::ConstantArea ||
        options.liquid_mass.mode != LiquidVolumeMode::CellMassInventory ||
        options.range_policy != FreeSurfaceRangePolicy::Error ||
        options.liquid_mass.depletion_policy != FreeSurfaceRangePolicy::Error ||
        options.headspace.mode != HeadspaceMode::Vented ||
        options.headspace.temperature_mode != HeadspaceTemperatureMode::Fixed)
    {
        throw std::invalid_argument(
            "planarALE initially requires constantArea, cellMassInventory, fixed-temperature vented headspace, "
            "and error range/depletion policies.");
    }
    if (stored_turbulence_model().enabled())
    {
        throw std::invalid_argument("planarALE does not yet support RANS transports, wall distance, or wall laws.");
    }
    if ((d_boiling_source_model && d_boiling_source_model->enabled()) ||
        (d_precursor_model && d_precursor_model->enabled()))
    {
        throw std::invalid_argument(
            "planarALE initially rejects boiling/steam and delayed-neutron precursor transport.");
    }
    if (d_scalar_void_fraction_explicitly_configured &&
        (!d_radiolytic_gas_model || !d_radiolytic_gas_model->supplies_void_fraction()))
    {
        throw std::invalid_argument(
            "planarALE rejects scalar-void-only evolution; a conservative gas owner is required.");
    }
    if (!d_material_feedback_model ||
        d_material_feedback_model->options().density_mode != DensityFeedbackMode::BoussinesqTemperatureOnly ||
        d_material_feedback_model->options().thermal_expansion == scalar_type{})
    {
        throw std::invalid_argument("planarALE requires pure-liquid BoussinesqTemperatureOnly density feedback with "
                                    "nonzero thermal expansion.");
    }
    const auto gravity = d_problem.time_options().gravity_vector();
    const auto axis = static_cast<size_t>(options.gravity_axis);
    const std::array<scalar_type, 3> gravity_components{
        static_cast<scalar_type>(gravity.x), static_cast<scalar_type>(gravity.y), static_cast<scalar_type>(gravity.z)};
    bool transverse_gravity = false;
    for (size_t component = 0; component < gravity_components.size(); ++component)
    {
        transverse_gravity =
            transverse_gravity || (component != axis && gravity_components[component] != scalar_type{});
    }
    if (transverse_gravity || gravity_components[axis] > scalar_type{})
    {
        throw std::invalid_argument("planarALE gravity must be zero or point inward along the selected moving-top "
                                    "axis, with no transverse component.");
    }
    if (stored_material_properties().has_updater() || stored_temperature_sources().has_dynamic_updates())
    {
        throw std::invalid_argument(
            "planarALE rejects dynamic user material/source callbacks because they are not rollback-safe.");
    }

    // A spatially anchored fission profile would retain stale cell values after
    // the mesh moves.  Until that source owns an epoch-aware refresh contract,
    // every ALE heat/radiolysis use must be geometry-invariant.
    if (d_fission_power_source)
    {
        scalar_type local_minimum_fission = std::numeric_limits<scalar_type>::infinity();
        scalar_type local_maximum_fission = -std::numeric_limits<scalar_type>::infinity();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto value = d_fission_power_source->field().value(static_cast<local_ordinal_type>(owned));
            local_minimum_fission = std::min(local_minimum_fission, value);
            local_maximum_fission = std::max(local_maximum_fission, value);
        }
        scalar_type global_minimum_fission{};
        scalar_type global_maximum_fission{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_minimum_fission,
            &global_minimum_fission);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum_fission,
            &global_maximum_fission);
        if (global_minimum_fission != global_maximum_fission)
        {
            throw std::invalid_argument("planarALE currently requires a uniform geometry-invariant fission-power "
                                        "field; Gaussian and other spatial profiles are not epoch-aware.");
        }
    }

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
    {
        if (!d_fission_power_source)
        {
            throw std::invalid_argument(
                "planarALE Sheng radiolysis requires an authoritative fission-power source before setup.");
        }
        if (options.ale.maximum_correctors < 2)
        {
            throw std::invalid_argument(
                "planarALE Sheng radiolysis requires at least two outer correctors to couple gas state and geometry.");
        }
        const auto& gas = d_radiolytic_gas_model->options();
        if (gas.mode != RadiolyticGasMode::Sheng2024TwoPopulation ||
            gas.dissolved_transport != RadiolyticTransportMode::Advective ||
            gas.bubble_transport != BubbleTransportMode::General ||
            (gas.pressure_mode != RadiolyticPressureMode::Constant &&
                gas.pressure_mode != RadiolyticPressureMode::Reconstructed) ||
            gas.free_surface_patches != std::vector<std::string>{options.ale.top_boundary})
        {
            throw std::invalid_argument(
                "planarALE radiolysis requires Sheng2024TwoPopulation, advective dissolved H2, general bubble "
                "transport, constant/reconstructed pressure, and exactly the configured moving-top escape patch.");
        }
    }

    const auto& boundaries = d_problem.boundary_conditions();
    const auto pressure = boundaries.pressure.find(options.ale.top_boundary);
    if (pressure == boundaries.pressure.end() || pressure->second.type != BoundaryConditionType::Dirichlet ||
        pressure->second.value != scalar_type{})
    {
        throw std::invalid_argument(
            "planarALE moving top requires a zero-gauge Dirichlet physical-pressure boundary.");
    }
    const auto velocity_boundary = boundaries.velocity.find(options.ale.top_boundary);
    if (velocity_boundary == boundaries.velocity.end() ||
        (velocity_boundary->second.type != BoundaryConditionType::Slip &&
            velocity_boundary->second.type != BoundaryConditionType::Dirichlet) ||
        (velocity_boundary->second.type == BoundaryConditionType::Dirichlet &&
            (velocity_boundary->second.value.x != scalar_type{} ||
                velocity_boundary->second.value.y != scalar_type{} ||
                velocity_boundary->second.value.z != scalar_type{})))
    {
        throw std::invalid_argument(
            "planarALE moving top requires a configured slip or Dirichlet liquid-velocity boundary.");
    }
    int local_open_nonmoving_boundary = 0;
    for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
    {
        (void)batch;
        const auto name = d_mesh->boundary_batch_name(batch_id);
        if (name == options.ale.top_boundary)
        {
            continue;
        }
        const auto velocity = boundaries.velocity.find(name);
        const auto pressure_condition = boundaries.pressure.find(name);
        const bool closed_velocity = velocity != boundaries.velocity.end() &&
            (velocity->second.type == BoundaryConditionType::NoSlip ||
                velocity->second.type == BoundaryConditionType::Slip ||
                (velocity->second.type == BoundaryConditionType::Dirichlet &&
                    velocity->second.value.x == scalar_type{} &&
                    velocity->second.value.y == scalar_type{} &&
                    velocity->second.value.z == scalar_type{}));
        const bool neutral_pressure = pressure_condition == boundaries.pressure.end() ||
            (pressure_condition->second.type == BoundaryConditionType::Neumann &&
                pressure_condition->second.value == scalar_type{});
        if (!closed_velocity || !neutral_pressure)
        {
            local_open_nonmoving_boundary = 1;
        }
    }
    int any_open_nonmoving_boundary = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(),
        Teuchos::REDUCE_MAX, 1, &local_open_nonmoving_boundary,
        &any_open_nonmoving_boundary);
    if (any_open_nonmoving_boundary != 0)
    {
        throw std::invalid_argument(
            "planarALE initially rejects physical liquid inlets/outlets; every nonmoving boundary must be "
            "closed with homogeneous Neumann pressure.");
    }
    for (const auto& [name, condition] : boundaries.temperature)
    {
        (void)name;
        if (condition.type != BoundaryConditionType::Neumann || condition.value != scalar_type{})
        {
            throw std::invalid_argument(
                "planarALE initially supports adiabatic temperature boundaries only; use volumetric heating.");
        }
    }
}

/** @brief Lazily create the sole motion controller after accepted state exists. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::initialize_planar_ale_if_needed()
{
    if (!d_free_surface_model || d_free_surface_options.mode != FreeSurfaceMode::PlanarALE || d_ale_motion)
    {
        return;
    }
    if (!d_free_surface_model->initialized() || !d_liquid_mass_inventory || !d_liquid_mass_inventory->initialized())
    {
        return;
    }
    validate_planar_ale_support(d_free_surface_options);

    PlanarALEMeshMotionOptions motion_options;
    motion_options.axis = d_free_surface_options.gravity_axis;
    motion_options.deformation_start_elevation = d_free_surface_options.ale.deformation_start_elevation;
    motion_options.maximum_level_change = d_free_surface_options.ale.maximum_level_change;
    motion_options.quality_limits = d_free_surface_options.ale.quality_limits;
    motion_options.gcl_absolute_tolerance = d_free_surface_options.ale.gcl_absolute_tolerance;
    motion_options.gcl_relative_tolerance = d_free_surface_options.ale.gcl_relative_tolerance;

    auto motion = std::make_unique<ale_motion_type>(this->mutable_mesh_handle(), std::move(motion_options));
    auto boundary = std::make_unique<ale_boundary_type>(d_mesh, d_free_surface_options.ale.top_boundary,
        d_free_surface_options.gravity_axis, d_free_surface_model->volumeMap(),
        d_free_surface_options.coupling.volume_absolute_tolerance,
        d_free_surface_options.coupling.volume_relative_tolerance);
    const auto surface = d_free_surface_model->diagnostics();
    const auto level_scale = std::max({scalar_type{1}, std::abs(surface.pool_level),
        std::abs(boundary->diagnostics().surface_elevation)});
    const auto level_tolerance = static_cast<scalar_type>(d_free_surface_options.ale.level_absolute_tolerance) +
                                 static_cast<scalar_type>(d_free_surface_options.ale.level_relative_tolerance) *
                                     level_scale;
    if (std::abs(surface.pool_level - boundary->diagnostics().surface_elevation) > level_tolerance)
    {
        throw std::invalid_argument(
            "planarALE initial pool level must coincide with the selected mesh-top elevation.");
    }
    const auto volume_scale = std::max({scalar_type{1}, std::abs(surface.pool_volume),
        std::abs(boundary->diagnostics().global_mesh_volume)});
    const auto volume_tolerance = static_cast<scalar_type>(d_free_surface_options.coupling.volume_absolute_tolerance) +
                                  static_cast<scalar_type>(d_free_surface_options.coupling.volume_relative_tolerance) *
                                      volume_scale;
    if (std::abs(surface.pool_volume - boundary->diagnostics().global_mesh_volume) > volume_tolerance)
    {
        throw std::invalid_argument(
            "planarALE initial liquid-plus-bubble pool volume must equal the moving fluid-mesh volume.");
    }

    d_ale_boundary = std::move(boundary);
    d_ale_motion = std::move(motion);
    d_volume_continuity_model = std::make_unique<volume_continuity_model_type>(d_mesh,
        d_free_surface_options.coupling.volume_absolute_tolerance,
        d_free_surface_options.coupling.volume_relative_tolerance);
    d_mesh_relative_face_flux = std::make_unique<face_flux_field_type>(d_mesh, "meshRelativeFaceFlux");
    // Before the first motion trial the accepted mesh is stationary, so its
    // relative transport flux is the already accepted absolute flux. Preserve
    // that state when ALE is configured after earlier fixed-grid work.
    for (const auto face_lid : d_mesh_relative_face_flux->owned_face_ids())
    {
        d_mesh_relative_face_flux->set_owned_value(face_lid, projected_face_fluxes().value(face_lid));
    }
    d_mesh_relative_face_flux->sync_ghosts();
    d_bubble_slip_volume_flux = std::make_unique<face_flux_field_type>(d_mesh, "bubbleSlipVolumeFlux");
    d_mesh_volume_rate = std::make_unique<field_type>(d_mesh, "meshVolumeRate");
    d_continuity_residual = std::make_unique<field_type>(d_mesh, "continuityResidual");
    d_ale_old_density = std::make_unique<field_type>(d_mesh, "aleAcceptedOldDensity");
    d_ale_old_heat_capacity = std::make_unique<field_type>(d_mesh, "aleAcceptedOldHeatCapacity");
    d_planar_ale_diagnostics.initialized = true;
    d_planar_ale_diagnostics.old_geometry_epoch = d_mesh->geometry_epoch();
    d_planar_ale_diagnostics.new_geometry_epoch = d_mesh->geometry_epoch();
    d_planar_ale_diagnostics.old_mesh_volume = d_ale_boundary->diagnostics().global_mesh_volume;
    d_planar_ale_diagnostics.new_mesh_volume = d_ale_boundary->diagnostics().global_mesh_volume;
}

/** @brief Enforce top kinematics and derive a distinct carrier-relative flux. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::form_mesh_relative_flux(face_flux_field_type& absolute_flux)
{
    if (!d_active_ale || !d_ale_boundary || !d_mesh_relative_face_flux)
    {
        throw std::logic_error("planarALE relative flux requires one active geometry trial.");
    }
    // Pressure and coupled solvers also own this exact override. Reapplying it
    // here makes the transport boundary invariant explicit at every consumer.
    d_ale_boundary->enforce_kinematic_flux(*d_active_ale, absolute_flux);
    FVM::mesh_relative_face_fluxes(absolute_flux, *d_active_ale, *d_mesh_relative_face_flux);
}

/** @brief Build extensive liquid-plus-raw-bubble material volume per local cell. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::material_volumes(const field_type& liquid_mass_density,
    const field_type& pure_liquid_density_field, std::span<const real_t> cell_volumes) const
    -> std::vector<scalar_type>
{
    if (cell_volumes.size() != d_mesh->num_local_cells())
    {
        throw std::invalid_argument("planarALE material-volume assembly requires mesh-local cell-volume order.");
    }
    std::vector<scalar_type> result(cell_volumes.size());
    const field_type* raw_bubble = d_radiolytic_gas_model && d_radiolytic_gas_model->enabled()
        ? &d_radiolytic_gas_model->raw_bubble_volume_fraction()
        : nullptr;
    int local_invalid = 0;
    for (size_t local = 0; local < result.size(); ++local)
    {
        const auto cell = static_cast<local_ordinal_type>(local);
        const bool owned = local < d_mesh->num_owned_cells();
        const auto density = owned ? pure_liquid_density_field.value(cell)
                                   : pure_liquid_density_field.local_value(cell);
        const auto mass_density = owned ? liquid_mass_density.value(cell)
                                        : liquid_mass_density.local_value(cell);
        const auto volume = static_cast<scalar_type>(cell_volumes[local]);
        const auto bubble_fraction = raw_bubble
            ? (owned ? raw_bubble->value(cell) : raw_bubble->local_value(cell))
            : scalar_type{};
        result[local] = volume * (mass_density / density + bubble_fraction);
        local_invalid = local_invalid || !std::isfinite(density) || density <= scalar_type{} ||
                        !std::isfinite(mass_density) || mass_density < scalar_type{} ||
                        !std::isfinite(bubble_fraction) || bubble_fraction < scalar_type{} ||
                        !std::isfinite(result[local]) || result[local] < scalar_type{};
    }
    int any_invalid = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
        1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::runtime_error("planarALE material-volume state is non-finite or physically invalid.");
    }
    return result;
}

/** @brief Volume-weighted mean of the dynamic/gauge pressure field. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::volume_weighted_mean_pressure() const -> scalar_type
{
    scalar_type local_pressure_volume{};
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
        local_pressure_volume += pressure().value(cell) * volume;
        local_volume += volume;
    }
    std::array<scalar_type, 2> local{local_pressure_volume, local_volume};
    std::array<scalar_type, 2> global{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
        static_cast<int>(local.size()), local.data(), global.data());
    return global[1] > scalar_type{} ? global[0] / global[1] : scalar_type{};
}

/** @brief Integrate carrier-density*cp*T on supplied owned-cell geometry. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::total_sensible_energy(const field_type& density,
    const field_type& heat_capacity, std::span<const real_t> cell_volumes) const -> scalar_type
{
    if (cell_volumes.size() != d_mesh->num_local_cells())
    {
        throw std::invalid_argument("planarALE energy accounting requires mesh-local cell-volume order.");
    }
    scalar_type local_energy{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        local_energy += static_cast<scalar_type>(cell_volumes[owned]) * density.value(cell) *
                        heat_capacity.value(cell) * temperature().value(cell);
    }
    scalar_type global_energy{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
        1, &local_energy, &global_energy);
    return global_energy;
}

/** @brief Prescribe the exact swept-volume rate on the dynamic-pressure top. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::configure_ale_pressure_boundary(std::uint64_t generation)
{
    if (!d_active_ale || !d_ale_boundary)
    {
        throw std::logic_error("planarALE pressure boundary requires an active motion trial.");
    }
    native_pressure_projection().set_fixed_boundary_flux_provider(
        {d_ale_boundary->name()},
        [this](int, size_t, local_ordinal_type face_lid)
        {
            if (!d_active_ale || !d_ale_boundary || !d_ale_boundary->contains(face_lid))
            {
                throw std::logic_error("planarALE fixed-flux provider was used outside its active moving patch.");
            }
            return static_cast<scalar_type>(d_active_ale->face_mesh_fluxes()[static_cast<size_t>(face_lid)]);
        },
        generation);
    boussinesq_coupled_pressure_velocity_solver().set_fixed_boundary_flux_provider(
        {d_ale_boundary->name()},
        [this](int, size_t, local_ordinal_type face_lid)
        {
            if (!d_active_ale || !d_ale_boundary || !d_ale_boundary->contains(face_lid))
            {
                throw std::logic_error("planarALE coupled fixed-flux provider was used outside its active patch.");
            }
            return static_cast<scalar_type>(d_active_ale->face_mesh_fluxes()[static_cast<size_t>(face_lid)]);
        },
        generation);
}

/** @brief Restore ordinary physical-pressure boundary behavior. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::clear_ale_pressure_boundary() noexcept
{
    if (uses_legacy_backend())
    {
        return;
    }
    try
    {
        native_pressure_projection().clear_fixed_boundary_flux_provider();
        boussinesq_coupled_pressure_velocity_solver().clear_fixed_boundary_flux_provider();
    }
    catch (...)
    {
    }
}

/** @brief Evaluate pure liquid density without reusing a void mixture. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pure_liquid_density(local_ordinal_type cell_lid) const -> scalar_type
{
    if (d_material_feedback_model)
    {
        const auto density_mode = d_material_feedback_model->options().density_mode;
        if (density_mode == DensityFeedbackMode::BoussinesqVoid || density_mode == DensityFeedbackMode::Mixture)
        {
            return d_material_feedback_model->pure_liquid_density(temperature().value(cell_lid));
        }
    }
    return stored_material_properties().density.value(cell_lid);
}

/** @brief Return the configured headspace temperature for one closure. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::headspace_temperature(scalar_type time) const -> scalar_type
{
    const auto& options = d_free_surface_options.headspace;
    if (options.temperature_mode == HeadspaceTemperatureMode::Fixed)
    {
        return options.initial_temperature;
    }
    if (options.temperature_mode == HeadspaceTemperatureMode::Prescribed)
    {
        return prescribed_headspace_temperature(options, time);
    }

    scalar_type local_temperature_volume{};
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
        local_temperature_volume += temperature().value(cell_lid) * volume;
        local_volume += volume;
    }
    scalar_type local_values[]{local_temperature_volume, local_volume};
    scalar_type global_values[2]{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local_values, global_values);
    if (!(global_values[1] > scalar_type{}) || !std::isfinite(global_values[0]))
    {
        throw std::runtime_error("Bulk-liquid headspace temperature requires finite temperature "
                                 "and positive global mesh volume.");
    }
    const auto average = global_values[0] / global_values[1];
    if (!std::isfinite(average) || average <= scalar_type{})
    {
        throw std::runtime_error("Bulk-liquid headspace temperature must be finite and positive.");
    }
    return average;
}

/** @brief Integrate the authoritative scalar void over owned cells. */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::scalar_void_volume() const -> scalar_type
{
    if (!d_scalar_void_fraction_model)
    {
        return {};
    }
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        local_volume += d_scalar_void_fraction_model->alpha_g().value(cell_lid) *
                        static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
    }
    scalar_type global_volume{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    return global_volume;
}

/** @brief Assemble callbacks and conservative gas compartments. */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::make_free_surface_update(
    scalar_type time, bool initializing, const FreeSurfaceAccountingPreview* preview) -> FreeSurfaceUpdate
{
    if (!d_liquid_mass_inventory || !d_free_surface_model)
    {
        throw std::logic_error("Free-surface update requires both model and liquid inventory.");
    }

    FreeSurfaceUpdate update;
    const auto liquid_volume = preview ? preview->liquid.liquid_volume : d_liquid_mass_inventory->liquidVolume();
    update.liquid_volume_at_pressure = [liquid_volume](real_t) { return static_cast<real_t>(liquid_volume); };
    update.liquid_volume_deficit = preview ? static_cast<real_t>(preview->liquid_volume_deficit) : real_t{};
    update.bubble_volume_at_pressure = [this](real_t pressure)
    {
        if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
            d_radiolytic_gas_model->supplies_void_fraction())
        {
            const auto pressure_mode = d_radiolytic_gas_model->options().pressure_mode;
            if (pressure_mode != RadiolyticPressureMode::Constant &&
                pressure_mode != RadiolyticPressureMode::Reconstructed)
            {
                throw std::logic_error("Planar free-surface bubble closure requires constant or "
                                       "reconstructed radiolytic pressure.");
            }
            auto pressure_offset = static_cast<scalar_type>(pressure);
            if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
                pressure_mode == RadiolyticPressureMode::Reconstructed)
            {
                pressure_offset += volume_weighted_mean_pressure();
            }
            return static_cast<real_t>(
                d_radiolytic_gas_model->evaluate_submerged_bubble_volume(pressure_offset));
        }
        if (d_boiling_source_model && d_boiling_source_model->enabled())
        {
            return static_cast<real_t>(d_boiling_source_model->global_submerged_steam_volume());
        }
        return real_t{};
    };

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() && d_radiolytic_gas_model->supplies_void_fraction())
    {
        update.gas.generated_moles["H2"] = d_radiolytic_gas_model->cumulative_hydrogen_produced();
        update.gas.dissolved_moles["H2"] = d_radiolytic_gas_model->global_dissolved_hydrogen_moles();
        const auto microbubble_moles = d_radiolytic_gas_model->global_microbubble_hydrogen_moles();
        const auto large_bubble_moles = d_radiolytic_gas_model->global_large_bubble_hydrogen_moles();
        update.gas.submerged_population_moles["microbubble"]["H2"] = microbubble_moles;
        update.gas.submerged_population_moles["largeBubble"]["H2"] = large_bubble_moles;
        update.gas.submerged_moles["H2"] = microbubble_moles + large_bubble_moles;
        update.gas.other_sink_moles["H2"] = d_radiolytic_gas_model->cumulative_dissolved_hydrogen_outflow();
        update.minimum_valid_absolute_pressure =
            static_cast<real_t>(d_radiolytic_gas_model->minimum_valid_absolute_pressure_offset());
        if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
            d_radiolytic_gas_model->options().pressure_mode == RadiolyticPressureMode::Reconstructed)
        {
            update.minimum_valid_absolute_pressure -= static_cast<real_t>(volume_weighted_mean_pressure());
        }
        if (!initializing)
        {
            const auto cumulative_escape =
                static_cast<scalar_type>(d_radiolytic_gas_model->cumulative_submerged_bubble_hydrogen_escaped());
            const auto committed_escape = static_cast<scalar_type>(
                free_surface_species_moles(d_free_surface_model->committedEscapedMoles(), "H2"));
            auto pending_escape = cumulative_escape - committed_escape;
            const auto roundoff = scalar_type{64} * std::numeric_limits<scalar_type>::epsilon() *
                                  std::max({scalar_type{1}, std::abs(cumulative_escape), std::abs(committed_escape)});
            const int local_invalid_escape = !std::isfinite(pending_escape) || pending_escape < -roundoff ? 1 : 0;
            int any_invalid_escape = 0;
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid_escape,
                &any_invalid_escape);
            if (any_invalid_escape != 0)
            {
                throw std::runtime_error(
                    "Cumulative submerged-bubble escape is smaller than the committed headspace/vent transfer.");
            }
            if (pending_escape < scalar_type{})
            {
                pending_escape = scalar_type{};
            }
            update.gas.escaped_moles_this_step["H2"] = static_cast<real_t>(pending_escape);
        }
    }
    update.headspace_temperature = headspace_temperature(time);
    update.time = time;
    update.time_step = initializing ? real_t{} : static_cast<real_t>(d_problem.time_options().time_step);
    return update;
}

/** @brief Initialize a configured free-surface after fields are usable. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_free_surface_if_needed(
    bool allow_default_fields, bool dependencies_already_refreshed)
{
    if (!d_free_surface_model || d_free_surface_model->initialized() ||
        (!d_primary_fields_initialized && !allow_default_fields))
    {
        return;
    }
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("Configured free-surface model lacks its liquid-mass inventory.");
    }
    try
    {
        if (!dependencies_already_refreshed && d_physical_model_enabled)
        {
            update_context_type context{d_time, d_step_index, *d_mesh, temperature(), pressure(), velocity()};
            stored_material_properties().update(context);
        }
        if (!dependencies_already_refreshed && d_material_feedback_model)
        {
            refresh_material_feedback(d_time);
        }
        if (!dependencies_already_refreshed)
        {
            initialize_radiolytic_gas_state(d_step_index == 0);
            if (d_material_feedback_model && d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction())
            {
                refresh_material_feedback(d_time);
            }
        }

        const auto initial_volume = configured_initial_liquid_volume(d_free_surface_options).value_or(real_t{});
        d_liquid_mass_inventory->initialize(static_cast<scalar_type>(initial_volume),
            [this](local_ordinal_type cell_lid) { return pure_liquid_density(cell_lid); });
        const auto configured_volume = configured_initial_liquid_volume(d_free_surface_options);
        if (d_free_surface_options.liquid_mass.initial_liquid_mass && configured_volume)
        {
            const auto actual_volume = static_cast<real_t>(d_liquid_mass_inventory->liquidVolume());
            const auto scale = std::max(std::abs(*configured_volume), std::abs(actual_volume));
            const auto tolerance = d_free_surface_options.coupling.volume_absolute_tolerance +
                                   d_free_surface_options.coupling.volume_relative_tolerance * scale;
            if (std::abs(actual_volume - *configured_volume) > tolerance)
            {
                throw std::invalid_argument("Configured initial liquid mass and fill volume/level are "
                                            "inconsistent at the initialized pure-liquid density.");
            }
        }
        d_free_surface_model->initialize(make_free_surface_update(d_time, true));
        if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
            d_radiolytic_gas_model->supplies_void_fraction())
        {
            const auto pressure_mode = d_radiolytic_gas_model->options().pressure_mode;
            if (pressure_mode == RadiolyticPressureMode::Constant ||
                pressure_mode == RadiolyticPressureMode::Reconstructed)
            {
                auto offset = static_cast<scalar_type>(d_free_surface_model->headspacePressure());
                if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
                    pressure_mode == RadiolyticPressureMode::Reconstructed)
                {
                    offset += volume_weighted_mean_pressure();
                }
                d_radiolytic_gas_model->set_absolute_pressure_offset(offset);
            }
        }
        // Lazy primary-field initialization must complete the same ALE
        // geometry/support preflight as configure_free_surface().  Do this
        // before publishing fields or accepted history.
        initialize_planar_ale_if_needed();
        publish_free_surface_fields();
        record_free_surface_history();
    }
    catch (...)
    {
        remove_free_surface_model();
        throw;
    }
}

/** @brief Apply accepted phase change and close the planar volume budget. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::advance_free_surface(scalar_type time_step)
{
    if (!d_free_surface_model)
    {
        return;
    }
    initialize_free_surface_if_needed(true);

    scalar_type evaporated_mass{};
    scalar_type condensed_mass{};
    const field_type* evaporation_mass_rate = nullptr;
    const field_type* condensation_mass_rate = nullptr;
    if (d_boiling_source_model && d_boiling_source_model->enabled())
    {
        if (d_boiling_source_model->phase_change_completion_pending())
        {
            throw std::logic_error("Free-surface liquid mass cannot advance before boiling "
                                   "scalar-void completion.");
        }
        evaporated_mass = d_boiling_source_model->accepted_evaporation_mass_this_step();
        condensed_mass = d_boiling_source_model->condensed_liquid_mass_this_step();
        evaporation_mass_rate = &d_boiling_source_model->phase_change_mass_rate();
        condensation_mass_rate = &d_boiling_source_model->condensation_mass_rate();
    }
    d_liquid_mass_inventory->updatePureLiquidDensity(
        [this](local_ordinal_type cell_lid) { return pure_liquid_density(cell_lid); });

    FreeSurfaceAccountingPreview preview;
    const auto current_liquid = d_liquid_mass_inventory->diagnostics();
    const auto phase_change_preview = [&]
    {
        if (d_liquid_mass_inventory->mode() == LiquidVolumeMode::CellMassInventory)
        {
            return d_liquid_mass_inventory->previewCellwiseAdvance(time_step, projected_face_fluxes(),
                evaporation_mass_rate, condensation_mass_rate, LinearSolverOptions{});
        }
        return d_liquid_mass_inventory->previewPhaseChange(evaporated_mass, condensed_mass);
    }();
    preview.liquid = phase_change_preview.diagnostics();
    if (d_liquid_mass_inventory->mode() == LiquidVolumeMode::CellMassInventory && d_boiling_source_model &&
        d_boiling_source_model->enabled())
    {
        const auto integrated_evaporation =
            preview.liquid.cumulative_evaporated_mass - current_liquid.cumulative_evaporated_mass;
        const auto integrated_condensation =
            preview.liquid.cumulative_condensed_mass - current_liquid.cumulative_condensed_mass;
        const auto scale = std::max({scalar_type{1}, std::abs(evaporated_mass), std::abs(condensed_mass)});
        const auto tolerance = scalar_type{1024} * std::numeric_limits<scalar_type>::epsilon() * scale;
        if (std::abs(integrated_evaporation - evaporated_mass) > tolerance ||
            std::abs(integrated_condensation - condensed_mass) > tolerance)
        {
            throw std::runtime_error(
                "Cellwise liquid-mass sources disagree with the accepted boiling phase-change totals.");
        }
    }
    preview.liquid_volume_deficit = (preview.liquid.dryout_mass_deficit - current_liquid.dryout_mass_deficit) *
                                    preview.liquid.mass_weighted_specific_volume;
    d_free_surface_model->update(make_free_surface_update(d_time + time_step, false, &preview));
    d_liquid_mass_inventory->commitPhaseChange(phase_change_preview);
    if (phase_change_preview.transportStatistics())
    {
        d_last_step_statistics.add(*phase_change_preview.transportStatistics());
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() && d_radiolytic_gas_model->supplies_void_fraction())
    {
        const auto pressure_mode = d_radiolytic_gas_model->options().pressure_mode;
        if (pressure_mode == RadiolyticPressureMode::Constant || pressure_mode == RadiolyticPressureMode::Reconstructed)
        {
            d_radiolytic_gas_model->set_absolute_pressure_offset(d_free_surface_model->headspacePressure());
        }
    }
    publish_free_surface_fields();
}

/** @brief Publish global diagnostics and a cell-centre occupancy mask. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::publish_free_surface_fields()
{
    if (!d_free_surface_model || !d_free_surface_model->initialized() || !d_clear_level || !d_pool_level ||
        !d_headspace_pressure || !d_pool_occupancy)
    {
        throw std::logic_error("Free-surface output fields require initialized owned state.");
    }
    const auto diagnostics = d_free_surface_model->diagnostics();
    d_clear_level->put_scalar(diagnostics.clear_level);
    d_pool_level->put_scalar(diagnostics.pool_level);
    d_headspace_pressure->put_scalar(diagnostics.headspace.pressure);

    scalar_type local_occupancy_volume{};
    const auto axis = static_cast<size_t>(d_free_surface_options.gravity_axis);
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto coordinate = static_cast<scalar_type>(d_mesh->cell_centroid(cell_lid).component(axis));
        const auto occupancy = coordinate <= diagnostics.pool_level ? scalar_type{1} : scalar_type{};
        d_pool_occupancy->set_owned_value(cell_lid, occupancy);
        local_occupancy_volume += occupancy * static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
    }
    d_pool_occupancy->sync_ghosts();
    scalar_type global_occupancy_volume{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_occupancy_volume,
        &global_occupancy_volume);
    d_pool_occupancy_volume_error = global_occupancy_volume - static_cast<scalar_type>(diagnostics.pool_volume);
}

/** @brief Append one accepted, already reduced free-surface history record. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::record_free_surface_history()
{
    if (!d_free_surface_model || !d_free_surface_model->initialized() || !d_liquid_mass_inventory ||
        !d_liquid_mass_inventory->initialized())
    {
        throw std::logic_error("Free-surface history requires initialized model and liquid inventory state.");
    }
    FreeSurfaceHistoryRecord record;
    record.free_surface = d_free_surface_model->diagnostics();
    record.liquid_mass = d_liquid_mass_inventory->diagnostics();
    record.pool_occupancy_volume_error = d_pool_occupancy_volume_error;
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() && d_radiolytic_gas_model->supplies_void_fraction())
    {
        auto population_moles = [&record](const std::string& population, const std::string& species)
        {
            const auto population_iterator = record.free_surface.submerged_population_gas_moles.find(population);
            if (population_iterator == record.free_surface.submerged_population_gas_moles.end())
            {
                return real_t{};
            }
            return free_surface_species_moles(population_iterator->second, species);
        };
        record.microbubble_hydrogen_moles = population_moles("microbubble", "H2");
        record.large_bubble_hydrogen_moles = population_moles("largeBubble", "H2");
    }
    if (d_boiling_source_model && d_boiling_source_model->enabled())
    {
        record.boiling = d_boiling_source_model->last_phase_change_diagnostics();
    }
    if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE && d_planar_ale_diagnostics.initialized)
    {
        record.planar_ale = d_planar_ale_diagnostics;
    }
    d_free_surface_history.push_back(std::move(record));
}

/**
 * @brief Create the default scalar void-fraction model when absent.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::ensure_scalar_void_fraction_model()
{
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model =
            std::make_unique<scalar_void_fraction_model_type>(d_mesh, ScalarVoidFractionOptions{});
    }
}

/**
 * @brief Select the authoritative gas void-fraction field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active gas fraction field, or nullptr when no model supplies one.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::active_alpha_g_field() const noexcept -> const field_type*
{
    if (d_scalar_void_fraction_model)
    {
        return &d_scalar_void_fraction_model->alpha_g();
    }
    if (d_radiolytic_gas_model)
    {
        return &d_radiolytic_gas_model->alpha_g();
    }
    return nullptr;
}

/**
 * @brief Select the authoritative liquid fraction field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active liquid fraction field, or nullptr when no model supplies one.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::active_alpha_l_field() const noexcept -> const field_type*
{
    if (d_scalar_void_fraction_model)
    {
        return &d_scalar_void_fraction_model->alpha_l();
    }
    if (d_radiolytic_gas_model)
    {
        return &d_radiolytic_gas_model->alpha_l();
    }
    return nullptr;
}

/**
 * @brief Advance or mirror the scalar void-fraction model for one step.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::update_void_fraction_models(scalar_type time_step)
{
    if (!d_scalar_void_fraction_model)
    {
        return;
    }

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->supplies_void_fraction())
    {
        d_scalar_void_fraction_model->mirror(d_radiolytic_gas_model->alpha_g(), time_step);
        return;
    }

    d_scalar_void_fraction_model->update_explicit(time_step,
        d_radiolytic_gas_model ? &d_radiolytic_gas_model->source_alpha_rad() : nullptr,
        d_boiling_source_model ? &d_boiling_source_model->source_alpha_boil() : nullptr);

    // This is the sole accepted steam-inventory completion point.  It must
    // immediately follow the scalar explicit update so the same accepted
    // source and collapse are used for liquid-mass accounting.
    if (d_boiling_source_model)
    {
        d_boiling_source_model->complete_void_fraction_update(
            time_step, *d_scalar_void_fraction_model, d_free_surface_model != nullptr);
    }

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
    {
        d_radiolytic_gas_model->synchronize_void_fraction(
            d_scalar_void_fraction_model->alpha_g(), d_scalar_void_fraction_model->options().alpha_max);
    }
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
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::refresh_geometry_dependent_state()
{
    boussinesq_momentum_equation().refresh_geometry();
    temperature_equation().refresh_geometry();
    if (uses_legacy_backend())
    {
        boussinesq_pressure_face_flux_workspace().refresh_geometry();
        boussinesq_coupled_pressure_velocity_solver().clear_cache();
    }
    this->refresh_pressure_velocity_geometry_state();
    if (d_radiolytic_gas_model)
    {
        d_radiolytic_gas_model->refresh_geometry();
    }
    if (d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model->refresh_geometry();
    }
    if (d_liquid_mass_inventory)
    {
        d_liquid_mass_inventory->refresh_geometry();
    }
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
    const face_flux_field_type* momentum_flux = &old_face_fluxes();
    if (d_active_ale)
    {
        form_mesh_relative_flux(old_face_fluxes());
        momentum_flux = d_mesh_relative_face_flux.get();
    }
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
            d_problem.linear_options(), turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
            turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr,
            d_active_ale ? &*d_active_ale : nullptr);
    }
    return boussinesq_momentum_equation().advance_velocity(velocity(), *momentum_flux, temperature(),
        boussinesq_velocity_boundary_cache(), d_problem.time_options(), velocity(), pressure_source,
        d_problem.linear_options(), d_active_ale ? &*d_active_ale : nullptr);
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

/**
 * @brief Assemble the coupled Boussinesq velocity-pressure system.
 *
 * @tparam Pack Tpetra type pack.
 * @return Monolithic coupled system with optional material and turbulence terms.
 */
template<TpetraTypePack Pack> auto BoussinesqSolver<Pack>::assemble_coupled_system() -> coupled_system_type
{
    const auto* turbulence = find_turbulence_model();
    if (d_active_ale)
    {
        form_mesh_relative_flux(old_face_fluxes());
        if (const auto* target = volume_continuity_target())
        {
            return boussinesq_coupled_pressure_velocity_solver().assemble(boussinesq_momentum_equation(), velocity(),
                pressure(), temperature(), *d_mesh_relative_face_flux, boussinesq_velocity_boundary_cache(),
                d_problem.boundary_conditions(), d_problem.time_options(), *target,
                physical_transport_enabled() ? &stored_material_properties() : nullptr,
                d_model_options.reference_density, d_model_options.density_feedback_enabled,
                turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
                turbulence != nullptr ? &turbulence->turbulent_kinetic_energy_gradient() : nullptr,
                turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr,
                &*d_active_ale);
        }
        throw std::logic_error("planarALE coupled assembly requires an active volume-continuity target.");
    }
    return boussinesq_coupled_pressure_velocity_solver().assemble(boussinesq_momentum_equation(), velocity(),
        pressure(), temperature(), old_face_fluxes(), boussinesq_velocity_boundary_cache(),
        d_problem.boundary_conditions(), d_problem.time_options(),
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
    const bool pending_free_surface = d_free_surface_model && !d_free_surface_model->initialized();
    if (!pending_free_surface)
    {
        if (d_free_surface_model && d_material_feedback_model)
        {
            refresh_material_feedback(d_time);
        }
        initialize_radiolytic_gas_state(true);
    }
    initialize_free_surface_if_needed();
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
 * @brief Reject incompatible cross-model selections before a step starts.
 *
 * Field availability remains a phase-local validation because authoritative
 * fraction fields may be initialized or updated earlier in the same step.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::validate_step_coupling() const
{
    validate_collective_model_state();
    if (d_free_surface_step_failed)
    {
        throw std::logic_error(
            "A prior step failed after upstream multiphysics state advanced; reconstruct the solver from its last "
            "accepted state before retrying.");
    }
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

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
        d_radiolytic_gas_model->supplies_void_fraction() && d_boiling_source_model && d_boiling_source_model->enabled())
    {
        throw std::logic_error("Sheng two-population radiolysis cannot advance with boiling "
                               "until vapor mass is coupled to bubble inventories.");
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
        d_radiolytic_gas_model->supplies_void_fraction() && d_scalar_void_fraction_model &&
        (d_scalar_void_fraction_model->options().alpha_min != d_radiolytic_gas_model->options().alpha_min ||
            d_scalar_void_fraction_model->options().alpha_max != d_radiolytic_gas_model->options().alpha_max))
    {
        throw std::logic_error("Sheng radiolysis cannot advance with mismatched scalar mirror "
                               "void-fraction bounds.");
    }
    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() &&
        d_radiolytic_gas_model->supplies_void_fraction() && d_scalar_void_fraction_model &&
        std::isfinite(d_scalar_void_fraction_model->options().alpha_collapse_time))
    {
        throw std::logic_error("Sheng radiolysis cannot advance with scalar void collapse "
                               "until bubble inventory removal is coupled conservatively.");
    }
    if (d_free_surface_model)
    {
        validate_free_surface_configuration(d_free_surface_options);
    }
}

/**
 * @brief Advance the active turbulence equations after flow coupling.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size captured after flow coupling.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::advance_turbulence(scalar_type time_step)
{
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
 * @brief Advance optional sources whose state is needed by temperature transport.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 * @return true when Sheng radiolysis must be advanced after temperature.
 */
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::advance_pre_temperature_models(scalar_type time_step)
{
    const auto sheng_after_temperature =
        d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() && d_radiolytic_gas_model->supplies_void_fraction();

    if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled() && !sheng_after_temperature)
    {
        if (!d_scalar_void_fraction_model)
        {
            throw std::runtime_error("Ideal radiolysis requires the authoritative scalar void model.");
        }
        d_radiolytic_gas_model->advance(d_time + time_step, time_step, temperature(), pressure(), velocity(),
            projected_face_fluxes(), stored_material_properties(),
            d_fission_power_source ? &d_fission_power_source->field() : nullptr,
            d_scalar_void_fraction_model->alpha_g(), d_scalar_void_fraction_model->options().alpha_max);
    }

    if (!sheng_after_temperature)
    {
        if (d_boiling_source_model)
        {
            if (!d_scalar_void_fraction_model)
            {
                throw std::runtime_error("Boiling requires the authoritative scalar void model.");
            }
            d_boiling_source_model->update(time_step, temperature(), stored_material_properties(),
                *d_scalar_void_fraction_model,
                d_radiolytic_gas_model && d_radiolytic_gas_model->enabled()
                    ? &d_radiolytic_gas_model->source_alpha_rad()
                    : nullptr);
        }
        update_void_fraction_models(time_step);
    }

    return sheng_after_temperature;
}

/**
 * @brief Advance temperature, record its solve, and synchronize its ghosts.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::advance_temperature_transport(scalar_type time_step)
{
    LinearSolveStatistics temperature_statistics;
    if (physical_transport_enabled())
    {
        const auto* turbulence = find_turbulence_model();
        auto total_power_density = [&](local_ordinal_type cell_lid)
        {
            auto total = stored_temperature_sources().total_power_density(cell_lid);
            if (d_boiling_source_model)
            {
                total += d_boiling_source_model->temperature_source(cell_lid);
            }
            return total;
        };
        const auto& transport_flux = d_active_ale ? *d_mesh_relative_face_flux : projected_face_fluxes();
        temperature_statistics = temperature_equation().advance_physical(temperature(), transport_flux,
            time_step, stored_material_properties(), temperature(), total_power_density,
            d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options(),
            turbulence != nullptr ? &turbulence->effective_thermal_conductivity() : nullptr,
            turbulence != nullptr ? turbulence->effective_thermal_conductivity_boundary_cache() : nullptr,
            d_problem.time_options().coefficient_interpolation, d_active_ale ? &*d_active_ale : nullptr,
            d_active_ale && d_liquid_mass_inventory ? &d_liquid_mass_inventory->cellMassInventory() : nullptr,
            d_active_ale ? d_ale_old_heat_capacity.get() : nullptr,
            d_active_ale ? d_ale_temperature_density : nullptr);
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

/**
 * @brief Advance temperature-dependent models and refresh transport fields.
 *
 * Required authoritative fields are checked here, after upstream models have
 * had their normal opportunity to publish them.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 * @param sheng_after_temperature Whether Sheng radiolysis advances in this phase.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::advance_post_temperature_models(scalar_type time_step, bool sheng_after_temperature)
{
    if (sheng_after_temperature)
    {
        d_radiolytic_gas_model->advance(d_time + time_step, time_step, temperature(), pressure(), velocity(),
            projected_face_fluxes(), stored_material_properties(),
            d_fission_power_source ? &d_fission_power_source->field() : nullptr);
        update_void_fraction_models(time_step);
    }

    if (d_precursor_model)
    {
        const auto* alpha_l = active_alpha_l_field();
        if (alpha_l == nullptr)
        {
            throw std::runtime_error("Precursor model requires a liquid fraction field.");
        }
        d_precursor_model->advance(time_step, *alpha_l,
            d_fission_power_source ? &d_fission_power_source->field() : nullptr, &projected_face_fluxes());
    }

    refresh_material_feedback(d_time + time_step);
    advance_free_surface(time_step);
    if (auto* turbulence = find_turbulence_model())
    {
        turbulence->refresh_effective_properties(stored_material_properties(), d_model_options.reference_density);
    }
}

/** @brief Execute one rollback-safe solver-integrated planar-ALE step. */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::step_planar_ale()
{
    d_ale_temperature_density = nullptr;
    if (d_step_index == 0)
    {
        temperature().sync_ghosts();
    }
    if (d_physical_model_enabled)
    {
        refresh_physical_models();
    }
    initialize_free_surface_if_needed(true, d_physical_model_enabled);
    initialize_planar_ale_if_needed();
    if (!d_ale_motion || !d_ale_boundary || !d_volume_continuity_model || !d_mesh_relative_face_flux ||
        !d_bubble_slip_volume_flux || !d_mesh_volume_rate || !d_continuity_residual || !d_ale_old_density ||
        !d_ale_old_heat_capacity)
    {
        throw std::logic_error("planarALE did not initialize its geometry, flux, and volume-source owners.");
    }

    // Everything below this line is either restored from these snapshots or
    // committed together with the one active geometry trial.
    FieldStateSnapshot pressure_snapshot(pressure());
    FieldStateSnapshot pressure_correction_snapshot(this->pressure_correction());
    FieldStateSnapshot velocity_snapshot(velocity());
    FieldStateSnapshot predictor_gradient_snapshot(predictor_pressure_gradient());
    FieldStateSnapshot predictor_velocity_snapshot(this->predictor_velocity());
    FieldStateSnapshot old_flux_snapshot(old_face_fluxes());
    FieldStateSnapshot projected_flux_snapshot(projected_face_fluxes());
    FieldStateSnapshot relative_flux_snapshot(*d_mesh_relative_face_flux);
    FieldStateSnapshot bubble_slip_flux_snapshot(*d_bubble_slip_volume_flux);
    FieldStateSnapshot mesh_volume_rate_snapshot(*d_mesh_volume_rate);
    FieldStateSnapshot temperature_snapshot(temperature());
    FieldStateSnapshot clear_level_snapshot(*d_clear_level);
    FieldStateSnapshot pool_level_snapshot(*d_pool_level);
    FieldStateSnapshot headspace_pressure_snapshot(*d_headspace_pressure);
    FieldStateSnapshot occupancy_snapshot(*d_pool_occupancy);
    FieldStateSnapshot continuity_residual_snapshot(*d_continuity_residual);
    const auto material_snapshot = stored_material_properties().snapshot();
    std::optional<typename material_feedback_model_type::StateSnapshot> material_feedback_snapshot;
    if (d_material_feedback_model)
    {
        material_feedback_snapshot.emplace(d_material_feedback_model->snapshot());
    }
    const auto liquid_snapshot = d_liquid_mass_inventory->snapshot();
    const auto free_surface_snapshot = d_free_surface_model->snapshot();
    const auto volume_model_snapshot = d_volume_continuity_model->snapshot();
    std::optional<typename radiolytic_gas_model_type::StateSnapshot> radiolytic_snapshot;
    if (d_radiolytic_gas_model)
    {
        radiolytic_snapshot.emplace(d_radiolytic_gas_model->snapshot());
    }
    std::optional<typename scalar_void_fraction_model_type::StateSnapshot> void_snapshot;
    if (d_scalar_void_fraction_model)
    {
        void_snapshot.emplace(d_scalar_void_fraction_model->snapshot());
    }
    const auto accepted_statistics = d_last_step_statistics;
    const auto accepted_residuals = pressure_velocity_residuals();
    const auto accepted_volume_residuals = this->last_volume_continuity_residuals();
    const auto accepted_time = d_time;
    const auto accepted_step = d_step_index;
    const auto accepted_history_size = d_free_surface_history.size();
    const auto accepted_ale_diagnostics = d_planar_ale_diagnostics;
    const auto accepted_occupancy_volume_error = d_pool_occupancy_volume_error;

    std::vector<real_t> accepted_cell_volumes(d_mesh->num_local_cells());
    for (size_t local = 0; local < accepted_cell_volumes.size(); ++local)
    {
        accepted_cell_volumes[local] = d_mesh->cell_volume(static_cast<local_ordinal_type>(local));
    }
    scalar_type local_accepted_mesh_volume{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        local_accepted_mesh_volume += static_cast<scalar_type>(accepted_cell_volumes[owned]);
    }
    scalar_type accepted_mesh_volume{};
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
        1, &local_accepted_mesh_volume, &accepted_mesh_volume);
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        d_ale_old_density->set_owned_value(cell, stored_material_properties().density.value(cell));
        d_ale_old_heat_capacity->set_owned_value(cell,
            stored_material_properties().specific_heat_capacity.value(cell));
    }
    d_ale_old_density->sync_ghosts();
    d_ale_old_heat_capacity->sync_ghosts();
    const auto old_material_volume = material_volumes(d_liquid_mass_inventory->cellMassInventory(),
        d_liquid_mass_inventory->pureLiquidDensity(), accepted_cell_volumes);
    const auto old_energy = total_sensible_energy(d_liquid_mass_inventory->cellMassInventory(),
        *d_ale_old_heat_capacity, accepted_cell_volumes);
    const auto accepted_surface = d_free_surface_model->diagnostics();
    const auto time_step = static_cast<scalar_type>(d_problem.time_options().time_step);

    auto restore_accepted = [&]
    {
        clear_volume_continuity_target();
        clear_ale_pressure_boundary();
        d_ale_temperature_density = nullptr;
        d_active_ale.reset();
        if (d_ale_motion->has_active_trial())
        {
            d_ale_motion->rollback_trial();
        }
        refresh_geometry_dependent_state();
        d_ale_boundary->refresh(d_free_surface_model->volumeMap());
        pressure_snapshot.restore(pressure());
        pressure_correction_snapshot.restore(this->pressure_correction());
        velocity_snapshot.restore(velocity());
        predictor_gradient_snapshot.restore(predictor_pressure_gradient());
        predictor_velocity_snapshot.restore(this->predictor_velocity());
        old_flux_snapshot.restore(old_face_fluxes());
        projected_flux_snapshot.restore(projected_face_fluxes());
        relative_flux_snapshot.restore(*d_mesh_relative_face_flux);
        bubble_slip_flux_snapshot.restore(*d_bubble_slip_volume_flux);
        mesh_volume_rate_snapshot.restore(*d_mesh_volume_rate);
        temperature_snapshot.restore(temperature());
        clear_level_snapshot.restore(*d_clear_level);
        pool_level_snapshot.restore(*d_pool_level);
        headspace_pressure_snapshot.restore(*d_headspace_pressure);
        occupancy_snapshot.restore(*d_pool_occupancy);
        continuity_residual_snapshot.restore(*d_continuity_residual);
        stored_material_properties().restore(material_snapshot);
        if (material_feedback_snapshot)
        {
            d_material_feedback_model->restore(*material_feedback_snapshot);
        }
        d_liquid_mass_inventory->restore(liquid_snapshot);
        d_free_surface_model->restore(free_surface_snapshot);
        d_volume_continuity_model->restore(volume_model_snapshot);
        if (radiolytic_snapshot)
        {
            d_radiolytic_gas_model->restore(*radiolytic_snapshot);
        }
        if (void_snapshot)
        {
            d_scalar_void_fraction_model->restore(*void_snapshot);
        }
        d_last_step_statistics = accepted_statistics;
        pressure_velocity_residuals() = accepted_residuals;
        this->d_last_volume_continuity_residuals = accepted_volume_residuals;
        d_time = accepted_time;
        d_step_index = accepted_step;
        d_free_surface_history.resize(accepted_history_size);
        d_planar_ale_diagnostics = accepted_ale_diagnostics;
        d_pool_occupancy_volume_error = accepted_occupancy_volume_error;
    };

    std::vector<scalar_type> target_guess(d_mesh->num_owned_cells(), scalar_type{});
    std::vector<scalar_type> density_guess(d_mesh->num_owned_cells());
    std::vector<scalar_type> heat_capacity_guess(d_mesh->num_owned_cells());
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        density_guess[owned] = d_ale_old_density->value(cell);
        heat_capacity_guess[owned] = d_ale_old_heat_capacity->value(cell);
    }
    auto candidate_level = static_cast<scalar_type>(accepted_surface.pool_level);
    if (d_volume_continuity_model->generation() != 0)
    {
        for (size_t owned = 0; owned < target_guess.size(); ++owned)
        {
            target_guess[owned] = d_volume_continuity_model->continuity_target_field().value(
                static_cast<local_ordinal_type>(owned));
        }
        candidate_level += time_step * static_cast<scalar_type>(accepted_surface.pool_level_rate);
    }

    scalar_type last_level_residual{};
    scalar_type last_target_change{};
    scalar_type last_continuity_maximum{};
    scalar_type last_level_tolerance{};
    scalar_type last_target_tolerance{};
    scalar_type last_pool_volume_mismatch{};
    scalar_type last_pool_volume_tolerance{};
    scalar_type last_material_state_residual{};
    scalar_type last_gas_state_residual{};
    constexpr scalar_type state_relative_tolerance{1.0e-10};
    std::vector<scalar_type> previous_gas_state;
    std::vector<scalar_type> level_residual_history;
    std::vector<scalar_type> target_change_history;
    std::vector<scalar_type> continuity_maximum_history;
    std::vector<scalar_type> material_state_residual_history;
    std::vector<scalar_type> gas_state_residual_history;
    try
    {
        for (int corrector = 1; corrector <= d_free_surface_options.ale.maximum_correctors; ++corrector)
        {
            d_ale_motion->begin_trial(candidate_level, time_step);
            d_active_ale.emplace(FVM::make_ale_control_volume_state(*d_mesh, *d_ale_motion));
            refresh_geometry_dependent_state();
            // Picard data from the preceding outer trial supplies the
            // new-time rho/cp coefficients. Accepted-old coefficients remain
            // in d_ale_old_* for the conservative transient RHS.
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                stored_material_properties().density.set_owned_value(cell, density_guess[owned]);
                stored_material_properties().specific_heat_capacity.set_owned_value(
                    cell, heat_capacity_guess[owned]);
            }
            stored_material_properties().density.sync_ghosts();
            stored_material_properties().specific_heat_capacity.sync_ghosts();
            d_ale_boundary->refresh(d_free_surface_model->volumeMap());
            d_ale_boundary->apply_kinematic_velocity(*d_active_ale, boussinesq_velocity_boundary_cache());

            if (d_ale_target_generation == std::numeric_limits<std::uint64_t>::max())
            {
                throw std::overflow_error("planarALE continuity-target generation exhausted.");
            }
            const auto solve_generation = ++d_ale_target_generation;
            const continuity_target_type solve_target(d_mesh, target_guess, solve_generation);
            set_volume_continuity_target(solve_target);
            configure_ale_pressure_boundary(solve_generation);
            solve_pressure_velocity_coupling();
            scalar_type local_solve_target_scale{};
            for (const auto value : target_guess)
            {
                local_solve_target_scale = std::max(local_solve_target_scale, std::abs(value));
            }
            scalar_type solve_target_scale{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
                1, &local_solve_target_scale, &solve_target_scale);
            const auto strict_pressure_tolerance =
                static_cast<scalar_type>(d_free_surface_options.coupling.volume_absolute_tolerance) / time_step +
                static_cast<scalar_type>(d_free_surface_options.coupling.volume_relative_tolerance) *
                    std::max(scalar_type{1}, solve_target_scale);
            refine_volume_continuity(
                d_free_surface_options.ale.maximum_correctors, strict_pressure_tolerance);
            form_mesh_relative_flux(projected_face_fluxes());

            // Temperature stores sensible energy in the transported liquid,
            // not in the bubble-displaced mixture volume.  Build the
            // conservative trial liquid-mass density on the candidate mesh
            // before assembling rho*cp*T at the new time level.
            const auto temperature_mass_preview = d_liquid_mass_inventory->previewCellwiseAdvance(time_step,
                *d_mesh_relative_face_flux, nullptr, nullptr, d_problem.linear_options(), &*d_active_ale);
            d_ale_temperature_density =
                &d_liquid_mass_inventory->trialCellMassInventory(temperature_mass_preview);
            if (temperature_mass_preview.transportStatistics())
            {
                d_last_step_statistics.add(*temperature_mass_preview.transportStatistics());
            }
            try
            {
                advance_temperature_transport(time_step);
            }
            catch (...)
            {
                d_ale_temperature_density = nullptr;
                throw;
            }
            d_ale_temperature_density = nullptr;
            if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
            {
                const auto pressure_mode = d_radiolytic_gas_model->options().pressure_mode;
                auto offset = static_cast<scalar_type>(d_free_surface_model->headspacePressure());
                if (pressure_mode == RadiolyticPressureMode::Reconstructed)
                {
                    // The radiolytic reconstruction subtracts the gauge mean;
                    // this offset therefore makes p_abs=p_h on the p_gauge=0 top.
                    offset += volume_weighted_mean_pressure();
                }
                d_radiolytic_gas_model->set_absolute_pressure_offset(offset);
                d_radiolytic_gas_model->advance(d_time + time_step, time_step, temperature(), pressure(), velocity(),
                    *d_mesh_relative_face_flux, stored_material_properties(),
                    d_fission_power_source ? &d_fission_power_source->field() : nullptr, &*d_active_ale,
                    d_free_surface_options.gravity_axis);
                update_void_fraction_models(time_step);
            }

            refresh_material_feedback(d_time + time_step);
            std::vector<scalar_type> next_density_guess(d_mesh->num_owned_cells());
            std::vector<scalar_type> next_heat_capacity_guess(d_mesh->num_owned_cells());
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                next_density_guess[owned] = stored_material_properties().density.value(cell);
                next_heat_capacity_guess[owned] =
                    stored_material_properties().specific_heat_capacity.value(cell);
            }
            scalar_type local_material_state_residual{};
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto density_scale = std::max({scalar_type{1},
                    std::abs(density_guess[owned]),
                    std::abs(next_density_guess[owned])});
                const auto heat_capacity_scale = std::max({scalar_type{1},
                    std::abs(heat_capacity_guess[owned]),
                    std::abs(next_heat_capacity_guess[owned])});
                local_material_state_residual = std::max(
                    local_material_state_residual,
                    std::max(
                        std::abs(next_density_guess[owned] - density_guess[owned]) /
                            density_scale,
                        std::abs(next_heat_capacity_guess[owned] -
                            heat_capacity_guess[owned]) /
                            heat_capacity_scale));
            }
            scalar_type material_state_residual{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(),
                Teuchos::REDUCE_MAX, 1, &local_material_state_residual,
                &material_state_residual);

            std::vector<scalar_type> current_gas_state;
            scalar_type local_gas_state_residual{};
            if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
            {
                const std::array<const field_type*, 5> gas_fields{
                    &d_radiolytic_gas_model->dissolved_hydrogen_inventory(),
                    &d_radiolytic_gas_model->micro_number_density(),
                    &d_radiolytic_gas_model->micro_moles(),
                    &d_radiolytic_gas_model->large_number_density(),
                    &d_radiolytic_gas_model->large_moles()};
                current_gas_state.reserve(
                    gas_fields.size() * d_mesh->num_owned_cells());
                for (const auto* field : gas_fields)
                {
                    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                    {
                        current_gas_state.push_back(field->value(
                            static_cast<local_ordinal_type>(owned)));
                    }
                }
                if (previous_gas_state.size() != current_gas_state.size())
                {
                    local_gas_state_residual = scalar_type{1};
                }
                else
                {
                    for (size_t entry = 0; entry < current_gas_state.size(); ++entry)
                    {
                        const auto scale = std::max({scalar_type{1},
                            std::abs(previous_gas_state[entry]),
                            std::abs(current_gas_state[entry])});
                        local_gas_state_residual = std::max(
                            local_gas_state_residual,
                            std::abs(current_gas_state[entry] -
                                previous_gas_state[entry]) /
                                scale);
                    }
                }
            }
            scalar_type gas_state_residual{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(),
                Teuchos::REDUCE_MAX, 1, &local_gas_state_residual,
                &gas_state_residual);
            d_liquid_mass_inventory->updatePureLiquidDensity(
                [this](local_ordinal_type cell) { return pure_liquid_density(cell); });
            const auto liquid_preview = d_liquid_mass_inventory->previewCellwiseAdvance(time_step,
                *d_mesh_relative_face_flux, nullptr, nullptr, d_problem.linear_options(), &*d_active_ale);
            FreeSurfaceAccountingPreview accounting;
            accounting.liquid = liquid_preview.diagnostics();

            scalar_type gas_inventory_residual{};
            if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
            {
                const auto& gas_statistics =
                    d_radiolytic_gas_model->last_statistics();
                gas_inventory_residual = gas_statistics.inventory_error;
                const auto gas_scale = std::abs(gas_statistics.hydrogen_before) +
                    std::abs(gas_statistics.hydrogen_produced) +
                    std::abs(gas_statistics.hydrogen_escaped) +
                    std::abs(gas_statistics.hydrogen_after);
                const auto gas_tolerance = static_cast<scalar_type>(
                    d_free_surface_options.coupling.gas_absolute_tolerance) +
                    static_cast<scalar_type>(
                        d_free_surface_options.coupling.gas_relative_tolerance) *
                        gas_scale;
                if (std::abs(gas_inventory_residual) > gas_tolerance)
                {
                    std::ostringstream message;
                    message << std::scientific
                            << std::setprecision(
                                   std::numeric_limits<scalar_type>::max_digits10)
                            << "planarALE radiolytic H2 step-inventory closure exceeded its physical tolerance: residual="
                            << gas_inventory_residual << " mol, tolerance="
                            << gas_tolerance << " mol.";
                    throw std::runtime_error(message.str());
                }
            }
            const auto surface_preview = d_free_surface_model->previewUpdate(
                make_free_surface_update(d_time + time_step, false, &accounting));

            const auto& trial_mass = d_liquid_mass_inventory->trialCellMassInventory(liquid_preview);
            if (d_radiolytic_gas_model && d_radiolytic_gas_model->enabled())
            {
                const auto& transported_slip =
                    d_radiolytic_gas_model->transported_bubble_slip_volume_flux();
                for (const auto face_lid : d_bubble_slip_volume_flux->owned_face_ids())
                {
                    d_bubble_slip_volume_flux->set_owned_value(
                        face_lid, transported_slip.value(face_lid));
                }
                d_bubble_slip_volume_flux->sync_ghosts();
            }
            else
            {
                d_bubble_slip_volume_flux->put_scalar(scalar_type{});
                d_bubble_slip_volume_flux->sync_ghosts();
            }

            scalar_type local_escape_volume_rate{};
            for (const auto face_lid : d_bubble_slip_volume_flux->owned_face_ids())
            {
                if (d_ale_boundary->contains(face_lid))
                {
                    local_escape_volume_rate += std::max(
                        d_bubble_slip_volume_flux->value(face_lid), scalar_type{});
                }
            }
            scalar_type escape_volume_rate{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
                1, &local_escape_volume_rate, &escape_volume_rate);

            const auto new_material_volume = material_volumes(trial_mass,
                d_liquid_mass_inventory->pureLiquidDensity(), d_active_ale->new_cell_volumes());
            face_flux_field_type carrier_material_volume_flux(
                d_mesh, 0.0, "carrierMaterialVolumeFlux");
            const auto* transported_bubble_carrier =
                d_radiolytic_gas_model && d_radiolytic_gas_model->enabled()
                ? &d_radiolytic_gas_model->transported_bubble_carrier_volume_flux()
                : nullptr;
            for (const auto face_lid : carrier_material_volume_flux.owned_face_ids())
            {
                const auto carrier_flux =
                    d_mesh_relative_face_flux->value(face_lid);
                const auto owner = d_mesh->owner_cell(face_lid);
                auto upwind = owner;
                if (carrier_flux < scalar_type{} &&
                    d_mesh->is_interior_face(face_lid))
                {
                    upwind = d_mesh->opposite_or_periodic_neighbor_cell(
                        face_lid, owner);
                }
                const auto liquid_fraction =
                    trial_mass.local_value(upwind) /
                    d_liquid_mass_inventory->pureLiquidDensity().local_value(upwind);
                const auto bubble_carrier_flux =
                    transported_bubble_carrier == nullptr
                    ? scalar_type{}
                    : transported_bubble_carrier->value(face_lid);
                carrier_material_volume_flux.set_owned_value(face_lid,
                    carrier_flux * liquid_fraction + bubble_carrier_flux);
            }
            carrier_material_volume_flux.sync_ghosts();
            if (d_ale_target_generation == std::numeric_limits<std::uint64_t>::max())
            {
                throw std::overflow_error("planarALE continuity-target generation exhausted.");
            }
            const auto ledger_generation = ++d_ale_target_generation;
            const typename volume_continuity_model_type::Inputs volume_inputs{.ale = *d_active_ale,
                .old_material_volume = old_material_volume,
                .new_material_volume = new_material_volume,
                .carrier_relative_flux = *d_mesh_relative_face_flux,
                .carrier_material_volume_flux = &carrier_material_volume_flux,
                .bubble_slip_volume_flux = d_bubble_slip_volume_flux.get(),
                .old_pool_volume = static_cast<scalar_type>(accepted_surface.pool_volume),
                .new_pool_volume = static_cast<scalar_type>(surface_preview.diagnostics().pool_volume),
                .bubble_escape_volume_rate = escape_volume_rate,
                .other_outflow_volume_rate = scalar_type{},
                .previous_target = target_guess};
            const auto volume_trial = d_volume_continuity_model->preview(volume_inputs, ledger_generation);

            const auto new_energy = total_sensible_energy(trial_mass,
                stored_material_properties().specific_heat_capacity, d_active_ale->new_cell_volumes());
            scalar_type local_added_energy{};
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                local_added_energy += static_cast<scalar_type>(d_active_ale->new_cell_volumes()[owned]) * time_step *
                                      stored_temperature_sources().total_power_density(cell);
            }
            scalar_type added_energy{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
                1, &local_added_energy, &added_energy);
            const auto energy_residual = new_energy - old_energy - added_energy;
            const auto energy_scale = std::max({scalar_type{1}, std::abs(new_energy), std::abs(old_energy),
                std::abs(added_energy)});
            const auto energy_tolerance = scalar_type{4096} * std::numeric_limits<scalar_type>::epsilon() *
                                              energy_scale +
                                          scalar_type{1.0e-9} * energy_scale;
            if (std::abs(energy_residual) > energy_tolerance)
            {
                throw std::runtime_error("planarALE sensible-energy closure exceeded its physical tolerance: residual=" +
                                         std::to_string(energy_residual) + " J, tolerance=" +
                                         std::to_string(energy_tolerance) + " J.");
            }

            const auto level_residual = static_cast<scalar_type>(surface_preview.diagnostics().pool_level) -
                                        candidate_level;
            last_level_residual = level_residual;
            const auto level_scale = std::max({scalar_type{1}, std::abs(candidate_level),
                std::abs(static_cast<scalar_type>(surface_preview.diagnostics().pool_level))});
            const auto level_tolerance = static_cast<scalar_type>(d_free_surface_options.ale.level_absolute_tolerance) +
                                         static_cast<scalar_type>(d_free_surface_options.ale.level_relative_tolerance) *
                                             level_scale;
            scalar_type local_target_scale{};
            for (size_t owned = 0; owned < target_guess.size(); ++owned)
            {
                local_target_scale = std::max(local_target_scale,
                    std::max(std::abs(target_guess[owned]),
                        std::abs(volume_trial.target().integrated_rate(static_cast<local_ordinal_type>(owned)))));
            }
            scalar_type target_scale{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
                1, &local_target_scale, &target_scale);
            const auto target_tolerance = static_cast<scalar_type>(d_free_surface_options.coupling.volume_absolute_tolerance) /
                                              time_step +
                                          static_cast<scalar_type>(d_free_surface_options.coupling.volume_relative_tolerance) *
                                              std::max(scalar_type{1}, target_scale);
            last_level_tolerance = level_tolerance;
            last_target_tolerance = target_tolerance;
            scalar_type local_final_l2_squared{};
            scalar_type local_final_normalization_squared{};
            scalar_type local_final_maximum{};
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto balance = FVM::cell_flux_balance<Pack>(*d_mesh, projected_face_fluxes(), cell);
                const auto target = volume_trial.target().integrated_rate(cell);
                const auto residual = balance - target;
                d_continuity_residual->set_owned_value(cell, residual);
                local_final_l2_squared += residual * residual;
                local_final_normalization_squared += std::max(balance * balance, target * target);
                local_final_maximum = std::max(local_final_maximum, std::abs(residual));
            }
            d_continuity_residual->sync_ghosts();
            const std::array<scalar_type, 2> local_continuity_squared{
                local_final_l2_squared, local_final_normalization_squared};
            std::array<scalar_type, 2> global_continuity_squared{};
            scalar_type global_final_maximum{};
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
                static_cast<int>(local_continuity_squared.size()), local_continuity_squared.data(),
                global_continuity_squared.data());
            Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
                1, &local_final_maximum, &global_final_maximum);
            const auto final_l2 = std::sqrt(global_continuity_squared[0]);
            const auto final_normalization = std::sqrt(global_continuity_squared[1]);
            const continuity_residual_type final_continuity{final_l2, global_final_maximum,
                final_normalization > scalar_type{} ? final_l2 / final_normalization : final_l2,
                final_normalization};
            const auto continuity_tolerance = target_tolerance;
            const auto pool_volume_mismatch =
                d_ale_boundary->diagnostics().global_mesh_volume -
                static_cast<scalar_type>(surface_preview.diagnostics().pool_volume);
            const auto pool_volume_scale = std::max({scalar_type{1},
                std::abs(d_ale_boundary->diagnostics().global_mesh_volume),
                std::abs(static_cast<scalar_type>(
                    surface_preview.diagnostics().pool_volume))});
            const auto pool_volume_tolerance =
                static_cast<scalar_type>(
                    d_free_surface_options.coupling.volume_absolute_tolerance) +
                static_cast<scalar_type>(
                    d_free_surface_options.coupling.volume_relative_tolerance) *
                    pool_volume_scale;
            last_target_change = volume_trial.diagnostics().maximum_target_change;
            last_continuity_maximum = final_continuity.maximum;
            last_pool_volume_mismatch = pool_volume_mismatch;
            last_pool_volume_tolerance = pool_volume_tolerance;
            last_material_state_residual = material_state_residual;
            last_gas_state_residual = gas_state_residual;
            level_residual_history.push_back(level_residual);
            target_change_history.push_back(last_target_change);
            continuity_maximum_history.push_back(last_continuity_maximum);
            material_state_residual_history.push_back(material_state_residual);
            gas_state_residual_history.push_back(gas_state_residual);
            const bool converged = std::abs(level_residual) <= level_tolerance &&
                                   volume_trial.diagnostics().maximum_target_change <= target_tolerance &&
                                   final_continuity.maximum <= continuity_tolerance &&
                                   std::abs(pool_volume_mismatch) <= pool_volume_tolerance &&
                                   material_state_residual <= state_relative_tolerance &&
                                   gas_state_residual <= state_relative_tolerance;

            if (converged)
            {
                const auto motion_diagnostics = d_ale_motion->diagnostics();
                for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell = static_cast<local_ordinal_type>(owned);
                    d_mesh_volume_rate->set_owned_value(cell,
                        static_cast<scalar_type>((d_active_ale->new_cell_volumes()[owned] -
                                                    d_active_ale->old_cell_volumes()[owned]) /
                                                time_step));
                }
                d_mesh_volume_rate->sync_ghosts();
                d_free_surface_model->commitUpdate(surface_preview);
                d_liquid_mass_inventory->commitPhaseChange(liquid_preview);
                if (liquid_preview.transportStatistics())
                {
                    d_last_step_statistics.add(*liquid_preview.transportStatistics());
                }
                d_volume_continuity_model->commit(volume_trial);
                this->d_last_volume_continuity_residuals = final_continuity;
                pressure_velocity_residuals().continuity = final_continuity.l2;

                PlanarALEStepDiagnostics accepted_diagnostics;
                accepted_diagnostics.initialized = true;
                accepted_diagnostics.old_geometry_epoch = motion_diagnostics.old_geometry_epoch;
                accepted_diagnostics.new_geometry_epoch = motion_diagnostics.new_geometry_epoch;
                accepted_diagnostics.old_mesh_volume = accepted_mesh_volume;
                accepted_diagnostics.new_mesh_volume = d_ale_boundary->diagnostics().global_mesh_volume;
                accepted_diagnostics.mesh_vessel_mismatch = pool_volume_mismatch;
                accepted_diagnostics.maximum_gcl_residual = motion_diagnostics.maximum_absolute_gcl_residual;
                accepted_diagnostics.maximum_normalized_gcl_residual =
                    motion_diagnostics.maximum_normalized_gcl_residual;
                accepted_diagnostics.mesh_quality = motion_diagnostics.mesh_quality;
                accepted_diagnostics.outer_correctors = corrector;
                accepted_diagnostics.level_residual = level_residual;
                accepted_diagnostics.pressure_residual = static_cast<scalar_type>(
                    surface_preview.headspacePressure() - d_free_surface_model->headspacePressure());
                accepted_diagnostics.material_state_residual = material_state_residual;
                accepted_diagnostics.gas_state_residual = gas_state_residual;
                accepted_diagnostics.energy_residual = energy_residual;
                accepted_diagnostics.normalized_energy_residual = energy_residual / energy_scale;
                accepted_diagnostics.liquid_mass_residual = liquid_preview.diagnostics().step_mass_balance_residual;
                accepted_diagnostics.gas_inventory_residual = gas_inventory_residual;
                accepted_diagnostics.volume_source = volume_trial.diagnostics();
                accepted_diagnostics.continuity = final_continuity;
                accepted_diagnostics.level_residual_history = std::move(level_residual_history);
                accepted_diagnostics.target_change_history = std::move(target_change_history);
                accepted_diagnostics.continuity_maximum_history = std::move(continuity_maximum_history);
                accepted_diagnostics.material_state_residual_history =
                    std::move(material_state_residual_history);
                accepted_diagnostics.gas_state_residual_history =
                    std::move(gas_state_residual_history);
                accepted_diagnostics.rejected_transactions = accepted_ale_diagnostics.rejected_transactions;
                d_planar_ale_diagnostics = std::move(accepted_diagnostics);

                publish_free_surface_fields();
                record_free_surface_history();
                // Leave the geometry trial active until every other
                // potentially throwing accepted-state publication has
                // completed.  A failure above or in finish_step() can then
                // still roll geometry, fields, ledgers, time, and history
                // back together.
                finish_step();
                d_ale_motion->accept_trial();
                d_ale_temperature_density = nullptr;
                d_active_ale.reset();
                clear_volume_continuity_target();
                clear_ale_pressure_boundary();
                return;
            }

            std::vector<scalar_type> next_target(target_guess.size());
            const auto relaxation = static_cast<scalar_type>(d_free_surface_options.ale.relaxation);
            for (size_t owned = 0; owned < next_target.size(); ++owned)
            {
                next_target[owned] = target_guess[owned] + relaxation *
                    (volume_trial.target().integrated_rate(static_cast<local_ordinal_type>(owned)) -
                        target_guess[owned]);
            }
            const auto next_level = candidate_level + relaxation * level_residual;
            restore_accepted();
            target_guess = std::move(next_target);
            density_guess = std::move(next_density_guess);
            heat_capacity_guess = std::move(next_heat_capacity_guess);
            previous_gas_state = std::move(current_gas_state);
            candidate_level = next_level;
        }
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "planarALE outer level/continuity corrector did not converge in "
                << d_free_surface_options.ale.maximum_correctors << " iterations: level residual="
                << last_level_residual << " m (tolerance=" << last_level_tolerance << "), target change="
                << last_target_change << " m^3/s (tolerance=" << last_target_tolerance
                << "), continuity maximum=" << last_continuity_maximum
                << " m^3/s, mesh/pool-volume mismatch="
                << last_pool_volume_mismatch << " m^3 (tolerance="
                << last_pool_volume_tolerance
                << " m^3), material-state residual="
                << last_material_state_residual << " (tolerance="
                << state_relative_tolerance << "), gas-state residual="
                << last_gas_state_residual << " (tolerance="
                << state_relative_tolerance << ").";
        throw std::runtime_error(message.str());
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        std::string reason = "unknown planarALE transaction failure";
        try
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
        catch (const std::exception& error)
        {
            reason = error.what();
        }
        catch (...)
        {
        }
        restore_accepted();
        d_planar_ale_diagnostics.rejected_transactions =
            accepted_ale_diagnostics.rejected_transactions + 1;
        d_planar_ale_diagnostics.last_rejection_reason = std::move(reason);
        std::rethrow_exception(failure);
    }
}

/**
 * @brief Advance the solution by one time step.
 *
 * Performs: face-flux computation, momentum advance, pressure projection,
 * corrected face fluxes, and semi-implicit temperature advance.  Periodic
 * boundary synchronisation is applied at the start of the first step and
 * after the updates.
 *
 * @tparam Pack Tpetra type pack.
 * @throws std::logic_error if incompatible Sheng radiolysis coupling is active.
 * @throws std::runtime_error if an enabled model lacks its required fraction field.
 */
template<TpetraTypePack Pack> void BoussinesqSolver<Pack>::step()
{
    validate_step_coupling();
    if (d_free_surface_model && d_free_surface_options.mode == FreeSurfaceMode::PlanarALE)
    {
        // begin_step() intentionally clears per-step numerical statistics.
        // Preserve the last accepted report outside the ALE transaction so a
        // rejected later step restores it rather than publishing a cleared
        // or partially accumulated trial report.
        const auto accepted_statistics = d_last_step_statistics;
        const auto accepted_volume_residuals = this->d_last_volume_continuity_residuals;
        begin_step();
        try
        {
            step_planar_ale();
        }
        catch (...)
        {
            d_last_step_statistics = accepted_statistics;
            this->d_last_volume_continuity_residuals = accepted_volume_residuals;
            throw;
        }
        return;
    }
    begin_step();
    const bool free_surface_active = d_free_surface_model != nullptr;
    try
    {
        if (d_step_index == 0)
        {
            temperature().sync_ghosts();
        }
        if (d_physical_model_enabled)
        {
            refresh_physical_models();
        }
        initialize_free_surface_if_needed(true, d_physical_model_enabled);
        if (auto* turbulence = find_turbulence_model())
        {
            turbulence->refresh_effective_properties(stored_material_properties(), d_model_options.reference_density);
        }

        solve_pressure_velocity_coupling();
        const auto time_step = d_problem.time_options().time_step;
        advance_turbulence(time_step);
        const auto sheng_after_temperature = advance_pre_temperature_models(time_step);
        advance_temperature_transport(time_step);
        advance_post_temperature_models(time_step, sheng_after_temperature);

        if (uses_legacy_backend())
        {
            sync_temperature_to_legacy();
        }
        finish_step();
        if (d_free_surface_model)
        {
            record_free_surface_history();
        }
    }
    catch (...)
    {
        if (free_surface_active)
        {
            d_free_surface_step_failed = true;
        }
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
auto BoussinesqSolver<Pack>::solution_writer(const SolutionOutputOptions& output_options) const -> VTUWriter
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
        if (d_material_feedback_model)
        {
            for (const auto& [name, field] : d_material_feedback_model->output_fields())
            {
                writer.add_scalar_cell_data(name, collect_scalar_field(*field));
            }
        }
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
    {
        for (const auto& [name, source] : stored_temperature_sources().entries())
        {
            writer.add_scalar_cell_data(name, collect_scalar_field(source->field()));
        }
        if (d_radiolytic_gas_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_rad", collect_scalar_field(d_radiolytic_gas_model->source_alpha_rad()));
        }
        if (d_boiling_source_model)
        {
            for (const auto& [name, field] : d_boiling_source_model->output_fields())
            {
                writer.add_scalar_cell_data(name, collect_scalar_field(*field));
            }
        }
        if (d_scalar_void_fraction_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total", collect_scalar_field(d_scalar_void_fraction_model->source_alpha_total()));
        }
    }
    if (output_options.include_radiolytic_gas_fields && d_scalar_void_fraction_model)
    {
        writer.add_scalar_cell_data("alpha_g", collect_scalar_field(d_scalar_void_fraction_model->alpha_g()));
        writer.add_scalar_cell_data("alpha_l", collect_scalar_field(d_scalar_void_fraction_model->alpha_l()));
        if (!output_options.include_sources)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total", collect_scalar_field(d_scalar_void_fraction_model->source_alpha_total()));
        }
    }
    if (output_options.include_radiolytic_gas_fields && d_radiolytic_gas_model)
    {
        for (const auto& [name, field] : d_radiolytic_gas_model->output_fields())
        {
            if (name == "S_alpha_rad")
                continue;
            if (d_scalar_void_fraction_model && (name == "alpha_g" || name == "alpha_l"))
            {
                continue;
            }
            writer.add_scalar_cell_data(name, collect_scalar_field(*field));
        }
    }
    if (output_options.include_precursor_fields && d_precursor_model)
    {
        for (const auto& [name, field] : d_precursor_model->output_fields())
        {
            writer.add_scalar_cell_data(name, collect_scalar_field(*field));
        }
    }
    if (output_options.include_free_surface_fields && d_free_surface_model && d_free_surface_model->initialized())
    {
        writer.add_scalar_cell_data("rhoLiquid", collect_scalar_field(rho_liquid()));
        if (d_liquid_mass_inventory->mode() == LiquidVolumeMode::CellMassInventory)
        {
            writer.add_scalar_cell_data(
                "liquidMassInventory", collect_scalar_field(d_liquid_mass_inventory->cellMassInventory()));
        }
        writer.add_scalar_cell_data("clearLevel", collect_scalar_field(clear_level()));
        writer.add_scalar_cell_data("poolLevel", collect_scalar_field(pool_level()));
        writer.add_scalar_cell_data("headspacePressure", collect_scalar_field(headspace_pressure()));
        writer.add_scalar_cell_data("poolOccupancy", collect_scalar_field(pool_occupancy()));
        if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE && d_volume_continuity_model &&
            d_mesh_volume_rate && d_continuity_residual)
        {
            writer.add_scalar_cell_data("meshVolumeRate", collect_scalar_field(*d_mesh_volume_rate));
            writer.add_scalar_cell_data(
                "volumeSourceRate", collect_scalar_field(d_volume_continuity_model->material_source_field()));
            writer.add_scalar_cell_data(
                "bubbleSlipVolumeRate", collect_scalar_field(d_volume_continuity_model->slip_contribution_field()));
            writer.add_scalar_cell_data(
                "continuityTarget", collect_scalar_field(d_volume_continuity_model->continuity_target_field()));
            writer.add_scalar_cell_data("continuityResidual", collect_scalar_field(*d_continuity_residual));
        }
    }
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
                                 (static_cast<int>(output_options.include_radiolytic_gas_fields) << 2) |
                                 (static_cast<int>(output_options.include_precursor_fields) << 3) |
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

} // namespace SimpleFluid
