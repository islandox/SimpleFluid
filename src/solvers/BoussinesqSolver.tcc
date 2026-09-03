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

#include <array>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

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
    : BoussinesqSolver(std::make_shared<MeshHandle<Pack>>(require_mesh(std::move(mesh))),
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
    : BoussinesqSolver(std::make_shared<MeshHandle<Pack>>(require_mesh(std::move(mesh))),
          std::move(boundary_conditions), time_options, linear_options, std::move(model_options), true,
          PhysicalModelTag{})
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
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_radiolytic_gas_model() noexcept
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
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_boiling_source_model() noexcept
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
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_material_feedback_model() noexcept
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

/** @brief Configure optional fixed-grid liquid-level accounting. */
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
    }
    catch (...)
    {
        remove_free_surface_model();
        throw;
    }
    return d_free_surface_model.get();
}

/** @brief Configure fixed-grid liquid-level accounting from a Database. */
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
template<TpetraTypePack Pack> bool BoussinesqSolver<Pack>::remove_free_surface_model() noexcept
{
    const auto removed = static_cast<bool>(d_free_surface_model) || static_cast<bool>(d_liquid_mass_inventory);
    d_free_surface_model.reset();
    d_liquid_mass_inventory.reset();
    d_clear_level.reset();
    d_pool_level.reset();
    d_headspace_pressure.reset();
    d_pool_occupancy.reset();
    d_free_surface_options = {};
    d_pool_occupancy_volume_error = {};
    d_free_surface_history.clear();
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
    output << "time_s,time_step_s,liquid_mass_kg,cumulative_evaporated_mass_kg,"
              "cumulative_condensed_mass_kg,dryout_mass_deficit_kg,liquid_mass_residual_kg,"
              "liquid_mass_residual_normalized,liquid_volume_m3,submerged_bubble_volume_m3,pool_volume_m3,"
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
              "boiling_void_residual_m3,boiling_latent_energy_residual_j\n";
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
               << liquid.normalized_mass_balance_residual << ',' << surface.liquid_volume << ','
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
               << boiling.latent_energy_balance_residual << '\n';
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
            return static_cast<real_t>(d_radiolytic_gas_model->evaluate_submerged_bubble_volume(pressure));
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
            if (d_material_feedback_model && d_radiolytic_gas_model &&
                d_radiolytic_gas_model->supplies_void_fraction())
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
                d_radiolytic_gas_model->set_absolute_pressure_offset(d_free_surface_model->headspacePressure());
            }
        }
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
    if (d_boiling_source_model && d_boiling_source_model->enabled())
    {
        if (d_boiling_source_model->phase_change_completion_pending())
        {
            throw std::logic_error("Free-surface liquid mass cannot advance before boiling "
                                   "scalar-void completion.");
        }
        evaporated_mass = d_boiling_source_model->accepted_evaporation_mass_this_step();
        condensed_mass = d_boiling_source_model->condensed_liquid_mass_this_step();
    }
    d_liquid_mass_inventory->updatePureLiquidDensity(
        [this](local_ordinal_type cell_lid) { return pure_liquid_density(cell_lid); });

    FreeSurfaceAccountingPreview preview;
    const auto current_liquid = d_liquid_mass_inventory->diagnostics();
    const auto phase_change_preview = d_liquid_mass_inventory->previewPhaseChange(evaporated_mass, condensed_mass);
    preview.liquid = phase_change_preview.diagnostics();
    preview.liquid_volume_deficit = (preview.liquid.dryout_mass_deficit - current_liquid.dryout_mass_deficit) *
                                    preview.liquid.mass_weighted_specific_volume;
    d_free_surface_model->update(make_free_surface_update(d_time + time_step, false, &preview));
    d_liquid_mass_inventory->commitPhaseChange(phase_change_preview);
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
        return boussinesq_momentum_equation().advance_velocity_physical(velocity(), old_face_fluxes(), temperature(),
            boussinesq_velocity_boundary_cache(), d_problem.time_options(), stored_material_properties(),
            d_model_options.reference_density, d_model_options.density_feedback_enabled, velocity(), pressure_source,
            d_problem.linear_options(), turbulence != nullptr ? &turbulence->effective_dynamic_viscosity() : nullptr,
            turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr);
    }
    return boussinesq_momentum_equation().advance_velocity(velocity(), old_face_fluxes(), temperature(),
        boussinesq_velocity_boundary_cache(), d_problem.time_options(), velocity(), pressure_source,
        d_problem.linear_options());
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
        temperature_statistics = temperature_equation().advance_physical(temperature(), projected_face_fluxes(),
            time_step, stored_material_properties(), temperature(), total_power_density,
            d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options(),
            turbulence != nullptr ? &turbulence->effective_thermal_conductivity() : nullptr,
            turbulence != nullptr ? turbulence->effective_thermal_conductivity_boundary_cache() : nullptr,
            d_problem.time_options().coefficient_interpolation);
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
        writer.add_scalar_cell_data("clearLevel", collect_scalar_field(clear_level()));
        writer.add_scalar_cell_data("poolLevel", collect_scalar_field(pool_level()));
        writer.add_scalar_cell_data("headspacePressure", collect_scalar_field(headspace_pressure()));
        writer.add_scalar_cell_data("poolOccupancy", collect_scalar_field(pool_occupancy()));
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
