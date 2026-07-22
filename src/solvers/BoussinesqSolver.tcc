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
#include <limits>
#include <optional>

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
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : BoussinesqSolver(
          std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh))),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          BoussinesqModelOptions::legacy_defaults(time_options),
          false,
          PhysicalModelTag{})
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
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    BoussinesqModelOptions model_options)
    : BoussinesqSolver(
          std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh))),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          std::move(model_options),
          true,
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
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : BoussinesqSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          BoussinesqModelOptions::legacy_defaults(time_options),
          false,
          PhysicalModelTag{})
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
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    BoussinesqModelOptions model_options)
    : BoussinesqSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          std::move(model_options),
          true,
          PhysicalModelTag{})
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
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    BoussinesqModelOptions model_options,
    bool physical_model_enabled,
    PhysicalModelTag)
    : base_type(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          typename base_type::DeferredMomentumEquationTag{}),
      d_model_options(std::move(model_options)),
      d_physical_model_enabled(physical_model_enabled)
{
    detail::validate_model_options(
        d_model_options, d_problem.time_options());

    d_problem.template emplace_object<TemperatureDiffusionEquation<Pack>>(
        "temperature_equation",
        d_mesh,
        d_problem.boundary_conditions());
    d_problem.template emplace_object<BoussinesqMomentumEquation<Pack>>(
        "momentum_equation", d_mesh);
    d_problem.template emplace_object<field_type>(
        "temperature", d_mesh, "temperature");
    d_problem.template emplace_object<MaterialPropertyFields<Pack>>(
        "material_properties",
        d_mesh,
        d_model_options,
        d_problem.time_options());
    d_problem.template emplace_object<TurbulenceModel<Pack>>(
        "turbulence_model",
        d_mesh,
        d_problem.boundary_conditions());
    auto& sources =
        d_problem.template emplace_object<
            TemperatureSourceRegistry<Pack>>(
                "temperature_sources", d_mesh);
    for (size_t index = 0;
         index < d_model_options.temperature_source_names.size();
         ++index)
    {
        sources.add(
            d_model_options.temperature_source_names[index],
            d_model_options.temperature_source_power_densities[index]);
    }
}

/**
 * @brief Return the immutable temperature field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored cell-centered temperature field.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() const noexcept
    -> const field_type&
{
    return d_problem.template object<field_type>("temperature");
}

/**
 * @brief Return the mutable temperature field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored cell-centered temperature field.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() noexcept -> field_type&
{
    return d_problem.template object<field_type>("temperature");
}

/**
 * @brief Return the internally stored mutable material-property fields.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_material_properties()
    -> MaterialPropertyFields<Pack>&
{
    return d_problem.template object<
        MaterialPropertyFields<Pack>>("material_properties");
}

/**
 * @brief Return the internally stored immutable material-property fields.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_material_properties() const
    -> const MaterialPropertyFields<Pack>&
{
    return d_problem.template object<
        MaterialPropertyFields<Pack>>("material_properties");
}

/**
 * @brief Enable physical transport and return mutable material properties.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned material-property fields.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::material_properties() noexcept
    -> MaterialPropertyFields<Pack>&
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
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::material_properties() const noexcept
    -> const MaterialPropertyFields<Pack>&
{
    return stored_material_properties();
}

/**
 * @brief Return the internally stored mutable turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned turbulence model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_turbulence_model()
    -> TurbulenceModel<Pack>&
{
    return d_problem.template object<TurbulenceModel<Pack>>(
        "turbulence_model");
}

/**
 * @brief Return the internally stored immutable turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned turbulence model.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_turbulence_model() const
    -> const TurbulenceModel<Pack>&
{
    return d_problem.template object<TurbulenceModel<Pack>>(
        "turbulence_model");
}

/**
 * @brief Report whether dimensional or turbulent transport is active.
 *
 * @tparam Pack Tpetra type pack.
 * @return true when a physical transport path must be used.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::physical_transport_enabled() const noexcept
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
auto BoussinesqSolver<Pack>::configure_turbulence(
    const TurbulenceModelOptions& options) -> TurbulenceModel<Pack>&
{
    auto& model = stored_turbulence_model();
    model.configure(
        options,
        stored_material_properties(),
        d_model_options.reference_density);
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
auto BoussinesqSolver<Pack>::configure_turbulence(
    const Database& database) -> TurbulenceModel<Pack>&
{
    auto& model = stored_turbulence_model();
    model.configure(
        database,
        stored_material_properties(),
        d_model_options.reference_density);
    return model;
}

/**
 * @brief Disable the active turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if an enabled model was disabled.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_turbulence_model() noexcept
{
    return stored_turbulence_model().disable();
}

/**
 * @brief Find the mutable active turbulence model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active model, or nullptr in laminar mode.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_turbulence_model() noexcept
    -> TurbulenceModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_turbulence_model() const noexcept
    -> const TurbulenceModel<Pack>*
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
auto BoussinesqSolver<Pack>::stored_temperature_sources()
    -> TemperatureSourceRegistry<Pack>&
{
    return d_problem.template object<
        TemperatureSourceRegistry<Pack>>("temperature_sources");
}

/**
 * @brief Return the internally stored immutable temperature-source registry.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_temperature_sources() const
    -> const TemperatureSourceRegistry<Pack>&
{
    return d_problem.template object<
        TemperatureSourceRegistry<Pack>>("temperature_sources");
}

/**
 * @brief Enable physical transport and return mutable temperature sources.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_sources() noexcept
    -> TemperatureSourceRegistry<Pack>&
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
auto BoussinesqSolver<Pack>::temperature_sources() const noexcept
    -> const TemperatureSourceRegistry<Pack>&
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
auto BoussinesqSolver<Pack>::add_temperature_source(
    std::string name,
    scalar_type initial_power_density)
    -> VolumetricScalarSource<Pack>&
{
    d_physical_model_enabled = true;
    return stored_temperature_sources().add(
        std::move(name), initial_power_density);
}

/**
 * @brief Create the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Newly created fission source.
 * @throws std::invalid_argument if a fission source already exists.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::add_fission_power_source()
    -> FissionPowerSource<Pack>&
{
    if (d_fission_power_source)
    {
        throw std::invalid_argument(
            "BoussinesqSolver already has a fission power source.");
    }
    d_physical_model_enabled = true;
    d_fission_power_source =
        std::make_unique<FissionPowerSource<Pack>>(
            d_mesh, stored_temperature_sources());
    return *d_fission_power_source;
}

/**
 * @brief Configure or disable the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Fission source profile and normalization options.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::configure_fission_power_source(
    const FissionPowerSourceOptions& options)
{
    detail::validate_fission_power_options(options);
    if (options.profile == FissionPowerProfile::Disabled)
    {
        remove_fission_power_source();
        return;
    }

    auto& source = d_fission_power_source
        ? *d_fission_power_source
        : add_fission_power_source();
    source.configure(options);
}

/**
 * @brief Remove the reserved fission power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a source was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_fission_power_source() noexcept
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
auto BoussinesqSolver<Pack>::find_fission_power_source() noexcept
    -> FissionPowerSource<Pack>*
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
auto BoussinesqSolver<Pack>::find_fission_power_source() const noexcept
    -> const FissionPowerSource<Pack>*
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
auto BoussinesqSolver<Pack>::configure_radiolytic_gas(
    const RadiolyticGasOptions& options) -> RadiolyticGasModel<Pack>&
{
    validate_radiolytic_gas_options(options);
    bool scalar_void_was_reset = false;
    if (options.mode == RadiolyticGasMode::Sheng2024TwoPopulation
        && d_boiling_source_model
        && d_boiling_source_model->enabled())
    {
        throw std::invalid_argument(
            "Sheng two-population radiolysis cannot be combined with "
            "boiling until vapor mass is coupled to bubble inventories.");
    }
    d_physical_model_enabled = true;
    if (options.mode != RadiolyticGasMode::Disabled
        && (!d_scalar_void_fraction_model
            || (!d_scalar_void_fraction_explicitly_configured
                && d_step_index == 0)))
    {
        ScalarVoidFractionOptions void_options;
        void_options.alpha_min = options.alpha_min;
        void_options.alpha_max = options.alpha_max;
        void_options.initial_alpha = options.alpha_min;
        if (!d_scalar_void_fraction_model)
        {
            d_scalar_void_fraction_model =
                std::make_unique<ScalarVoidFractionModel<Pack>>(
                    d_mesh, void_options);
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
        const auto& void_options =
            d_scalar_void_fraction_model->options();
        if (void_options.alpha_min != options.alpha_min
            || void_options.alpha_max != options.alpha_max)
        {
            throw std::invalid_argument(
                "Sheng radiolysis and its scalar mirror require identical "
                "void-fraction bounds.");
        }
        if (std::isfinite(void_options.alpha_collapse_time))
        {
            throw std::invalid_argument(
                "Sheng radiolysis cannot use scalar void collapse until "
                "bubble inventory removal is coupled conservatively.");
        }
    }
    if (!d_radiolytic_gas_model)
    {
        d_radiolytic_gas_model =
            std::make_unique<RadiolyticGasModel<Pack>>(
                d_mesh, options);
    }
    else
    {
        d_radiolytic_gas_model->configure(options);
    }
    if (options.mode == RadiolyticGasMode::Sheng2024TwoPopulation
        && d_primary_fields_initialized)
    {
        initialize_radiolytic_gas_state();
    }
    else if (scalar_void_was_reset
        && d_precursor_model
        && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(
            d_scalar_void_fraction_model->alpha_l());
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
auto BoussinesqSolver<Pack>::configure_radiolytic_gas(
    const Database& database) -> RadiolyticGasModel<Pack>&
{
    return configure_radiolytic_gas(
        radiolytic_gas_options_from_database(database));
}

/**
 * @brief Remove the optional radiolytic-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_radiolytic_gas_model() noexcept
{
    if (!d_radiolytic_gas_model)
        return false;
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
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() noexcept
    -> RadiolyticGasModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() const noexcept
    -> const RadiolyticGasModel<Pack>*
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
auto BoussinesqSolver<Pack>::configure_boiling_source(
    const BoilingSourceOptions& options) -> BoilingSourceModel<Pack>&
{
    validate_boiling_source_options(options);
    if ((options.enable_bulk_boiling || options.enable_wall_boiling)
        && d_radiolytic_gas_model
        && d_radiolytic_gas_model->supplies_void_fraction())
    {
        throw std::invalid_argument(
            "Boiling cannot be combined with Sheng two-population "
            "radiolysis until vapor mass is coupled to bubble inventories.");
    }
    d_physical_model_enabled = true;
    ensure_scalar_void_fraction_model();
    if (!d_boiling_source_model)
    {
        d_boiling_source_model =
            std::make_unique<BoilingSourceModel<Pack>>(
                d_mesh, options);
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
auto BoussinesqSolver<Pack>::configure_boiling_source(
    const Database& database) -> BoilingSourceModel<Pack>&
{
    return configure_boiling_source(
        boiling_source_options_from_database(database));
}

/**
 * @brief Remove the optional boiling source model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_boiling_source_model() noexcept
{
    if (!d_boiling_source_model)
        return false;
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
auto BoussinesqSolver<Pack>::find_boiling_source_model() noexcept
    -> BoilingSourceModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_boiling_source_model() const noexcept
    -> const BoilingSourceModel<Pack>*
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
auto BoussinesqSolver<Pack>::configure_scalar_void_fraction(
    const ScalarVoidFractionOptions& options)
    -> ScalarVoidFractionModel<Pack>&
{
    validate_scalar_void_fraction_options(options);
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->supplies_void_fraction()
        && (options.alpha_min
                != d_radiolytic_gas_model->options().alpha_min
            || options.alpha_max
                != d_radiolytic_gas_model->options().alpha_max))
    {
        throw std::invalid_argument(
            "A Sheng radiolysis mirror requires its scalar void bounds.");
    }
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->supplies_void_fraction()
        && std::isfinite(options.alpha_collapse_time))
    {
        throw std::invalid_argument(
            "Sheng radiolysis cannot use scalar void collapse until "
            "bubble inventory removal is coupled conservatively.");
    }
    d_physical_model_enabled = true;
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model =
            std::make_unique<ScalarVoidFractionModel<Pack>>(
                d_mesh, options);
    }
    else
    {
        d_scalar_void_fraction_model->configure(options);
    }
    d_scalar_void_fraction_explicitly_configured = true;
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->supplies_void_fraction()
        && d_radiolytic_gas_model->initial_state_initialized())
    {
        d_scalar_void_fraction_model->initialize_from(
            d_radiolytic_gas_model->alpha_g());
    }
    if (d_precursor_model && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(
            d_scalar_void_fraction_model->alpha_l());
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
auto BoussinesqSolver<Pack>::configure_scalar_void_fraction(
    const Database& database) -> ScalarVoidFractionModel<Pack>&
{
    return configure_scalar_void_fraction(
        scalar_void_fraction_options_from_database(database));
}

/**
 * @brief Find the mutable scalar void-fraction model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() noexcept
    -> ScalarVoidFractionModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() const noexcept
    -> const ScalarVoidFractionModel<Pack>*
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
auto BoussinesqSolver<Pack>::configure_material_feedback(
    const MaterialFeedbackOptions& options)
    -> MaterialFeedbackModel<Pack>&
{
    validate_material_feedback_options(options);
    d_physical_model_enabled = true;
    if (!d_material_feedback_model)
    {
        d_material_feedback_model =
            std::make_unique<MaterialFeedbackModel<Pack>>(
                d_mesh, options);
    }
    else
    {
        d_material_feedback_model->configure(options);
    }
    d_model_options.density_feedback_enabled =
        d_material_feedback_model->density_feedback_enabled();
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
auto BoussinesqSolver<Pack>::configure_material_feedback(
    const Database& database) -> MaterialFeedbackModel<Pack>&
{
    return configure_material_feedback(
        material_feedback_options_from_database(
            database, d_model_options, d_problem.time_options()));
}

/**
 * @brief Remove the optional material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_material_feedback_model() noexcept
{
    if (!d_material_feedback_model)
        return false;
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
auto BoussinesqSolver<Pack>::find_material_feedback_model() noexcept
    -> MaterialFeedbackModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_material_feedback_model() const noexcept
    -> const MaterialFeedbackModel<Pack>*
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
auto BoussinesqSolver<Pack>::configure_precursors(
    const DelayedNeutronPrecursorOptions& options)
    -> DelayedNeutronPrecursorModel<Pack>&
{
    validate_delayed_neutron_precursor_options(options);
    d_physical_model_enabled = true;
    if (d_primary_fields_initialized)
    {
        initialize_radiolytic_gas_state();
    }
    ensure_scalar_void_fraction_model();
    if (!d_precursor_model)
    {
        d_precursor_model =
            std::make_unique<DelayedNeutronPrecursorModel<Pack>>(
                d_mesh, options);
    }
    else
    {
        d_precursor_model->configure(options);
    }
    d_precursor_model->initialize_inventory(
        *active_alpha_l_field());
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
auto BoussinesqSolver<Pack>::configure_precursors(
    const Database& database) -> DelayedNeutronPrecursorModel<Pack>&
{
    return configure_precursors(
        delayed_neutron_precursor_options_from_database(database));
}

/**
 * @brief Remove the optional precursor model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_precursor_model() noexcept
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
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_precursor_model() noexcept
    -> DelayedNeutronPrecursorModel<Pack>*
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
auto BoussinesqSolver<Pack>::find_precursor_model() const noexcept
    -> const DelayedNeutronPrecursorModel<Pack>*
{
    return d_precursor_model.get();
}

/**
 * @brief Remove a named volumetric temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return true if a source was removed.
 */
template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_temperature_source(
    const std::string& name)
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
auto BoussinesqSolver<Pack>::find_temperature_source(
    const std::string& name) noexcept
    -> VolumetricScalarSource<Pack>*
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
auto BoussinesqSolver<Pack>::find_temperature_source(
    const std::string& name) const noexcept
    -> const VolumetricScalarSource<Pack>*
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
void BoussinesqSolver<Pack>::set_material_updater(
    typename MaterialPropertyFields<Pack>::updater_type updater)
{
    d_physical_model_enabled = true;
    stored_material_properties().set_updater(std::move(updater));
}

/**
 * @brief Remove the custom material-property updater.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::clear_material_updater() noexcept
{
    stored_material_properties().clear_updater();
}

/**
 * @brief Refresh material properties, feedback, and registered heat sources.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::refresh_physical_models()
{
    BoussinesqUpdateContext<Pack> context{
        d_time,
        d_step_index,
        *d_mesh,
        temperature(),
        pressure(),
        velocity()};
    stored_material_properties().update(context);
    initialize_radiolytic_gas_state(d_step_index == 0);
    refresh_material_feedback(d_time);
    stored_temperature_sources().update(context);
}

/**
 * @brief Initialize Sheng radiolysis inventories and dependent liquid fractions.
 *
 * @tparam Pack Tpetra type pack.
 * @param force Whether to reinitialize an already initialized model.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_radiolytic_gas_state(
    bool force)
{
    if (!d_radiolytic_gas_model
        || !d_radiolytic_gas_model->supplies_void_fraction()
        || (!force
            && d_radiolytic_gas_model->initial_state_initialized()))
    {
        return;
    }

    d_radiolytic_gas_model->initialize_state(
        d_time,
        temperature(),
        pressure(),
        velocity(),
        stored_material_properties(),
        force);
    if (d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model->initialize_from(
            d_radiolytic_gas_model->alpha_g());
    }
    if (d_precursor_model && d_step_index == 0)
    {
        d_precursor_model->initialize_inventory(
            *active_alpha_l_field());
    }
}

/**
 * @brief Apply material feedback for the supplied physical time.
 *
 * @tparam Pack Tpetra type pack.
 * @param time Physical time passed to feedback callbacks.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::refresh_material_feedback(scalar_type time)
{
    if (!d_material_feedback_model)
    {
        return;
    }
    BoussinesqUpdateContext<Pack> context{
        time,
        d_step_index,
        *d_mesh,
        temperature(),
        pressure(),
        velocity()};
    d_material_feedback_model->apply(
        context, active_alpha_g_field(), stored_material_properties());
}

/**
 * @brief Create the default scalar void-fraction model when absent.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::ensure_scalar_void_fraction_model()
{
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model =
            std::make_unique<ScalarVoidFractionModel<Pack>>(
                d_mesh, ScalarVoidFractionOptions{});
    }
}

/**
 * @brief Select the authoritative gas void-fraction field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active gas fraction field, or nullptr when no model supplies one.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::active_alpha_g_field() const noexcept
    -> const field_type*
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
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::active_alpha_l_field() const noexcept
    -> const field_type*
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
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::update_void_fraction_models(
    scalar_type time_step)
{
    if (!d_scalar_void_fraction_model)
    {
        return;
    }

    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->supplies_void_fraction())
    {
        d_scalar_void_fraction_model->mirror(
            d_radiolytic_gas_model->alpha_g(),
            time_step);
        return;
    }

    d_scalar_void_fraction_model->update_explicit(
        time_step,
        d_radiolytic_gas_model
            ? &d_radiolytic_gas_model->source_alpha_rad()
            : nullptr,
        d_boiling_source_model
            ? &d_boiling_source_model->source_alpha_boil()
            : nullptr);

    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled())
    {
        d_radiolytic_gas_model->synchronize_void_fraction(
            d_scalar_void_fraction_model->alpha_g(),
            d_scalar_void_fraction_model->options().alpha_max);
    }
}

/**
 * @brief Return the Problem-owned temperature equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored temperature diffusion equation.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_equation()
    -> TemperatureDiffusionEquation<Pack>&
{
    return d_problem.template object<
        TemperatureDiffusionEquation<Pack>>("temperature_equation");
}

/**
 * @brief Return the Problem-owned Boussinesq momentum equation.
 *
 * @tparam Pack Tpetra type pack.
 * @return Stored momentum equation.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::momentum_equation()
    -> BoussinesqMomentumEquation<Pack>&
{
    return d_problem.template object<
        BoussinesqMomentumEquation<Pack>>("momentum_equation");
}

/**
 * @brief Advance momentum with buoyancy and optional physical transport.
 *
 * @tparam Pack Tpetra type pack.
 * @return Aggregated linear solve summary.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::advance_momentum() -> LinearSolveSummary
{
    FVM::cell_gradient(
        pressure(),
        d_problem.boundary_conditions().pressure,
        predictor_pressure_gradient());
    const auto inverse_reference_density =
        scalar_type{1} / pressure_reference_density();
    const auto* turbulence = find_turbulence_model();
    auto pressure_source =
        [&](local_ordinal_type cell_lid) -> vec_type
    {
        auto acceleration = predictor_pressure_gradient().value(cell_lid)
                          * (-inverse_reference_density);
        if (turbulence != nullptr)
        {
            acceleration = acceleration +
                turbulence->turbulent_kinetic_energy_gradient().value(cell_lid)
                * scalar_type{-2.0 / 3.0};
        }
        return acceleration;
    };

    if (physical_transport_enabled())
    {
        return momentum_equation().advance_velocity_physical(
            velocity(),
            old_face_fluxes(),
            temperature(),
            velocity_boundary_cache(),
            d_problem.time_options(),
            stored_material_properties(),
            d_model_options.reference_density,
            d_model_options.density_feedback_enabled,
            velocity(),
            pressure_source,
            d_problem.linear_options(),
            turbulence != nullptr
                ? &turbulence->effective_dynamic_viscosity()
                : nullptr,
            turbulence != nullptr
                ? turbulence->effective_dynamic_viscosity_boundary_cache()
                : nullptr);
    }
    return momentum_equation().advance_velocity(
        velocity(),
        old_face_fluxes(),
        temperature(),
        velocity_boundary_cache(),
        d_problem.time_options(),
        velocity(),
        pressure_source,
        d_problem.linear_options());
}

/**
 * @brief Return the density used to normalize pressure operators.
 *
 * @tparam Pack Tpetra type pack.
 * @return Positive reference density.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_reference_density() const noexcept
    -> scalar_type
{
    return d_model_options.reference_density;
}

/**
 * @brief Assemble the coupled Boussinesq velocity-pressure system.
 *
 * @tparam Pack Tpetra type pack.
 * @return Monolithic coupled system with optional material and turbulence terms.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::assemble_coupled_system()
    -> coupled_system_type
{
    const auto* turbulence = find_turbulence_model();
    return coupled_pressure_velocity_solver().assemble(
        momentum_equation(),
        velocity(),
        pressure(),
        temperature(),
        old_face_fluxes(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions(),
        d_problem.time_options(),
        physical_transport_enabled()
            ? &stored_material_properties()
            : nullptr,
        d_model_options.reference_density,
        d_model_options.density_feedback_enabled,
        turbulence != nullptr
            ? &turbulence->effective_dynamic_viscosity()
            : nullptr,
        turbulence != nullptr
            ? &turbulence->turbulent_kinetic_energy_gradient()
            : nullptr,
        turbulence != nullptr
            ? turbulence->effective_dynamic_viscosity_boundary_cache()
            : nullptr);
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
    const vec_type& direction,
    scalar_type hot_at_min,
    scalar_type cold_at_max,
    scalar_type initial_pressure)
{
    if (direction.norm() <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a nonzero initialization direction.");
    }
    if (d_mesh->num_owned_cells() > 0)
    {
        auto min_projected = d_mesh->cell_centroid(0).dot(direction);
        auto max_projected = min_projected;
        for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(cell);
            const auto projected =
                d_mesh->cell_centroid(cell_lid).dot(direction);
            min_projected = std::min(min_projected, projected);
            max_projected = std::max(max_projected, projected);
        }

        const auto width = max_projected > min_projected
                         ? max_projected - min_projected
                         : 1.0;
        for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(cell);
            const auto projected =
                d_mesh->cell_centroid(cell_lid).dot(direction);
            const auto blend = (projected - min_projected) / width;
            temperature().set_value(
                cell_lid,
                hot_at_min * (1.0 - blend) + cold_at_max * blend);
            pressure().set_value(cell_lid, initial_pressure);
            velocity().set_value(cell_lid, {});
        }
    }

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(pressure());
    d_mesh->sync_periodic_boundaries(velocity());
    d_primary_fields_initialized = true;
    initialize_radiolytic_gas_state(true);
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
    scalar_type hot_temperature,
    scalar_type cold_temperature,
    scalar_type initial_pressure)
{
    initialize_linear_temperature({1.0, 0.0, 0.0},
                                  hot_temperature,
                                  cold_temperature,
                                  initial_pressure);
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
    scalar_type hot_temperature,
    scalar_type cold_temperature,
    scalar_type initial_pressure)
{
    initialize_linear_temperature({0.0, 0.0, 1.0},
                                  hot_temperature,
                                  cold_temperature,
                                  initial_pressure);
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
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled()
        && d_radiolytic_gas_model->supplies_void_fraction()
        && d_boiling_source_model
        && d_boiling_source_model->enabled())
    {
        throw std::logic_error(
            "Sheng two-population radiolysis cannot advance with boiling "
            "until vapor mass is coupled to bubble inventories.");
    }
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled()
        && d_radiolytic_gas_model->supplies_void_fraction()
        && d_scalar_void_fraction_model
        && (d_scalar_void_fraction_model->options().alpha_min
                != d_radiolytic_gas_model->options().alpha_min
            || d_scalar_void_fraction_model->options().alpha_max
                != d_radiolytic_gas_model->options().alpha_max))
    {
        throw std::logic_error(
            "Sheng radiolysis cannot advance with mismatched scalar mirror "
            "void-fraction bounds.");
    }
    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled()
        && d_radiolytic_gas_model->supplies_void_fraction()
        && d_scalar_void_fraction_model
        && std::isfinite(
            d_scalar_void_fraction_model->options()
                .alpha_collapse_time))
    {
        throw std::logic_error(
            "Sheng radiolysis cannot advance with scalar void collapse "
            "until bubble inventory removal is coupled conservatively.");
    }

    begin_step();
    if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(temperature());
    }
    if (d_physical_model_enabled)
    {
        refresh_physical_models();
    }
    if (auto* turbulence = find_turbulence_model())
    {
        turbulence->refresh_effective_properties(
            stored_material_properties(),
            d_model_options.reference_density);
    }

    solve_pressure_velocity_coupling();
    const auto time_step = d_problem.time_options().time_step;
    if (auto* turbulence = find_turbulence_model())
    {
        const auto turbulence_statistics = turbulence->advance(
            velocity(),
            projected_face_fluxes(),
            velocity_boundary_cache(),
            time_step,
            stored_material_properties(),
            d_model_options.reference_density,
            d_problem.time_options().non_orthogonal_treatment,
            d_problem.linear_options());
        d_last_step_statistics.add(turbulence_statistics);
    }
    const auto advanced_radiolysis =
        d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled()
        && d_radiolytic_gas_model->supplies_void_fraction();

    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled()
        && !advanced_radiolysis)
    {
        if (!d_scalar_void_fraction_model)
        {
            throw std::runtime_error(
                "Ideal radiolysis requires the authoritative scalar void model.");
        }
        d_radiolytic_gas_model->advance(
            d_time + time_step,
            time_step,
            temperature(),
            pressure(),
            velocity(),
            projected_face_fluxes(),
            stored_material_properties(),
            d_fission_power_source
                ? &d_fission_power_source->field()
                : nullptr,
            d_scalar_void_fraction_model->alpha_g(),
            d_scalar_void_fraction_model->options().alpha_max);
    }

    if (!advanced_radiolysis)
    {
        if (d_boiling_source_model)
        {
            if (!d_scalar_void_fraction_model)
            {
                throw std::runtime_error(
                    "Boiling requires the authoritative scalar void model.");
            }
            d_boiling_source_model->update(
                time_step,
                temperature(),
                stored_material_properties(),
                *d_scalar_void_fraction_model,
                d_radiolytic_gas_model
                        && d_radiolytic_gas_model->enabled()
                    ? &d_radiolytic_gas_model->source_alpha_rad()
                    : nullptr);
        }
        update_void_fraction_models(time_step);
    }

    LinearSolveStatistics temperature_statistics;
    if (physical_transport_enabled())
    {
        const auto* turbulence = find_turbulence_model();
        auto total_power_density =
            [&](local_ordinal_type cell_lid)
        {
            auto total = stored_temperature_sources()
                .total_power_density(cell_lid);
            if (d_boiling_source_model)
            {
                total += d_boiling_source_model
                    ->temperature_source(cell_lid);
            }
            return total;
        };
        temperature_statistics =
            temperature_equation().advance_physical(
                temperature(),
                projected_face_fluxes(),
                time_step,
                stored_material_properties(),
                temperature(),
                total_power_density,
                d_problem.time_options().non_orthogonal_treatment,
                d_problem.linear_options(),
                turbulence != nullptr
                    ? &turbulence->effective_thermal_conductivity()
                    : nullptr,
                turbulence != nullptr
                    ? turbulence->effective_thermal_conductivity_boundary_cache()
                    : nullptr);
    }
    else
    {
        temperature_statistics =
            temperature_equation().advance_semi_implicit(
                temperature(),
                projected_face_fluxes(),
                time_step,
                d_problem.time_options().thermal_diffusivity,
                temperature(),
                d_problem.linear_options());
    }
    d_last_step_statistics.add(temperature_statistics);
    d_last_step_statistics.temperature =
        temperature_statistics.achieved_tolerance;

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(velocity());

    if (advanced_radiolysis)
    {
        d_radiolytic_gas_model->advance(
            d_time + time_step,
            time_step,
            temperature(),
            pressure(),
            velocity(),
            projected_face_fluxes(),
            stored_material_properties(),
            d_fission_power_source
                ? &d_fission_power_source->field()
                : nullptr);
        update_void_fraction_models(time_step);
    }

    if (d_precursor_model && d_precursor_model->enabled())
    {
        const auto* alpha_l = active_alpha_l_field();
        if (alpha_l == nullptr)
        {
            throw std::runtime_error(
                "Precursor model requires a liquid fraction field.");
        }
        d_precursor_model->advance(
            time_step,
            *alpha_l,
            d_fission_power_source
                ? &d_fission_power_source->field()
                : nullptr);
    }

    refresh_material_feedback(
        d_time + time_step);
    if (auto* turbulence = find_turbulence_model())
    {
        turbulence->refresh_effective_properties(
            stored_material_properties(),
            d_model_options.reference_density);
    }

    finish_step();
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
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_solution_vtu(const std::string& filename) const
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
auto BoussinesqSolver<Pack>::solution_writer(
    const SolutionOutputOptions& output_options) const -> VTUWriter
{
    auto writer = fluid_solution_writer();
    writer.add_scalar_cell_data(
        "temperature", collect_scalar_field(temperature()));

    if (output_options.include_material_properties)
    {
        const auto& material = stored_material_properties();
        writer.add_scalar_cell_data(
            "density", collect_scalar_field(material.density));
        writer.add_scalar_cell_data(
            "specific_heat_capacity",
            collect_scalar_field(material.specific_heat_capacity));
        writer.add_scalar_cell_data(
            "dynamic_viscosity",
            collect_scalar_field(material.dynamic_viscosity));
        writer.add_scalar_cell_data(
            "thermal_conductivity",
            collect_scalar_field(material.thermal_conductivity));
        if (d_material_feedback_model)
        {
            for (const auto& [name, field] :
                 d_material_feedback_model->output_fields())
            {
                writer.add_scalar_cell_data(
                    name, collect_scalar_field(*field));
            }
        }
    }
    if (output_options.include_turbulence_fields)
    {
        if (const auto* turbulence = find_turbulence_model())
        {
            for (const auto& [name, field] : turbulence->output_fields())
            {
                writer.add_scalar_cell_data(
                    name, collect_scalar_field(*field));
            }
        }
    }
    if (output_options.include_sources)
    {
        for (const auto& [name, source] :
             stored_temperature_sources().entries())
        {
            writer.add_scalar_cell_data(
                name, collect_scalar_field(source->field()));
        }
        if (d_radiolytic_gas_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_rad",
                collect_scalar_field(
                    d_radiolytic_gas_model->source_alpha_rad()));
        }
        if (d_boiling_source_model)
        {
            for (const auto& [name, field] :
                 d_boiling_source_model->output_fields())
            {
                writer.add_scalar_cell_data(
                    name, collect_scalar_field(*field));
            }
        }
        if (d_scalar_void_fraction_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total",
                collect_scalar_field(
                    d_scalar_void_fraction_model
                        ->source_alpha_total()));
        }
    }
    if (output_options.include_radiolytic_gas_fields
        && d_scalar_void_fraction_model)
    {
        writer.add_scalar_cell_data(
            "alpha_g",
            collect_scalar_field(d_scalar_void_fraction_model->alpha_g()));
        writer.add_scalar_cell_data(
            "alpha_l",
            collect_scalar_field(d_scalar_void_fraction_model->alpha_l()));
        if (!output_options.include_sources)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total",
                collect_scalar_field(
                    d_scalar_void_fraction_model
                        ->source_alpha_total()));
        }
    }
    if (output_options.include_radiolytic_gas_fields
        && d_radiolytic_gas_model)
    {
        for (const auto& [name, field] :
             d_radiolytic_gas_model->output_fields())
        {
            if (name == "S_alpha_rad")
                continue;
            if (d_scalar_void_fraction_model
                && (name == "alpha_g" || name == "alpha_l"))
            {
                continue;
            }
            writer.add_scalar_cell_data(
                name, collect_scalar_field(*field));
        }
    }
    if (output_options.include_precursor_fields
        && d_precursor_model)
    {
        for (const auto& [name, field] :
             d_precursor_model->output_fields())
        {
            writer.add_scalar_cell_data(
                name, collect_scalar_field(*field));
        }
    }
    return writer;
}

/** @brief Write selected solution fields to one VTU piece. */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_solution_vtu(
    const std::string& filename,
    const SolutionOutputOptions& output_options) const
{
    solution_writer(output_options).write(
        filename, VTUWriter::Encoding::AppendedBinary);
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
    const std::string& filename,
    const SolutionOutputOptions& output_options) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count > 1)
    {
        const auto option_mask =
            static_cast<int>(output_options.include_sources)
          | (static_cast<int>(
                 output_options.include_material_properties) << 1)
          | (static_cast<int>(
                 output_options.include_radiolytic_gas_fields) << 2)
          | (static_cast<int>(
                 output_options.include_precursor_fields) << 3)
          | (static_cast<int>(
                 output_options.include_turbulence_fields) << 4);
        std::array<long long, 2> root_arguments{
            rank == 0
                ? static_cast<long long>(filename.size())
                : 0LL,
            rank == 0 ? static_cast<long long>(option_mask) : 0LL};
        Teuchos::broadcast(
            *communicator,
            0,
            static_cast<int>(root_arguments.size()),
            root_arguments.data());
        if (root_arguments[0] < 0
            || root_arguments[0]
               > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(
                "Parallel VTU filename is too long for MPI broadcast.");
        }

        std::string root_filename(
            static_cast<size_t>(root_arguments[0]), '\0');
        if (rank == 0)
        {
            root_filename = filename;
        }
        if (!root_filename.empty())
        {
            Teuchos::broadcast(
                *communicator,
                0,
                static_cast<int>(root_filename.size()),
                root_filename.data());
        }

        const int local_arguments_mismatch =
            (filename != root_filename
             || option_mask != static_cast<int>(root_arguments[1]))
                ? 1
                : 0;
        int any_arguments_mismatch = 0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_MAX,
            1,
            &local_arguments_mismatch,
            &any_arguments_mismatch);
        if (any_arguments_mismatch != 0)
        {
            throw std::invalid_argument(
                "Parallel VTU filename and output options must agree on "
                "every rank.");
        }
    }
    const auto piece_filename = VTUWriter::rank_piece_filename(
        filename, rank, rank_count);
    if (rank_count <= 1)
    {
        solution_writer(output_options).write(
            piece_filename, VTUWriter::Encoding::AppendedBinary);
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
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_preparation_failed,
        &any_preparation_failed);
    if (any_preparation_failed != 0)
    {
        if (preparation_error)
        {
            std::rethrow_exception(preparation_error);
        }
        throw std::runtime_error(
            "VTU output preparation failed on another MPI rank.");
    }

    const int local_schema_size_is_valid =
        local_schema_key.size()
                <= static_cast<size_t>(
                    std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_schema_size_is_valid = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_schema_size_is_valid,
        &global_schema_size_is_valid);
    if (global_schema_size_is_valid == 0)
    {
        throw std::invalid_argument(
            "Parallel VTU CellData schema is too large for MPI "
            "broadcast.");
    }

    int root_schema_size = rank == 0
        ? static_cast<int>(local_schema_key.size())
        : 0;
    Teuchos::broadcast(
        *communicator, 0, 1, &root_schema_size);
    std::string root_schema_key(
        static_cast<size_t>(root_schema_size), '\0');
    if (rank == 0)
    {
        root_schema_key = local_schema_key;
    }
    if (!root_schema_key.empty())
    {
        Teuchos::broadcast(
            *communicator,
            0,
            root_schema_size,
            root_schema_key.data());
    }
    const int local_schema_mismatch =
        local_schema_key == root_schema_key ? 0 : 1;
    int any_schema_mismatch = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_schema_mismatch,
        &any_schema_mismatch);
    if (any_schema_mismatch != 0)
    {
        throw std::invalid_argument(
            "Parallel VTU CellData names, types, and component counts "
            "must agree on every rank.");
    }

    std::exception_ptr piece_error;
    try
    {
        writer->write(
            piece_filename, VTUWriter::Encoding::AppendedBinary);
    }
    catch (...)
    {
        piece_error = std::current_exception();
    }
    const int local_piece_failed = piece_error ? 1 : 0;
    int any_piece_failed = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_piece_failed,
        &any_piece_failed);
    if (any_piece_failed != 0)
    {
        if (piece_error)
        {
            std::rethrow_exception(piece_error);
        }
        throw std::runtime_error(
            "A VTU piece failed to write on another MPI rank.");
    }

    std::exception_ptr index_error;
    if (rank == 0)
    {
        try
        {
            std::vector<std::string> piece_filenames;
            piece_filenames.reserve(static_cast<size_t>(rank_count));
            for (int piece_rank = 0;
                 piece_rank < rank_count;
                 ++piece_rank)
            {
                piece_filenames.push_back(
                    VTUWriter::rank_piece_filename(
                        filename, piece_rank, rank_count));
            }
            writer->write_parallel_index(
                VTUWriter::parallel_index_filename(filename),
                piece_filenames);
        }
        catch (...)
        {
            index_error = std::current_exception();
        }
    }
    int local_index_failed = index_error ? 1 : 0;
    int any_index_failed = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_index_failed,
        &any_index_failed);
    if (any_index_failed != 0)
    {
        if (index_error)
        {
            std::rethrow_exception(index_error);
        }
        throw std::runtime_error(
            "The PVTU index failed to write on another MPI rank.");
    }
}

} // namespace SimpleFluid
