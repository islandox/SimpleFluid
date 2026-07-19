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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() const noexcept
    -> const field_type&
{
    return d_problem.template object<field_type>("temperature");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() noexcept -> field_type&
{
    return d_problem.template object<field_type>("temperature");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_material_properties()
    -> MaterialPropertyFields<Pack>&
{
    return d_problem.template object<
        MaterialPropertyFields<Pack>>("material_properties");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_material_properties() const
    -> const MaterialPropertyFields<Pack>&
{
    return d_problem.template object<
        MaterialPropertyFields<Pack>>("material_properties");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::material_properties() noexcept
    -> MaterialPropertyFields<Pack>&
{
    d_physical_model_enabled = true;
    return stored_material_properties();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::material_properties() const noexcept
    -> const MaterialPropertyFields<Pack>&
{
    return stored_material_properties();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_turbulence_model()
    -> TurbulenceModel<Pack>&
{
    return d_problem.template object<TurbulenceModel<Pack>>(
        "turbulence_model");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_turbulence_model() const
    -> const TurbulenceModel<Pack>&
{
    return d_problem.template object<TurbulenceModel<Pack>>(
        "turbulence_model");
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::physical_transport_enabled() const noexcept
{
    return d_physical_model_enabled || stored_turbulence_model().enabled();
}

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

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_turbulence_model() noexcept
{
    return stored_turbulence_model().disable();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_turbulence_model() noexcept
    -> TurbulenceModel<Pack>*
{
    auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_turbulence_model() const noexcept
    -> const TurbulenceModel<Pack>*
{
    const auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_temperature_sources()
    -> TemperatureSourceRegistry<Pack>&
{
    return d_problem.template object<
        TemperatureSourceRegistry<Pack>>("temperature_sources");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::stored_temperature_sources() const
    -> const TemperatureSourceRegistry<Pack>&
{
    return d_problem.template object<
        TemperatureSourceRegistry<Pack>>("temperature_sources");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_sources() noexcept
    -> TemperatureSourceRegistry<Pack>&
{
    d_physical_model_enabled = true;
    return stored_temperature_sources();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_sources() const noexcept
    -> const TemperatureSourceRegistry<Pack>&
{
    return stored_temperature_sources();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_fission_power_source() noexcept
    -> FissionPowerSource<Pack>*
{
    return d_fission_power_source.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_fission_power_source() const noexcept
    -> const FissionPowerSource<Pack>*
{
    return d_fission_power_source.get();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_radiolytic_gas(
    const Database& database) -> RadiolyticGasModel<Pack>&
{
    return configure_radiolytic_gas(
        radiolytic_gas_options_from_database(database));
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_radiolytic_gas_model() noexcept
{
    if (!d_radiolytic_gas_model)
        return false;
    d_radiolytic_gas_model.reset();
    return true;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() noexcept
    -> RadiolyticGasModel<Pack>*
{
    return d_radiolytic_gas_model.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_radiolytic_gas_model() const noexcept
    -> const RadiolyticGasModel<Pack>*
{
    return d_radiolytic_gas_model.get();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_boiling_source(
    const Database& database) -> BoilingSourceModel<Pack>&
{
    return configure_boiling_source(
        boiling_source_options_from_database(database));
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_boiling_source_model() noexcept
{
    if (!d_boiling_source_model)
        return false;
    d_boiling_source_model.reset();
    return true;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_boiling_source_model() noexcept
    -> BoilingSourceModel<Pack>*
{
    return d_boiling_source_model.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_boiling_source_model() const noexcept
    -> const BoilingSourceModel<Pack>*
{
    return d_boiling_source_model.get();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_scalar_void_fraction(
    const Database& database) -> ScalarVoidFractionModel<Pack>&
{
    return configure_scalar_void_fraction(
        scalar_void_fraction_options_from_database(database));
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() noexcept
    -> ScalarVoidFractionModel<Pack>*
{
    return d_scalar_void_fraction_model.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_scalar_void_fraction_model() const noexcept
    -> const ScalarVoidFractionModel<Pack>*
{
    return d_scalar_void_fraction_model.get();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_material_feedback(
    const Database& database) -> MaterialFeedbackModel<Pack>&
{
    return configure_material_feedback(
        material_feedback_options_from_database(
            database, d_model_options, d_problem.time_options()));
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_material_feedback_model() noexcept
{
    if (!d_material_feedback_model)
        return false;
    d_material_feedback_model.reset();
    d_model_options.density_feedback_enabled = false;
    return true;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_material_feedback_model() noexcept
    -> MaterialFeedbackModel<Pack>*
{
    return d_material_feedback_model.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_material_feedback_model() const noexcept
    -> const MaterialFeedbackModel<Pack>*
{
    return d_material_feedback_model.get();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::configure_precursors(
    const Database& database) -> DelayedNeutronPrecursorModel<Pack>&
{
    return configure_precursors(
        delayed_neutron_precursor_options_from_database(database));
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_precursor_model() noexcept
{
    if (!d_precursor_model)
        return false;
    d_precursor_model.reset();
    return true;
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_precursor_model() noexcept
    -> DelayedNeutronPrecursorModel<Pack>*
{
    return d_precursor_model.get();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_precursor_model() const noexcept
    -> const DelayedNeutronPrecursorModel<Pack>*
{
    return d_precursor_model.get();
}

template<TpetraTypePack Pack>
bool BoussinesqSolver<Pack>::remove_temperature_source(
    const std::string& name)
{
    return stored_temperature_sources().remove(name);
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_temperature_source(
    const std::string& name) noexcept
    -> VolumetricScalarSource<Pack>*
{
    return stored_temperature_sources().find(name);
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::find_temperature_source(
    const std::string& name) const noexcept
    -> const VolumetricScalarSource<Pack>*
{
    return stored_temperature_sources().find(name);
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::set_material_updater(
    typename MaterialPropertyFields<Pack>::updater_type updater)
{
    d_physical_model_enabled = true;
    stored_material_properties().set_updater(std::move(updater));
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::clear_material_updater() noexcept
{
    stored_material_properties().clear_updater();
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_equation()
    -> TemperatureDiffusionEquation<Pack>&
{
    return d_problem.template object<
        TemperatureDiffusionEquation<Pack>>("temperature_equation");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::momentum_equation()
    -> BoussinesqMomentumEquation<Pack>&
{
    return d_problem.template object<
        BoussinesqMomentumEquation<Pack>>("momentum_equation");
}

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_reference_density() const noexcept
    -> scalar_type
{
    return d_model_options.reference_density;
}

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

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_solution_vtu(
    const std::string& filename,
    const SolutionOutputOptions& output_options) const
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
    writer.write(filename);
}

} // namespace SimpleFluid
