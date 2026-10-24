/**
 * @file MultiphysicsSolverCoupler.tcc
 * @brief Optional-model orchestration, planar ALE trials, and checkpoint replay.
 */
#include "FVM/BoundaryCache.hh"
#include "equations/BoussFiniteVapor.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/FieldStateTransaction.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "solvers/SafeguardedAitken.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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

[[nodiscard]] std::string encode_headspace_metadata(const HeadspaceMetadata& metadata)
{
    std::ostringstream encoded;
    encoded << std::hexfloat << metadata.catalog_fingerprint << ':' << metadata.provider_identity.size() << ':'
            << metadata.provider_identity << ':' << metadata.provider_revision << ':'
            << metadata.prescribed_temperature_only;
    for (const auto bound : metadata.validity_bounds)
        encoded << ':' << bound;
    return encoded.str();
}

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
    scalar(options.ale.level_tolerance_from_volume);
    scalar(options.ale.level_absolute_tolerance);
    scalar(options.ale.level_relative_tolerance);
    scalar(options.ale.relaxation);
    scalar(options.ale.aitken);
    scalar(options.ale.initial_relaxation);
    scalar(options.ale.minimum_relaxation);
    scalar(options.ale.maximum_relaxation);
    optional_scalar(options.ale.quality_limits.maximum_growth_ratio);
    optional_scalar(options.ale.quality_limits.maximum_non_orthogonality_degrees);
    optional_scalar(options.ale.quality_limits.maximum_skewness);
    optional_scalar(options.ale.quality_limits.maximum_aspect_ratio);
    return encoded.str();
}

/** Deterministically encode every rank-global choice used by planar ALE. */
[[nodiscard]] std::string encode_planar_ale_runtime_options(const TimeStepperOptions& time,
    const LinearSolverOptions& linear, const LinearSolverOptions& pressure_linear,
    const LinearSolverOptions& momentum_linear, const BoussinesqModelOptions& model,
    const MaterialFeedbackOptions* feedback, const ReactingGasOptions* gas, const BoundaryConditionSet& boundaries,
    const std::vector<std::pair<std::string, bool>>& temperature_sources, bool mutable_mesh, bool legacy_backend,
    bool physical_transport, bool turbulence_enabled, bool boiling_enabled,
    bool scalar_void_explicit, bool material_updater, bool dynamic_temperature_sources)
{
    std::ostringstream encoded;
    encoded << std::hexfloat;
    const auto scalar = [&encoded](const auto value) { encoded << value << ';'; };
    const auto string = [&encoded](const std::string& value) { encoded << value.size() << ':' << value << ';'; };
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
            (void) condition;
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
            (void) condition;
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
    linear_options(momentum_linear);

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
    if (gas)
    {
        scalar(static_cast<int>(gas->mode));
        scalar(static_cast<int>(gas->source_schedule));
        scalar(static_cast<int>(gas->pressure_mode));
        scalar(static_cast<int>(gas->solution_transport));
        scalar(static_cast<int>(gas->bubble_transport));
        scalar(static_cast<int>(gas->rise_velocity_mode));
        scalar(static_cast<int>(gas->surface_tension_mode));
        scalar(static_cast<int>(gas->mass_transfer_mode));
        scalar(static_cast<int>(gas->bulk_diffusion));
        scalar(gas->reference_pressure);
        scalar(gas->minimum_absolute_pressure);
        scalar(gas->alpha_min);
        scalar(gas->alpha_max);
        scalar(gas->surface_tension);
        scalar(gas->constant_slip_velocity);
        scalar(gas->bubble_gravity);
        scalar(gas->rise_velocity_tolerance);
        scalar(gas->max_rise_velocity_iterations);
        scalar(gas->min_radius);
        scalar(gas->max_radius);
        scalar(gas->transport_solver_tolerance);
        scalar(gas->max_subcycles);
        scalar(gas->micro_dissolution_frequency);
        scalar(gas->large_dissolution_frequency);
        scalar(gas->micro_conversion_frequency);
        scalar(static_cast<bool>(gas->species_source));
        scalar(static_cast<bool>(gas->population_process));
        scalar(static_cast<bool>(gas->initial_species_state));
        scalar(static_cast<bool>(gas->initial_population_numbers));
        scalar(static_cast<bool>(gas->surface_tension_correlation));
        scalar(gas->reaction_advance.maximumSubstep);
        scalar(gas->reaction_advance.maximumSubsteps);
        scalar(gas->reaction_advance.maximumRetries);
        scalar(static_cast<bool>(gas->reaction_mechanism));
        if (gas->reaction_mechanism)
        {
            scalar(gas->reaction_mechanism->catalog().fingerprint());
            scalar(static_cast<int>(gas->reaction_mechanism->energy_convention()));
            for (const auto& reaction : gas->reaction_mechanism->reactions())
            {
                scalar(reaction.id);
                string(reaction.name);
                scalar(reaction.heatPerExtent);
                scalar(reaction.changes.size());
                for (const auto& term : reaction.changes)
                {
                    scalar(term.species);
                    scalar(static_cast<int>(term.phase));
                    scalar(term.coefficient);
                    string(term.population);
                }
            }
        }
        scalar(gas->species_properties.size());
        for (const auto& property : gas->species_properties)
        {
            scalar(property.species);
            scalar(property.diffusivity);
            scalar(property.henry_coefficient);
            scalar(static_cast<bool>(property.diffusivity_provider));
            scalar(static_cast<bool>(property.equilibrium_concentration));
            scalar(static_cast<bool>(property.carried_molar_enthalpy));
        }
        for (const auto& eos : {gas->liquid_thermodynamics, gas->gas_thermodynamics})
        {
            scalar(static_cast<bool>(eos));
            if (!eos)
                continue;
            scalar(eos->catalog().fingerprint());
            scalar(static_cast<int>(eos->kind()));
            scalar(static_cast<int>(eos->energy_convention()));
            scalar(eos->bounds().minimum_temperature);
            scalar(eos->bounds().maximum_temperature);
            scalar(eos->bounds().minimum_pressure);
            scalar(eos->bounds().maximum_pressure);
        }
        scalar(static_cast<bool>(gas->bouss_reference));
        if (gas->bouss_reference)
            scalar(Reacting::bouss_configuration_signature(*gas->bouss_reference, *gas->bouss_sources));
        real_array(gas->pressure_history_times);
        real_array(gas->pressure_history_values);
        string_array(gas->population_ids);
        string_array(gas->free_surface_patches);
    }

    scalar_boundaries(boundaries.temperature);
    string(convection_detail::configuration_signature(boundaries.convection));
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
 * @brief Enable physical transport and return mutable temperature sources.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::temperature_sources() -> temperature_source_registry_type&
{
    d_solver->d_physical_model_enabled = true;
    return d_solver->stored_temperature_sources();
}

/**
 * @brief Return the immutable temperature-source registry.
 *
 * @tparam Pack Tpetra type pack.
 * @return Problem-owned source registry.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::temperature_sources() const -> const temperature_source_registry_type&
{
    return d_solver->stored_temperature_sources();
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
auto MultiphysicsSolverCoupler<Pack>::add_temperature_source(std::string name, scalar_type initial_power_density)
    -> volumetric_source_type&
{
    d_solver->d_physical_model_enabled = true;
    return d_solver->stored_temperature_sources().add(std::move(name), initial_power_density);
}

/**
 * @brief Create the reserved prescribed_heat power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Newly created prescribed_heat source.
 * @throws std::invalid_argument if a prescribed_heat source already exists.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::add_prescribed_heat_source() -> prescribed_heat_source_type&
{
    if (d_prescribed_heat_source)
    {
        throw std::invalid_argument("BoussinesqSolver already has a prescribed_heat power source.");
    }
    d_solver->d_physical_model_enabled = true;
    d_prescribed_heat_source =
        std::make_unique<prescribed_heat_source_type>(d_solver->d_mesh, d_solver->stored_temperature_sources());
    return *d_prescribed_heat_source;
}

/**
 * @brief Configure or disable the reserved prescribed_heat power source.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Prescribed heat source profile and normalization options.
 */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::configure_prescribed_heat_source(const PrescribedHeatSourceOptions& options)
{
    detail::validate_prescribed_heat_options(options);
    if (options.profile == PrescribedHeatProfile::Disabled)
    {
        remove_prescribed_heat_source();
        return;
    }

    auto& source = d_prescribed_heat_source ? *d_prescribed_heat_source : add_prescribed_heat_source();
    source.configure(options);
}

/**
 * @brief Remove the reserved prescribed_heat power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a source was removed.
 */
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_prescribed_heat_source() noexcept
{
    if (!d_prescribed_heat_source)
    {
        return false;
    }
    d_prescribed_heat_source.reset();
    return true;
}

/**
 * @brief Find the mutable prescribed_heat power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured source, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_prescribed_heat_source() noexcept -> prescribed_heat_source_type*
{
    return d_prescribed_heat_source.get();
}

/**
 * @brief Find the immutable prescribed_heat power source.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured source, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_prescribed_heat_source() const noexcept -> const prescribed_heat_source_type*
{
    return d_prescribed_heat_source.get();
}

/**
 * @brief Configure the optional reacting-gas model from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options ReactingGas mode, kinetics, and void-fraction bounds.
 * @return Configured reacting-gas model.
 * @throws std::invalid_argument if options or coupled-model combinations are invalid.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_reacting_gas(const ReactingGasOptions& supplied_options)
    -> reacting_gas_model_type&
{
    validate_collective_model_state();
    std::optional<ReactingGasOptions> staged_options;
    collective_detail::collective_local_validation(
        *d_solver->d_mesh, "Reacting options copy", [&] { staged_options.emplace(supplied_options); });
    // Even default population-vector construction belongs inside the local
    // allocation fence. The private preflight receives only the staged move.
    auto options = reacting_gas_model_type::require_options(d_solver->d_mesh, std::move(*staged_options));
    const bool replace_mirror = options.mode != ReactingGasMode::Disabled &&
                                (!d_scalar_void_fraction_model ||
                                    (!d_scalar_void_fraction_explicitly_configured && d_solver->d_step_index == 0));
    collective_detail::require_uniform_value(
        *d_solver->d_mesh, static_cast<int>(replace_mirror), "Reacting scalar mirror setup branch");
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Reacting coupled support preflight",
        [&]
        {
            if (options.source_schedule == ReactingSourceSchedule::PrescribedCarrierAfterTemperature &&
                ((d_boiling_source_model && d_boiling_source_model->enabled()) || d_free_surface_model ||
                    d_thermal_participant || options.bouss_reference || options.bouss_sources))
                throw std::invalid_argument(
                    "Post-temperature reference setup cannot own boiling, surface geometry or thermal participation.");
            if (d_bouss_wall_trial)
                throw std::logic_error("Cannot replace a reacting model during certified wall admission.");
            if (d_free_surface_model && d_free_surface_model->initialized())
                throw std::logic_error("Remove the initialized free-surface model before reconfiguring reacting gas.");
            if (options.reaction_mechanism && options.mode != ReactingGasMode::Disabled &&
                options.reaction_mechanism->energy_convention() != Reacting::EnergyConvention::SensibleEnergy)
                throw std::invalid_argument("Boussinesq temperature storage requires sensible-energy chemistry.");
            if (d_free_surface_model && options.mode == ReactingGasMode::Species &&
                options.pressure_mode != ReactingPressureMode::Constant &&
                options.pressure_mode != ReactingPressureMode::Reconstructed)
                throw std::invalid_argument("Planar surface closure requires constant or reconstructed gas pressure.");
            if (options.mode == ReactingGasMode::Species && d_boiling_source_model && d_boiling_source_model->enabled())
                throw std::invalid_argument("Independent boiling cannot also own coupled gas/vapor inventory.");
            if (options.bouss_reference &&
                (d_free_surface_model || d_thermal_participant || d_active_ale ||
                    d_solver->stored_material_properties().has_updater() ||
                    options.bouss_reference->heatCapacity != d_solver->d_model_options.specific_heat_capacity ||
                    options.bouss_reference->thermalDensity != d_solver->d_model_options.reference_density))
                throw std::invalid_argument("Finite Bouss sources require fixed reference-density/cp temperature "
                                            "storage without surface/ALE/native thermal participation.");
            if (options.mode == ReactingGasMode::Species && !replace_mirror)
            {
                const auto& mirror = d_scalar_void_fraction_model->options();
                if (mirror.alpha_min != options.alpha_min || mirror.alpha_max != options.alpha_max)
                    throw std::invalid_argument(
                        "Reacting inventories and retained scalar mirror need identical bounds.");
                if (std::isfinite(mirror.alpha_collapse_time))
                    throw std::invalid_argument("Scalar collapse has no conservative bubble inventory owner.");
            }
        });
    if (options.bouss_reference)
    {
        if (!d_bouss_temperature_density)
            d_bouss_temperature_density =
                std::make_unique<field_type>(d_solver->d_mesh, "boussReferenceThermalDensity");
        d_bouss_temperature_density->put_scalar(options.bouss_reference->thermalDensity);
    }
    else
        d_bouss_temperature_density.reset();
    d_solver->d_physical_model_enabled = true;
    if (replace_mirror)
    {
        ScalarVoidFractionOptions mirror;
        mirror.alpha_min = options.alpha_min;
        mirror.alpha_max = options.alpha_max;
        mirror.initial_alpha = options.alpha_min;
        if (!d_scalar_void_fraction_model)
            d_scalar_void_fraction_model = std::make_unique<scalar_void_fraction_model_type>(d_solver->d_mesh, mirror);
        else
            d_scalar_void_fraction_model->configure(mirror);
    }
    else
        ensure_scalar_void_fraction_model();
    if (!d_reacting_gas_model)
        d_reacting_gas_model = std::make_unique<reacting_gas_model_type>(d_solver->d_mesh, options);
    else
        d_reacting_gas_model->configure(options);
    if (options.mode == ReactingGasMode::Species && d_solver->d_primary_fields_initialized)
        initialize_reacting_gas_state();
    return *d_reacting_gas_model;
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::configure_bouss_sources(
    std::shared_ptr<const Reacting::BoussCoordinatedSources> sources)
{
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Bouss source replacement",
        [&]
        {
            if (d_bouss_wall_trial || !d_reacting_gas_model || !d_reacting_gas_model->options().bouss_reference ||
                !sources || d_free_surface_model || d_thermal_participant || d_active_ale ||
                d_reacting_gas_model->d_source_configuration_generation == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Bouss source replacement needs an existing fixed-grid canonical "
                                            "reference binding and immutable supplied laws.");
            auto options = d_reacting_gas_model->options();
            options.bouss_sources = sources;
            validate_reacting_gas_options(options);
        });
    const auto& reference = *d_reacting_gas_model->options().bouss_reference;
    collective_detail::require_uniform_value(*d_solver->d_mesh,
        static_cast<unsigned long long>(Reacting::bouss_configuration_signature(reference, *sources)),
        "Bouss replacement source metadata");
    d_reacting_gas_model->d_options.bouss_sources = std::move(sources);
    ++d_reacting_gas_model->d_source_configuration_generation;
}

/**
 * @brief Configure the optional reacting-gas model from database keys.
 *
 * @tparam Pack Tpetra type pack.
 * @param database Database containing reacting_gas options.
 * @return Configured reacting-gas model.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_reacting_gas(const Database& database) -> reacting_gas_model_type&
{
    ReactingGasOptions options;
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Reacting gas database options", [&]
        { options = reacting_gas_options_from_database(database); });
    return configure_reacting_gas(options);
}

/**
 * @brief Remove the optional reacting-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_reacting_gas_model()
{
    if (!d_reacting_gas_model)
        return false;
    if (d_free_surface_model)
    {
        remove_free_surface_model();
    }
    d_reacting_gas_model.reset();
    return true;
}

/**
 * @brief Find the mutable reacting-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_reacting_gas_model() noexcept -> reacting_gas_model_type*
{
    return d_reacting_gas_model.get();
}

/**
 * @brief Find the immutable reacting-gas model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_reacting_gas_model() const noexcept -> const reacting_gas_model_type*
{
    return d_reacting_gas_model.get();
}

/**
 * @brief Configure the optional boiling source from explicit options.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Bulk and wall boiling configuration.
 * @return Configured boiling source model.
 * @throws std::invalid_argument if options conflict with active reacting_gas.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_boiling_source(const BoilingSourceOptions& options)
    -> boiling_source_model_type&
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
    if ((options.enable_bulk_boiling || options.enable_wall_boiling) && d_reacting_gas_model &&
        d_reacting_gas_model->supplies_void_fraction())
    {
        throw std::invalid_argument("Independent boiling cannot be combined with schema-owned "
                                    "reacting_gas until vapor mass is coupled to bubble inventories.");
    }
    d_solver->d_physical_model_enabled = true;
    ensure_scalar_void_fraction_model();
    if (!d_boiling_source_model)
    {
        d_boiling_source_model = std::make_unique<boiling_source_model_type>(d_solver->d_mesh, options);
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
auto MultiphysicsSolverCoupler<Pack>::configure_boiling_source(const Database& database) -> boiling_source_model_type&
{
    return configure_boiling_source(boiling_source_options_from_database(database));
}

/**
 * @brief Remove the optional boiling source model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_boiling_source_model()
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
auto MultiphysicsSolverCoupler<Pack>::find_boiling_source_model() noexcept -> boiling_source_model_type*
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
auto MultiphysicsSolverCoupler<Pack>::find_boiling_source_model() const noexcept -> const boiling_source_model_type*
{
    return d_boiling_source_model.get();
}

/**
 * @brief Configure the authoritative scalar void-fraction model.
 *
 * @tparam Pack Tpetra type pack.
 * @param options Void bounds, initial value, and collapse options.
 * @return Configured scalar void-fraction model.
 * @throws std::invalid_argument if options conflict with Schema-owned reacting gas.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_scalar_void_fraction(const ScalarVoidFractionOptions& options)
    -> scalar_void_fraction_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "scalar void, then configure free surface last.");
    }
    validate_scalar_void_fraction_options(options);
    if (d_free_surface_model && options.initial_alpha != scalar_type{} &&
        !(d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction()))
    {
        throw std::invalid_argument("The planar free-surface budget rejects nonzero scalar void "
                                    "without a conservative gas-inventory owner.");
    }
    if (d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction() &&
        (options.alpha_min != d_reacting_gas_model->options().alpha_min ||
            options.alpha_max != d_reacting_gas_model->options().alpha_max))
    {
        throw std::invalid_argument("A Schema-owned reacting gas mirror requires its scalar void bounds.");
    }
    if (d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction() &&
        std::isfinite(options.alpha_collapse_time))
    {
        throw std::invalid_argument("Schema-owned reacting gas cannot use scalar void collapse until "
                                    "bubble inventory removal is coupled conservatively.");
    }
    d_solver->d_physical_model_enabled = true;
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model = std::make_unique<scalar_void_fraction_model_type>(d_solver->d_mesh, options);
    }
    else
    {
        d_scalar_void_fraction_model->configure(options);
    }
    d_scalar_void_fraction_explicitly_configured = true;
    if (d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction() &&
        d_reacting_gas_model->initial_state_initialized())
    {
        d_scalar_void_fraction_model->initialize_from(d_reacting_gas_model->alpha_g());
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
auto MultiphysicsSolverCoupler<Pack>::configure_scalar_void_fraction(const Database& database)
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
auto MultiphysicsSolverCoupler<Pack>::find_scalar_void_fraction_model() noexcept -> scalar_void_fraction_model_type*
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
auto MultiphysicsSolverCoupler<Pack>::find_scalar_void_fraction_model() const noexcept
    -> const scalar_void_fraction_model_type*
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
auto MultiphysicsSolverCoupler<Pack>::configure_material_feedback(const MaterialFeedbackOptions& options)
    -> material_feedback_model_type&
{
    if (d_free_surface_model && d_free_surface_model->initialized())
    {
        throw std::logic_error("Remove the initialized free-surface model before reconfiguring "
                               "material feedback, then configure free surface last.");
    }
    validate_material_feedback_options(options);
    d_solver->d_physical_model_enabled = true;
    if (!d_material_feedback_model)
    {
        d_material_feedback_model = std::make_unique<material_feedback_model_type>(d_solver->d_mesh, options);
    }
    else
    {
        d_material_feedback_model->configure(options);
    }
    d_solver->d_model_options.density_feedback_enabled = d_material_feedback_model->density_feedback_enabled();
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
auto MultiphysicsSolverCoupler<Pack>::configure_material_feedback(const Database& database)
    -> material_feedback_model_type&
{
    return configure_material_feedback(material_feedback_options_from_database(
        database, d_solver->d_model_options, d_solver->d_problem.time_options()));
}

/**
 * @brief Remove the optional material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return true if a model was removed.
 */
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_material_feedback_model()
{
    if (!d_material_feedback_model)
        return false;
    if (d_free_surface_model)
    {
        remove_free_surface_model();
    }
    d_material_feedback_model.reset();
    d_solver->d_model_options.density_feedback_enabled = false;
    return true;
}

/**
 * @brief Find the mutable material-feedback model.
 *
 * @tparam Pack Tpetra type pack.
 * @return Configured model, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_material_feedback_model() noexcept -> material_feedback_model_type*
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
auto MultiphysicsSolverCoupler<Pack>::find_material_feedback_model() const noexcept
    -> const material_feedback_model_type*
{
    return d_material_feedback_model.get();
}

/** @brief Configure optional fixed-grid or solver-integrated planar free-surface coupling. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_free_surface(const FreeSurfaceOptions& options)
    -> free_surface_model_type*
{
    return configure_free_surface(options, nullptr);
}

template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::configure_free_surface(
    const FreeSurfaceOptions& options, std::unique_ptr<HeadspaceModel> supplied_headspace) -> free_surface_model_type*
{
    const auto& mesh = *d_solver->d_mesh;
    collective_detail::require_uniform_value(
        mesh, static_cast<int>(bool(supplied_headspace)), "Supplied headspace selection");
    std::string supplied_signature;
    collective_detail::collective_local_validation(mesh, "Supplied headspace metadata",
        [&]
        {
            const bool typed =
                d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled();
            if (supplied_headspace)
            {
                auto bound_options = options;
                bound_options.headspace = supplied_headspace->options();
                if (supplied_headspace->mode() != options.headspace.mode ||
                    encode_free_surface_options(bound_options) != encode_free_surface_options(options))
                    throw std::invalid_argument("Supplied headspace options do not match the configured surface.");
                const auto& metadata = supplied_headspace->metadata();
                if (supplied_headspace->state().volume != real_t{} || !supplied_headspace->headspaceMoles().empty() ||
                    !supplied_headspace->ventedMoles().empty())
                    throw std::invalid_argument("Inject a fresh uninitialized headspace closure.");
                supplied_signature = encode_headspace_metadata(metadata);
                if (supplied_signature.size() > std::size_t(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("Supplied headspace metadata exceeds MPI ordinal range.");
                if (typed && options.headspace.mode == HeadspaceMode::Closed)
                {
                    const auto& catalog = d_reacting_gas_model->options().reaction_mechanism->catalog();
                    if (options.headspace.infer_initial_moles ||
                        metadata.catalog_fingerprint != catalog.fingerprint() || metadata.provider_identity.empty() ||
                        metadata.provider_revision == 0 || !metadata.prescribed_temperature_only)
                        throw std::invalid_argument(
                            "Typed closed headspace requires an explicit catalog-bound prescribed-T EOS.");
                    for (const auto& [name, amount] : options.headspace.initial_moles)
                    {
                        const auto found =
                            std::ranges::find(catalog.species(), name, &Reacting::SpeciesDefinition::name);
                        if (found == catalog.species().end() || !std::isfinite(amount) || amount < 0 ||
                            (amount > 0 && !catalog.allows(found->id, Reacting::SpeciesPhase::Resolved) &&
                                !catalog.allows(found->id, Reacting::SpeciesPhase::Dispersed)))
                            throw std::invalid_argument(
                                "Initial headspace gas must name actual allowed catalog species.");
                    }
                }
            }
            else if (options.enabled && typed && options.headspace.mode == HeadspaceMode::Closed)
                throw std::invalid_argument(
                    "Typed closed coupling requires explicit supplied HeadspaceModel ownership.");
        });
    collective_detail::require_uniform_value(mesh, supplied_signature.size(), "Headspace signature size");
    std::string root_signature = supplied_signature;
    if (!root_signature.empty())
        Teuchos::broadcast(
            *mesh.owned_cell_map()->getComm(), 0, static_cast<int>(root_signature.size()), root_signature.data());
    collective_detail::collective_local_validation(mesh, "Collective headspace metadata",
        [&]
        {
            if (root_signature != supplied_signature)
                throw std::invalid_argument("Headspace provider metadata differs across ranks.");
        });
    const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
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
    validate_free_surface_configuration(options, supplied_headspace.get());
    if (options.enabled && options.mode == FreeSurfaceMode::PlanarALE)
    {
        PlanarALEMeshMotionOptions motion_options;
        motion_options.axis = options.gravity_axis;
        motion_options.deformation_start_elevation = options.ale.deformation_start_elevation;
        motion_options.maximum_level_change = options.ale.maximum_level_change;
        motion_options.quality_limits = options.ale.quality_limits;
        motion_options.gcl_absolute_tolerance = options.ale.gcl_absolute_tolerance;
        motion_options.gcl_relative_tolerance = options.ale.gcl_relative_tolerance;
        const auto volume_map = make_vessel_volume_map(options);
        if (d_ale_motion)
        {
            if (!d_ale_boundary || d_ale_boundary->axis() != options.gravity_axis ||
                d_ale_boundary->name() != options.ale.top_boundary)
            {
                throw std::invalid_argument(
                    "Remove the active planarALE model before changing its motion axis or top boundary.");
            }
            ale_boundary_type geometry_preflight(d_solver->d_mesh, options.ale.top_boundary, options.gravity_axis,
                *volume_map, options.coupling.volume_absolute_tolerance, options.coupling.volume_relative_tolerance);
        }
        else
        {
            // Constructor/destructor acquire and release the motion lease
            // without changing geometry; the boundary check proves current
            // patch planarity, area, and vessel/mesh-volume consistency.
            ale_motion_type motion_preflight(d_solver->mutable_mesh_handle(), std::move(motion_options));
            ale_boundary_type geometry_preflight(d_solver->d_mesh, options.ale.top_boundary, options.gravity_axis,
                *volume_map, options.coupling.volume_absolute_tolerance, options.coupling.volume_relative_tolerance);
        }
    }
    if (!options.enabled)
    {
        remove_free_surface_model();
        d_free_surface_options = options;
        return nullptr;
    }
    if (!d_solver->physical_transport_enabled())
    {
        throw std::invalid_argument(
            "The planar free-surface budget requires the dimensional physical-model Boussinesq constructor or an "
            "already configured physical model.");
    }

    auto model = [&]()
    {
        if (!supplied_headspace)
            return make_planar_free_surface_model(options);
        auto volume_map = make_vessel_volume_map(options);
        real_t underflow{}, overflow{};
        if (options.initial_clear_level)
        {
            const auto range = volume_map->evaluateVolumeBelow(*options.initial_clear_level);
            underflow = range.underflow;
            overflow = range.overflow;
        }
        return std::make_unique<free_surface_model_type>(std::move(volume_map), std::move(supplied_headspace),
            options.coupling, options.validity_warning_relative_level_change, underflow, overflow);
    }();
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled())
    {
        const auto mesh = d_solver->d_mesh;
        collective_detail::collective_local_validation(*mesh, "Surface collective guard installation",
            [&]
            {
                model->install_collective_guards([mesh](std::string_view label, const std::function<void()>& operation)
                    { collective_detail::collective_local_validation(*mesh, label, operation); },
                    [mesh](std::string_view label, const std::string& local)
                    {
                        collective_detail::collective_local_validation(*mesh, std::string(label) + " size",
                            [&]
                            {
                                if (local.size() > std::size_t(std::numeric_limits<int>::max()))
                                    throw std::invalid_argument("Replicated surface result exceeds MPI ordinal range.");
                            });
                        collective_detail::require_uniform_value(*mesh, static_cast<int>(local.size()), label);
                        std::string root;
                        collective_detail::collective_local_validation(
                            *mesh, std::string(label) + " buffer", [&] { root = local; });
                        if (!root.empty())
                            Teuchos::broadcast(
                                *mesh->owned_cell_map()->getComm(), 0, static_cast<int>(root.size()), root.data());
                        collective_detail::collective_local_validation(*mesh, label,
                            [&]
                            {
                                if (root != local)
                                    throw std::invalid_argument("Headspace/volume result differs across ranks.");
                            });
                    });
            });
    }
    auto inventory = std::make_unique<liquid_mass_inventory_type>(d_solver->d_mesh, options.liquid_mass);
    auto clear_level = std::make_unique<field_type>(d_solver->d_mesh, "clearLevel");
    auto pool_level = std::make_unique<field_type>(d_solver->d_mesh, "poolLevel");
    auto headspace_pressure = std::make_unique<field_type>(d_solver->d_mesh, "headspacePressure");
    auto pool_occupancy = std::make_unique<field_type>(d_solver->d_mesh, "poolOccupancy");

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
auto MultiphysicsSolverCoupler<Pack>::configure_free_surface(const Database& database) -> free_surface_model_type*
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
    const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
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
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_free_surface_model()
{
    const auto removed = static_cast<bool>(d_free_surface_model) || static_cast<bool>(d_liquid_mass_inventory);
    const bool removed_planar_ale = static_cast<bool>(d_ale_motion);
    if (removed && d_boiling_source_model)
    {
        const auto local_steam_mass = d_boiling_source_model->global_submerged_steam_mass();
        const int local_invalid_mass = !std::isfinite(local_steam_mass) || local_steam_mass < scalar_type{} ? 1 : 0;
        int any_invalid_mass = 0;
        const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
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
        d_solver->boussinesq_velocity_boundary_cache() =
            FVM::cache_velocity_boundary_conditions<Pack>(d_solver->d_mesh, d_solver->d_problem.boundary_conditions());
    }
    d_solver->clear_volume_continuity_target();
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
    d_species_surface_accounting = {};
    d_pool_occupancy_volume_error = {};
    d_free_surface_history.clear();
    d_planar_ale_diagnostics = {};
    d_ale_target_generation = 0;
    return removed;
}

/** @brief Find mutable configured free-surface state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_free_surface_model() noexcept -> free_surface_model_type*
{
    return d_free_surface_model.get();
}

/** @brief Find immutable configured free-surface state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_free_surface_model() const noexcept -> const free_surface_model_type*
{
    return d_free_surface_model.get();
}

/** @brief Return the accepted free-surface diagnostics. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::free_surface_diagnostics() const -> free_surface_diagnostics_type
{
    if (!d_free_surface_model)
    {
        throw std::logic_error("BoussinesqSolver has no configured free-surface model.");
    }
    return d_free_surface_model->diagnostics();
}

/** @brief Return mutable liquid-mass state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::liquid_mass_inventory() -> liquid_mass_inventory_type&
{
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("BoussinesqSolver has no configured liquid-mass inventory.");
    }
    return *d_liquid_mass_inventory;
}

/** @brief Return immutable liquid-mass state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::liquid_mass_inventory() const -> const liquid_mass_inventory_type&
{
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("BoussinesqSolver has no configured liquid-mass inventory.");
    }
    return *d_liquid_mass_inventory;
}

/** @brief Find mutable liquid-mass state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_liquid_mass_inventory() noexcept -> liquid_mass_inventory_type*
{
    return d_liquid_mass_inventory.get();
}

/** @brief Find immutable liquid-mass state. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_liquid_mass_inventory() const noexcept -> const liquid_mass_inventory_type*
{
    return d_liquid_mass_inventory.get();
}

/** @brief Return the pure-liquid density field. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::rho_liquid() const -> const field_type&
{
    return liquid_mass_inventory().rhoLiquid();
}

/** @brief Return the replicated clear-level field. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::clear_level() const -> const field_type&
{
    if (!d_clear_level)
    {
        throw std::logic_error("BoussinesqSolver has no configured clear-level field.");
    }
    return *d_clear_level;
}

/** @brief Return the replicated pool-level field. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::pool_level() const -> const field_type&
{
    if (!d_pool_level)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-level field.");
    }
    return *d_pool_level;
}

/** @brief Return the replicated absolute headspace-pressure field. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::headspace_pressure() const -> const field_type&
{
    if (!d_headspace_pressure)
    {
        throw std::logic_error("BoussinesqSolver has no configured headspace-pressure field.");
    }
    return *d_headspace_pressure;
}

/** @brief Return the approximate cell-centre pool indicator. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::pool_occupancy() const -> const field_type&
{
    if (!d_pool_occupancy)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-occupancy field.");
    }
    return *d_pool_occupancy;
}

/** @brief Return the signed cell-centre occupancy-volume error. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::pool_occupancy_volume_error() const -> scalar_type
{
    if (!d_pool_occupancy)
    {
        throw std::logic_error("BoussinesqSolver has no configured pool-occupancy field.");
    }
    return d_pool_occupancy_volume_error;
}

/** @brief Return the accepted carrier flux relative to the moving mesh. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::mesh_relative_face_fluxes() const -> const face_flux_field_type&
{
    if (!d_mesh_relative_face_flux)
    {
        throw std::logic_error("BoussinesqSolver has no configured planar-ALE relative-flux field.");
    }
    return *d_mesh_relative_face_flux;
}

/** @brief Return immutable accepted free-surface history. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::free_surface_history() const noexcept
    -> const std::vector<FreeSurfaceHistoryRecord>&
{
    return d_free_surface_history;
}

/** @brief Write accepted free-surface history on mesh rank zero. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::write_free_surface_history_csv(const std::string& filename) const
{
    if (d_solver->d_mesh->owned_cell_map()->getComm()->getRank() != 0)
    {
        return;
    }
    std::ofstream output(filename, std::ios::out | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("Could not create free-surface history CSV file '" + filename + "'.");
    }
    if (species_surface_enabled())
    {
        const auto quote = [](std::string value)
        {
            std::string result = "\"";
            for (const auto character : value)
            {
                if (character == '"')
                    result += '"';
                result += character;
            }
            return result + "\"";
        };
        output << "time_s,time_step_s,schema_fingerprint,event_revision,publication_nonce,species_id,species_name,"
                  "gas_subset_generated_mol,gas_subset_consumed_mol,gas_subset_net_source_mol,dissolved_mol,submerged_"
                  "mol,headspace_mol,vented_mol,"
                  "escaped_this_step_mol,other_sink_mol,closure_mol,liquid_mass_kg,liquid_volume_m3,pool_volume_m3,"
                  "headspace_pressure_pa,headspace_temperature_k,headspace_thermal_contract";
        for (const auto& population : d_reacting_gas_model->options().population_ids)
            output << ',' << quote("population_moles:" + population);
        output << '\n';
        output << std::setprecision(std::numeric_limits<real_t>::max_digits10);
        const auto& catalog = d_reacting_gas_model->options().reaction_mechanism->catalog();
        for (const auto& record : d_free_surface_history)
        {
            if (!record.catalogued_species || record.species_event.schema_fingerprint != catalog.fingerprint())
                throw std::logic_error("History record belongs to a foreign chemical schema.");
            const auto& surface = record.free_surface;
            for (const auto& species : catalog.species())
            {
                const auto value = [&](const GasMolesBySpecies& amounts)
                { return free_surface_species_moles(amounts, species.name); };
                if (!surface.net_source_gas_moles)
                    throw std::logic_error("Typed history lacks its authoritative signed source report.");
                output << surface.time << ',' << surface.time_step << ',' << record.species_event.schema_fingerprint
                       << ',' << record.species_event.revision << ',' << record.species_event.publication_nonce << ','
                       << species.id << ',' << quote(species.name) << ',' << value(surface.generated_gas_moles) << ','
                       << value(surface.consumed_gas_moles) << ',' << value(*surface.net_source_gas_moles) << ','
                       << value(surface.dissolved_gas_moles) << ',' << value(surface.submerged_gas_moles) << ','
                       << value(surface.headspace_gas_moles) << ',' << value(surface.vented_gas_moles) << ','
                       << value(surface.escaped_gas_moles_this_step) << ',' << value(surface.other_sink_gas_moles)
                       << ',' << value(surface.gas_closure_by_species) << ',' << record.liquid_mass.total_mass << ','
                       << surface.liquid_volume << ',' << surface.pool_volume << ',' << surface.headspace.pressure
                       << ',' << surface.headspace.temperature << ",prescribed_temperature_pressure_only";
                for (const auto& population : d_reacting_gas_model->options().population_ids)
                {
                    const auto found = surface.submerged_population_gas_moles.find(population);
                    output << ','
                           << (found == surface.submerged_population_gas_moles.end() ? real_t{} : value(found->second));
                }
                output << '\n';
            }
        }
        return;
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
              "nonlinear_residual_pa,"
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
                  "ale_energy_residual_normalized,ale_rejected_transactions,ale_mesh_volume_correction_m3_per_s";
    }
    output << '\n';
    output << std::setprecision(std::numeric_limits<scalar_type>::max_digits10);
    for (const auto& record : d_free_surface_history)
    {
        const auto& surface = record.free_surface;
        const auto& liquid = record.liquid_mass;
        const auto boiling = record.boiling.value_or(boiling_diagnostics_type{});
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
               << surface.gas_closure_residual << ',' << surface.normalized_gas_closure_residual << ','
               << boiling.requested_evaporation_mass << ',' << boiling.accepted_evaporation_mass << ','
               << boiling.rejected_vapor_mass << ',' << boiling.condensed_liquid_mass << ','
               << boiling.submerged_steam_mass << ',' << boiling.submerged_steam_volume << ','
               << boiling.condensation_latent_energy_release << ',' << boiling.mass_balance_residual << ','
               << boiling.void_balance_residual << ',' << boiling.latent_energy_balance_residual;
        if (include_planar_ale)
        {
            const auto ale = record.planar_ale.value_or(PlanarALEStepDiagnostics{});
            output << ',' << ale.old_geometry_epoch << ',' << ale.new_geometry_epoch << ',' << ale.old_mesh_volume
                   << ',' << ale.new_mesh_volume << ',' << ale.mesh_vessel_mismatch << ',' << ale.maximum_gcl_residual
                   << ',' << ale.maximum_normalized_gcl_residual << ',' << ale.outer_correctors << ','
                   << ale.level_residual << ',' << ale.pressure_residual << ',' << ale.material_state_residual << ','
                   << ale.gas_state_residual << ',' << ale.volume_source.global_material_source << ','
                   << ale.volume_source.global_carrier_transport << ','
                   << ale.volume_source.global_bubble_slip_divergence << ','
                   << ale.volume_source.bubble_escape_volume_rate << ','
                   << ale.volume_source.source_pool_closure_residual << ',' << ale.continuity.l2 << ','
                   << ale.continuity.maximum << ',' << ale.continuity.normalized_l2 << ',' << ale.liquid_mass_residual
                   << ',' << ale.gas_inventory_residual << ',' << ale.energy_residual << ','
                   << ale.normalized_energy_residual << ',' << ale.rejected_transactions << ','
                   << ale.volume_source.global_mesh_volume_correction;
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
template<TpetraTypePack Pack> bool MultiphysicsSolverCoupler<Pack>::remove_temperature_source(const std::string& name)
{
    return d_solver->stored_temperature_sources().remove(name);
}

/**
 * @brief Find a mutable named temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return Source pointer, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_temperature_source(const std::string& name) noexcept
    -> volumetric_source_type*
{
    return d_solver->stored_temperature_sources().find(name);
}

/**
 * @brief Find an immutable named temperature source.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Registered source name.
 * @return Source pointer, or nullptr when absent.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::find_temperature_source(const std::string& name) const noexcept
    -> const volumetric_source_type*
{
    return d_solver->stored_temperature_sources().find(name);
}

/**
 * @brief Refresh material properties, feedback, and registered heat sources.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::refresh_physical_models()
{
    const auto phase_timing = d_solver->d_solver_timings.scope(SolverPhase::PhysicalModels);
    update_context_type context{d_solver->d_time, d_solver->d_step_index, *d_solver->d_mesh, d_solver->temperature(),
        d_solver->pressure(), d_solver->velocity()};
    d_solver->stored_material_properties().update(context);
    if (d_free_surface_model && !d_free_surface_model->initialized() && d_material_feedback_model)
    {
        // Lazy free-surface initialization needs the same accepted material
        // state as explicit primary-field initialization before reacting
        // bubble radii and liquid reference mass are reconstructed.
        refresh_material_feedback(d_solver->d_time);
    }
    // Explicit primary-field initialization owns typed finite inventories. A
    // first-step material refresh must not reset them or invalidate the snapshot
    // already captured for rollback. The no-gas refresh behavior is retained.
    initialize_reacting_gas_state(d_solver->d_step_index == 0 &&
        !(d_free_surface_model && d_free_surface_model->initialized()) &&
        !(d_reacting_gas_model && d_reacting_gas_model->species_enabled()));
    refresh_material_feedback(d_solver->d_time);
    d_solver->stored_temperature_sources().update(context);
}

/**
 * @brief Initialize Schema-owned reacting gas inventories and dependent liquid fractions.
 *
 * @tparam Pack Tpetra type pack.
 * @param force Whether to reinitialize an already initialized model.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::initialize_reacting_gas_state(bool force)
{
    if (!d_reacting_gas_model || !d_reacting_gas_model->supplies_void_fraction() ||
        (!force && d_reacting_gas_model->initial_state_initialized()))
    {
        return;
    }

    d_reacting_gas_model->initialize_state(d_solver->d_time, d_solver->temperature(), d_solver->pressure(),
        d_solver->velocity(), d_solver->stored_material_properties(), force);
    if (d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model->initialize_from(d_reacting_gas_model->alpha_g());
    }
}

/**
 * @brief Apply material feedback for the supplied physical time.
 *
 * @tparam Pack Tpetra type pack.
 * @param time Physical time passed to feedback callbacks.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::refresh_material_feedback(scalar_type time)
{
    if (!d_material_feedback_model)
    {
        return;
    }
    update_context_type context{time, d_solver->d_step_index, *d_solver->d_mesh, d_solver->temperature(),
        d_solver->pressure(), d_solver->velocity()};
    d_material_feedback_model->apply(context, active_alpha_g_field(), d_solver->stored_material_properties());
}

/** @brief Reject rank-divergent optional-model ownership before conditional collectives. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::validate_collective_model_state() const
{
    const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
    constexpr int absent = -1;
    const std::array<int, 26> local_model_state{static_cast<int>(d_free_surface_model != nullptr),
        d_free_surface_model ? static_cast<int>(d_free_surface_model->initialized()) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.enabled) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.mode) : absent,
        d_free_surface_model ? static_cast<int>(d_free_surface_options.headspace.mode) : absent,
        static_cast<int>(d_liquid_mass_inventory != nullptr),
        d_liquid_mass_inventory ? static_cast<int>(d_liquid_mass_inventory->initialized()) : absent,
        static_cast<int>(d_clear_level != nullptr), static_cast<int>(d_pool_level != nullptr),
        static_cast<int>(d_headspace_pressure != nullptr), static_cast<int>(d_pool_occupancy != nullptr),
        static_cast<int>(d_reacting_gas_model != nullptr),
        d_reacting_gas_model ? static_cast<int>(d_reacting_gas_model->mode()) : absent,
        d_reacting_gas_model ? static_cast<int>(d_reacting_gas_model->options().pressure_mode) : absent,
        d_reacting_gas_model ? static_cast<int>(d_reacting_gas_model->initial_state_initialized()) : absent,
        static_cast<int>(d_boiling_source_model != nullptr),
        d_boiling_source_model ? static_cast<int>(d_boiling_source_model->enabled()) : absent,
        d_boiling_source_model ? static_cast<int>(d_boiling_source_model->phase_change_completion_pending()) : absent,
        static_cast<int>(d_scalar_void_fraction_model != nullptr),
        static_cast<int>(d_material_feedback_model != nullptr),
        d_material_feedback_model ? static_cast<int>(d_material_feedback_model->options().density_mode) : absent,
        static_cast<int>(d_prescribed_heat_source != nullptr),
        static_cast<int>(d_solver->stored_turbulence_model().enabled()),
        static_cast<int>(d_solver->d_physical_model_enabled), static_cast<int>(d_solver->d_primary_fields_initialized),
        static_cast<int>(d_free_surface_step_failed)};
    std::array<int, 26> minimum_model_state{};
    std::array<int, 26> maximum_model_state{};
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
    collective_detail::require_uniform_value(*d_solver->d_mesh,
        static_cast<int>(d_reacting_gas_model && bool(d_reacting_gas_model->options().bouss_reference)),
        "Bouss reference thermal owner presence");
}

/**
 * @brief Validate free-surface ownership and pressure-coupling seams.
 *
 * Scalar void is accepted only when a conservative model owns it.  The
 * schema-owned reacting gas model owns its bounded mirror; standalone boiling owns only
 * the globally tracked submerged-steam volume.
 */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::validate_free_surface_configuration(
    const FreeSurfaceOptions& options, const HeadspaceModel* proposed_headspace) const
{
    if (d_reacting_gas_model && d_reacting_gas_model->options().bouss_reference)
        throw std::invalid_argument("Reference-volume Bouss vapor has no supplied-EOS free-surface/ALE contract.");

    validate_free_surface_options(options);
    if (!options.enabled)
    {
        return;
    }
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled())
    {
        if (d_reacting_gas_model->options().source_schedule ==
            ReactingSourceSchedule::PrescribedCarrierAfterTemperature)
            throw std::invalid_argument("Post-temperature gas references cannot configure a free-surface owner.");
        if (options.liquid_mass.mode != LiquidVolumeMode::CellMassInventory ||
            options.liquid_mass.depletion_policy != FreeSurfaceRangePolicy::Error ||
            (d_boiling_source_model && d_boiling_source_model->enabled()))
            throw std::invalid_argument(
                "Typed species surface requires a cellMassInventory/error follower and no independent boiling owner.");
        if (options.headspace.mode == HeadspaceMode::Closed)
        {
            const auto* closure = proposed_headspace
                                      ? proposed_headspace
                                      : (d_free_surface_model ? &d_free_surface_model->headspace() : nullptr);
            if (!closure || options.headspace.infer_initial_moles)
                throw std::invalid_argument("Typed closed headspace requires an explicit catalog-bound closure.");
            const auto& metadata = closure->metadata();
            const auto& bounds = metadata.validity_bounds;
            if (metadata.catalog_fingerprint !=
                    d_reacting_gas_model->options().reaction_mechanism->catalog().fingerprint() ||
                metadata.provider_identity.empty() || metadata.provider_revision == 0 ||
                !metadata.prescribed_temperature_only ||
                !std::ranges::all_of(bounds, [](auto value) { return std::isfinite(value); }) ||
                !(bounds[0] > 0 && bounds[1] > bounds[0] && bounds[2] > 0 && bounds[3] > bounds[2]))
                throw std::invalid_argument("Supplied closed-mixture metadata/schema/bounds are invalid.");
        }
    }
    if (options.headspace.temperature_mode == HeadspaceTemperatureMode::External &&
        (options.mode != FreeSurfaceMode::PlanarALE || !d_thermal_participant))
        throw std::invalid_argument("External headspace temperature requires a typed planar ALE thermal participant.");
    if (options.mode == FreeSurfaceMode::PlanarALE)
    {
        validate_planar_ale_support(options);
    }

    if (d_reacting_gas_model && d_reacting_gas_model->mode() == ReactingGasMode::Species &&
        d_reacting_gas_model->options().pressure_mode != ReactingPressureMode::Constant &&
        d_reacting_gas_model->options().pressure_mode != ReactingPressureMode::Reconstructed)
    {
        throw std::invalid_argument("Planar free-surface coupling requires constant or reconstructed "
                                    "reacting absolute pressure.");
    }
    if (options.headspace.mode == HeadspaceMode::Closed && d_boiling_source_model && d_boiling_source_model->enabled())
    {
        throw std::invalid_argument(
            "Closed free-surface coupling cannot use the fixed-temperature boiling model; pressure-dependent "
            "saturation from absolute headspace pressure is not implemented.");
    }

    if (!d_scalar_void_fraction_model || (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
                                             d_reacting_gas_model->supplies_void_fraction()))
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
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::validate_planar_ale_runtime_parity() const
{
    const auto* feedback_options = d_material_feedback_model ? &d_material_feedback_model->options() : nullptr;
    const auto* gas_options = d_reacting_gas_model ? &d_reacting_gas_model->options() : nullptr;
    std::vector<std::pair<std::string, bool>> temperature_sources;
    temperature_sources.reserve(d_solver->stored_temperature_sources().entries().size());
    for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
    {
        temperature_sources.emplace_back(name, source->enabled());
    }
    const auto local_encoding = encode_planar_ale_runtime_options(d_solver->d_problem.time_options(),
        d_solver->d_problem.linear_options(), d_solver->pressure_linear_solver_options(),
        d_solver->momentum_linear_solver_options(), d_solver->d_model_options, feedback_options, gas_options,
        d_solver->d_problem.boundary_conditions(), temperature_sources, d_solver->has_mutable_mesh_handle(),
        d_solver->uses_legacy_backend(), d_solver->physical_transport_enabled(),
        d_solver->stored_turbulence_model().enabled(), d_boiling_source_model && d_boiling_source_model->enabled(),
        d_scalar_void_fraction_explicitly_configured,
        d_solver->stored_material_properties().has_updater(),
        d_solver->stored_temperature_sources().has_dynamic_updates());

    const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    long long root_size = rank == 0 ? static_cast<long long>(local_encoding.size()) : 0LL;
    Teuchos::broadcast(*communicator, 0, 1, &root_size);
    if (root_size < 0 || root_size > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("planarALE runtime-option encoding is too large for MPI broadcast.");
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
void MultiphysicsSolverCoupler<Pack>::validate_planar_ale_support(const FreeSurfaceOptions& options) const
{
    validate_planar_ale_runtime_parity();
    if (!d_solver->has_mutable_mesh_handle() || d_solver->uses_legacy_backend())
    {
        throw std::invalid_argument(
            std::string(planar_ale_unavailable_diagnostic) + " A const-only or legacy solver mesh cannot be moved.");
    }
    if (!d_solver->physical_transport_enabled())
    {
        throw std::invalid_argument("planarALE requires dimensional physical Boussinesq temperature transport.");
    }
    if (options.vessel.mode != VesselVolumeMapMode::ConstantArea ||
        options.liquid_mass.mode != LiquidVolumeMode::CellMassInventory ||
        options.range_policy != FreeSurfaceRangePolicy::Error ||
        options.liquid_mass.depletion_policy != FreeSurfaceRangePolicy::Error ||
        options.headspace.mode != HeadspaceMode::Vented ||
        (options.headspace.temperature_mode != HeadspaceTemperatureMode::Fixed &&
            !(options.headspace.temperature_mode == HeadspaceTemperatureMode::External && d_thermal_participant)))
    {
        throw std::invalid_argument("planarALE initially requires constantArea, cellMassInventory, fixed or typed "
                                    "external-temperature vented headspace, "
                                    "and error range/depletion policies.");
    }
    if (d_solver->stored_turbulence_model().enabled())
    {
        throw std::invalid_argument("planarALE does not yet support RANS transports, wall distance, or wall laws.");
    }
    if (d_boiling_source_model && d_boiling_source_model->enabled())
    {
        throw std::invalid_argument(
            "planarALE initially rejects boiling/steam transport.");
    }
    if (d_scalar_void_fraction_explicitly_configured &&
        (!d_reacting_gas_model || !d_reacting_gas_model->supplies_void_fraction()))
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
    const auto gravity = d_solver->d_problem.time_options().gravity_vector();
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
    if (d_solver->stored_material_properties().has_updater() ||
        d_solver->stored_temperature_sources().has_dynamic_updates())
    {
        throw std::invalid_argument(
            "planarALE rejects dynamic user material/source callbacks because they are not rollback-safe.");
    }

    // A spatially anchored prescribed_heat profile would retain stale cell values after
    // the mesh moves.  Until that source owns an epoch-aware refresh contract,
    // every ALE heat/reacting_gas use must be geometry-invariant.
    if (d_prescribed_heat_source && !d_prescribed_heat_source->has_interval_energy())
    {
        scalar_type local_minimum_prescribed_heat = std::numeric_limits<scalar_type>::infinity();
        scalar_type local_maximum_prescribed_heat = -std::numeric_limits<scalar_type>::infinity();
        for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
        {
            const auto value = d_prescribed_heat_source->field().value(static_cast<local_ordinal_type>(owned));
            local_minimum_prescribed_heat = std::min(local_minimum_prescribed_heat, value);
            local_maximum_prescribed_heat = std::max(local_maximum_prescribed_heat, value);
        }
        scalar_type global_minimum_prescribed_heat{};
        scalar_type global_maximum_prescribed_heat{};
        Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1,
            &local_minimum_prescribed_heat, &global_minimum_prescribed_heat);
        Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
            &local_maximum_prescribed_heat, &global_maximum_prescribed_heat);
        if (global_minimum_prescribed_heat != global_maximum_prescribed_heat)
        {
            throw std::invalid_argument("planarALE currently requires a uniform geometry-invariant prescribed_heat-power "
                                        "field; Gaussian and other spatial profiles are not epoch-aware.");
        }
    }

    if (d_reacting_gas_model && d_reacting_gas_model->enabled())
    {
        if (!d_reacting_gas_model->species_enabled() && !d_prescribed_heat_source)
        {
            throw std::invalid_argument("planarALE Schema-owned reacting gas requires an authoritative "
                                        "prescribed_heat-power source before setup.");
        }
        if (options.ale.maximum_correctors < 2)
        {
            throw std::invalid_argument("planarALE Schema-owned reacting gas requires at least two outer correctors to "
                                        "couple gas state and geometry.");
        }
        const auto& gas = d_reacting_gas_model->options();
        if (gas.mode != ReactingGasMode::Species || gas.solution_transport != ReactingTransportMode::Advective ||
            gas.bubble_transport != BubbleTransportMode::General ||
            (gas.pressure_mode != ReactingPressureMode::Constant &&
                gas.pressure_mode != ReactingPressureMode::Reconstructed) ||
            gas.free_surface_patches != std::vector<std::string>{options.ale.top_boundary})
        {
            throw std::invalid_argument(
                "planarALE reacting_gas requires Species, advective solution, general bubble "
                "transport, constant/reconstructed pressure, and exactly the configured moving-top escape patch.");
        }
    }

    const auto& boundaries = d_solver->d_problem.boundary_conditions();
    const auto pressure = boundaries.pressure.find(options.ale.top_boundary);
    if (pressure == boundaries.pressure.end() || pressure->second.type != BoundaryConditionType::Dirichlet ||
        pressure->second.value != scalar_type{})
    {
        throw std::invalid_argument("planarALE moving top requires a zero-gauge Dirichlet physical-pressure boundary.");
    }
    const auto velocity_boundary = boundaries.velocity.find(options.ale.top_boundary);
    if (velocity_boundary == boundaries.velocity.end() ||
        (velocity_boundary->second.type != BoundaryConditionType::Slip &&
            velocity_boundary->second.type != BoundaryConditionType::Dirichlet) ||
        (velocity_boundary->second.type == BoundaryConditionType::Dirichlet &&
            (velocity_boundary->second.value.x != scalar_type{} || velocity_boundary->second.value.y != scalar_type{} ||
                velocity_boundary->second.value.z != scalar_type{})))
    {
        throw std::invalid_argument(
            "planarALE moving top requires a configured slip or Dirichlet liquid-velocity boundary.");
    }
    int local_open_nonmoving_boundary = 0;
    for (const auto& [batch_id, batch] : d_solver->d_mesh->boundary_batches())
    {
        (void) batch;
        const auto name = d_solver->d_mesh->boundary_batch_name(batch_id);
        if (name == options.ale.top_boundary)
        {
            continue;
        }
        const auto velocity = boundaries.velocity.find(name);
        const auto pressure_condition = boundaries.pressure.find(name);
        const bool closed_velocity =
            velocity != boundaries.velocity.end() &&
            (velocity->second.type == BoundaryConditionType::NoSlip ||
                velocity->second.type == BoundaryConditionType::Slip ||
                (velocity->second.type == BoundaryConditionType::Dirichlet &&
                    velocity->second.value.x == scalar_type{} && velocity->second.value.y == scalar_type{} &&
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
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
        &local_open_nonmoving_boundary, &any_open_nonmoving_boundary);
    if (any_open_nonmoving_boundary != 0)
    {
        throw std::invalid_argument(
            "planarALE initially rejects physical liquid inlets/outlets; every nonmoving boundary must be "
            "closed with homogeneous Neumann pressure.");
    }
    for (const auto& [name, condition] : boundaries.temperature)
    {
        (void) name;
        if (condition.type != BoundaryConditionType::Neumann || condition.value != scalar_type{})
        {
            throw std::invalid_argument(
                "planarALE initially supports adiabatic temperature boundaries only; use volumetric heating.");
        }
    }
}

/** @brief Lazily create the sole motion controller after accepted state exists. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::initialize_planar_ale_if_needed()
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

    auto motion = std::make_unique<ale_motion_type>(d_solver->mutable_mesh_handle(), std::move(motion_options));
    auto boundary = std::make_unique<ale_boundary_type>(d_solver->d_mesh, d_free_surface_options.ale.top_boundary,
        d_free_surface_options.gravity_axis, d_free_surface_model->volumeMap(),
        d_free_surface_options.coupling.volume_absolute_tolerance,
        d_free_surface_options.coupling.volume_relative_tolerance);
    const auto surface = d_free_surface_model->diagnostics();
    const auto level_scale =
        std::max({scalar_type{1}, std::abs(surface.pool_level), std::abs(boundary->diagnostics().surface_elevation)});
    const auto volume_scale =
        std::max({scalar_type{1}, std::abs(surface.pool_volume), std::abs(boundary->diagnostics().global_mesh_volume)});
    const auto volume_tolerance =
        static_cast<scalar_type>(d_free_surface_options.coupling.volume_absolute_tolerance) +
        static_cast<scalar_type>(d_free_surface_options.coupling.volume_relative_tolerance) * volume_scale;
    const auto level_tolerance =
        d_free_surface_options.ale.level_tolerance_from_volume
            ? volume_tolerance / boundary->diagnostics().global_area
            : static_cast<scalar_type>(d_free_surface_options.ale.level_absolute_tolerance) +
                  static_cast<scalar_type>(d_free_surface_options.ale.level_relative_tolerance) * level_scale;
    if (std::abs(surface.pool_level - boundary->diagnostics().surface_elevation) > level_tolerance)
    {
        throw std::invalid_argument("planarALE initial pool level must coincide with the selected mesh-top elevation.");
    }
    if (std::abs(surface.pool_volume - boundary->diagnostics().global_mesh_volume) > volume_tolerance)
    {
        throw std::invalid_argument(
            "planarALE initial liquid-plus-bubble pool volume must equal the moving fluid-mesh volume.");
    }

    d_ale_boundary = std::move(boundary);
    d_ale_motion = std::move(motion);
    d_volume_continuity_model = std::make_unique<volume_continuity_model_type>(d_solver->d_mesh,
        d_free_surface_options.coupling.volume_absolute_tolerance,
        d_free_surface_options.coupling.volume_relative_tolerance);
    d_mesh_relative_face_flux = std::make_unique<face_flux_field_type>(d_solver->d_mesh, "meshRelativeFaceFlux");
    // Before the first motion trial the accepted mesh is stationary, so its
    // relative transport flux is the already accepted absolute flux. Preserve
    // that state when ALE is configured after earlier fixed-grid work.
    for (const auto face_lid : d_mesh_relative_face_flux->owned_face_ids())
    {
        d_mesh_relative_face_flux->set_owned_value(face_lid, d_solver->projected_face_fluxes().value(face_lid));
    }
    d_mesh_relative_face_flux->sync_ghosts();
    d_bubble_slip_volume_flux = std::make_unique<face_flux_field_type>(d_solver->d_mesh, "bubbleSlipVolumeFlux");
    d_mesh_volume_rate = std::make_unique<field_type>(d_solver->d_mesh, "meshVolumeRate");
    d_continuity_residual = std::make_unique<field_type>(d_solver->d_mesh, "continuityResidual");
    d_ale_old_density = std::make_unique<field_type>(d_solver->d_mesh, "aleAcceptedOldDensity");
    d_ale_old_heat_capacity = std::make_unique<field_type>(d_solver->d_mesh, "aleAcceptedOldHeatCapacity");
    d_planar_ale_diagnostics.initialized = true;
    d_planar_ale_diagnostics.old_geometry_epoch = d_solver->d_mesh->geometry_epoch();
    d_planar_ale_diagnostics.new_geometry_epoch = d_solver->d_mesh->geometry_epoch();
    d_planar_ale_diagnostics.old_mesh_volume = d_ale_boundary->diagnostics().global_mesh_volume;
    d_planar_ale_diagnostics.new_mesh_volume = d_ale_boundary->diagnostics().global_mesh_volume;
}

/** @brief Enforce top kinematics and derive a distinct carrier-relative flux. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::form_mesh_relative_flux(face_flux_field_type& absolute_flux)
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
auto MultiphysicsSolverCoupler<Pack>::material_volumes(const field_type& liquid_mass_density,
    const field_type& pure_liquid_density_field, std::span<const real_t> cell_volumes) const -> std::vector<scalar_type>
{
    if (cell_volumes.size() != d_solver->d_mesh->num_local_cells())
    {
        throw std::invalid_argument("planarALE material-volume assembly requires mesh-local cell-volume order.");
    }
    std::vector<scalar_type> result(cell_volumes.size());
    const field_type* raw_bubble = d_reacting_gas_model && d_reacting_gas_model->enabled()
                                       ? &d_reacting_gas_model->raw_bubble_volume_fraction()
                                       : nullptr;
    int local_invalid = 0;
    {
        // Owned storage is authoritative; overlap supplies only ghost cells.
        const auto density_owned = pure_liquid_density_field.owned_read_view();
        const auto density_local = pure_liquid_density_field.local_read_view();
        const auto mass_owned = liquid_mass_density.owned_read_view();
        const auto mass_local = liquid_mass_density.local_read_view();
        const auto bubble_owned = raw_bubble ? raw_bubble->owned_read_view() : decltype(mass_owned){};
        const auto bubble_local = raw_bubble ? raw_bubble->local_read_view() : decltype(mass_local){};
        for (size_t local = 0; local < result.size(); ++local)
        {
            const bool owned = local < d_solver->d_mesh->num_owned_cells();
            const auto density = owned ? density_owned(local, 0) : density_local(local, 0);
            const auto mass_density = owned ? mass_owned(local, 0) : mass_local(local, 0);
            const auto volume = static_cast<scalar_type>(cell_volumes[local]);
            const auto bubble_fraction =
                raw_bubble ? (owned ? bubble_owned(local, 0) : bubble_local(local, 0)) : scalar_type{};
            result[local] = volume * (mass_density / density + bubble_fraction);
            local_invalid = local_invalid || !std::isfinite(density) || density <= scalar_type{} ||
                            !std::isfinite(mass_density) || mass_density < scalar_type{} ||
                            !std::isfinite(bubble_fraction) || bubble_fraction < scalar_type{} ||
                            !std::isfinite(result[local]) || result[local] < scalar_type{};
        }
    }
    int any_invalid = 0;
    Teuchos::reduceAll(
        *d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::runtime_error("planarALE material-volume state is non-finite or physically invalid.");
    }
    return result;
}

/** @brief Volume-weighted mean of the dynamic/gauge pressure field. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::volume_weighted_mean_pressure() const -> scalar_type
{
    scalar_type local_pressure_volume{};
    scalar_type local_volume{};
    {
        const auto values = d_solver->pressure().owned_read_view();
        for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto volume = static_cast<scalar_type>(d_solver->d_mesh->cell_volume(cell));
            local_pressure_volume += values(owned, 0) * volume;
            local_volume += volume;
        }
    }
    std::array<scalar_type, 2> local{local_pressure_volume, local_volume};
    std::array<scalar_type, 2> global{};
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
        static_cast<int>(local.size()), local.data(), global.data());
    return global[1] > scalar_type{} ? global[0] / global[1] : scalar_type{};
}

/** @brief Integrate carrier-density*cp*T on supplied owned-cell geometry. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::total_sensible_energy(const field_type& density, const field_type& heat_capacity,
    std::span<const real_t> cell_volumes) const -> scalar_type
{
    if (cell_volumes.size() != d_solver->d_mesh->num_local_cells())
    {
        throw std::invalid_argument("planarALE energy accounting requires mesh-local cell-volume order.");
    }
    scalar_type local_energy{};
    {
        const auto density_values = density.owned_read_view();
        const auto heat_capacity_values = heat_capacity.owned_read_view();
        const auto temperature_values = d_solver->temperature().owned_read_view();
        for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
        {
            local_energy += static_cast<scalar_type>(cell_volumes[owned]) * density_values(owned, 0) *
                            heat_capacity_values(owned, 0) * temperature_values(owned, 0);
        }
    }
    scalar_type global_energy{};
    Teuchos::reduceAll(
        *d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_energy, &global_energy);
    return global_energy;
}

/** @brief Prescribe the exact swept-volume rate on the dynamic-pressure top. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::configure_ale_pressure_boundary(std::uint64_t generation)
{
    if (!d_active_ale || !d_ale_boundary)
    {
        throw std::logic_error("planarALE pressure boundary requires an active motion trial.");
    }
    d_solver->native_pressure_projection().set_fixed_boundary_flux_provider(
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
    d_solver->boussinesq_coupled_pressure_velocity_solver().set_fixed_boundary_flux_provider(
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
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::clear_ale_pressure_boundary() noexcept
{
    if (d_solver->uses_legacy_backend())
    {
        return;
    }
    try
    {
        d_solver->native_pressure_projection().clear_fixed_boundary_flux_provider();
        d_solver->boussinesq_coupled_pressure_velocity_solver().clear_fixed_boundary_flux_provider();
    }
    catch (...)
    {
    }
}

/** @brief Evaluate pure liquid density without reusing a void mixture. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::pure_liquid_density(local_ordinal_type cell_lid) const -> scalar_type
{
    if (d_material_feedback_model)
    {
        const auto density_mode = d_material_feedback_model->options().density_mode;
        if (density_mode == DensityFeedbackMode::BoussinesqVoid || density_mode == DensityFeedbackMode::Mixture)
        {
            return d_material_feedback_model->pure_liquid_density(d_solver->temperature().value(cell_lid));
        }
    }
    return d_solver->stored_material_properties().density.value(cell_lid);
}

/** @brief Return the configured headspace temperature for one closure. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::headspace_temperature(scalar_type time) const -> scalar_type
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
    if (options.temperature_mode == HeadspaceTemperatureMode::External)
    {
        std::optional<scalar_type> supplied;
        collective_detail::collective_local_validation(*d_solver->d_mesh, "External headspace temperature",
            [&]
            {
                if (!d_thermal_participant)
                    throw std::invalid_argument("External headspace temperature has no thermal owner.");
                supplied = d_thermal_participant->trial_cavity_temperature();
                if (!supplied || !std::isfinite(*supplied) || *supplied <= scalar_type{})
                    throw std::invalid_argument(
                        "Thermal participant must supply a finite positive cavity temperature.");
            });
        scalar_type low{}, high{};
        Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &*supplied, &low);
        Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &*supplied, &high);
        if (low != high)
            throw std::invalid_argument("External cavity temperature must agree on every mesh rank.");
        return *supplied;
    }

    scalar_type local_temperature_volume{};
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(d_solver->d_mesh->cell_volume(cell_lid));
        local_temperature_volume += d_solver->temperature().value(cell_lid) * volume;
        local_volume += volume;
    }
    scalar_type local_values[]{local_temperature_volume, local_volume};
    scalar_type global_values[2]{};
    Teuchos::reduceAll(
        *d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local_values, global_values);
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
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::scalar_void_volume() const -> scalar_type
{
    if (!d_scalar_void_fraction_model)
    {
        return {};
    }
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        local_volume += d_scalar_void_fraction_model->alpha_g().value(cell_lid) *
                        static_cast<scalar_type>(d_solver->d_mesh->cell_volume(cell_lid));
    }
    scalar_type global_volume{};
    Teuchos::reduceAll(
        *d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    return global_volume;
}

/** @brief Assemble callbacks and conservative gas compartments. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::make_free_surface_update(
    scalar_type time, bool initializing, const FreeSurfaceAccountingPreview* preview) -> FreeSurfaceUpdate
{
    if (!d_liquid_mass_inventory || !d_free_surface_model)
    {
        throw std::logic_error("Free-surface update requires both model and liquid inventory.");
    }

    FreeSurfaceUpdate update;
    const auto liquid_volume = preview ? preview->liquid.liquid_volume : d_liquid_mass_inventory->liquidVolume();
    if (species_surface_enabled())
        update.liquid_volume_at_pressure = [this](real_t pressure)
        {
            return static_cast<real_t>(d_reacting_gas_model->evaluate_liquid_volume(surface_pressure_offset(pressure)));
        };
    else
        update.liquid_volume_at_pressure = [liquid_volume](real_t) { return static_cast<real_t>(liquid_volume); };
    update.liquid_volume_deficit = preview ? static_cast<real_t>(preview->liquid_volume_deficit) : real_t{};
    update.bubble_volume_at_pressure = [this](real_t pressure)
    {
        if (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
            d_reacting_gas_model->supplies_void_fraction())
        {
            const auto pressure_mode = d_reacting_gas_model->options().pressure_mode;
            if (pressure_mode != ReactingPressureMode::Constant &&
                pressure_mode != ReactingPressureMode::Reconstructed)
            {
                throw std::logic_error("Planar free-surface bubble closure requires constant or "
                                       "reconstructed reacting pressure.");
            }
            auto pressure_offset = static_cast<scalar_type>(pressure);
            if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
                pressure_mode == ReactingPressureMode::Reconstructed)
            {
                pressure_offset += volume_weighted_mean_pressure();
            }
            return static_cast<real_t>(d_reacting_gas_model->evaluate_submerged_bubble_volume(pressure_offset));
        }
        if (d_boiling_source_model && d_boiling_source_model->enabled())
        {
            return static_cast<real_t>(d_boiling_source_model->global_submerged_steam_volume());
        }
        return real_t{};
    };

    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->supplies_void_fraction())
    {
        if (species_surface_enabled())
        {
            if (!initializing && (!preview || !preview->species || !preview->species_gas))
                throw std::logic_error("An accepted typed surface step requires its exact event receipt preview.");
            const auto typed = initializing ? prepare_species_surface_preview(time, scalar_type{}, true) : *preview;
            update.gas = *typed.species_gas;
            update.minimum_valid_absolute_pressure = d_reacting_gas_model->minimum_valid_absolute_pressure_offset();
            if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
                d_reacting_gas_model->options().pressure_mode == ReactingPressureMode::Reconstructed)
                update.minimum_valid_absolute_pressure -= volume_weighted_mean_pressure();
            update.headspace_temperature = headspace_temperature(time);
            update.time = time;
            update.time_step = initializing ? real_t{} : d_solver->time_step();
            return update;
        }
    }
    update.headspace_temperature = headspace_temperature(time);
    update.time = time;
    update.time_step = initializing ? real_t{} : static_cast<real_t>(d_solver->d_problem.time_options().time_step);
    return update;
}

/** Pure collective current gas-subset inventory. All species/population identities
 * come from the immutable schema; no reserved gas slot or phase renormalization.
 */
template<TpetraTypePack Pack>
FreeSurfaceGasInventory MultiphysicsSolverCoupler<Pack>::current_species_surface_inventory() const
{
    const auto& mesh = *d_solver->d_mesh;
    const auto& options = d_reacting_gas_model->options();
    const auto& catalog = options.reaction_mechanism->catalog();
    const auto ns = catalog.species().size(), stride = std::size_t{2} + options.population_ids.size();
    std::vector<detail::surface_species::Sum> sums;
    std::vector<scalar_type> local, global;
    collective_detail::collective_local_validation(mesh, "Species surface inventory reduction",
        [&]
        {
            if (ns > std::size_t(std::numeric_limits<int>::max()) / stride)
                throw std::invalid_argument("Species surface reduction exceeds MPI ordinal range.");
            sums.resize(ns * stride);
            for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
            {
                const auto state = d_reacting_gas_model->cell_species_state(static_cast<local_ordinal_type>(owned));
                for (std::size_t s = 0; s < ns; ++s)
                {
                    const auto id = catalog.species()[s].id;
                    sums[s * stride] += state.amount(catalog, id, Reacting::SpeciesPhase::Dissolved);
                    sums[s * stride + 1] += state.amount(catalog, id, Reacting::SpeciesPhase::Resolved);
                    for (std::size_t p = 0; p < options.population_ids.size(); ++p)
                        sums[s * stride + p + 2] +=
                            state.amount(catalog, id, Reacting::SpeciesPhase::Dispersed, options.population_ids[p]);
                }
            }
            local.resize(sums.size());
            global.resize(sums.size());
            for (std::size_t i = 0; i < sums.size(); ++i)
                local[i] = detail::surface_species::narrow(sums[i].value());
        });
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
        local.data(), global.data());
    FreeSurfaceGasInventory gas;
    collective_detail::collective_local_validation(mesh, "Species surface inventory publication",
        [&]
        {
            for (std::size_t s = 0; s < ns; ++s)
            {
                const auto& name = catalog.species()[s].name;
                gas.dissolved_moles[name] = detail::surface_species::narrow(global[s * stride]);
                detail::surface_species::Sum submerged;
                submerged += global[s * stride + 1];
                for (std::size_t p = 0; p < options.population_ids.size(); ++p)
                {
                    const auto amount = detail::surface_species::narrow(global[s * stride + p + 2]);
                    gas.submerged_population_moles[options.population_ids[p]][name] = amount;
                    submerged += amount;
                }
                gas.submerged_moles[name] = detail::surface_species::narrow(submerged.value());
            }
        });
    return gas;
}

template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::prepare_species_surface_preview(
    scalar_type interval_end, scalar_type time_step, bool initializing) -> FreeSurfaceAccountingPreview
{
    const auto& mesh = *d_solver->d_mesh;
    const auto& catalog = d_reacting_gas_model->options().reaction_mechanism->catalog();
    FreeSurfaceAccountingPreview result;
    result.species_gas.emplace(current_species_surface_inventory());
    auto& gas = *result.species_gas;
    collective_detail::collective_local_validation(mesh, "Species surface accepted event receipt",
        [&]
        {
            if (initializing)
                result.species = d_species_surface_accounting;
            else
                result.species = d_species_surface_accounting.preview(catalog,
                    d_reacting_gas_model->species_event_receipt(), interval_end, time_step,
                    d_reacting_gas_model->species_step_events(), d_reacting_gas_model->species_subset_source_changes(),
                    d_reacting_gas_model->physical_escape_events(), gas.escaped_moles_this_step,
                    result.species_liquid_source, result.species_liquid_outflow);
            gas.initial_moles = result.species->initial;
            gas.generated_moles = detail::surface_species::Accounting::report(result.species->production);
            gas.consumed_moles = detail::surface_species::Accounting::report(result.species->consumption);
            gas.net_source_moles = detail::surface_species::Accounting::report(result.species->net_source);
            gas.other_sink_moles = detail::surface_species::Accounting::report(result.species->dissolved_outflow);
        });
    return result;
}

template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::surface_pressure_offset(scalar_type headspace_pressure) const -> scalar_type
{
    if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
        d_reacting_gas_model->options().pressure_mode == ReactingPressureMode::Reconstructed)
        return headspace_pressure + volume_weighted_mean_pressure();
    return headspace_pressure;
}

/** Fixed-grid typed reporting/closure transaction. No liquid transport solve,
 * no source reevaluation, and no phase-energy debit inferred from escape traces.
 * The enclosing physical-step rollback also owns this model/bridge state.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::advance_species_free_surface(scalar_type time_step)
{
    auto accounting = prepare_species_surface_preview(d_solver->d_time + time_step, time_step);
    auto mass = d_liquid_mass_inventory->previewSpeciesMass(d_reacting_gas_model->liquid_mass_density(),
        d_reacting_gas_model->pure_liquid_density(), accounting.species_liquid_source,
        accounting.species_liquid_outflow);
    accounting.liquid = mass.diagnostics();
    const auto surface =
        d_free_surface_model->previewUpdate(make_free_surface_update(d_solver->d_time + time_step, false, &accounting));
    // Only a physically accepted closure may change the retained pressure.
    const auto offset = surface_pressure_offset(surface.headspacePressure());
    d_reacting_gas_model->set_absolute_pressure_offset(offset);
    // Mass is unchanged by pressure publication; report the accepted EOS density,
    // not the lagged pressure used for the source/transport substep.
    mass = d_liquid_mass_inventory->previewSpeciesMass(d_reacting_gas_model->liquid_mass_density(),
        d_reacting_gas_model->pure_liquid_density(), accounting.species_liquid_source,
        accounting.species_liquid_outflow);
    d_free_surface_model->commitUpdate(surface);
    d_liquid_mass_inventory->commitPhaseChange(mass);
    d_species_surface_accounting = std::move(*accounting.species);
    publish_free_surface_fields();
}

/** @brief Initialize a configured free-surface after fields are usable. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::initialize_free_surface_if_needed(
    bool allow_default_fields, bool dependencies_already_refreshed, bool preserve_on_failure)
{
    if (!d_free_surface_model || d_free_surface_model->initialized() ||
        (!d_solver->d_primary_fields_initialized && !allow_default_fields))
    {
        return;
    }
    if (!d_liquid_mass_inventory)
    {
        throw std::logic_error("Configured free-surface model lacks its liquid-mass inventory.");
    }
    try
    {
        if (!dependencies_already_refreshed && d_solver->d_physical_model_enabled)
        {
            update_context_type context{d_solver->d_time, d_solver->d_step_index, *d_solver->d_mesh,
                d_solver->temperature(), d_solver->pressure(), d_solver->velocity()};
            d_solver->stored_material_properties().update(context);
        }
        if (!dependencies_already_refreshed && d_material_feedback_model)
        {
            refresh_material_feedback(d_solver->d_time);
        }
        if (!dependencies_already_refreshed)
        {
            initialize_reacting_gas_state(d_solver->d_step_index == 0 &&
                !(d_reacting_gas_model && d_reacting_gas_model->species_enabled()));
            if (d_material_feedback_model && d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction())
            {
                refresh_material_feedback(d_solver->d_time);
            }
        }

        const auto initial_volume = configured_initial_liquid_volume(d_free_surface_options).value_or(real_t{});
        if (species_surface_enabled())
        {
            d_liquid_mass_inventory->initializeFromSpecies(
                d_reacting_gas_model->liquid_mass_density(), d_reacting_gas_model->pure_liquid_density());
            auto initial_gas = current_species_surface_inventory();
            auto amounts = initial_gas.dissolved_moles;
            for (const auto& [name, amount] : initial_gas.submerged_moles)
                amounts[name] += amount;
            for (const auto& [name, amount] : d_free_surface_options.headspace.initial_moles)
                amounts[name] += amount;
            d_species_surface_accounting.initialize(d_reacting_gas_model->options().reaction_mechanism->catalog(),
                d_reacting_gas_model->species_event_receipt(), std::move(amounts), d_solver->d_time);
        }
        else
            d_liquid_mass_inventory->initialize(static_cast<scalar_type>(initial_volume),
                [this](local_ordinal_type cell_lid) { return pure_liquid_density(cell_lid); });
        const auto configured_volume = configured_initial_liquid_volume(d_free_surface_options);
        if (!species_surface_enabled() && d_free_surface_options.liquid_mass.initial_liquid_mass && configured_volume)
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
        d_free_surface_model->initialize(make_free_surface_update(d_solver->d_time, true));
        if (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
            d_reacting_gas_model->supplies_void_fraction())
        {
            const auto pressure_mode = d_reacting_gas_model->options().pressure_mode;
            if (pressure_mode == ReactingPressureMode::Constant ||
                pressure_mode == ReactingPressureMode::Reconstructed)
            {
                auto offset = static_cast<scalar_type>(d_free_surface_model->headspacePressure());
                if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE &&
                    pressure_mode == ReactingPressureMode::Reconstructed)
                {
                    offset += volume_weighted_mean_pressure();
                }
                d_reacting_gas_model->set_absolute_pressure_offset(offset);
                if (species_surface_enabled())
                {
                    const auto eos_report = d_liquid_mass_inventory->previewSpeciesMass(
                        d_reacting_gas_model->liquid_mass_density(), d_reacting_gas_model->pure_liquid_density(), 0, 0);
                    d_liquid_mass_inventory->commitPhaseChange(eos_report);
                    if (configured_volume)
                    {
                        const auto actual = static_cast<real_t>(d_liquid_mass_inventory->liquidVolume());
                        const auto scale = std::max(std::abs(*configured_volume), std::abs(actual));
                        const auto tolerance = d_free_surface_options.coupling.volume_absolute_tolerance +
                                               d_free_surface_options.coupling.volume_relative_tolerance * scale;
                        if (std::abs(actual - *configured_volume) > tolerance)
                            throw std::invalid_argument("Configured fill disagrees with actual species liquid volume "
                                                        "at the accepted initial pressure.");
                    }
                }
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
        if (!preserve_on_failure)
            remove_free_surface_model();
        throw;
    }
}

/** @brief Apply accepted phase change and close the planar volume budget. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::advance_free_surface(scalar_type time_step)
{
    const auto phase_timing = d_solver->d_solver_timings.scope(SolverPhase::FreeSurface);
    if (!d_free_surface_model)
    {
        return;
    }
    initialize_free_surface_if_needed(true);
    if (species_surface_enabled())
    {
        advance_species_free_surface(time_step);
        return;
    }

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
            return d_liquid_mass_inventory->previewCellwiseAdvance(time_step, d_solver->projected_face_fluxes(),
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
    d_free_surface_model->update(make_free_surface_update(d_solver->d_time + time_step, false, &preview));
    d_liquid_mass_inventory->commitPhaseChange(phase_change_preview);
    if (phase_change_preview.transportStatistics())
    {
        d_solver->d_last_step_statistics.add(*phase_change_preview.transportStatistics());
    }
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->supplies_void_fraction())
    {
        const auto pressure_mode = d_reacting_gas_model->options().pressure_mode;
        if (pressure_mode == ReactingPressureMode::Constant || pressure_mode == ReactingPressureMode::Reconstructed)
        {
            d_reacting_gas_model->set_absolute_pressure_offset(d_free_surface_model->headspacePressure());
        }
    }
    publish_free_surface_fields();
}

/** @brief Publish global diagnostics and a cell-centre occupancy mask. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::publish_free_surface_fields()
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
    for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto coordinate = static_cast<scalar_type>(d_solver->d_mesh->cell_centroid(cell_lid).component(axis));
        const auto occupancy = coordinate <= diagnostics.pool_level ? scalar_type{1} : scalar_type{};
        d_pool_occupancy->set_owned_value(cell_lid, occupancy);
        local_occupancy_volume += occupancy * static_cast<scalar_type>(d_solver->d_mesh->cell_volume(cell_lid));
    }
    d_pool_occupancy->sync_ghosts();
    scalar_type global_occupancy_volume{};
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_occupancy_volume,
        &global_occupancy_volume);
    d_pool_occupancy_volume_error = global_occupancy_volume - static_cast<scalar_type>(diagnostics.pool_volume);
}

/** @brief Append one accepted, already reduced free-surface history record. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::record_free_surface_history()
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
    if (species_surface_enabled())
    {
        record.catalogued_species = true;
        record.species_event = d_reacting_gas_model->species_event_receipt();
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
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::ensure_scalar_void_fraction_model()
{
    if (!d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model =
            std::make_unique<scalar_void_fraction_model_type>(d_solver->d_mesh, ScalarVoidFractionOptions{});
    }
}

/**
 * @brief Select the authoritative gas void-fraction field.
 *
 * @tparam Pack Tpetra type pack.
 * @return Active gas fraction field, or nullptr when no model supplies one.
 */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::active_alpha_g_field() const noexcept -> const field_type*
{
    if (d_scalar_void_fraction_model)
    {
        return &d_scalar_void_fraction_model->alpha_g();
    }
    if (d_reacting_gas_model)
    {
        return &d_reacting_gas_model->alpha_g();
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
auto MultiphysicsSolverCoupler<Pack>::active_alpha_l_field() const noexcept -> const field_type*
{
    if (d_scalar_void_fraction_model)
    {
        return &d_scalar_void_fraction_model->alpha_l();
    }
    if (d_reacting_gas_model)
    {
        return &d_reacting_gas_model->alpha_l();
    }
    return nullptr;
}

/**
 * @brief Advance or mirror the scalar void-fraction model for one step.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::update_void_fraction_models(scalar_type time_step)
{
    if (!d_scalar_void_fraction_model)
    {
        return;
    }

    if (d_reacting_gas_model && d_reacting_gas_model->supplies_void_fraction())
    {
        d_scalar_void_fraction_model->mirror(d_reacting_gas_model->alpha_g(), time_step);
        return;
    }

    d_scalar_void_fraction_model->update_explicit(time_step,
        d_reacting_gas_model ? &d_reacting_gas_model->source_alpha_reaction() : nullptr,
        d_boiling_source_model ? &d_boiling_source_model->source_alpha_boil() : nullptr);

    // This is the sole accepted steam-inventory completion point.  It must
    // immediately follow the scalar explicit update so the same accepted
    // source and collapse are used for liquid-mass accounting.
    if (d_boiling_source_model)
    {
        d_boiling_source_model->complete_void_fraction_update(
            time_step, *d_scalar_void_fraction_model, d_free_surface_model != nullptr);
    }
}

/**
 * @brief Reject incompatible cross-model selections before a step starts.
 *
 * Field availability remains a phase-local validation because authoritative
 * fraction fields may be initialized or updated earlier in the same step.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::validate_step_coupling() const
{
    validate_collective_model_state();
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Post-temperature reference support",
        [&]
        {
            if (d_reacting_gas_model && d_reacting_gas_model->species_enabled() &&
                d_reacting_gas_model->options().source_schedule ==
                    ReactingSourceSchedule::PrescribedCarrierAfterTemperature &&
                ((d_boiling_source_model && d_boiling_source_model->enabled()) || d_free_surface_model ||
                    planar_ale_enabled() || d_thermal_participant))
                throw std::invalid_argument(
                    "Post-temperature gas references require fixed-grid carrier storage without boiling "
                    "or surface ownership.");
        });
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Coupled nonlinear model support",
        [&]
        {
            if (d_solver->d_problem.time_options().pressure_velocity_coupling ==
                    PressureVelocityCoupling::CoupledNonlinear &&
                (d_free_surface_model || d_boiling_source_model))
                throw std::invalid_argument("coupledNonlinear does not support free surface or boiling.");
        });
    if (d_thermal_participant)
        collective_detail::collective_local_validation(*d_solver->d_mesh, "Thermal participant step support",
            [&]
            {
                if (d_thermal_participant && !planar_ale_enabled())
                    throw std::invalid_argument("Thermal participant is supported only by planar ALE advances.");
            });
    if (d_free_surface_step_failed)
    {
        throw std::logic_error(
            "A prior step failed after upstream multiphysics state advanced; reconstruct the solver from its last "
            "accepted state before retrying.");
    }
    d_solver->validate_step_state();
    if (d_reacting_gas_model && d_reacting_gas_model->options().bouss_reference)
    {
        collective_detail::collective_local_validation(*d_solver->d_mesh, "Coordinated Bouss scalar storage",
            [&]
            {
                const auto& reference = *d_reacting_gas_model->options().bouss_reference;
                if (d_free_surface_model || d_active_ale || d_thermal_participant ||
                    (d_boiling_source_model && d_boiling_source_model->enabled()) ||
                    d_solver->d_problem.time_options().pressure_velocity_coupling ==
                        PressureVelocityCoupling::CoupledNonlinear ||
                    !d_bouss_temperature_density || !d_solver->physical_transport_enabled() ||
                    d_solver->stored_material_properties().has_updater() ||
                    reference.thermalDensity != d_solver->d_model_options.reference_density)
                    throw std::invalid_argument("Coordinated Bouss vapor supports fixed-grid sensible storage only; "
                                                "independent boiling, coupled nonlinear flow, free surface/ALE and "
                                                "phase-U owners are not combined.");
                const auto& cp = d_solver->stored_material_properties().specific_heat_capacity;
                for (std::size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
                    if (cp.value(static_cast<local_ordinal_type>(owned)) != reference.heatCapacity)
                        throw std::invalid_argument(
                            "Coordinated Bouss vapor requires its declared constant cp; "
                            "variable-cp storage schedules are not implemented by this adapter.");
            });
    }

    if (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
        d_reacting_gas_model->supplies_void_fraction() && d_boiling_source_model && d_boiling_source_model->enabled())
    {
        throw std::logic_error("Schema-owned reacting gas cannot advance with boiling "
                               "until vapor mass is coupled to bubble inventories.");
    }
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
        d_reacting_gas_model->supplies_void_fraction() && d_scalar_void_fraction_model &&
        (d_scalar_void_fraction_model->options().alpha_min != d_reacting_gas_model->options().alpha_min ||
            d_scalar_void_fraction_model->options().alpha_max != d_reacting_gas_model->options().alpha_max))
    {
        throw std::logic_error("Schema-owned reacting gas cannot advance with mismatched scalar mirror "
                               "void-fraction bounds.");
    }
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() &&
        d_reacting_gas_model->supplies_void_fraction() && d_scalar_void_fraction_model &&
        std::isfinite(d_scalar_void_fraction_model->options().alpha_collapse_time))
    {
        throw std::logic_error("Schema-owned reacting gas cannot advance with scalar void collapse "
                               "until bubble inventory removal is coupled conservatively.");
    }
    if (d_free_surface_model)
    {
        validate_free_surface_configuration(d_free_surface_options);
    }
}

/**
 * @brief Advance optional sources whose state is needed by temperature transport.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 * @return true when Schema-owned reacting gas must be advanced after temperature.
 */
template<TpetraTypePack Pack>
bool MultiphysicsSolverCoupler<Pack>::advance_pre_temperature_models(scalar_type time_step)
{
    const auto phase_timing = d_solver->d_solver_timings.scope(SolverPhase::PreTemperatureModels);
    prepare_bouss_wall_heat(time_step);
    if (d_reacting_gas_model && d_reacting_gas_model->options().bouss_reference)
        collective_detail::collective_local_validation(*d_solver->d_mesh, "Updated Bouss scalar cp",
            [&]
            {
                const auto cp = d_reacting_gas_model->options().bouss_reference->heatCapacity;
                for (std::size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
                    if (d_solver->stored_material_properties().specific_heat_capacity.value(
                            static_cast<local_ordinal_type>(owned)) != cp)
                        throw std::invalid_argument(
                            "Updated variable cp is outside the fixed-capacity Bouss vapor contract.");
            });

    const bool species_before_temperature =
        d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled() &&
        d_reacting_gas_model->options().source_schedule == ReactingSourceSchedule::BeforeTemperature;
    const auto species_after_temperature = d_reacting_gas_model && d_reacting_gas_model->enabled() &&
                                           d_reacting_gas_model->supplies_void_fraction() &&
                                           !species_before_temperature;
    if (species_before_temperature)
    {
        if (species_surface_enabled())
            collective_detail::collective_local_validation(*d_solver->d_mesh, "Species surface pre-advance receipt",
                [&]
                {
                    d_species_surface_accounting.require_before_advance(
                        d_reacting_gas_model->species_event_receipt(), d_solver->d_time);
                });
        const auto gas_timing = d_solver->d_solver_timings.scope(SolverPhase::Gas);
        d_reacting_gas_model->advance(d_solver->d_time + time_step, time_step, d_solver->temperature(),
            d_solver->pressure(), d_solver->velocity(), d_solver->projected_face_fluxes(),
            d_solver->stored_material_properties());
    }

    if (!species_after_temperature)
    {
        if (d_boiling_source_model)
        {
            if (!d_scalar_void_fraction_model)
            {
                throw std::runtime_error("Boiling requires the authoritative scalar void model.");
            }
            d_boiling_source_model->update(time_step, d_solver->temperature(), d_solver->stored_material_properties(),
                *d_scalar_void_fraction_model,
                d_reacting_gas_model && d_reacting_gas_model->enabled()
                    ? &d_reacting_gas_model->source_alpha_reaction()
                    : nullptr);
        }
        update_void_fraction_models(time_step);
    }

    return species_after_temperature;
}

/**
 * @brief Advance temperature-dependent models and refresh transport fields.
 *
 * Required authoritative fields are checked here, after upstream models have
 * had their normal opportunity to publish them.
 *
 * @tparam Pack Tpetra type pack.
 * @param time_step Physical time-step size.
 * @param species_after_temperature Whether Schema-owned reacting gas advances in this phase.
 */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::advance_post_temperature_models(
    scalar_type time_step, bool species_after_temperature)
{
    const auto phase_timing = d_solver->d_solver_timings.scope(SolverPhase::PostTemperatureModels);
    if (species_after_temperature)
    {
        {
            const auto gas_timing = d_solver->d_solver_timings.scope(SolverPhase::Gas);
            d_reacting_gas_model->advance(d_solver->d_time + time_step, time_step, d_solver->temperature(),
                d_solver->pressure(), d_solver->velocity(), d_solver->projected_face_fluxes(),
                d_solver->stored_material_properties());
        }
        update_void_fraction_models(time_step);
    }

    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled())
    {
        d_reacting_gas_model->refresh_species_thermodynamics(d_solver->d_time + time_step, d_solver->temperature(),
            d_solver->pressure(), d_solver->velocity(), d_solver->stored_material_properties());
        // Fixed reference molar volumes cannot change in this refresh. Mirroring
        // twice would erase the accepted whole-step scalar void source diagnostic.
        if (!d_reacting_gas_model->options().bouss_reference)
            update_void_fraction_models(time_step);
    }
    refresh_material_feedback(d_solver->d_time + time_step);
    advance_free_surface(time_step);
    if (auto* turbulence = d_solver->find_turbulence_model())
    {
        turbulence->refresh_effective_properties(
            d_solver->stored_material_properties(), d_solver->d_model_options.reference_density);
    }
}

/** Complete physical interval state. Geometry epochs and preview nonces stay monotonic. */
template<TpetraTypePack Pack> struct MultiphysicsSolverCoupler<Pack>::CouplingState
{
    const MultiphysicsSolverCoupler* owner = nullptr;
    std::string configuration;
    std::vector<const void*> model_owners;
    std::vector<std::pair<field_type*, FieldStateSnapshot<field_type>>> scalar_fields;
    std::vector<std::pair<velocity_field_type*, FieldStateSnapshot<velocity_field_type>>> vector_fields;
    std::vector<std::pair<face_flux_field_type*, FieldStateSnapshot<face_flux_field_type>>> face_fields;
    std::vector<std::pair<volumetric_source_type*, bool>> sources;
    SP<const typename thermal_participant_type::State> thermal;
    SP<const typename temperature_equation_type::boundary_cache_type> thermal_flux;
    typename ale_motion_type::StateSnapshot geometry;
    typename material_type::StateSnapshot material;
    typename material_feedback_model_type::StateSnapshot feedback;
    typename liquid_mass_inventory_type::StateSnapshot liquid;
    std::optional<typename free_surface_model_type::StateSnapshot> surface;
    detail::surface_species::Accounting species_surface;
    std::optional<typename volume_continuity_model_type::StateSnapshot> continuity;
    std::optional<typename reacting_gas_model_type::StateSnapshot> gas;
    std::optional<typename scalar_void_fraction_model_type::StateSnapshot> void_state;
    std::optional<typename prescribed_heat_source_type::StateSnapshot> prescribed_heat;
    scalar_type time = {}, time_step = {}, occupancy_error = {};
    int step_index = 0;
    bool primary_initialized = false, failed = false;
    step_statistics_type statistics;
    residual_type residuals;
    continuity_residual_type volume_residuals;
    NonlinearSolveResult nonlinear_result;
    PlanarALEStepDiagnostics ale_diagnostics;
    size_t history_size = 0;
};

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::set_thermal_step_participant(SP<thermal_participant_type> participant)
{
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE thermal participant installation",
        [&]
        {
            if (participant && d_reacting_gas_model && d_reacting_gas_model->options().bouss_reference)
                throw std::invalid_argument(
                    "A finite Bouss cp*T/latent owner cannot also own native phase-U thermal coupling.");
            if (!d_active_coupling_checkpoint.expired() || d_active_ale ||
                (d_ale_motion && d_ale_motion->has_active_trial()))
                throw std::logic_error("Cannot replace thermal participant during a live coupling transaction.");
            if (participant && (d_solver->uses_legacy_backend() || !d_solver->physical_transport_enabled()))
                throw std::invalid_argument("Thermal participant requires mapped physical temperature transport.");
        });
    int local = participant ? 1 : 0, low = 0, high = 0;
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &low);
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &high);
    if (low != high)
        throw std::invalid_argument("Thermal participant installation differs across ranks.");
    std::string signature;
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE thermal configuration",
        [&]
        {
            if (participant)
                signature = participant->configuration_signature();
            if (signature.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Thermal configuration signature is too large.");
        });
    auto comm = d_solver->d_mesh->owned_cell_map()->getComm();
    int root_size = static_cast<int>(signature.size());
    Teuchos::broadcast(*comm, 0, 1, &root_size);
    std::string root_signature = comm->getRank() == 0 ? signature : std::string(root_size, '\0');
    if (root_size)
        Teuchos::broadcast(*comm, 0, root_size, root_signature.data());
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE thermal rank agreement",
        [&]
        {
            if (signature != root_signature)
                throw std::invalid_argument("Thermal participant configuration must agree on every rank.");
        });
    // Removing a participant removes its prescribed thermal boundary snapshot.
    if (d_thermal_participant && !participant)
        d_solver->temperature_equation().set_boundary_heat_flux_snapshot(
            std::make_shared<const typename temperature_equation_type::boundary_cache_type>(
                typename temperature_equation_type::boundary_cache_type{{}, d_solver->d_mesh}));
    d_thermal_participant = std::move(participant);
}

template<TpetraTypePack Pack> std::string MultiphysicsSolverCoupler<Pack>::coupling_configuration_signature() const
{
    auto time_options = d_solver->d_problem.time_options();
    time_options.time_step = 1.0; // Subcycling may change dt; configuration may not change.
    std::vector<std::pair<std::string, bool>> sources;
    for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
        sources.emplace_back(name, true); // Enabled flags are checkpointed physical source state.
    return (d_thermal_participant ? d_thermal_participant->configuration_signature() : std::string{}) +
           (d_free_surface_model ? encode_headspace_metadata(d_free_surface_model->headspace().metadata())
                                 : std::string{}) +
           encode_free_surface_options(d_free_surface_options) +
           encode_planar_ale_runtime_options(time_options, d_solver->d_problem.linear_options(),
               d_solver->pressure_linear_solver_options(), d_solver->momentum_linear_solver_options(),
               d_solver->d_model_options, d_material_feedback_model ? &d_material_feedback_model->options() : nullptr,
               d_reacting_gas_model ? &d_reacting_gas_model->options() : nullptr,
               d_solver->d_problem.boundary_conditions(), sources, d_solver->has_mutable_mesh_handle(),
               d_solver->uses_legacy_backend(), d_solver->physical_transport_enabled(),
               d_solver->stored_turbulence_model().enabled(),
               d_boiling_source_model && d_boiling_source_model->enabled(),
               d_scalar_void_fraction_explicitly_configured, d_solver->stored_material_properties().has_updater(),
               d_solver->stored_temperature_sources().has_dynamic_updates());
}

template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::create_coupling_checkpoint() -> CouplingCheckpoint
{
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE coupling checkpoint",
        [&]
        {
            if (!planar_ale_enabled() || !d_active_coupling_checkpoint.expired() || d_active_ale ||
                d_free_surface_step_failed ||
                d_solver->d_problem.time_options().pressure_velocity_coupling != PressureVelocityCoupling::PISO)
                throw std::logic_error(
                    "Coupling checkpoints require idle, healthy planar ALE/PISO and no live checkpoint.");
        });
    validate_collective_model_state();
    validate_planar_ale_support(d_free_surface_options);
    initialize_free_surface_if_needed(true, false);
    initialize_planar_ale_if_needed();
    auto saved = std::make_shared<CouplingState>();
    saved->owner = this;
    saved->configuration = coupling_configuration_signature();
    saved->model_owners = {d_ale_motion.get(), d_free_surface_model.get(), d_liquid_mass_inventory.get(),
        d_volume_continuity_model.get(), d_material_feedback_model.get(), d_reacting_gas_model.get(),
        d_scalar_void_fraction_model.get(), d_prescribed_heat_source.get(), d_thermal_participant.get()};
    if (d_thermal_participant)
        saved->thermal = d_thermal_participant->capture_state();
    if (d_thermal_participant)
        collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE thermal checkpoint state",
            [&]
            {
                if (d_thermal_participant && !saved->thermal)
                    throw std::invalid_argument("Thermal participant returned a null checkpoint state.");
            });
    saved->thermal_flux = d_solver->temperature_equation().boundary_heat_flux();
    saved->geometry = d_ale_motion->snapshot();
    for (auto* field :
        {&d_solver->pressure(), &d_solver->pressure_correction(), &d_solver->temperature(), d_clear_level.get(),
            d_pool_level.get(), d_headspace_pressure.get(), d_pool_occupancy.get(), d_mesh_volume_rate.get(),
            d_continuity_residual.get(), d_ale_old_density.get(), d_ale_old_heat_capacity.get()})
        saved->scalar_fields.emplace_back(field, FieldStateSnapshot(*field));
    for (auto* field :
        {&d_solver->velocity(), &d_solver->predictor_pressure_gradient(), &d_solver->predictor_velocity()})
        saved->vector_fields.emplace_back(field, FieldStateSnapshot(*field));
    for (auto* field : {&d_solver->old_face_fluxes(), &d_solver->projected_face_fluxes(),
             d_mesh_relative_face_flux.get(), d_bubble_slip_volume_flux.get()})
        saved->face_fields.emplace_back(field, FieldStateSnapshot(*field));
    for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
    {
        saved->sources.emplace_back(source.get(), source->enabled());
        saved->scalar_fields.emplace_back(&source->field(), FieldStateSnapshot(source->field()));
    }
    saved->material = d_solver->stored_material_properties().snapshot();
    saved->feedback = d_material_feedback_model->snapshot();
    saved->liquid = d_liquid_mass_inventory->snapshot();
    saved->surface.emplace(d_free_surface_model->snapshot());
    saved->species_surface = d_species_surface_accounting;
    saved->continuity.emplace(d_volume_continuity_model->snapshot());
    if (d_reacting_gas_model)
        saved->gas.emplace(d_reacting_gas_model->snapshot());
    if (d_scalar_void_fraction_model)
        saved->void_state.emplace(d_scalar_void_fraction_model->snapshot());
    if (d_prescribed_heat_source)
        saved->prescribed_heat.emplace(d_prescribed_heat_source->snapshot());
    saved->time = d_solver->d_time;
    saved->time_step = d_solver->time_step();
    saved->step_index = d_solver->d_step_index;
    saved->primary_initialized = d_solver->d_primary_fields_initialized;
    saved->failed = d_free_surface_step_failed;
    saved->statistics = d_solver->d_last_step_statistics;
    saved->residuals = d_solver->pressure_velocity_residuals();
    saved->volume_residuals = d_solver->d_last_volume_continuity_residuals;
    saved->nonlinear_result = d_solver->d_last_nonlinear_result;
    saved->ale_diagnostics = d_planar_ale_diagnostics;
    saved->occupancy_error = d_pool_occupancy_volume_error;
    saved->history_size = d_free_surface_history.size();
    d_active_coupling_checkpoint = saved;
    return CouplingCheckpoint(std::move(saved));
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::validate_coupling_checkpoint(const CouplingCheckpoint& checkpoint) const
{
    const auto& saved = checkpoint.d_state;
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE interval checkpoint validation",
        [&]
        {
            const std::vector<const void*> owners{d_ale_motion.get(), d_free_surface_model.get(),
                d_liquid_mass_inventory.get(), d_volume_continuity_model.get(), d_material_feedback_model.get(),
                d_reacting_gas_model.get(), d_scalar_void_fraction_model.get(), d_prescribed_heat_source.get(),
                d_thermal_participant.get()};
            if (!saved || saved->owner != this || saved.get() != d_active_coupling_checkpoint.lock().get() ||
                saved->model_owners != owners || saved->configuration != coupling_configuration_signature() ||
                d_active_ale || !d_ale_motion || d_ale_motion->has_active_trial() ||
                d_free_surface_history.size() < saved->history_size)
                throw std::invalid_argument("Coupling checkpoint is stale, foreign, active, or reconfigured.");
            size_t index = 0;
            if (saved->sources.size() != d_solver->stored_temperature_sources().entries().size())
                throw std::invalid_argument("Coupling source registry changed after checkpoint.");
            for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
                if (saved->sources[index++].first != source.get())
                    throw std::invalid_argument("Coupling source ownership changed after checkpoint.");
        });
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::restore_coupling_checkpoint(const CouplingCheckpoint& checkpoint)
{
    validate_coupling_checkpoint(checkpoint);
    const auto& saved = *checkpoint.d_state;
    d_solver->clear_volume_continuity_target();
    clear_ale_pressure_boundary();
    d_ale_temperature_density = nullptr;
    d_active_ale.reset();
    d_ale_motion->restore(saved.geometry);
    refresh_geometry_dependent_state();
    d_ale_boundary->refresh(d_free_surface_model->volumeMap());
    for (const auto& [field, state] : saved.scalar_fields)
        state.restore(*field);
    for (const auto& [field, state] : saved.vector_fields)
        state.restore(*field);
    for (const auto& [field, state] : saved.face_fields)
        state.restore(*field);
    for (const auto& [source, enabled] : saved.sources)
        source->set_enabled(enabled);
    d_solver->stored_material_properties().restore(saved.material);
    d_material_feedback_model->restore(saved.feedback);
    d_liquid_mass_inventory->restore(saved.liquid);
    d_free_surface_model->restore(*saved.surface);
    d_species_surface_accounting = saved.species_surface;
    d_volume_continuity_model->restore(*saved.continuity);
    if (saved.gas)
        d_reacting_gas_model->restore(*saved.gas);
    if (saved.void_state)
        d_scalar_void_fraction_model->restore(*saved.void_state);
    if (saved.prescribed_heat)
        d_prescribed_heat_source->restore(*saved.prescribed_heat);
    if (saved.thermal)
        d_thermal_participant->restore_state(*saved.thermal);
    if (d_thermal_participant)
        d_solver->temperature_equation().set_boundary_heat_flux_snapshot(saved.thermal_flux);
    d_solver->d_time = saved.time;
    d_solver->set_time_step(saved.time_step);
    d_solver->d_step_index = saved.step_index;
    d_solver->d_primary_fields_initialized = saved.primary_initialized;
    d_free_surface_step_failed = saved.failed;
    d_solver->d_last_step_statistics = saved.statistics;
    d_solver->pressure_velocity_residuals() = saved.residuals;
    d_solver->d_last_volume_continuity_residuals = saved.volume_residuals;
    d_solver->d_last_nonlinear_result = saved.nonlinear_result;
    d_planar_ale_diagnostics = saved.ale_diagnostics;
    d_pool_occupancy_volume_error = saved.occupancy_error;
    // A valid checkpoint preserves the accepted prefix; trials only append.
    d_free_surface_history.resize(saved.history_size);
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::validate_coupling_checkpoint_acceptance(
    const CouplingCheckpoint& checkpoint) const
{
    validate_coupling_checkpoint(checkpoint);
    collective_detail::collective_local_validation(*d_solver->d_mesh, "ALE interval acceptance",
        [&]
        {
            if (d_free_surface_step_failed)
                throw std::logic_error("A failed coupling interval cannot be accepted.");
            if (d_prescribed_heat_source && d_prescribed_heat_source->has_interval_energy())
            {
                const auto end = d_prescribed_heat_source->interval_end_time();
                // A clock-scale tolerance can exceed a short energy interval and
                // accept its budget without executing any physical substep.
                const auto tolerance = scalar_type{32} * std::numeric_limits<scalar_type>::epsilon() *
                                       d_prescribed_heat_source->interval_duration();
                if (!std::isfinite(d_solver->d_time) || std::abs(d_solver->d_time - end) > tolerance)
                    throw std::logic_error("Accept the complete prescribed_heat-energy interval, not a partial subcycle.");
            }
        });
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::accept_coupling_checkpoint(CouplingCheckpoint& checkpoint)
{
    validate_coupling_checkpoint_acceptance(checkpoint);
    d_active_coupling_checkpoint.reset();
    checkpoint.d_state.reset();
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::set_coupling_interval_energy(
    std::span<const scalar_type> owned_cell_energy_joules, scalar_type interval_duration)
{
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Coupling interval source selection",
        [&]
        {
            const auto checkpoint = d_active_coupling_checkpoint.lock();
            if (checkpoint && d_solver->d_time != checkpoint->time)
                throw std::logic_error("Restore the coupling checkpoint before replacing an interval energy budget.");
            if (!planar_ale_enabled() || !d_prescribed_heat_source || d_active_ale ||
                (d_ale_motion && d_ale_motion->has_active_trial()))
                throw std::logic_error("Interval energy requires an idle planar ALE solver with a prescribed_heat source.");
        });
    d_prescribed_heat_source->set_interval_energy(owned_cell_energy_joules, d_solver->d_time, interval_duration);
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
 * @throws std::logic_error if incompatible Schema-owned reacting gas coupling is active.
 * @throws std::runtime_error if an enabled model lacks its required fraction field.
 */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::step()
{
    const auto phase_timing = d_solver->d_solver_timings.scope(SolverPhase::Step);
    d_solver->validate_pressure_velocity_selection();
    validate_step_coupling();
    if (d_prescribed_heat_source && d_prescribed_heat_source->has_interval_energy())
        d_prescribed_heat_source->refresh_interval_energy(d_solver->d_time, d_solver->time_step());
    if (d_free_surface_model && d_free_surface_options.mode == FreeSurfaceMode::PlanarALE)
    {
        // begin_step() intentionally clears per-step numerical statistics.
        // Preserve the last accepted report outside the ALE transaction so a
        // rejected later step restores it rather than publishing a cleared
        // or partially accumulated trial report.
        const auto accepted_statistics = d_solver->d_last_step_statistics;
        const auto accepted_volume_residuals = d_solver->d_last_volume_continuity_residuals;
        d_solver->begin_step();
        try
        {
            step_planar_ale();
        }
        catch (...)
        {
            d_solver->d_last_step_statistics = accepted_statistics;
            d_solver->d_last_volume_continuity_residuals = accepted_volume_residuals;
            throw;
        }
        return;
    }
    AcceptedStateRollback physics_rollback;
    auto* sas_model = d_solver->find_turbulence_model();
    const bool active_sas =
        sas_model && sas_model->type() == TurbulenceModelType::SSTKOmegaSAS && sas_model->options().sas.enabled;
    const bool transactional_physics = active_sas ||
        (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled());
    if (sas_model)
        sas_model->validate_time_mode(d_solver->d_problem.time_options().physical_time);
    if (transactional_physics)
    {
        if (d_material_feedback_model)
            physics_rollback.capture_model(*d_material_feedback_model);
        if (d_scalar_void_fraction_model)
            physics_rollback.capture_model(*d_scalar_void_fraction_model);
        if (d_reacting_gas_model)
            physics_rollback.add([this, saved = d_reacting_gas_model->snapshot()]
                { d_reacting_gas_model->restore_for_initialization_rollback(saved); });
        if (d_boiling_source_model)
            physics_rollback.capture_model(*d_boiling_source_model);
        if (d_free_surface_model)
            physics_rollback.capture_model(*d_free_surface_model);
        if (d_liquid_mass_inventory)
            physics_rollback.capture_model(*d_liquid_mass_inventory);
        physics_rollback.capture_model(d_species_surface_accounting);
        for (auto* field :
            {d_clear_level.get(), d_pool_level.get(), d_headspace_pressure.get(), d_pool_occupancy.get()})
            if (field)
                physics_rollback.capture(*field);
        physics_rollback.add(
            [this, history_size = d_free_surface_history.size(), occupancy_error = d_pool_occupancy_volume_error,
                failed = d_free_surface_step_failed]
            {
                d_free_surface_history.resize(history_size);
                d_pool_occupancy_volume_error = occupancy_error;
                d_free_surface_step_failed = failed;
            });
        physics_rollback.capture(d_solver->temperature());
        for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
            physics_rollback.capture(source->field());
        physics_rollback.capture(d_solver->pressure());
        physics_rollback.capture(d_solver->pressure_correction());
        physics_rollback.capture(d_solver->velocity());
        physics_rollback.capture(d_solver->predictor_pressure_gradient());
        physics_rollback.capture(d_solver->predictor_velocity());
        physics_rollback.capture(d_solver->old_face_fluxes());
        physics_rollback.capture(d_solver->projected_face_fluxes());
        physics_rollback.add(
            [this, residuals = d_solver->pressure_velocity_residuals(),
                volume = d_solver->d_last_volume_continuity_residuals]
            {
                d_solver->pressure_velocity_residuals() = residuals;
                d_solver->d_last_volume_continuity_residuals = volume;
            });
        physics_rollback.add([this, saved = d_solver->stored_material_properties().snapshot()]
            { d_solver->stored_material_properties().restore(saved); });
        if (sas_model)
            physics_rollback.add([sas_model, saved = sas_model->snapshot()] { sas_model->restore(saved); });
        physics_rollback.add(
            [this, saved = d_solver->d_last_nonlinear_result] { d_solver->d_last_nonlinear_result = saved; });
        physics_rollback.add(
            [this, time = d_solver->d_time, step = d_solver->d_step_index,
                statistics = d_solver->d_last_step_statistics, primary_initialized = d_solver->d_primary_fields_initialized]
            {
                d_solver->d_time = time;
                d_solver->d_step_index = step;
                d_solver->d_last_step_statistics = statistics;
                d_solver->d_primary_fields_initialized = primary_initialized;
            });
    }
    const bool nonlinear_flow =
        d_solver->d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::CoupledNonlinear;
    // Preserve accepted reports and primary ghosts until the private flow
    // solve succeeds. Model refresh occurs once outside all NOX callbacks.
    if (!nonlinear_flow)
        d_solver->begin_step();
    std::optional<typename material_type::StateSnapshot> nonlinear_material_snapshot;
    std::optional<typename material_feedback_model_type::StateSnapshot> nonlinear_feedback_snapshot;
    std::optional<typename reacting_gas_model_type::StateSnapshot> nonlinear_gas_snapshot;
    std::optional<typename scalar_void_fraction_model_type::StateSnapshot> nonlinear_void_snapshot;
    if (nonlinear_flow && !transactional_physics)
    {
        nonlinear_material_snapshot.emplace(d_solver->stored_material_properties().snapshot());
        if (d_material_feedback_model)
            nonlinear_feedback_snapshot.emplace(d_material_feedback_model->snapshot());
        if (d_reacting_gas_model)
            nonlinear_gas_snapshot.emplace(d_reacting_gas_model->snapshot());
        if (d_scalar_void_fraction_model)
            nonlinear_void_snapshot.emplace(d_scalar_void_fraction_model->snapshot());
    }
    bool flow_accepted = false;
    const bool free_surface_active = d_free_surface_model != nullptr;
    try
    {
        if (d_solver->d_step_index == 0)
        {
            d_solver->temperature().sync_ghosts();
        }
        if (d_solver->d_physical_model_enabled)
        {
            refresh_physical_models();
        }
        initialize_free_surface_if_needed(true, d_solver->d_physical_model_enabled, transactional_physics);
        if (auto* turbulence = d_solver->find_turbulence_model())
        {
            turbulence->refresh_effective_properties(
                d_solver->stored_material_properties(), d_solver->d_model_options.reference_density);
        }

        d_solver->solve_flow({d_active_ale ? &*d_active_ale : nullptr, d_mesh_relative_face_flux.get()});
        flow_accepted = true;
        const auto time_step = d_solver->d_problem.time_options().time_step;
        d_solver->advance_turbulence(time_step);
        const auto species_after_temperature = advance_pre_temperature_models(time_step);
        advance_temperature_transport(time_step);
        advance_post_temperature_models(time_step, species_after_temperature);

        if (d_solver->uses_legacy_backend())
        {
            d_solver->sync_temperature_to_legacy();
        }
        d_solver->finish_step();
        if (d_free_surface_model)
        {
            record_free_surface_history();
        }
    }
    catch (...)
    {
        restore_bouss_wall_heat();
        if (nonlinear_flow && !flow_accepted && !transactional_physics)
        {
            d_solver->stored_material_properties().restore(*nonlinear_material_snapshot);
            if (nonlinear_feedback_snapshot)
                d_material_feedback_model->restore(*nonlinear_feedback_snapshot);
            if (nonlinear_gas_snapshot)
                d_reacting_gas_model->restore(*nonlinear_gas_snapshot);
            if (nonlinear_void_snapshot)
                d_scalar_void_fraction_model->restore(*nonlinear_void_snapshot);
            if (auto* turbulence = d_solver->find_turbulence_model())
                turbulence->refresh_effective_properties(
                    d_solver->stored_material_properties(), d_solver->d_model_options.reference_density);
        }
        if (transactional_physics)
        {
            physics_rollback.restore();
            if (d_solver->uses_legacy_backend())
            {
                d_solver->sync_primary_fields_to_legacy();
                d_solver->sync_temperature_to_legacy();
            }
        }
        if (!transactional_physics && (free_surface_active || (nonlinear_flow && flow_accepted)))
        {
            d_free_surface_step_failed = true;
        }
        throw;
    }
}

/** @brief Refresh optional-model geometry after the bound equations refresh. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::refresh_geometry_dependent_models()
{
    const auto& geometry = d_solver->d_shared_transport_geometry->shared_geometry();
    if (d_reacting_gas_model)
    {
        d_reacting_gas_model->refresh_geometry(geometry);
    }
    if (d_scalar_void_fraction_model)
    {
        d_scalar_void_fraction_model->refresh_geometry(geometry);
    }
    if (d_liquid_mass_inventory)
    {
        d_liquid_mass_inventory->refresh_geometry();
    }
}

/** @brief Initialize gas and free-surface dependencies after primary fields publish. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::initialize_primary_state()
{
    const bool pending_free_surface = d_free_surface_model && !d_free_surface_model->initialized();
    if (!pending_free_surface)
    {
        if (d_free_surface_model && d_material_feedback_model)
        {
            refresh_material_feedback(d_solver->d_time);
        }
        initialize_reacting_gas_state(true);
    }
    initialize_free_surface_if_needed();
}

/** @brief Append configured source, gas, void, and surface fields. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::append_solution_fields(
    VTUWriter& writer, const SolutionOutputOptions& output_options) const
{
    if (output_options.include_sources)
    {
        for (const auto& [name, source] : d_solver->stored_temperature_sources().entries())
        {
            writer.add_scalar_cell_data(name, d_solver->collect_scalar_field(source->field()));
        }
        if (d_reacting_gas_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_reaction", d_solver->collect_scalar_field(d_reacting_gas_model->source_alpha_reaction()));
        }
        if (d_boiling_source_model)
        {
            for (const auto& [name, field] : d_boiling_source_model->output_fields())
            {
                writer.add_scalar_cell_data(name, d_solver->collect_scalar_field(*field));
            }
        }
        if (d_scalar_void_fraction_model)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total", d_solver->collect_scalar_field(d_scalar_void_fraction_model->source_alpha_total()));
        }
    }
    if (output_options.include_reacting_gas_fields && d_scalar_void_fraction_model)
    {
        writer.add_scalar_cell_data("alpha_g", d_solver->collect_scalar_field(d_scalar_void_fraction_model->alpha_g()));
        writer.add_scalar_cell_data("alpha_l", d_solver->collect_scalar_field(d_scalar_void_fraction_model->alpha_l()));
        if (!output_options.include_sources)
        {
            writer.add_scalar_cell_data(
                "S_alpha_total", d_solver->collect_scalar_field(d_scalar_void_fraction_model->source_alpha_total()));
        }
    }
    if (output_options.include_reacting_gas_fields && d_reacting_gas_model)
    {
        for (const auto& [name, field] : d_reacting_gas_model->output_fields())
        {
            if (name == "S_alpha_reaction")
                continue;
            if (d_scalar_void_fraction_model && (name == "alpha_g" || name == "alpha_l"))
            {
                continue;
            }
            writer.add_scalar_cell_data(name, d_solver->collect_scalar_field(*field));
        }
    }
    if (output_options.include_free_surface_fields && d_free_surface_model && d_free_surface_model->initialized())
    {
        writer.add_scalar_cell_data("rhoLiquid", d_solver->collect_scalar_field(rho_liquid()));
        if (d_liquid_mass_inventory->mode() == LiquidVolumeMode::CellMassInventory)
        {
            writer.add_scalar_cell_data(
                "liquidMassInventory", d_solver->collect_scalar_field(d_liquid_mass_inventory->cellMassInventory()));
        }
        writer.add_scalar_cell_data("clearLevel", d_solver->collect_scalar_field(clear_level()));
        writer.add_scalar_cell_data("poolLevel", d_solver->collect_scalar_field(pool_level()));
        writer.add_scalar_cell_data("headspacePressure", d_solver->collect_scalar_field(headspace_pressure()));
        writer.add_scalar_cell_data("poolOccupancy", d_solver->collect_scalar_field(pool_occupancy()));
        if (d_free_surface_options.mode == FreeSurfaceMode::PlanarALE && d_volume_continuity_model &&
            d_mesh_volume_rate && d_continuity_residual)
        {
            writer.add_scalar_cell_data("meshVolumeRate", d_solver->collect_scalar_field(*d_mesh_volume_rate));
            writer.add_scalar_cell_data(
                "volumeSourceRate", d_solver->collect_scalar_field(d_volume_continuity_model->material_source_field()));
            writer.add_scalar_cell_data("bubbleSlipVolumeRate",
                d_solver->collect_scalar_field(d_volume_continuity_model->slip_contribution_field()));
            writer.add_scalar_cell_data("continuityTarget",
                d_solver->collect_scalar_field(d_volume_continuity_model->continuity_target_field()));
            writer.add_scalar_cell_data("continuityResidual", d_solver->collect_scalar_field(*d_continuity_residual));
        }
    }
}

/** @brief Retain the solver and acquire one external-controller lease. */
template<TpetraTypePack Pack>
ALEConvergenceStatistics MultiphysicsSolverCoupler<Pack>::ale_convergence_statistics() const noexcept
{
    return d_ale_convergence_statistics;
}

template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::set_ale_convergence_observer(ALEConvergenceObserver observer)
{
    d_ale_convergence_observer = std::move(observer);
}

template<TpetraTypePack Pack> struct MultiphysicsSolverCoupler<Pack>::BoussWallTrial
{
    using Cache = typename temperature_equation_type::boundary_cache_type;
    struct Face
    {
        int batch = 0;
        std::size_t slot = 0;
        local_ordinal_type owner = 0;
        scalar_type conductivity = 0, owned_conductivity = 0;
    };
    SP<const Cache> original, flux;
    std::shared_ptr<const Reacting::ReactionMechanism> mechanism;
    const field_type* conductivity = nullptr;
    const Cache* boundary_conductivity = nullptr;
    std::vector<Face> faces;
    std::vector<Reacting::BoussWallHeatBudget> cells;
    std::uint64_t epoch = 0, signature = 0;
    scalar_type interval_end = 0, time_step = 0;
};

template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::prepare_bouss_wall_heat(scalar_type time_step)
{
    if (!d_reacting_gas_model || !d_reacting_gas_model->options().bouss_reference ||
        !d_reacting_gas_model->options().bouss_sources->vapor ||
        d_reacting_gas_model->options().bouss_sources->vapor->walls.empty())
        return;
    using Cache = typename temperature_equation_type::boundary_cache_type;
    const auto& options = d_reacting_gas_model->options();
    const auto& walls = options.bouss_sources->vapor->walls;
    const auto& mesh = *d_solver->d_mesh;
    std::shared_ptr<BoussWallTrial> trial;
    std::shared_ptr<Cache> flux;
    std::string wall_flux_name;
    std::vector<int> face_present, global_face_present;
    const auto* turbulence = d_solver->find_turbulence_model();
    collective_detail::collective_local_validation(mesh, "Finite vapor wall owner storage",
        [&]
        {
            if (d_bouss_wall_trial || d_active_ale || !std::isfinite(time_step) || time_step <= 0)
                throw std::logic_error("Wall heat preparation needs a new fixed-grid physical interval.");
            wall_flux_name = "certifiedBoussWallHeatFlux";
            trial = std::make_shared<BoussWallTrial>();
            trial->original = d_solver->temperature_equation().boundary_heat_flux();
            if (!trial->original || trial->original->mesh.get() != &mesh)
                throw std::invalid_argument("Wall heat owner is null or belongs to another mesh.");
            flux = std::make_shared<Cache>(*trial->original);
            trial->conductivity = turbulence ? &turbulence->effective_thermal_conductivity()
                                             : &d_solver->stored_material_properties().thermal_conductivity;
            trial->boundary_conductivity =
                turbulence ? turbulence->effective_thermal_conductivity_boundary_cache() : nullptr;
            trial->epoch = mesh_geometry_epoch(mesh);
            trial->signature =
                Reacting::bouss_configuration_signature(*options.bouss_reference, *options.bouss_sources);
            trial->mechanism = options.reaction_mechanism;
            trial->interval_end = d_solver->d_time + time_step;
            trial->time_step = time_step;
            trial->cells.resize(mesh.num_owned_cells());
            for (auto& cell : trial->cells)
            {
                cell.producer = flux.get();
                cell.geometryEpoch = trial->epoch;
                cell.configurationSignature = trial->signature;
                cell.intervalEnd = trial->interval_end;
                cell.duration = time_step;
            }
            face_present.assign(walls.size(), 0);
            global_face_present.resize(walls.size());
        });
    // Inline construction introduces no extra rank-local heap allocation before
    // FieldStored's communicating constructor. Canonical face import also gives
    // ghost-only/empty owners the exact producer values, not a recomputed flux.
    std::optional<face_flux_field_type> wall_flux;
    wall_flux.emplace(d_solver->d_mesh, scalar_type{}, std::move(wall_flux_name));
    collective_detail::collective_local_validation(mesh, "Finite vapor actual wall heat owner",
        [&]
        {
            const auto& boundaries = d_solver->d_problem.boundary_conditions();
            for (const auto& [batch_id, batch] : mesh.boundary_batches())
            {
                const auto name = mesh.boundary_batch_name(batch_id);
                for (std::size_t wall = 0; wall < walls.size(); ++wall)
                {
                    if (walls[wall].patch != name)
                        continue;
                    auto& values = flux->value[batch_id];
                    values.resize(batch.face_lids.size());
                    const auto prescribed = trial->original->value.find(batch_id);
                    const bool explicit_flux = prescribed != trial->original->value.end();
                    const auto condition = boundaries.temperature.find(name);
                    if (explicit_flux && prescribed->second.size() != batch.face_lids.size())
                        throw std::invalid_argument("Wall heat owner has a stale local boundary shape.");
                    for (std::size_t slot = 0; slot < batch.face_lids.size(); ++slot)
                    {
                        const auto face = batch.face_lids[slot];
                        const auto owner = mesh.owner_cell(face);
                        if (!mesh.is_owned_cell(owner))
                            continue;
                        if (!mesh.is_geometry_exterior_face(face) || !wall_flux->is_owned_face(face))
                            throw std::invalid_argument("Wall vapor needs a physical, canonical owned boundary face.");
                        if (!explicit_flux &&
                            (boundaries.convection.contains(name) || condition == boundaries.temperature.end() ||
                                condition->second.type != BoundaryConditionType::Neumann))
                            throw std::invalid_argument(
                                "Finite wall vapor requires actual prescribed flux or Neumann "
                                "heat; Dirichlet/Robin/convection heat is not known before this solve.");
                        const auto owned_k = trial->conductivity->value(owner);
                        if (!std::isfinite(owned_k) || owned_k < 0)
                            throw std::invalid_argument("Wall vapor requires valid owned thermal conductivity.");
                        const auto k = FVM::boundary_coefficient<Pack>(
                            trial->boundary_conductivity, batch_id, slot, owned_k);
                        const auto outward = explicit_flux ? prescribed->second[slot] : -k * condition->second.value;
                        if (!std::isfinite(k) || k < 0 || !std::isfinite(outward) || (outward != 0 && k == 0) ||
                            outward > 0)
                            throw std::invalid_argument(
                                "Wall vapor needs valid conductivity and nonnegative inward heat.");
                        const auto inward = -outward;
                        const auto scale = std::max(std::abs(inward), std::abs(walls[wall].inwardHeatFlux));
                        if (std::abs(inward - walls[wall].inwardHeatFlux) >
                            64 * std::numeric_limits<scalar_type>::epsilon() * scale)
                            throw std::invalid_argument(
                                "Wall vapor heat mismatches the actual temperature boundary owner.");
                        if (d_solver->projected_face_fluxes().value(face) != 0)
                            throw std::invalid_argument("Wall phase change requires a closed zero-carrier-flux wall.");
                        wall_flux->set_owned_value(face, outward);
                        trial->faces.push_back({batch_id, slot, owner, k, owned_k});
                        face_present[wall] = 1;
                        const auto energy = static_cast<long double>(inward) * mesh.face_area(face) * time_step;
                        if (!std::isfinite(energy) || (inward != 0 && energy == 0))
                            throw std::invalid_argument(
                                "Wall heat event overflows or underflows its extensive energy.");
                        trial->cells[static_cast<std::size_t>(owner)].packets.push_back(
                            {wall, static_cast<std::int64_t>(mesh.face_geometry_global_id(face)), energy});
                    }
                }
            }
        });
    wall_flux->sync_ghosts();
    collective_detail::collective_local_validation(mesh, "Finite vapor canonical wall heat snapshot",
        [&]
        {
            for (const auto& [batch_id, batch] : mesh.boundary_batches())
                if (std::ranges::find(walls, mesh.boundary_batch_name(batch_id),
                        &Reacting::BoussWallHeatBinding::patch) != walls.end())
                    for (std::size_t slot = 0; slot < batch.face_lids.size(); ++slot)
                        flux->value.at(batch_id)[slot] = wall_flux->local_value(batch.face_lids[slot]);
            trial->flux = flux;
        });
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, static_cast<int>(face_present.size()),
        face_present.data(), global_face_present.data());
    collective_detail::collective_local_validation(mesh, "Finite vapor wall patch ownership",
        [&]
        {
            for (const auto present : global_face_present)
                if (!present)
                    throw std::invalid_argument("A configured wall vapor patch has no physical boundary faces.");
        });
    d_solver->temperature_equation().set_boundary_heat_flux_snapshot(trial->flux);
    d_bouss_wall_trial = std::move(trial);
    d_reacting_gas_model->d_active_bouss_wall_heat = &d_bouss_wall_trial->cells;
}

template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::validate_bouss_wall_heat() const
{
    if (!d_bouss_wall_trial)
        return;
    const auto& trial = *d_bouss_wall_trial;
    const auto* turbulence = d_solver->find_turbulence_model();
    const auto* conductivity = turbulence ? &turbulence->effective_thermal_conductivity()
                                          : &d_solver->stored_material_properties().thermal_conductivity;
    const auto* boundary_k = turbulence ? turbulence->effective_thermal_conductivity_boundary_cache() : nullptr;
    collective_detail::collective_local_validation(*d_solver->d_mesh, "Finite wall immutable heat-owner receipt",
        [&]
        {
            if (trial.epoch != mesh_geometry_epoch(*d_solver->d_mesh) ||
                trial.flux.get() != d_solver->temperature_equation().boundary_heat_flux().get() ||
                conductivity != trial.conductivity || boundary_k != trial.boundary_conductivity ||
                trial.mechanism.get() != d_reacting_gas_model->options().reaction_mechanism.get() ||
                trial.interval_end != d_solver->d_time + trial.time_step ||
                trial.signature !=
                    Reacting::bouss_configuration_signature(*d_reacting_gas_model->options().bouss_reference,
                        *d_reacting_gas_model->options().bouss_sources))
                throw std::logic_error(
                    "Wall source heat owner, geometry, interval or configuration changed during the trial.");
            for (const auto& face : trial.faces)
            {
                const auto owned_k = conductivity->value(face.owner);
                if (owned_k != face.owned_conductivity ||
                    FVM::boundary_coefficient<Pack>(boundary_k, face.batch, face.slot, owned_k) != face.conductivity)
                    throw std::logic_error(
                        "Wall thermal conductivity changed after the admitted physical heat snapshot.");
            }
        });
}
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::restore_bouss_wall_heat()
{
    if (!d_bouss_wall_trial)
        return;
    if (d_reacting_gas_model)
        d_reacting_gas_model->d_active_bouss_wall_heat = nullptr;
    d_solver->temperature_equation().set_boundary_heat_flux_snapshot(d_bouss_wall_trial->original);
    d_bouss_wall_trial.reset();
}

template<TpetraTypePack Pack>
MultiphysicsSolverCoupler<Pack>::MultiphysicsSolverCoupler(solver_pointer solver) : d_solver(std::move(solver))
{
    if (!d_solver)
        throw std::invalid_argument("MultiphysicsSolverCoupler requires non-null shared solver ownership.");

    collective_detail::collective_local_validation(*d_solver->d_mesh, "External multiphysics controller",
        [&]
        {
            if (!d_solver->d_external_step_controller.expired() || d_solver->d_step_failed)
                throw std::logic_error("A timestep controller is already bound to this Boussinesq solver.");
            d_controller_lease = std::make_shared<const int>(0);
        });
    d_solver->d_external_step_controller = d_controller_lease;
}

/** @brief Initialize fields through the referenced solver, then dependent inventories. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::initialize_linear_temperature(
    const vec_type& direction, scalar_type hot_at_min, scalar_type cold_at_max, scalar_type initial_pressure)
{
    d_solver->initialize_linear_temperature(direction, hot_at_min, cold_at_max, initial_pressure);
    initialize_primary_state();
}
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::initialize_heated_box(scalar_type hot, scalar_type cold, scalar_type pressure)
{
    initialize_linear_temperature({1, 0, 0}, hot, cold, pressure);
}
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::initialize_bottom_hot_top_cold(
    scalar_type hot, scalar_type cold, scalar_type pressure)
{
    initialize_linear_temperature({0, 0, 1}, hot, cold, pressure);
}
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::set_material_updater(typename material_type::updater_type updater)
{
    if (d_liquid_mass_inventory && d_liquid_mass_inventory->initialized())
        throw std::logic_error(
            "Configure material updates before initializing the free-surface liquid-mass reference state.");
    d_solver->set_material_updater(std::move(updater));
}

/** @brief Refresh equation geometry, then the optional model caches owned here. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::refresh_geometry_dependent_state()
{
    d_solver->refresh_geometry_dependent_state();
    refresh_geometry_dependent_models();
}

/** @brief Pass numeric carrier, storage, and source inputs to the temperature equation stage. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::advance_temperature_transport(scalar_type time_step)
{
    const field_type* power_release = d_boiling_source_model
        ? &d_boiling_source_model->condensation_latent_heat_release() : nullptr;
    if (d_reacting_gas_model && d_reacting_gas_model->enabled() && d_reacting_gas_model->species_enabled())
    {
        if (!d_combined_heat_release)
            d_combined_heat_release = std::make_unique<field_type>(d_solver->d_mesh, "combinedReactionHeat");
        collective_detail::collective_local_validation(*d_solver->d_mesh, "Combined reaction thermal source", [&]
        {
            for (size_t c = 0; c < d_solver->d_mesh->num_owned_cells(); ++c)
            {
                const auto cell = static_cast<local_ordinal_type>(c);
                const auto value = d_reacting_gas_model->thermal_source().value(cell) +
                    (power_release ? power_release->value(cell) : scalar_type{});
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite combined reaction heat source.");
                d_combined_heat_release->set_owned_value(cell, value);
            }
        });
        d_combined_heat_release->sync_ghosts();
        power_release = d_combined_heat_release.get();
    }
    const typename solver_type::TemperatureTransportInputs inputs{d_active_ale ? &*d_active_ale : nullptr,
        d_active_ale && d_liquid_mass_inventory ? &d_liquid_mass_inventory->cellMassInventory() : nullptr,
        d_active_ale ? d_ale_old_heat_capacity.get() : nullptr,
        d_reacting_gas_model && d_reacting_gas_model->options().bouss_reference
            ? d_bouss_temperature_density.get()
            : (d_active_ale ? d_ale_temperature_density : nullptr),
        d_active_ale ? d_mesh_relative_face_flux.get() : nullptr,
        d_boiling_source_model ? &d_boiling_source_model->latent_heat_sink() : nullptr, power_release};
    validate_bouss_wall_heat();
    try
    {
        d_solver->advance_temperature_transport(time_step, inputs);
    }
    catch (...)
    {
        restore_bouss_wall_heat();
        throw;
    }
    restore_bouss_wall_heat();
}

/** @brief Drive only this controller's coupled step, preserving core progress formatting. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::step(ProgressStream& progress)
{
    step();
    if (d_solver->d_mesh->owned_cell_map()->getComm()->getRank() == 0)
        progress.write(d_solver->step_index(), d_solver->d_problem.time_options().steps, d_solver->time(),
            d_solver->last_step_statistics());
}
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::run(int steps)
{
    if (steps < 0)
        throw std::invalid_argument("MultiphysicsSolverCoupler::run steps cannot be negative.");
    for (int i = 0; i < steps; ++i)
        step();
}
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::run(int steps, ProgressStream& progress)
{
    if (steps < 0)
        throw std::invalid_argument("MultiphysicsSolverCoupler::run steps cannot be negative.");
    const auto final_step = d_solver->d_step_index + steps;
    for (int i = 0; i < steps; ++i)
    {
        step();
        if (d_solver->d_mesh->owned_cell_map()->getComm()->getRank() == 0)
            progress.write(d_solver->step_index(), final_step, d_solver->time(), d_solver->last_step_statistics());
    }
}
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::run()
{
    run(d_solver->d_problem.time_options().steps);
}
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::run(ProgressStream& progress)
{
    run(d_solver->d_problem.time_options().steps, progress);
}

/** @brief Compute the accepted transport Courant diagnostic without a reverse solver callback. */
template<TpetraTypePack Pack> auto MultiphysicsSolverCoupler<Pack>::maximum_courant_number() const -> scalar_type
{
    if (!planar_ale_enabled() || !d_mesh_relative_face_flux)
        return d_solver->maximum_courant_number();
    scalar_type local{};
    for (size_t owned = 0; owned < d_solver->d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        scalar_type sum{};
        for (const auto face : d_solver->d_mesh->faces(cell))
            sum += std::abs(d_mesh_relative_face_flux->local_value(face));
        local = std::max(local, scalar_type{0.5} * d_solver->time_step() * sum /
                                    static_cast<scalar_type>(d_solver->d_mesh->cell_volume(cell)));
    }
    scalar_type global{};
    Teuchos::reduceAll(*d_solver->d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
    return global;
}

/** @brief Compose model output around the referenced solver's primary/material fields. */
template<TpetraTypePack Pack>
auto MultiphysicsSolverCoupler<Pack>::solution_writer(const SolutionOutputOptions& options) const -> VTUWriter
{
    auto core_options = options;
    core_options.include_sources = false;
    auto writer = d_solver->solution_writer(
        core_options, d_material_feedback_model ? &d_material_feedback_model->output_fields() : nullptr);
    append_solution_fields(writer, options);
    return writer;
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
void MultiphysicsSolverCoupler<Pack>::write_solution_vtu(const std::string& filename) const
{
    write_solution_vtu(filename, {});
}

/** @brief Write selected solution fields to one VTU piece. */
template<TpetraTypePack Pack>
void MultiphysicsSolverCoupler<Pack>::write_solution_vtu(
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
void MultiphysicsSolverCoupler<Pack>::write_parallel_solution_vtu(
    const std::string& filename, const SolutionOutputOptions& output_options) const
{
    const auto communicator = d_solver->d_mesh->owned_cell_map()->getComm();
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
} // namespace SimpleFluid

#include "solvers/MultiphysicsSolverCouplerPlanarALE.tcc"
