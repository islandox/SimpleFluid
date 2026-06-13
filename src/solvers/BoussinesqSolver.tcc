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
#include "geometry/MeshFactory.hh"

namespace SimpleFluid
{

/**
 * @brief Validate and store a non-null mesh pointer.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return The validated mesh shared pointer.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "BoussinesqSolver");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_legacy_mesh(
    const SP<const MeshHandle<Pack>>& mesh) -> SP<const mesh_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires a non-null mesh handle.");
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
            "BoussinesqSolver currently supports STK and Cartesian "
            "mesh handles.");
    }
    if (mesh->num_owned_cells() != (*cartesian)->num_cells())
    {
        throw std::invalid_argument(
            "BoussinesqSolver Cartesian compatibility currently "
            "requires a serial mesh.");
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
    : d_mesh(require_legacy_mesh(mesh)),
      d_problem(std::make_shared<MeshHandle<Pack>>(d_mesh),
                std::move(boundary_conditions),
                time_options,
                linear_options),
      d_model_options(std::move(model_options)),
      d_physical_model_enabled(physical_model_enabled)
{
    if (d_problem.time_options().time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }
    detail::validate_model_options(
        d_model_options, d_problem.time_options());

    d_problem.template emplace_object<FVM::VelocityBoundaryCache<Pack>>(
        "velocity_boundary_cache",
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions()));
    d_problem.template emplace_object<TemperatureDiffusionEquation<Pack>>(
        "temperature_equation",
        d_mesh,
        d_problem.boundary_conditions());
    d_problem.template emplace_object<BoussinesqMomentumEquation<Pack>>(
        "momentum_equation", d_mesh);
    d_problem.template emplace_object<PressureProjectionEquation<Pack>>(
        "pressure_projection", d_mesh, d_problem.linear_options());
    d_problem.template emplace_object<CoupledPressureVelocitySolver<Pack>>(
        "coupled_pressure_velocity_solver", d_mesh);
    d_problem.template emplace_object<field_type>(
        "temperature", d_mesh, "temperature");
    d_problem.template emplace_object<field_type>(
        "pressure", d_mesh, "pressure");
    d_problem.template emplace_object<field_type>(
        "pressure_correction", d_mesh, "pressure_correction");
    d_problem.template emplace_object<velocity_field_type>(
        "velocity", d_mesh, "velocity");
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
    d_problem.template emplace_object<MaterialPropertyFields<Pack>>(
        "material_properties",
        d_mesh,
        d_model_options,
        d_problem.time_options());
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
auto BoussinesqSolver<Pack>::pressure() const noexcept
    -> const field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity() const noexcept
    -> const velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() noexcept -> field_type&
{
    return d_problem.template object<field_type>("temperature");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure() noexcept -> field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity() noexcept
    -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
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
    d_physical_model_enabled = true;
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
    stored_temperature_sources().update(context);
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::last_pressure_velocity_residuals()
    const noexcept -> const residual_type&
{
    return pressure_velocity_residuals();
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
auto BoussinesqSolver<Pack>::pressure_projection()
    -> PressureProjectionEquation<Pack>&
{
    return d_problem.template object<
        PressureProjectionEquation<Pack>>("pressure_projection");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::coupled_pressure_velocity_solver()
    -> CoupledPressureVelocitySolver<Pack>&
{
    return d_problem.template object<
        CoupledPressureVelocitySolver<Pack>>(
            "coupled_pressure_velocity_solver");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity_boundary_cache()
    -> FVM::VelocityBoundaryCache<Pack>&
{
    return d_problem.template object<
        FVM::VelocityBoundaryCache<Pack>>("velocity_boundary_cache");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_correction() -> field_type&
{
    return d_problem.template object<field_type>("pressure_correction");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::predictor_velocity()
    -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>(
        "pressure_velocity_predictor");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::old_face_fluxes()
    -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>("old_face_flux");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::projected_face_fluxes()
    -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>(
        "projected_face_flux");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_velocity_residuals()
    -> residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_velocity_residuals()
    const -> const residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

/**
 * @brief Compute the volume-weighted L2 norm of a velocity-field update.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity_update_norm(
    const velocity_field_type& before,
    const velocity_field_type& after) const -> scalar_type
{
    EquationValidation::require_mesh_match(*d_mesh, before, "BoussinesqSolver");
    EquationValidation::require_mesh_match(*d_mesh, after, "BoussinesqSolver");

    scalar_type norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto delta = after.value(cell_lid) - before.value(cell_lid);
        norm_squared += delta.dot(delta) * d_mesh->cell_volume(cell_lid);
    }

    using std::sqrt;
    return sqrt(norm_squared);
}

/**
 * @brief Solve the semi-implicit momentum predictor and report its update norm.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::run_momentum_predictor()
    -> LinearSolveSummary
{
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        predictor_velocity().set_value(cell_lid, velocity().value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), old_face_fluxes());
    LinearSolveSummary linear_summary;
    if (d_physical_model_enabled)
    {
        auto zero_source =
            [](local_ordinal_type)
                -> typename velocity_field_type::vec_type
        {
            return {};
        };
        linear_summary =
            momentum_equation().advance_velocity_physical(
                velocity(),
                old_face_fluxes(),
                temperature(),
                velocity_boundary_cache(),
                d_problem.time_options(),
                stored_material_properties(),
                d_model_options.reference_density,
                d_model_options.density_feedback_enabled,
                velocity(),
                zero_source,
                d_problem.linear_options());
    }
    else
    {
        linear_summary =
            momentum_equation().advance_velocity(
                velocity(),
                old_face_fluxes(),
                temperature(),
                velocity_boundary_cache(),
                d_problem.time_options(),
                velocity(),
                d_problem.linear_options());
    }
    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
    return linear_summary;
}

/**
 * @brief Run one pressure-correction solve and accumulate pressure.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::run_pressure_correction()
    -> typename PressureProjectionEquation<Pack>::ProjectionResult
{
    const auto result =
        pressure_projection().project(pressure_correction(),
                                      d_problem.time_options().time_step,
                                      velocity_boundary_cache(),
                                      velocity());
    pressure().owned_data().update(1.0, pressure_correction().owned_data(), 1.0);
    d_mesh->sync_periodic_boundaries(pressure());
    return result;
}

/**
 * @brief Assemble and solve the monolithic velocity-pressure system.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::solve_coupled_krylov()
{
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        predictor_velocity().set_value(
            cell_lid, velocity().value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), old_face_fluxes());
    const auto system =
        coupled_pressure_velocity_solver().assemble(
            momentum_equation(),
            velocity(),
            pressure(),
            temperature(),
            old_face_fluxes(),
            velocity_boundary_cache(),
            d_problem.boundary_conditions(),
            d_problem.time_options(),
            d_physical_model_enabled
                ? &stored_material_properties()
                : nullptr,
            d_model_options.reference_density,
            d_model_options.density_feedback_enabled);
    const auto result =
        coupled_pressure_velocity_solver().solve(
            system,
            velocity(),
            pressure(),
            d_problem.linear_options());
    if (!result.converged)
    {
        throw std::runtime_error(
            "BoussinesqSolver coupled Krylov solve did not converge.");
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
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), projected_face_fluxes());
    scalar_type continuity_norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto balance =
            FVM::cell_flux_balance<Pack>(
                *d_mesh, projected_face_fluxes(), cell_lid);
        continuity_norm_squared += balance * balance;
    }
    using std::sqrt;
    pressure_velocity_residuals().continuity =
        sqrt(continuity_norm_squared);
}

/**
 * @brief Dispatch SIMPLE, PISO, or PIMPLE pressure-velocity coupling loops.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::solve_pressure_velocity_coupling()
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
            "BoussinesqSolver requires at least one pressure corrector.");
    }
    if (d_problem.time_options().n_outer_correctors < 1)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires at least one outer corrector.");
    }

    const auto pressure_corrections =
        d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::SIMPLE
      ? 1
      : d_problem.time_options().n_pressure_correctors;
    const auto outer_corrections =
        d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::PIMPLE
      ? d_problem.time_options().n_outer_correctors
      : 1;

    for (int outer = 0; outer < outer_corrections; ++outer)
    {
        ++d_last_step_statistics.nonlinear_iterations;
        d_last_step_statistics.add(run_momentum_predictor());

        typename PressureProjectionEquation<Pack>::ProjectionResult result;
        for (int corrector = 0; corrector < pressure_corrections; ++corrector)
        {
            result = run_pressure_correction();
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

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), projected_face_fluxes());
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
 * @param initial_pressure Uniform initial pressure value.
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
    if (d_mesh->num_owned_cells() == 0)
    {
        return;
    }

    auto min_projected = d_mesh->cell_centroid(0).dot(direction);
    auto max_projected = min_projected;
    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        min_projected = std::min(min_projected, projected);
        max_projected = std::max(max_projected, projected);
    }

    const auto width = max_projected > min_projected
                     ? max_projected - min_projected
                     : 1.0;
    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        const auto blend = (projected - min_projected) / width;
        temperature().set_value(cell_lid,
                                hot_at_min * (1.0 - blend)
                              + cold_at_max * blend);
        pressure().set_value(cell_lid, initial_pressure);
        velocity().set_value(cell_lid, {});
    }

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(pressure());
    d_mesh->sync_periodic_boundaries(velocity());
}

/**
 * @brief Initialise fields for a heated-box problem with temperature gradient along X.
 *
 * Delegates to initialize_linear_temperature with direction (1, 0, 0).
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the hot (xmin) boundary.
 * @param cold_temperature Temperature at the cold (xmax) boundary.
 * @param initial_pressure Uniform initial pressure value.
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
 * @param initial_pressure Uniform initial pressure value.
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
    d_last_step_statistics = {};
    if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(temperature());
        d_mesh->sync_periodic_boundaries(velocity());
    }
    if (d_physical_model_enabled)
    {
        refresh_physical_models();
    }

    solve_pressure_velocity_coupling();
    LinearSolveStatistics temperature_statistics;
    if (d_physical_model_enabled)
    {
        auto total_power_density =
            [&](local_ordinal_type cell_lid)
        {
            return stored_temperature_sources()
                .total_power_density(cell_lid);
        };
        temperature_statistics =
            temperature_equation().advance_physical(
                temperature(),
                projected_face_fluxes(),
                d_problem.time_options().time_step,
                stored_material_properties(),
                temperature(),
                total_power_density,
                d_problem.time_options().non_orthogonal_treatment,
                d_problem.linear_options());
    }
    else
    {
        temperature_statistics =
            temperature_equation().advance_semi_implicit(
                temperature(),
                projected_face_fluxes(),
                d_problem.time_options().time_step,
                d_problem.time_options().thermal_diffusivity,
                temperature(),
                d_problem.linear_options());
    }
    d_last_step_statistics.add(temperature_statistics);
    d_last_step_statistics.momentum =
        pressure_velocity_residuals().momentum;
    d_last_step_statistics.pressure =
        pressure_velocity_residuals().pressure;
    d_last_step_statistics.temperature =
        temperature_statistics.achieved_tolerance;
    d_last_step_statistics.continuity =
        pressure_velocity_residuals().continuity;

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(velocity());

    if (d_radiolytic_gas_model
        && d_radiolytic_gas_model->enabled())
    {
        d_radiolytic_gas_model->advance(
            d_time + d_problem.time_options().time_step,
            d_problem.time_options().time_step,
            temperature(),
            pressure(),
            velocity(),
            projected_face_fluxes(),
            stored_material_properties(),
            d_fission_power_source
                ? &d_fission_power_source->field()
                : nullptr);
    }

    d_time += d_problem.time_options().time_step;
    ++d_step_index;
}

/**
 * @brief Run the solver for a specified number of time steps.
 *
 * @tparam Pack Tpetra type pack.
 * @param steps Number of time steps to execute (defaults to the value configured
 *              in TimeStepperOptions).
 * @throws std::invalid_argument If steps is negative.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::run(int steps)
{
    if (steps < 0)
    {
        throw std::invalid_argument("BoussinesqSolver::run steps cannot be negative.");
    }

    for (int step_id = 0; step_id < steps; ++step_id)
    {
        step();
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
    std::unordered_map<global_ordinal_type, global_index_t> node_lid;
    VTUWriter::VectorData node_coords;
    VTUWriter::Int64Data cell_node_offsets;
    VTUWriter::Int64Data cell_node_ids;
    VTUWriter::UInt8Data cell_types;

    auto append_node = [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto iter = node_lid.find(node_gid);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = static_cast<global_index_t>(node_coords.size());
        node_lid.emplace(node_gid, lid);
        node_coords.push_back(d_mesh->node_coord(node_gid));
        return lid;
    };

    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto& cell_info = d_mesh->cell(static_cast<local_ordinal_type>(lid));
        for (const auto node_gid : cell_info.node_gids)
        {
            cell_node_ids.push_back(append_node(node_gid));
        }
        cell_node_offsets.push_back(static_cast<global_index_t>(cell_node_ids.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(cell_info.type)));
    }

    VTUWriter::ScalarData temperature_values;
    VTUWriter::ScalarData pressure_values;
    VTUWriter::VectorData velocity_values;
    temperature_values.reserve(d_mesh->num_local_cells());
    pressure_values.reserve(d_mesh->num_local_cells());
    velocity_values.reserve(d_mesh->num_local_cells());
    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(lid);
        temperature_values.push_back(static_cast<real_t>(
            temperature().local_value(cell_lid)));
        pressure_values.push_back(static_cast<real_t>(
            pressure().local_value(cell_lid)));
        velocity_values.push_back(velocity().local_value(cell_lid));
    }

    VTUWriter writer;
    writer.set_points(std::move(node_coords));
    writer.set_cells(std::move(cell_node_ids),
                     std::move(cell_node_offsets),
                     std::move(cell_types));
    writer.add_scalar_cell_data("temperature", std::move(temperature_values));
    writer.add_scalar_cell_data("pressure", std::move(pressure_values));
    writer.add_vector_cell_data("velocity", std::move(velocity_values));

    auto collect_scalar_field =
        [&](const field_type& field)
    {
        VTUWriter::ScalarData values;
        values.reserve(d_mesh->num_local_cells());
        for (size_t lid = 0;
             lid < d_mesh->num_local_cells();
             ++lid)
        {
            values.push_back(static_cast<real_t>(
                field.local_value(
                    static_cast<local_ordinal_type>(lid))));
        }
        return values;
    };

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
    }
    if (output_options.include_radiolytic_gas_fields
        && d_radiolytic_gas_model)
    {
        for (const auto& [name, field] :
             d_radiolytic_gas_model->output_fields())
        {
            if (name == "S_alpha_rad")
                continue;
            writer.add_scalar_cell_data(
                name, collect_scalar_field(*field));
        }
    }
    writer.write(filename);
}

} // namespace SimpleFluid
