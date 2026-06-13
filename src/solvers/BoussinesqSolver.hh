/**
 * @file BoussinesqSolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Minimal transient Boussinesq natural-convection driver.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/FissionPowerSource.hh"
#include "equations/RadiolyticGasModel.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "solvers/FluidSolver.hh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Minimal transient Boussinesq natural-convection solver.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqSolver : public FluidSolver<Pack>
{
public:
    using base_type = FluidSolver<Pack>;
    using typename base_type::mesh_type;
    using typename base_type::field_type;
    using typename base_type::velocity_field_type;
    using typename base_type::face_flux_field_type;
    using typename base_type::scalar_type;
    using typename base_type::local_ordinal_type;
    using typename base_type::global_ordinal_type;
    using typename base_type::vec_type;
    using typename base_type::residual_type;
    using typename base_type::step_statistics_type;
    using typename base_type::coupled_system_type;
    using cell_type = typename mesh_type::CellType;
    using base_type::pressure;
    using base_type::velocity;

    BoussinesqSolver(SP<const mesh_type> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options = {},
                     LinearSolverOptions linear_options = {});

    BoussinesqSolver(SP<const mesh_type> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options,
                     LinearSolverOptions linear_options,
                     BoussinesqModelOptions model_options);

    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options = {},
                     LinearSolverOptions linear_options = {});

    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options,
                     LinearSolverOptions linear_options,
                     BoussinesqModelOptions model_options);

    void initialize_linear_temperature(const vec_type& direction,
                                       scalar_type hot_at_min,
                                       scalar_type cold_at_max,
                                       scalar_type initial_pressure = 0.0);

    void initialize_heated_box(scalar_type hot_temperature,
                               scalar_type cold_temperature,
                               scalar_type initial_pressure = 0.0);

    void initialize_bottom_hot_top_cold(scalar_type hot_temperature,
                                        scalar_type cold_temperature,
                                        scalar_type initial_pressure = 0.0);

    void step() override;

    const field_type& temperature() const noexcept;
    field_type& temperature() noexcept;

    MaterialPropertyFields<Pack>& material_properties() noexcept;
    const MaterialPropertyFields<Pack>& material_properties() const noexcept;

    VolumetricScalarSource<Pack>& add_temperature_source(
        std::string name,
        scalar_type initial_power_density = {});
    bool remove_temperature_source(const std::string& name);
    VolumetricScalarSource<Pack>* find_temperature_source(
        const std::string& name) noexcept;
    const VolumetricScalarSource<Pack>* find_temperature_source(
        const std::string& name) const noexcept;
    TemperatureSourceRegistry<Pack>& temperature_sources() noexcept;
    const TemperatureSourceRegistry<Pack>& temperature_sources() const noexcept;

    FissionPowerSource<Pack>& add_fission_power_source();
    void configure_fission_power_source(
        const FissionPowerSourceOptions& options);
    bool remove_fission_power_source() noexcept;
    FissionPowerSource<Pack>* find_fission_power_source() noexcept;
    const FissionPowerSource<Pack>* find_fission_power_source() const noexcept;

    RadiolyticGasModel<Pack>& configure_radiolytic_gas(
        const RadiolyticGasOptions& options);
    RadiolyticGasModel<Pack>& configure_radiolytic_gas(
        const Database& database);
    bool remove_radiolytic_gas_model() noexcept;
    RadiolyticGasModel<Pack>* find_radiolytic_gas_model() noexcept;
    const RadiolyticGasModel<Pack>*
    find_radiolytic_gas_model() const noexcept;

    void set_material_updater(
        typename MaterialPropertyFields<Pack>::updater_type updater);
    void clear_material_updater() noexcept;

    void write_solution_vtu(const std::string& filename) const;
    void write_solution_vtu(
        const std::string& filename,
        const SolutionOutputOptions& output_options) const;

private:
    using base_type::begin_step;
    using base_type::collect_scalar_field;
    using base_type::coupled_pressure_velocity_solver;
    using base_type::d_last_step_statistics;
    using base_type::d_mesh;
    using base_type::d_problem;
    using base_type::d_step_index;
    using base_type::d_time;
    using base_type::finish_step;
    using base_type::fluid_solution_writer;
    using base_type::old_face_fluxes;
    using base_type::pressure_velocity_residuals;
    using base_type::projected_face_fluxes;
    using base_type::require_mesh;
    using base_type::solve_pressure_velocity_coupling;
    using base_type::velocity_boundary_cache;

    struct PhysicalModelTag {};

    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options,
                     LinearSolverOptions linear_options,
                     BoussinesqModelOptions model_options,
                     bool physical_model_enabled,
                     PhysicalModelTag);

    TemperatureDiffusionEquation<Pack>& temperature_equation();
    BoussinesqMomentumEquation<Pack>& momentum_equation() override;
    LinearSolveSummary advance_momentum() override;
    coupled_system_type assemble_coupled_system() override;
    MaterialPropertyFields<Pack>& stored_material_properties();
    const MaterialPropertyFields<Pack>& stored_material_properties() const;
    TemperatureSourceRegistry<Pack>& stored_temperature_sources();
    const TemperatureSourceRegistry<Pack>& stored_temperature_sources() const;
    void refresh_physical_models();

    BoussinesqModelOptions d_model_options;
    bool d_physical_model_enabled = false;
    std::unique_ptr<FissionPowerSource<Pack>> d_fission_power_source;
    std::unique_ptr<RadiolyticGasModel<Pack>> d_radiolytic_gas_model;
};

} // namespace SimpleFluid
