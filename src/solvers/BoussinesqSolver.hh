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

#include "equations/BoilingSourceModel.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FissionPowerSource.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/RadiolyticGasModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
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

    using base_type::step;
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

    /**
     * @brief Create the reserved qdot_fission source if needed.
     */
    FissionPowerSource<Pack>& add_fission_power_source();
    /**
     * @brief Configure the reserved fission power-density source.
     */
    void configure_fission_power_source(
        const FissionPowerSourceOptions& options);
    /**
     * @brief Remove the reserved fission power source, if present.
     */
    bool remove_fission_power_source() noexcept;
    /**
     * @brief Return the mutable fission source, or nullptr when absent.
     */
    FissionPowerSource<Pack>* find_fission_power_source() noexcept;
    /**
     * @brief Return the fission source, or nullptr when absent.
     */
    const FissionPowerSource<Pack>* find_fission_power_source() const noexcept;

    /**
     * @brief Configure the optional radiolytic gas model from explicit options.
     */
    RadiolyticGasModel<Pack>& configure_radiolytic_gas(
        const RadiolyticGasOptions& options);
    /**
     * @brief Configure the optional radiolytic gas model from database keys.
     */
    RadiolyticGasModel<Pack>& configure_radiolytic_gas(
        const Database& database);
    /**
     * @brief Remove the optional radiolytic gas model, if present.
     */
    bool remove_radiolytic_gas_model() noexcept;
    /**
     * @brief Return the mutable radiolytic gas model, or nullptr when absent.
     */
    RadiolyticGasModel<Pack>* find_radiolytic_gas_model() noexcept;
    /**
     * @brief Return the radiolytic gas model, or nullptr when absent.
     */
    const RadiolyticGasModel<Pack>*
    find_radiolytic_gas_model() const noexcept;

    /**
     * @brief Configure the optional boiling source from explicit options.
     */
    BoilingSourceModel<Pack>& configure_boiling_source(
        const BoilingSourceOptions& options);
    /**
     * @brief Configure the optional boiling source from database keys.
     */
    BoilingSourceModel<Pack>& configure_boiling_source(
        const Database& database);
    /**
     * @brief Remove the optional boiling source model, if present.
     */
    bool remove_boiling_source_model() noexcept;
    /**
     * @brief Return the mutable boiling source model, or nullptr when absent.
     */
    BoilingSourceModel<Pack>* find_boiling_source_model() noexcept;
    /**
     * @brief Return the boiling source model, or nullptr when absent.
     */
    const BoilingSourceModel<Pack>*
    find_boiling_source_model() const noexcept;

    /**
     * @brief Configure the scalar void-fraction model from explicit options.
     */
    ScalarVoidFractionModel<Pack>& configure_scalar_void_fraction(
        const ScalarVoidFractionOptions& options);
    /**
     * @brief Configure the scalar void-fraction model from database keys.
     */
    ScalarVoidFractionModel<Pack>& configure_scalar_void_fraction(
        const Database& database);
    /**
     * @brief Return the mutable scalar void-fraction model, or nullptr.
     */
    ScalarVoidFractionModel<Pack>* find_scalar_void_fraction_model()
        noexcept;
    /**
     * @brief Return the scalar void-fraction model, or nullptr when absent.
     */
    const ScalarVoidFractionModel<Pack>*
    find_scalar_void_fraction_model() const noexcept;

    /**
     * @brief Configure material-property feedback from explicit options.
     */
    MaterialFeedbackModel<Pack>& configure_material_feedback(
        const MaterialFeedbackOptions& options);
    /**
     * @brief Configure material-property feedback from database keys.
     */
    MaterialFeedbackModel<Pack>& configure_material_feedback(
        const Database& database);
    /**
     * @brief Remove the optional material-feedback model, if present.
     */
    bool remove_material_feedback_model() noexcept;
    /**
     * @brief Return the mutable material-feedback model, or nullptr.
     */
    MaterialFeedbackModel<Pack>* find_material_feedback_model() noexcept;
    /**
     * @brief Return the material-feedback model, or nullptr when absent.
     */
    const MaterialFeedbackModel<Pack>*
    find_material_feedback_model() const noexcept;

    /**
     * @brief Configure delayed-neutron precursor groups from explicit options.
     */
    DelayedNeutronPrecursorModel<Pack>& configure_precursors(
        const DelayedNeutronPrecursorOptions& options);
    /**
     * @brief Configure delayed-neutron precursor groups from database keys.
     */
    DelayedNeutronPrecursorModel<Pack>& configure_precursors(
        const Database& database);
    /**
     * @brief Remove the optional precursor model, if present.
     */
    bool remove_precursor_model() noexcept;
    /**
     * @brief Return the mutable precursor model, or nullptr when absent.
     */
    DelayedNeutronPrecursorModel<Pack>* find_precursor_model() noexcept;
    /**
     * @brief Return the precursor model, or nullptr when absent.
     */
    const DelayedNeutronPrecursorModel<Pack>*
    find_precursor_model() const noexcept;

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
    using base_type::predictor_pressure_gradient;
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
    scalar_type pressure_reference_density() const noexcept override;
    MaterialPropertyFields<Pack>& stored_material_properties();
    const MaterialPropertyFields<Pack>& stored_material_properties() const;
    TemperatureSourceRegistry<Pack>& stored_temperature_sources();
    const TemperatureSourceRegistry<Pack>& stored_temperature_sources() const;
    void refresh_physical_models();
    void refresh_material_feedback(scalar_type time);
    void initialize_radiolytic_gas_state(bool force = false);
    void update_void_fraction_models(scalar_type time_step);
    const field_type* active_alpha_g_field() const noexcept;
    const field_type* active_alpha_l_field() const noexcept;
    void ensure_scalar_void_fraction_model();

    BoussinesqModelOptions d_model_options;
    bool d_physical_model_enabled = false;
    bool d_primary_fields_initialized = false;
    std::unique_ptr<FissionPowerSource<Pack>> d_fission_power_source;
    std::unique_ptr<RadiolyticGasModel<Pack>> d_radiolytic_gas_model;
    std::unique_ptr<BoilingSourceModel<Pack>> d_boiling_source_model;
    std::unique_ptr<ScalarVoidFractionModel<Pack>>
        d_scalar_void_fraction_model;
    bool d_scalar_void_fraction_explicitly_configured = false;
    std::unique_ptr<MaterialFeedbackModel<Pack>>
        d_material_feedback_model;
    std::unique_ptr<DelayedNeutronPrecursorModel<Pack>>
        d_precursor_model;
};

} // namespace SimpleFluid
