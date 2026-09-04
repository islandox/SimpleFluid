/**
 * @file planar_ale_verification.cc
 * @brief Deterministic solver-integrated checks for constrained planar ALE.
 */

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;
using scalar_type = Pack::scalar_type;
using local_ordinal_type = Pack::local_ordinal_type;

constexpr scalar_type time_step = 1.0e-2;
constexpr scalar_type initial_temperature = 300.0;
constexpr scalar_type density = 10.0;
constexpr scalar_type heat_capacity = 2.0;
constexpr scalar_type thermal_expansion = 1.0e-3;

struct ConfiguredCase
{
    std::shared_ptr<Mesh> mesh;
    std::unique_ptr<Solver> solver;
    SimpleFluid::VolumetricScalarSource<Pack, Mesh>* heat_source = nullptr;
};

struct VerificationResult
{
    std::string_view name;
    scalar_type pool_level = {};
    scalar_type mesh_volume = {};
    scalar_type maximum_gcl = {};
    scalar_type maximum_continuity = {};
    scalar_type liquid_mass_residual = {};
    scalar_type gas_inventory_residual = {};
    scalar_type mesh_pool_mismatch = {};
    scalar_type source_pool_residual = {};
    scalar_type energy_residual = {};
};

void require_near(std::string_view label, scalar_type actual, scalar_type expected, scalar_type absolute_tolerance,
    scalar_type relative_tolerance = {})
{
    const auto scale = std::max(std::abs(actual), std::abs(expected));
    const auto tolerance = absolute_tolerance + relative_tolerance * scale;
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance)
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17) << label << " failed: actual=" << actual
                << ", expected=" << expected << ", tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

void require_positive(std::string_view label, scalar_type value)
{
    if (!std::isfinite(value) || !(value > 0.0))
    {
        throw std::runtime_error(std::string(label) + " must be finite and positive");
    }
}

void require_residual(std::string_view label, scalar_type residual, scalar_type absolute_tolerance,
    scalar_type reference_scale, scalar_type relative_tolerance)
{
    const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(reference_scale);
    if (!std::isfinite(residual) || std::abs(residual) > tolerance)
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17) << label << " failed: residual=" << residual
                << ", tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

std::shared_ptr<Mesh> make_mesh()
{
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.5, 1.0}}});
    return std::make_shared<Mesh>(std::move(geometry));
}

SimpleFluid::BoundaryConditionSet boundaries()
{
    SimpleFluid::BoundaryConditionSet result;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        result.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        result.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
        result.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    result.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    result.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    return result;
}

SimpleFluid::TimeStepperOptions time_options()
{
    SimpleFluid::TimeStepperOptions result;
    result.time_step = time_step;
    result.steps = 1;
    result.thermal_diffusivity = 0.0;
    result.kinematic_viscosity = 0.0;
    result.thermal_expansion = 0.0;
    result.gravity_x = 0.0;
    result.gravity_y = 0.0;
    result.gravity_z = 0.0;
    result.reference_temperature = initial_temperature;
    result.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;
    result.n_pressure_correctors = 2;
    result.n_outer_correctors = 2;
    return result;
}

SimpleFluid::BoussinesqModelOptions model_options()
{
    SimpleFluid::BoussinesqModelOptions result;
    result.reference_density = density;
    result.density = density;
    result.specific_heat_capacity = heat_capacity;
    result.dynamic_viscosity = 1.0e-2;
    result.thermal_conductivity = 0.0;
    return result;
}

SimpleFluid::MaterialFeedbackOptions feedback_options()
{
    SimpleFluid::MaterialFeedbackOptions result;
    result.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    result.reference_density = density;
    result.liquid_density = density;
    result.gas_density = 1.0;
    result.reference_temperature = initial_temperature;
    result.thermal_expansion = thermal_expansion;
    result.reference_dynamic_viscosity = 1.0e-2;
    result.min_density = 1.0;
    return result;
}

SimpleFluid::FreeSurfaceOptions surface_options(int maximum_correctors)
{
    SimpleFluid::FreeSurfaceOptions result;
    result.enabled = true;
    result.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    result.gravity_axis = SimpleFluid::Dimension::Z;
    result.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    result.initial_liquid_volume = 1.0;
    result.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    result.vessel.bottom_elevation = 0.0;
    result.vessel.top_elevation = 2.0;
    result.vessel.cross_section_area = 1.0;
    result.vessel.total_internal_volume = 2.0;
    result.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    result.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    result.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    result.headspace.ambient_pressure = 101325.0;
    result.headspace.initial_pressure = 101325.0;
    result.headspace.initial_temperature = initial_temperature;
    result.ale.top_boundary = "zmax";
    result.ale.maximum_correctors = maximum_correctors;
    result.ale.level_absolute_tolerance = 1.0e-13;
    result.ale.level_relative_tolerance = 0.0;
    result.ale.relaxation = 1.0;
    return result;
}

SimpleFluid::RadiolyticGasOptions gas_options(scalar_type slip_velocity)
{
    SimpleFluid::RadiolyticGasOptions result;
    result.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    result.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    result.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    result.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    result.rise_velocity_mode = slip_velocity > 0.0 ? SimpleFluid::BubbleRiseVelocityMode::ConstantSlip
                                                    : SimpleFluid::BubbleRiseVelocityMode::ZeroSlip;
    result.constant_slip_velocity = slip_velocity;
    result.hydrogen_yield_mol_per_j = 2.0e-7;
    result.max_source_alpha_rate = 1.0;
    result.henry_coefficient = 1.0e-5;
    result.surface_tension = 0.07;
    result.hydrogen_diffusivity = 1.0e-5;
    result.uranium_concentration_mol_per_m3 = 1000.0;
    result.hydrogen_yield_molecules_per_100_ev = 1.8;
    result.reference_pressure = 101325.0;
    result.initial_dissolved_hydrogen = 1.0;
    result.free_surface_patches = {"zmax"};
    return result;
}

ConfiguredCase make_case(
    scalar_type heat, int maximum_correctors, bool include_gas = false, scalar_type slip_velocity = {})
{
    auto mesh = make_mesh();
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.0e-13;
    linear.max_iterations = 500;
    auto solver = std::make_unique<Solver>(mesh, boundaries(), time_options(), linear, model_options());
    solver->configure_material_feedback(feedback_options());
    auto* heat_source = &solver->add_temperature_source("ale_heat", heat);
    if (include_gas)
    {
        solver->add_fission_power_source().initialize_constant(0.0);
        solver->configure_radiolytic_gas(gas_options(slip_velocity));
    }
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);
    if (solver->configure_free_surface(surface_options(maximum_correctors)) == nullptr)
    {
        throw std::runtime_error("planar ALE verification failed to configure");
    }
    return {std::move(mesh), std::move(solver), heat_source};
}

ConfiguredCase make_complete_escape_case()
{
    constexpr scalar_type slip_velocity = 1.0e3;
    auto mesh = make_mesh();
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.0e-13;
    linear.max_iterations = 500;
    auto solver = std::make_unique<Solver>(mesh, boundaries(), time_options(), linear, model_options());
    solver->configure_material_feedback(feedback_options());
    auto* heat_source = &solver->add_temperature_source("ale_heat", 0.0);
    solver->add_fission_power_source().initialize_constant(0.0);

    auto gas = gas_options(slip_velocity);
    gas.initial_dissolved_hydrogen = 0.0;
    gas.initial_micro_number_density = 1.0e10;
    gas.initial_micro_moles = 2.0e-6;
    gas.initial_large_number_density = 2.0e8;
    gas.initial_large_moles = 6.0e-7;
    gas.microbubble_lifetime = 1.0e100;
    gas.large_bubble_dissolution_time = 1.0e100;
    gas.micro_to_large_conversion_coefficient = 0.0;
    // Keep the required positive Henry coefficient while making the residual
    // mass-transfer term negligible compared with the explicit molar checks.
    gas.henry_coefficient = 1.0e-100;
    solver->configure_radiolytic_gas(gas);
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);

    const auto* gas_model = solver->find_radiolytic_gas_model();
    if (gas_model == nullptr)
    {
        throw std::runtime_error("ALE complete-escape verification lacks its gas model");
    }
    const auto initial_bubble_volume = gas_model->global_submerged_bubble_volume();
    if (!std::isfinite(initial_bubble_volume) || !(initial_bubble_volume > 0.0) || !(initial_bubble_volume < 1.0))
    {
        throw std::runtime_error("ALE complete-escape seed produced an invalid raw bubble volume");
    }

    auto surface = surface_options(12);
    surface.initial_liquid_volume = 1.0 - initial_bubble_volume;
    if (solver->configure_free_surface(surface) == nullptr)
    {
        throw std::runtime_error("planar ALE complete-escape verification failed to configure");
    }
    return {std::move(mesh), std::move(solver), heat_source};
}

scalar_type global_mesh_volume(const Mesh& mesh)
{
    scalar_type local{};
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        local += mesh.cell_volume(static_cast<local_ordinal_type>(owned));
    }
    scalar_type global{};
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

scalar_type global_liquid_energy(const ConfiguredCase& state)
{
    const auto& mass_density = state.solver->liquid_mass_inventory().cellMassInventory();
    scalar_type local{};
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        local += mass_density.value(cell) * state.mesh->cell_volume(cell) * heat_capacity *
                 state.solver->temperature().value(cell);
    }
    scalar_type global{};
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

scalar_type species_value(const SimpleFluid::GasMolesBySpecies& values, std::string_view species)
{
    const auto iterator = values.find(std::string(species));
    return iterator == values.end() ? scalar_type{} : iterator->second;
}

VerificationResult result(std::string_view name, const ConfiguredCase& state)
{
    const auto surface = state.solver->free_surface_diagnostics();
    const auto& ale = state.solver->planar_ale_diagnostics();
    VerificationResult value{name, static_cast<scalar_type>(surface.pool_level), global_mesh_volume(*state.mesh),
        ale.maximum_gcl_residual, ale.continuity.maximum, ale.liquid_mass_residual, ale.gas_inventory_residual,
        ale.mesh_vessel_mismatch, ale.volume_source.source_pool_closure_residual, ale.energy_residual};

    // These are executable-level acceptance gates, independent of the linear
    // solver tolerance.  Each uses a physical absolute plus relative scale.
    const auto volume_rate_scale = value.mesh_volume / time_step;
    require_residual("ALE GCL closure", value.maximum_gcl, 1.0e-13, volume_rate_scale, 1.0e-14);
    require_residual("ALE continuity closure", value.maximum_continuity, 1.0e-10, volume_rate_scale, 2.0e-12);
    require_residual("ALE liquid-mass closure", value.liquid_mass_residual, 1.0e-12,
        state.solver->liquid_mass_inventory().totalMass(), 1.0e-13);
    require_residual("ALE gas-inventory closure", value.gas_inventory_residual, 1.0e-12,
        state.solver->find_radiolytic_gas_model() == nullptr
            ? 0.0
            : state.solver->find_radiolytic_gas_model()->global_submerged_hydrogen_moles(),
        1.0e-12);
    require_residual("ALE mesh/pool closure", value.mesh_pool_mismatch, 1.0e-12, value.mesh_volume, 2.0e-12);
    require_residual("ALE source/pool closure", value.source_pool_residual, 1.0e-12, volume_rate_scale, 2.0e-12);
    require_residual("ALE energy closure", value.energy_residual, 1.0e-9, 1.0, 1.0e-12);
    return value;
}

VerificationResult run_uniform_heating()
{
    constexpr scalar_type power_density = 1.0e-3;
    auto state = make_case(power_density, 6);
    const auto initial_mass = state.solver->liquid_mass_inventory().totalMass();
    const auto initial_energy = global_liquid_energy(state);
    state.solver->step();
    const auto temperature_scale = power_density * time_step / (density * heat_capacity);
    const auto discriminant = 1.0 - 4.0 * thermal_expansion * temperature_scale;
    if (!(discriminant > 0.0))
    {
        throw std::runtime_error("ALE heating analytic density root is not positive");
    }
    const auto temperature_change = 2.0 * temperature_scale / (1.0 + std::sqrt(discriminant));
    const auto expected_temperature = initial_temperature + temperature_change;
    const auto expected_density = density * (1.0 - thermal_expansion * temperature_change);
    const auto expected_level = initial_mass / expected_density;
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        require_near("ALE heated temperature",
            state.solver->temperature().value(static_cast<local_ordinal_type>(owned)), expected_temperature, 3.0e-9);
    }
    require_near("ALE heated level", state.solver->free_surface_diagnostics().pool_level, expected_level, 3.0e-12);
    require_near("ALE heated mesh volume", global_mesh_volume(*state.mesh), expected_level, 3.0e-12);
    require_near(
        "ALE heated liquid mass", state.solver->liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12, 1.0e-13);
    const auto expected_energy_input = power_density * global_mesh_volume(*state.mesh) * time_step;
    require_near(
        "ALE heated energy", global_liquid_energy(state) - initial_energy, expected_energy_input, 1.0e-10, 1.0e-9);
    return result("aleUniformHeating", state);
}

VerificationResult run_gas_generation()
{
    constexpr scalar_type power_density = 1.0e3;
    auto state = make_case(0.0, 10, true);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    if (gas == nullptr || fission == nullptr)
    {
        throw std::runtime_error("ALE gas verification lacks its model owners");
    }
    fission->initialize_constant(power_density);
    const auto before = gas->cumulative_hydrogen_produced();
    const auto before_submerged = gas->global_submerged_hydrogen_moles();
    const auto initial_mass = state.solver->liquid_mass_inventory().totalMass();
    state.solver->step();
    const auto surface = state.solver->free_surface_diagnostics();
    const auto expected_generation = power_density * global_mesh_volume(*state.mesh) *
                                     gas->options().hydrogen_yield_mol_per_j * gas->options().gas_release_efficiency *
                                     time_step;
    const auto old_volume_generation =
        power_density * gas->options().hydrogen_yield_mol_per_j * gas->options().gas_release_efficiency * time_step;
    if (!(std::abs(expected_generation - old_volume_generation) > 5.0e-14))
    {
        throw std::runtime_error("ALE gas source case cannot distinguish accepted-new from old volume");
    }
    require_near(
        "ALE generated H2", gas->cumulative_hydrogen_produced() - before, expected_generation, 5.0e-15, 5.0e-11);
    require_near("ALE generated H2 inventory", gas->global_submerged_hydrogen_moles() - before_submerged,
        expected_generation, 5.0e-13, 5.0e-11);
    const auto raw_bubble_volume = gas->global_submerged_bubble_volume();
    require_positive("ALE generated raw bubble volume", raw_bubble_volume);
    require_positive("ALE retained dissolved H2", gas->global_dissolved_hydrogen_moles());
    require_near("ALE gas raw-volume ownership", surface.submerged_bubble_volume, raw_bubble_volume, 1.0e-14, 1.0e-11);
    require_near(
        "ALE gas pool decomposition", surface.pool_volume, surface.liquid_volume + raw_bubble_volume, 1.0e-12, 1.0e-12);
    require_near("ALE gas displacement excludes dissolved H2", surface.pool_volume - surface.liquid_volume,
        raw_bubble_volume, 1.0e-14, 1.0e-11);
    require_near("ALE gas pool/mesh", surface.pool_volume, global_mesh_volume(*state.mesh), 3.0e-12);
    require_near("ALE gas pressure offset", gas->absolute_pressure_offset(), gas->options().reference_pressure, 1.0e-10,
        1.0e-13);
    const auto expected_temperature = initial_temperature + power_density * global_mesh_volume(*state.mesh) *
                                                                time_step / (initial_mass * heat_capacity);
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        require_near(
            "ALE gas thermal response", state.solver->temperature().value(cell), expected_temperature, 2.0e-9, 1.0e-11);
        require_near("ALE gas absolute pressure", gas->absolute_pressure().value(cell),
            gas->options().reference_pressure, 1.0e-10, 1.0e-13);
    }
    require_near("ALE gas inventory", gas->last_statistics().inventory_error, 0.0, 1.0e-12);
    return result("aleGasGeneration", state);
}

VerificationResult run_complete_escape()
{
    constexpr scalar_type seeded_micro_moles = 2.0e-6;
    constexpr scalar_type seeded_large_moles = 6.0e-7;
    constexpr scalar_type completion_absolute_tolerance = 1.0e-18;
    constexpr scalar_type completion_relative_tolerance = 1.0e-10;
    constexpr int maximum_escape_steps = 16;

    auto state = make_complete_escape_case();
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    if (gas == nullptr || fission == nullptr)
    {
        throw std::runtime_error("ALE complete-escape verification lacks its model owners");
    }
    const auto initial_surface = state.solver->free_surface_diagnostics();
    const auto initial_raw_bubble_volume = gas->global_submerged_bubble_volume();
    const auto initial_liquid_volume = initial_surface.liquid_volume;
    const auto initial_micro = gas->global_microbubble_hydrogen_moles();
    const auto initial_large = gas->global_large_bubble_hydrogen_moles();
    const auto initial_total = initial_micro + initial_large;
    const auto completion_tolerance = completion_absolute_tolerance + completion_relative_tolerance * initial_total;
    require_near("ALE seeded microbubble inventory", initial_micro, seeded_micro_moles, 1.0e-18, 1.0e-12);
    require_near("ALE seeded large-bubble inventory", initial_large, seeded_large_moles, 1.0e-18, 1.0e-12);
    require_near(
        "ALE initial pool/mesh match", initial_surface.pool_volume, global_mesh_volume(*state.mesh), 1.0e-12, 1.0e-12);
    require_near("ALE initial pool decomposition", initial_surface.pool_volume,
        initial_liquid_volume + initial_raw_bubble_volume, 1.0e-12, 1.0e-12);
    require_near("ALE complete-escape dissolved seed", gas->global_dissolved_hydrogen_moles(), 0.0, 0.0);
    require_near("ALE complete-escape initial generation", gas->cumulative_hydrogen_produced(), 0.0, 0.0);

    scalar_type accumulated_micro_escape{};
    scalar_type accumulated_large_escape{};
    int completed_steps = 0;
    bool observed_resolved_level_drop = false;
    auto remaining_bubble_moles = [gas]
    { return gas->global_microbubble_hydrogen_moles() + gas->global_large_bubble_hydrogen_moles(); };
    while (remaining_bubble_moles() > completion_tolerance && completed_steps < maximum_escape_steps)
    {
        const auto before_surface = state.solver->free_surface_diagnostics();
        const auto before_micro = gas->global_microbubble_hydrogen_moles();
        const auto before_large = gas->global_large_bubble_hydrogen_moles();
        const auto before_raw_bubble_volume = gas->global_submerged_bubble_volume();
        const auto before_cumulative_escape = gas->cumulative_submerged_bubble_hydrogen_escaped();
        const auto before_vent = species_value(before_surface.vented_gas_moles, "H2");

        state.solver->step();
        ++completed_steps;

        const auto& statistics = gas->last_statistics();
        const auto after_surface = state.solver->free_surface_diagnostics();
        const auto after_micro = gas->global_microbubble_hydrogen_moles();
        const auto after_large = gas->global_large_bubble_hydrogen_moles();
        const auto after_raw_bubble_volume = gas->global_submerged_bubble_volume();
        const auto population_tolerance =
            5.0e-18 + 5.0e-10 * std::max({std::abs(before_micro), std::abs(before_large),
                                    std::abs(statistics.submerged_bubble_hydrogen_escaped)});
        require_positive("ALE complete-escape step transfer", statistics.submerged_bubble_hydrogen_escaped);
        require_near("ALE microbubble transport decrement", before_micro - after_micro,
            statistics.microbubble_hydrogen_escaped, population_tolerance);
        require_near("ALE large-bubble transport decrement", before_large - after_large,
            statistics.large_bubble_hydrogen_escaped, population_tolerance);
        require_near("ALE population escape split",
            statistics.microbubble_hydrogen_escaped + statistics.large_bubble_hydrogen_escaped,
            statistics.submerged_bubble_hydrogen_escaped, population_tolerance);
        require_near("ALE cumulative escape",
            gas->cumulative_submerged_bubble_hydrogen_escaped() - before_cumulative_escape,
            statistics.submerged_bubble_hydrogen_escaped, population_tolerance);
        require_near("ALE vent transfer", species_value(after_surface.vented_gas_moles, "H2") - before_vent,
            statistics.submerged_bubble_hydrogen_escaped, population_tolerance);
        require_near("ALE accepted escaped-gas transfer",
            species_value(after_surface.escaped_gas_moles_this_step, "H2"),
            statistics.submerged_bubble_hydrogen_escaped, population_tolerance);
        require_near("ALE zero-generation escape step", statistics.hydrogen_produced, 0.0, 0.0);
        require_near("ALE zero cumulative generation", gas->cumulative_hydrogen_produced(), 0.0, 0.0);
        require_near("ALE constant liquid volume during escape", after_surface.liquid_volume, initial_liquid_volume,
            1.0e-12, 1.0e-12);
        require_near("ALE escape pool displacement", before_surface.pool_volume - after_surface.pool_volume,
            before_raw_bubble_volume - after_raw_bubble_volume, 2.0e-12, 1.0e-10);
        observed_resolved_level_drop =
            observed_resolved_level_drop || after_surface.pool_level < before_surface.pool_level;
        if (!(after_raw_bubble_volume < before_raw_bubble_volume) ||
            after_surface.pool_level > before_surface.pool_level)
        {
            std::ostringstream message;
            message << std::scientific << std::setprecision(17)
                    << "ALE complete-escape step did not reduce raw bubble volume or increased the pool level: step="
                    << completed_steps << ", bubble_before=" << before_raw_bubble_volume
                    << ", bubble_after=" << after_raw_bubble_volume << ", level_before=" << before_surface.pool_level
                    << ", level_after=" << after_surface.pool_level;
            throw std::runtime_error(message.str());
        }
        accumulated_micro_escape += statistics.microbubble_hydrogen_escaped;
        accumulated_large_escape += statistics.large_bubble_hydrogen_escaped;
    }

    const auto final_micro = gas->global_microbubble_hydrogen_moles();
    const auto final_large = gas->global_large_bubble_hydrogen_moles();
    const auto final_total = final_micro + final_large;
    if (!(completed_steps > 1) || final_total > completion_tolerance || !observed_resolved_level_drop ||
        !(state.solver->free_surface_diagnostics().pool_level < initial_surface.pool_level))
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "ALE complete escape did not reach its explicit inventory tolerance: remaining=" << final_total
                << ", tolerance=" << completion_tolerance << ", steps=" << completed_steps;
        throw std::runtime_error(message.str());
    }
    require_near(
        "ALE complete microbubble escape", accumulated_micro_escape, initial_micro - final_micro, 5.0e-18, 5.0e-10);
    require_near(
        "ALE complete large-bubble escape", accumulated_large_escape, initial_large - final_large, 5.0e-18, 5.0e-10);
    require_near("ALE complete cumulative escape", gas->cumulative_submerged_bubble_hydrogen_escaped(),
        initial_total - final_total, 5.0e-18, 5.0e-10);
    require_near("ALE complete vent transfer",
        species_value(state.solver->free_surface_diagnostics().vented_gas_moles, "H2"), initial_total - final_total,
        5.0e-18, 5.0e-10);
    require_near("ALE complete-escape residual dissolved inventory", gas->global_dissolved_hydrogen_moles(), 0.0,
        completion_absolute_tolerance);
    require_near("ALE complete pool/raw-volume drop",
        initial_surface.pool_volume - state.solver->free_surface_diagnostics().pool_volume,
        initial_raw_bubble_volume - gas->global_submerged_bubble_volume(), 2.0e-12, 1.0e-10);
    require_near("ALE final pool decomposition", state.solver->free_surface_diagnostics().pool_volume,
        state.solver->free_surface_diagnostics().liquid_volume + gas->global_submerged_bubble_volume(), 1.0e-12,
        1.0e-12);
    return result("aleCompleteEscape", state);
}

VerificationResult run_failure_rollback()
{
    auto state = make_case(1.0, 1);
    std::vector<scalar_type> accepted_temperature(state.mesh->num_owned_cells());
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        accepted_temperature[owned] = state.solver->temperature().value(static_cast<local_ordinal_type>(owned));
    }
    const auto accepted_volume = global_mesh_volume(*state.mesh);
    const auto accepted_mass = state.solver->liquid_mass_inventory().totalMass();
    const auto accepted_history = state.solver->free_surface_history().size();
    std::string rejection;
    try
    {
        state.solver->step();
    }
    catch (const std::runtime_error& error)
    {
        rejection = error.what();
    }
    if (rejection.find("outer level/continuity corrector did not converge") == std::string::npos)
    {
        throw std::runtime_error(
            "ALE rollback verification expected outer-corrector nonconvergence; got '" + rejection + "'");
    }
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        require_near("ALE rollback temperature",
            state.solver->temperature().value(static_cast<local_ordinal_type>(owned)), accepted_temperature[owned],
            0.0);
    }
    require_near("ALE rollback volume", global_mesh_volume(*state.mesh), accepted_volume, 0.0);
    require_near("ALE rollback liquid mass", state.solver->liquid_mass_inventory().totalMass(), accepted_mass, 0.0);
    if (state.solver->step_index() != 0 || state.solver->time() != 0.0 ||
        state.solver->free_surface_history().size() != accepted_history)
    {
        throw std::runtime_error("ALE rollback advanced accepted time/history");
    }
    if (state.solver->planar_ale_diagnostics().rejected_transactions != 1 ||
        state.solver->planar_ale_diagnostics().last_rejection_reason != rejection)
    {
        throw std::runtime_error("ALE rollback did not publish its rejection count/reason exactly once");
    }
    state.heat_source->set_enabled(false);
    state.solver->step();
    if (state.solver->step_index() != 1 || state.solver->planar_ale_diagnostics().rejected_transactions != 1)
    {
        throw std::runtime_error("ALE rollback retry did not accept exactly one step while retaining rejection count");
    }
    return result("aleFailureRollback", state);
}

VerificationResult run_case(std::string_view name)
{
    if (name == "aleUniformHeating")
        return run_uniform_heating();
    if (name == "aleGasGeneration")
        return run_gas_generation();
    if (name == "aleCompleteEscape")
        return run_complete_escape();
    if (name == "aleFailureRollback")
        return run_failure_rollback();
    throw std::invalid_argument("unknown subcase '" + std::string(name) +
                                "'; expected aleUniformHeating, aleGasGeneration, "
                                "aleCompleteEscape, or aleFailureRollback");
}

void print_results(const std::vector<VerificationResult>& results)
{
    if (Tpetra::getDefaultComm()->getRank() != 0)
        return;
    std::cout << "case,pool_level_m,mesh_volume_m3,gcl_max_m3_per_s,"
                 "continuity_max_m3_per_s,liquid_mass_residual_kg,"
                 "gas_inventory_residual_mol,mesh_pool_mismatch_m3,"
                 "source_pool_residual_m3_per_s,energy_residual_j\n";
    std::cout << std::scientific << std::setprecision(17);
    for (const auto& value : results)
    {
        std::cout << value.name << ',' << value.pool_level << ',' << value.mesh_volume << ',' << value.maximum_gcl
                  << ',' << value.maximum_continuity << ',' << value.liquid_mass_residual << ','
                  << value.gas_inventory_residual << ',' << value.mesh_pool_mismatch << ','
                  << value.source_pool_residual << ',' << value.energy_residual << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    try
    {
        if (argc > 2)
        {
            throw std::invalid_argument("usage: planar_ale_verification [subcase]");
        }
        std::vector<VerificationResult> results;
        if (argc == 2)
        {
            results.push_back(run_case(argv[1]));
        }
        else
        {
            for (const auto name : {"aleUniformHeating", "aleGasGeneration", "aleCompleteEscape", "aleFailureRollback"})
            {
                results.push_back(run_case(name));
            }
        }
        print_results(results);
    }
    catch (const std::exception& error)
    {
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            std::cerr << "planar_ale_verification: " << error.what() << '\n';
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
