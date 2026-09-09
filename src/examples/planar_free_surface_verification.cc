/**
 * @file planar_free_surface_verification.cc
 * @brief Deterministic analytic checks for the planar free-surface volume budget.
 */

#include "solvers/PlanarFreeSurfaceModel.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using SimpleFluid::FreeSurfaceDiagnostics;
using SimpleFluid::FreeSurfaceUpdate;
using SimpleFluid::GasMolesBySpecies;
using SimpleFluid::PlanarFreeSurfaceModel;
using SimpleFluid::real_t;

constexpr real_t bottom_elevation = 0.5;          // [m]
constexpr real_t top_elevation = 5.5;             // [m]
constexpr real_t cross_section_area = 2.0;        // [m^2]
constexpr real_t total_internal_volume = 12.;     // [m^3]
constexpr real_t ambient_pressure = 101325.;      // [Pa absolute]
constexpr real_t gas_constant = 8.31446261815324; // [J/(mol K)]

struct VerificationResult
{
    std::string_view name;
    FreeSurfaceDiagnostics diagnostics;
};

[[nodiscard]] real_t species_value(const GasMolesBySpecies& values, std::string_view species)
{
    const auto iterator = values.find(std::string(species));
    return iterator == values.end() ? 0.0 : iterator->second;
}

void require_near(std::string_view label, real_t actual, real_t expected, real_t absolute_tolerance = 5.0e-12,
    real_t relative_tolerance = 5.0e-12)
{
    const auto scale = std::max(std::abs(actual), std::abs(expected));
    const auto tolerance = absolute_tolerance + relative_tolerance * scale;
    const auto residual = std::abs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) || residual > tolerance)
    {
        std::ostringstream message;
        message << std::setprecision(17) << label << " analytic check failed: actual=" << actual
                << ", expected=" << expected << ", residual=" << residual << ", tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

void require_species(std::string_view label, const GasMolesBySpecies& values, std::string_view species, real_t expected)
{
    require_near(label, species_value(values, species), expected, 5.0e-12, 5.0e-12);
}

void require_gas_closure(const FreeSurfaceDiagnostics& diagnostics)
{
    for (const auto& [species, residual] : diagnostics.gas_closure_by_species)
    {
        require_near(std::string("gas closure for ") + species, residual, 0.0, 5.0e-12, 0.0);
    }
    require_near("total gas closure", diagnostics.gas_closure_residual, 0.0, 5.0e-12, 0.0);
}

void require_constant_area_state(const FreeSurfaceDiagnostics& diagnostics, real_t liquid_volume, real_t bubble_volume)
{
    const auto clear_level = bottom_elevation + liquid_volume / cross_section_area;
    const auto pool_level = bottom_elevation + (liquid_volume + bubble_volume) / cross_section_area;
    require_near("liquid volume", diagnostics.liquid_volume, liquid_volume);
    require_near("submerged bubble volume", diagnostics.submerged_bubble_volume, bubble_volume);
    require_near("pool volume", diagnostics.pool_volume, liquid_volume + bubble_volume);
    require_near("clear level", diagnostics.clear_level, clear_level);
    require_near("pool level", diagnostics.pool_level, pool_level);
    require_near("surface area", diagnostics.surface_area, cross_section_area);
    require_near("volume closure", diagnostics.volume_closure_residual, 0.0, 5.0e-12, 0.0);
    require_gas_closure(diagnostics);
}

[[nodiscard]] FreeSurfaceUpdate constant_volume_update(real_t liquid_volume, real_t bubble_volume)
{
    FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [liquid_volume](real_t) { return liquid_volume; };
    update.bubble_volume_at_pressure = [bubble_volume](real_t) { return bubble_volume; };
    return update;
}

[[nodiscard]] std::unique_ptr<PlanarFreeSurfaceModel> make_vented_model()
{
    auto volume_map =
        std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(bottom_elevation, top_elevation, cross_section_area);
    SimpleFluid::HeadspaceOptions options;
    options.mode = SimpleFluid::HeadspaceMode::Vented;
    options.ambient_pressure = ambient_pressure;
    options.total_internal_volume = total_internal_volume;
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(options);
    return std::make_unique<PlanarFreeSurfaceModel>(std::move(volume_map), std::move(headspace));
}

[[nodiscard]] VerificationResult run_uniform_heating()
{
    constexpr real_t initial_temperature = 300.0;    // [K]
    constexpr real_t final_temperature = 380.0;      // [K]
    constexpr real_t reference_density = 1000.0;     // [kg/m^3]
    constexpr real_t expansion_coefficient = 3.5e-4; // [1/K]
    constexpr real_t initial_liquid_volume = 3.0;    // [m^3]

    auto model = make_vented_model();
    model->initialize(constant_volume_update(initial_liquid_volume, 0.0));

    // This reciprocal density law makes the constant-mass volume change exact.
    const auto liquid_mass = reference_density * initial_liquid_volume;
    const auto final_density =
        reference_density / (1.0 + expansion_coefficient * (final_temperature - initial_temperature));
    const auto final_liquid_volume = liquid_mass / final_density;
    auto update = constant_volume_update(final_liquid_volume, 0.0);
    update.time = 80.0;
    update.time_step = 80.0;
    model->update(update);

    const auto diagnostics = model->diagnostics();
    require_constant_area_state(diagnostics, final_liquid_volume, 0.0);
    require_near("vented pressure", diagnostics.headspace.pressure, ambient_pressure, 1.0e-9, 0.0);
    return {"uniformHeating", diagnostics};
}

[[nodiscard]] VerificationResult run_gas_generation()
{
    constexpr real_t liquid_volume = 3.0;    // [m^3]
    constexpr real_t gas_temperature = 340.; // [K]
    const GasMolesBySpecies generated{{"H2", 0.60}, {"steam", 0.25}};
    const auto generated_moles = species_value(generated, "H2") + species_value(generated, "steam");

    auto model = make_vented_model();
    model->initialize(constant_volume_update(liquid_volume, 0.0));

    FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [=](real_t) { return liquid_volume; };
    update.bubble_volume_at_pressure = [=](real_t pressure)
    { return generated_moles * gas_constant * gas_temperature / pressure; };
    update.gas.generated_moles = generated;
    update.gas.submerged_moles = generated;
    update.time = 4.0;
    update.time_step = 4.0;
    model->update(update);

    const auto expected_bubble_volume = generated_moles * gas_constant * gas_temperature / ambient_pressure;
    const auto diagnostics = model->diagnostics();
    require_constant_area_state(diagnostics, liquid_volume, expected_bubble_volume);
    require_species("generated H2", diagnostics.generated_gas_moles, "H2", 0.60);
    require_species("generated steam", diagnostics.generated_gas_moles, "steam", 0.25);
    require_species("submerged H2", diagnostics.submerged_gas_moles, "H2", 0.60);
    require_species("submerged steam", diagnostics.submerged_gas_moles, "steam", 0.25);
    require_near("vented pressure", diagnostics.headspace.pressure, ambient_pressure, 1.0e-9, 0.0);
    return {"gasGeneration", diagnostics};
}

[[nodiscard]] VerificationResult run_complete_escape()
{
    constexpr real_t liquid_volume = 3.0;    // [m^3]
    constexpr real_t gas_temperature = 335.; // [K]
    const GasMolesBySpecies generated{{"H2", 0.40}, {"steam", 0.20}};
    const auto generated_moles = species_value(generated, "H2") + species_value(generated, "steam");

    auto model = make_vented_model();
    model->initialize(constant_volume_update(liquid_volume, 0.0));

    FreeSurfaceUpdate generation;
    generation.liquid_volume_at_pressure = [=](real_t) { return liquid_volume; };
    generation.bubble_volume_at_pressure = [=](real_t pressure)
    { return generated_moles * gas_constant * gas_temperature / pressure; };
    generation.gas.generated_moles = generated;
    generation.gas.submerged_moles = generated;
    generation.time = 1.0;
    generation.time_step = 1.0;
    model->update(generation);
    const auto generated_state = model->diagnostics();
    require_constant_area_state(
        generated_state, liquid_volume, generated_moles * gas_constant * gas_temperature / ambient_pressure);

    auto escape = constant_volume_update(liquid_volume, 0.0);
    escape.gas.generated_moles = generated;
    escape.gas.escaped_moles_this_step = generated;
    escape.time = 2.0;
    escape.time_step = 1.0;
    model->update(escape);

    const auto diagnostics = model->diagnostics();
    require_constant_area_state(diagnostics, liquid_volume, 0.0);
    require_species("vented H2", diagnostics.vented_gas_moles, "H2", 0.40);
    require_species("vented steam", diagnostics.vented_gas_moles, "steam", 0.20);
    require_species("remaining submerged H2", diagnostics.submerged_gas_moles, "H2", 0.0);
    require_species("remaining submerged steam", diagnostics.submerged_gas_moles, "steam", 0.0);
    require_near("vented pressure", diagnostics.headspace.pressure, ambient_pressure, 1.0e-9, 0.0);
    return {"completeEscape", diagnostics};
}

[[nodiscard]] VerificationResult run_closed_headspace()
{
    constexpr real_t initial_pressure = 120000.0; // [Pa absolute]
    constexpr real_t temperature = 330.0;         // [K]
    constexpr real_t laplace_pressure = 18000.0;  // [Pa]
    constexpr real_t liquid_volume = 4.0;         // [m^3]
    constexpr real_t initial_bubble_moles = 18.0; // [mol H2]
    constexpr real_t escaped_moles = 6.0;         // [mol H2]
    constexpr real_t final_bubble_moles = initial_bubble_moles - escaped_moles;

    const auto initial_bubble_volume =
        initial_bubble_moles * gas_constant * temperature / (initial_pressure + laplace_pressure);
    const auto initial_headspace_volume = total_internal_volume - liquid_volume - initial_bubble_volume;
    const auto initial_air_moles = initial_pressure * initial_headspace_volume / (gas_constant * temperature);
    const GasMolesBySpecies initial_inventory{{"H2", initial_bubble_moles}, {"air", initial_air_moles}};

    auto volume_map =
        std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(bottom_elevation, top_elevation, cross_section_area);
    SimpleFluid::HeadspaceOptions headspace_options;
    headspace_options.mode = SimpleFluid::HeadspaceMode::Closed;
    headspace_options.initial_pressure = initial_pressure;
    headspace_options.initial_temperature = temperature;
    headspace_options.gas_constant = gas_constant;
    headspace_options.total_internal_volume = total_internal_volume;
    headspace_options.initial_moles = {{"air", initial_air_moles}};
    headspace_options.infer_initial_moles = false;
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(headspace_options);
    SimpleFluid::FreeSurfaceCouplingOptions coupling;
    coupling.maximum_correctors = 200;
    coupling.absolute_tolerance = 1.0e-10;
    coupling.relative_tolerance = 1.0e-13;
    PlanarFreeSurfaceModel model(std::move(volume_map), std::move(headspace), coupling);

    FreeSurfaceUpdate initial;
    initial.liquid_volume_at_pressure = [=](real_t) { return liquid_volume; };
    initial.bubble_volume_at_pressure = [=](real_t pressure)
    { return initial_bubble_moles * gas_constant * temperature / (pressure + laplace_pressure); };
    initial.gas.initial_moles = initial_inventory;
    initial.gas.submerged_moles = {{"H2", initial_bubble_moles}};
    model.initialize(initial);
    const auto initial_diagnostics = model.diagnostics();
    require_near("initial closed pressure", initial_diagnostics.headspace.pressure, initial_pressure, 5.0e-7, 1.0e-12);

    FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [=](real_t) { return liquid_volume; };
    update.bubble_volume_at_pressure = [=](real_t pressure)
    { return final_bubble_moles * gas_constant * temperature / (pressure + laplace_pressure); };
    update.gas.initial_moles = initial_inventory;
    update.gas.submerged_moles = {{"H2", final_bubble_moles}};
    update.gas.escaped_moles_this_step = {{"H2", escaped_moles}};
    update.time = 3.0;
    update.time_step = 3.0;
    model.update(update);

    // The pressure-dependent bubble volume produces a quadratic closed-system
    // ideal-gas balance with one physical, positive root.
    const auto headspace_moles = initial_air_moles + escaped_moles;
    const auto nonliquid_volume = total_internal_volume - liquid_volume;
    const auto a = nonliquid_volume;
    const auto b =
        nonliquid_volume * laplace_pressure - (final_bubble_moles + headspace_moles) * gas_constant * temperature;
    const auto c = -headspace_moles * gas_constant * temperature * laplace_pressure;
    const auto discriminant = b * b - 4.0 * a * c;
    const auto expected_pressure = (-b + std::sqrt(discriminant)) / (2.0 * a);
    const auto expected_bubble_volume =
        final_bubble_moles * gas_constant * temperature / (expected_pressure + laplace_pressure);
    const auto expected_headspace_volume = total_internal_volume - liquid_volume - expected_bubble_volume;

    const auto diagnostics = model.diagnostics();
    require_constant_area_state(diagnostics, liquid_volume, expected_bubble_volume);
    require_near("closed pressure", diagnostics.headspace.pressure, expected_pressure, 5.0e-7, 2.0e-12);
    require_near("closed headspace volume", diagnostics.headspace.volume, expected_headspace_volume, 5.0e-11, 2.0e-12);
    require_near("closed headspace moles", diagnostics.headspace.total_moles, headspace_moles);
    require_near("closed ideal-gas residual",
        diagnostics.headspace.pressure * diagnostics.headspace.volume -
            diagnostics.headspace.total_moles * gas_constant * temperature,
        0.0, 2.0e-6, 0.0);
    require_near("closed nonlinear residual", diagnostics.nonlinear_residual, 0.0, 5.0e-8, 0.0);
    require_species("closed headspace air", diagnostics.headspace_gas_moles, "air", initial_air_moles);
    require_species("closed headspace H2", diagnostics.headspace_gas_moles, "H2", escaped_moles);
    require_species("closed submerged H2", diagnostics.submerged_gas_moles, "H2", final_bubble_moles);
    require_near("closed vented H2", species_value(diagnostics.vented_gas_moles, "H2"), 0.0);
    if (!(diagnostics.headspace.pressure > initial_diagnostics.headspace.pressure) ||
        !(diagnostics.pool_level < initial_diagnostics.pool_level))
    {
        throw std::runtime_error("closed-headspace transfer must raise pressure while lowering the pool level");
    }
    return {"closedHeadspace", diagnostics};
}

[[nodiscard]] VerificationResult run_boiling_mass_loss()
{
    constexpr real_t liquid_density = 1000.0;       // [kg/m^3]
    constexpr real_t initial_liquid_volume = 3.0;   // [m^3]
    constexpr real_t evaporated_liquid_mass = 18.0; // [kg]
    const auto initial_liquid_mass = liquid_density * initial_liquid_volume;
    const auto final_liquid_volume = (initial_liquid_mass - evaporated_liquid_mass) / liquid_density;

    auto model = make_vented_model();
    model->initialize(constant_volume_update(initial_liquid_volume, 0.0));

    // Verify only the accepted liquid-mass decrement here. BoilingSourceModel
    // integration tests cover submerged boiling mass accounting; steam-moles
    // generation and escape are not yet supported by that production path.
    auto update = constant_volume_update(final_liquid_volume, 0.0);
    update.time = 10.0;
    update.time_step = 10.0;
    model->update(update);

    const auto diagnostics = model->diagnostics();
    require_constant_area_state(diagnostics, final_liquid_volume, 0.0);
    require_near("boiling level decrement", diagnostics.clear_level,
        bottom_elevation + (initial_liquid_mass - evaporated_liquid_mass) / (liquid_density * cross_section_area));
    require_near("vented pressure", diagnostics.headspace.pressure, ambient_pressure, 1.0e-9, 0.0);
    return {"boilingMassLoss", diagnostics};
}

[[nodiscard]] VerificationResult run_case(std::string_view name)
{
    if (name == "uniformHeating")
    {
        return run_uniform_heating();
    }
    if (name == "gasGeneration")
    {
        return run_gas_generation();
    }
    if (name == "completeEscape")
    {
        return run_complete_escape();
    }
    if (name == "closedHeadspace")
    {
        return run_closed_headspace();
    }
    if (name == "boilingMassLoss")
    {
        return run_boiling_mass_loss();
    }
    throw std::invalid_argument("unknown subcase '" + std::string(name) +
                                "'; expected uniformHeating, gasGeneration, completeEscape, closedHeadspace, or "
                                "boilingMassLoss");
}

void print_results(const std::vector<VerificationResult>& results)
{
    std::cout << "case,time_s,liquid_volume_m3,bubble_volume_m3,clear_level_m,pool_level_m,pressure_pa,"
                 "pressure_residual_pa,volume_residual_m3,gas_residual_mol\n";
    std::cout << std::scientific << std::setprecision(9);
    for (const auto& result : results)
    {
        const auto& diagnostics = result.diagnostics;
        std::cout << result.name << ',' << diagnostics.time << ',' << diagnostics.liquid_volume << ','
                  << diagnostics.submerged_bubble_volume << ',' << diagnostics.clear_level << ','
                  << diagnostics.pool_level << ',' << diagnostics.headspace.pressure << ','
                  << diagnostics.nonlinear_residual << ',' << diagnostics.volume_closure_residual << ','
                  << diagnostics.gas_closure_residual << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 2)
        {
            throw std::invalid_argument("usage: planar_free_surface_verification [subcase]");
        }

        std::vector<VerificationResult> results;
        if (argc == 2)
        {
            results.push_back(run_case(argv[1]));
        }
        else
        {
            for (const auto name :
                {"uniformHeating", "gasGeneration", "completeEscape", "closedHeadspace", "boilingMassLoss"})
            {
                results.push_back(run_case(name));
            }
        }
        print_results(results);
    }
    catch (const std::exception& error)
    {
        std::cerr << "planar_free_surface_verification: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
