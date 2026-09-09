/** @file dispersed_bubble_verification.cc
 * @brief Matched OpenFOAM verification of the production microbubble moment transport.
 */
#include "IF97ReferenceWater.hh"
#include "VerificationLinearSolvers.hh"
#include "VerificationMesh.hh"
#include "VerificationParallel.hh"
#include "equations/RadiolyticGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"

#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, Mesh>;
using Velocity = SimpleFluid::VectorCellFieldStored<Pack, Mesh>;
using Flux = SimpleFluid::ScalarFaceFieldStored<Pack, Mesh>;
using Model = SimpleFluid::RadiolyticGasModel<Pack, Mesh>;

void require(bool condition, const std::string& message)
{
    SimpleFluid::Verification::ParallelContext(Tpetra::getDefaultComm()).require(condition, message);
}

std::map<std::string, double> read_parameters(const std::string& path)
{
    std::ifstream input(path);
    require(input.good(), "Cannot read parameters: " + path);
    std::map<std::string, double> result;
    for (std::string line; std::getline(input, line);)
    {
        line = line.substr(0, line.find('#'));
        std::istringstream fields(line);
        std::string key, extra;
        double value;
        if (!(fields >> key))
            continue;
        require(bool(fields >> value) && std::isfinite(value) && !(fields >> extra), "Malformed parameter: " + line);
        require(result.emplace(key, value).second, "Duplicate parameter: " + key);
    }
    return result;
}

int run(int argc, char** argv)
{
    std::string mode = "transient", output = "dispersed_bubble_output";
    int requested_steps = 0;
    SimpleFluid::Verification::LinearSolverControls linear_controls;
    std::string parameters = "verification/openfoam/dispersedBubbleFlow/reference.properties";
    std::string water_parameters = "verification/openfoam/reference_water.properties";
    std::string mesh_file = "verification/openfoam/dispersedBubbleFlow/mesh.dat";
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        require(i + 1 < argc, "Expected value after " + option);
        const std::string value = argv[++i];
        if (option == "--mode")
            mode = value;
        else if (option == "--output")
            output = value;
        else if (option == "--parameters")
            parameters = value;
        else if (option == "--water-properties")
            water_parameters = value;
        else if (option == "--steps")
            requested_steps = std::stoi(value);
        else if (option == "--mesh-file")
            mesh_file = value;
        else if (linear_controls.parse(option, value, false))
            continue;
        else
            throw std::runtime_error("Unknown argument " + option);
    }
    require(mode == "steady" || mode == "transient", "Mode must be steady or transient");
    const auto values = read_parameters(parameters);
    const auto water = SimpleFluid::Verification::load_if97_reference_water(water_parameters);
    const auto& liquid = water.liquid;
    const auto p = [&](const std::string& key) { return values.at(key); };
    const auto grid = SimpleFluid::Verification::read_verification_mesh(mesh_file);
    const auto& z = grid.z;
    const int cells = static_cast<int>(z.size() - 1);
    const double height = p("height"), width = p("width"), dt = p("dt");
    const double speed = p("carrier_velocity") + p("slip_velocity");
    const double end = p(mode + "_end_time"), interval = p(mode + "_write_interval");
    require(cells > 1 && height > 0 && width > 0 && dt > 0 && speed > 0, "Invalid mesh/time/velocity parameters");
    const int steps = static_cast<int>(std::llround(end / dt));
    const int run_steps = requested_steps ? requested_steps : steps;
    require(run_steps > 0 && run_steps <= steps, "Smoke steps must be between 1 and the configured step count");
    const int write_steps = static_cast<int>(std::llround(interval / dt));
    require(steps > 0 && write_steps > 0 && std::abs(steps * dt - end) < 1e-12 &&
                std::abs(write_steps * dt - interval) < 1e-12 && steps % write_steps == 0,
        "End time and write interval must align with dt");
    require(std::abs(grid.x.back() - width) < 1e-12 && std::abs(grid.y.back() - width) < 1e-12 &&
                std::abs(z.back() - height) < 1e-12,
        "Shared mesh differs from bubble column");
    auto geometry = std::make_shared<Mesh::Cartesian>(grid.coordinates());
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
    const SimpleFluid::Verification::ParallelContext parallel(mesh->owned_cell_map()->getComm());
    const int local_cells = static_cast<int>(mesh->num_owned_cells());
    require(
        parallel.sum(local_cells) == static_cast<int>(grid.cells()), "Partitioned mesh differs from shared cell count");
    std::vector<size_t> samples(local_cells);
    for (int i = 0; i < local_cells; ++i)
        samples[i] = grid.interval(z, mesh->cell_centroid(i).z);
    Field temperature(mesh, liquid.temperature, "T"), pressure(mesh, 0.0, "p");
    Field power(mesh, mode == "steady" ? p("power_density") : 0.0, "qdot");
    Velocity velocity(mesh, Mesh::Vec3{0.0, 0.0, p("carrier_velocity")}, "U");
    Flux flux(mesh, 0.0, "phi");
    for (const auto face : flux.owned_face_ids())
        flux.set_owned_value(face, p("carrier_velocity") * mesh->face_area_vector(face).z);
    flux.sync_ghosts();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.kinematic_viscosity = liquid.kinematic_viscosity();
    time_options.thermal_diffusivity = liquid.thermal_diffusivity();
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.reference_density = material_options.density = liquid.density;
    material_options.specific_heat_capacity = liquid.specific_heat_capacity;
    material_options.dynamic_viscosity = liquid.dynamic_viscosity;
    material_options.thermal_conductivity = liquid.thermal_conductivity;
    SimpleFluid::MaterialPropertyFields<Pack, Mesh> material(mesh, material_options, time_options);

    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    options.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = p("slip_velocity");
    options.reference_pressure = liquid.absolute_pressure;
    options.atmospheric_pressure = p("atmospheric_pressure");
    options.gas_constant = p("gas_constant");
    options.surface_tension_mode = SimpleFluid::SurfaceTensionMode::Constant;
    options.surface_tension = water.surface_tension;
    options.henry_coefficient = 1e-5;
    options.hydrogen_diffusivity = 1e-8;
    options.hydrogen_yield_mol_per_j = p("yield_mol_per_j");
    options.gas_release_efficiency = p("release_efficiency");
    options.max_source_alpha_rate = 1.0;
    options.uranium_concentration_mol_per_m3 = p("uranium_concentration");
    options.hydrogen_yield_molecules_per_100_ev = p("yield_molecules_per_100_ev");
    options.microbubble_lifetime = options.large_bubble_dissolution_time = 1e100;
    options.micro_to_large_conversion_coefficient = 0.0;
    options.max_subcycles = 1;
    options.local_ode_tolerance = 1e-12;
    options.min_radius = 1e-12;
    options.max_radius = 1e-3;
    options.free_surface_patches = {"zmax"};
    const double radius = p("nucleation_radius");
    const double nucleation =
        SimpleFluid::RadiolyticGasPhysics::sheng2024_nucleation_radius(liquid.temperature, p("uranium_concentration"),
            p("yield_molecules_per_100_ev"), liquid.absolute_pressure, p("atmospheric_pressure"));
    require(std::abs(nucleation / radius - 1.0) < 1e-12, "Reference nucleation radius disagrees with configured state");
    const double bubble_volume = 4.0 * std::numbers::pi / 3.0 * radius * radius * radius;
    const double moles_per_bubble = bubble_volume * (liquid.absolute_pressure + 2 * water.surface_tension / radius) /
                                    (p("gas_constant") * liquid.temperature);
    const double initial = p(mode + "_initial_moles");
    const double source = mode == "steady" ? p("power_density") * p("yield_mol_per_j") * p("release_efficiency") : 0.0;
    options.initial_micro_moles = initial;
    options.initial_micro_number_density = initial / moles_per_bubble;
    Model gas(mesh, options);
    linear_controls.apply_gas(gas);
    gas.initialize_state(0.0, temperature, pressure, velocity, material);
    const double volume = height * width * width;
    const double volume_scale = volume; // Original reference volume: 1 m3.
    const double initial_inventory = initial * volume;
    double escaped = 0.0, produced = 0.0, escaped_number = 0.0, last_escape = 0.0;
    double maximum_change = 0.0;
    int steady_checks = 0;
    output = parallel.output_directory(output).string();
    std::filesystem::create_directories(output);
    SimpleFluid::Verification::LinearSolverHistory linear_history(output);
    std::ofstream profiles(std::filesystem::path(output) / "profiles.csv");
    std::ofstream history(std::filesystem::path(output) / "history.csv");
    std::ofstream fields(std::filesystem::path(output) / "fields.csv");
    require(profiles.good() && history.good() && fields.good(), "Cannot create verification CSV files");
    fields << std::setprecision(17)
           << "time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,"
              "uz_m_s\n";
    profiles << std::setprecision(17)
             << "time_s,sample,z_m,micro_moles_mol_m3,micro_number_m3,alpha_g,temperature_K,absolute_pressure_Pa,"
                "density_kg_m3,"
                "specific_heat_capacity_J_kg_K,dynamic_viscosity_Pa_s,thermal_conductivity_W_m_K,"
                "kinematic_viscosity_m2_s,thermal_diffusivity_m2_s,surface_tension_N_m,"
                "hydrogen_balance_mol,number_balance_relative\n";
    history
        << std::setprecision(17)
        << "time_s,inventory_mol,produced_mol,escaped_mol,outlet_mol_s,hydrogen_balance_mol,maximum_change_mol_m3\n";
    auto write = [&](double time)
    {
        const double inventory = gas.global_microbubble_hydrogen_moles();
        double number = 0.0;
        for (int i = 0; i < local_cells; ++i)
            number += gas.micro_number_density().value(i) * mesh->cell_volume(i);
        number = parallel.sum(number);
        const double balance = inventory + escaped - initial_inventory - produced;
        const double number_balance = (number + escaped_number - (initial_inventory + produced) / moles_per_bubble) /
                                      ((initial_inventory + produced) / moles_per_bubble);
        require(std::abs(balance) < 2e-13 * volume_scale && std::abs(number_balance) < 2e-8,
            "Global bubble conservation gate failed");
        for (int i = 0; i < local_cells; ++i)
        {
            const auto sample = samples[i];
            const double rho = material.density.value(i), cp = material.specific_heat_capacity.value(i);
            const double mu = material.dynamic_viscosity.value(i), k = material.thermal_conductivity.value(i);
            const auto u = velocity.value(i);
            const auto center = mesh->cell_centroid(i);
            const auto ix = grid.interval(grid.x, center.x), iy = grid.interval(grid.y, center.y);
            if (iy != grid.ny() / 2)
                continue;
            fields << time << ',' << sample * grid.nx() + ix << ',' << grid.x[ix] << ',' << grid.x[ix + 1] << ','
                   << z[sample] << ',' << z[sample + 1] << ',' << temperature.value(i) << ',' << rho << ','
                   << gas.alpha_g().value(i) << ',' << u.x << ',' << u.y << ',' << u.z << '\n';
            if (ix != grid.nx() / 2)
                continue;
            profiles << time << ',' << sample << ',' << 0.5 * (z[sample] + z[sample + 1]) << ','
                     << gas.micro_moles().value(i) << ',' << gas.micro_number_density().value(i) << ','
                     << gas.alpha_g().value(i) << ',' << temperature.value(i) << ',' << gas.absolute_pressure().value(i)
                     << ',' << rho << ',' << cp << ',' << mu << ',' << k << ',' << mu / rho << ',' << k / (rho * cp)
                     << ',' << options.surface_tension << ',' << balance << ',' << number_balance << '\n';
        }
        history << time << ',' << inventory << ',' << produced << ',' << escaped << ',' << last_escape / dt << ','
                << balance << ',' << maximum_change << '\n';
    };
    write(0.0);
    const SimpleFluid::Verification::LoopTimer timer;
    for (int step = 1; step <= run_steps; ++step)
    {
        std::vector<double> previous;
        previous.reserve(local_cells);
        for (int i = 0; i < local_cells; ++i)
            previous.push_back(gas.micro_moles().value(i));
        gas.advance(step * dt, dt, temperature, pressure, velocity, flux, material, &power);
        const auto& statistics = gas.last_statistics();
        linear_history.write_gas(step, step * dt, statistics.transport_linear);
        require(statistics.clipped_cells == 0 && statistics.radius_solver_failures == 0 &&
                    statistics.maximum_subcycles == 1,
            "Bubble clipping, radius failure, or unexpected subcycling");
        require(
            std::abs(statistics.inventory_error) < 2e-13 * volume_scale, "Per-step hydrogen conservation gate failed");
        escaped += statistics.microbubble_hydrogen_escaped;
        escaped_number += statistics.escaped_microbubble_count;
        produced += statistics.hydrogen_produced;
        last_escape = statistics.microbubble_hydrogen_escaped;
        maximum_change = 0.0;
        for (int i = 0; i < local_cells; ++i)
            maximum_change = std::max(maximum_change, std::abs(gas.micro_moles().value(i) - previous[i]));
        maximum_change = parallel.max(maximum_change);
        if (mode == "steady" && maximum_change < 1e-12 &&
            std::abs(last_escape / dt - source * volume) < 2e-11 * volume_scale)
            ++steady_checks;
        else
            steady_checks = 0;
        if (step % write_steps == 0 || step == run_steps)
            write(step * dt);
    }
    if (run_steps == steps && mode == "steady")
    {
        require(steady_checks >= 5, "Steady profile/source balance must converge for five consecutive steps");
        require(std::abs(last_escape / dt - source * volume) < 2e-11 * volume_scale,
            "Steady outlet must balance nonzero bubble production");
        bool continuum_passed = true;
        for (int i = 0; i < local_cells; ++i)
        {
            const auto sample = samples[i];
            const double dz = z[sample + 1] - z[sample];
            const double continuum = source * 0.5 * (z[sample] + z[sample + 1]) / speed;
            const double truncation = source * (0.5 * dz / speed + dt);
            continuum_passed =
                continuum_passed && std::abs(gas.micro_moles().value(i) - continuum) < truncation * 1.01 + 1e-12;
        }
        require(parallel.all(continuum_passed),
            "Steady continuum profile exceeds upwind and split-source truncation bound");
    }
    else if (run_steps == steps)
    {
        double l1 = 0.0;
        for (int i = 0; i < local_cells; ++i)
        {
            const auto sample = samples[i];
            const double dz = z[sample + 1] - z[sample];
            const double exact_cell = initial * std::clamp((z[sample + 1] - speed * end) / dz, 0.0, 1.0);
            l1 += std::abs(gas.micro_moles().value(i) - exact_cell) * mesh->cell_volume(i) / (initial * volume);
        }
        l1 = parallel.sum(l1);
        // Retained dz, dt and transient travel distance imply the same absolute
        // front-smearing allowance (0.12 m), normalized by the taller column.
        require(l1 < 0.12 / height, "Transient translating front exceeds first-order L1 error gate");
        const double escaped_fraction = std::min(1.0, speed * end / height);
        require(escaped > 0.8 * escaped_fraction * initial_inventory &&
                    escaped < std::min(1.0, 1.2 * escaped_fraction) * initial_inventory,
            "Transient escape does not match the prescribed residence-time fraction");
    }
    profiles.flush();
    history.flush();
    fields.flush();
    linear_history.flush();
    require(profiles.good() && history.good() && fields.good(), "Failed writing verification CSV files");
    parallel.write_timing(output, timer.wall_seconds(), timer.cpu_seconds());
    if (parallel.is_root())
        std::cout << (run_steps == steps ? mode : "partial smoke")
                  << " dispersed microbubble checks passed; profiles: " << output << "/profiles.csv\n";
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Dispersed bubble verification failed: " << error.what() << '\n';
        return 1;
    }
}
