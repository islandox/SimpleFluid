/** @file dispersed_bubble_verification.cc
 * @brief Matched OpenFOAM verification of the production microbubble moment transport.
 */
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
    if (!condition)
        throw std::runtime_error(message);
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
    std::string parameters = "verification/openfoam/dispersedBubbleFlow/reference.properties";
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
        else
            throw std::runtime_error("Unknown argument " + option);
    }
    require(mode == "steady" || mode == "transient", "Mode must be steady or transient");
    require(Tpetra::getDefaultComm()->getSize() == 1, "This matched Cartesian verification is serial");
    const auto values = read_parameters(parameters);
    const auto p = [&](const std::string& key) { return values.at(key); };
    const int cells = static_cast<int>(p("cells"));
    const double height = p("height"), width = p("width"), dt = p("dt");
    const double speed = p("carrier_velocity") + p("slip_velocity");
    const double end = p(mode + "_end_time"), interval = p(mode + "_write_interval");
    require(cells > 1 && cells == p("cells") && height > 0 && width > 0 && dt > 0 && speed > 0,
        "Invalid mesh/time/velocity parameters");
    const int steps = static_cast<int>(std::llround(end / dt));
    const int write_steps = static_cast<int>(std::llround(interval / dt));
    require(steps > 0 && write_steps > 0 && std::abs(steps * dt - end) < 1e-12 &&
                std::abs(write_steps * dt - interval) < 1e-12 && steps % write_steps == 0,
        "End time and write interval must align with dt");
    SimpleFluid::ArrReal z;
    for (int i = 0; i <= cells; ++i)
        z.push_back(height * i / cells);
    auto geometry =
        std::make_shared<Mesh::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, width}, {0.0, width}, z}});
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
    Field temperature(mesh, p("temperature"), "T"), pressure(mesh, 0.0, "p");
    Field power(mesh, mode == "steady" ? p("power_density") : 0.0, "qdot");
    Velocity velocity(mesh, Mesh::Vec3{0.0, 0.0, p("carrier_velocity")}, "U");
    Flux flux(mesh, 0.0, "phi");
    for (const auto face : flux.owned_face_ids())
        flux.set_owned_value(face, p("carrier_velocity") * mesh->face_area_vector(face).z);
    flux.sync_ghosts();
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.reference_density = material_options.density = 1000.0;
    material_options.specific_heat_capacity = 4200.0;
    material_options.dynamic_viscosity = 1e-3;
    material_options.thermal_conductivity = 0.6;
    SimpleFluid::MaterialPropertyFields<Pack, Mesh> material(mesh, material_options, time_options);

    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    options.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = p("slip_velocity");
    options.reference_pressure = p("pressure");
    options.atmospheric_pressure = p("atmospheric_pressure");
    options.gas_constant = p("gas_constant");
    options.surface_tension = p("surface_tension");
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
    const double nucleation = SimpleFluid::RadiolyticGasPhysics::sheng2024_nucleation_radius(p("temperature"),
        p("uranium_concentration"), p("yield_molecules_per_100_ev"), p("pressure"), p("atmospheric_pressure"));
    require(std::abs(nucleation / radius - 1.0) < 1e-12, "Reference nucleation radius disagrees with configured state");
    const double bubble_volume = 4.0 * std::numbers::pi / 3.0 * radius * radius * radius;
    const double moles_per_bubble =
        bubble_volume * (p("pressure") + 2 * p("surface_tension") / radius) / (p("gas_constant") * p("temperature"));
    const double initial = p(mode + "_initial_moles");
    const double source = mode == "steady" ? p("power_density") * p("yield_mol_per_j") * p("release_efficiency") : 0.0;
    options.initial_micro_moles = initial;
    options.initial_micro_number_density = initial / moles_per_bubble;
    Model gas(mesh, options);
    gas.initialize_state(0.0, temperature, pressure, velocity, material);
    const double volume = height * width * width, dz = height / cells;
    const double initial_inventory = initial * volume;
    double escaped = 0.0, produced = 0.0, escaped_number = 0.0, last_escape = 0.0;
    double maximum_change = 0.0;
    int steady_checks = 0;
    std::filesystem::create_directories(output);
    std::ofstream profiles(std::filesystem::path(output) / "profiles.csv");
    std::ofstream history(std::filesystem::path(output) / "history.csv");
    require(profiles.good() && history.good(), "Cannot create verification CSV files");
    profiles << std::setprecision(17)
             << "time_s,sample,z_m,micro_moles_mol_m3,micro_number_m3,alpha_g,hydrogen_balance_mol,number_balance_"
                "relative\n";
    history
        << std::setprecision(17)
        << "time_s,inventory_mol,produced_mol,escaped_mol,outlet_mol_s,hydrogen_balance_mol,maximum_change_mol_m3\n";
    auto write = [&](double time)
    {
        const double inventory = gas.global_microbubble_hydrogen_moles();
        double number = 0.0;
        for (int i = 0; i < cells; ++i)
            number += gas.micro_number_density().value(i) * volume / cells;
        const double balance = inventory + escaped - initial_inventory - produced;
        const double number_balance = (number + escaped_number - (initial_inventory + produced) / moles_per_bubble) /
                                      ((initial_inventory + produced) / moles_per_bubble);
        require(std::abs(balance) < 2e-13 && std::abs(number_balance) < 2e-8, "Global bubble conservation gate failed");
        for (int i = 0; i < cells; ++i)
        {
            profiles << time << ',' << i << ',' << (i + 0.5) * dz << ',' << gas.micro_moles().value(i) << ','
                     << gas.micro_number_density().value(i) << ',' << gas.alpha_g().value(i) << ',' << balance << ','
                     << number_balance << '\n';
        }
        history << time << ',' << inventory << ',' << produced << ',' << escaped << ',' << last_escape / dt << ','
                << balance << ',' << maximum_change << '\n';
    };
    write(0.0);
    for (int step = 1; step <= steps; ++step)
    {
        std::vector<double> previous;
        for (int i = 0; i < cells; ++i)
            previous.push_back(gas.micro_moles().value(i));
        gas.advance(step * dt, dt, temperature, pressure, velocity, flux, material, &power);
        const auto& statistics = gas.last_statistics();
        require(statistics.clipped_cells == 0 && statistics.radius_solver_failures == 0 &&
                    statistics.maximum_subcycles == 1,
            "Bubble clipping, radius failure, or unexpected subcycling");
        require(std::abs(statistics.inventory_error) < 2e-13, "Per-step hydrogen conservation gate failed");
        escaped += statistics.microbubble_hydrogen_escaped;
        escaped_number += statistics.escaped_microbubble_count;
        produced += statistics.hydrogen_produced;
        last_escape = statistics.microbubble_hydrogen_escaped;
        maximum_change = 0.0;
        for (int i = 0; i < cells; ++i)
            maximum_change = std::max(maximum_change, std::abs(gas.micro_moles().value(i) - previous[i]));
        if (mode == "steady" && maximum_change < 1e-12 && std::abs(last_escape / dt - source * volume) < 2e-11)
            ++steady_checks;
        else
            steady_checks = 0;
        if (step % write_steps == 0)
            write(step * dt);
    }
    if (mode == "steady")
    {
        require(steady_checks >= 5, "Steady profile/source balance must converge for five consecutive steps");
        require(std::abs(last_escape / dt - source * volume) < 2e-11,
            "Steady outlet must balance nonzero bubble production");
        for (int i = 0; i < cells; ++i)
        {
            const double continuum = source * (i + 0.5) * dz / speed;
            const double truncation = source * (0.5 * dz / speed + dt);
            require(std::abs(gas.micro_moles().value(i) - continuum) < truncation * 1.01 + 1e-12,
                "Steady continuum profile exceeds upwind and split-source truncation bound");
        }
    }
    else
    {
        double l1 = 0.0;
        for (int i = 0; i < cells; ++i)
        {
            const double exact_cell = initial * std::clamp(((i + 1) * dz - speed * end) / dz, 0.0, 1.0);
            l1 += std::abs(gas.micro_moles().value(i) - exact_cell) * dz / (initial * height);
        }
        require(l1 < 0.12, "Transient translating front exceeds first-order L1 error gate");
        require(escaped > 0.4 * initial_inventory && escaped < 0.6 * initial_inventory,
            "Transient escape does not match half-column residence time");
    }
    profiles.flush();
    history.flush();
    require(profiles.good() && history.good(), "Failed writing verification CSV files");
    std::cout << mode << " dispersed microbubble verification passed; profiles: " << output << "/profiles.csv\n";
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
