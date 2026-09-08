/** Bottom-localized heat and H2 production with solved buoyant circulation. */
#include "IF97ReferenceWater.hh"
#include "VerificationMesh.hh"
#include "VerificationLinearSolvers.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, Mesh>;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int run(int argc, char** argv)
{
    std::filesystem::path output = "bottom_heated_bubbly_convection";
    std::filesystem::path properties = "verification/openfoam/bottomHeatedBubblyConvection/reference.properties";
    std::filesystem::path water_file = "verification/openfoam/reference_water.properties";
    std::filesystem::path mesh_file = "verification/openfoam/bottomHeatedBubblyConvection/mesh.dat";
    int requested_steps = 0;
    double source_scale = 1.0;
    SimpleFluid::Verification::LinearSolverControls linear_controls;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        require(i + 1 < argc, "Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--output")
            output = value;
        else if (arg == "--properties")
            properties = value;
        else if (arg == "--water-properties")
            water_file = value;
        else if (arg == "--mesh-file")
            mesh_file = value;
        else if (arg == "--steps")
            requested_steps = std::stoi(value);
        else if (arg == "--source-scale")
            source_scale = std::stod(value);
        else if (linear_controls.parse(arg, value))
            continue;
        else
            throw std::invalid_argument("Unknown argument " + arg);
    }
    require(Tpetra::getDefaultComm()->getSize() == 1, "This matched convection fixture is serial");
    require(std::isfinite(source_scale) && source_scale >= 0 && requested_steps >= 0, "Invalid run override");
    std::ifstream input(properties);
    require(input.good(), "Cannot read " + properties.string());
    std::map<std::string, double> params;
    for (std::string line; std::getline(input, line);)
    {
        line = line.substr(0, line.find('#'));
        std::istringstream stream(line);
        std::string key, extra;
        double value;
        if (!(stream >> key))
            continue;
        require(
            bool(stream >> value) && std::isfinite(value) && !(stream >> extra) && params.emplace(key, value).second,
            "Invalid or duplicate case parameter");
    }
    const auto p = [&](const std::string& key) { return params.at(key); };
    const auto reference = SimpleFluid::Verification::load_if97_reference_water(water_file);
    const auto& water = reference.liquid;
    const auto grid = SimpleFluid::Verification::read_verification_mesh(mesh_file);
    const auto& x = grid.x;
    const auto& z = grid.z;
    const int nx = static_cast<int>(x.size() - 1), nz = static_cast<int>(z.size() - 1);
    const double width = p("width"), height = p("height"), depth = p("depth"), dt = p("dt");
    const double beta = reference.thermal_expansion;
    const int total_steps = static_cast<int>(std::llround(p("end_time") / dt));
    const int steps = requested_steps ? requested_steps : total_steps;
    const int stride = static_cast<int>(std::llround(p("write_interval") / dt));
    require(nx >= 4 && nz >= 4 && width > 0 && height > 0 && depth > 0 && dt > 0 && stride > 0,
        "Invalid mesh or time parameters");
    require(std::abs(x.front()) < 1e-12 && std::abs(z.front()) < 1e-12 && std::abs(grid.y.front()) < 1e-12 &&
                std::abs(x.back() - width) < 1e-12 && std::abs(z.back() - height) < 1e-12 &&
                std::abs(grid.y.back() - depth) < 1e-12,
        "Shared mesh extents differ from physical case");
    auto geometry = std::make_shared<Mesh::Cartesian>(grid.coordinates());
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
    SimpleFluid::BoundaryConditionSet bc;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bc.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0};
        bc.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
        bc.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0};
    }
    for (const auto* name : {"xmin", "xmax", "zmax"})
        bc.temperature[name] = {SimpleFluid::BoundaryConditionType::Dirichlet, water.temperature};
    for (const auto* name : {"ymin", "ymax", "zmax"})
        bc.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = dt;
    time.steps = 1;
    time.gravity_x = time.gravity_y = 0;
    time.gravity_z = -p("gravity");
    time.reference_temperature = water.temperature;
    time.thermal_expansion = beta;
    time.kinematic_viscosity = water.kinematic_viscosity();
    time.thermal_diffusivity = water.thermal_diffusivity();
    time.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;
    time.n_pressure_correctors = 3;
    time.n_outer_correctors = 1;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1e-11;
    linear.max_iterations = 1000;
    SimpleFluid::BoussinesqModelOptions model;
    model.reference_density = model.density = water.density;
    model.specific_heat_capacity = water.specific_heat_capacity;
    model.dynamic_viscosity = water.dynamic_viscosity;
    model.thermal_conductivity = water.thermal_conductivity;
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bc, time, linear, model);
    linear_controls.apply_flow(solver);
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
    feedback.reference_density = feedback.liquid_density = water.density;
    feedback.gas_density = water.absolute_pressure * p("hydrogen_molar_mass") / (p("gas_constant") * water.temperature);
    feedback.reference_temperature = water.temperature;
    feedback.thermal_expansion = beta;
    feedback.reference_dynamic_viscosity = water.dynamic_viscosity;
    feedback.min_density = water.density * 0.9;
    solver.configure_material_feedback(feedback);
    Field power(mesh, 0.0, "bottom_power");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto c = mesh->cell_centroid(static_cast<Pack::local_ordinal_type>(cell));
        if (c.x > width / 3 && c.x < 2 * width / 3 && c.z < height / 8)
            power.set_owned_value(static_cast<Pack::local_ordinal_type>(cell), source_scale * p("power_density"));
    }
    power.sync_ghosts();
    auto& fission = solver.add_fission_power_source();
    fission.initialize_from_power_density(power);
    const double expected_power = source_scale * p("power_density") * (width / 3) * (height / 8) * depth;
    require(std::abs(fission.integrated_power() - expected_power) < 1e-12,
        "Mesh must align with source boundaries and preserve integrated heating");
    SimpleFluid::RadiolyticGasOptions gas;
    gas.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    gas.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    gas.reference_pressure = gas.atmospheric_pressure = water.absolute_pressure;
    gas.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    gas.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    gas.constant_slip_velocity = p("slip_velocity");
    gas.surface_tension_mode = SimpleFluid::SurfaceTensionMode::Constant;
    gas.surface_tension = reference.surface_tension;
    gas.hydrogen_yield_mol_per_j = p("yield_mol_per_j");
    gas.hydrogen_yield_molecules_per_100_ev = p("yield_molecules_per_100_ev");
    gas.uranium_concentration_mol_per_m3 = p("uranium_concentration");
    gas.gas_constant = p("gas_constant");
    gas.henry_coefficient = 1e-5;
    gas.hydrogen_diffusivity = 1e-8;
    gas.microbubble_lifetime = gas.large_bubble_dissolution_time = 1e100;
    gas.micro_to_large_conversion_coefficient = 0;
    gas.max_source_alpha_rate = 1;
    gas.max_subcycles = 1;
    gas.transport_solver_tolerance=p("gas_transport_tolerance");
    gas.min_radius = 1e-12;
    gas.max_radius = 1e-3;
    gas.free_surface_patches = {"zmax"};
    solver.configure_radiolytic_gas(gas);
    solver.initialize_linear_temperature({0, 0, 1}, water.temperature, water.temperature);
    auto* bubbles = solver.find_radiolytic_gas_model();
    linear_controls.apply_gas(*bubbles);
    SimpleFluid::Verification::LinearSolverHistory linear_history(output);
    std::filesystem::create_directories(output);
    std::ofstream fields(output / "fields.csv"), history(output / "history.csv");
    fields.exceptions(std::ios::badbit | std::ios::failbit);
    history.exceptions(std::ios::badbit | std::ios::failbit);
    fields << std::setprecision(17)
           << "time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,"
              "uz_m_s\n";
    history
        << std::setprecision(17)
        << "time_s,sample,temperature_max_K,temperature_mean_K,alpha_max,speed_max_m_s,uz_min_m_s,uz_max_m_s,hydrogen_"
           "mol,produced_mol,escaped_mol,hydrogen_balance_mol,heat_power_W,continuity_per_s,thermal_step_residual_J\n";
    double thermal_residual = 0, maximum_speed = 0, maximum_temperature = water.temperature, alpha_max = 0, uzmin = 0,
           uzmax = 0;
    auto write = [&](int step)
    {
        double mean_temperature = 0, max_div = 0;
        maximum_speed = alpha_max = uzmin = uzmax = 0;
        maximum_temperature = water.temperature;
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto id = static_cast<Pack::local_ordinal_type>(cell);
            const auto c = mesh->cell_centroid(id);
            const auto u = solver.velocity().value(id);
            const auto T = solver.temperature().value(id), a = bubbles->alpha_g().value(id);
            const double speed = std::sqrt(u.dot(u));
            require(std::isfinite(T) && T > 290 && T < 320 && std::isfinite(speed) && a >= 0 && a < 0.02,
                "Convection left the dilute, reference-water operating envelope");
            maximum_temperature = std::max(maximum_temperature, T);
            maximum_speed = std::max(maximum_speed, speed);
            alpha_max = std::max(alpha_max, a);
            uzmin = std::min(uzmin, u.z);
            uzmax = std::max(uzmax, u.z);
            mean_temperature += T * mesh->cell_volume(id) / (width * height * depth);
            double divergence = 0;
            for (const auto face : mesh->faces(id))
                divergence +=
                    (mesh->owner_cell(face) == id ? 1 : -1) * solver.pressure_corrected_face_fluxes().value(face);
            max_div = std::max(max_div, std::abs(divergence) / mesh->cell_volume(id));
            const auto ix = grid.interval(x, c.x), iz = grid.interval(z, c.z);
            fields << solver.time() << ',' << iz * nx + ix << ',' << x[ix] << ',' << x[ix + 1] << ',' << z[iz] << ','
                   << z[iz + 1] << ',' << T << ',' << water.density * (1 - beta * (T - water.temperature)) << ',' << a
                   << ',' << u.x << ',' << u.y << ',' << u.z << '\n';
        }
        const double inventory = bubbles->global_submerged_hydrogen_moles();
        const double produced = bubbles->cumulative_hydrogen_produced();
        const double escaped = bubbles->cumulative_submerged_bubble_hydrogen_escaped();
        const double balance = inventory + escaped - produced;
        require(std::abs(balance) < 1e-13 && max_div < 1e-6, "Hydrogen or incompressible continuity gate failed");
        require(std::abs(thermal_residual) < 1e-6, "Discrete heat-source/conduction step budget failed");
        history << solver.time() << ",global," << maximum_temperature << ',' << mean_temperature << ',' << alpha_max
                << ',' << maximum_speed << ',' << uzmin << ',' << uzmax << ',' << inventory << ',' << produced << ','
                << escaped << ',' << balance << ',' << fission.integrated_power() << ',' << max_div << ','
                << thermal_residual << '\n';
        std::cout << "step=" << step << " t=" << solver.time() << " Tmax=" << maximum_temperature
                  << " alpha=" << alpha_max << " Umax=" << maximum_speed << " uz=[" << uzmin << ',' << uzmax << "]\n";
    };
    write(0);
    for (int step = 1; step <= steps; ++step)
    {
        std::vector<double> old_temperature(mesh->num_owned_cells()), capacity(mesh->num_owned_cells());
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto id = static_cast<Pack::local_ordinal_type>(cell);
            old_temperature[cell] = solver.temperature().value(id);
            capacity[cell] = solver.material_properties().density.value(id) * water.specific_heat_capacity;
        }
        try { solver.step(); }
        catch(const std::exception& error)
        {
            throw std::runtime_error("Step "+std::to_string(step)+": "+error.what());
        }
        linear_history.write(step, solver.time(), solver.last_step_statistics(),
            bubbles->last_statistics().transport_linear);
        require(std::abs(solver.time() - step * dt) < 1e-10, "Accepted physical time mismatch");
        thermal_residual = -fission.integrated_power() * dt;
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto id = static_cast<Pack::local_ordinal_type>(cell);
            const double T = solver.temperature().value(id);
            thermal_residual += capacity[cell] * mesh->cell_volume(id) * (T - old_temperature[cell]);
            for (const auto face : mesh->faces(id))
            {
                if (!mesh->is_exterior_face(face))
                    continue;
                const auto f = mesh->face_centroid(face);
                if (std::abs(f.x) < 1e-12 || std::abs(f.x - width) < 1e-12 || std::abs(f.z - height) < 1e-12)
                    thermal_residual += dt * water.thermal_conductivity * mesh->face_area(face) *
                                        (T - water.temperature) / mesh->cell_to_face_distance(face, id);
            }
        }
        if (step % stride == 0 || step == steps)
            write(step);
    }
    if (source_scale == 0)
        require(maximum_speed < 1e-9 && alpha_max == 0 && maximum_temperature - water.temperature < 1e-8,
            "Zero-source control moved or heated");
    else if (steps >= total_steps)
        require(maximum_temperature - water.temperature > 0.05 && alpha_max > 1e-6 && uzmax > 1e-5 && uzmin < -1e-5,
            "Bottom source failed to develop heated bubbly upward flow and downward return flow");
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
        std::cerr << "Bottom convection: " << error.what() << '\n';
        return 1;
    }
}
