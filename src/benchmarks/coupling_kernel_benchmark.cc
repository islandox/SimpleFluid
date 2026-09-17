/**
 * @file coupling_kernel_benchmark.cc
 * @brief Opt-in gas kinetics and annular ALE checkpoint replay measurements.
 *
 * Run both builds with identical arguments, one thread per rank, and fresh output
 * directories. Gas repeats restore the same initial state outside the measured
 * advance; ALE measures checkpoint creation, all trials/replays, and acceptance.
 * CSV output and additional verification are outside the measured regions.
 * Defaults reproduce the stiff gas and R100 coupling regression parameters.
 */
#include "equations/RadiolyticGasModel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, Mesh>;
using Gas = SimpleFluid::RadiolyticGasModel<Pack, Mesh>;
using Local = Pack::local_ordinal_type;
using Clock = std::chrono::steady_clock;

struct Options
{
    std::string mode = "gas";
    std::filesystem::path output;
    int nx = 0, ny = 0, nz = 0, steps = 0, replays = 2;
};

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help")
            throw std::invalid_argument("usage: coupling_kernel_benchmark --mode gas|ale --output NEW_DIR "
                "[--nx N --ny N --nz N --steps N --replays N]; gas defaults: 8x8x8,20 advances; "
                "ALE defaults: 4x8x8,4 intervals,2 trials per interval; each axis >=2");
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + argument);
        const std::string value = argv[++i];
        if (argument == "--mode") options.mode = value;
        else if (argument == "--output") options.output = value;
        else if (argument == "--nx") options.nx = std::stoi(value);
        else if (argument == "--ny") options.ny = std::stoi(value);
        else if (argument == "--nz") options.nz = std::stoi(value);
        else if (argument == "--steps") options.steps = std::stoi(value);
        else if (argument == "--replays") options.replays = std::stoi(value);
        else throw std::invalid_argument("Unknown argument " + argument);
    }
    if (options.mode != "gas" && options.mode != "ale") throw std::invalid_argument("Invalid mode");
    const bool gas = options.mode == "gas";
    if (!options.nx) options.nx = gas ? 8 : 4;
    if (!options.ny) options.ny = 8;
    if (!options.nz) options.nz = 8;
    if (!options.steps) options.steps = gas ? 20 : 4;
    if (options.output.empty() || options.nx < 2 || options.ny < 2 || options.nz < 2 ||
        options.nx > 128 || options.ny > 128 || options.nz > 128 ||
        options.steps < 1 || options.steps > 1000 || options.replays < 1 || options.replays > 100)
        throw std::invalid_argument("Invalid output, grid (2..128), steps (1..1000), or replays (1..100)");
    return options;
}

double reduce(double local, Teuchos::EReductionType operation = Teuchos::REDUCE_SUM)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), operation, 1, &local, &result);
    return result;
}

void require(bool local, const std::string& message)
{
    if (reduce(local ? 1.0 : 0.0, Teuchos::REDUCE_MIN) == 0.0)
        throw std::runtime_error(message);
}

SimpleFluid::ArrReal edges(double lower, double upper, int cells)
{
    SimpleFluid::ArrReal values(static_cast<size_t>(cells) + 1);
    for (int i = 0; i <= cells; ++i) values[static_cast<size_t>(i)] = std::lerp(lower, upper, double(i) / cells);
    return values;
}

Mesh::DistributionOptions distribution()
{
    const auto comm = Tpetra::getDefaultComm();
    Mesh::DistributionOptions result;
    result.partition = static_cast<size_t>(comm->getRank());
    result.partitions = static_cast<size_t>(comm->getSize());
    result.allow_empty_partitions = true;
    return result;
}

std::ofstream output_file(const Options& options, const std::string& name)
{
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(options.output / (name + "-rank" + std::to_string(Tpetra::getDefaultComm()->getRank()) + ".csv"));
    stream << std::setprecision(17);
    return stream;
}

void write_fields(const Options& options, const Mesh& mesh,
    const std::map<std::string, const Field*>& fields,
    const SimpleFluid::VectorCellFieldStored<Pack, Mesh>* velocity = nullptr)
{
    bool finite = true;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        for (const auto& [name, field] : fields) finite = finite && std::isfinite(field->value(cell));
        if (velocity)
        {
            const auto value = velocity->value(cell);
            finite = finite && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }
    }
    require(finite, "Non-finite final field state");
    auto stream = output_file(options, "fields");
    stream << "mode,rank,gid,volume_m3,x_m,y_m,z_m";
    for (const auto& [name, field] : fields) stream << ',' << name;
    if (velocity) stream << ",ux_m_s,uy_m_s,uz_m_s";
    stream << '\n';
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto center = mesh.cell_centroid(cell);
        stream << options.mode << ',' << Tpetra::getDefaultComm()->getRank() << ',' << mesh.cell_geometry_global_id(cell)
               << ',' << mesh.cell_volume(cell) << ',' << center.x << ',' << center.y << ',' << center.z;
        for (const auto& [name, field] : fields) stream << ',' << field->value(cell);
        if (velocity)
        {
            const auto value = velocity->value(cell);
            stream << ',' << value.x << ',' << value.y << ',' << value.z;
        }
        stream << '\n';
    }
}

void write_summary(const Options& options, const Mesh& mesh, double wall, double cpu, int advances,
    const Gas& gas, double time, double mass = 0.0, double energy = 0.0, double top = 0.0)
{
    const auto& statistics = gas.last_statistics();
    const double max_wall = reduce(wall, Teuchos::REDUCE_MAX);
    const double sum_cpu = reduce(cpu);
    const double inventory = gas.global_submerged_hydrogen_moles();
    const double bubble_volume = gas.global_submerged_bubble_volume();
    auto stream = output_file(options, "summary");
    stream << "mode,rank,ranks,nx,ny,nz,global_cells,steps,trials_per_interval,advances,time_s,wall_s,max_rank_wall_s,"
              "cpu_s,sum_rank_cpu_s,hydrogen_mol,produced_mol,escaped_mol,inventory_error_mol,donor_error_mol,"
              "bubble_volume_m3,maximum_subcycles,clipped_cells,radius_solver_failures,mass_kg,energy_J,top_m\n";
    stream << options.mode << ',' << Tpetra::getDefaultComm()->getRank() << ',' << Tpetra::getDefaultComm()->getSize()
           << ',' << options.nx << ',' << options.ny << ',' << options.nz << ',' << mesh.owned_cell_map()->getGlobalNumElements()
           << ',' << options.steps << ',' << (options.mode == "ale" ? options.replays : 1) << ',' << advances
           << ',' << time << ',' << wall << ',' << max_wall << ',' << cpu << ',' << sum_cpu << ',' << inventory
           << ',' << gas.cumulative_hydrogen_produced() << ',' << gas.cumulative_submerged_bubble_hydrogen_escaped()
           << ',' << statistics.inventory_error << ',' << statistics.donor_inventory_error << ',' << bubble_volume
           << ',' << statistics.maximum_subcycles << ',' << statistics.clipped_cells << ',' << statistics.radius_solver_failures
           << ',' << mass << ',' << energy << ',' << top << '\n';
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << std::setprecision(17) << options.mode << ": " << advances << " advances, max-rank wall "
                  << max_wall << " s, summed CPU " << sum_cpu << " s; output " << options.output << '\n';
}

void run_gas(const Options& options)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        edges(0, 1, options.nx), edges(0, 1, options.ny), edges(0, 1, options.nz)}});
    auto mesh = std::make_shared<Mesh>(geometry, distribution());
    SimpleFluid::RadiolyticGasOptions settings;
    settings.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    settings.hydrogen_yield_mol_per_j = 2.e-7;
    settings.max_source_alpha_rate = 1.0;
    settings.reference_pressure = 1.e5;
    settings.henry_coefficient = 1.e-5;
    settings.surface_tension = 0.07;
    settings.hydrogen_diffusivity = 1.e-8;
    settings.uranium_concentration_mol_per_m3 = 1000.0;
    settings.hydrogen_yield_molecules_per_100_ev = 1.8;
    settings.min_radius = 1.e-12;
    settings.max_radius = 1.e-3;
    settings.min_population = 1.e-40;
    settings.max_population = 1.e40;
    settings.initial_dissolved_hydrogen = 1.e3;
    settings.initial_micro_number_density = 1.e8;
    settings.initial_micro_moles = 1.e-8;
    settings.initial_large_number_density = 1.e6;
    settings.initial_large_moles = 1.e-7;
    settings.microbubble_lifetime = settings.large_bubble_dissolution_time = 1.e-7;
    settings.micro_to_large_conversion_coefficient = 1.e-8;
    settings.max_subcycles = 100;
    Gas gas(mesh, settings);
    Field temperature(mesh, 300.0, "temperature"), pressure(mesh, 0.0, "pressure"), power(mesh, 1.e5, "power");
    SimpleFluid::VectorCellFieldStored<Pack, Mesh> velocity(mesh, Mesh::Vec3{}, "velocity");
    SimpleFluid::ScalarFaceFieldStored<Pack, Mesh> flux(mesh, 0.0, "flux");
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.reference_density = material_options.density = 1000.0;
    material_options.specific_heat_capacity = 4200.0;
    material_options.dynamic_viscosity = 1.e-3;
    material_options.thermal_conductivity = 0.6;
    SimpleFluid::MaterialPropertyFields<Pack, Mesh> material(mesh, material_options, SimpleFluid::TimeStepperOptions{});
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto center = mesh->cell_centroid(cell);
        temperature.set_owned_value(cell, 298.0 + 4.0 * center.z);
        material.density.set_owned_value(cell, 995.0 + 10.0 * center.x);
    }
    temperature.sync_ghosts();
    material.density.sync_ghosts();
    gas.initialize_state(0.0, temperature, pressure, velocity, material);
    const auto initial = gas.snapshot();
    constexpr double dt = 2.e-6;
    double elapsed = 0.0, cpu = 0.0;
    // A warmup builds transport workspaces but is excluded from all measurements.
    gas.advance(dt, dt, temperature, pressure, velocity, flux, material, &power);
    for (int step = 0; step < options.steps; ++step)
    {
        gas.restore(initial);
        Tpetra::getDefaultComm()->barrier();
        const auto start = Clock::now();
        const auto cpu_start = std::clock();
        gas.advance(dt, dt, temperature, pressure, velocity, flux, material, &power);
        cpu += double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
        elapsed += std::chrono::duration<double>(Clock::now() - start).count();
        const auto& statistics = gas.last_statistics();
        require(statistics.maximum_subcycles == 100, "Gas benchmark did not exercise 100 subcycles");
        require(std::isfinite(statistics.inventory_error) && std::abs(statistics.inventory_error) <=
            1.e-10 * std::max(1.0, std::abs(statistics.hydrogen_before)), "Gas inventory closure failed");
    }
    auto fields = gas.output_fields();
    fields.emplace("temperature_K", &temperature);
    fields.emplace("density_kg_m3", &material.density);
    fields.emplace("power_W_m3", &power);
    write_fields(options, *mesh, fields);
    write_summary(options, *mesh, elapsed, cpu, options.steps, gas, dt);
}

void run_ale(const Options& options)
{
    constexpr double inner = 0.03815, outer = 0.25, bottom = 0.1, initial_top = 0.60852, vessel_top = 2.1;
    constexpr double initial_mass = 151.09200565656749, initial_temperature = 298.85;
    constexpr double interval_dt = 0.01, interval_energy = 0.01, yield = 1.5e-7;
    const double area = std::numbers::pi * (outer * outer - inner * inner);
    const double density = initial_mass / (area * (initial_top - bottom));
    const double mass_tolerance = 4096 * std::numeric_limits<double>::epsilon() * initial_mass;
    auto geometry = std::make_shared<Mesh::Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        edges(inner, outer, options.nx), edges(0, 2 * std::numbers::pi, options.ny),
        edges(bottom, initial_top, options.nz)}});
    auto mesh = std::make_shared<Mesh>(geometry, distribution());
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"rmin", "rmax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundaries.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = 0.005;
    time.steps = 1;
    time.reference_temperature = initial_temperature;
    time.thermal_expansion = 2.1e-4;
    time.gravity_z = -9.81;
    time.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;
    time.n_pressure_correctors = 2;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-11;
    linear.max_iterations = 1000;
    SimpleFluid::BoussinesqModelOptions material;
    material.reference_density = material.density = density;
    material.specific_heat_capacity = 4200.0;
    material.dynamic_viscosity = 1.e-3;
    material.thermal_conductivity = 0.6;
    Solver solver(mesh, boundaries, time, linear, material);
    SimpleFluid::MaterialFeedbackOptions thermal;
    thermal.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    thermal.reference_density = thermal.liquid_density = density;
    thermal.reference_temperature = initial_temperature;
    thermal.thermal_expansion = 2.1e-4;
    thermal.reference_dynamic_viscosity = 1.e-3;
    solver.configure_material_feedback(thermal);
    solver.add_fission_power_source().initialize_constant(0.0);
    SimpleFluid::RadiolyticGasOptions gas_options;
    gas_options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    gas_options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    gas_options.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    gas_options.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    gas_options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    gas_options.constant_slip_velocity = 0.01;
    gas_options.hydrogen_yield_mol_per_j = yield;
    gas_options.gas_release_efficiency = 1.0;
    gas_options.hydrogen_yield_molecules_per_100_ev = 1.8;
    gas_options.max_source_alpha_rate = 10.0;
    gas_options.henry_coefficient = 1.e-5;
    gas_options.surface_tension = 0.07;
    gas_options.hydrogen_diffusivity = 4.5e-9;
    gas_options.transport_solver_tolerance = 1.e-14;
    gas_options.uranium_concentration_mol_per_m3 = 1000.0;
    gas_options.initial_dissolved_hydrogen = 0.0;
    gas_options.reference_pressure = 101325.0;
    gas_options.free_surface_patches = {"zmax"};
    auto& gas = solver.configure_radiolytic_gas(gas_options);
    gas.enable_donor_hydrogen_deficit_tracking();
    solver.initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);
    SimpleFluid::FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    surface.gravity_axis = SimpleFluid::Dimension::Z;
    surface.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.initial_liquid_volume = area * (initial_top - bottom);
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = bottom;
    surface.vessel.top_elevation = vessel_top;
    surface.vessel.cross_section_area = area;
    surface.vessel.total_internal_volume = area * (vessel_top - bottom);
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = surface.headspace.initial_pressure = 101325.0;
    surface.headspace.initial_temperature = initial_temperature;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    require(solver.configure_free_surface(surface) != nullptr, "ALE setup failed");
    const std::array<double, 8> axial_shape{0.5, 0.8, 1.2, 1.5, 1.4, 1.1, 0.8, 0.5};
    const std::array<double, 4> radial_shape{1.0, 1.2, 0.9, 0.7};
    std::vector<double> energy(mesh->num_owned_cells());
    double local_weight = 0.0;
    for (size_t owned = 0; owned < energy.size(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto gid = static_cast<size_t>(mesh->cell_geometry_global_id(cell));
        const auto radial = (gid % options.nx) * radial_shape.size() / options.nx;
        const auto axial = (gid / (options.nx * options.ny)) * axial_shape.size() / options.nz;
        energy[owned] = mesh->cell_volume(cell) * radial_shape[radial] * axial_shape[axial];
        local_weight += energy[owned];
    }
    const auto total_weight = reduce(local_weight);
    for (auto& value : energy) value *= interval_energy / total_weight;
    auto history = output_file(options, "history");
    history << "interval,trial,time_s,mass_kg,produced_mol,hydrogen_mol,escaped_mol,top_m,gcl_m3_s,continuity_1_s\n";
    const auto top = [&]
    {
        double result = bottom;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
            if (mesh->boundary_batch_name(batch_id) == "zmax")
                for (const auto face : batch.face_lids)
                    if (mesh->is_owned_face(face)) result = std::max(result, mesh->face_centroid(face).z);
        return reduce(result, Teuchos::REDUCE_MAX);
    };
    double elapsed = 0.0, cpu = 0.0;
    int advances = 0;
    for (int interval = 0; interval < options.steps; ++interval)
    {
        const double start = interval * interval_dt, end = (interval + 1) * interval_dt, duration = end - start;
        const int substeps = static_cast<int>(std::ceil(duration / time.time_step));
        Tpetra::getDefaultComm()->barrier();
        auto begin = Clock::now();
        auto cpu_begin = std::clock();
        auto checkpoint = solver.create_coupling_checkpoint();
        elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
        cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
        std::vector<double> first_temperature;
        double first_top = 0.0;
        for (int trial = 0; trial < options.replays; ++trial)
        {
            Tpetra::getDefaultComm()->barrier();
            begin = Clock::now();
            cpu_begin = std::clock();
            solver.restore_coupling_checkpoint(checkpoint);
            solver.set_coupling_interval_energy(energy, duration);
            for (int substep = 0; substep < substeps; ++substep)
            {
                const double endpoint = substep + 1 == substeps ? end : start + duration * (substep + 1) / substeps;
                solver.set_time_step(endpoint - solver.time());
                solver.step();
                ++advances;
            }
            elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
            cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
            const double mass = solver.liquid_mass_inventory().totalMass();
            const double produced = (interval + 1) * interval_energy * yield;
            const double retained = gas.global_submerged_hydrogen_moles();
            const double escaped = gas.cumulative_submerged_bubble_hydrogen_escaped();
            const double height = top();
            require(std::abs(mass - initial_mass) <= mass_tolerance && solver.time() == end &&
                std::abs(gas.cumulative_hydrogen_produced() - produced) <= 1.e-18 &&
                std::abs(retained + escaped - produced) <= 1.e-17, "ALE time/mass/hydrogen closure failed");
            bool replay_matches = true;
            for (size_t owned = 0; owned < energy.size(); ++owned)
            {
                const double temperature = solver.temperature().value(static_cast<Local>(owned));
                if (trial == 0) first_temperature.push_back(temperature);
                else replay_matches = replay_matches && std::abs(temperature - first_temperature[owned]) <= 1.e-11;
            }
            if (trial == 0) first_top = height;
            require(replay_matches && std::abs(height - first_top) <= 1.e-13, "ALE replay field mismatch");
            const auto& diagnostics = solver.planar_ale_diagnostics();
            history << interval << ',' << trial << ',' << solver.time() << ',' << mass << ','
                    << gas.cumulative_hydrogen_produced() << ',' << retained << ',' << escaped << ',' << height << ','
                    << diagnostics.maximum_gcl_residual << ',' << diagnostics.continuity.maximum << '\n';
        }
        begin = Clock::now();
        cpu_begin = std::clock();
        solver.accept_coupling_checkpoint(checkpoint);
        elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
        cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
    }
    auto fields = gas.output_fields();
    fields.emplace("temperature_K", &solver.temperature());
    fields.emplace("pressure_Pa", &solver.pressure());
    fields.emplace("density_kg_m3", &solver.material_properties().density);
    fields.emplace("liquid_inventory_kg_m3", &solver.liquid_mass_inventory().cellMassInventory());
    fields.emplace("donor_deficit_mol_m3", &gas.donor_hydrogen_deficit());
    fields.emplace("power_W_m3", &solver.find_fission_power_source()->field());
    write_fields(options, *mesh, fields, &solver.velocity());
    long double local_energy = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        local_energy += static_cast<long double>(solver.liquid_mass_inventory().cellMassInventory().value(cell)) *
            mesh->cell_volume(cell) * material.specific_heat_capacity * solver.temperature().value(cell);
    }
    const double total_energy = reduce(static_cast<double>(local_energy));
    const double mass = solver.liquid_mass_inventory().totalMass();
    const double height = top();
    write_summary(options, *mesh, elapsed, cpu, advances, gas, solver.time(), mass, total_energy, height);
}
} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    try
    {
        const auto options = parse(argc, argv);
        bool created = true;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            std::error_code error;
            created = std::filesystem::create_directories(options.output, error) && !error;
        }
        require(created, "Output directory must be new and writable");
        Tpetra::getDefaultComm()->barrier();
        if (options.mode == "gas") run_gas(options);
        else run_ale(options);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "coupling_kernel_benchmark rank " << Tpetra::getDefaultComm()->getRank() << ": " << error.what() << '\n';
        return 1;
    }
}
